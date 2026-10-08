"""The built-in paper-trading simulator: fake money, real (free) prices, no account or sign-up.

It behaves like an Alpaca paper account from the rest of the app's point of view (same methods, same dict shapes):
- market orders fill at the latest price plus a little slippage; sent while the market is closed, they wait and
  fill at the next open (like a real broker)
- every buy/short is a bracket: a take-profit limit and a stop-loss, checked against real 1-minute price bars during
  regular hours. If one minute touched both, the stop-loss is assumed to have filled first (the cautious choice)
- today's P/L is measured from the account value at the previous close; the account-value chart is recorded as you go
- everything is saved in the app's database, so the account survives restarts. Settings -> Trading can reset it.

Prices come from free_data.FreeMarketData (Yahoo Finance + Nasdaq's symbol list + a built-in NYSE calendar).
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from datetime import UTC, date, datetime, timedelta
from typing import Any

from ..db import Database, iso, parse_iso
from . import nyse_calendar
from .free_data import FreeMarketData

log = logging.getLogger(__name__)

OPEN_STATES = ("new", "accepted", "held", "pending_new", "partially_filled")
PROCESS_EVERY = 10.0  # seconds between automatic order checks (the trader polls every 15 s)
EQUITY_EVERY = timedelta(minutes=5)
KEEP_ORDERS_DAYS = 120


def _now() -> datetime:
    return datetime.now(UTC)


def _stamp(dt: datetime) -> str:
    # milliseconds, like the rest of the database: the realized-P/L calculation orders fills by time
    return dt.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class SimBroker:
    paper = True
    mode = "paper"  # never live: real money always goes through Alpaca and the live-trading lock
    kind = "simulator"
    has_news_history = False  # backtests / model training need Alpaca's historical news

    def __init__(self, db: Database, data: FreeMarketData | None = None, starting_cash: float = 100_000.0,
                 slippage_pct: float = 0.05, clock=_now):
        self.db = db
        self.data = data or FreeMarketData()
        self.data_feed = self.data.data_feed
        self.starting_cash = float(starting_cash)
        self.slippage = max(0.0, float(slippage_pct)) / 100
        self._clock = clock
        self._lock = threading.RLock()
        self._last_process = 0.0
        self.state = self._load()

    # ------------------------------------------------------------------ persistence
    def _load(self) -> dict:
        row = self.db.query_one("SELECT value FROM sim_state WHERE key = 'account'")
        if row is not None:
            try:
                return json.loads(row["value"])
            except ValueError:
                log.warning("Simulated account data was unreadable - starting a new one")
        return self._fresh()

    def _fresh(self) -> dict:
        now = self._clock()
        return {"account_number": f"SIM-{uuid.uuid4().hex[:8].upper()}", "created_at": _stamp(now),
                "starting_cash": self.starting_cash, "cash": self.starting_cash, "last_equity": self.starting_cash,
                "closed_day": None, "positions": {}, "orders": [], "equity_at": None}

    def _save(self) -> None:
        cutoff = _stamp(self._clock() - timedelta(days=KEEP_ORDERS_DAYS))
        self.state["orders"] = [o for o in self.state["orders"]
                                if o["status"] in OPEN_STATES or (o.get("updated_at") or "") >= cutoff][-3000:]
        self.db.execute("INSERT INTO sim_state (key, value, updated_at) VALUES ('account', ?, ?) "
                        "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
                        (json.dumps(self.state), iso()))

    def reset(self, starting_cash: float | None = None) -> None:
        """Start over with a fresh account (all positions, orders and history cleared)."""
        with self._lock:
            if starting_cash is not None:
                self.starting_cash = float(starting_cash)
            self.state = self._fresh()
            self.db.execute("DELETE FROM sim_equity")
            self._save()

    # ------------------------------------------------------------------ prices
    def latest_price(self, symbol: str) -> float | None:
        return self.data.latest_price(symbol)

    def bars(self, symbol: str, start: datetime, end: datetime, timeframe: str = "1Min") -> list[dict]:
        return self.data.bars(symbol, start, end, timeframe)

    def bars_multi(self, symbols, start, end, timeframe="1Min", feed=None) -> dict[str, list[dict]]:
        return self.data.bars_multi(symbols, start, end, timeframe, feed)

    def snapshots(self, symbols, feed=None) -> dict[str, dict]:
        return self.data.snapshots(symbols, feed)

    def movers(self, top: int = 20) -> dict:
        return self.data.movers(top)

    def most_actives(self, top: int = 20, by: str = "volume") -> dict:
        return self.data.most_actives(top, by)

    def news(self, start, end, symbols=None, limit=200, include_content=True, newest_first=False) -> list[dict]:
        return self.data.news(start, end, symbols, limit, include_content, newest_first)

    def all_assets(self) -> list[dict]:
        return self.data.all_assets()

    def asset(self, symbol: str) -> dict | None:
        row = self.db.query_one("SELECT symbol, name, exchange, tradable, shortable, easy_to_borrow, fractionable "
                                "FROM tickers WHERE symbol = ?", (symbol.upper(),))
        if row is not None:
            return {**row, "tradable": bool(row["tradable"]), "shortable": bool(row["shortable"]),
                    "easy_to_borrow": bool(row["easy_to_borrow"]), "fractionable": bool(row["fractionable"]),
                    "status": "active"}
        if self.latest_price(symbol) is None:
            return None
        return {"symbol": symbol.upper(), "name": symbol.upper(), "exchange": "", "tradable": True, "shortable": True,
                "easy_to_borrow": True, "fractionable": False, "status": "active"}

    # ------------------------------------------------------------------ market clock
    def clock(self) -> dict:
        return nyse_calendar.clock(self._clock())

    def calendar(self, start: date, end: date) -> list[dict]:
        return nyse_calendar.calendar(start, end)

    def _session_now(self) -> tuple[datetime, datetime] | None:
        """Today's regular session if the market is open right now."""
        now = self._clock()
        s = nyse_calendar.session(now.astimezone(nyse_calendar.ET).date())
        return s if s is not None and s[0] <= now < s[1] else None

    # ------------------------------------------------------------------ account
    def _mark(self, symbol: str, fallback: float) -> float:
        return self.latest_price(symbol) or fallback

    def account(self) -> dict:
        self.process()
        with self._lock:
            long_mv = short_mv = 0.0
            for sym, p in self.state["positions"].items():
                mv = p["qty"] * self._mark(sym, p["avg"])
                if p["qty"] > 0:
                    long_mv += mv
                else:
                    short_mv += mv
            equity = self.state["cash"] + long_mv + short_mv
            last = self.state.get("last_equity") or equity
            return {"account_number": self.state["account_number"], "status": "ACTIVE", "equity": equity,
                    "last_equity": last, "day_pl": equity - last,
                    "day_pl_pct": (equity - last) / last * 100 if last else 0.0, "cash": self.state["cash"],
                    "buying_power": max(0.0, equity - long_mv - abs(short_mv)), "long_market_value": long_mv,
                    "short_market_value": short_mv, "shorting_enabled": True, "trading_blocked": False,
                    "pattern_day_trader": False, "daytrade_count": 0}

    def positions(self) -> list[dict]:
        self.process()
        out = []
        with self._lock:
            for sym, p in self.state["positions"].items():
                price = self._mark(sym, p["avg"])
                qty, avg = p["qty"], p["avg"]
                mv = qty * price
                upl = (price - avg) * qty
                prev = p.get("prev_close") or avg
                out.append({"symbol": sym, "qty": qty, "qty_available": qty, "side": "long" if qty > 0 else "short",
                            "avg_entry_price": avg, "current_price": price, "market_value": mv,
                            "cost_basis": avg * qty, "unrealized_pl": upl,
                            "unrealized_plpc": upl / abs(avg * qty) * 100 if qty else 0.0,
                            "unrealized_intraday_pl": (price - prev) * qty,
                            "change_today": (price - prev) / prev * 100 if prev else 0.0, "lastday_price": prev})
        return out

    def portfolio_history(self, period: str = "1M", timeframe: str | None = None) -> dict:
        days = {"1D": 1, "1W": 7, "1M": 31, "3M": 92, "1A": 366}.get(period, 31)
        since = iso(self._clock() - timedelta(days=days))
        rows = self.db.query("SELECT ts, equity FROM sim_equity WHERE ts >= ? ORDER BY ts", (since,))
        if not rows:
            eq = self.account()["equity"]
            rows = [{"ts": iso(self._clock()), "equity": eq}]
        base = rows[0]["equity"]
        return {"timestamp": [r["ts"] for r in rows], "equity": [r["equity"] for r in rows],
                "profit_loss": [r["equity"] - base for r in rows], "base_value": base,
                "timeframe": "5Min" if days <= 7 else "1D"}

    # ------------------------------------------------------------------ orders: queries
    def orders(self, status: str = "open", limit: int = 100, after: datetime | None = None,
               symbols: list[str] | None = None) -> list[dict]:
        self.process()
        with self._lock:
            rows = [o for o in self.state["orders"] if o.get("parent") is None]
            if symbols:
                wanted = {s.upper() for s in symbols}
                rows = [o for o in rows if o["symbol"] in wanted]
            if after is not None:
                cutoff = _stamp(after)
                rows = [o for o in rows if (o.get("submitted_at") or "") >= cutoff]
            if status == "open":
                rows = [o for o in rows if o["status"] in OPEN_STATES
                        or any(leg["status"] in OPEN_STATES for leg in self._legs(o))]
            elif status == "closed":
                rows = [o for o in rows if o["status"] not in OPEN_STATES
                        and not any(leg["status"] in OPEN_STATES for leg in self._legs(o))]
            out = [self._public(o) for o in rows]
        return out[-limit:]

    def recent_orders(self, days: int = 7) -> list[dict]:
        return self.orders("all", limit=500, after=self._clock() - timedelta(days=days))

    def _legs(self, order: dict) -> list[dict]:
        return [o for o in self.state["orders"] if o.get("parent") == order["id"]]

    def _public(self, o: dict) -> dict:
        out = {k: v for k, v in o.items() if not k.startswith("_")}
        if o.get("parent") is None:
            out["legs"] = [self._public(leg) for leg in self._legs(o)]
        else:
            out["legs"] = []
        return out

    def _find(self, order_id: str) -> dict | None:
        return next((o for o in self.state["orders"] if o["id"] == order_id), None)

    # ------------------------------------------------------------------ orders: placing
    def _order(self, symbol: str, side: str, qty: float, order_type: str, order_class: str, client_order_id: str,
               parent: str | None = None, limit_price: float | None = None, stop_price: float | None = None,
               status: str = "accepted") -> dict:
        now = _stamp(self._clock())
        return {"id": f"sim-{uuid.uuid4().hex[:12]}", "client_order_id": client_order_id, "symbol": symbol.upper(),
                "side": side, "qty": float(qty), "filled_qty": 0.0, "filled_avg_price": None,
                "order_type": order_type, "order_class": order_class, "status": status,
                "limit_price": limit_price, "stop_price": stop_price,
                "time_in_force": "gtc" if order_class == "bracket" else "day", "submitted_at": now, "filled_at": None,
                "updated_at": now, "parent": parent}

    def submit_bracket(self, symbol: str, side: str, qty: int, take_profit: float, stop_loss: float,
                       client_order_id: str) -> dict:
        symbol = symbol.upper()
        if qty <= 0:
            raise ValueError("quantity must be at least 1 share")
        price = self.latest_price(symbol)
        if price is None:
            raise RuntimeError(f"No price available for {symbol} right now (free price data didn't answer)")
        with self._lock:
            acct = self.account()
            if price * qty > acct["buying_power"] + 1e-6:
                raise RuntimeError(f"insufficient buying power (need ${price * qty:,.2f}, have "
                                   f"${acct['buying_power']:,.2f})")
            if side == "sell" and symbol in self.state["positions"] and self.state["positions"][symbol]["qty"] > 0:
                raise RuntimeError("a short sale can't be opened while you hold the stock")
            parent = self._order(symbol, side, qty, "market", "bracket", client_order_id)
            exit_side = "sell" if side == "buy" else "buy"
            tp = self._order(symbol, exit_side, qty, "limit", "bracket", f"{client_order_id}-tp", parent["id"],
                             limit_price=round(take_profit, 2), status="new")
            sl = self._order(symbol, exit_side, qty, "stop", "bracket", f"{client_order_id}-sl", parent["id"],
                             stop_price=round(stop_loss, 2), status="held")
            self.state["orders"].extend([parent, tp, sl])
            if self._session_now() is not None:
                self._fill(parent, self._with_slippage(price, side))
            self._save()
            return self._public(parent)

    def _with_slippage(self, price: float, side: str) -> float:
        return round(price * (1 + self.slippage) if side == "buy" else price * (1 - self.slippage), 4)

    def _fill(self, order: dict, price: float, when: datetime | None = None) -> None:
        when = when or self._clock()
        qty, side, sym = order["qty"], order["side"], order["symbol"]
        signed = qty if side == "buy" else -qty
        pos = self.state["positions"].get(sym)
        if pos is None:
            self.state["positions"][sym] = {"qty": signed, "avg": price, "opened_at": _stamp(when),
                                            "prev_close": None}
        else:
            new_qty = pos["qty"] + signed
            if abs(new_qty) < 1e-9:
                del self.state["positions"][sym]
            elif (pos["qty"] > 0) == (signed > 0):
                pos["avg"] = (pos["avg"] * pos["qty"] + price * signed) / new_qty
                pos["qty"] = new_qty
            elif (pos["qty"] > 0) != (new_qty > 0):  # flipped from long to short (or back)
                self.state["positions"][sym] = {"qty": new_qty, "avg": price, "opened_at": _stamp(when),
                                                "prev_close": None}
            else:
                pos["qty"] = new_qty
        self.state["cash"] -= signed * price
        order.update(status="filled", filled_qty=qty, filled_avg_price=round(price, 4), filled_at=_stamp(when),
                     updated_at=_stamp(when))
        if order.get("parent") is None:
            for leg in self._legs(order):  # the exits start watching the price from the entry onwards
                leg["_armed_at"] = _stamp(when)
        log.info("Simulator: %s %g %s @ $%.2f", "bought" if side == "buy" else "sold", qty, sym, price)

    # ------------------------------------------------------------------ orders: cancelling / closing
    def cancel_order(self, order_id: str) -> None:
        with self._lock:
            o = self._find(order_id)
            if o is None:
                raise RuntimeError(f"order {order_id} not found")
            if o["status"] in OPEN_STATES:
                self._cancel(o)
                if o.get("parent") is None:
                    for leg in self._legs(o):
                        if leg["status"] in OPEN_STATES:
                            self._cancel(leg)
            self._save()

    def _cancel(self, o: dict) -> None:
        o.update(status="canceled", updated_at=_stamp(self._clock()))

    def cancel_orders_for(self, symbol: str) -> int:
        with self._lock:
            n = 0
            for o in self.state["orders"]:
                if o["symbol"] == symbol.upper() and o["status"] in OPEN_STATES:
                    self._cancel(o)
                    n += 1
            self._save()
            return n

    def cancel_all_orders(self) -> int:
        with self._lock:
            n = 0
            for o in self.state["orders"]:
                if o["status"] in OPEN_STATES:
                    self._cancel(o)
                    n += 1
            self._save()
            return n

    def close_position(self, symbol: str, wait_seconds: float = 0) -> dict:
        symbol = symbol.upper()
        with self._lock:
            self.cancel_orders_for(symbol)
            pos = self.state["positions"].get(symbol)
            if pos is None:
                raise RuntimeError(f"no position in {symbol}")
            side = "sell" if pos["qty"] > 0 else "buy"
            o = self._order(symbol, side, abs(pos["qty"]), "market", "simple", f"close-{uuid.uuid4().hex[:8]}")
            self.state["orders"].append(o)
            price = self.latest_price(symbol)
            if self._session_now() is not None and price:
                self._fill(o, self._with_slippage(price, side))
            self._save()
            return self._public(o)

    def close_all_positions(self) -> int:
        with self._lock:
            syms = list(self.state["positions"])
            for s in syms:
                self.close_position(s)
            return len(syms)

    # ------------------------------------------------------------------ the matching engine
    def process(self, force: bool = False) -> None:
        """Fill waiting orders and triggered stop-loss / take-profit legs; book the previous close; record the
        account value. Runs at most every PROCESS_EVERY seconds (the app's 15-second refresh calls it)."""
        if not force and time.monotonic() - self._last_process < PROCESS_EVERY:
            return
        with self._lock:
            self._last_process = time.monotonic()
            try:
                self._book_close()
                session = self._session_now()
                if session is not None:
                    self._fill_waiting(session)
                    self._check_exits(session)
                self._record_equity()
                self._save()
            except Exception as exc:  # free data hiccup: try again next time
                log.debug("simulator check failed: %s", exc)

    def _fill_waiting(self, session: tuple[datetime, datetime]) -> None:
        """Market orders sent while the market was closed fill at today's opening price."""
        for o in [o for o in self.state["orders"] if o["status"] in ("accepted", "new") and o["order_type"] == "market"]:
            sent = parse_iso(o["submitted_at"])
            price = None
            if sent is not None and sent < session[0]:
                bars = self._session_bars(o["symbol"], session, session[0])
                price = bars[0]["o"] if bars else None
            price = price or self.latest_price(o["symbol"])
            if price:
                self._fill(o, self._with_slippage(price, o["side"]))

    def _session_bars(self, symbol: str, session: tuple[datetime, datetime], since: datetime) -> list[dict]:
        """Regular-hours 1-minute bars of the current session from `since` on."""
        now = self._clock()
        try:
            rows = self.data.bars_multi([symbol], max(since, session[0]), now, "1Min").get(symbol.upper(), [])
        except Exception as exc:
            log.debug("bars for %s unavailable: %s", symbol, exc)
            return []
        lo, hi = _stamp(max(since, session[0])), _stamp(session[1])
        return [b for b in rows if lo <= b["t"] < hi]

    def _check_exits(self, session: tuple[datetime, datetime]) -> None:
        for parent in [o for o in self.state["orders"] if o.get("parent") is None and o["status"] == "filled"]:
            legs = [leg for leg in self._legs(parent) if leg["status"] in OPEN_STATES]
            if len(legs) == 0:
                continue
            tp = next((leg for leg in legs if leg["order_type"] == "limit"), None)
            sl = next((leg for leg in legs if leg["order_type"] == "stop"), None)
            armed = parse_iso(max(leg.get("_checked") or leg.get("_armed_at") or parent["filled_at"] for leg in legs))
            if armed is None:
                continue
            bars = self._session_bars(parent["symbol"], session, armed)
            # a bar that started before the entry may hold prices from before we bought: skip it
            bars = [b for b in bars if parse_iso(b["t"]) >= armed - timedelta(seconds=59)]
            long = parent["side"] == "buy"
            hit = None
            for b in bars:
                stop_hit = sl is not None and (b["l"] <= sl["stop_price"] if long else b["h"] >= sl["stop_price"])
                take_hit = tp is not None and (b["h"] >= tp["limit_price"] if long else b["l"] <= tp["limit_price"])
                when = parse_iso(b["t"]) + timedelta(seconds=59)
                if stop_hit:  # both in one minute -> assume the stop filled first (cautious)
                    # a gap through the stop fills at the opening price, which is worse
                    gap = b["o"] < sl["stop_price"] if long else b["o"] > sl["stop_price"]
                    hit = (sl, b["o"] if gap else sl["stop_price"], when)
                    break
                if take_hit:
                    gap = b["o"] > tp["limit_price"] if long else b["o"] < tp["limit_price"]
                    hit = (tp, b["o"] if gap else tp["limit_price"], when)
                    break
            if hit is not None:
                leg, price, when = hit
                self._fill(leg, price, min(when, self._clock()))
                for other in legs:
                    if other is not leg and other["status"] in OPEN_STATES:
                        self._cancel(other)  # one-cancels-other
            elif bars:
                for leg in legs:
                    leg["_checked"] = bars[-1]["t"]

    def _book_close(self) -> None:
        """Once per trading day: remember the account value at the last close (for today's P/L)."""
        now = self._clock()
        today = now.astimezone(nyse_calendar.ET).date()
        last_close_day = None
        d = today
        for _ in range(10):
            s = nyse_calendar.session(d)
            if s is not None and s[1] <= now:
                last_close_day = d
                break
            d -= timedelta(days=1)
        if last_close_day is None or self.state.get("closed_day") == last_close_day.isoformat():
            return
        # value the positions at that close: after today's close that's today's last regular price; if the app
        # was off and the next session already started, it's the previous close Yahoo reports
        syms = list(self.state["positions"])
        snaps = self.data.snapshots(syms) if syms else {}
        in_next_session = last_close_day != today or self._session_now() is not None
        value = self.state["cash"]
        for sym, p in self.state["positions"].items():
            snap = snaps.get(sym) or {}
            close = (snap.get("prev_close") if in_next_session and last_close_day != today else snap.get("price"))
            close = close or self._mark(sym, p["avg"])
            p["prev_close"] = close
            value += p["qty"] * close
        self.state["last_equity"] = value
        self.state["closed_day"] = last_close_day.isoformat()

    def _record_equity(self) -> None:
        now = self._clock()
        last = parse_iso(self.state.get("equity_at")) if self.state.get("equity_at") else None
        if last is not None and now - last < EQUITY_EVERY:
            return
        value = self.state["cash"] + sum(p["qty"] * self._mark(s, p["avg"]) for s, p in self.state["positions"].items())
        self.db.execute("INSERT OR REPLACE INTO sim_equity (ts, equity) VALUES (?, ?)", (iso(now), value))
        self.state["equity_at"] = iso(now)

    # ------------------------------------------------------------------ for the UI
    def describe(self) -> dict[str, Any]:
        return {"kind": "simulator", "account_number": self.state["account_number"],
                "created_at": self.state["created_at"], "starting_cash": self.state["starting_cash"],
                "data": "Yahoo Finance (free, unofficial) + Nasdaq symbol list", "data_error": self.data.last_error}
