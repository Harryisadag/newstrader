"""In-memory stand-in for Alpaca. Used by the tests, and by the UI demo mode (NEWSTRADER_FAKE_BROKER=1).

It never talks to the internet and never places a real order.
"""

from __future__ import annotations

import itertools
import math
import random
from datetime import UTC, date, datetime, timedelta


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _stamp(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


# Demo mode: (price now, day change %) for the market gauges, world-market ETFs and a few stocks.
DEMO_MARKET = {
    "SPY": (664.2, -1.2), "QQQ": (598.4, -1.6), "IWM": (244.1, -0.4), "DIA": (463.5, -0.9),
    "EWJ": (77.6, -2.3), "FXI": (40.3, 1.4), "EWG": (41.2, -0.6), "EWU": (41.4, -0.3),
    "INDA": (53.1, 0.5), "EWZ": (29.2, 1.1), "EZU": (59.8, -0.7), "EWY": (74.9, 0.8),
    "AAPL": (233.1, 0.8), "NVDA": (176.9, -1.7), "TSLA": (262.5, 7.0), "MSFT": (510.0, 0.3),
    "AMD": (164.8, 3.2), "PLTR": (179.6, -4.1), "AMZN": (221.3, 0.6), "META": (709.4, -0.2),
}
DEMO_SPIKE = "TSLA"  # jumps ~5% in the last few minutes on heavy volume


def demo_bars(end: datetime, minutes: int = 60, seed: int = 7) -> tuple[dict[str, list[dict]], dict[str, float]]:
    """A short, repeatable random walk of 1-minute bars per demo symbol, ending at the last full minute before
    `end` and scaled to finish at the demo price. Returns (bars by symbol, previous close by symbol)."""
    rng = random.Random(seed)
    last = end.replace(second=0, microsecond=0) - timedelta(minutes=1)
    bars: dict[str, list[dict]] = {}
    prev_close: dict[str, float] = {}
    for sym, (price, day_pct) in DEMO_MARKET.items():
        p, rows = 100.0, []
        base_vol = rng.uniform(2_000, 20_000)
        for i in range(minutes):
            spike = sym == DEMO_SPIKE and i >= minutes - 4
            o = p
            p = p * (1 + (0.014 if spike else rng.gauss(0, 0.0008)))
            v = base_vol * (8 if spike else rng.uniform(0.6, 1.4))
            rows.append({"t": _stamp(last - timedelta(minutes=minutes - 1 - i)), "o": o, "h": max(o, p) * 1.0004,
                         "l": min(o, p) * 0.9996, "c": p, "v": round(v)})
        scale = price / rows[-1]["c"]
        bars[sym] = [{**r, **{k: round(r[k] * scale, 4) for k in ("o", "h", "l", "c")}} for r in rows]
        prev_close[sym] = round(price / (1 + day_pct / 100), 2)
    return bars, prev_close


class FakeBroker:
    paper = True
    mode = "paper"

    def __init__(self, equity: float = 100_000.0, prices: dict[str, float] | None = None, market_open: bool = True):
        self.cash = equity
        self.last_equity = equity
        self.prices: dict[str, float] = dict(prices or {"AAPL": 230.0, "NVDA": 180.0, "TSLA": 250.0, "MSFT": 510.0})
        self.market_open = market_open
        self._positions: dict[str, dict] = {}  # symbol -> {qty, avg}
        self._orders: list[dict] = []
        self._ids = itertools.count(1)
        self.shorting_enabled = True
        self.trading_blocked = False
        self.assets: dict[str, dict] = {}
        self.fail_next_submit: str | None = None
        self.calls: list[tuple] = []
        self.news_items: list[dict] = []  # for backtests
        self.bar_data: dict[str, list[dict]] = {}  # symbol -> 1-minute bars
        self.prev_close: dict[str, float] = {}  # symbol -> yesterday's close (snapshots / movers)
        self.screener_error: str | None = None  # set to make movers() / most_actives() fail

    @classmethod
    def demo(cls) -> FakeBroker:
        """A fake account with a couple of positions and an hour of made-up price history (including one sudden
        spike), for trying the UI without Alpaca keys."""
        fb = cls()
        fb.submit_bracket("AAPL", "buy", 4, 239.2, 225.4, "demo-1")
        fb.submit_bracket("NVDA", "buy", 5, 187.2, 176.4, "demo-2")
        fb.bar_data, fb.prev_close = demo_bars(datetime.now(UTC))
        fb.prices.update({sym: rows[-1]["c"] for sym, rows in fb.bar_data.items()})
        return fb

    # ---- account ----
    def _equity(self) -> float:
        mv = sum(p["qty"] * self.prices.get(s, p["avg"]) for s, p in self._positions.items())
        return self.cash + mv

    def account(self) -> dict:
        eq = self._equity()
        return {"account_number": "PA-FAKE", "status": "ACTIVE", "equity": eq, "last_equity": self.last_equity,
                "day_pl": eq - self.last_equity,
                "day_pl_pct": (eq - self.last_equity) / self.last_equity * 100 if self.last_equity else 0,
                "cash": self.cash, "buying_power": max(self.cash, 0) * 2, "long_market_value": 0,
                "short_market_value": 0, "shorting_enabled": self.shorting_enabled,
                "trading_blocked": self.trading_blocked, "pattern_day_trader": False, "daytrade_count": 0}

    def clock(self) -> dict:
        now = datetime.now(UTC)
        return {"is_open": self.market_open, "next_open": (now + timedelta(hours=12)).isoformat(),
                "next_close": (now + timedelta(hours=3)).isoformat(), "timestamp": now.isoformat()}

    def calendar(self, start: date, end: date) -> list[dict]:
        out, d = [], start
        while d <= end:
            if d.weekday() < 5:
                out.append({"date": d.isoformat(), "open": f"{d}T13:30:00Z", "close": f"{d}T20:00:00Z"})  # EDT hours
            d += timedelta(days=1)
        return out

    def positions(self) -> list[dict]:
        out = []
        for s, p in self._positions.items():
            price = self.prices.get(s, p["avg"])
            mv = p["qty"] * price
            upl = (price - p["avg"]) * p["qty"]
            out.append({"symbol": s, "qty": p["qty"], "qty_available": p["qty"], "side": "long" if p["qty"] > 0 else "short",
                        "avg_entry_price": p["avg"], "current_price": price, "market_value": mv,
                        "cost_basis": p["avg"] * p["qty"], "unrealized_pl": upl,
                        "unrealized_plpc": upl / abs(p["avg"] * p["qty"]) * 100 if p["qty"] else 0,
                        "unrealized_intraday_pl": upl, "change_today": 0.0, "lastday_price": p["avg"]})
        return out

    def orders(self, status: str = "open", limit: int = 100, after=None, symbols=None) -> list[dict]:
        rows = [o for o in self._orders if o.get("parent") is None]
        if symbols:
            rows = [o for o in rows if o["symbol"] in symbols]
        if status == "open":
            rows = [o for o in rows if o["status"] in ("new", "accepted", "held", "partially_filled")
                    or any(leg["status"] in ("new", "held", "accepted") for leg in o["legs"])]
        return [dict(o) for o in rows][-limit:]

    def recent_orders(self, days: int = 7) -> list[dict]:
        return self.orders("all", limit=500)

    def asset(self, symbol: str) -> dict | None:
        if symbol in self.assets:
            return self.assets[symbol]
        if symbol not in self.prices:
            return None
        return {"symbol": symbol, "name": symbol, "exchange": "NASDAQ", "tradable": True, "shortable": True,
                "easy_to_borrow": True, "fractionable": True, "status": "active"}

    def all_assets(self) -> list[dict]:
        return [self.asset(s) for s in self.prices]

    def portfolio_history(self, period: str = "1M", timeframe: str | None = None) -> dict:
        now = datetime.now(UTC)
        pts = 30
        ts = [(now - timedelta(days=pts - i)).isoformat() for i in range(pts)]
        eq = [self.last_equity * (1 + 0.002 * math.sin(i / 3)) for i in range(pts)]
        return {"timestamp": ts, "equity": eq, "profit_loss": [e - eq[0] for e in eq], "base_value": eq[0],
                "timeframe": "1D"}

    # ---- prices / data ----
    def latest_price(self, symbol: str) -> float | None:
        return self.prices.get(symbol)

    def bars(self, symbol, start, end, timeframe="1Min") -> list[dict]:
        if symbol in self.bar_data:
            from ..db import parse_iso

            return [b for b in self.bar_data[symbol] if start <= parse_iso(b["t"]) <= end]
        p = self.prices.get(symbol)
        if p is None:
            return []
        return [{"t": start.isoformat(), "o": p, "h": p, "l": p, "c": p, "v": 100}]

    def bars_multi(self, symbols, start, end, timeframe="1Min", feed=None) -> dict[str, list[dict]]:
        self.calls.append(("bars_multi", tuple(symbols), feed))
        if feed == "sip" and getattr(self, "refuse_sip", False):
            raise RuntimeError("subscription does not permit querying recent SIP data")
        out = {}
        for sym in symbols:
            if sym in self.bar_data:
                rows = self.bars(sym, start, end, timeframe)
                if rows:
                    out[sym] = rows
        return out

    def _snapshot(self, sym: str) -> dict | None:
        rows = self.bar_data.get(sym) or []
        price = self.prices.get(sym, rows[-1]["c"] if rows else None)
        if price is None:
            return None
        prev = self.prev_close.get(sym) or (rows[0]["o"] if rows else price)
        day = {"t": rows[0]["t"] if rows else _now(), "o": rows[0]["o"] if rows else price,
               "h": max([r["h"] for r in rows] + [price]), "l": min([r["l"] for r in rows] + [price]), "c": price,
               "v": sum(r["v"] for r in rows)}
        return {"price": price, "prev_close": prev, "change_pct": (price - prev) / prev * 100 if prev else None,
                "day_volume": day["v"], "time": rows[-1]["t"] if rows else _now(),
                "minute_bar": dict(rows[-1]) if rows else None, "daily_bar": day,
                "prev_daily_bar": {"t": None, "o": prev, "h": prev, "l": prev, "c": prev, "v": None}}

    def snapshots(self, symbols, feed=None) -> dict[str, dict]:
        self.calls.append(("snapshots", tuple(symbols), feed))
        out = {}
        for sym in symbols:
            snap = self._snapshot(sym)
            if snap is not None:
                out[sym] = snap
        return out

    def _all_snapshots(self) -> dict[str, dict]:
        out = {}
        for sym in dict.fromkeys([*self.prices, *self.bar_data]):
            snap = self._snapshot(sym)
            if snap is not None:
                out[sym] = snap
        return out

    def movers(self, top: int = 20) -> dict:
        self.calls.append(("movers", top))
        if self.screener_error:
            raise RuntimeError(self.screener_error)
        rows = [{"symbol": s, "price": d["price"], "change": round(d["price"] - d["prev_close"], 4),
                 "change_pct": round(d["change_pct"], 4)}
                for s, d in self._all_snapshots().items() if d["change_pct"]]
        return {"gainers": sorted([r for r in rows if r["change_pct"] > 0], key=lambda r: -r["change_pct"])[:top],
                "losers": sorted([r for r in rows if r["change_pct"] < 0], key=lambda r: r["change_pct"])[:top],
                "updated": _now()}

    def most_actives(self, top: int = 20, by: str = "volume") -> dict:
        self.calls.append(("most_actives", top, by))
        if self.screener_error:
            raise RuntimeError(self.screener_error)
        rows = [{"symbol": s, "volume": float(d["day_volume"] or 0),
                 "trade_count": float(len(self.bar_data.get(s) or []) * 25)}
                for s, d in self._all_snapshots().items() if d["day_volume"]]
        key = "trade_count" if by == "trades" else "volume"
        return {"items": sorted(rows, key=lambda r: -r[key])[:top], "updated": _now()}

    def news(self, start, end, symbols=None, limit=200, include_content=True, newest_first=False) -> list[dict]:
        out = [n for n in self.news_items if start <= n["created_at"] <= end
               and (not symbols or set(symbols) & set(n.get("symbols") or []))]
        if newest_first:
            out.reverse()
        return out[:limit]

    # ---- orders ----
    def _fill(self, symbol: str, side: str, qty: float, price: float) -> None:
        signed = qty if side == "buy" else -qty
        pos = self._positions.get(symbol)
        if pos is None:
            self._positions[symbol] = {"qty": signed, "avg": price}
        else:
            new_qty = pos["qty"] + signed
            if abs(new_qty) < 1e-9:
                del self._positions[symbol]
            elif (pos["qty"] > 0) == (signed > 0):
                pos["avg"] = (pos["avg"] * pos["qty"] + price * signed) / new_qty
                pos["qty"] = new_qty
            else:
                pos["qty"] = new_qty
        self.cash -= signed * price

    def submit_bracket(self, symbol, side, qty, take_profit, stop_loss, client_order_id) -> dict:
        self.calls.append(("submit_bracket", symbol, side, qty, take_profit, stop_loss))
        if self.fail_next_submit:
            msg, self.fail_next_submit = self.fail_next_submit, None
            raise RuntimeError(msg)
        price = self.prices[symbol]
        oid = f"fake-{next(self._ids)}"
        exit_side = "sell" if side == "buy" else "buy"
        legs = [
            {"id": f"{oid}-tp", "symbol": symbol, "side": exit_side, "qty": qty, "filled_qty": 0, "filled_avg_price": None,
             "order_type": "limit", "order_class": "bracket", "status": "new", "limit_price": take_profit,
             "stop_price": None, "submitted_at": _now(), "filled_at": None, "updated_at": _now(), "legs": [],
             "client_order_id": f"{client_order_id}-tp", "parent": oid},
            {"id": f"{oid}-sl", "symbol": symbol, "side": exit_side, "qty": qty, "filled_qty": 0, "filled_avg_price": None,
             "order_type": "stop", "order_class": "bracket", "status": "held", "limit_price": None,
             "stop_price": stop_loss, "submitted_at": _now(), "filled_at": None, "updated_at": _now(), "legs": [],
             "client_order_id": f"{client_order_id}-sl", "parent": oid},
        ]
        order = {"id": oid, "client_order_id": client_order_id, "symbol": symbol, "side": side, "qty": qty,
                 "filled_qty": qty, "filled_avg_price": price, "order_type": "market", "order_class": "bracket",
                 "status": "filled", "limit_price": None, "stop_price": None, "time_in_force": "gtc",
                 "submitted_at": _now(), "filled_at": _now(), "updated_at": _now(), "legs": legs, "parent": None}
        self._orders.append(order)
        self._orders.extend(legs)
        self._fill(symbol, side, qty, price)
        return dict(order)

    def cancel_order(self, order_id: str) -> None:
        for o in self._orders:
            if o["id"] == order_id and o["status"] not in ("filled", "canceled"):
                o["status"] = "canceled"

    def cancel_orders_for(self, symbol: str) -> int:
        n = 0
        for o in self._orders:
            if o["symbol"] == symbol and o["status"] in ("new", "held", "accepted"):
                o["status"] = "canceled"
                n += 1
        return n

    def cancel_all_orders(self) -> int:
        n = 0
        for o in self._orders:
            if o["status"] in ("new", "held", "accepted"):
                o["status"] = "canceled"
                n += 1
        self.calls.append(("cancel_all_orders",))
        return n

    def close_position(self, symbol: str, wait_seconds: float = 0) -> dict:
        self.cancel_orders_for(symbol)
        pos = self._positions.get(symbol)
        if pos is None:
            raise RuntimeError(f"no position in {symbol}")
        qty = abs(pos["qty"])
        side = "sell" if pos["qty"] > 0 else "buy"
        price = self.prices[symbol]
        oid = f"fake-{next(self._ids)}"
        order = {"id": oid, "client_order_id": f"close-{oid}", "symbol": symbol, "side": side, "qty": qty,
                 "filled_qty": qty, "filled_avg_price": price, "order_type": "market", "order_class": "simple",
                 "status": "filled", "limit_price": None, "stop_price": None, "time_in_force": "day",
                 "submitted_at": _now(), "filled_at": _now(), "updated_at": _now(), "legs": [], "parent": None}
        self._orders.append(order)
        self._fill(symbol, side, qty, price)
        self.calls.append(("close_position", symbol))
        return dict(order)

    def close_all_positions(self) -> int:
        syms = list(self._positions)
        for s in syms:
            self.close_position(s)
        return len(syms)
