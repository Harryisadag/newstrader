"""The Trader service: keeps Alpaca account state fresh, turns signals into bracket orders through the
risk checks, tracks order fills, and owns the kill switch.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from ..context import AppContext
from ..db import iso, parse_iso
from . import live_guard
from .fake_broker import FakeBroker
from .pnl import realized_pnl
from .risk import AccountSnapshot, AssetInfo, PositionInfo, RiskContext, check_entry, check_exit

log = logging.getLogger(__name__)

OPEN_STATUSES = {"new", "accepted", "pending_new", "held", "partially_filled", "accepted_for_bidding",
                 "pending_replace", "replaced", "calculated"}


def decide_action(direction: str, confidence: int, trading, holding_long: bool) -> tuple[str, str]:
    """What a signal should do, before risk checks. Returns (action, reason).

    action: buy | short | sell | review | ignore
    """
    if direction not in ("bullish", "bearish"):
        return "ignore", "Neutral signal - no trade."
    if confidence < trading.review_threshold:
        return "ignore", f"Confidence {confidence} is below the review threshold ({trading.review_threshold})."
    if confidence < trading.buy_threshold:
        return "review", (f"Confidence {confidence} is in the manual-review band "
                          f"({trading.review_threshold}-{trading.buy_threshold - 1}).")
    if direction == "bullish":
        return "buy", f"Bullish with confidence {confidence} (buy threshold {trading.buy_threshold})."
    if holding_long:
        if trading.sell_on_bearish:
            return "sell", f"Bearish with confidence {confidence} and you hold it."
        return "ignore", "Bearish and you hold it, but sell-on-bearish is off."
    if trading.allow_shorting:
        return "short", f"Bearish with confidence {confidence}; shorting is on."
    return "ignore", "Bearish, but you don't hold it and shorting is off."


class Trader:
    name = "trader"

    def __init__(self, ctx: AppContext, broker_factory=None):
        self.ctx = ctx
        self.broker: Any = None
        self._factory = broker_factory or self._default_factory
        self._lock = asyncio.Lock()
        self._tasks: list[asyncio.Task] = []
        self._stream_task: asyncio.Task | None = None
        self._stream: Any = None
        self.account: dict | None = None
        self.clock: dict | None = None
        self.positions: list[dict] = []
        self.open_orders: list[dict] = []
        self._clock_at = 0.0
        self._orders_synced_at = 0.0
        self._snapshot_at = 0.0
        self._asset_cache: dict[str, tuple[float, dict | None]] = {}
        self.last_error: str | None = None

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        await self.connect()
        self._tasks.append(asyncio.create_task(self._poll_loop(), name="trader-poll"))

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t
        self._tasks.clear()
        await self._stop_stream()

    async def on_keys_changed(self) -> None:
        await self.connect()

    async def on_mode_changed(self) -> None:
        await self.connect()

    def _default_factory(self):
        if os.environ.get("NEWSTRADER_FAKE_BROKER") == "1":  # demo / UI testing only - never real orders
            return FakeBroker.demo()
        creds = live_guard.credentials_for_mode(self.ctx.state, self.ctx.keys.keys)
        if creds is None:
            return None
        from .broker import Broker

        return Broker(creds, self.ctx.state, data_feed=self.ctx.config.settings.trading.data_feed)

    async def connect(self) -> None:
        await self._stop_stream()
        self.account = None
        self.positions = []
        self.open_orders = []
        try:
            self.broker = await asyncio.to_thread(self._factory)
        except Exception as exc:
            self.broker = None
            self.last_error = str(exc)
            self.ctx.state.set_status("alpaca", "error", f"Couldn't create Alpaca client: {exc}")
            log.error("Alpaca client error: %s", exc)
            return
        if self.broker is None:
            self.ctx.state.set_status("alpaca", "error", "No Alpaca paper keys - add them in Settings -> API keys")
            return
        self.ctx.state.set_status("alpaca", "starting", f"Connecting ({self.broker.mode})...")
        await self.refresh(force=True)
        if self.account is not None:
            self._start_stream()

    # ------------------------------------------------------------------ polling
    async def _poll_loop(self) -> None:
        while True:
            await asyncio.sleep(15)
            if self.broker is None:
                continue
            try:
                await self.refresh()
            except Exception:
                log.debug("refresh failed", exc_info=True)

    async def refresh(self, force: bool = False) -> None:
        b = self.broker
        if b is None:
            return
        try:
            now = time.monotonic()
            account = await asyncio.to_thread(b.account)
            positions = await asyncio.to_thread(b.positions)
            open_orders = await asyncio.to_thread(b.orders, "open")
            if force or now - self._clock_at > 60:
                self.clock = await asyncio.to_thread(b.clock)
                self._clock_at = now
            self.account, self.positions, self.open_orders = account, positions, open_orders
            self.last_error = None
            mkt = "open" if self.clock and self.clock.get("is_open") else "closed"
            self.ctx.state.set_status("alpaca", "ok",
                                      f"{b.mode.upper()} account {account.get('account_number', '')} - market {mkt}")
            self.ctx.bus.publish("account", self.summary())
            if force or now - self._orders_synced_at > 30:
                self._orders_synced_at = now
                await self.sync_orders()
            if force or now - self._snapshot_at > 300:
                self._snapshot_at = now
                self.ctx.db.execute(
                    "INSERT OR IGNORE INTO equity_snapshots (ts, equity, cash, buying_power, mode) VALUES (?,?,?,?,?)",
                    (iso(), account["equity"], account["cash"], account["buying_power"], b.mode))
            await self._check_daily_loss()
        except Exception as exc:
            self.last_error = str(exc)
            self.ctx.state.set_status("alpaca", "error", f"Alpaca error: {exc}")
            log.warning("Alpaca refresh failed: %s", exc)

    async def _check_daily_loss(self) -> None:
        if self.account is None or self.ctx.state.halted_today:
            return
        limit = self.ctx.config.settings.risk.daily_loss_limit_usd
        loss = -float(self.account.get("day_pl") or 0)
        if loss >= limit:
            reason = f"Down ${loss:,.2f} today (limit ${limit:,.2f}). No new trades until the next trading day."
            self.ctx.state.halt_for_today(reason)
            log.warning("Daily loss limit hit: %s", reason)
            await self._alert("daily_loss_limit", "Daily loss limit hit - trading stopped", reason, "error")

    def summary(self) -> dict:
        acct = self.account
        return {
            "broker_connected": acct is not None,
            "broker_error": self.last_error,
            "market": self.clock,
            "account": None if acct is None else {
                "equity": acct["equity"], "day_pl": acct["day_pl"], "day_pl_pct": acct["day_pl_pct"],
                "buying_power": acct["buying_power"], "cash": acct["cash"],
            },
            "positions_count": len([p for p in self.positions if p.get("qty")]),
        }

    # ------------------------------------------------------------------ trade stream (instant fill updates)
    def _start_stream(self) -> None:
        if not hasattr(self.broker, "creds"):
            return  # fake broker
        try:
            from alpaca.trading.stream import TradingStream

            creds = self.broker.creds
            self._stream = TradingStream(creds.api_key, creds.secret_key, paper=creds.paper)

            async def on_update(data):
                try:
                    from .broker import order_to_dict

                    await self._upsert_order(order_to_dict(data.order))
                    self.ctx.bus.publish("orders_changed", {})
                except Exception:
                    log.debug("trade update handling failed", exc_info=True)

            self._stream.subscribe_trade_updates(on_update)
            self._stream_task = asyncio.create_task(self._stream._run_forever(), name="alpaca-trade-stream")
        except Exception as exc:
            log.warning("Trade update stream unavailable (polling instead): %s", exc)

    async def _stop_stream(self) -> None:
        if self._stream is not None:
            with contextlib.suppress(Exception):
                await self._stream.stop_ws()
        if self._stream_task is not None:
            self._stream_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._stream_task
        self._stream = None
        self._stream_task = None

    # ------------------------------------------------------------------ orders <-> database
    async def sync_orders(self) -> None:
        if self.broker is None:
            return
        orders = await asyncio.to_thread(self.broker.recent_orders, 7)
        changed = False
        for o in orders:
            changed |= await self._upsert_order(o)
        if changed:
            self.ctx.bus.publish("orders_changed", {})

    async def _upsert_order(self, o: dict, parent_id: str | None = None) -> bool:
        """Insert or update an order (and its bracket legs). Returns True if anything changed."""
        db = self.ctx.db
        row = db.query_one("SELECT * FROM orders WHERE alpaca_order_id = ?", (o["id"],))
        changed = False
        fields = {"status": o.get("status"), "filled_qty": o.get("filled_qty"),
                  "filled_avg_price": o.get("filled_avg_price"), "filled_at": o.get("filled_at"),
                  "updated_at": o.get("updated_at") or iso()}
        if row is None:
            intent = "external"
            if parent_id:
                intent = "take_profit" if o.get("order_type") == "limit" else "stop_loss"
            db.insert("orders", {
                "alpaca_order_id": o["id"], "client_order_id": o.get("client_order_id"), "symbol": o["symbol"],
                "side": o.get("side"), "intent": intent, "qty": o.get("qty"), "order_type": o.get("order_type"),
                "order_class": o.get("order_class"), "stop_price": o.get("stop_price"),
                "take_profit_price": o.get("limit_price") if intent == "take_profit" else None,
                "submitted_at": o.get("submitted_at"), "mode": getattr(self.broker, "mode", "paper"),
                "parent_alpaca_id": parent_id,
                "reason": "Bracket exit leg" if parent_id else "Order placed outside NewsTrader",
                "raw": json.dumps(o)[:20000], **fields})
            changed = True
            if o.get("status") == "filled" and parent_id:
                await self._on_filled(o, intent)
        elif any(row.get(k) != v for k, v in fields.items() if k != "updated_at"):
            was_filled = row.get("status") == "filled"
            db.update("orders", row["id"], fields)
            changed = True
            if o.get("status") == "filled" and not was_filled:
                await self._on_filled(o, row.get("intent") or "external")
        for leg in o.get("legs") or []:
            changed |= await self._upsert_order(leg, parent_id=o["id"])
        return changed

    async def _on_filled(self, o: dict, intent: str) -> None:
        price = o.get("filled_avg_price")
        qty = o.get("filled_qty")
        label = {"open_long": "Bought", "open_short": "Shorted", "close_long": "Sold", "close_short": "Covered",
                 "take_profit": "Take-profit hit", "stop_loss": "Stop-loss hit"}.get(intent, "Filled")
        msg = f"{label}: {qty:g} {o['symbol']} @ ${price:,.2f}" if price and qty else f"{label}: {o['symbol']}"
        if intent in ("take_profit", "stop_loss", "close_long", "close_short"):
            pnl = self.realized_pnl_map().get(o["id"])
            if pnl is not None:
                msg += f" (P/L {'+' if pnl >= 0 else '-'}${abs(pnl):,.2f})"
        log.info(msg)
        await self._alert("trade_filled", msg, "", "trade")

    def realized_pnl_map(self) -> dict:
        rows = self.ctx.db.query(
            "SELECT alpaca_order_id AS id, symbol, side, filled_qty, filled_avg_price, filled_at FROM orders "
            "WHERE filled_qty > 0 AND filled_avg_price IS NOT NULL AND mode = ?", (self.ctx.state.mode,))
        return realized_pnl(rows)

    # ------------------------------------------------------------------ risk context
    def _last_entries(self) -> dict[str, datetime]:
        rows = self.ctx.db.query(
            "SELECT symbol, MAX(submitted_at) AS last FROM orders WHERE intent IN ('open_long','open_short') "
            "AND status NOT IN ('rejected') GROUP BY symbol")
        out = {}
        for r in rows:
            dt = parse_iso(r["last"])
            if dt:
                out[r["symbol"]] = dt
        return out

    async def _asset(self, symbol: str) -> dict | None:
        cached = self._asset_cache.get(symbol)
        if cached and time.monotonic() - cached[0] < 3600:
            return cached[1]
        info = await asyncio.to_thread(self.broker.asset, symbol)
        self._asset_cache[symbol] = (time.monotonic(), info)
        return info

    async def _risk_context(self, symbol: str) -> RiskContext:
        if self.account is None:
            raise RuntimeError(self.last_error or "Alpaca account unavailable")
        s = self.ctx.config.settings
        a = self.account
        # entry orders (always market orders) that haven't filled yet
        pending = {o["symbol"] for o in self.open_orders
                   if o.get("status") in OPEN_STATUSES and not (o.get("filled_qty") or 0)
                   and o.get("order_type") == "market"}
        asset = await self._asset(symbol)
        return RiskContext(
            trading=s.trading, risk=s.risk,
            account=AccountSnapshot(equity=a["equity"], last_equity=a["last_equity"], buying_power=a["buying_power"],
                                    cash=a["cash"] or 0, shorting_enabled=a["shorting_enabled"],
                                    trading_blocked=a["trading_blocked"]),
            positions=[PositionInfo(p["symbol"], p["qty"] or 0, p["market_value"] or 0) for p in self.positions],
            market_open=bool(self.clock and self.clock.get("is_open")),
            now=datetime.now(UTC),
            kill_engaged=self.ctx.state.kill_engaged,
            halted_today=self.ctx.state.halted_today,
            pending_symbols=pending,
            last_entry_at=self._last_entries(),
            asset=None if asset is None else AssetInfo(asset["tradable"], asset["shortable"], asset["easy_to_borrow"]),
        )

    def holding_long(self, symbol: str) -> bool:
        return any(p["symbol"] == symbol and (p.get("qty") or 0) > 0 for p in self.positions)

    # ------------------------------------------------------------------ signal handling
    async def handle_signal(self, signal: dict, manual: bool = False) -> dict:
        """signal needs: id (optional), ticker, direction, confidence. Returns the action taken."""
        symbol = signal["ticker"].upper()
        if self.broker is None:
            return self._result("blocked", "Not connected to Alpaca (check API keys).")
        async with self._lock:
            try:
                await self.refresh()
                action, why = decide_action(signal["direction"], int(signal["confidence"]),
                                            self.ctx.config.settings.trading, self.holding_long(symbol))
                if manual and action in ("review", "ignore") and signal["direction"] in ("bullish", "bearish"):
                    # You approved it by hand: act as if it cleared the buy threshold.
                    action = "buy" if signal["direction"] == "bullish" else (
                        "sell" if self.holding_long(symbol) else "short")
                    why = "Manually approved."
                if action == "review":
                    return self._result("review", why)
                if action == "ignore":
                    return self._result("ignored", why)
                if action == "sell":
                    return await self._close_long(symbol, signal, manual, why)
                return await self._open(symbol, "buy" if action == "buy" else "short", signal, manual, why)
            except Exception as exc:
                log.exception("Trade for %s failed", symbol)
                await self._alert("error", f"Trade error ({symbol})", str(exc), "error")
                return self._result("error", f"Error: {exc}")

    async def _open(self, symbol: str, side: str, signal: dict, manual: bool, why: str) -> dict:
        rctx = await self._risk_context(symbol)
        price = await asyncio.to_thread(self.broker.latest_price, symbol)
        decision = check_entry(symbol, side, price, rctx, manual=manual)
        if decision.halt_for_day and not self.ctx.state.halted_today:
            self.ctx.state.halt_for_today(decision.reason)
            await self._alert("daily_loss_limit", "Daily loss limit hit - trading stopped", decision.reason, "error")
        if not decision.allowed:
            log.info("Blocked %s %s: %s", side, symbol, decision.reason)
            return self._result("blocked", decision.reason)
        coid = f"nt-{signal.get('id') or 'm'}-{uuid.uuid4().hex[:10]}"
        try:
            order = await asyncio.to_thread(self.broker.submit_bracket, symbol, "buy" if side == "buy" else "sell",
                                            decision.qty, decision.take_profit_price, decision.stop_price, coid)
        except Exception as exc:
            msg = f"Alpaca rejected the {side} order for {symbol}: {exc}"
            log.error(msg)
            await self._alert("error", f"Order rejected ({symbol})", str(exc), "error")
            return self._result("error", msg)
        intent = "open_long" if side == "buy" else "open_short"
        db_id = self._record_order(order, signal, intent, decision.reason + (" [manual]" if manual else ""),
                                   est_price=decision.price, stop=decision.stop_price, tp=decision.take_profit_price)
        verb = "Bought" if side == "buy" else "Shorted"
        title = f"{verb} {decision.qty} {symbol} (~${decision.notional:,.0f})"
        detail = (f"Confidence {signal.get('confidence')} · stop ${decision.stop_price:,.2f} · "
                  f"target ${decision.take_profit_price:,.2f}")
        if signal.get("reasoning"):
            detail += f"\n{signal['reasoning']}"
        log.info("%s - %s", title, detail.replace("\n", " | "))
        await self._alert("trade_placed", title, detail, "trade")
        await self.refresh(force=True)
        return self._result("bought" if side == "buy" else "shorted", decision.reason, traded=True,
                            order_db_id=db_id, alpaca_order_id=order["id"])

    async def _close_long(self, symbol: str, signal: dict, manual: bool, why: str) -> dict:
        rctx = await self._risk_context(symbol)
        decision = check_exit(symbol, rctx, manual=manual)
        if not decision.allowed:
            return self._result("blocked", decision.reason)
        try:
            order = await asyncio.to_thread(self.broker.close_position, symbol)
        except Exception as exc:
            await self._alert("error", f"Couldn't sell {symbol}", str(exc), "error")
            return self._result("error", f"Couldn't sell {symbol}: {exc}")
        db_id = self._record_order(order, signal, "close_long", decision.reason)
        title = f"Sold all {symbol} on bearish signal"
        await self._alert("trade_placed", title, signal.get("reasoning") or "", "trade")
        await self.refresh(force=True)
        return self._result("sold", decision.reason, traded=True, order_db_id=db_id, alpaca_order_id=order["id"])

    def _save_order_row(self, values: dict) -> int:
        """Insert, or update if the trade stream already inserted this Alpaca order id."""
        existing = self.ctx.db.query_one("SELECT id FROM orders WHERE alpaca_order_id = ?", (values["alpaca_order_id"],))
        if existing is None:
            return self.ctx.db.insert("orders", values)
        keep_status = {k: v for k, v in values.items() if k not in ("status", "filled_qty", "filled_avg_price", "filled_at")}
        self.ctx.db.update("orders", existing["id"], keep_status)
        return existing["id"]

    def _record_order(self, order: dict, signal: dict, intent: str, reason: str, est_price=None, stop=None,
                      tp=None) -> int:
        db_id = self._save_order_row({
            "signal_id": signal.get("id"), "alpaca_order_id": order["id"], "client_order_id": order.get("client_order_id"),
            "symbol": order["symbol"], "side": order.get("side"), "intent": intent, "qty": order.get("qty"),
            "order_type": order.get("order_type"), "order_class": order.get("order_class"),
            "status": order.get("status"), "entry_price_est": est_price, "stop_price": stop, "take_profit_price": tp,
            "filled_qty": order.get("filled_qty"), "filled_avg_price": order.get("filled_avg_price"),
            "submitted_at": order.get("submitted_at") or iso(), "filled_at": order.get("filled_at"),
            "updated_at": iso(), "mode": self.broker.mode, "reason": reason, "raw": json.dumps(order)[:20000],
        })
        for leg in order.get("legs") or []:
            intent_leg = "take_profit" if leg.get("order_type") == "limit" else "stop_loss"
            self._save_order_row({
                "signal_id": signal.get("id"), "alpaca_order_id": leg["id"],
                "client_order_id": leg.get("client_order_id"), "symbol": leg["symbol"], "side": leg.get("side"),
                "intent": intent_leg, "qty": leg.get("qty"), "order_type": leg.get("order_type"),
                "order_class": leg.get("order_class"), "status": leg.get("status"),
                "stop_price": leg.get("stop_price"),
                "take_profit_price": leg.get("limit_price") if intent_leg == "take_profit" else None,
                "submitted_at": leg.get("submitted_at") or iso(), "updated_at": iso(), "mode": self.broker.mode,
                "parent_alpaca_id": order["id"], "reason": "Bracket exit leg", "raw": json.dumps(leg)[:20000]})
        self.ctx.bus.publish("orders_changed", {})
        return db_id

    @staticmethod
    def _result(action: str, reason: str, traded: bool = False, order_db_id: int | None = None,
                alpaca_order_id: str | None = None) -> dict:
        return {"action": action, "reason": reason, "traded": traded, "order_db_id": order_db_id,
                "alpaca_order_id": alpaca_order_id}

    # ------------------------------------------------------------------ manual controls
    async def manual_close(self, symbol: str) -> dict:
        if self.broker is None:
            raise RuntimeError("Not connected to Alpaca")
        async with self._lock:
            order = await asyncio.to_thread(self.broker.close_position, symbol.upper())
            pos_long = order.get("side") == "sell"
            self._record_order(order, {}, "close_long" if pos_long else "close_short", "Closed manually")
            log.info("Manually closed %s", symbol)
        await self.refresh(force=True)
        return order

    async def cancel_order(self, alpaca_order_id: str) -> None:
        if self.broker is None:
            raise RuntimeError("Not connected to Alpaca")
        await asyncio.to_thread(self.broker.cancel_order, alpaca_order_id)
        await self.refresh(force=True)

    async def kill(self, close_positions: bool = False, reason: str = "Kill switch pressed") -> dict:
        """Stop all trading and cancel every open order (optionally flatten every position)."""
        self.ctx.state.engage_kill_switch(reason)
        log.warning("KILL SWITCH engaged (%s). close_positions=%s", reason, close_positions)
        result = {"cancelled": 0, "closed": 0, "errors": []}
        if self.broker is not None:
            async with self._lock:
                try:
                    result["cancelled"] = await asyncio.to_thread(self.broker.cancel_all_orders)
                except Exception as exc:
                    result["errors"].append(f"cancel orders: {exc}")
                if close_positions:
                    try:
                        result["closed"] = await asyncio.to_thread(self.broker.close_all_positions)
                    except Exception as exc:
                        result["errors"].append(f"close positions: {exc}")
            await self.refresh(force=True)
        msg = f"Cancelled {result['cancelled']} open orders"
        if close_positions:
            msg += f", closing {result['closed']} positions"
        if result["errors"]:
            msg += ". Problems: " + "; ".join(result["errors"])
        await self._alert("kill_switch", "KILL SWITCH engaged - all trading stopped", msg, "error")
        return result

    async def rearm(self) -> None:
        self.ctx.state.release_kill_switch()
        log.warning("Trading re-armed (kill switch released)")
        await self._alert("kill_switch", "Trading re-armed", "Kill switch released by you.", "info")

    # ------------------------------------------------------------------ trade log
    def trade_log(self, limit: int = 500, symbol: str | None = None, mode: str | None = None) -> list[dict]:
        sql = ("SELECT o.*, s.confidence, s.direction, s.reasoning AS signal_reasoning, s.source_name "
               "FROM orders o LEFT JOIN signals s ON s.id = o.signal_id WHERE 1=1")
        params: list = []
        if symbol:
            sql += " AND o.symbol = ?"
            params.append(symbol.upper())
        if mode:
            sql += " AND o.mode = ?"
            params.append(mode)
        sql += " ORDER BY COALESCE(o.submitted_at, o.updated_at) DESC LIMIT ?"
        params.append(limit)
        rows = self.ctx.db.query(sql, params)
        pnl = self.realized_pnl_map()
        for r in rows:
            r.pop("raw", None)
            r["realized_pl"] = pnl.get(r["alpaca_order_id"])
        return rows

    def today_realized(self) -> float:
        day_start = datetime.now(UTC) - timedelta(hours=24)
        pnl = self.realized_pnl_map()
        rows = self.ctx.db.query("SELECT alpaca_order_id FROM orders WHERE filled_at >= ?", (iso(day_start),))
        return round(sum(pnl.get(r["alpaca_order_id"], 0) for r in rows), 2)

    # ------------------------------------------------------------------ alerts
    async def _alert(self, kind: str, title: str, message: str, level: str = "info") -> None:
        alerts = self.ctx.service("alerts")
        if alerts is not None:
            try:
                await alerts.send(kind, title, message, level)
                return
            except Exception:
                log.debug("alert failed", exc_info=True)
        self.ctx.bus.publish("toast", {"kind": "error" if level == "error" else level, "title": title,
                                       "message": message})

