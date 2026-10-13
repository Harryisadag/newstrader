"""The CryptoExitManager: the stop-loss and take-profit for crypto positions, which Alpaca can't do with a bracket
order.

Every crypto buy gets an exit plan (table crypto_exits):
- a SAFETY STOP held at Alpaca: a good-until-cancelled stop-limit sell for the whole position at the stop price (its
  limit a little lower, crypto.STOP_LIMIT_GAP_PCT). It works even while NewsTrader is closed.
- a TAKE-PROFIT NewsTrader watches itself, about every CHECK_EVERY seconds, 24/7 while the app runs: when the price
  reaches the target it cancels the safety stop, then sells the whole position at market. So the take-profit only
  works while NewsTrader is open.

Every check - the first one right after start-up - re-syncs with Alpaca: a buy that filled gets its safety stop, a buy
that never filled ends its plan, a plan whose position is gone is closed (the stop filled, or it was sold), and a
position whose safety stop isn't working any more gets a new one. If the price falls through the stop's limit while
the stop is still open (a drop too fast for a stop-limit to fill) it sells at market. Only positions NewsTrader
opened are managed. Orders are only placed while holding the trader's lock, so a check never races a trade or the
kill switch. While the kill switch is engaged (it cancels the stops like every other order) or an update is being
installed it does nothing; once trading is re-armed the next check puts the safety stops back.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid

from ..context import AppContext
from ..db import Database, iso
from . import crypto

log = logging.getLogger(__name__)

CHECK_EVERY = 15.0     # seconds between checks
FAST_CHECK = 3.0       # ...while a buy is waiting to fill
START_DELAY = 2.0      # seconds after start-up before the first check (the trader connects first)
CANCEL_WAIT = 6.0      # seconds to wait for a cancelled safety stop to let go of the coins
OPEN_PLANS = ("pending", "active", "closing")
WORKING = {"new", "accepted", "held", "pending_new", "partially_filled", "accepted_for_bidding", "pending_replace",
           "replaced", "calculated"}
ENDED = {"canceled", "expired", "rejected", "done_for_day", "stopped", "suspended"}
DUST = 1e-9            # a position smaller than this is gone
CHANGED = {"stop placed", "stop placed again", "sold", "closed", "buy didn't fill"}  # results that changed orders


# ---------------------------------------------------------------------------------------------- plans in the database
def create_plan(db: Database, symbol: str, mode: str, signal_id, order: dict, stop: float | None,
                stop_limit: float | None, take_profit: float | None, entry_price: float | None) -> int:
    """A new plan for a crypto buy that was just sent (status "pending" until it fills)."""
    now = iso()
    return db.insert("crypto_exits", {
        "symbol": crypto.normalize(symbol), "mode": mode, "signal_id": signal_id, "entry_order_id": order["id"],
        "entry_price": entry_price, "qty": None, "stop_price": stop, "stop_limit_price": stop_limit,
        "take_profit_price": take_profit, "status": "pending", "reason": "Waiting for the buy to fill",
        "created_at": now, "updated_at": now})


def open_plans(db: Database, mode: str | None = None, symbol: str | None = None) -> list[dict]:
    sql = f"SELECT * FROM crypto_exits WHERE status IN ({', '.join('?' * len(OPEN_PLANS))})"
    params: list = list(OPEN_PLANS)
    if mode:
        sql += " AND mode = ?"
        params.append(mode)
    if symbol:
        sql += " AND symbol = ?"
        params.append(crypto.normalize(symbol))
    return db.query(sql + " ORDER BY id", params)


def mark_closing(db: Database, symbol: str, mode: str, exit_order_id: str | None, reason: str) -> int:
    """A sell of the whole position was sent (bearish news, or by hand): the plans for it end when it fills."""
    n = 0
    for plan in open_plans(db, mode, symbol):
        _update(db, plan["id"], status="closing", exit_order_id=exit_order_id, reason=reason)
        n += 1
    return n


def _update(db: Database, plan_id: int, **values) -> None:
    values["updated_at"] = iso()
    if values.get("status") == "closed":
        values["closed_at"] = values["updated_at"]
    db.update("crypto_exits", plan_id, values)


def _num(value) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


class Skip(Exception):
    """The plan changed while the check was waiting for the trade lock - look again next time."""


# ---------------------------------------------------------------------------------------------- the service
class CryptoExitManager:
    name = "crypto_exits"

    def __init__(self, ctx: AppContext):
        self.ctx = ctx
        self._task: asyncio.Task | None = None
        self._wake = asyncio.Event()
        self._running = asyncio.Lock()
        self.last_check: str | None = None
        self.last_error: str | None = None
        self.watching = 0

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        loop = asyncio.get_running_loop()

        def on_settings(_settings) -> None:  # called from whichever thread saved the settings
            if not loop.is_closed():
                loop.call_soon_threadsafe(self._wake.set)

        self.ctx.config.on_change(on_settings)
        self._task = asyncio.create_task(self._loop(), name="crypto-exits")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task

    async def on_keys_changed(self) -> None:
        self.kick()

    def kick(self) -> None:
        """Check soon (a buy was just sent, a fill came in, or trading was re-armed)."""
        self._wake.set()

    async def _loop(self) -> None:
        await asyncio.sleep(START_DELAY)
        while True:
            try:
                await self.run_once()
                self.last_error = None
            except Exception as exc:
                self.last_error = str(exc)
                self._status("warn", f"Crypto exits: the last check failed ({exc})")
                log.warning("Crypto exit check failed: %s", exc)
                log.debug("crypto exit traceback", exc_info=True)
            pending = any(p["status"] == "pending" for p in open_plans(self.ctx.db))
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._wake.wait(), FAST_CHECK if pending else CHECK_EVERY)
            self._wake.clear()

    # ------------------------------------------------------------------ helpers
    def _trader(self):
        return self.ctx.service("trader")

    def _broker(self):
        return getattr(self._trader(), "broker", None)

    def _status(self, level: str, detail: str) -> None:
        self.ctx.state.set_status("crypto", level, detail)

    def _paused(self) -> str:
        if self.ctx.state.kill_engaged:
            return ("Kill switch engaged - crypto safety stops were cancelled with every other order. They are put "
                    "back when you re-arm trading.")
        if getattr(self._trader(), "_held_for_update", False):
            return "Paused while NewsTrader installs an update."
        return ""

    @contextlib.asynccontextmanager
    async def _trade_lock(self, plan: dict):
        """Hold the trader's lock (so no trade or kill switch runs at the same time) and make sure the plan hasn't
        changed since the check read it."""
        lock = getattr(self._trader(), "_lock", None)
        async with lock if lock is not None else contextlib.nullcontext():
            if self._paused():
                raise Skip()
            fresh = self.ctx.db.query_one("SELECT * FROM crypto_exits WHERE id = ?", (plan["id"],))
            if fresh is None or fresh["status"] != plan["status"] or fresh["updated_at"] != plan["updated_at"]:
                raise Skip()
            yield fresh

    def _record(self, order: dict, plan: dict, intent: str, reason: str, stop=None, tp=None) -> None:
        trader = self._trader()
        if trader is not None and hasattr(trader, "record_exit_order"):
            trader.record_exit_order(order, plan.get("signal_id"), intent, reason, parent_id=plan["entry_order_id"],
                                     stop=stop, tp=tp)

    async def _alert(self, kind: str, title: str, message: str, level: str = "trade") -> None:
        trader = self._trader()
        if trader is not None and hasattr(trader, "_alert"):
            await trader._alert(kind, title, message, level)

    def plans_by_symbol(self, mode: str | None = None) -> dict[str, dict]:
        """The open plan of each crypto position (for the Portfolio tab)."""
        broker = self._broker()
        mode = mode or getattr(broker, "mode", None)
        return {p["symbol"]: p for p in open_plans(self.ctx.db, mode)}

    def summary(self) -> dict:
        s = self.ctx.config.settings.crypto
        broker = self._broker()
        return {"enabled": s.enabled, "supported": bool(getattr(broker, "supports_crypto", False)),
                "watching": self.watching, "last_check": self.last_check, "error": self.last_error}

    # ------------------------------------------------------------------ one check
    async def run_once(self) -> dict:
        async with self._running:
            return await self._check()

    async def _check(self) -> dict:
        broker = self._broker()
        enabled = self.ctx.config.settings.crypto.enabled
        if broker is None or not getattr(broker, "supports_crypto", False):
            if getattr(broker, "kind", "") == "simulator":
                self._status("off", "Crypto needs an Alpaca paper account - the built-in simulator can't trade it")
            elif enabled:
                self._status("off", "Crypto: waiting for an Alpaca connection")
            else:
                self._status("off", "Crypto is off (Settings -> Crypto)")
            return {"status": "no_broker"}
        plans = await asyncio.to_thread(open_plans, self.ctx.db, broker.mode)
        self.watching = sum(1 for p in plans if p["status"] != "pending")
        if not plans:
            self.last_check = iso()
            self._status("ok" if enabled else "off", "Crypto: no positions to look after" if enabled
                         else "Crypto is off (Settings -> Crypto)")
            return {"status": "idle"}
        paused = self._paused()
        if paused:
            self._status("warn", paused)
            return {"status": "paused"}
        positions = {p["symbol"]: p for p in await asyncio.to_thread(broker.positions)
                     if crypto.is_crypto(p["symbol"])}
        orders = await asyncio.to_thread(broker.orders, "open", 500)
        working = {o["id"]: o for o in orders if o.get("status") in WORKING}
        done: dict[int, str] = {}
        for plan in plans:
            try:
                done[plan["id"]] = await self._handle(plan, broker, positions, working)
            except Skip:
                done[plan["id"]] = "changed meanwhile"
            except Exception as exc:
                done[plan["id"]] = f"error: {exc}"
                log.warning("Crypto exit for %s: %s", plan["symbol"], exc)
                with contextlib.suppress(Exception):
                    _update(self.ctx.db, plan["id"], reason=f"Problem: {exc}"[:300])
        self.last_check = iso()
        trader = self._trader()
        if any(d in CHANGED for d in done.values()) and hasattr(trader, "refresh"):
            with contextlib.suppress(Exception):
                await trader.refresh()  # the Portfolio tab shows the new stop / sale straight away
        left = await asyncio.to_thread(open_plans, self.ctx.db, broker.mode)
        self.watching = sum(1 for p in left if p["status"] != "pending")
        problems = [d for d in done.values() if d.startswith("error")]
        n = self.watching
        text = (f"Crypto: looking after {n} position{'' if n == 1 else 's'} (safety stops at Alpaca, take-profits "
                f"while NewsTrader is open)")
        self._status("warn" if problems else "ok", text + (f" · {problems[0]}" if problems else ""))
        return {"status": "ok", "plans": done}

    async def _handle(self, plan: dict, broker, positions: dict[str, dict], working: dict[str, dict]) -> str:
        sym = plan["symbol"]
        pos = positions.get(sym)
        qty = _num(pos.get("qty")) if pos else 0.0
        if plan["status"] == "pending":
            return await self._pending(plan, broker, pos)
        if qty <= DUST:
            return await self._gone(plan, broker, working)
        stop = working.get(plan["stop_order_id"]) if plan["stop_order_id"] else None
        if plan["status"] == "closing":
            if plan["exit_order_id"] and plan["exit_order_id"] in working:
                return "selling"
            if plan["exit_order_id"]:
                sale = await asyncio.to_thread(broker.order, plan["exit_order_id"])
                if sale.get("status") == "filled" and _num(sale.get("filled_qty")) >= qty - DUST:
                    return await self._gone(plan, broker, working)  # Alpaca's position list is a moment behind
            async with self._trade_lock(plan) as fresh:  # the sell is over but some coins are left: protect them
                _update(self.ctx.db, fresh["id"], status="active", reason="Some coins were left after the sale")
            plan = self.ctx.db.query_one("SELECT * FROM crypto_exits WHERE id = ?", (plan["id"],))
        price = await asyncio.to_thread(broker.latest_price, sym)
        if price and plan["take_profit_price"] and price >= plan["take_profit_price"]:
            return await self._sell(plan, broker, stop, "take_profit",
                                    f"Crypto take-profit: {sym} reached ${price:,.2f} (target "
                                    f"${plan['take_profit_price']:,.2f}) - NewsTrader sold it")
        if stop is None:
            if price and plan["stop_price"] and price <= plan["stop_price"]:
                return await self._sell(plan, broker, None, "stop_loss",
                                        f"Crypto safety stop: {sym} is at ${price:,.2f}, already below the "
                                        f"${plan['stop_price']:,.2f} stop - NewsTrader sold it at market")
            return await self._place_stop(plan, broker, pos)
        if price and plan["stop_limit_price"] and price < plan["stop_limit_price"]:
            return await self._sell(plan, broker, stop, "stop_loss",
                                    f"Crypto safety stop: {sym} fell to ${price:,.2f}, past the stop's limit "
                                    f"(${plan['stop_limit_price']:,.2f}) - NewsTrader sold it at market")
        return "watching"

    # ------------------------------------------------------------------ steps
    async def _pending(self, plan: dict, broker, pos: dict | None) -> str:
        """The buy was sent: once it has filled (and Alpaca shows the coins), place the safety stop."""
        order = await asyncio.to_thread(broker.order, plan["entry_order_id"])
        trader = self._trader()
        if trader is not None and hasattr(trader, "track_order"):
            await trader.track_order(order)
        status = order.get("status")
        filled = _num(order.get("filled_qty"))
        if status != "filled" and not (status in ENDED and filled > 0):
            if status in ENDED:
                async with self._trade_lock(plan) as fresh:
                    _update(self.ctx.db, fresh["id"], status="closed", reason=f"The buy didn't fill ({status})")
                return "buy didn't fill"
            return "waiting for the fill"
        if pos is None or _num(pos.get("qty")) <= DUST:
            return "waiting for the position"  # Alpaca shows the position a moment after the fill
        entry = _num(order.get("filled_avg_price")) or _num(pos.get("avg_entry_price"))
        s = self.ctx.config.settings.crypto
        asset = await asyncio.to_thread(broker.asset, plan["symbol"]) if hasattr(broker, "asset") else None
        stop, limit, tp = crypto.exit_prices(entry, s.stop_loss_pct, s.take_profit_pct,
                                             (asset or {}).get("price_increment"))
        async with self._trade_lock(plan) as fresh:
            _update(self.ctx.db, fresh["id"], status="active", entry_price=entry, stop_price=stop,
                    stop_limit_price=limit, take_profit_price=tp, reason="")
        plan = self.ctx.db.query_one("SELECT * FROM crypto_exits WHERE id = ?", (plan["id"],))
        return await self._place_stop(plan, broker, pos)

    async def _place_stop(self, plan: dict, broker, pos: dict) -> str:
        """A safety stop for the whole position, held at Alpaca."""
        asset = await asyncio.to_thread(broker.asset, plan["symbol"]) if hasattr(broker, "asset") else None
        qty = _num(pos.get("qty"))
        if pos.get("qty_available") is not None:
            qty = min(qty, _num(pos.get("qty_available")))
        qty = crypto.round_down(qty, (asset or {}).get("min_trade_increment"))
        if qty <= DUST:
            return "no coins free for a stop yet"
        async with self._trade_lock(plan) as fresh:
            first = not fresh["stop_order_id"]
            coid = f"nt-stop-{fresh['id']}-{uuid.uuid4().hex[:8]}"
            order = await asyncio.to_thread(broker.submit_crypto_stop, fresh["symbol"], qty, fresh["stop_price"],
                                            fresh["stop_limit_price"], coid)
            _update(self.ctx.db, fresh["id"], stop_order_id=order["id"], qty=qty, status="active", reason="")
            self._record(order, fresh, "stop_loss", "Crypto safety stop (held at Alpaca)", stop=fresh["stop_price"])
        if first:
            log.info("Crypto safety stop for %g %s at $%s (limit $%s)", qty, plan["symbol"], plan["stop_price"],
                     plan["stop_limit_price"])
            return "stop placed"
        log.warning("Crypto safety stop for %s wasn't working any more - placed a new one", plan["symbol"])
        return "stop placed again"

    async def _gone(self, plan: dict, broker, working: dict[str, dict]) -> str:
        """The position is gone: the safety stop filled, the take-profit sold it, or it was sold another way."""
        why = plan["reason"] if plan["status"] == "closing" and plan["reason"] else "The position was closed"
        async with self._trade_lock(plan) as fresh:
            if fresh["stop_order_id"]:
                if fresh["stop_order_id"] in working:  # nothing left to protect
                    with contextlib.suppress(Exception):
                        await asyncio.to_thread(broker.cancel_order, fresh["stop_order_id"])
                else:
                    with contextlib.suppress(Exception):
                        stop = await asyncio.to_thread(broker.order, fresh["stop_order_id"])
                        if stop.get("status") == "filled":
                            why = f"The safety stop sold it at ${_num(stop.get('filled_avg_price')):,.2f}"
            _update(self.ctx.db, fresh["id"], status="closed", reason=why[:300])
        return "closed"

    async def _sell(self, plan: dict, broker, stop: dict | None, intent: str, reason: str) -> str:
        """Cancel the safety stop, wait until it lets go of the coins, then sell the whole position at market."""
        sym = plan["symbol"]
        async with self._trade_lock(plan) as fresh:
            if stop is not None:
                with contextlib.suppress(Exception):  # it may have just filled or been cancelled
                    await asyncio.to_thread(broker.cancel_order, stop["id"])
                deadline = time.monotonic() + CANCEL_WAIT
                while time.monotonic() < deadline:
                    o = await asyncio.to_thread(broker.order, stop["id"])
                    if o.get("status") not in WORKING and o.get("status") != "pending_cancel":
                        break
                    await asyncio.sleep(0.5)
            order = await asyncio.to_thread(broker.close_position, sym)
            _update(self.ctx.db, fresh["id"], status="closing", exit_order_id=order["id"], reason=reason[:300])
            self._record(order, fresh, intent, reason, tp=fresh["take_profit_price"] if intent == "take_profit" else None)
        title = (f"Crypto take-profit: selling {sym}" if intent == "take_profit" else f"Crypto stop: selling {sym}")
        log.info(reason)
        await self._alert("trade_placed", title, reason)
        return "sold"
