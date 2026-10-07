"""Records the stock price when each signal fired, then again +5 minutes, +1 hour and +1 trading day later,
so the Performance tab can show whether the AI's calls actually worked.

Works from the database, so it catches up after a restart (missed checkpoints are filled from history).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import UTC, date, datetime, timedelta

from ..context import AppContext
from ..db import iso, parse_iso, utcnow
from .prices import directional_return, fallback_next_day, next_session_same_time, price_at

log = logging.getLogger(__name__)

CHECK_EVERY = 30  # seconds
DATA_DELAY = timedelta(minutes=2)  # wait for the minute bar to be published
BATCH = 40  # max price look-ups per cycle (Alpaca data API limit is 200/min)


class PerformanceTracker:
    name = "performance"

    def __init__(self, ctx: AppContext):
        self.ctx = ctx
        self._task: asyncio.Task | None = None
        self._calendar: list[dict] = []
        self._calendar_until: date | None = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._loop(), name="performance")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task

    def _broker(self):
        trader = self.ctx.service("trader")
        return getattr(trader, "broker", None)

    async def _loop(self) -> None:
        await asyncio.sleep(5)
        while True:
            try:
                await self.run_once()
            except Exception:
                log.debug("performance cycle failed", exc_info=True)
            await asyncio.sleep(CHECK_EVERY)

    async def run_once(self, now: datetime | None = None) -> int:
        broker = self._broker()
        if broker is None:
            return 0
        now = now or utcnow()
        done = await self._register_new(broker, now)
        done += await self._fill_due(broker, now)
        return done

    # ------------------------------------------------------------------ calendar
    async def _sessions(self, broker, around: datetime) -> list[dict]:
        need_until = (around + timedelta(days=10)).date()
        if self._calendar_until is None or need_until > self._calendar_until or not self._calendar:
            start = (around - timedelta(days=5)).date()
            try:
                self._calendar = await asyncio.to_thread(broker.calendar, start, start + timedelta(days=40))
                self._calendar_until = start + timedelta(days=40)
            except Exception as exc:
                log.debug("calendar unavailable: %s", exc)
                return []
        return self._calendar

    # ------------------------------------------------------------------ steps
    async def _register_new(self, broker, now: datetime) -> int:
        rows = self.ctx.db.query(
            "SELECT s.id, s.ticker, s.direction, s.created_at FROM signals s "
            "LEFT JOIN signal_prices p ON p.signal_id = s.id "
            "WHERE p.signal_id IS NULL AND s.merged_into IS NULL AND s.direction IN ('bullish','bearish') "
            "ORDER BY s.id LIMIT ?", (BATCH,))
        n = 0
        for r in rows:
            t0 = parse_iso(r["created_at"]) or now
            sessions = await self._sessions(broker, t0)
            due_1d = next_session_same_time(t0, sessions) if sessions else None
            due_1d = due_1d or fallback_next_day(t0)
            if now - t0 < timedelta(minutes=2):
                p0 = await asyncio.to_thread(broker.latest_price, r["ticker"])
            else:
                p0 = await self._historical_price(broker, r["ticker"], t0)
            self.ctx.db.insert("signal_prices", {
                "signal_id": r["id"], "ticker": r["ticker"], "direction": r["direction"], "t0": iso(t0),
                "price_t0": p0, "due_5m": iso(t0 + timedelta(minutes=5)), "due_1h": iso(t0 + timedelta(hours=1)),
                "due_1d": iso(due_1d.astimezone(UTC)), "status": "pending" if p0 else "no_price", "updated_at": iso(),
            })
            if p0:
                self.ctx.db.update("signals", r["id"], {"price_at_signal": p0})
            n += 1
        return n

    async def _historical_price(self, broker, ticker: str, when: datetime) -> float | None:
        try:
            bars = await asyncio.to_thread(broker.bars, ticker, when - timedelta(hours=4), when + timedelta(minutes=1))
        except Exception as exc:
            log.debug("bars failed for %s: %s", ticker, exc)
            return None
        return price_at(bars, when)

    async def _fill_due(self, broker, now: datetime) -> int:
        ready = iso(now - DATA_DELAY)
        rows = self.ctx.db.query(
            "SELECT * FROM signal_prices WHERE status = 'pending' AND "
            "((price_5m IS NULL AND due_5m <= ?) OR (price_1h IS NULL AND due_1h <= ?) OR (price_1d IS NULL AND due_1d <= ?)) "
            "ORDER BY signal_id LIMIT ?", (ready, ready, ready, BATCH))
        n = 0
        for r in rows:
            updates: dict = {}
            for key in ("5m", "1h", "1d"):
                due = parse_iso(r[f"due_{key}"])
                if r[f"price_{key}"] is None and due is not None and due <= now - DATA_DELAY:
                    p = await self._historical_price(broker, r["ticker"], due)
                    if p is None and now - due < timedelta(minutes=10):
                        p = await asyncio.to_thread(broker.latest_price, r["ticker"])
                    if p is not None:
                        updates[f"price_{key}"] = p
                        updates[f"ret_{key}"] = directional_return(r["direction"], r["price_t0"], p)
                    elif now - due > timedelta(days=3):
                        updates[f"price_{key}"] = 0  # give up - no trades in that stock
            if updates:
                merged = {**r, **updates}
                if all(merged[f"price_{k}"] is not None for k in ("5m", "1h", "1d")):
                    updates["status"] = "complete"
                updates["updated_at"] = iso()
                self.ctx.db.update("signal_prices", r["signal_id"], updates, id_col="signal_id")
                n += 1
        if n:
            self.ctx.bus.publish("performance_updated", {"updated": n})
        return n
