"""The MarketMonitor service. About once a minute while the market is open it checks:

- the stocks you care about (positions, open orders, watchlist, stocks with recent signals, today's top movers)
  for sudden spikes, and looks for the news behind each one,
- the market as a whole (S&P 500, Nasdaq... through their ETFs) and world markets (US-listed country ETFs),

and keeps today's top movers for the Market tab. The detection rules are in detect.py, the wording in wording.py.

Alerts go through the AlertManager ("market_spike" / "market_move"). The monitor throttles them itself: one alert
per stock and kind per cooldown, and only the most significant few per check. Every event is still stored (and
shown on the Market tab) even when it isn't alerted. When the market is closed it only refreshes prices and movers
every 10 minutes for the tab - no spike or move alerts.

Data problems are logged as warnings, never errors (errors become alerts).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
from datetime import UTC, datetime, timedelta

from ..context import AppContext
from ..db import iso, parse_iso, utcnow
from ..performance.prices import price_at
from ..state import MARKET_TZ, trading_day
from . import wording
from .detect import (
    BASELINE_MINUTES,
    MARKET_WINDOW_MINUTES,
    DayLevels,
    EventMemory,
    WindowStats,
    detect_spike,
    market_phase,
    market_window_move,
    window_stats,
    world_alert_worthy,
)

log = logging.getLogger(__name__)

START_DELAY = 8                       # seconds after start-up before the first check
BUSY_FACTOR = 3                       # check this many times less often while a model trains or a backtest runs
MOVERS_EVERY = timedelta(minutes=5)   # Alpaca's top movers / most active lists
CLOSED_REFRESH = timedelta(minutes=10)  # market closed: refresh prices for the tab this often
BARS_CHUNK = 50                       # symbols per bars request
NEWS_WINDOW = timedelta(minutes=90)   # news behind a spike: signals / stories from the last 90 minutes...
NEWS_LOOKUPS_PER_CHECK = 3            # ...else ask Alpaca's news for at most this many stocks per check
BARS_CACHE_FRESH = timedelta(minutes=5)  # the chase guard may reuse bars this recent
KEEP_DAYS = 30
RECENT_EVENTS = 100
TOP_SUMMARY = 5
_SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")


def build_universe(groups: list[tuple[str, list[str]]], max_symbols: int, index_symbols: list[str],
                   world_symbols: list[str]) -> tuple[dict[str, list[str]], int]:
    """Which symbols to watch, and why. Returns ({symbol: [reasons]}, number of stocks left out).

    groups come in priority order (positions, open orders, watchlist, recent signals, top movers, most active):
    once max_symbols stocks are in, later ones are left out. The market gauges ("index") and world-market ETFs
    ("world") are always added on top."""
    out: dict[str, list[str]] = {}
    dropped: set[str] = set()
    for reason, symbols in groups:
        for raw in symbols or []:
            sym = str(raw or "").strip().upper()
            if not _SYMBOL_RE.match(sym):
                continue
            if sym in out:
                if reason not in out[sym]:
                    out[sym].append(reason)
            elif len(out) < max_symbols:
                out[sym] = [reason]
            else:
                dropped.add(sym)
    for reason, symbols in (("index", index_symbols), ("world", world_symbols)):
        for sym in symbols:
            reasons = out.setdefault(sym, [])
            if reason not in reasons:
                reasons.append(reason)
    return out, len(dropped)


def load_events(db, limit: int = RECENT_EVENTS, symbol: str = "", kind: str = "", days: int | None = None) -> list[dict]:
    """Stored market events, newest first (detail parsed from JSON)."""
    sql = "SELECT * FROM market_events WHERE 1=1"
    params: list = []
    if symbol:
        sql += " AND symbol = ?"
        params.append(symbol.upper().strip())
    if kind == "spike":
        sql += " AND kind IN ('spike_up', 'spike_down')"
    elif kind:
        sql += " AND kind = ?"
        params.append(kind)
    if days:
        sql += " AND ts >= ?"
        params.append(iso(utcnow() - timedelta(days=days)))
    sql += " ORDER BY ts DESC, id DESC LIMIT ?"
    params.append(limit)
    rows = db.query(sql, params)
    for r in rows:
        try:
            r["detail"] = json.loads(r["detail"] or "{}")
        except (TypeError, ValueError):
            r["detail"] = {}
    return rows


def _r(v: float | None, digits: int = 2) -> float | None:
    return None if v is None else round(v, digits)


def _tile(sym: str, snap: dict | None, move: float | None = None) -> dict:
    snap = snap or {}
    return {"symbol": sym, "name": wording.market_name(sym), "price": snap.get("price"),
            "prev_close": snap.get("prev_close"), "change_pct": _r(snap.get("change_pct")), "move_15m": _r(move)}


def empty_payload(settings) -> dict:
    """What the Market tab gets when the monitor isn't running."""
    s = settings.market
    return {
        "enabled": s.enabled, "running": False, "phase": None, "feed": settings.trading.data_feed.upper(),
        "last_scan": None, "last_refresh": None, "error": None,
        "indices": [_tile(sym, None) for sym in s.index_symbols],
        "world": [_tile(sym, None) for sym in s.world_symbols],
        "movers": {"gainers": [], "losers": [], "updated": None, "error": None},
        "most_actives": {"items": [], "updated": None, "error": None},
        "watching": [], "counts": {"watching": 0, "events_today": 0, "truncated": 0},
        "thresholds": _thresholds(s),
    }


def _thresholds(s) -> dict:
    return {"spike_pct": s.spike_pct, "spike_window_minutes": s.spike_window_minutes, "volume_ratio": s.volume_ratio,
            "market_move_pct": s.market_move_pct, "market_day_step_pct": s.market_day_step_pct,
            "min_price": s.min_price}


class MarketMonitor:
    name = "market"

    def __init__(self, ctx: AppContext, tickers=None):
        self.ctx = ctx
        self._tickers = tickers
        self._task: asyncio.Task | None = None
        self._changed = asyncio.Event()
        self._lock = asyncio.Lock()
        self.levels = DayLevels()
        self.memory = EventMemory()
        self._alerted_at: dict[tuple[str, str], datetime] = {}
        self._session: tuple[str, dict | None] | None = None
        self._movers_at: datetime | None = None
        self._refreshed_at: datetime | None = None
        self._truncation_noted = False
        self._warned: set[str] = set()
        self._events_today: tuple[str, int] = ("", 0)
        self.phase = "closed"
        self.last_scan: str | None = None
        self.last_error: str | None = None
        self.universe: dict[str, list[str]] = {}
        self.truncated = 0
        self.recent_bars: dict[str, list[dict]] = {}
        self._bars_at: datetime | None = None
        self.stats: dict[str, WindowStats] = {}
        self.index_moves: dict[str, float | None] = {}
        self.snapshots: dict[str, dict] = {}
        self.movers: dict = {"gainers": [], "losers": [], "updated": None}
        self.most_actives: dict = {"items": [], "updated": None}
        self.movers_error: str | None = None

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        self.ctx.config.on_change(lambda _s: self._signal_change())
        await asyncio.to_thread(self._load_state)
        if not self.ctx.config.settings.market.enabled:
            self._set_status("off", "Turned off (Settings -> Market monitor)")
        self._task = asyncio.create_task(self._loop(), name="market-monitor")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task

    async def on_keys_changed(self) -> None:
        self._movers_at = self._refreshed_at = None
        self._signal_change()

    def _signal_change(self) -> None:
        loop = self.ctx.loop
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(self._changed.set)

    def _settings_changed(self) -> None:
        self._truncation_noted = False
        self._movers_at = self._refreshed_at = None

    async def _loop(self) -> None:
        await asyncio.sleep(START_DELAY)
        while True:
            try:
                await self.run_once()
            except Exception as exc:
                self.last_error = str(exc)
                self._set_status("warn", f"Market check failed: {exc}")
                log.warning("Market check failed: %s", exc)
                log.debug("market check traceback", exc_info=True)
            woke = False
            try:
                await asyncio.wait_for(self._changed.wait(), timeout=self.next_interval())
                woke = True
            except TimeoutError:
                pass
            if woke:
                self._changed.clear()
                self._settings_changed()
                await asyncio.sleep(0.5)  # let rapid saves settle

    def busy(self) -> bool:
        """A model is training or a backtest is running - they need the market-data allowance more."""
        return any(getattr(self.ctx.service(name), "running", False) for name in ("ml_trainer", "backtest"))

    def next_interval(self) -> float:
        secs = float(self.ctx.config.settings.market.scan_seconds)
        return secs * BUSY_FACTOR if self.busy() else secs

    def _load_state(self, now: datetime | None = None) -> None:
        """Prune old events and pick up today's state after a restart (so nothing is re-alerted)."""
        db = self.ctx.db
        now = now or utcnow()
        db.execute("DELETE FROM market_events WHERE ts < ?", (iso(now - timedelta(days=KEEP_DAYS)),))
        day = trading_day(now)
        self._events_today = (day, int(db.scalar("SELECT COUNT(*) FROM market_events WHERE trading_day = ?",
                                                 (day,)) or 0))
        for r in db.query("SELECT symbol, kind, ts, change_pct, window_min, alerted, detail FROM market_events "
                          "WHERE trading_day = ? OR ts >= ? ORDER BY ts", (day, iso(now - timedelta(hours=2)))):
            ts = parse_iso(r["ts"])
            if ts is None or not r["symbol"]:
                continue
            try:
                detail = json.loads(r["detail"] or "{}")
            except ValueError:
                detail = {}
            if r["kind"] == "market_move" and detail.get("level_pct") is not None:
                if trading_day(ts) == day:
                    self.levels.mark(r["symbol"], day, float(detail["level_pct"]), float(detail.get("step_pct") or 1))
                continue
            key = (r["symbol"], "market_window" if r["kind"] == "market_move" else r["kind"])
            self.memory.remember(key, ts, float(r["change_pct"] or 0))
            if r["alerted"]:
                self._alerted_at[(r["symbol"], r["kind"])] = ts

    # ------------------------------------------------------------------ helpers
    def _broker(self):
        return getattr(self.ctx.service("trader"), "broker", None)

    def _ticker_table(self):
        return self._tickers or getattr(self.ctx.service("pipeline"), "tickers", None)

    def _feed(self) -> str:
        broker = self._broker()
        return str(getattr(broker, "data_feed", None) or self.ctx.config.settings.trading.data_feed).upper()

    def _clock_text(self, when: datetime) -> str:
        tz = getattr(getattr(self.ctx.config.settings, "ui", None), "time_zone", "local")
        if tz == "market":
            return when.astimezone(MARKET_TZ).strftime("%H:%M") + " ET"
        if tz == "utc":
            return when.astimezone(UTC).strftime("%H:%M") + " UTC"
        return when.astimezone().strftime("%H:%M")

    def _set_status(self, level: str, detail: str) -> None:
        self.ctx.state.set_status("market", level, detail)

    def _warn_once(self, key: str, msg: str, *args) -> None:
        """Log a data problem as a warning the first time, then quietly until it works again."""
        if key in self._warned:
            log.debug(msg, *args)
        else:
            self._warned.add(key)
            log.warning(msg, *args)

    def events_today(self) -> int:
        day, n = self._events_today
        return n if day == trading_day() else 0

    def _stock_symbols(self) -> list[str]:
        return [s for s, why in self.universe.items() if "index" not in why and "world" not in why]

    def _reference_symbol(self) -> str | None:
        """SPY (or the first market gauge) - what "the market" means for relative moves."""
        idx = self.ctx.config.settings.market.index_symbols
        return "SPY" if "SPY" in idx else (idx[0] if idx else None)

    # ------------------------------------------------------------------ public
    def summary(self) -> dict:
        """Tiny summary for the 5-second heartbeat."""
        spy = self.snapshots.get(self._reference_symbol() or "") or {}
        return {"enabled": self.ctx.config.settings.market.enabled, "phase": self.phase,
                "last_scan": self.last_scan, "spy_change_pct": _r(spy.get("change_pct")),
                "events_today": self.events_today()}

    def payload(self) -> dict:
        """Everything the Market tab shows (except the event list, which the API reads from the database)."""
        settings = self.ctx.config.settings
        s = settings.market
        out = empty_payload(settings)
        out.update({
            "running": True, "phase": self.phase, "feed": self._feed(), "last_scan": self.last_scan,
            "last_refresh": iso(self._refreshed_at) if self._refreshed_at else None, "error": self.last_error,
            "indices": [_tile(sym, self.snapshots.get(sym), self.index_moves.get(sym)) for sym in s.index_symbols],
            "world": [_tile(sym, self.snapshots.get(sym)) for sym in s.world_symbols],
            "movers": {**self.movers, "error": self.movers_error},
            "most_actives": {**self.most_actives, "error": self.movers_error},
            "watching": self._watching(),
            "counts": {"watching": len(self._stock_symbols()), "events_today": self.events_today(),
                       "truncated": self.truncated},
        })
        return out

    def _watching(self) -> list[dict]:
        table = self._ticker_table()
        w = self.ctx.config.settings.market.spike_window_minutes
        out = []
        for sym in self._stock_symbols():
            snap = self.snapshots.get(sym) or {}
            st = self.stats.get(sym)
            info = table.get(sym) if table is not None else None
            out.append({"symbol": sym, "name": wording.company_name(info.name) if info else "",
                        "reasons": self.universe[sym],
                        "price": (st.price if st and st.price else None) or snap.get("price"),
                        "change_pct": _r(snap.get("change_pct")), "move_pct": _r(st.change_pct) if st else None,
                        "window_min": w, "volume_ratio": _r(st.volume_ratio, 1) if st else None,
                        "stale": st.stale if st else None, "bars": st.bars if st else 0})
        return out

    def price_near(self, symbol: str, when: datetime, now: datetime | None = None) -> float | None:
        """The price at `when` from the bars the last check fetched (no API call). None when that check is more
        than a few minutes old or its bars don't reach back that far."""
        bars = self.recent_bars.get(symbol.upper())
        now = now or utcnow()
        if not bars or self._bars_at is None or now - self._bars_at > BARS_CACHE_FRESH:
            return None
        first = parse_iso(bars[0].get("t"))
        if first is None or first > when:
            return None
        return price_at(bars, when)

    async def run_once(self, now: datetime | None = None, force: bool = False) -> dict:
        """One check (public so tests and "Check now" can call it). force: refresh even when the market is closed
        and the last refresh is recent."""
        async with self._lock:
            return await self._check(now or utcnow(), force)

    # ------------------------------------------------------------------ one check
    async def _check(self, now: datetime, force: bool) -> dict:
        s = self.ctx.config.settings.market
        if not s.enabled:
            self.phase = "off"
            self._set_status("off", "Turned off (Settings -> Market monitor)")
            return {"status": "off"}
        broker = self._broker()
        if broker is None:
            self._set_status("off", "Waiting for an Alpaca connection (Settings -> API keys)")
            return {"status": "no_broker"}
        self.phase = await self._phase(broker, now, s)
        if self.phase == "closed":
            if force or self._refreshed_at is None or now - self._refreshed_at >= CLOSED_REFRESH:
                await self._refresh_closed(broker, s, now)
            when = f" · prices from {self._clock_text(self._refreshed_at)}" if self._refreshed_at else ""
            self._set_status("ok", f"Market closed - no spike alerts{when} · {self._feed()}")
            return {"status": "closed", "watching": len(self._stock_symbols())}
        return await self._scan(broker, s, now)

    async def _phase(self, broker, now: datetime, s) -> str:
        clock = getattr(self.ctx.service("trader"), "clock", None)
        if clock and clock.get("is_open"):
            return "open"
        session = await self._trading_session(broker, now) if (s.extended_hours or clock is None) else None
        return market_phase(now, clock, session, s.extended_hours)

    async def _trading_session(self, broker, now: datetime) -> dict | None:
        day = now.astimezone(MARKET_TZ).date()
        if self._session is not None and self._session[0] == day.isoformat():
            return self._session[1]
        try:
            days = await asyncio.to_thread(broker.calendar, day, day)
        except Exception as exc:
            log.debug("market calendar unavailable: %s", exc)
            return None
        session = next((d for d in days or [] if d.get("date") == day.isoformat()), None)
        self._session = (day.isoformat(), session)
        return session

    async def _refresh_closed(self, broker, s, now: datetime) -> None:
        """Market closed: keep the tab's prices and movers reasonably fresh. No detection, no alerts."""
        self._refreshed_at = now
        await self._refresh_movers(broker, s, now, force=True)
        self.universe, self.truncated = await self._build_universe(s, now)
        self.stats, self.index_moves = {}, {}
        await self._fetch_snapshots(broker)
        self._publish_summary()

    async def _scan(self, broker, s, now: datetime) -> dict:
        self._refreshed_at = now
        await self._refresh_movers(broker, s, now)
        self.universe, self.truncated = await self._build_universe(s, now)
        if self.truncated and not self._truncation_noted:
            self._truncation_noted = True
            log.info("Market monitor: watching the first %d stocks; %d more were left out (Settings -> Market "
                     "monitor -> Max stocks watched)", s.max_symbols, self.truncated)
        try:
            bars = await self._fetch_bars(broker, list(self.universe), now, s)
            self._warned.discard("bars")
        except Exception as exc:
            self.last_error = f"Couldn't get prices: {exc}"
            self._set_status("warn", self.last_error)
            self._warn_once("bars", "Market monitor couldn't get prices: %s", exc)
            return {"status": "error", "error": str(exc)}
        await self._fetch_snapshots(broker)
        self.recent_bars, self._bars_at = bars, now
        events = self._detect(bars, s, now)
        await self._attach_news(broker, events, s, now)
        for ev in events:
            self._word(ev)
        to_alert = self._plan_alerts(events, s, now)
        stored = await asyncio.to_thread(self._store, events)
        for row in stored:
            log.info("Market: %s", row["detail"].get("title"))
            self.ctx.bus.publish("market_event", row)
        for ev in to_alert:
            await self._send_alert(ev)
        self.last_scan, self.last_error = iso(now), None
        self._publish_summary()
        n = len(self._stock_symbols())
        extra = " (outside regular market hours)" if self.phase == "extended" else ""
        self._set_status("ok", f"Watching {n} stock{'' if n == 1 else 's'}{extra} · {self._feed()} · "
                               f"last check {self._clock_text(now)}")
        return {"status": "ok", "phase": self.phase, "watching": len(self._stock_symbols()),
                "events": len(stored), "alerts": len(to_alert)}

    # ------------------------------------------------------------------ data
    async def _build_universe(self, s, now: datetime) -> tuple[dict[str, list[str]], int]:
        trader = self.ctx.service("trader")
        groups: list[tuple[str, list[str]]] = []
        if s.watch_positions and trader is not None:
            groups.append(("position", [p["symbol"] for p in getattr(trader, "positions", None) or []
                                        if p.get("qty")]))
            groups.append(("order", [o["symbol"] for o in getattr(trader, "open_orders", None) or []]))
        groups.append(("watchlist", list(s.watchlist)))
        if s.watch_signals_minutes:
            since = iso(now - timedelta(minutes=s.watch_signals_minutes))
            rows = await asyncio.to_thread(
                self.ctx.db.query,
                "SELECT ticker, MAX(created_at) AS last FROM signals WHERE created_at >= ? AND merged_into IS NULL "
                "AND direction IN ('bullish', 'bearish') AND COALESCE(action, '') != 'watch' GROUP BY ticker "
                "ORDER BY last DESC", (since,))
            groups.append(("signal", [r["ticker"] for r in rows]))
        if s.watch_movers:
            groups.append(("mover", [m["symbol"] for m in self.movers["gainers"] + self.movers["losers"]]))
            groups.append(("active", [m["symbol"] for m in self.most_actives["items"]]))
        return build_universe(groups, s.max_symbols, s.index_symbols, s.world_symbols)

    async def _refresh_movers(self, broker, s, now: datetime, force: bool = False) -> None:
        if not force and self._movers_at is not None and now - self._movers_at < MOVERS_EVERY:
            return
        self._movers_at = now
        problems = []
        if hasattr(broker, "movers"):
            try:
                raw = await asyncio.to_thread(broker.movers, s.movers_top)
                self.movers = {"gainers": self._screen(raw.get("gainers"), s.min_price),
                               "losers": self._screen(raw.get("losers"), s.min_price), "updated": raw.get("updated")}
            except Exception as exc:
                problems.append(str(exc))
        if hasattr(broker, "most_actives"):
            try:
                raw = await asyncio.to_thread(broker.most_actives, s.movers_top)
                self.most_actives = {"items": self._screen(raw.get("items"), s.min_price),
                                     "updated": raw.get("updated")}
            except Exception as exc:
                problems.append(str(exc))
        if problems:
            self.movers_error = "Alpaca's top-movers lists aren't available right now (some accounts don't have them)."
            self._warn_once("movers", "Top movers unavailable: %s", problems[0])
        else:
            self.movers_error = None
            self._warned.discard("movers")

    def _screen(self, rows: list[dict] | None, min_price: float) -> list[dict]:
        """Keep tradable, US-listed stocks (no warrants or OTC) priced at least min_price, with their names."""
        table = self._ticker_table()
        checkable = table is not None and getattr(table, "loaded", False)
        out = []
        for r in rows or []:
            sym = str(r.get("symbol") or "").upper()
            info = table.get(sym) if checkable else None
            if checkable and (info is None or not info.tradable):
                continue
            price = r.get("price") or (self.snapshots.get(sym) or {}).get("price")
            if price is not None and price < min_price:
                continue
            name = wording.company_name(info.name) if info else ""
            out.append({**r, "symbol": sym, "name": "" if name == sym else name, "price": price})
        return out

    async def _fetch_bars(self, broker, symbols: list[str], now: datetime, s) -> dict[str, list[dict]]:
        """The last ~35 minutes of 1-minute bars for every watched symbol (one request per 50 symbols)."""
        lookback = timedelta(minutes=max(35, s.spike_window_minutes + BASELINE_MINUTES + 5,
                                         MARKET_WINDOW_MINUTES + 10))
        out: dict[str, list[dict]] = {}
        for i in range(0, len(symbols), BARS_CHUNK):
            out.update(await asyncio.to_thread(broker.bars_multi, symbols[i:i + BARS_CHUNK], now - lookback, now,
                                               "1Min"))
        return out

    async def _fetch_snapshots(self, broker) -> None:
        if not hasattr(broker, "snapshots") or not self.universe:
            return
        try:
            self.snapshots = await asyncio.to_thread(broker.snapshots, list(self.universe))
            self._warned.discard("snapshots")
        except Exception as exc:
            self._warn_once("snapshots", "Market monitor couldn't get day changes: %s", exc)

    # ------------------------------------------------------------------ detection
    def _detect(self, bars: dict[str, list[dict]], s, now: datetime) -> list[dict]:
        w = s.spike_window_minutes
        day = trading_day(now)
        ref = self._reference_symbol()
        market_change = window_stats(ref, bars.get(ref, []), now, w).change_pct if ref else None
        events: list[dict] = []
        stats: dict[str, WindowStats] = {}
        index_moves: dict[str, float | None] = {}
        for sym, reasons in self.universe.items():
            rows = bars.get(sym, [])
            snap = self.snapshots.get(sym) or {}
            if "index" in reasons or "world" in reasons:
                stats[sym] = window_stats(sym, rows, now, w)
                world = "index" not in reasons
                if not world:
                    st15 = window_stats(sym, rows, now, MARKET_WINDOW_MINUTES)
                    index_moves[sym] = st15.change_pct
                    if (market_window_move(st15, s.market_move_pct)
                            and self.memory.is_new((sym, "market_window"), now, MARKET_WINDOW_MINUTES,
                                                   st15.change_pct, s.market_move_pct)):
                        events.append(self._market_window_event(sym, st15, s, now))
                level = self.levels.new_level(sym, day, snap.get("change_pct"), s.market_day_step_pct)
                if level is not None:
                    events.append(self._day_level_event(sym, level, snap, s, now, world))
                continue
            st = window_stats(sym, rows, now, w, None if sym == ref else market_change)
            stats[sym] = st
            screened_only = set(reasons) <= {"mover", "active"}
            kind = detect_spike(st, s.spike_pct, s.volume_ratio, s.min_price if screened_only else 0.0)
            if kind and self.memory.is_new((sym, kind), now, w, st.change_pct or 0.0, s.spike_pct):
                events.append(self._spike_event(sym, kind, st, reasons, s, now))
        self.stats, self.index_moves = stats, index_moves
        return events

    @staticmethod
    def _base(now: datetime, kind: str, scope: str, sym: str, **values) -> dict:
        return {"ts": iso(now), "trading_day": trading_day(now), "kind": kind, "scope": scope, "symbol": sym,
                "price": None, "ref_price": None, "change_pct": None, "window_min": None, "volume": None,
                "volume_ratio": None, "level": "warn", "news_id": None, "signal_id": None, "headline": None,
                "url": None, "alerted": 0, **values}

    def _spike_event(self, sym: str, kind: str, st: WindowStats, reasons: list[str], s, now: datetime) -> dict:
        surge = kind == "volume_surge"
        ev = self._base(now, kind, "stock", sym, price=st.price, ref_price=st.ref_price,
                        change_pct=_r(st.change_pct), window_min=st.window_min, volume=st.volume,
                        volume_ratio=_r(st.volume_ratio, 1), level="info" if surge else "warn")
        ev["detail"] = {"reasons": reasons, "rel_change_pct": _r(st.rel_change_pct),
                        "market_change_pct": _r(st.market_change_pct), "baseline_volume": _r(st.baseline_volume, 0),
                        "bars": st.bars}
        ev["_alert"] = None if surge else "market_spike"
        ev["_cooldown"] = (sym, kind)
        # sudden moves outrank slow day-change levels; a stock you hold comes first
        ev["_score"] = 1 + abs(st.change_pct or 0) / s.spike_pct + (1 if "position" in reasons else 0)
        return ev

    def _market_window_event(self, sym: str, st: WindowStats, s, now: datetime) -> dict:
        ev = self._base(now, "market_move", "market", sym, price=st.price, ref_price=st.ref_price,
                        change_pct=_r(st.change_pct), window_min=st.window_min, volume=st.volume)
        ev["detail"] = {"name": wording.market_name(sym)}
        ev["_alert"] = "market_move"
        ev["_cooldown"] = (sym, "market_move")
        ev["_score"] = 2 + abs(st.change_pct or 0) / s.market_move_pct  # a fast market-wide move comes first
        return ev

    def _day_level_event(self, sym: str, level: float, snap: dict, s, now: datetime, world: bool) -> dict:
        change = snap.get("change_pct")
        alertable = world_alert_worthy(change, s.market_day_step_pct) if world else True
        ev = self._base(now, "market_move", "world" if world else "market", sym, price=snap.get("price"),
                        ref_price=snap.get("prev_close"), change_pct=_r(change), volume=snap.get("day_volume"),
                        level="warn" if alertable else "info")
        ev["detail"] = {"name": wording.market_name(sym), "level_pct": level, "step_pct": s.market_day_step_pct}
        ev["_alert"] = "market_move" if alertable else None
        ev["_cooldown"] = None  # each level is reported once a day anyway
        ev["_score"] = abs(level) / s.market_day_step_pct + (0 if world else 1)
        return ev

    # ------------------------------------------------------------------ news behind a spike
    async def _attach_news(self, broker, events: list[dict], s, now: datetime) -> None:
        lookups = 0
        for ev in events:
            if ev["scope"] != "stock":
                continue
            found = await asyncio.to_thread(self._news_from_db, ev["symbol"], now)
            if found is None and s.lookup_news and lookups < NEWS_LOOKUPS_PER_CHECK and hasattr(broker, "news"):
                lookups += 1
                found = await self._news_from_alpaca(broker, ev["symbol"], now)
            if found:
                ev["detail"]["news_source"] = found.pop("source")
                ev.update(found)

    def _news_from_db(self, sym: str, now: datetime) -> dict | None:
        """The newest signal for the stock in the last 90 minutes, else the newest story tagged with it."""
        db = self.ctx.db
        since = iso(now - NEWS_WINDOW)
        sig = db.query_one("SELECT id, headline, url FROM signals WHERE ticker = ? AND created_at >= ? "
                           "AND merged_into IS NULL AND COALESCE(action, '') != 'watch' "
                           "ORDER BY created_at DESC, id DESC LIMIT 1", (sym, since))
        if sig is not None:
            return {"signal_id": sig["id"], "headline": sig["headline"], "url": sig["url"], "source": "signal"}
        rows = db.query("SELECT id, title, url, symbols FROM news_items WHERE received_at >= ? AND symbols LIKE ? "
                        "ORDER BY id DESC LIMIT 20", (since, f'%"{sym}"%'))
        for r in rows:
            try:
                tagged = sym in json.loads(r["symbols"] or "[]")
            except ValueError:
                tagged = False
            if tagged:
                return {"news_id": r["id"], "headline": r["title"], "url": r["url"], "source": "news"}
        return None

    async def _news_from_alpaca(self, broker, sym: str, now: datetime) -> dict | None:
        try:
            items = await asyncio.to_thread(broker.news, now - timedelta(hours=2), now, symbols=[sym], limit=3,
                                            include_content=False, newest_first=True)
        except Exception as exc:
            log.debug("news lookup for %s failed: %s", sym, exc)
            return None
        for n in items or []:
            if n.get("headline"):
                return {"headline": str(n["headline"])[:500], "url": n.get("url"), "source": "alpaca"}
        return None

    # ------------------------------------------------------------------ alerts
    def _word(self, ev: dict) -> None:
        d = ev["detail"]
        sym, change = ev["symbol"], ev["change_pct"] or 0.0
        if ev["scope"] == "stock":
            d["title"] = wording.spike_title(sym, ev["kind"], change, ev["window_min"], ev["volume_ratio"])
            d["message"] = wording.spike_message(ev["price"], ev["ref_price"], ev["volume_ratio"], ev["headline"])
            d["fields"] = {"Price": f"${ev['price']:,.2f}" if ev["price"] else "?",
                           "Move": f"{change:+.1f}% in {ev['window_min']} min",
                           "Volume": f"{ev['volume_ratio']:.1f}x usual" if ev["volume_ratio"] else "not enough data",
                           "News": ev["headline"] or "none found"}
        elif ev["window_min"]:
            d["title"] = wording.market_window_title(sym, change, ev["window_min"])
            d["message"] = wording.market_window_message(ev["price"], ev["ref_price"])
            d["fields"] = {"Move": f"{change:+.1f}% in {ev['window_min']} min"}
        else:
            d["title"] = wording.day_level_title(sym, d["level_pct"])
            d["message"] = wording.day_level_message(sym, ev["price"], ev["ref_price"], ev["change_pct"],
                                                     ev["scope"] == "world")
            d["fields"] = {"Today": f"{change:+.1f}%"}

    def _plan_alerts(self, events: list[dict], s, now: datetime) -> list[dict]:
        """Most significant first; skip turned-off alert types, stocks alerted within the cooldown, a second alert
        for the same symbol in one check, and anything past max_alerts_per_scan."""
        toggles = self.ctx.config.settings.alerts
        cooldown = timedelta(minutes=s.alert_cooldown_minutes)
        chosen: list[dict] = []
        symbols: set[str] = set()
        for ev in sorted(events, key=lambda e: -e["_score"]):
            kind = ev["_alert"]
            if kind is None or len(chosen) >= s.max_alerts_per_scan or ev["symbol"] in symbols:
                continue
            if not (toggles.on_market_spike if kind == "market_spike" else toggles.on_market_move):
                continue
            key = ev["_cooldown"]
            last = self._alerted_at.get(key) if key else None
            if last is not None and now - last < cooldown:
                continue
            ev["alerted"] = 1
            if key:
                self._alerted_at[key] = now
            symbols.add(ev["symbol"])
            chosen.append(ev)
        return chosen

    async def _send_alert(self, ev: dict) -> None:
        d = ev["detail"]
        alerts = self.ctx.service("alerts")
        if alerts is None:
            self.ctx.bus.publish("toast", {"kind": "warn", "title": d["title"], "message": d["message"]})
            return
        try:
            await alerts.send(ev["_alert"], d["title"], d["message"], "warn", fields=d.get("fields"))
        except Exception:
            log.debug("market alert failed", exc_info=True)

    def _store(self, events: list[dict]) -> list[dict]:
        out = []
        for ev in events:
            row = {k: v for k, v in ev.items() if not k.startswith("_")}
            detail = row.pop("detail")
            row["id"] = self.ctx.db.insert("market_events", {**row, "detail": json.dumps(detail)})
            out.append({**row, "detail": detail})
        if out:
            day, n = self._events_today
            today = trading_day()
            self._events_today = (today, (n if day == today else 0) + len(out))
        return out

    def _publish_summary(self) -> None:
        """A compact update for the Market tab after each check."""
        s = self.ctx.config.settings.market
        self.ctx.bus.publish("market", {
            "phase": self.phase, "last_scan": self.last_scan,
            "last_refresh": iso(self._refreshed_at) if self._refreshed_at else None,
            "indices": [_tile(sym, self.snapshots.get(sym), self.index_moves.get(sym)) for sym in s.index_symbols],
            "world": [_tile(sym, self.snapshots.get(sym)) for sym in s.world_symbols],
            "gainers": self.movers["gainers"][:TOP_SUMMARY], "losers": self.movers["losers"][:TOP_SUMMARY],
            "counts": {"watching": len(self._stock_symbols()), "events_today": self.events_today(),
                       "truncated": self.truncated},
        })
