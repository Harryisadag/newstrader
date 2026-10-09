"""The ChartService: chart readings (reading.py) for the app, without hammering the market-data API.

Bars it keeps for each stock:
- 1-minute bars from 4:00 New York time on the trading day before the latest one (two sessions: "normal volume for
  this time of day" needs an earlier session). After the first download they are kept up to date from the bars the
  market monitor fetches every minute anyway (market/monitor.py), so a watched stock costs no extra requests; any
  other stock only fetches the minutes it is missing, at most about once a minute.
- daily bars for about a year (the 200-day average needs 200 days), downloaded once per stock per day.

Requests ask for many stocks at once and stay under CALLS_PER_MINUTE (each page of a big answer counts), well inside
the 200 a minute the free Alpaca plan allows the whole app. Prices come from the same feed as the rest of the app: on
the free IEX feed volume is only part of all trading, but relative volume compares IEX with IEX, so it is still a fair
guide. A reading is kept for READING_AGE.

Two jobs use the readings:
- check(): the chart check on a news signal before the Trader acts on it (settings chart.confirm; rules in gate()).
- chart signals (settings chart.signals): right after a market-monitor check while the market is open (at most about
  once a minute), every watched stock's chart is read and chart_signal() run. A new one - at most one per stock and
  direction every COOLDOWN - is stored as a signal from "Chart patterns" (engine "chart"): watch-only by default, or
  manual review. They are never traded by themselves, and they are price-tracked like every signal so the scoreboard
  can measure them.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import re
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from ..context import AppContext
from ..db import iso, utcnow
from ..market import wording
from ..performance.speed import signal_speed
from ..state import trading_day
from ..trading import nyse_calendar
from . import indicators as ind
from .patterns import BEARISH, BULLISH
from .reading import ChartReading, ChartSignal, chart_signal, confirm, read_chart

log = logging.getLogger(__name__)

READING_AGE = timedelta(seconds=30)    # a chart reading is reused this long
CALLS_PER_MINUTE = 60                  # data requests a minute for charts (the free Alpaca plan: 200 for everything)
PAGE_BARS = 10_000                     # Alpaca sends at most this many bars per page of an answer
BARS_CHUNK = 50                        # stocks per bars request
DAILY_DAYS = 380                       # calendar days of daily bars (about a year of trading days)
MINUTES_FRESH = timedelta(seconds=60)  # 1-minute bars synced this recently are complete enough to read
TAIL_OVERLAP = timedelta(minutes=2)    # fetching the newest minutes: start a little before the last sync
DAILY_RETRY = timedelta(minutes=10)    # daily bars that failed to download are tried again after this
FORGET_AFTER = timedelta(hours=2)      # a stock's bars are dropped when nothing read them for this long
SCAN_EVERY = timedelta(seconds=50)     # chart signals: at most one look per this long
COOLDOWN = timedelta(minutes=30)       # at most one chart signal per stock and direction in this window
CHECK_TIMEOUT = 8.0                    # seconds a news signal waits for its chart at most
MAX_BOOST = 10                         # the most confidence a chart that agrees ever adds (reading.confirm)
SOURCE_NAME = "Chart patterns"
AGREES, NEUTRAL, AGAINST, STRETCHED, UNAVAILABLE = "agrees", "neutral", "against", "stretched", "unavailable"
NO_CHART = "No up-to-date chart for this stock (no trades in the last 30 minutes, or no price data) - not checked."
_SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")
# the trigger patterns behind chart signals, in plain English (stored as the signal's "news event")
TRIGGER_WORDS = {
    "breakout": "Breakout", "breakdown": "Breakdown", "opening_range_breakout": "Opening range breakout",
    "opening_range_breakdown": "Opening range breakdown", "double_top": "Double top", "double_bottom": "Double bottom",
    "bull_flag": "Bull flag", "bear_flag": "Bear flag", "rsi_divergence": "RSI divergence",
    "vwap_reclaim": "Back above VWAP", "vwap_lost": "Fell below VWAP",
}


class NoChartData(RuntimeError):
    """There is nothing to draw (not connected, or no prices for that stock)."""


# ---------------------------------------------------------------------------------------------- the news check
@dataclass
class ChartCheck:
    """What the chart check did to a news signal."""

    verdict: str             # agrees | neutral | against | stretched | unavailable
    reason: str
    score: float | None      # the chart's score: -1 (very bearish) .. +1 (very bullish)
    adjust: int              # confidence points the chart added (+) or took away (-)
    confidence: int          # the signal's confidence after the check
    review: str = ""         # send it to manual review with this reason instead of auto-trading ("" = no)

    def columns(self) -> dict:
        score = None if self.score is None or not math.isfinite(self.score) else round(float(self.score), 2)
        return {"chart_verdict": self.verdict, "chart_reason": self.reason[:600], "chart_score": score,
                "chart_adjust": int(self.adjust)}


def gate(direction: str, confidence: int, verdict: str, adjust: int, reason: str, score: float | None, mode: str,
         buy_threshold: int, closing: bool = False) -> ChartCheck:
    """The chart check's rules (pure). verdict / adjust / reason come from reading.confirm().

    - "stretched" (the move already happened): the confidence stays, but a signal that would be auto-traded goes to
      manual review instead - like the "don't chase" check.
    - "against": soft mode lowers the confidence by the chart's adjustment (5-15 points); strict mode sends a signal
      that would be auto-traded to manual review instead.
    - "agrees": a small boost (2-10 points), but a boost never lifts a signal from manual review into an auto-trade -
      one that started under the buy threshold ends at most one point under it.
    - closing: selling a stock you hold is never held back or lowered by the chart.
    - "neutral", "unavailable" or mode "off": nothing changes."""
    out = ChartCheck(verdict, reason, score, 0, int(confidence))
    if mode == "off":
        return out
    if verdict == AGREES and adjust > 0:
        boosted = min(100, confidence + adjust)
        if confidence < buy_threshold:
            boosted = min(boosted, max(confidence, buy_threshold - 1))
        out.adjust, out.confidence = boosted - confidence, boosted
    elif verdict in (AGAINST, STRETCHED) and closing:
        out.reason = f"{reason} (Selling a stock you hold is never held back by the chart.)"
    elif verdict == STRETCHED:
        out.review = f"{reason} Sent for manual review instead of auto-trading it."
    elif verdict == AGAINST and mode == "strict":
        out.review = f"{reason} Sent for manual review (Settings -> Charts -> strict)."
    elif verdict == AGAINST and adjust < 0:
        lowered = max(0, confidence + adjust)
        out.adjust, out.confidence = lowered - confidence, lowered
    return out


def worth_checking(direction: str, confidence: int, holding_long: bool, trading) -> bool:
    """Could the chart change what happens to this news signal? Not for a neutral one, one too weak to reach manual
    review even with a boost, or a bearish one on a stock you don't hold while shorting is off (ignored anyway)."""
    if direction not in (BULLISH, BEARISH) or int(confidence) + MAX_BOOST < trading.review_threshold:
        return False
    return direction == BULLISH or holding_long or trading.allow_shorting


# ---------------------------------------------------------------------------------------------- helpers
class Pacer:
    """At most `per_minute` data requests in any rolling minute. An answer that came in several pages is charged for
    the extra pages afterwards."""

    def __init__(self, per_minute: int = CALLS_PER_MINUTE, clock=time.monotonic, sleep=asyncio.sleep):
        self.per_minute = max(1, int(per_minute))
        self._clock = clock
        self._sleep = sleep
        self._times: deque[float] = deque()
        self._lock = asyncio.Lock()

    def _prune(self, now: float) -> None:
        while self._times and now - self._times[0] >= 60:
            self._times.popleft()

    async def wait(self) -> None:
        async with self._lock:
            while True:
                now = self._clock()
                self._prune(now)
                if len(self._times) < self.per_minute:
                    self._times.append(now)
                    return
                await self._sleep(max(0.05, 60 - (now - self._times[0])))

    def charge(self, pages: int) -> None:
        now = self._clock()
        self._times.extend([now] * max(0, int(pages)))

    def used(self) -> int:
        self._prune(self._clock())
        return len(self._times)


@dataclass
class _Minutes:
    bars: list[dict]       # 1-minute bars, oldest first (only finished ones)
    synced: datetime       # every bar that started before this is in
    used: datetime         # last time something read them


def _t(bar: dict) -> float:
    s = ind.seconds(bar.get("t"))
    return -1.0 if s is None else s


def _minute(dt: datetime) -> datetime:
    return dt.replace(second=0, microsecond=0)


def _merge(m: _Minutes, new: list[dict], since: datetime, until: datetime) -> None:
    """Replace the bars from `since` on with `new` - every bar between `since` and `until` - keeping only the ones
    that had finished by `until`. Builds a new list (a reading in another thread may hold the old one)."""
    cut, end = since.timestamp(), _minute(until).timestamp()
    fresh = sorted((b for b in new or [] if isinstance(b, dict) and cut <= _t(b) < end), key=_t)
    m.bars = [b for b in m.bars if _t(b) < cut] + fresh
    m.synced = max(m.synced, _minute(until))


def history_start(now: datetime) -> datetime:
    """4:00 New York time on the trading day before the latest one: two sessions of 1-minute bars."""
    d = now.astimezone(nyse_calendar.ET).date()
    days: list = []
    while len(days) < 2:
        if nyse_calendar.is_trading_day(d):
            days.append(d)
        d -= timedelta(days=1)
    return datetime.combine(days[1], nyse_calendar.PRE_OPEN, nyse_calendar.ET)


def _clean(symbol: str) -> str:
    return str(symbol or "").strip().upper().lstrip("$")


def _num(x: float) -> float | None:
    return float(f"{x:.6g}") if x is not None and math.isfinite(x) else None


def price_series(bars: list[dict], now: datetime) -> dict:
    """The latest session's 1-minute closes with VWAP, for the chart card ({"t": [...], "close": [...], "vwap": [...]};
    empty lists without data)."""
    b = ind.to_arrays(bars, now=now)
    if not len(b):
        return {"t": [], "close": [], "vwap": []}
    day = ind.day_parts(b.t)[0]
    keep = np.flatnonzero(day == day[-1])
    vw = ind.vwap(b)
    return {"t": [b.iso(int(i)) for i in keep], "close": [_num(float(b.close[i])) for i in keep],
            "vwap": [_num(float(vw[i])) for i in keep]}


def _quiet(task: asyncio.Task) -> None:
    """Collect a background task's error (it is logged where it matters) so asyncio doesn't complain."""
    if not task.cancelled():
        task.exception()


def _read_all(items: list[tuple[str, list[dict], list[dict]]], now: datetime) -> list[ChartReading]:
    return [read_chart(sym, minutes, daily, now) for sym, minutes, daily in items]


# ---------------------------------------------------------------------------------------------- the service
class ChartService:
    name = "chart"

    def __init__(self, ctx: AppContext, pacer: Pacer | None = None):
        self.ctx = ctx
        self.pacer = pacer or Pacer()
        self._minutes: dict[str, _Minutes] = {}
        self._daily: dict[str, tuple[str, list[dict]]] = {}   # symbol -> (New York date downloaded, bars)
        self._daily_failed: dict[str, datetime] = {}
        self._readings: dict[str, tuple[datetime, ChartReading]] = {}
        self._errors: dict[str, str] = {}                     # symbol -> why its bars couldn't be downloaded
        self._fetch_lock = asyncio.Lock()
        self._scan_lock = asyncio.Lock()
        self._wake = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._inflight: dict[str, asyncio.Task] = {}          # symbol -> a reading started by prefetch()
        self._scanned_at: datetime | None = None
        self._warned: set[str] = set()
        self.requests = 0                                     # data requests made (each page counted once)
        self.last_scan: str | None = None

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        self._task = asyncio.create_task(self._loop(), name="chart-signals")
        if self.ctx.config.settings.chart.signals == "off":
            self._set_status("off", "Chart signals are off (Settings -> Charts)")
        else:
            self._set_status("ok", "Chart signals start with the market monitor's next check")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task

    async def on_keys_changed(self) -> None:
        self.forget()

    def forget(self) -> None:
        """Drop every cached bar and reading (another account or data feed)."""
        self._minutes.clear()
        self._daily.clear()
        self._daily_failed.clear()
        self._readings.clear()
        self._errors.clear()

    def market_checked(self, now: datetime | None = None) -> None:
        """Called by the market monitor after each check while the market is open: time to look for chart signals."""
        self._wake.set()

    async def _loop(self) -> None:
        while True:
            await self._wake.wait()
            self._wake.clear()
            try:
                await self.scan()
            except Exception as exc:
                self._set_status("warn", f"Chart signals: the last check failed ({exc})")
                log.warning("Chart signal check failed: %s", exc)
                log.debug("chart scan traceback", exc_info=True)

    # ------------------------------------------------------------------ helpers
    def _broker(self):
        return getattr(self.ctx.service("trader"), "broker", None)

    def _feed(self) -> str:
        broker = self._broker()
        return str(getattr(broker, "data_feed", None) or self.ctx.config.settings.trading.data_feed).upper()

    def _set_status(self, level: str, detail: str) -> None:
        self.ctx.state.set_status("chart", level, detail)

    def _warn_once(self, key: str, msg: str, *args) -> None:
        if key in self._warned:
            log.debug(msg, *args)
        else:
            self._warned.add(key)
            log.warning(msg, *args)

    def _company(self, symbol: str) -> str:
        table = getattr(self.ctx.service("pipeline"), "tickers", None)
        info = table.get(symbol) if table is not None else None
        name = wording.company_name(info.name) if info is not None else ""
        name = name or wording.market_name(symbol)
        return "" if name == symbol else name

    # ------------------------------------------------------------------ bars
    async def _fetch(self, symbols: list[str], start: datetime, end: datetime, timeframe: str) -> dict[str, list[dict]]:
        broker = self._broker()
        if broker is None:
            raise NoChartData("Not connected to a market-data account (Settings -> API keys).")
        await self.pacer.wait()
        self.requests += 1
        rows = await asyncio.to_thread(broker.bars_multi, symbols, start, end, timeframe)
        out = {str(k).upper(): list(v or []) for k, v in (rows or {}).items()}
        extra = sum(len(v) for v in out.values()) // PAGE_BARS
        self.pacer.charge(extra)
        self.requests += extra
        return out

    async def _ensure(self, symbols: list[str], now: datetime) -> None:
        """Bring the bars of `symbols` up to date, a chunk at a time (so a news signal's check never waits long behind
        a big batch). A chunk that fails is noted per stock and skipped."""
        for i in range(0, len(symbols), BARS_CHUNK):
            chunk = symbols[i:i + BARS_CHUNK]
            async with self._fetch_lock:
                minutes, _daily = await asyncio.gather(self._ensure_minutes(chunk, now),
                                                       self._ensure_daily(chunk, now), return_exceptions=True)
            if isinstance(minutes, BaseException):
                for sym in chunk:
                    self._errors[sym] = f"Couldn't get prices: {minutes}"
                self._warn_once("minutes", "Charts couldn't get 1-minute bars: %s", minutes)
            else:
                for sym in chunk:
                    self._errors.pop(sym, None)
                self._warned.discard("minutes")

    async def _ensure_minutes(self, symbols: list[str], now: datetime) -> None:
        start = history_start(now)
        monitor = self.ctx.service("market")
        fresh_bars = getattr(monitor, "fresh_bars", None)
        full, tail = [], []
        for sym in symbols:
            m = self._minutes.get(sym)
            if m is None:
                full.append(sym)
                continue
            m.used = now
            if m.bars and _t(m.bars[0]) < start.timestamp():
                m.bars = [b for b in m.bars if _t(b) >= start.timestamp()]
            if now - m.synced < MINUTES_FRESH:
                continue
            got = fresh_bars(sym, now) if fresh_bars is not None else None
            if got is not None and got[1] <= m.synced < got[2]:
                _merge(m, got[0], got[1], got[2])  # the monitor's bars of a minute ago: no request needed
                if now - m.synced < MINUTES_FRESH:
                    continue
            tail.append(sym)
        if full:
            rows = await self._fetch(full, start, now, "1Min")
            for sym in full:
                m = _Minutes([], start, now)
                _merge(m, rows.get(sym, []), start, now)
                self._minutes[sym] = m
        if tail:
            series = {sym: self._minutes[sym] for sym in tail}
            since = min(m.synced for m in series.values()) - TAIL_OVERLAP
            rows = await self._fetch(tail, since, now, "1Min")
            for sym, m in series.items():
                _merge(m, rows.get(sym, []), since, now)

    async def _ensure_daily(self, symbols: list[str], now: datetime) -> None:
        day = trading_day(now)
        need = [sym for sym in symbols if self._daily.get(sym, ("",))[0] != day
                and not (sym in self._daily_failed and now - self._daily_failed[sym] < DAILY_RETRY)]
        if not need:
            return
        try:
            rows = await self._fetch(need, now - timedelta(days=DAILY_DAYS), now, "1Day")
            self._warned.discard("daily")
        except Exception as exc:  # the chart still works from 1-minute bars (with an estimated daily range)
            for sym in need:
                self._daily_failed[sym] = now
            self._warn_once("daily", "Charts couldn't get daily bars: %s", exc)
            return
        for sym in need:
            self._daily[sym] = (day, rows.get(sym, []))
            self._daily_failed.pop(sym, None)

    def _forget_old(self, now: datetime) -> None:
        for sym in [s for s, m in self._minutes.items() if now - m.used > FORGET_AFTER]:
            self._minutes.pop(sym, None)
            self._daily.pop(sym, None)
        day = trading_day(now)
        for sym in [s for s, (d, _bars) in self._daily.items() if d != day and s not in self._minutes]:
            self._daily.pop(sym, None)
        for sym in [s for s, (at, _r) in self._readings.items() if now - at > 10 * READING_AGE]:
            self._readings.pop(sym, None)

    # ------------------------------------------------------------------ readings
    async def reading(self, symbol: str, now: datetime | None = None) -> ChartReading:
        """The chart reading for one stock (reused for READING_AGE)."""
        return (await self.readings([symbol], now))[0]

    async def readings(self, symbols: list[str], now: datetime | None = None) -> list[ChartReading]:
        """Chart readings for several stocks, with their bars fetched together."""
        now = now or utcnow()
        wanted = list(dict.fromkeys(_clean(s) for s in symbols if _clean(s)))
        out: dict[str, ChartReading] = {}
        todo = []
        for sym in wanted:
            hit = self._readings.get(sym)
            if hit is not None and timedelta(0) <= now - hit[0] < READING_AGE:
                out[sym] = hit[1]
            else:
                todo.append(sym)
        if todo:
            await self._ensure(todo, now)
            items = [(sym, self._minutes[sym].bars if sym in self._minutes else [],
                      self._daily.get(sym, ("", []))[1]) for sym in todo]
            for r in await asyncio.to_thread(_read_all, items, now):
                out[r.symbol] = r
                self._readings[r.symbol] = (now, r)
        self._forget_old(now)
        return [out[sym] for sym in wanted]

    async def card(self, symbol: str, now: datetime | None = None) -> dict:
        """Everything the Market tab's chart card shows for one stock: the latest session's price and VWAP, the reading
        (indicators, patterns with their explanations, support / resistance, summary) and any chart signal."""
        sym = _clean(symbol)
        if not _SYMBOL_RE.match(sym):
            raise ValueError(f"'{symbol}' doesn't look like a ticker")
        if self._broker() is None:
            raise NoChartData("Not connected to a market-data account (Settings -> API keys).")
        now = now or utcnow()
        r = await self.reading(sym, now)
        if r.price is None:
            raise NoChartData(self._errors.get(sym) or f"No price data for {sym} on the {self._feed()} feed yet.")
        m = self._minutes.get(sym)
        sig = chart_signal(r)
        s = self.ctx.config.settings.chart
        return {"symbol": sym, "name": self._company(sym), "reading": r.as_dict(),
                "signal": sig.as_dict() if sig is not None else None,
                "series": price_series(m.bars if m is not None else [], now), "feed": self._feed(),
                "confirm": s.confirm, "signals": s.signals}

    # ------------------------------------------------------------------ the check on news signals
    def prefetch(self, symbol: str) -> asyncio.Task | None:
        """Start reading a stock's chart in the background. The trader calls this the moment a signal arrives, so
        downloading the chart overlaps with its other checks instead of adding to the time it takes to trade."""
        if self.ctx.config.settings.chart.confirm == "off" or not _clean(symbol):
            return None
        sym = _clean(symbol)
        task = self._inflight.get(sym)
        if task is None or task.done():
            task = asyncio.ensure_future(self.reading(sym))
            task.add_done_callback(_quiet)
            self._inflight = {k: t for k, t in self._inflight.items() if not t.done()}
            self._inflight[sym] = task
        return task

    async def check(self, signal: dict, holding_long: bool = False) -> ChartCheck | None:
        """Check a news signal against its chart before it is traded (None when the check is off or couldn't change
        anything - see worth_checking()). The result is saved on the signal. Never waits more than CHECK_TIMEOUT."""
        settings = self.ctx.config.settings
        mode = settings.chart.confirm
        direction = signal.get("direction")
        if mode == "off" or not signal.get("ticker") or not worth_checking(
                direction, signal["confidence"], holding_long, settings.trading):
            return None
        reading = None
        task = self.prefetch(signal["ticker"])  # (already running if the trader started it)
        try:  # a reading that takes too long finishes in the background and fills the cache anyway
            reading = await asyncio.wait_for(asyncio.shield(task), CHECK_TIMEOUT)
        except Exception as exc:
            log.info("Chart check for %s skipped: %s", signal["ticker"], str(exc) or "the chart took too long")
        if reading is None or reading.price is None or reading.stale:
            verdict, adjust, reason, score = UNAVAILABLE, 0, NO_CHART, None
        else:
            verdict, adjust, reason = confirm(direction, reading)
            score = reading.score
        out = gate(direction, int(signal["confidence"]), verdict, adjust, reason, score, mode,
                   settings.trading.buy_threshold, closing=direction == BEARISH and holding_long)
        if signal.get("id"):
            self.ctx.db.update("signals", signal["id"], out.columns())
        return out

    # ------------------------------------------------------------------ chart signals
    async def scan(self, now: datetime | None = None, force: bool = False) -> dict:
        """Read every watched stock's chart and store any new chart signal (public so tests can call it)."""
        now = now or utcnow()
        mode = self.ctx.config.settings.chart.signals
        if mode == "off":
            self._set_status("off", "Chart signals are off (Settings -> Charts)")
            return {"status": "off"}
        if not force and self._scanned_at is not None and timedelta(0) <= now - self._scanned_at < SCAN_EVERY:
            return {"status": "too_soon"}
        clock = getattr(self.ctx.service("trader"), "clock", None)
        if not (clock and clock.get("is_open")):
            self._set_status("ok", "Market closed - chart signals start again at the open")
            return {"status": "closed"}
        monitor = self.ctx.service("market")
        symbols = monitor.watched_stocks() if hasattr(monitor, "watched_stocks") else []
        if not symbols:
            self._set_status("ok", "No stocks to read yet - chart signals use the market monitor's list")
            return {"status": "nothing"}
        if self._scan_lock.locked():
            return {"status": "busy"}
        async with self._scan_lock:
            self._scanned_at = now
            readings = await self.readings(symbols, now)
            stored = []
            for r in readings:
                sig = chart_signal(r)
                if sig is not None and not await asyncio.to_thread(self._cooling, sig, now):
                    stored.append(await self._store(sig, r, mode, now))
        self.last_scan = iso(now)
        read = sum(1 for r in readings if r.price is not None and not r.stale)
        what = "watching only" if mode == "watch" else "new ones go to manual review"
        found = f"{len(stored)} new signal{'' if len(stored) == 1 else 's'}" if stored else "nothing new"
        self._set_status("ok", f"Chart signals ({what}) · read {read} of {len(readings)} charts · {found} · "
                               f"{self._feed()} · {self.pacer.used()} data requests in the last minute")
        return {"status": "ok", "read": read, "signals": [s["id"] for s in stored]}

    def _cooling(self, sig: ChartSignal, now: datetime) -> bool:
        """A chart signal for this stock and direction within COOLDOWN (also after a restart)."""
        return bool(self.ctx.db.scalar(
            "SELECT COUNT(*) FROM signals WHERE engine = 'chart' AND ticker = ? AND direction = ? AND created_at >= ?",
            (sig.symbol, sig.direction, iso(now - COOLDOWN))))

    async def _store(self, sig: ChartSignal, r: ChartReading, mode: str, now: datetime) -> dict:
        """Save a chart signal. Never sent to the trader: watch-only, or manual review for you to approve."""
        review = mode == "review"
        trigger = sig.patterns[0] if sig.patterns else ""
        event = TRIGGER_WORDS.get(trigger) or trigger.replace("_", " ").capitalize()
        if review:
            why = ("From the chart alone - it is never traded by itself. Approve it to trade (every risk check still "
                   "applies), or dismiss it.")
        else:
            why = ("Chart signals are only being watched - never traded or alerted. Performance -> By AI engine shows "
                   "whether they work (Settings -> Charts can send them to manual review instead).")
        row = {
            "analysis_id": None, "created_at": iso(now), "ticker": sig.symbol, "company": self._company(sig.symbol),
            "direction": sig.direction, "confidence": int(sig.confidence), "speaker": SOURCE_NAME,
            "source_id": "chart", "source_name": SOURCE_NAME, "source_type": "chart", "reasoning": sig.reason,
            "time_sensitivity": "immediate", "headline": sig.reason[:500], "url": None,
            "sources_seen": json.dumps([SOURCE_NAME]), "engine": "chart", "event": f"Chart: {event}",
            "flags": None, "decided_at": iso(now), "action": "review" if review else "watch", "action_reason": why,
            "traded": 0, "review_status": "pending" if review else None, "price_at_signal": r.price,
            "chart_verdict": None, "chart_reason": r.summary[:600],
            "chart_score": None if not math.isfinite(r.score) else round(float(r.score), 2), "chart_adjust": None,
        }
        row["id"] = await asyncio.to_thread(self.ctx.db.insert, "signals", row)
        event_row = self.ctx.db.query_one("SELECT * FROM signals WHERE id = ?", (row["id"],)) or row
        event_row["speed"] = signal_speed(event_row, None)
        self.ctx.bus.publish("signal", event_row)
        log.info("Chart signal #%s %s %s conf %s -> %s: %s", row["id"], sig.symbol, sig.direction, sig.confidence,
                 "manual review" if review else "watching", sig.reason)
        if review:
            await self._review_alert(event_row)
        return event_row

    async def _review_alert(self, sig: dict) -> None:
        title = f"Manual review (chart): {sig['ticker']} {sig['direction']} ({sig['confidence']})"
        msg = f"{sig['reasoning']}\nFrom the chart alone. Approve or dismiss in the Signals tab."
        alerts = self.ctx.service("alerts")
        if alerts is None:
            self.ctx.bus.publish("toast", {"kind": "warn", "title": title, "message": msg})
            return
        try:
            await alerts.send("manual_review", title, msg, "warn")
        except Exception:
            log.debug("chart review alert failed", exc_info=True)
