"""Spike and market-move detection - pure functions, no network or database, fully unit-tested.

Everything works on 1-minute bars ({"t": iso, "o", "h", "l", "c", "v"}) as the broker returns them. The monitor
(monitor.py) fetches the data, calls these functions and decides what to store and what to alert.

The rules (numbers in brackets are the defaults from Settings -> Market monitor):
- A stock SPIKES when its price moves at least spike_pct (3%) within spike_window_minutes (5) on at least
  volume_ratio (3x) its usual volume for that many minutes. When the usual volume can't be judged (too few
  earlier bars - common on the free IEX feed, which sees only part of all trading) the move must be 1.5x as big
  instead (4.5%).
- A VOLUME SURGE is twice the volume ratio (6x) with at least half the move (1.5%). Shown, never alerted.
- A move that is mostly the whole market moving doesn't count: the stock's own part of the move (its move minus
  SPY's over the same minutes) must be in the same direction and at least half the required move.
- The MARKET MOVES when an index ETF (SPY, QQQ...) moves market_move_pct (1%) within 15 minutes, and each time
  its change on the day crosses another market_day_step_pct (1%, 2%, 3%... up or down) - once per level per day.
- WORLD-MARKET ETFs (EWJ = Japan...) use the same day steps, but are only alert-worthy from twice the step (2%).
- Data that isn't current (the last bar is more than 3 minutes old) never makes a spike.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, time, timedelta

from ..db import iso, parse_iso
from ..state import MARKET_TZ

STALE_AFTER = timedelta(minutes=3)    # last bar older than this -> the price isn't current
REF_MAX_GAP = timedelta(minutes=15)   # the "before" price must be from at most this long before the window
BASELINE_MINUTES = 25                 # usual volume: the minutes just before the window...
MIN_BASELINE_BARS = 8                 # ...but only if at least this many of them had trades
MARKET_WINDOW_MINUTES = 15            # index ETFs: the move that counts as a market-wide move
UNKNOWN_VOLUME_FACTOR = 1.5           # without a volume baseline, the move must be this much bigger
SURGE_FACTOR = 2.0                    # volume surge: this many times the spike volume ratio
WORLD_ALERT_STEPS = 2                 # world ETFs alert from this many day steps (2 x 1% = 2%)


@dataclass
class WindowStats:
    """How a stock moved over the last few minutes (from its 1-minute bars)."""

    symbol: str
    window_min: int
    price: float | None = None            # close of the latest bar
    last_bar_at: datetime | None = None
    stale: bool = True                    # no bars, or the latest is more than STALE_AFTER old
    ref_price: float | None = None        # the price window_min minutes before the latest bar
    change_pct: float | None = None       # move over the window
    market_change_pct: float | None = None  # SPY's move over the same window
    rel_change_pct: float | None = None   # the stock's own part of the move (change - market change)
    volume: float = 0.0                   # shares traded in the window
    baseline_volume: float | None = None  # usual volume for that many minutes (None = can't tell)
    volume_ratio: float | None = None     # volume / baseline_volume
    bars: int = 0                         # bars available (IEX is sparse for quieter stocks)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["last_bar_at"] = iso(self.last_bar_at) if self.last_bar_at else None
        for key in ("change_pct", "market_change_pct", "rel_change_pct", "volume_ratio"):
            if d[key] is not None:
                d[key] = round(d[key], 2)
        return d


def _parsed(bars: list[dict], now: datetime) -> list[tuple[datetime, dict]]:
    rows = []
    for b in bars or []:
        t = parse_iso(b.get("t"))
        if t is not None and t <= now and b.get("c"):
            rows.append((t, b))
    rows.sort(key=lambda r: r[0])
    return rows


def pct_change(before: float | None, after: float | None) -> float | None:
    if not before or after is None:
        return None
    return (after - before) / before * 100


def window_stats(symbol: str, bars: list[dict], now: datetime, window_min: int,
                 market_change_pct: float | None = None) -> WindowStats:
    """Price and volume over the window_min minutes ending at the latest bar.

    The "before" price is the close of the last bar that started window_min minutes or more before the latest
    one (so a 5-minute window compares the latest close with the close 5 minutes earlier). The usual volume is
    the median volume of the bars in the BASELINE_MINUTES before the window, times window_min; with fewer than
    MIN_BASELINE_BARS such bars it is unknown (None)."""
    rows = _parsed(bars, now)
    st = WindowStats(symbol=symbol, window_min=window_min, bars=len(rows))
    if not rows:
        return st
    t_last, last = rows[-1]
    st.price = float(last["c"])
    st.last_bar_at = t_last
    st.stale = now - t_last > STALE_AFTER
    ref_t = t_last - timedelta(minutes=window_min)
    base_start = ref_t - timedelta(minutes=BASELINE_MINUTES)
    ref: tuple[datetime, dict] | None = None
    base: list[float] = []
    window_volume = 0.0
    for t, b in rows:
        v = float(b.get("v") or 0)
        if t <= ref_t:
            ref = (t, b)
            if t > base_start:
                base.append(v)
        else:
            window_volume += v
    st.volume = window_volume
    if ref is not None and ref_t - ref[0] <= REF_MAX_GAP:
        st.ref_price = float(ref[1]["c"])
        st.change_pct = pct_change(st.ref_price, st.price)
    if len(base) >= MIN_BASELINE_BARS:
        median = statistics.median(base)
        if median > 0:
            st.baseline_volume = median * window_min
            st.volume_ratio = window_volume / st.baseline_volume
    if market_change_pct is not None and st.change_pct is not None:
        st.market_change_pct = market_change_pct
        st.rel_change_pct = st.change_pct - market_change_pct
    return st


def _own_move(st: WindowStats, needed_pct: float) -> bool:
    """False when the move is mostly the whole market moving (compared with SPY over the same minutes)."""
    rel = st.rel_change_pct
    if rel is None or st.change_pct is None:
        return True
    return rel * st.change_pct > 0 and abs(rel) >= needed_pct / 2


def detect_spike(st: WindowStats, spike_pct: float, volume_ratio: float, min_price: float = 0.0) -> str | None:
    """'spike_up' | 'spike_down' | 'volume_surge' | None (see the module docstring for the rules)."""
    if st.stale or st.price is None or st.change_pct is None or st.price < min_price:
        return None
    move = abs(st.change_pct)
    ratio = st.volume_ratio
    if move >= spike_pct and _own_move(st, spike_pct):
        busy = ratio is not None and ratio >= volume_ratio
        big_enough_alone = ratio is None and move >= UNKNOWN_VOLUME_FACTOR * spike_pct
        if busy or big_enough_alone:
            return "spike_up" if st.change_pct > 0 else "spike_down"
    surge = ratio is not None and ratio >= SURGE_FACTOR * volume_ratio
    if surge and move >= spike_pct / 2 and _own_move(st, spike_pct / 2):
        return "volume_surge"
    return None


def market_window_move(st: WindowStats, market_move_pct: float) -> bool:
    """An index ETF moved at least market_move_pct within its window (MARKET_WINDOW_MINUTES)."""
    return not st.stale and st.change_pct is not None and abs(st.change_pct) >= market_move_pct


def world_alert_worthy(day_change_pct: float | None, step_pct: float) -> bool:
    """World-market ETFs only alert from WORLD_ALERT_STEPS day steps (2 x 1% = 2% by default)."""
    return day_change_pct is not None and abs(day_change_pct) >= WORLD_ALERT_STEPS * step_pct - 1e-9


def day_levels(change_pct: float, step_pct: float) -> list[float]:
    """The day-change levels already reached: -2.3% with 1% steps -> [-1.0, -2.0]."""
    if not step_pct or step_pct <= 0 or not change_pct:
        return []
    n = int(math.floor(abs(change_pct) / step_pct + 1e-9))
    sign = 1 if change_pct > 0 else -1
    return [round(sign * k * step_pct, 4) for k in range(1, n + 1)]


class DayLevels:
    """Remembers which day-change levels each symbol has crossed today, so each one is reported only once per
    symbol per day (a market that swings back and forth around -1% doesn't alert again and again)."""

    def __init__(self) -> None:
        self._seen: dict[str, tuple[str, set[float]]] = {}

    def _levels_for(self, symbol: str, day: str) -> set[float]:
        entry = self._seen.get(symbol)
        if entry is None or entry[0] != day:
            entry = (day, set())
            self._seen[symbol] = entry
        return entry[1]

    def new_level(self, symbol: str, day: str, change_pct: float | None, step_pct: float) -> float | None:
        """The biggest level newly crossed today (e.g. -2.0), or None. Smaller levels crossed at the same time
        are marked as seen too, so a jump straight to -2.3% makes one event, not two."""
        if change_pct is None:
            return None
        levels = day_levels(change_pct, step_pct)
        seen = self._levels_for(symbol, day)
        new = [lvl for lvl in levels if lvl not in seen]
        seen.update(levels)
        return max(new, key=abs) if new else None

    def mark(self, symbol: str, day: str, level: float, step_pct: float) -> None:
        """Remember a level reported earlier today (used after a restart)."""
        self._levels_for(symbol, day).update(day_levels(level, step_pct) or [round(level, 4)])


class EventMemory:
    """Stops one move from being stored again on every check while it is still inside its window.

    A (symbol, kind) is "new" if nothing was recorded for it in the last hold_minutes, or if the move has
    grown by another regrow_pct since (a second leg)."""

    def __init__(self) -> None:
        self._last: dict[tuple[str, str], tuple[datetime, float]] = {}

    def is_new(self, key: tuple[str, str], now: datetime, hold_minutes: float, change_pct: float,
               regrow_pct: float) -> bool:
        prev = self._last.get(key)
        fresh = (prev is None or now - prev[0] >= timedelta(minutes=hold_minutes)
                 or (prev[1] * change_pct > 0 and abs(change_pct) >= abs(prev[1]) + regrow_pct))
        if fresh:
            self.remember(key, now, change_pct)
        return fresh

    def remember(self, key: tuple[str, str], when: datetime, change_pct: float) -> None:
        self._last[key] = (when, change_pct)
        if len(self._last) > 2000:
            cutoff = when - timedelta(hours=2)
            self._last = {k: v for k, v in self._last.items() if v[0] >= cutoff}


def market_phase(now: datetime, clock: dict | None, session: dict | None, extended_hours: bool) -> str:
    """'open' (regular hours), 'extended' (pre-market 4:00-9:30 / after-hours until 8pm ET, only when
    extended_hours is on) or 'closed'.

    clock is Alpaca's market clock ({"is_open": ...}); session is today's trading session from the calendar
    ({"open": iso, "close": iso}) or None on a weekend / holiday."""
    if clock and clock.get("is_open"):
        return "open"
    open_t = parse_iso(session.get("open")) if session else None
    close_t = parse_iso(session.get("close")) if session else None
    if open_t is None or close_t is None:
        return "closed"
    if clock is None and open_t <= now < close_t:
        return "open"
    if extended_hours:
        day = now.astimezone(MARKET_TZ).date()
        pre_start = datetime.combine(day, time(4, 0), MARKET_TZ)
        post_end = min(close_t + timedelta(hours=4), datetime.combine(day, time(20, 0), MARKET_TZ))
        if pre_start <= now < open_t or close_t <= now < post_end:
            return "extended"
    return "closed"
