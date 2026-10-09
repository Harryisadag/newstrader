"""Chart indicators - pure numpy functions, no network or database, fully unit-tested.

Every function takes price arrays (or lists) and returns numpy arrays as long as its input. Values that can't be
worked out yet are NaN (an RSI needs 15 prices before its first value). Short, flat, zero-volume or NaN input never
raises - it just gives NaN (or a neutral value) where there is nothing to say. Each value only uses the bars up to
its own, so nothing here can peek at the future.

The standard settings match TradingView: EMAs start from a simple average, RSI and ATR use Wilder's smoothing and
Bollinger Bands use the population standard deviation.

Bars arrive as dicts ({"t": iso, "o", "h", "l", "c", "v"}, like market/detect.py) and to_arrays() turns them into a
Bars object of arrays. Stock bars are read in New York time: pre-market until 9:30, the regular session 9:30-16:00
(13:00 on early-close days), after-hours after that. Crypto trades around the clock and its "day" is the UTC day.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, date, datetime
from functools import lru_cache
from zoneinfo import ZoneInfo

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ..trading.nyse_calendar import early_closes

NY = ZoneInfo("America/New_York")
DAY = 86400
OPEN = 9 * 3600 + 30 * 60          # 9:30 New York time, in seconds after midnight
CLOSE = 16 * 3600                  # 16:00
EARLY_CLOSE = 13 * 3600            # 13:00 on early-close days
PRE, REGULAR, AFTER = 0, 1, 2      # the parts of a stock's trading day
_EPOCH = date(1970, 1, 1)


@dataclass(frozen=True, eq=False)
class Bars:
    """Bars as arrays, oldest first. t is when each bar starts (whole seconds since 1970, UTC)."""

    t: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray

    def __len__(self) -> int:
        return len(self.t)

    def __getitem__(self, key) -> Bars:
        return Bars(self.t[key], self.open[key], self.high[key], self.low[key], self.close[key], self.volume[key])

    def iso(self, i: int) -> str:
        return iso_time(int(self.t[i]))

    @property
    def spacing(self) -> int:
        """Usual seconds between bars (60 for 1-minute bars, 86400 for daily ones)."""
        if len(self.t) < 2:
            return 60
        return max(int(np.median(np.diff(self.t))), 1)

    @staticmethod
    def empty() -> Bars:
        z = np.zeros(0)
        return Bars(np.zeros(0, dtype=np.int64), z, z, z, z, z)

    @staticmethod
    def concat(a: Bars, b: Bars) -> Bars:
        return Bars(*(np.concatenate([x, y]) for x, y in zip(_cols(a), _cols(b), strict=True)))


def _cols(b: Bars) -> tuple[np.ndarray, ...]:
    return b.t, b.open, b.high, b.low, b.close, b.volume


def iso_time(ts: float) -> str:
    """Seconds since 1970 -> '2026-10-07T15:00:00.000Z' (the app's timestamp format)."""
    return datetime.fromtimestamp(ts, UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def seconds(value) -> float | None:
    """A time (ISO text, datetime, or seconds / milliseconds since 1970) as seconds since 1970, or None."""
    if isinstance(value, datetime):
        return (value if value.tzinfo else value.replace(tzinfo=UTC)).timestamp()
    if isinstance(value, int | float) and not isinstance(value, bool):
        if not math.isfinite(value):
            return None
        return value / 1000 if value > 1e11 else float(value)
    if isinstance(value, str) and value.strip():
        try:
            dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
        return (dt if dt.tzinfo else dt.replace(tzinfo=UTC)).timestamp()
    return None


def _num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def _floats(values: list) -> np.ndarray:
    try:
        out = np.array(values, dtype=float)
        if out.ndim == 1:
            return out
    except (TypeError, ValueError):
        pass
    return np.array([_num(x) for x in values], dtype=float)


def _times(values: list) -> np.ndarray:
    """Bar times as seconds since 1970 (NaN where unusable). The usual "...Z" text is parsed in one go."""
    if values and all(isinstance(x, str) and x.endswith("Z") for x in values):
        try:
            parsed = np.array([x[:-1] for x in values], dtype="datetime64[ms]")
            return np.where(np.isnat(parsed), np.nan, parsed.astype(np.int64) / 1000.0)
        except ValueError:
            pass
    return np.array([math.nan if (s := seconds(x)) is None else s for x in values], dtype=float)


def to_arrays(rows, now=None, bar_seconds: int = 60) -> Bars:
    """Bar dicts -> Bars, oldest first.

    Bars without a usable time or a positive close are skipped; a missing open/high/low falls back to the close
    and a missing volume counts as 0. With `now`, only bars already finished by then are kept (start +
    bar_seconds <= now), so a half-built bar never sneaks in. A repeated time keeps the last copy."""
    rows = [r for r in rows or [] if isinstance(r, dict)]
    if not rows:
        return Bars.empty()
    t = _times([r.get("t") for r in rows])
    c = _floats([r.get("c") for r in rows])
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(t) & (t > 0) & np.isfinite(c) & (c > 0)
        limit = seconds(now) if now is not None else None
        if limit is not None:
            ok &= t + bar_seconds <= limit + 1e-6
    if not ok.any():
        return Bars.empty()
    ta = np.floor(t[ok]).astype(np.int64)
    ca = c[ok]
    oa = _positive(_floats([r.get("o") for r in rows])[ok], ca)
    ha = np.fmax(_positive(_floats([r.get("h") for r in rows])[ok], ca), np.fmax(oa, ca))
    la = np.fmin(_positive(_floats([r.get("l") for r in rows])[ok], ca), np.fmin(oa, ca))
    va = _floats([r.get("v") for r in rows])[ok]
    with np.errstate(invalid="ignore"):
        va = np.where(np.isfinite(va) & (va > 0), va, 0.0)
    order = np.argsort(ta, kind="stable")
    ta, oa, ha, la, ca, va = (x[order] for x in (ta, oa, ha, la, ca, va))
    keep = np.r_[ta[1:] != ta[:-1], True]
    return Bars(ta[keep], oa[keep], ha[keep], la[keep], ca[keep], va[keep])


def _positive(x: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(x) & (x > 0), x, fallback)


# ---------------------------------------------------------------------------------------------- time of day
def ny_offset(t: np.ndarray) -> np.ndarray:
    """New York's offset from UTC in seconds for each time (-14400 in summer, -18000 in winter)."""
    t = np.asarray(t, dtype=np.int64)
    if not len(t):
        return np.zeros(0, dtype=np.int64)
    hours, inv = np.unique(t // 3600, return_inverse=True)  # clocks change on the hour
    return np.array([_hour_offset(int(hr)) for hr in hours], dtype=np.int64)[inv]


@lru_cache(maxsize=100_000)
def _hour_offset(hour: int) -> int:
    return int(datetime.fromtimestamp(hour * 3600, NY).utcoffset().total_seconds())


def day_parts(t: np.ndarray, crypto: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each bar time: its day (days since 1970 - New York date for stocks, UTC date for crypto), seconds after
    midnight that day, and which part of the trading day it is in (PRE / REGULAR / AFTER; crypto is all REGULAR)."""
    t = np.asarray(t, dtype=np.int64)
    local = t if crypto else t + ny_offset(t)
    day, sod = np.divmod(local, DAY)
    if crypto:
        return day, sod, np.full(len(t), REGULAR)
    close = np.full(len(t), CLOSE)
    for d in np.unique(day):
        when = _date(int(d))
        if when in early_closes(when.year):
            close[day == d] = EARLY_CLOSE
    seg = np.where(sod < OPEN, PRE, np.where(sod < close, REGULAR, AFTER))
    return day, sod, seg


def _date(day: int) -> date:
    return date.fromordinal(_EPOCH.toordinal() + day)


def daily_days(t: np.ndarray, crypto: bool = False) -> np.ndarray:
    """The date (days since 1970) each daily bar is for. Works whether the data provider stamps daily bars at
    midnight New York time or midnight UTC (both land on the right date after adding 12 hours)."""
    t = np.asarray(t, dtype=np.int64) + 12 * 3600
    return (t if crypto else t + ny_offset(t)) // DAY


def trading_day(ts: float, crypto: bool = False) -> int:
    """The day (days since 1970) a moment falls on: New York date for stocks, UTC date for crypto."""
    return int(day_parts(np.array([int(ts)]), crypto)[0][0])


def resample(bars: Bars, minutes: int, now=None, crypto: bool = False, bar_seconds: int = 60) -> Bars:
    """1-minute bars -> `minutes`-minute bars.

    Stock bars line up with the 9:30 New York open (9:30-9:35, 9:35-9:40...; pre-market ones count back from
    9:30), crypto bars with midnight UTC. A bar is only made once its time is over - by `now`, or (without `now`)
    when a later bar exists or its last minute is there - so the newest bar is never half-built."""
    size = int(minutes * 60)
    limit = seconds(now) if now is not None else None
    if limit is not None and len(bars):
        bars = bars[bars.t + bar_seconds <= limit + 1e-6]
    if not len(bars) or size <= 0:
        return Bars.empty()
    t = bars.t
    if crypto:
        start = t // size * size
    else:
        sod = (t + ny_offset(t)) % DAY
        start = t - sod + OPEN + np.floor_divide(sod - OPEN, size) * size
    first = np.flatnonzero(np.r_[True, start[1:] != start[:-1]])
    last = np.r_[first[1:] - 1, len(t) - 1]
    end = start[first] + size
    if limit is not None:
        done = end <= limit + 1e-6
    else:
        done = np.r_[np.ones(len(first) - 1, dtype=bool), t[last[-1]] + bar_seconds >= end[-1]]
    out = Bars(start[first], bars.open[first], np.fmax.reduceat(bars.high, first),
               np.fmin.reduceat(bars.low, first), bars.close[last], np.add.reduceat(bars.volume, first))
    return out[done]


# ---------------------------------------------------------------------------------------------- basics
def _arr(x) -> np.ndarray:
    a = np.asarray(x, dtype=float).ravel()
    return np.where(np.isfinite(a), a, np.nan)


def _windows(x: np.ndarray, n: int, reduce) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if n >= 1 and len(x) >= n:
        out[n - 1:] = reduce(sliding_window_view(x, n), axis=1)
    return out


def sma(x, n: int) -> np.ndarray:
    """Simple moving average of the last n values (NaN when any of them is missing)."""
    return _windows(_arr(x), int(n), np.mean)


def _smooth(x: np.ndarray, n: int, alpha: float) -> np.ndarray:
    """Exponential smoothing that starts from the simple average of the first n values. Missing values are
    skipped (NaN out, the average carries on)."""
    out = [math.nan] * len(x)
    if n < 1:
        return np.array(out, dtype=float)
    prev = None
    total, count = 0.0, 0
    for i, val in enumerate(x.tolist()):
        if val != val:
            continue
        if prev is None:
            total += val
            count += 1
            if count == n:
                prev = total / n
                out[i] = prev
        else:
            prev += alpha * (val - prev)
            out[i] = prev
    return np.array(out, dtype=float)


def ema(x, n: int) -> np.ndarray:
    """Exponential moving average (weight 2/(n+1)), starting from the simple average of the first n values."""
    n = int(n)
    return _smooth(_arr(x), n, 2 / (n + 1))


def rma(x, n: int) -> np.ndarray:
    """Wilder's smoothing (weight 1/n), starting from the simple average of the first n values."""
    n = int(n)
    return _smooth(_arr(x), n, 1 / n)


def rsi(close, n: int = 14) -> np.ndarray:
    """Relative Strength Index 0-100 (Wilder). Above 70 = overbought, below 30 = oversold. A price that hasn't
    moved at all reads 50."""
    c = _arr(close)
    out = np.full(len(c), np.nan)
    if len(c) <= n:
        return out
    d = np.diff(c)
    gain = rma(np.where(np.isnan(d), np.nan, np.maximum(d, 0.0)), n)
    loss = rma(np.where(np.isnan(d), np.nan, np.maximum(-d, 0.0)), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = 100 - 100 / (1 + gain / loss)
    r = np.where(loss == 0, np.where(gain == 0, 50.0, 100.0), r)
    out[1:] = np.where(np.isnan(gain) | np.isnan(loss), np.nan, r)
    return out


def macd(close, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MACD line (fast EMA - slow EMA), its signal line (EMA of the line) and the histogram (line - signal)."""
    c = _arr(close)
    line = ema(c, fast) - ema(c, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


def bollinger(close, n: int = 20, k: float = 2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bollinger Bands: (middle = n-bar SMA, upper = middle + k std, lower = middle - k std), population std."""
    c = _arr(close)
    mid = sma(c, n)
    sd = _windows(c, int(n), np.std)
    return mid, mid + k * sd, mid - k * sd


def bandwidth(upper, lower, mid) -> np.ndarray:
    """How wide the Bollinger Bands are, as a fraction of the middle line (small = a "squeeze")."""
    with np.errstate(divide="ignore", invalid="ignore"):
        bw = (_arr(upper) - _arr(lower)) / _arr(mid)
    return np.where(np.isfinite(bw), bw, np.nan)


def true_range(high, low, close) -> np.ndarray:
    h, lo, c = _arr(high), _arr(low), _arr(close)
    pc = np.r_[np.nan, c[:-1]]
    return np.fmax(h - lo, np.fmax(np.abs(h - pc), np.abs(lo - pc)))


def atr(high, low, close, n: int = 14) -> np.ndarray:
    """Average True Range (Wilder): how far the price usually travels in one bar, gaps included."""
    return rma(true_range(high, low, close), n)


def stochastic(high, low, close, k: int = 14, smooth_k: int = 3, d: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Slow stochastic (14, 3, 3): %K = where the close sits in the last k bars' range (0-100, smoothed),
    %D = average of %K. A flat range reads 50."""
    h, lo, c = _arr(high), _arr(low), _arr(close)
    hh, ll = highest(h, k), lowest(lo, k)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = np.where(hh > ll, 100 * (c - ll) / (hh - ll), 50.0)
    raw = np.where(np.isnan(hh) | np.isnan(ll) | np.isnan(c), np.nan, raw)
    pk = sma(raw, smooth_k)
    return pk, sma(pk, d)


def obv(close, volume) -> np.ndarray:
    """On-Balance Volume: running total of volume, added on up closes and taken away on down closes."""
    c, v = _arr(close), np.nan_to_num(_arr(volume))
    if not len(c):
        return np.zeros(0)
    step = np.nan_to_num(np.sign(np.diff(c))) * v[1:]
    return np.r_[0.0, np.cumsum(step)]


def roc(x, n: int) -> np.ndarray:
    """Rate of change: % change from n bars earlier."""
    a = _arr(x)
    out = np.full(len(a), np.nan)
    if 0 < n < len(a):
        with np.errstate(divide="ignore", invalid="ignore"):
            out[n:] = 100 * (a[n:] - a[:-n]) / a[:-n]
    return np.where(np.isfinite(out), out, np.nan)


def highest(x, n: int) -> np.ndarray:
    """Highest value over the last n bars (this one included), ignoring missing values."""
    return _windows(_arr(x), int(n), np.fmax.reduce)


def lowest(x, n: int) -> np.ndarray:
    """Lowest value over the last n bars (this one included), ignoring missing values."""
    return _windows(_arr(x), int(n), np.fmin.reduce)


def pct_diff(value, ref):
    """How far value is from ref, in % of ref (NaN when ref is 0 or missing). Works on numbers and arrays."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 100 * (np.asarray(value, dtype=float) - np.asarray(ref, dtype=float)) / np.asarray(ref, dtype=float)
    out = np.where(np.isfinite(out), out, np.nan)
    return float(out) if out.ndim == 0 else out


def atr_distance(value, ref, atr_value):
    """How far value is from ref, measured in ATRs (2.0 = two usual bar ranges away)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (np.asarray(value, dtype=float) - np.asarray(ref, dtype=float)) / np.asarray(atr_value, dtype=float)
    out = np.where(np.isfinite(out), out, np.nan)
    return float(out) if out.ndim == 0 else out


def pivots(high, low, left: int = 3, right: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Swing highs and lows: bar j is a swing high when its high is above the `left` bars before it and not below
    the `right` bars after it (a swing low the other way round). Careful: a swing at bar j is only known `right`
    bars later - callers must not use it before bar j + right."""
    h, lo = _arr(high), _arr(low)
    n = len(h)
    is_high, is_low = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    if n < left + right + 1:
        return is_high, is_low
    span = slice(left, n - right)
    before_h = sliding_window_view(h, left)[: n - left - right] if left else None
    after_h = sliding_window_view(h, right)[left + 1:] if right else None
    before_l = sliding_window_view(lo, left)[: n - left - right] if left else None
    after_l = sliding_window_view(lo, right)[left + 1:] if right else None
    mid_h, mid_l = h[span], lo[span]
    with np.errstate(invalid="ignore"):
        ok_h = np.isfinite(mid_h)
        ok_l = np.isfinite(mid_l)
        if left:
            ok_h &= np.all(mid_h[:, None] > before_h, axis=1)
            ok_l &= np.all(mid_l[:, None] < before_l, axis=1)
        if right:
            ok_h &= np.all(mid_h[:, None] >= after_h, axis=1)
            ok_l &= np.all(mid_l[:, None] <= after_l, axis=1)
    is_high[span], is_low[span] = ok_h, ok_l
    return is_high, is_low


# ---------------------------------------------------------------------------------------------- VWAP and volume
def vwap(bars: Bars, crypto: bool = False, rolling_hours: float | None = None) -> np.ndarray:
    """Volume-weighted average price: the average price paid, weighted by how much traded.

    Stocks: starts afresh each New York day - once for the pre-market and again at the 9:30 open; after-hours
    bars carry on the day's VWAP. Crypto: starts afresh at midnight UTC. rolling_hours (24 for crypto) instead
    gives a continuous VWAP over that many hours back. NaN until some volume has traded."""
    n = len(bars)
    if not n:
        return np.zeros(0)
    tp = (bars.high + bars.low + bars.close) / 3
    ok = np.isfinite(tp)
    vol = np.where(ok, np.nan_to_num(bars.volume), 0.0)
    cpv = np.r_[0.0, np.cumsum(np.where(ok, tp, 0.0) * vol)]
    cv = np.r_[0.0, np.cumsum(vol)]
    idx = np.arange(n)
    if rolling_hours:
        start = np.searchsorted(bars.t, bars.t - int(rolling_hours * 3600), side="right")
    else:
        day, sod, _ = day_parts(bars.t, crypto)
        key = day * 2 + (0 if crypto else (sod >= OPEN))
        new = np.r_[True, key[1:] != key[:-1]]
        start = np.maximum.accumulate(np.where(new, idx, 0))
    num = cpv[idx + 1] - cpv[start]
    den = cv[idx + 1] - cv[start]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    return np.where(den > 0, out, np.nan)


def _recent_volume(v: np.ndarray, key: np.ndarray, n: int) -> np.ndarray:
    """Average volume of the n bars before each bar with the same key (NaN with fewer than a few of them)."""
    m = len(v)
    idx = np.arange(m)
    new = np.r_[True, key[1:] != key[:-1]] if m else np.zeros(0, dtype=bool)
    run_start = np.maximum.accumulate(np.where(new, idx, 0)) if m else idx
    start = np.maximum(idx - n, run_start)
    count = idx - start
    cs = np.r_[0.0, np.cumsum(v)]
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = (cs[idx] - cs[start]) / count
    return np.where(count >= max(3, n // 4), mean, np.nan)


def _minute_grid(bars: Bars, crypto: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bars laid out as rows = sessions (only days with bars) x columns = minute of the day."""
    day, sod, seg = day_parts(bars.t, crypto)
    days, row = np.unique(day, return_inverse=True)
    minute = (sod // 60).astype(np.int64)
    vol = np.zeros((len(days), 1440))
    cnt = np.zeros((len(days), 1440))
    np.add.at(vol, (row, minute), np.nan_to_num(bars.volume))
    np.add.at(cnt, (row, minute), 1.0)
    return row, minute, vol, cnt, seg


def _segment_bounds(crypto: bool) -> tuple[np.ndarray, np.ndarray]:
    """First and last minute of the part of the day each minute belongs to (smoothing never crosses 9:30/16:00)."""
    minutes = np.arange(1440)
    if crypto:
        return np.zeros(1440, dtype=np.int64), np.full(1440, 1439)
    cuts = [0, OPEN // 60, CLOSE // 60, 1440]
    lo = np.zeros(1440, dtype=np.int64)
    hi = np.zeros(1440, dtype=np.int64)
    for a, b in zip(cuts[:-1], cuts[1:], strict=True):
        part = (minutes >= a) & (minutes < b)
        lo[part], hi[part] = a, b - 1
    return lo, hi


def relative_volume(bars: Bars, n: int = 20, days: int = 10, crypto: bool = False,
                    half_window: int | None = None) -> np.ndarray:
    """Each bar's volume compared with what's normal (2.0 = twice the usual volume, NaN = can't tell).

    Intraday bars are compared with the average volume at the same time of day over up to `days` earlier
    sessions (minute bars use the minutes around it too: half_window, default 2). Without earlier sessions to
    compare with - and for daily bars - it's the average of the n bars before it; for stocks only bars from the
    same part of the day count, so a quiet pre-market doesn't make the open look busy. Crypto's "day" is the UTC
    day."""
    m = len(bars)
    if not m:
        return np.zeros(0)
    v = np.nan_to_num(bars.volume)
    spacing = bars.spacing
    if spacing >= 6 * 3600 or crypto:
        base = _recent_volume(v, np.zeros(m, dtype=np.int64), n)
    else:
        day, _, seg = day_parts(bars.t)
        base = _recent_volume(v, day * 3 + seg, n)
    if spacing < 6 * 3600:
        half = (2 if spacing <= 60 else 0) if half_window is None else half_window
        tod = _time_of_day_volume(bars, days, crypto, half)
        base = np.where(np.isfinite(tod), tod, base)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = v / base
    return np.where(np.isfinite(out) & (base > 0), out, np.nan)


def _time_of_day_volume(bars: Bars, days: int, crypto: bool, half: int) -> np.ndarray:
    row, minute, vol, cnt, _ = _minute_grid(bars, crypto)
    if half > 0:
        lo, hi = _segment_bounds(crypto)
        lo = np.maximum(np.arange(1440) - half, lo)
        hi = np.minimum(np.arange(1440) + half, hi)
        cv = np.concatenate([np.zeros((len(vol), 1)), np.cumsum(vol, axis=1)], axis=1)
        cc = np.concatenate([np.zeros((len(cnt), 1)), np.cumsum(cnt, axis=1)], axis=1)
        vol = cv[:, hi + 1] - cv[:, lo]
        cnt = cc[:, hi + 1] - cc[:, lo]
    pv = np.concatenate([np.zeros((1, 1440)), np.cumsum(vol, axis=0)])
    pc = np.concatenate([np.zeros((1, 1440)), np.cumsum(cnt, axis=0)])
    first = np.maximum(row - days, 0)
    total = pv[row, minute] - pv[first, minute]
    count = pc[row, minute] - pc[first, minute]
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = total / count
    return np.where(count > 0, mean, np.nan)


def day_relative_volume(bars: Bars, days: int = 10, crypto: bool = False) -> np.ndarray:
    """Volume so far today compared with earlier sessions by the same time of day (2.0 = twice as busy as usual).

    Stocks count from the 9:30 open (NaN in the pre-market), crypto from midnight UTC. Only earlier sessions
    whose bars start near the open are used (a session the data only half covers would make today look busy).
    NaN without such a session."""
    m = len(bars)
    if not m:
        return np.zeros(0)
    day, sod, seg = day_parts(bars.t, crypto)
    start = 0 if crypto else OPEN
    use = (sod >= start) & (seg == REGULAR)
    if not use.any():
        return np.full(m, np.nan)
    days_list, row = np.unique(day, return_inverse=True)
    minute = (sod // 60).astype(np.int64)
    vol = np.zeros((len(days_list), 1440))
    np.add.at(vol, (row[use], minute[use]), np.nan_to_num(bars.volume[use]))
    cum = np.cumsum(vol, axis=1)
    first_min = np.full(len(days_list), 1440)
    np.minimum.at(first_min, row[use], minute[use])
    full = (first_min <= start // 60 + 15).astype(float)
    pf = np.r_[0.0, np.cumsum(full)]
    psum = np.concatenate([np.zeros((1, 1440)), np.cumsum(cum * full[:, None], axis=0)])
    lo = np.maximum(row - days, 0)
    count = pf[row] - pf[lo]
    base = (psum[row, minute] - psum[lo, minute])
    with np.errstate(divide="ignore", invalid="ignore"):
        out = cum[row, minute] / (base / count)
    return np.where(use & (count > 0) & (base > 0) & np.isfinite(out), out, np.nan)
