"""Chart patterns: candlesticks, indicator signals and chart structure, each as a plain-English Pattern.

A Frame is one chart (1-minute, 5-minute or daily bars) plus its indicators, each worked out once. Every detector
looks at one bar i and only uses bars up to i - never later ones - so what it finds at 10:05 is exactly what you
could have seen at 10:05. (A swing high needs a few lower bars after it before it counts, so it is only used once
those bars exist.)

direction is what the pattern usually means for the price: "bullish" (up), "bearish" (down) or "neutral".
strength runs 0-1. kind says what sort of evidence it is: "candle", "indicator", "structure", "trend", or
"caution" (overbought / stretched: a reason to wait, not a signal). Candlestick reversal patterns need the right
trend before them: a hammer only counts as bullish after a fall to a new low - after a rise it's a "hanging man".

The numbers below are judgement calls, kept here so they're easy to find and tune once the scoreboard shows how
the chart signals really do.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import cached_property

import numpy as np

from . import indicators as ind
from .indicators import OPEN, REGULAR, Bars

CONTEXT_BARS = 10            # the trend going into a candlestick pattern: the last 10 bars...
CONTEXT_ATR = 1.0            # ...must have moved at least 1 ATR (else it's "flat")
TREND_SLOPE_ATR = 0.1        # an up/downtrend's 20-EMA moves at least 0.1 ATR over 5 bars
RSI_HIGH, RSI_VERY_HIGH = 70, 80
RSI_LOW, RSI_VERY_LOW = 30, 20
HEAVY_VOLUME = 1.5           # "heavy volume" = 1.5x normal for the time of day
BREAKOUT_VOLUME = 1.5        # breakouts / breakdowns need a close past the level on at least this
CLIMAX_VOLUME = 3.0          # volume climax: 3x normal on a wide bar after a big run
STRETCH_ATR = 2.0            # "stretched": price this many daily ATRs (normal days' moves) away from VWAP
GAP_MIN_PCT = 2.0            # a gap smaller than this isn't worth a mention
OPENING_RANGE_MIN = 15       # the opening range: the first 15 minutes after 9:30
DOUBLE_TOL_PCT = 1.0         # double top / bottom: the two peaks within 1% (and within 1 ATR)
MACD_MIN_ATR = 0.05          # a MACD cross only counts if the histogram was at least this far from 0 just before
PIVOTS = {"live": (5, 5), "intraday": (3, 3), "daily": (3, 3)}       # swing points: bars before / after
LEVEL_LOOKBACK = {"live": 390, "intraday": 160, "daily": 250}        # support / resistance from this many bars
PATTERN_LOOKBACK = {"live": 120, "intraday": 60, "daily": 60}        # double tops, divergences...

BULLISH, BEARISH, NEUTRAL = "bullish", "bearish", "neutral"


@dataclass
class Pattern:
    """Something the chart is showing at one bar."""

    name: str                   # e.g. "breakout", "hammer", "rsi_overbought"
    direction: str              # "bullish" | "bearish" | "neutral"
    strength: float             # 0-1
    at: str                     # time of the bar it showed up on (bar start, ISO UTC)
    level: float | None = None  # the price that matters, if any (the level broken, the gap's close...)
    explain: str = ""           # plain English for the UI
    kind: str = "indicator"     # "candle" | "indicator" | "structure" | "trend" | "caution"
    timeframe: str = ""         # "1m" | "5m" | "1d"

    def as_dict(self) -> dict:
        d = asdict(self)
        d["strength"] = round(self.strength, 2)
        d["level"] = _round_price(self.level)
        return d


@dataclass
class Level:
    """A support or resistance price."""

    price: float
    kind: str          # "support" (below the price) | "resistance" (above it)
    touches: int       # how many swing highs/lows bunched up there
    source: str        # "swing points", "yesterday's high"...

    def as_dict(self) -> dict:
        return {"price": _round_price(self.price), "kind": self.kind, "touches": self.touches, "source": self.source}


def _round_price(x: float | None) -> float | None:
    if x is None or not math.isfinite(x):
        return None
    return float(f"{x:.6g}")


def money(x: float) -> str:
    """$101.20, $1,234.50 or $0.0001234 (tiny crypto prices)."""
    if not math.isfinite(x):
        return "?"
    if abs(x) >= 1000:
        return f"${x:,.2f}"
    return f"${x:.2f}" if abs(x) >= 1 else f"${x:.4g}"


def _ok(*vals) -> bool:
    return all(v is not None and math.isfinite(v) for v in vals)


class Frame:
    """One chart: bars of one timeframe and their indicators (each worked out once, when first needed).

    timeframe "1m" (the live chart), "1d" (daily) or anything else intraday ("5m"). daily = finished daily bars,
    for yesterday's close / high / low and the daily ATR on intraday charts. partial_last = the last bar is today's
    unfinished daily bar (candlestick patterns skip it)."""

    def __init__(self, bars: Bars, timeframe: str = "5m", crypto: bool = False, daily: Bars | None = None,
                 partial_last: bool = False):
        self.bars = bars
        self.timeframe = timeframe
        self.role = "live" if timeframe == "1m" else "daily" if timeframe == "1d" else "intraday"
        self.crypto = crypto
        self.daily = daily
        self.partial_last = partial_last
        self.n = len(bars)
        self.o, self.h, self.lo, self.c = bars.open, bars.high, bars.low, bars.close
        self.left, self.right = PIVOTS[self.role]

    @cached_property
    def atr(self) -> np.ndarray:
        return ind.atr(self.h, self.lo, self.c, 14)

    @cached_property
    def rsi(self) -> np.ndarray:
        return ind.rsi(self.c, 14)

    @cached_property
    def macd(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return ind.macd(self.c)

    @cached_property
    def bb(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return ind.bollinger(self.c)

    @cached_property
    def bw(self) -> np.ndarray:
        mid, up, low = self.bb
        return ind.bandwidth(up, low, mid)

    @cached_property
    def ema20(self) -> np.ndarray:
        return ind.ema(self.c, 20)

    @cached_property
    def ema50(self) -> np.ndarray:
        return ind.ema(self.c, 50)

    @cached_property
    def sma50(self) -> np.ndarray:
        return ind.sma(self.c, 50)

    @cached_property
    def sma200(self) -> np.ndarray:
        return ind.sma(self.c, 200)

    @cached_property
    def vwap(self) -> np.ndarray:
        return ind.vwap(self.bars, self.crypto, 24 if self.crypto else None)

    @cached_property
    def rvol(self) -> np.ndarray:
        return ind.relative_volume(self.bars, crypto=self.crypto)

    @cached_property
    def parts(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return ind.day_parts(self.bars.t, self.crypto)

    @cached_property
    def swings(self) -> tuple[np.ndarray, np.ndarray]:
        return ind.pivots(self.h, self.lo, self.left, self.right)

    @cached_property
    def session_start(self) -> np.ndarray:
        """For each bar: index of the first regular-session bar of its day (-1 if there's none up to it)."""
        day, _, seg = self.parts
        out = np.full(self.n, -1)
        for d in np.unique(day):
            idx = np.flatnonzero((day == d) & (seg == REGULAR))
            if len(idx):
                same = np.flatnonzero(day == d)
                out[same[same >= idx[0]]] = idx[0]
        return out

    @cached_property
    def prior(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """For each bar: the previous trading day's close, high and low, and the daily ATR as of that day (NaN
        when unknown). From the finished daily bars, else from this chart's own bars of that day; the ATR comes
        from the latest daily bar before the bar's day."""
        out = tuple(np.full(self.n, np.nan) for _ in range(4))
        if not self.n:
            return out
        day, _, seg = self.parts
        known: dict[int, tuple[float, float, float]] = {}
        for d in np.unique(day):
            idx = np.flatnonzero((day == d) & (seg == REGULAR))
            if len(idx):
                known[int(d)] = (self.c[idx[-1]], self.h[idx].max(), self.lo[idx].min())
        atr_by_day: dict[int, float] = {}
        if self.daily is not None and len(self.daily):
            daily_atr = ind.atr(self.daily.high, self.daily.low, self.daily.close, 14)
            for k, d in enumerate(ind.daily_days(self.daily.t, self.crypto).tolist()):
                known[int(d)] = (self.daily.close[k], self.daily.high[k], self.daily.low[k])
                atr_by_day[int(d)] = daily_atr[k]
        keys, atr_keys = sorted(known), sorted(atr_by_day)
        for d in np.unique(day):
            mask = day == d
            pos = bisect_left(keys, int(d)) - 1
            if pos >= 0:
                for arr, val in zip(out[:3], known[keys[pos]], strict=True):
                    arr[mask] = val
            pos = bisect_left(atr_keys, int(d)) - 1
            if pos >= 0:
                out[3][mask] = atr_by_day[atr_keys[pos]]
        return out

    @cached_property
    def ref_atr(self) -> np.ndarray:
        """A normal day's move for each bar: yesterday's daily ATR, else estimated from this chart's own ATR
        (scaled up by the square root of the number of bars in a day)."""
        if self.role == "daily":
            return self.atr
        per_day = (1440 if self.crypto else 390) * 60 / max(self.bars.spacing, 60)
        return np.where(np.isfinite(self.prior[3]), self.prior[3], self.atr * math.sqrt(per_day))

    def make(self, name: str, direction: str, strength: float, i: int, explain: str, kind: str,
             level: float | None = None) -> Pattern:
        lvl = float(level) if level is not None and math.isfinite(level) else None
        return Pattern(name, direction, round(min(max(float(strength), 0.0), 1.0), 3), self.bars.iso(i), lvl,
                       explain, kind, self.timeframe)

    def volume_note(self, i: int) -> tuple[float, str]:
        """(strength bonus, " on 2.4x normal volume") when bar i traded heavily, else (0, "")."""
        r = self.rvol[i] if 0 <= i < self.n else math.nan
        if _ok(r) and r >= HEAVY_VOLUME:
            return (0.15 if r >= 3 else 0.1), f" on {r:.1f}x normal volume"
        return 0.0, ""


# ---------------------------------------------------------------------------------------------- shared helpers
def trend_state(f: Frame, i: int) -> str:
    """'up', 'down', 'sideways' or 'unknown' at bar i: the price against its 20- and 50-bar EMAs, and whether the
    20-EMA is rising or falling (at least TREND_SLOPE_ATR over 5 bars)."""
    if i < 5 or i >= f.n:
        return "unknown"
    c, e20, e50, a, e20_before = f.c[i], f.ema20[i], f.ema50[i], f.atr[i], f.ema20[i - 5]
    if not _ok(c, e20, a, e20_before) or a <= 0:
        return "unknown"
    slope = (e20 - e20_before) / a
    have50 = _ok(e50)
    if c > e20 and (not have50 or e20 > e50) and slope > TREND_SLOPE_ATR:
        return "up"
    if c < e20 and (not have50 or e20 < e50) and slope < -TREND_SLOPE_ATR:
        return "down"
    return "sideways"


def _before(f: Frame, j: int) -> str:
    """The trend going into bar j: 'down', 'up', 'flat' or 'unknown' (move of the last CONTEXT_BARS closes, in ATRs)."""
    k = j - 1
    if k - CONTEXT_BARS < 0:
        return "unknown"
    a = f.atr[k]
    if not _ok(a, f.c[k], f.c[k - CONTEXT_BARS]) or a <= 0:
        return "unknown"
    move = (f.c[k] - f.c[k - CONTEXT_BARS]) / a
    return "down" if move <= -CONTEXT_ATR else "up" if move >= CONTEXT_ATR else "flat"


def _new_low(f: Frame, first: int, last: int) -> bool:
    """Bars first..last reached a low at or below the lowest of the CONTEXT_BARS bars before them."""
    if first - CONTEXT_BARS < 0:
        return False
    return bool(np.nanmin(f.lo[first:last + 1]) <= np.nanmin(f.lo[first - CONTEXT_BARS:first]))


def _new_high(f: Frame, first: int, last: int) -> bool:
    if first - CONTEXT_BARS < 0:
        return False
    return bool(np.nanmax(f.h[first:last + 1]) >= np.nanmax(f.h[first - CONTEXT_BARS:first]))


def _candle(f: Frame, j: int) -> tuple[float, float, float, float]:
    """(range, body, upper wick, lower wick) of bar j."""
    o, h, lo, c = f.o[j], f.h[j], f.lo[j], f.c[j]
    return h - lo, abs(c - o), h - max(o, c), min(o, c) - lo


def _cluster(prices: list[float], tol: float) -> list[tuple[float, int]]:
    """Group sorted prices that lie within tol of their group's average: [(average, count)]."""
    groups: list[list[float]] = []
    for p in prices:
        if groups and p - sum(groups[-1]) / len(groups[-1]) <= tol:
            groups[-1].append(p)
        else:
            groups.append([p])
    return [(sum(g) / len(g), len(g)) for g in groups]


def levels(f: Frame, i: int) -> list[Level]:
    """Support and resistance at bar i, lowest first: prices where swing highs/lows bunched up (at least 2
    touches), plus yesterday's high, low and close on intraday charts. Below the close = support, above =
    resistance. Only swings already confirmed by bar i count."""
    if i < 0 or i >= f.n or not _ok(f.c[i]):
        return []
    hi_sw, lo_sw = f.swings
    first, last = max(0, i - LEVEL_LOOKBACK[f.role]), i - f.right
    prices: list[float] = []
    if last >= first:
        prices += f.h[first:last + 1][hi_sw[first:last + 1]].tolist()
        prices += f.lo[first:last + 1][lo_sw[first:last + 1]].tolist()
    price, a = f.c[i], f.atr[i]
    tol = max(0.5 * a if _ok(a) else 0.0, price * 0.0005)
    found = [Level(p, "", n, "swing points") for p, n in _cluster(sorted(x for x in prices if _ok(x)), tol)
             if n >= 2]
    if f.role != "daily":
        pc, ph, pl, _ = (x[i] for x in f.prior)
        for value, name in ((ph, "yesterday's high"), (pl, "yesterday's low"), (pc, "yesterday's close")):
            if not _ok(value):
                continue
            near = [lv for lv in found if abs(lv.price - value) <= tol and lv.source == "swing points"]
            if near:
                near[0].price, near[0].source, near[0].touches = float(value), name, near[0].touches + 1
            else:
                found.append(Level(float(value), "", 1, name))
    for lv in found:
        lv.kind = "resistance" if lv.price > price else "support"
    return sorted(found, key=lambda lv: lv.price)


# ---------------------------------------------------------------------------------------------- candlesticks
def _single_candle(f: Frame, i: int) -> Pattern | None:
    """Hammer / hanging man, shooting star / inverted hammer, doji at a top or bottom."""
    if i < 1 or (f.partial_last and i == f.n - 1):
        return None
    rng, body, upper, lower = _candle(f, i)
    a = f.atr[i - 1]
    if not _ok(rng, a) or rng <= 0 or a <= 0:
        return None
    ctx = _before(f, i)
    at_low, at_high = _new_low(f, i, i), _new_high(f, i, i)
    bonus, vol = f.volume_note(i)
    if rng >= 0.8 * a and lower >= 2 * body and lower >= 0.55 * rng and upper <= 0.15 * rng:
        if ctx == "down" and at_low:
            return f.make("hammer", BULLISH, 0.45 + bonus, i,
                          f"Hammer{vol}: after a fall, sellers pushed the price down to {money(f.lo[i])} but "
                          "buyers pushed it right back up - a possible bottom", "candle", f.lo[i])
        if ctx == "up" and at_high:
            return f.make("hanging_man", BEARISH, 0.3 + bonus / 2, i,
                          "Hanging man: after a rise, a candle with a long lower tail - buyers may be tiring",
                          "candle", f.h[i])
    if rng >= 0.8 * a and upper >= 2 * body and upper >= 0.55 * rng and lower <= 0.15 * rng:
        if ctx == "up" and at_high:
            return f.make("shooting_star", BEARISH, 0.45 + bonus, i,
                          f"Shooting star{vol}: after a rise, buyers pushed the price up to {money(f.h[i])} but "
                          "it fell right back - a possible top", "candle", f.h[i])
        if ctx == "down" and at_low:
            return f.make("inverted_hammer", BULLISH, 0.3 + bonus / 2, i,
                          "Inverted hammer: after a fall, buyers tried to push the price up - a possible bottom "
                          "if the next candle follows through", "candle", f.lo[i])
    if body <= 0.1 * rng and rng >= 0.5 * a:
        if ctx == "up" and at_high:
            return f.make("doji", BEARISH, 0.3, i, "Doji at a high: after a rise the price ended where it "
                          "started - buyers and sellers are evenly matched, the rise may stall", "candle", f.h[i])
        if ctx == "down" and at_low:
            return f.make("doji", BULLISH, 0.3, i, "Doji at a low: after a fall the price ended where it "
                          "started - the selling may be running out", "candle", f.lo[i])
    return None


def _engulfing(f: Frame, i: int) -> Pattern | None:
    if i < 2 or (f.partial_last and i == f.n - 1):
        return None
    o1, c1, o2, c2, a = f.o[i - 1], f.c[i - 1], f.o[i], f.c[i], f.atr[i - 1]
    if not _ok(o1, c1, o2, c2, a) or a <= 0:
        return None
    b1, b2 = abs(c1 - o1), abs(c2 - o2)
    if b2 < 0.6 * a or b2 <= b1 or b1 < 0.15 * a:
        return None
    ctx = _before(f, i - 1)
    bonus, vol = f.volume_note(i)
    if c1 < o1 and c2 > o2 and o2 <= c1 and c2 >= o1 and ctx == "down" and _new_low(f, i - 1, i):
        return f.make("bullish_engulfing", BULLISH, 0.5 + bonus, i,
                      f"Bullish engulfing{vol}: after a fall, one strong up-candle swallowed the whole previous "
                      "down-candle - buyers took over", "candle", f.lo[i - 1:i + 1].min())
    if c1 > o1 and c2 < o2 and o2 >= c1 and c2 <= o1 and ctx == "up" and _new_high(f, i - 1, i):
        return f.make("bearish_engulfing", BEARISH, 0.5 + bonus, i,
                      f"Bearish engulfing{vol}: after a rise, one strong down-candle swallowed the whole previous "
                      "up-candle - sellers took over", "candle", f.h[i - 1:i + 1].max())
    return None


def _star(f: Frame, i: int) -> Pattern | None:
    """Morning star (down candle, small candle, strong up candle) and evening star."""
    if i < 3 or (f.partial_last and i == f.n - 1):
        return None
    a = f.atr[i - 1]
    o1, c1, o2, c2, o3, c3 = f.o[i - 2], f.c[i - 2], f.o[i - 1], f.c[i - 1], f.o[i], f.c[i]
    if not _ok(a, o1, c1, o2, c2, o3, c3) or a <= 0:
        return None
    b1, b2, b3 = abs(c1 - o1), abs(c2 - o2), abs(c3 - o3)
    if b1 < 0.6 * a or b2 > 0.35 * b1 or b3 < 0.4 * a:
        return None
    ctx = _before(f, i - 2)
    bonus, vol = f.volume_note(i)
    if (c1 < o1 and max(o2, c2) <= c1 + 0.25 * b1 and c3 > o3 and c3 >= (o1 + c1) / 2 and ctx == "down"
            and _new_low(f, i - 2, i)):
        return f.make("morning_star", BULLISH, 0.55 + bonus, i,
                      f"Morning star{vol}: a big down-candle, a small pause, then a strong up-candle - the fall "
                      "may be over", "candle", f.lo[i - 2:i + 1].min())
    if (c1 > o1 and min(o2, c2) >= c1 - 0.25 * b1 and c3 < o3 and c3 <= (o1 + c1) / 2 and ctx == "up"
            and _new_high(f, i - 2, i)):
        return f.make("evening_star", BEARISH, 0.55 + bonus, i,
                      f"Evening star{vol}: a big up-candle, a small pause, then a strong down-candle - the rise "
                      "may be over", "candle", f.h[i - 2:i + 1].max())
    return None


def _three(f: Frame, i: int) -> Pattern | None:
    """Three white soldiers / three black crows: three strong candles in a row, each closing near its end."""
    if i < 3 or (f.partial_last and i == f.n - 1):
        return None
    a = f.atr[i - 3]
    idx = (i - 2, i - 1, i)
    if not _ok(a, *(f.o[k] for k in idx), *(f.c[k] for k in idx)) or a <= 0:
        return None
    slack = 0.05 * a
    ctx = _before(f, i - 2)

    def solid(k: int, up: bool) -> bool:
        rng = f.h[k] - f.lo[k]
        body = (f.c[k] - f.o[k]) if up else (f.o[k] - f.c[k])
        tail = (f.h[k] - f.c[k]) if up else (f.c[k] - f.lo[k])
        return body >= 0.5 * a and tail <= 0.3 * rng

    if all(solid(k, True) for k in idx) and ctx in ("down", "flat") and all(
            f.c[k] > f.c[k - 1] and f.c[k - 1] + slack >= f.o[k] >= f.o[k - 1] - slack for k in idx[1:]):
        bonus, vol = f.volume_note(i)
        return f.make("three_white_soldiers", BULLISH, 0.5 + bonus, i,
                      f"Three white soldiers{vol}: three strong up-candles in a row, each closing near its high - "
                      "steady buying", "candle")
    if all(solid(k, False) for k in idx) and ctx in ("up", "flat") and all(
            f.c[k] < f.c[k - 1] and f.c[k - 1] - slack <= f.o[k] <= f.o[k - 1] + slack for k in idx[1:]):
        bonus, vol = f.volume_note(i)
        return f.make("three_black_crows", BEARISH, 0.5 + bonus, i,
                      f"Three black crows{vol}: three strong down-candles in a row, each closing near its low - "
                      "steady selling", "candle")
    return None


# ---------------------------------------------------------------------------------------------- indicators
def _rsi_extreme(f: Frame, i: int) -> Pattern | None:
    r = f.rsi[i]
    if not _ok(r):
        return None
    if r > RSI_HIGH:
        very = r > RSI_VERY_HIGH
        return f.make("rsi_overbought", BEARISH, 0.8 if very else 0.5, i,
                      f"RSI is {r:.0f} - {'very ' if very else ''}overbought: the price has risen fast and often "
                      "pauses or pulls back from here", "caution")
    if r < RSI_LOW:
        very = r < RSI_VERY_LOW
        return f.make("rsi_oversold", BULLISH, 0.8 if very else 0.5, i,
                      f"RSI is {r:.0f} - {'very ' if very else ''}oversold: the price has fallen fast and often "
                      "bounces from here", "caution")
    return None


def _rsi_divergence(f: Frame, i: int) -> Pattern | None:
    """Price makes a higher swing high but RSI a lower one (bearish), or a lower low with a higher RSI low.
    Shows up on the bar that confirms the second swing."""
    j = i - f.right
    if j < 1:
        return None
    hi_sw, lo_sw = f.swings
    first = max(0, j - PATTERN_LOOKBACK[f.role])
    rsi = f.rsi
    if hi_sw[j]:
        prev = np.flatnonzero(hi_sw[first:j - 4]) + first
        if len(prev):
            p = int(prev[-1])
            if _ok(rsi[p], rsi[j]) and f.h[j] > f.h[p] and rsi[j] < rsi[p] - 3 and rsi[p] >= 55:
                return f.make("rsi_divergence", BEARISH, 0.5, i,
                              f"Bearish divergence: the price made a higher high ({money(f.h[j])}) but RSI made a "
                              f"lower one ({rsi[j]:.0f} vs {rsi[p]:.0f}) - the push up is losing steam",
                              "indicator", f.h[j])
    if lo_sw[j]:
        prev = np.flatnonzero(lo_sw[first:j - 4]) + first
        if len(prev):
            p = int(prev[-1])
            if _ok(rsi[p], rsi[j]) and f.lo[j] < f.lo[p] and rsi[j] > rsi[p] + 3 and rsi[p] <= 45:
                return f.make("rsi_divergence", BULLISH, 0.5, i,
                              f"Bullish divergence: the price made a lower low ({money(f.lo[j])}) but RSI made a "
                              f"higher one ({rsi[j]:.0f} vs {rsi[p]:.0f}) - the selling is losing steam",
                              "indicator", f.lo[j])
    return None


def _macd(f: Frame, i: int) -> list[Pattern]:
    line, _, hist = f.macd
    a = f.atr[i]
    if i < 7 or not _ok(hist[i], hist[i - 1], line[i], line[i - 1], a) or a <= 0:
        return []
    out = []
    recent_h, recent_l = hist[i - 6:i], line[i - 6:i]
    if np.all(np.isfinite(recent_h)):
        if hist[i - 1] <= 0 < hist[i] and recent_h.min() <= -MACD_MIN_ATR * a:
            early = line[i] < 0
            out.append(f.make("macd_cross", BULLISH, 0.45 if early else 0.4, i,
                              "MACD crossed above its signal line - upward momentum is picking up"
                              + (" (early: still below zero)" if early else ""), "indicator"))
        elif hist[i - 1] >= 0 > hist[i] and recent_h.max() >= MACD_MIN_ATR * a:
            early = line[i] > 0
            out.append(f.make("macd_cross", BEARISH, 0.45 if early else 0.4, i,
                              "MACD crossed below its signal line - downward momentum is picking up"
                              + (" (early: still above zero)" if early else ""), "indicator"))
    if np.all(np.isfinite(recent_l)):
        if line[i - 1] <= 0 < line[i] and recent_l.min() <= -2 * MACD_MIN_ATR * a:
            out.append(f.make("macd_zero_cross", BULLISH, 0.45, i, "MACD crossed above zero - the short-term "
                              "average price is now above the longer-term one", "indicator"))
        elif line[i - 1] >= 0 > line[i] and recent_l.max() >= 2 * MACD_MIN_ATR * a:
            out.append(f.make("macd_zero_cross", BEARISH, 0.45, i, "MACD crossed below zero - the short-term "
                              "average price is now below the longer-term one", "indicator"))
    return out


def _bollinger(f: Frame, i: int) -> Pattern | None:
    """A squeeze (bands unusually tight) and the breakout that often follows it."""
    _, up, low = f.bb
    bw = f.bw
    if i < 40 or not _ok(f.c[i], f.c[i - 1], up[i], low[i], up[i - 1], low[i - 1], bw[i]):
        return None
    history = bw[max(0, i - 120):i]
    history = history[np.isfinite(history)]
    if len(history) < 20:
        return None
    recent = bw[i - 10:i]
    usual = np.median(history)
    tightest = np.nanmin(recent) if np.isfinite(recent).any() else math.inf
    squeezed = tightest <= np.percentile(history, 20) and tightest < 0.75 * usual
    bonus, vol = f.volume_note(i)
    if squeezed and f.c[i] > up[i] and f.c[i - 1] <= up[i - 1]:
        return f.make("bollinger_breakout", BULLISH, 0.55 + bonus, i,
                      f"Squeeze breakout{vol}: after a quiet, tight stretch the price closed above the upper "
                      "Bollinger Band", "indicator", up[i])
    if squeezed and f.c[i] < low[i] and f.c[i - 1] >= low[i - 1]:
        return f.make("bollinger_breakout", BEARISH, 0.55 + bonus, i,
                      f"Squeeze breakdown{vol}: after a quiet, tight stretch the price closed below the lower "
                      "Bollinger Band", "indicator", low[i])
    if bw[i] <= np.percentile(history, 10) and bw[i] < 0.75 * usual:
        return f.make("bollinger_squeeze", NEUTRAL, 0.25, i,
                      "Bollinger squeeze: the price is moving in the tightest range in a while - a big move often "
                      "follows (either way)", "indicator")
    return None


def _vwap_session_start(f: Frame, i: int, back: int) -> bool:
    """Bars i-back..i are all in the same VWAP session (crypto's rolling VWAP never restarts)."""
    if f.crypto:
        return True
    day, sod, _ = f.parts
    return day[i - back] == day[i] and (sod[i - back] >= OPEN) == (sod[i] >= OPEN)


def _vwap_cross(f: Frame, i: int) -> Pattern | None:
    """A close back above VWAP after being clearly below it for most of the last 5 bars (or the other way)."""
    vw = f.vwap
    if i < 6 or not _ok(f.c[i], f.c[i - 1], vw[i], vw[i - 1]) or not _vwap_session_start(f, i, 5):
        return None
    side = f.c[i - 5:i] - vw[i - 5:i]
    a = f.atr[i]
    if not np.all(np.isfinite(side)) or not _ok(a):
        return None
    bonus, vol = f.volume_note(i)
    if f.c[i - 1] < vw[i - 1] and f.c[i] > vw[i] and (side < 0).sum() >= 3 and side.min() <= -0.15 * a:
        return f.make("vwap_reclaim", BULLISH, 0.4 + bonus, i,
                      f"Climbed back above VWAP{vol} (the average price paid today, {money(vw[i])}) - buyers are "
                      "back in control", "indicator", vw[i])
    if f.c[i - 1] > vw[i - 1] and f.c[i] < vw[i] and (side > 0).sum() >= 3 and side.max() >= 0.15 * a:
        return f.make("vwap_lost", BEARISH, 0.4 + bonus, i,
                      f"Fell below VWAP{vol} (the average price paid today, {money(vw[i])}) - sellers are in "
                      "control", "indicator", vw[i])
    return None


def _vwap_stretch(f: Frame, i: int) -> Pattern | None:
    vw, ra, c = f.vwap[i], f.ref_atr[i], f.c[i]
    if not _ok(vw, ra, c) or ra <= 0:
        return None
    d = (c - vw) / ra
    if abs(d) < STRETCH_ATR:
        return None
    strength = min(0.9, 0.5 + 0.15 * (abs(d) - STRETCH_ATR))
    if d > 0:
        return f.make("stretched_above_vwap", BEARISH, strength, i,
                      f"Stretched: the price is {d:.1f}x a normal day's move above VWAP (the average price paid "
                      f"today, {money(vw)}) - it often snaps back toward it", "caution", vw)
    return f.make("stretched_below_vwap", BULLISH, strength, i,
                  f"Stretched: the price is {-d:.1f}x a normal day's move below VWAP (the average price paid "
                  f"today, {money(vw)}) - it often snaps back toward it", "caution", vw)


def _golden_cross(f: Frame, i: int) -> Pattern | None:
    """The 50-day average crossing the 200-day one - only after it was clearly (half an ATR) on the other side
    within the last 60 days, so two flat averages wobbling around each other don't count."""
    s50, s200, a = f.sma50, f.sma200, f.atr[i]
    if i < 1 or not _ok(s50[i], s200[i], s50[i - 1], s200[i - 1], a):
        return None
    gap = (s50 - s200)[max(0, i - 60):i]
    if s50[i - 1] <= s200[i - 1] and s50[i] > s200[i] and np.nanmin(gap) <= -0.5 * a:
        return f.make("golden_cross", BULLISH, 0.55, i, "Golden cross: the 50-day average price crossed above "
                      "the 200-day average - the long-term trend may be turning up", "indicator", s200[i])
    if s50[i - 1] >= s200[i - 1] and s50[i] < s200[i] and np.nanmax(gap) >= 0.5 * a:
        return f.make("death_cross", BEARISH, 0.55, i, "Death cross: the 50-day average price crossed below "
                      "the 200-day average - the long-term trend may be turning down", "indicator", s200[i])
    return None


def _ma_trend(f: Frame, i: int) -> Pattern | None:
    state = trend_state(f, i)
    if state not in ("up", "down"):
        return None
    slope = abs(f.ema20[i] - f.ema20[i - 5]) / f.atr[i]
    strength = 0.4 + min(0.3, 0.1 * slope)
    span = "20- and 50-day" if f.role == "daily" else "20- and 50-bar"
    if state == "up":
        return f.make("uptrend", BULLISH, strength, i, f"Uptrend: the price is above its {span} averages and "
                      "they're rising", "trend")
    return f.make("downtrend", BEARISH, strength, i, f"Downtrend: the price is below its {span} averages and "
                  "they're falling", "trend")


# ---------------------------------------------------------------------------------------------- chart structure
def _breakout(f: Frame, i: int) -> Pattern | None:
    """A close through support / resistance (as it stood one bar earlier) on heavy volume."""
    if i < 1 or not _ok(f.c[i], f.c[i - 1]):
        return None
    r = f.rvol[i]
    if not _ok(r) or r < BREAKOUT_VOLUME:
        return None
    candidates = [lv for lv in levels(f, i - 1) if lv.source != "yesterday's close"]
    up = [lv for lv in candidates if f.c[i - 1] <= lv.price < f.c[i]]
    down = [lv for lv in candidates if f.c[i] < lv.price <= f.c[i - 1]]
    bonus = 0.15 if r >= 3 else 0.1
    span = " in the past year" if f.role == "daily" else ""
    if up:
        lv = max(up, key=lambda x: (x.touches, x.price))
        where = (f"{lv.source} ({money(lv.price)})" if lv.source != "swing points" else
                 f"resistance at {money(lv.price)} (the price turned back there {lv.touches} times{span})")
        return f.make("breakout", BULLISH, 0.5 + 0.07 * min(lv.touches - 1, 3) + bonus, i,
                      f"Broke above {where} on {r:.1f}x normal volume", "structure", lv.price)
    if down:
        lv = min(down, key=lambda x: (-x.touches, x.price))
        where = (f"{lv.source} ({money(lv.price)})" if lv.source != "swing points" else
                 f"support at {money(lv.price)} (the price bounced there {lv.touches} times{span})")
        return f.make("breakdown", BEARISH, 0.5 + 0.07 * min(lv.touches - 1, 3) + bonus, i,
                      f"Broke below {where} on {r:.1f}x normal volume", "structure", lv.price)
    return None


def _opening_range(f: Frame, i: int) -> Pattern | None:
    """The first close above (below) the high (low) of the first 15 minutes after the 9:30 open."""
    ss = int(f.session_start[i])
    day, sod, seg = f.parts
    end = OPEN + OPENING_RANGE_MIN * 60
    if ss < 0 or seg[i] != REGULAR or sod[i] < end or sod[ss] > OPEN + 300 or not _ok(f.c[i]):
        return None
    k = ss
    while k <= i and day[k] == day[i] and sod[k] < end:
        k += 1
    orh, orl = np.nanmax(f.h[ss:k]), np.nanmin(f.lo[ss:k])
    after = f.c[k:i]
    bonus, vol = f.volume_note(i)
    morning = 0.1 if sod[i] < 11 * 3600 else 0.0
    if f.c[i] > orh and (not len(after) or np.nanmax(after) <= orh):
        return f.make("opening_range_breakout", BULLISH, 0.45 + morning + bonus, i,
                      f"Broke above the opening range{vol} (the high of the first 15 minutes, {money(orh)})",
                      "structure", orh)
    if f.c[i] < orl and (not len(after) or np.nanmin(after) >= orl):
        return f.make("opening_range_breakdown", BEARISH, 0.45 + morning + bonus, i,
                      f"Broke below the opening range{vol} (the low of the first 15 minutes, {money(orl)})",
                      "structure", orl)
    return None


def _gap(f: Frame, i: int) -> Pattern | None:
    """A gap at the open vs yesterday's close (while it holds), and the moment it gets filled."""
    ss = int(f.session_start[i])
    pc = f.prior[0][i]
    if ss < 0 or f.parts[2][i] != REGULAR or not _ok(pc, f.o[ss]) or pc <= 0:
        return None
    gap = (f.o[ss] - pc) / pc * 100
    if abs(gap) < GAP_MIN_PCT:
        return None
    if gap > 0:
        if i > ss and np.nanmin(f.lo[ss:i]) <= pc:
            return None
        if f.lo[i] <= pc:
            return f.make("gap_fill", BEARISH, 0.5, i, f"Gap filled: it opened {gap:.1f}% higher but has now fallen "
                          f"all the way back to yesterday's close ({money(pc)})", "structure", pc)
        return f.make("gap_up", BULLISH, 0.35 + min(0.25, gap / 20), i,
                      f"Gapped up {gap:.1f}% at the open (yesterday's close was {money(pc)}) and is holding it",
                      "structure", pc)
    if i > ss and np.nanmax(f.h[ss:i]) >= pc:
        return None
    if f.h[i] >= pc:
        return f.make("gap_fill", BULLISH, 0.5, i, f"Gap filled: it opened {-gap:.1f}% lower but has now climbed "
                      f"all the way back to yesterday's close ({money(pc)})", "structure", pc)
    return f.make("gap_down", BEARISH, 0.35 + min(0.25, -gap / 20), i,
                  f"Gapped down {-gap:.1f}% at the open (yesterday's close was {money(pc)}) and is staying down",
                  "structure", pc)


def _day_extreme(f: Frame, i: int) -> Pattern | None:
    """A new high / low of the day (regular session, from 9:45 on - before that every bar is one)."""
    ss = int(f.session_start[i])
    _, sod, seg = f.parts
    if ss < 0 or i <= ss or seg[i] != REGULAR or sod[i] < OPEN + OPENING_RANGE_MIN * 60:
        return None
    bonus, vol = f.volume_note(i)
    if f.h[i] > np.nanmax(f.h[ss:i]):
        return f.make("new_high_of_day", BULLISH, 0.35 + bonus, i, f"New high of the day ({money(f.h[i])}){vol}",
                      "structure", f.h[i])
    if f.lo[i] < np.nanmin(f.lo[ss:i]):
        return f.make("new_low_of_day", BEARISH, 0.35 + bonus, i, f"New low of the day ({money(f.lo[i])}){vol}",
                      "structure", f.lo[i])
    return None


def _double(f: Frame, i: int) -> Pattern | None:
    """Double top / bottom: two swing highs (lows) within ~1%, then a close through the dip (bump) between them."""
    a = f.atr[i] if i < f.n else math.nan
    if i < 2 or not _ok(a, f.c[i], f.c[i - 1]) or a <= 0:
        return None
    hi_sw, lo_sw = f.swings
    first, last = max(0, i - PATTERN_LOOKBACK[f.role]), i - f.right
    if last <= first:
        return None
    tops = np.flatnonzero(hi_sw[first:last + 1]) + first
    if len(tops) >= 2:
        p1, p2 = int(tops[-2]), int(tops[-1])
        t1, t2 = f.h[p1], f.h[p2]
        top = max(t1, t2)
        neck = np.nanmin(f.lo[p1:p2 + 1])
        if (p2 - p1 >= 4 and abs(t1 - t2) <= min(DOUBLE_TOL_PCT / 100 * top, a)
                and np.nanmax(f.h[p1 + 1:p2], initial=-np.inf) <= min(t1, t2) and top - neck >= 1.5 * a
                and np.nanmax(f.h[p2 + 1:i + 1], initial=-np.inf) <= top and f.c[i] < neck <= f.c[i - 1]):
            bonus, vol = f.volume_note(i)
            return f.make("double_top", BEARISH, 0.6 + bonus, i,
                          f"Double top{vol}: the price failed twice near {money(top)} and has now closed below "
                          f"the dip between them ({money(neck)})", "structure", neck)
    bottoms = np.flatnonzero(lo_sw[first:last + 1]) + first
    if len(bottoms) >= 2:
        p1, p2 = int(bottoms[-2]), int(bottoms[-1])
        b1, b2 = f.lo[p1], f.lo[p2]
        bottom = min(b1, b2)
        neck = np.nanmax(f.h[p1:p2 + 1])
        if (p2 - p1 >= 4 and abs(b1 - b2) <= min(DOUBLE_TOL_PCT / 100 * bottom, a)
                and np.nanmin(f.lo[p1 + 1:p2], initial=np.inf) >= max(b1, b2) and neck - bottom >= 1.5 * a
                and np.nanmin(f.lo[p2 + 1:i + 1], initial=np.inf) >= bottom and f.c[i] > neck >= f.c[i - 1]):
            bonus, vol = f.volume_note(i)
            return f.make("double_bottom", BULLISH, 0.6 + bonus, i,
                          f"Double bottom{vol}: the price held twice near {money(bottom)} and has now closed above "
                          f"the bump between them ({money(neck)})", "structure", neck)
    return None


def _flag(f: Frame, i: int) -> Pattern | None:
    """Bull flag: a sharp rise (the pole: at least 3 ATR within 10 bars), a calm drift down or sideways for 3-12
    bars that gives back at most half of it, then a close above the flag. Bear flag: the same upside down."""
    if i < 16:
        return None
    a = f.atr[i - 1]
    if not _ok(a, f.c[i], f.c[i - 1]) or a <= 0:
        return None
    lo_i = i - 16
    h_win, l_win = f.h[lo_i:i], f.lo[lo_i:i]
    if not (np.all(np.isfinite(h_win)) and np.all(np.isfinite(l_win))):
        return None
    for bull in (True, False):
        pt = lo_i + int(np.argmax(h_win) if bull else np.argmin(l_win))
        k = i - 1 - pt
        if not 3 <= k <= 12:
            continue
        ps = max(0, pt - 10)
        if bull:
            pb = ps + int(np.argmin(f.lo[ps:pt + 1]))
            pole = f.h[pt] - f.lo[pb]
        else:
            pb = ps + int(np.argmax(f.h[ps:pt + 1]))
            pole = f.h[pb] - f.lo[pt]
        pole_atr = f.atr[pb]  # measured before the pole's own big bars
        if pb >= pt or not _ok(pole, pole_atr) or pole < 3 * pole_atr:
            continue
        fh, fl = np.nanmax(f.h[pt + 1:i]), np.nanmin(f.lo[pt + 1:i])
        drift = (f.c[i - 1] - f.c[pt + 1]) / k
        bonus, vol = f.volume_note(i)
        if bull and f.h[pt] - fl <= 0.5 * pole and drift <= 0.1 * a and f.c[i] > fh >= f.c[i - 1]:
            return f.make("bull_flag", BULLISH, 0.6 + bonus, i,
                          f"Bull flag{vol}: a sharp rise, a short calm pause, and now a break above the pause "
                          f"({money(fh)}) - the rise may be starting again", "structure", fh)
        if not bull and fh - f.lo[pt] <= 0.5 * pole and drift >= -0.1 * a and f.c[i] < fl <= f.c[i - 1]:
            return f.make("bear_flag", BEARISH, 0.6 + bonus, i,
                          f"Bear flag{vol}: a sharp fall, a short calm pause, and now a break below the pause "
                          f"({money(fl)}) - the fall may be starting again", "structure", fl)
    return None


def _climax(f: Frame, i: int) -> Pattern | None:
    """Volume climax: a wide bar on huge volume after a big run - moves often end like this. Called exhaustion
    (a reversal sign) when the bar also closed well away from its extreme."""
    if i < CONTEXT_BARS + 1:
        return None
    r, a = f.rvol[i], f.atr[i - 1]
    rng, _, upper, lower = _candle(f, i)
    if not _ok(r, a, rng) or a <= 0 or r < CLIMAX_VOLUME or rng < 1.5 * a:
        return None
    move = (f.c[i - 1] - f.c[i - 1 - CONTEXT_BARS]) / a
    if move >= 3:
        if upper >= 0.4 * rng or f.c[i] <= (f.h[i] + f.lo[i]) / 2:
            return f.make("volume_climax", BEARISH, 0.55, i,
                          f"Buying exhaustion: {r:.1f}x normal volume after a big run-up, and the bar closed well "
                          "off its high - buyers may be running out", "structure", f.h[i])
        return f.make("volume_climax", NEUTRAL, 0.3, i,
                      f"Volume climax: {r:.1f}x normal volume after a big run-up - moves often end like this, "
                      "watch for a reversal", "structure", f.h[i])
    if move <= -3:
        if lower >= 0.4 * rng or f.c[i] >= (f.h[i] + f.lo[i]) / 2:
            return f.make("volume_climax", BULLISH, 0.55, i,
                          f"Selling exhaustion: {r:.1f}x normal volume after a big drop, and the bar closed well "
                          "off its low - sellers may be running out", "structure", f.lo[i])
        return f.make("volume_climax", NEUTRAL, 0.3, i,
                      f"Volume climax: {r:.1f}x normal volume after a big drop - moves often end like this, "
                      "watch for a reversal", "structure", f.lo[i])
    return None


# ---------------------------------------------------------------------------------------------- running them
CHARTS = frozenset({"intraday", "daily"})
Detector = Callable[[Frame, int], "Pattern | list[Pattern] | None"]
# (detector, chart roles it runs on, stocks only)
DETECTORS: tuple[tuple[Detector, frozenset[str], bool], ...] = (
    (_single_candle, CHARTS, False),
    (_engulfing, CHARTS, False),
    (_star, CHARTS, False),
    (_three, CHARTS, False),
    (_rsi_extreme, CHARTS, False),
    (_rsi_divergence, CHARTS, False),
    (_macd, CHARTS, False),
    (_bollinger, CHARTS, False),
    (_ma_trend, CHARTS, False),
    (_breakout, CHARTS, False),
    (_double, CHARTS, False),
    (_flag, CHARTS, False),
    (_climax, CHARTS, False),
    (_golden_cross, frozenset({"daily"}), False),
    (_vwap_cross, frozenset({"intraday"}), False),
    (_opening_range, frozenset({"intraday"}), True),
    (_vwap_stretch, frozenset({"live"}), False),
    (_gap, frozenset({"live"}), True),
    (_day_extreme, frozenset({"live"}), True),
)


def find_patterns(f: Frame, i: int | None = None) -> list[Pattern]:
    """Every pattern showing at bar i (default: the latest bar), using only bars up to i."""
    if i is None:
        i = f.n - 1
    if i < 0 or i >= f.n:
        return []
    out: list[Pattern] = []
    with np.errstate(all="ignore"):
        for detector, roles, stock_only in DETECTORS:
            if f.role not in roles or (stock_only and f.crypto):
                continue
            found = detector(f, i)
            if isinstance(found, Pattern):
                out.append(found)
            elif found:
                out.extend(found)
    return out


def recent_patterns(f: Frame, bars: int = 3) -> list[Pattern]:
    """Patterns from the last few bars, newest first. One that keeps showing (an uptrend, overbought...) is
    listed once, at its latest bar."""
    seen: set[tuple[str, str]] = set()
    out: list[Pattern] = []
    for i in range(f.n - 1, max(f.n - bars, 0) - 1, -1):
        for p in find_patterns(f, i):
            key = (p.name, p.direction)
            if key not in seen:
                seen.add(key)
                out.append(p)
    return out
