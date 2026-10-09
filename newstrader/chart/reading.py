"""Reading a chart: an indicator snapshot, the patterns showing right now, an overall score and a plain-English
summary - plus the two helpers the app uses with it:

- confirm(direction, reading): does the chart agree with a news signal? "agrees" / "neutral" / "against" /
  "stretched" (the move already happened - buying now would be chasing it), with a confidence nudge.
- chart_signal(reading): a signal from the chart alone. Conservative on purpose (a trigger pattern on heavy volume
  plus at least one more agreeing piece of evidence, never a lone candlestick, confidence at most 75) - and the
  app keeps these watch-only until the scoreboard shows they work.

Charts used: the 1-minute bars (live price, VWAP, gaps, new highs of the day), 5-minute bars built from them (most
patterns - 1-minute candles are mostly noise) and daily bars (the bigger trend). Only bars finished by `now` are
used, so reading the same data "as of" an earlier time gives exactly what you'd have seen then.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import indicators as ind
from .indicators import REGULAR, Bars
from .patterns import (
    BEARISH,
    BULLISH,
    HEAVY_VOLUME,
    NEUTRAL,
    STRETCH_ATR,
    Frame,
    Level,
    Pattern,
    levels,
    recent_patterns,
    trend_state,
)

STALE_MINUTES = 30           # no 1-minute bar for this long -> the reading is "stale"
STRETCH_RSI = 80             # RSI above this (below 100 - this for bearish) = stretched
VERY_STRETCHED_RSI = 90
VERY_STRETCHED_ATR = 3.0
AGREE_SCORE = 0.3            # confirm(): score (in the signal's direction) at or above this = "agrees"
AGAINST_SCORE = -0.35        # ...at or below this = "against" (a mild trend + the side of VWAP alone is about 0.4)
SIGNAL_MIN_EVIDENCE = 3      # chart_signal(): a trigger pattern + heavy volume + at least 1 more agreeing piece
SIGNAL_MIN_SCORE = 0.2       # ...and the overall score must lean the same way
SIGNAL_MAX_CONFIDENCE = 75
RECENT_BARS = {"1m": 5, "5m": 3, "1d": 1}   # "recent" patterns: from this many latest bars of each chart

# patterns that can start a chart signal - only fresh ones from the intraday charts (daily ones are context)
TRIGGERS = frozenset({
    "breakout", "breakdown", "opening_range_breakout", "opening_range_breakdown", "double_top", "double_bottom",
    "bull_flag", "bear_flag", "bollinger_breakout", "rsi_divergence", "vwap_reclaim", "vwap_lost",
})
_KIND_WEIGHT = {"structure": 1.0, "indicator": 0.6, "candle": 0.35, "caution": 0.8}
_TREND_WORDS = {"up": "Uptrend", "down": "Downtrend", "sideways": "Sideways"}
_EVENT_WORDS = {
    "breakout": "breaking out", "breakdown": "breaking down",
    "opening_range_breakout": "breaking out of the opening range",
    "opening_range_breakdown": "breaking below the opening range",
    "bull_flag": "breaking out of a bull flag", "bear_flag": "breaking down from a bear flag",
    "double_top": "double top", "double_bottom": "double bottom",
    "vwap_reclaim": "just climbed back above VWAP", "vwap_lost": "just fell below VWAP",
    "gap_up": "gapped up", "gap_down": "gapped down", "gap_fill": "gap filled",
    "new_high_of_day": "at a new high of the day", "new_low_of_day": "at a new low of the day",
    "golden_cross": "golden cross", "death_cross": "death cross", "rsi_divergence": "RSI divergence",
    "macd_cross": "MACD cross", "macd_zero_cross": "MACD crossed zero", "volume_climax": "volume climax",
}


@dataclass
class ChartReading:
    """What the chart says about one symbol at one moment. as_dict() is ready for the UI (plain JSON)."""

    symbol: str
    asset_class: str = "stock"
    at: str | None = None                  # when the chart was read (ISO UTC)
    price: float | None = None             # latest 1-minute close
    bar_at: str | None = None              # start of that bar
    stale: bool = True                     # no 1-minute bar within STALE_MINUTES
    prev_close: float | None = None        # yesterday's close
    day_change_pct: float | None = None
    gap_pct: float | None = None           # today's open vs yesterday's close (stocks)
    vwap: float | None = None
    vwap_pct: float | None = None          # % above (+) / below (-) VWAP
    vwap_atr: float | None = None          # distance from VWAP in normal days' moves (daily ATRs)
    atr: float | None = None               # a normal day's move in $ (daily ATR 14)
    atr_pct: float | None = None           # ...as % of the price
    atr_basis: str = ""                    # "daily" or "estimated" (from intraday bars, no daily bars)
    rsi: float | None = None               # RSI 14 on the 5-minute chart
    rsi_daily: float | None = None         # RSI 14 on the daily chart (today so far counts as a day)
    macd_hist: float | None = None         # 5-minute MACD histogram (> 0 = momentum up)
    rel_volume: float | None = None        # volume vs normal (see rel_volume_basis)
    rel_volume_basis: str = ""
    bar_rel_volume: float | None = None    # the latest 5-minute bar's volume vs normal
    trend: str = "unknown"                 # 5-minute chart: "up" | "down" | "sideways" | "unknown"
    trend_daily: str = "unknown"
    patterns: list[Pattern] = field(default_factory=list)   # strongest first
    levels: list[Level] = field(default_factory=list)       # nearest support / resistance, lowest first
    score: float = 0.0                     # -1 (very bearish) .. +1 (very bullish)
    summary: str = ""
    bars: dict = field(default_factory=dict)                 # bars used per chart

    def as_dict(self) -> dict:
        def num(x, nd: int | None = 2):
            if x is None or not math.isfinite(x):
                return None
            return float(f"{x:.6g}") if nd is None else round(float(x), nd)

        return {
            "symbol": self.symbol, "asset_class": self.asset_class, "at": self.at, "price": num(self.price, None),
            "bar_at": self.bar_at, "stale": bool(self.stale), "prev_close": num(self.prev_close, None),
            "day_change_pct": num(self.day_change_pct), "gap_pct": num(self.gap_pct), "vwap": num(self.vwap, None),
            "vwap_pct": num(self.vwap_pct), "vwap_atr": num(self.vwap_atr), "atr": num(self.atr, None),
            "atr_pct": num(self.atr_pct), "atr_basis": self.atr_basis, "rsi": num(self.rsi, 1),
            "rsi_daily": num(self.rsi_daily, 1), "macd_hist": num(self.macd_hist, None),
            "rel_volume": num(self.rel_volume), "rel_volume_basis": self.rel_volume_basis,
            "bar_rel_volume": num(self.bar_rel_volume), "trend": self.trend, "trend_daily": self.trend_daily,
            "patterns": [p.as_dict() for p in self.patterns], "levels": [lv.as_dict() for lv in self.levels],
            "score": round(float(self.score), 2), "summary": self.summary,
            "bars": {k: int(v) for k, v in self.bars.items()},
        }


@dataclass
class ChartSignal:
    """A buy/sell idea from the chart alone (the app keeps these watch-only until they prove themselves)."""

    symbol: str
    direction: str             # "bullish" | "bearish"
    confidence: int            # 0-100 (never above SIGNAL_MAX_CONFIDENCE)
    patterns: list[str]        # names of the patterns behind it
    reason: str                # plain English
    evidence: list[str] = field(default_factory=list)
    at: str | None = None

    def as_dict(self) -> dict:
        return {"symbol": self.symbol, "direction": self.direction, "confidence": int(self.confidence),
                "patterns": list(self.patterns), "reason": self.reason, "evidence": list(self.evidence),
                "at": self.at}


def _f(x) -> float | None:
    return float(x) if x is not None and math.isfinite(x) else None


def _today_bar(m1: Bars, today: int | None, crypto: bool) -> Bars | None:
    """Today's daily bar so far, built from today's regular-session 1-minute bars."""
    if today is None or not len(m1):
        return None
    day, sod, seg = ind.day_parts(m1.t, crypto)
    idx = np.flatnonzero((day == today) & (seg == REGULAR))
    if not len(idx):
        return None
    start = int(m1.t[idx[0]] - sod[idx[0]])  # midnight of that day
    return Bars(np.array([start], dtype=np.int64), m1.open[idx[:1]], np.array([np.nanmax(m1.high[idx])]),
                np.array([np.nanmin(m1.low[idx])]), m1.close[idx[-1:]], np.array([m1.volume[idx].sum()]))


def read_chart(symbol: str, minute_bars: list[dict] | None, daily_bars: list[dict] | None = None, now=None,
               asset_class: str = "stock", intraday_minutes: int = 5) -> ChartReading:
    """Read the chart of one symbol as of `now` (datetime / ISO; default: just after the last minute bar).

    minute_bars: 1-minute bars ({"t", "o", "h", "l", "c", "v"}), ideally the last 2 days (an earlier session
    makes "normal volume for the time of day" possible). daily_bars: daily bars, ideally a year (the 200-day
    average needs 200). Bars after `now` - or not finished by then - are ignored. asset_class "stock" or
    "crypto" (24/7: no opening range or gaps, a rolling 24-hour VWAP, UTC days)."""
    crypto = asset_class == "crypto"
    r = ChartReading(symbol=str(symbol or "").upper(), asset_class="crypto" if crypto else "stock")
    now_s = ind.seconds(now) if now is not None else None
    m1 = ind.to_arrays(minute_bars, now=now_s, bar_seconds=60)
    if now_s is None and len(m1):
        now_s = float(m1.t[-1] + 60)
    all_daily = ind.to_arrays(daily_bars)
    today = ind.trading_day(now_s, crypto) if now_s is not None else None
    done = all_daily
    if today is not None and len(all_daily):
        done = all_daily[ind.daily_days(all_daily.t, crypto) < today]
    partial = _today_bar(m1, today, crypto)
    daily = Bars.concat(done, partial) if partial is not None else done
    m5 = ind.resample(m1, intraday_minutes, now=now_s, crypto=crypto)
    tf = f"{intraday_minutes}m"
    f1 = Frame(m1, "1m", crypto, daily=done)
    f5 = Frame(m5, tf, crypto, daily=done)
    fd = Frame(daily, "1d", crypto, partial_last=partial is not None)
    r.at = ind.iso_time(now_s) if now_s is not None else None
    r.bars = {"1m": len(m1), tf: len(m5), "1d": len(daily)}
    if not len(m1) and not len(daily):
        r.summary = "No chart data yet"
        return r
    with np.errstate(all="ignore"):
        _snapshot(r, f1, f5, fd, now_s)
        found = (recent_patterns(f5, RECENT_BARS["5m"]) + recent_patterns(f1, RECENT_BARS["1m"])
                 + recent_patterns(fd, RECENT_BARS["1d"] + (1 if partial is not None else 0)))
        r.patterns = sorted(found, key=lambda p: -p.strength)
        r.levels = _nearby_levels(r.price, f5, fd)
    r.score = _score(r)
    r.summary = _summary(r)
    return r


def _snapshot(r: ChartReading, f1: Frame, f5: Frame, fd: Frame, now_s: float | None) -> None:
    if f1.n:
        i = f1.n - 1
        r.price = float(f1.c[i])
        r.bar_at = f1.bars.iso(i)
        r.stale = bool(now_s is None or now_s - (f1.bars.t[i] + 60) > STALE_MINUTES * 60)
        r.vwap = _f(f1.vwap[i])
        r.atr = _f(f1.ref_atr[i])
        r.atr_basis = "daily" if _f(f1.prior[3][i]) is not None else "estimated"
        r.prev_close = _f(f1.prior[0][i])
        ss = int(f1.session_start[i])
        if not f1.crypto and ss >= 0 and r.prev_close:
            r.gap_pct = _f(ind.pct_diff(f1.o[ss], r.prev_close))
        drv = ind.day_relative_volume(f1.bars, crypto=f1.crypto)
        if _f(drv[i]) is not None:
            r.rel_volume, r.rel_volume_basis = float(drv[i]), "today so far vs a normal day by this time"
    elif fd.n:
        i = fd.n - 1
        r.price, r.bar_at = float(fd.c[i]), fd.bars.iso(i)
        r.atr, r.atr_basis = _f(fd.atr[i]), "daily"
        r.prev_close = _f(fd.c[i - 1]) if i >= 1 else None
    if r.price is not None:
        r.day_change_pct = _f(ind.pct_diff(r.price, r.prev_close)) if r.prev_close else None
        r.vwap_pct = _f(ind.pct_diff(r.price, r.vwap)) if r.vwap else None
        r.vwap_atr = _f(ind.atr_distance(r.price, r.vwap, r.atr)) if r.vwap and r.atr else None
        r.atr_pct = 100 * r.atr / r.price if r.atr else None
    if f5.n:
        j = f5.n - 1
        r.rsi, r.macd_hist = _f(f5.rsi[j]), _f(f5.macd[2][j])
        r.bar_rel_volume = _f(f5.rvol[j])
        r.trend = trend_state(f5, j)
        if r.rel_volume is None:
            recent = f5.rvol[max(0, j - 2):j + 1]
            if np.isfinite(recent).any():
                r.rel_volume = float(np.nanmean(recent))
                r.rel_volume_basis = "the last few bars vs normal"
    if fd.n:
        r.rsi_daily = _f(fd.rsi[fd.n - 1])
        r.trend_daily = trend_state(fd, fd.n - 1)


def _nearby_levels(price: float | None, f5: Frame, fd: Frame, each: int = 3) -> list[Level]:
    """The closest few support levels below the price and resistance levels above it (5-minute + daily charts)."""
    if price is None:
        return []
    found = levels(f5, f5.n - 1) + [lv for lv in levels(fd, fd.n - 1) if lv.touches >= 2]
    out: list[Level] = []
    for lv in sorted(found, key=lambda x: abs(x.price - price)):
        if any(abs(lv.price - x.price) <= price * 0.001 for x in out):
            continue
        lv.kind = "resistance" if lv.price > price else "support"
        if sum(x.kind == lv.kind for x in out) < each:
            out.append(lv)
    return sorted(out, key=lambda x: x.price)


def _score(r: ChartReading) -> float:
    """-1..+1 from the trends, the side of VWAP and the recent patterns."""
    total = {"up": 0.25, "down": -0.25}.get(r.trend, 0.0) + {"up": 0.15, "down": -0.15}.get(r.trend_daily, 0.0)
    if r.vwap_pct is not None:
        total += 0.1 if r.vwap_pct > 0.05 else -0.1 if r.vwap_pct < -0.05 else 0.0
    for p in r.patterns:
        if p.kind == "trend" or p.direction == NEUTRAL:
            continue
        sign = 1 if p.direction == BULLISH else -1
        tf_weight = 1.0 if p.timeframe not in ("1m", "1d") else 0.8
        total += 0.3 * sign * p.strength * _KIND_WEIGHT.get(p.kind, 0.5) * tf_weight
    return round(math.tanh(1.2 * total), 2)


def _stretch(r: ChartReading, direction: str) -> tuple[list[str], bool]:
    """Why the price is already stretched in `direction` (empty = it isn't), and whether it's very stretched."""
    up = direction == BULLISH
    rsi = r.rsi if r.rsi is not None else 50.0
    rsi_d = r.rsi_daily if r.rsi_daily is not None else 50.0
    dist = r.vwap_atr if r.vwap_atr is not None else 0.0
    if not up:
        rsi, rsi_d, dist = 100 - rsi, 100 - rsi_d, -dist
    rsi_hit, daily_hit, vwap_hit = rsi > STRETCH_RSI, rsi_d > STRETCH_RSI, dist >= STRETCH_ATR
    if not (rsi_hit or daily_hit or vwap_hit):
        return [], False
    reasons = []
    if rsi_hit:
        reasons.append(f"RSI {r.rsi:.0f}")
    if daily_hit:
        reasons.append(f"daily RSI {r.rsi_daily:.0f}")
    if dist >= 1.0:
        reasons.append(f"{dist:.1f} ATR {'above' if up else 'below'} VWAP")
    very = rsi > VERY_STRETCHED_RSI or dist >= VERY_STRETCHED_ATR or (rsi_hit and vwap_hit)
    return reasons, very


def _lower_first(text: str) -> str:
    if len(text) > 1 and text[0].isupper() and text[1].islower():
        return text[0].lower() + text[1:]
    return text


def _event_words(p: Pattern) -> str:
    if p.name == "bollinger_breakout":
        return "breaking out of a squeeze" if p.direction == BULLISH else "breaking down out of a squeeze"
    return _EVENT_WORDS.get(p.name) or p.name.replace("_", " ") + " candle"


def _summary(r: ChartReading) -> str:
    if r.price is None:
        return "No chart data yet"
    for direction, label in ((BULLISH, "Stretched"), (BEARISH, "Stretched to the downside")):
        reasons, _ = _stretch(r, direction)
        if reasons:
            return f"{label}: {' and '.join(reasons)}"
    parts = []
    trend = _TREND_WORDS.get(r.trend) or {"up": "Daily uptrend", "down": "Daily downtrend"}.get(r.trend_daily)
    if trend:
        parts.append(trend)
    if r.vwap_pct is not None:
        parts.append("above VWAP" if r.vwap_pct > 0.05 else "below VWAP" if r.vwap_pct < -0.05 else "at VWAP")
    events = sorted((p for p in r.patterns if p.kind in ("structure", "indicator", "candle") and p.direction != NEUTRAL),
                    key=lambda p: (p.timeframe == "1d", -p.strength))
    if events:
        heavy = max(r.bar_rel_volume or 0.0, r.rel_volume or 0.0) >= HEAVY_VOLUME
        parts.append(_event_words(events[0]) + (" on heavy volume" if heavy else ""))
    elif r.trend in ("sideways", "unknown"):
        parts.append("nothing special on the chart")
    text = ", ".join(parts) if parts else "Nothing special on the chart"
    return text[0].upper() + text[1:]


def confirm(direction: str, reading: ChartReading | None) -> tuple[str, int, str]:
    """Does the chart back up a news signal? Returns (verdict, confidence_adjust, reason).

    verdict: "stretched" (the price already ran - RSI over 80 or 2+ ATR from VWAP in the signal's direction: the
    app sends it for review instead of chasing), "against" (the chart points the other way), "agrees" or
    "neutral". confidence_adjust is -15..+10 points."""
    if direction not in (BULLISH, BEARISH) or reading is None or reading.price is None:
        return "neutral", 0, "No chart data to check the signal against."
    reasons, very = _stretch(reading, direction)
    if reasons:
        verb = "Buying" if direction == BULLISH else "Selling"
        return ("stretched", -15 if very else -10,
                f"Chart says the move may already be done: {' and '.join(reasons)}. {verb} now would be chasing it.")
    ds = reading.score if direction == BULLISH else -reading.score
    opposite = BEARISH if direction == BULLISH else BULLISH
    against = [p for p in reading.patterns if p.direction == opposite and p.kind == "structure" and p.strength >= 0.5]
    if ds <= AGAINST_SCORE or (against and ds < 0.1):
        adjust = -5 - round(10 * min(1.0, max(0.0, (-ds + AGAINST_SCORE) / 0.5)))
        why = against[0].explain if against else _lower_first(reading.summary)
        return "against", adjust, f"Chart disagrees: {why}."
    if ds >= AGREE_SCORE:
        adjust = 2 + round(8 * min(1.0, (ds - AGREE_SCORE) / 0.55))
        return "agrees", adjust, f"Chart agrees: {_lower_first(reading.summary)}."
    return "neutral", 0, f"Chart is mixed: {_lower_first(reading.summary)}."


def chart_signal(reading: ChartReading | None) -> ChartSignal | None:
    """A signal from the chart alone, or None.

    Needs a fresh trigger pattern on the intraday charts (breakout, flag, double bottom, VWAP reclaim...), heavy
    volume (1.5x normal or more), at least SIGNAL_MIN_EVIDENCE agreeing pieces of evidence in all (trigger,
    volume, trend, daily trend, side of VWAP, MACD momentum, a candle), no trigger pointing the other way, the
    overall score leaning the same way, and a price that isn't already stretched. Never fires on a candlestick
    alone; confidence is capped at SIGNAL_MAX_CONFIDENCE."""
    if reading is None or reading.price is None or reading.stale:
        return None
    best = None
    for direction in (BULLISH, BEARISH):
        sig = _signal_for(reading, direction)
        if sig is not None and (best is None or sig.confidence > best.confidence):
            best = sig
    return best


def _signal_for(r: ChartReading, direction: str) -> ChartSignal | None:
    up = direction == BULLISH
    sign = 1 if up else -1
    fresh = [p for p in r.patterns if p.name in TRIGGERS and p.timeframe != "1d"]
    triggers = [p for p in fresh if p.direction == direction]
    if not triggers or any(p.direction not in (direction, NEUTRAL) for p in fresh):
        return None
    if _stretch(r, direction)[0] or sign * r.score < SIGNAL_MIN_SCORE:
        return None
    heavy = max(r.bar_rel_volume or 0.0, r.rel_volume or 0.0)
    if heavy < HEAVY_VOLUME:
        return None
    triggers = triggers[:2]
    names = [p.name for p in triggers]
    evidence = [triggers[0].explain] + [_event_words(p) for p in triggers[1:]]
    evidence.append(f"heavy volume ({heavy:.1f}x normal)")
    want = "up" if up else "down"
    if r.trend == want:
        evidence.append("the 5-minute trend agrees")
    if r.trend_daily == want:
        evidence.append("the daily trend agrees")
    if r.vwap_pct is not None and sign * r.vwap_pct > 0:
        evidence.append("above VWAP" if up else "below VWAP")
    if r.macd_hist is not None and sign * r.macd_hist > 0:
        evidence.append("MACD momentum agrees")
    candle = next((p for p in r.patterns if p.kind == "candle" and p.direction == direction), None)
    if candle is not None:
        evidence.append(f"{candle.name.replace('_', ' ')} candle")
        names.append(candle.name)
    if len(evidence) < SIGNAL_MIN_EVIDENCE:
        return None
    strength = max(p.strength for p in triggers)
    confidence = min(SIGNAL_MAX_CONFIDENCE, round(40 + 5 * (len(evidence) - SIGNAL_MIN_EVIDENCE) + 15 * strength))
    reason = evidence[0] + (" - plus " + ", ".join(evidence[1:]) if len(evidence) > 1 else "")
    return ChartSignal(r.symbol, direction, int(confidence), names, reason, evidence, r.at)
