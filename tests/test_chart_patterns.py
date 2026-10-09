"""Chart engine patterns: each detector on a made-up chart that should show it and one that shouldn't, support /
resistance levels, and the no-look-ahead rule (bars added later never change what was found earlier)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from newstrader.chart import indicators as ind
from newstrader.chart.indicators import to_arrays
from newstrader.chart.patterns import Frame, find_patterns, levels, recent_patterns, trend_state
from tests.test_chart_indicators import MONDAY, candles, daily_path, ny, path, walk

DAY = timedelta(days=1)
FIVE = timedelta(minutes=5)


# ---------------------------------------------------------------------------------------------- chart builders
def trend(start: float, step: float, n: int, wick: float = 0.3, vol: float = 1e6) -> list[tuple]:
    """n bars moving `step` each, every bar opening at the previous close."""
    rows, prev = [], start
    for _ in range(n):
        c = prev + step
        rows.append((prev, max(prev, c) + wick, min(prev, c) - wick, c, vol))
        prev = c
    return rows


def wiggle(level: float, n: int, size: float = 0.5, vol: float = 1e6) -> list[tuple]:
    """n bars going sideways around `level`."""
    rows = []
    for k in range(n):
        o, c = (level, level + size) if k % 2 == 0 else (level + size, level)
        rows.append((o, max(o, c) + 0.3, min(o, c) - 0.3, c, vol))
    return rows


DOWN = trend(130, -1, 20)      # ends at 110, lows down to 109.7
UP = trend(90, 1, 20)          # ends at 110, highs up to 110.3
FLAT = wiggle(110, 20)


def daily(rows: list[tuple]) -> Frame:
    return Frame(to_arrays(candles(rows, ny(date(2025, 1, 1), 0), DAY)), "1d")


def names(f: Frame, i: int | None = None) -> dict[str, str]:
    """{pattern name: direction} at bar i (default: the last one)."""
    return {p.name: p.direction for p in find_patterns(f, i)}


def five_minute(rows: list[tuple], day: date = MONDAY, hh: int = 9, mm: int = 30, **kw) -> Frame:
    return Frame(to_arrays(candles(rows, ny(day, hh, mm), FIVE)), "5m", **kw)


# ---------------------------------------------------------------------------------------------- candlesticks
HAMMER_LOW = (109.8, 109.95, 107.3, 109.9, 1e6)       # long lower tail, at a new low
HAMMER_HIGH = (110.2, 110.4, 107.8, 110.35, 1e6)      # the same shape, at a new high


def test_hammer_only_after_a_fall():
    hammer = next(p for p in find_patterns(daily(DOWN + [HAMMER_LOW])) if p.name == "hammer")
    assert hammer.direction == "bullish" and hammer.kind == "candle" and hammer.level == 107.3
    after_rise = names(daily(UP + [HAMMER_HIGH]))
    assert "hammer" not in after_rise and after_rise.get("hanging_man") == "bearish"
    sideways = names(daily(FLAT + [(110.4, 110.55, 108.0, 110.5, 1e6)]))
    assert not {"hammer", "hanging_man"} & set(sideways)


def test_shooting_star_and_inverted_hammer():
    assert names(daily(UP + [(110.2, 112.8, 110.05, 110.1, 1e6)])).get("shooting_star") == "bearish"
    assert names(daily(DOWN + [(109.6, 112.2, 109.55, 109.7, 1e6)])).get("inverted_hammer") == "bullish"
    assert "shooting_star" not in names(daily(DOWN + [(109.6, 112.2, 109.55, 109.7, 1e6)]))


def test_doji_only_at_an_extreme():
    assert names(daily(UP + [(110.3, 111.3, 109.5, 110.32, 1e6)])).get("doji") == "bearish"
    assert "doji" not in names(daily(FLAT + [(110.3, 111.0, 109.6, 110.32, 1e6)]))


def test_engulfing_needs_the_right_trend():
    bull = [(110, 110.1, 109.2, 109.3, 1e6), (109.2, 110.7, 109.1, 110.6, 1e6)]
    assert names(daily(DOWN + bull)).get("bullish_engulfing") == "bullish"
    assert "bullish_engulfing" not in names(daily(UP + bull))
    bear = [(110, 110.8, 109.9, 110.7, 1e6), (110.8, 110.9, 109.3, 109.4, 1e6)]
    assert names(daily(UP + bear)).get("bearish_engulfing") == "bearish"
    assert "bearish_engulfing" not in names(daily(DOWN + bear))


def test_morning_and_evening_star():
    morning = [(110, 110.1, 108.4, 108.5, 1e6), (108.4, 108.7, 108.0, 108.5, 1e6), (108.6, 109.7, 108.5, 109.6, 1e6)]
    assert names(daily(DOWN + morning)).get("morning_star") == "bullish"
    weak = morning[:2] + [(108.6, 108.9, 108.5, 108.8, 1e6)]          # third candle doesn't get back halfway
    assert "morning_star" not in names(daily(DOWN + weak))
    evening = [(110, 111.6, 109.9, 111.5, 1e6), (111.6, 112.0, 111.3, 111.5, 1e6), (111.4, 111.5, 110.3, 110.4, 1e6)]
    assert names(daily(UP + evening)).get("evening_star") == "bearish"
    assert "evening_star" not in names(daily(DOWN + evening))


def test_three_white_soldiers_and_black_crows():
    soldiers = [(110, 111.1, 109.9, 111.0, 1e6), (110.9, 112.1, 110.8, 112.0, 1e6), (111.9, 113.1, 111.8, 113.0, 1e6)]
    assert names(daily(DOWN + soldiers)).get("three_white_soldiers") == "bullish"
    assert "three_white_soldiers" not in names(daily(UP + soldiers))      # already ran up: not a fresh sign
    crows = [(110, 110.1, 108.9, 109.0, 1e6), (109.1, 109.2, 107.9, 108.0, 1e6), (108.1, 108.2, 106.9, 107.0, 1e6)]
    assert names(daily(UP + crows)).get("three_black_crows") == "bearish"
    small = [(110, 110.3, 109.9, 110.2, 1e6), (110.2, 110.5, 110.1, 110.4, 1e6), (110.4, 110.7, 110.3, 110.6, 1e6)]
    assert "three_white_soldiers" not in names(daily(DOWN + small))


def test_candles_skip_todays_unfinished_daily_bar():
    bars = to_arrays(candles(DOWN + [HAMMER_LOW], ny(date(2025, 1, 1), 0), DAY))
    assert "hammer" not in names(Frame(bars, "1d", partial_last=True))


# ---------------------------------------------------------------------------------------------- indicators
def closes_frame(closes, wick: float = 0.3, volumes=None) -> Frame:
    return Frame(to_arrays(path(closes, ny(date(2025, 1, 1), 0), DAY, volumes=volumes, wick=wick)), "1d")


def test_rsi_overbought_and_oversold():
    up = names(closes_frame(np.arange(100, 130.0)))
    assert up.get("rsi_overbought") == "bearish"
    strong = next(x for x in find_patterns(closes_frame(np.arange(100, 130.0))) if x.name == "rsi_overbought")
    assert strong.strength == 0.8 and "very overbought" in strong.explain and strong.kind == "caution"
    steps = np.cumsum([3, -1] * 20) + 100.0                  # RSI settles around 75: overbought, not very
    mild = next(x for x in find_patterns(closes_frame(steps)) if x.name == "rsi_overbought")
    assert mild.strength == 0.5 and "very" not in mild.explain
    assert names(closes_frame(np.arange(130, 100.0, -1))).get("rsi_oversold") == "bullish"
    assert not {"rsi_overbought", "rsi_oversold"} & set(names(closes_frame(np.cumsum([1, -1] * 20) + 100.0)))


def _divergence_closes(first_leg: float, second_leg: float) -> list[float]:
    closes = [100.0, 100.2] * 10
    for steps, size in ((8, first_leg), (5, -1), (10, second_leg), (4, -1)):
        for _ in range(steps):
            closes.append(closes[-1] + size)
    return closes


def test_rsi_divergence_when_the_second_high_is_weaker():
    f = closes_frame(_divergence_closes(2.0, 0.6))       # strong rise to 116, dip, slow crawl to 117
    found = [p for k in range(f.n) for p in find_patterns(f, k) if p.name == "rsi_divergence"]
    assert len(found) == 1 and found[0].direction == "bearish"
    assert found[0].level == pytest.approx(117.5)      # 117.2 close + 0.3 wick
    strong = closes_frame(_divergence_closes(0.8, 2.0))  # second rise stronger than the first: no divergence
    assert not [p for k in range(strong.n) for p in find_patterns(strong, k) if p.name == "rsi_divergence"]


def test_macd_cross_after_a_turn_but_not_on_a_steady_climb():
    falling_faster = [130 - 0.01 * k * k for k in range(50)]
    f = closes_frame(falling_faster + [falling_faster[-1] + 0.8 * k for k in range(1, 25)])
    hist = f.macd[2]
    k = int(np.flatnonzero((hist[:-1] <= 0) & (hist[1:] > 0))[-1]) + 1
    assert any(p.name == "macd_cross" and p.direction == "bullish" for p in find_patterns(f, k))
    assert not any(p.name == "macd_cross" for p in find_patterns(f, k - 1))
    ramp = closes_frame(np.arange(100, 200.0))
    assert not [p for k in range(ramp.n) for p in find_patterns(ramp, k) if p.name.startswith("macd")]


def test_bollinger_squeeze_then_breakout():
    wide = list(np.cumsum([2.0, -2.0] * 30) + 100)
    tight = list(np.cumsum([0.1, -0.1] * 13) + 100)
    vols = [1e6] * (len(wide) + len(tight)) + [3e6]
    f = closes_frame(wide + tight + [103.0], wick=0.05, volumes=vols)
    assert names(f).get("bollinger_breakout") == "bullish"
    assert names(f, f.n - 2).get("bollinger_squeeze") == "neutral"
    no_squeeze = closes_frame(wide + [110.0], wick=0.05)
    assert "bollinger_breakout" not in names(no_squeeze)


def test_golden_and_death_cross_fire_once():
    closes = list(np.linspace(150, 100, 220)) + list(np.linspace(100.5, 140, 80))
    f = closes_frame(closes)
    found = [(k, p.name) for k in range(f.n) for p in find_patterns(f, k) if p.name in ("golden_cross", "death_cross")]
    assert len(found) == 1 and found[0][1] == "golden_cross"
    k = found[0][0]
    assert f.sma50[k - 1] <= f.sma200[k - 1] and f.sma50[k] > f.sma200[k]
    down = closes_frame(list(np.linspace(100, 150, 220)) + list(np.linspace(149.5, 110, 80)))
    assert [p.name for k in range(down.n) for p in find_patterns(down, k) if "cross" in p.name and "macd" not in p.name] \
        == ["death_cross"]


def test_moving_average_trend():
    up = closes_frame(np.arange(100, 160.0))
    assert trend_state(up, up.n - 1) == "up" and names(up).get("uptrend") == "bullish"
    down = closes_frame(np.arange(160, 100.0, -1))
    assert trend_state(down, down.n - 1) == "down" and names(down).get("downtrend") == "bearish"
    flat = closes_frame(np.cumsum([1, -1] * 30) + 100.0)
    assert trend_state(flat, flat.n - 1) == "sideways" and not {"uptrend", "downtrend"} & set(names(flat))
    assert trend_state(closes_frame([100.0, 101.0]), 1) == "unknown"


def test_vwap_reclaim_and_loss_on_the_5_minute_chart():
    rows = wiggle(100, 6, 0.2, 1000) + trend(100, -0.4, 5, 0.1, 1000) + wiggle(98, 2, 0.2, 1000) \
        + [(98.2, 100.6, 98.1, 100.5, 3000)]
    f = five_minute(rows)
    assert f.c[-2] < f.vwap[-2] and f.c[-1] > f.vwap[-1]
    assert names(f).get("vwap_reclaim") == "bullish"
    chop = five_minute(wiggle(100, 10, 0.2, 1000) + [(100.2, 101.0, 100.1, 100.9, 1000)])
    assert "vwap_reclaim" not in names(chop)                                 # just wobbling around VWAP
    lost = five_minute(wiggle(100, 6, 0.2, 1000) + trend(100, 0.4, 5, 0.1, 1000) + wiggle(102, 2, 0.2, 1000)
                       + [(102.0, 102.1, 99.4, 99.5, 3000)])
    assert names(lost).get("vwap_lost") == "bearish"


def _daily_history(end: date, last_close: float = 100.0, n: int = 30) -> ind.Bars:
    """n flat-ish days ending yesterday with a close of last_close and a daily ATR of about 2."""
    return to_arrays(daily_path([last_close] * n, end, wick=1.0))


def test_stretched_far_from_vwap():
    hist = _daily_history(MONDAY)
    minutes = path([100.0] * 30 + [106.0] * 3, ny(MONDAY, 9, 30), wick=0.05)
    f = Frame(to_arrays(minutes), "1m", daily=hist)
    assert f.ref_atr[-1] == pytest.approx(2.0)
    p = next(x for x in find_patterns(f) if x.name == "stretched_above_vwap")
    assert p.direction == "bearish" and p.kind == "caution" and "normal day's move above VWAP" in p.explain
    calm = Frame(to_arrays(path([100.0] * 30 + [101.0] * 3, ny(MONDAY, 9, 30), wick=0.05)), "1m", daily=hist)
    assert "stretched_above_vwap" not in names(calm)
    low = Frame(to_arrays(path([100.0] * 30 + [94.0] * 3, ny(MONDAY, 9, 30), wick=0.05)), "1m", daily=hist)
    assert names(low).get("stretched_below_vwap") == "bullish"


# ---------------------------------------------------------------------------------------------- structure
def _range_then(last: list[tuple]) -> Frame:
    """Three trips between 100 and 105 (so 105 is resistance, 100 support), then `last`."""
    rows = wiggle(102, 20, 0.2)
    for _ in range(3):
        rows += trend(rows[-1][3], 1, 5 - (rows[-1][3] - 100)) if rows[-1][3] < 100 else []
        rows += trend(100, 1, 5) + trend(105, -1, 5)
    return daily(rows + last)


def test_support_and_resistance_levels_count_touches():
    f = _range_then(trend(100, 0.5, 6))
    found = levels(f, f.n - 1)
    res = [lv for lv in found if lv.kind == "resistance"]
    sup = [lv for lv in found if lv.kind == "support"]
    assert any(abs(lv.price - 105.3) < 0.2 and lv.touches >= 3 for lv in res)
    assert any(abs(lv.price - 99.7) < 0.2 and lv.touches >= 2 for lv in sup)
    assert all(lv.price > f.c[-1] for lv in res) and all(lv.price < f.c[-1] for lv in sup)


def test_breakout_needs_a_close_through_resistance_on_heavy_volume():
    approach = trend(100, 1, 4)        # up to 104
    heavy = _range_then(approach + [(104, 106.6, 103.9, 106.5, 3e6)])
    p = next(x for x in find_patterns(heavy) if x.name == "breakout")
    assert p.direction == "bullish" and p.kind == "structure" and p.level == pytest.approx(105.3, abs=0.2)
    assert "3.0x normal volume" in p.explain and "3 times" in p.explain
    quiet = _range_then(approach + [(104, 106.6, 103.9, 106.5, 1e6)])
    assert "breakout" not in names(quiet)
    wick_only = _range_then(approach + [(104, 106.6, 103.9, 104.9, 3e6)])    # poked above, closed below
    assert "breakout" not in names(wick_only)


def test_breakdown_below_support():
    f = _range_then(trend(100, 1, 1) + [(101, 101.1, 98.4, 98.5, 3e6)])
    assert names(f).get("breakdown") == "bearish"
    quiet = _range_then(trend(100, 1, 1) + [(101, 101.1, 98.4, 98.5, 1e6)])
    assert "breakdown" not in names(quiet)


def test_opening_range_breakout_first_time_only():
    rows = [(100, 101, 99.5, 100.5, 1000), (100.5, 100.9, 99, 99.5, 1000), (99.5, 100.4, 99.2, 100.2, 1000),
            (100.2, 100.8, 100.0, 100.6, 1000), (100.6, 101.6, 100.5, 101.5, 1000), (101.5, 102.1, 101.4, 102.0, 1000)]
    f = five_minute(rows)
    assert names(f, 4).get("opening_range_breakout") == "bullish"           # 9:50 closes above 101
    assert "opening_range_breakout" not in names(f, 5)                      # 9:55: not the first time
    assert "opening_range_breakout" not in names(f, 3)
    early = five_minute(rows[:2] + [(99.5, 102.4, 99.2, 102.2, 1000)])       # 9:40 is still inside the range
    assert not [p for k in range(early.n) for p in find_patterns(early, k) if p.name.startswith("opening_range")]
    down = five_minute(rows[:4] + [(100.6, 100.7, 98.4, 98.5, 1000)])
    assert names(down).get("opening_range_breakdown") == "bearish"
    crypto = five_minute(rows, crypto=True)
    assert "opening_range_breakout" not in names(crypto, 4)


def test_gap_up_then_gap_fill():
    hist = _daily_history(MONDAY)                                            # yesterday's close: 100
    today = path(list(np.linspace(104, 103, 20)) + [100.1, 102], ny(MONDAY, 9, 30), wick=0.2, first_open=104.0)
    f = Frame(to_arrays(today), "1m", daily=hist)
    p = next(x for x in find_patterns(f, 10) if x.name == "gap_up")
    assert p.direction == "bullish" and "4.0%" in p.explain and p.level == 100
    assert names(f, 20).get("gap_fill") == "bearish"                         # low 99.9: back to 100
    small = Frame(to_arrays(path([101.0] * 5, ny(MONDAY, 9, 30), first_open=101.0)), "1m", daily=hist)
    assert not {"gap_up", "gap_fill"} & set(names(small))


def test_gap_fill_fires_once_when_price_reaches_yesterdays_close():
    hist = _daily_history(MONDAY)
    today = path([104.0] * 10 + [99.8, 101.0, 102.0], ny(MONDAY, 9, 30), wick=0.1, first_open=104.0)
    f = Frame(to_arrays(today), "1m", daily=hist)
    fills = [k for k in range(f.n) for p in find_patterns(f, k) if p.name == "gap_fill"]
    assert fills == [10]
    assert not {"gap_up", "gap_fill"} & set(names(f))                        # after the fill: nothing


def test_new_high_of_day_after_the_opening_range():
    rows = path([100.0] * 20 + [101.0], ny(MONDAY, 9, 30), wick=0.3)
    f = Frame(to_arrays(rows), "1m")
    assert names(f).get("new_high_of_day") == "bullish"
    early = Frame(to_arrays(path([100.0] * 3 + [101.0], ny(MONDAY, 9, 30), wick=0.3)), "1m")
    assert "new_high_of_day" not in names(early)                             # 9:33: still the opening range


def _double(second_peak: float) -> Frame:
    rows = wiggle(100, 15, 0.3) + trend(100, 1, 10) + trend(110, -1, 6) + trend(104, 1, 6)
    if second_peak != 110:
        rows[-1] = (109, second_peak + 0.3, 108.7, second_peak, 1e6)
    return daily(rows + trend(rows[-1][3], -1, 7))


def test_double_top_on_the_neckline_break():
    f = _double(110)
    found = [(k, p) for k in range(f.n) for p in find_patterns(f, k) if p.name == "double_top"]
    assert len(found) == 1
    k, p = found[0]
    assert p.direction == "bearish" and p.level == pytest.approx(103.7) and f.c[k] < 103.7 <= f.c[k - 1]
    assert not [p for k in range(_double(115).n) for p in find_patterns(_double(115), k) if p.name == "double_top"]


def test_double_bottom_on_the_neckline_break():
    rows = wiggle(110, 15, 0.3) + trend(110, -1, 10) + trend(100, 1, 6) + trend(106, -1, 6) + trend(100, 1, 7)
    f = daily(rows)
    found = [p for k in range(f.n) for p in find_patterns(f, k) if p.name == "double_bottom"]
    assert len(found) == 1 and found[0].direction == "bullish" and found[0].level == pytest.approx(106.3)
    lower = daily(wiggle(110, 15, 0.3) + trend(110, -1, 10) + trend(100, 1, 6) + trend(106, -1.5, 6)
                  + trend(97, 1.5, 7))                                      # second low 3% lower: not a double
    assert not [p for k in range(lower.n) for p in find_patterns(lower, k) if p.name == "double_bottom"]


def test_bull_flag_and_bear_flag():
    base = wiggle(100, 20, 0.4)
    bull = daily(base + trend(100, 1.5, 5) + trend(107.5, -0.3, 5) + [(106, 108.3, 105.9, 108.2, 1e6)])
    assert names(bull).get("bull_flag") == "bullish"
    no_pole = daily(base + trend(100, 0.2, 5) + trend(101, -0.1, 5) + [(100.5, 102.3, 100.4, 102.2, 1e6)])
    assert "bull_flag" not in names(no_pole)
    deep = daily(base + trend(100, 1.5, 5) + trend(107.5, -1.0, 5) + [(102.5, 108.3, 102.4, 108.2, 1e6)])
    assert "bull_flag" not in names(deep)                                    # gave back too much: not a flag
    bear = daily(base + trend(100, -1.5, 5) + trend(92.5, 0.3, 5) + [(94, 94.1, 91.7, 91.8, 1e6)])
    assert names(bear).get("bear_flag") == "bearish"


def test_volume_climax_after_a_run():
    run = wiggle(100, 10, 0.3) + trend(100, 1, 15)
    exhausted = daily(run + [(115, 118, 114.5, 115.2, 4e6)])
    assert names(exhausted).get("volume_climax") == "bearish"
    still_strong = daily(run + [(115, 118.1, 114.9, 118.0, 4e6)])
    assert names(still_strong).get("volume_climax") == "neutral"
    normal_volume = daily(run + [(115, 118, 114.5, 115.2, 1e6)])
    assert "volume_climax" not in names(normal_volume)


# ---------------------------------------------------------------------------------------------- no look-ahead
def _random_minutes(seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for day, s in ((date(2026, 10, 6), seed), (date(2026, 10, 7), seed + 1)):
        vols = rng.integers(100, 3000, 960).astype(float)
        vols[rng.integers(0, 960, 25)] *= 6                       # some volume bursts
        out += path(walk(960, s, 100, 0.0015), ny(day, 4, 0), volumes=vols, wick=0.03)
    return out


@pytest.mark.parametrize("seed", [1, 7])
def test_adding_later_bars_never_changes_earlier_patterns(seed):
    minutes = to_arrays(_random_minutes(seed))
    daily_rows = to_arrays(daily_path(list(walk(260, seed, 100, 0.02)), date(2026, 10, 6)))
    m5 = ind.resample(minutes, 5)
    frames = [
        (lambda b: Frame(b, "5m", daily=daily_rows), m5),
        (lambda b: Frame(b, "1m", daily=daily_rows), minutes),
        (lambda b: Frame(b, "1d"), daily_rows),
    ]
    checked = fired = 0
    for build, bars in frames:
        full = build(bars)
        step = 20 if len(bars) > 1000 else 3 if len(bars) > 300 else 2
        for i in range(30, len(bars), step):
            later = [p.as_dict() for p in find_patterns(full, i)]
            then = [p.as_dict() for p in find_patterns(build(bars[: i + 1]), i)]
            assert later == then, (full.timeframe, i)
            checked += 1
            fired += len(then)
    assert checked > 250 and fired > 100       # the test only means something if patterns actually showed up


def test_recent_patterns_lists_each_pattern_once_newest_first():
    f = closes_frame(np.arange(100, 140.0))
    found = recent_patterns(f, 3)
    assert [p.name for p in found].count("uptrend") == 1
    assert found[0].at == f.bars.iso(f.n - 1)
    assert recent_patterns(Frame(ind.Bars.empty(), "5m")) == []
    assert find_patterns(Frame(ind.Bars.empty(), "1m")) == []
