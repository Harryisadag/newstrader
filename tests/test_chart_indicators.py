"""Chart engine indicators: values against hand-worked / textbook examples, awkward input (short, flat, NaN, zero
volume), VWAP's daily restart at the New York open (summer and winter time), relative volume and resampling."""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from newstrader.chart import indicators as ind
from newstrader.chart.indicators import Bars, to_arrays

NY = ZoneInfo("America/New_York")
FRIDAY, MONDAY = date(2026, 3, 6), date(2026, 3, 9)  # US clocks went forward on Sunday March 8, 2026


# ---------------------------------------------------------------------------------------------- test data helpers
def iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


def ny(day: date, hh: int, mm: int = 0) -> datetime:
    return datetime.combine(day, time(hh, mm), NY)


def candles(rows: list[tuple], start: datetime, step: timedelta = timedelta(minutes=1)) -> list[dict]:
    """Bar dicts from (open, high, low, close, volume) tuples, one every `step` from `start`."""
    return [{"t": iso(start + k * step), "o": o, "h": h, "l": lo, "c": c, "v": v}
            for k, (o, h, lo, c, v) in enumerate(rows)]


def path(closes, start: datetime, step: timedelta = timedelta(minutes=1), volumes=None, wick: float = 0.05,
         first_open: float | None = None) -> list[dict]:
    """Bars that walk through `closes`: each opens at the previous close, wicks stick out by `wick`."""
    rows, prev = [], closes[0] if first_open is None else first_open
    for k, c in enumerate(closes):
        v = volumes[k] if volumes is not None else 1000
        rows.append((prev, max(prev, c) + wick, min(prev, c) - wick, float(c), float(v)))
        prev = c
    return candles(rows, start, step)


def weekdays(end: date, n: int) -> list[date]:
    """The n weekdays before `end`, oldest first."""
    out, d = [], end
    while len(out) < n:
        d -= timedelta(days=1)
        if d.weekday() < 5:
            out.append(d)
    return out[::-1]


def daily_path(closes, end: date, volume: float = 1e6, wick: float = 1.0) -> list[dict]:
    """Daily bars (stamped at midnight New York time) for the weekdays before `end`."""
    days = weekdays(end, len(closes))
    rows, prev = [], closes[0]
    for d, c in zip(days, closes, strict=True):
        rows.append({"t": iso(ny(d, 0)), "o": prev, "h": max(prev, c) + wick, "l": min(prev, c) - wick,
                     "c": float(c), "v": volume})
        prev = c
    return rows


def walk(n: int, seed: int, start: float = 100.0, vol: float = 0.001) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return start * np.exp(np.cumsum(rng.normal(0, vol, n)))


# ---------------------------------------------------------------------------------------------- textbook values
WILDER_CLOSES = [44.3389, 44.0902, 44.1497, 43.6124, 44.3278, 44.8264, 45.0955, 45.4245, 45.8433, 46.0826,
                 45.8931, 46.0328, 45.614, 46.282, 46.282, 46.0028, 46.0328, 46.4116, 46.2222, 45.6439, 46.2122,
                 46.2521, 45.7137, 46.4515, 45.7835, 45.3548, 44.0288, 44.1783, 44.2181, 44.5672, 43.4205,
                 42.6628, 43.1314]
# StockCharts' worked RSI example (Wilder's method, 14 periods)
WILDER_RSI = [70.53, 66.32, 66.55, 69.41, 66.36, 57.97, 62.93, 63.26, 56.06, 62.38, 54.71, 50.42, 39.99, 41.46,
              41.87, 45.46, 37.30, 33.08, 37.77]


def test_rsi_matches_wilders_classic_example():
    r = ind.rsi(WILDER_CLOSES, 14)
    assert np.isnan(r[:14]).all()
    assert np.round(r[14:], 2).tolist() == pytest.approx(WILDER_RSI, abs=0.006)


def test_rsi_flat_rising_falling():
    assert ind.rsi([5.0] * 20)[-1] == 50       # no movement at all: neutral
    assert ind.rsi(list(range(30)))[-1] == 100
    assert ind.rsi(list(range(30, 0, -1)))[-1] == 0


def test_sma_and_ema_hand_worked():
    assert ind.sma([1, 2, 3, 4, 5], 3)[2:].tolist() == [2, 3, 4]
    # EMA 3 (weight 0.5) starts from the average of the first 3 values: 4, then 4+.5(8-4)=6, 6+.5(12-6)=9...
    e = ind.ema([2, 4, 6, 8, 12, 7], 3)
    assert np.isnan(e[:2]).all() and e[2:].tolist() == [4, 6, 9, 8]
    # Wilder's smoothing (weight 1/n): 3 -> 3 + (6-3)/3 = 4
    assert ind.rma([2, 3, 4, 6], 3)[2:].tolist() == pytest.approx([3, 4])


def test_macd_of_a_steady_climb_is_seven():
    """Price +1 every bar: EMA12 lags 5.5 and EMA26 lags 12.5, so the MACD line is exactly 7 and the histogram 0."""
    line, sig, hist = ind.macd(np.arange(60.0) + 100)
    assert np.isnan(line[:25]).all() and line[25:] == pytest.approx(7.0)
    assert np.isnan(sig[:33]).all() and sig[33:] == pytest.approx(7.0)
    assert hist[33:] == pytest.approx(0.0, abs=1e-9)


def test_macd_small_hand_worked():
    # fast EMA2: 1.5, 3.1667, 3.0556, 4.3519, 5.4506; slow EMA3: 2.3333, 2.6667, 3.8333, 4.9167
    line, sig, hist = ind.macd([1, 2, 4, 3, 5, 6], fast=2, slow=3, signal=2)
    assert line[2:] == pytest.approx([0.8333, 0.3889, 0.5185, 0.5340], abs=1e-4)
    assert sig[3:] == pytest.approx([0.6111, 0.5494, 0.5391], abs=1e-4)
    assert hist[5] == pytest.approx(-0.0051, abs=1e-4)


def test_bollinger_population_std():
    mid, up, low = ind.bollinger([1, 2, 3, 4, 5], 5)
    assert mid[4] == 3 and up[4] == pytest.approx(3 + 2 * math.sqrt(2)) and low[4] == pytest.approx(3 - 2 * math.sqrt(2))
    mid, up, low = ind.bollinger([2, 4, 6, 8], 3)   # windows (2,4,6) and (4,6,8): std sqrt(8/3)
    assert mid[2:].tolist() == [4, 6]
    assert up[2:] == pytest.approx([4 + 2 * math.sqrt(8 / 3), 6 + 2 * math.sqrt(8 / 3)])
    assert ind.bandwidth(up, low, mid)[3] == pytest.approx(4 * math.sqrt(8 / 3) / 6)


def test_atr_true_range_and_wilder():
    # true ranges: 1 (first bar: high-low), max(2, |3-1.5|, |1-1.5|) = 2, max(2, 1.5, .5) = 2
    assert ind.true_range([2, 3, 4], [1, 1, 2], [1.5, 2.5, 3]).tolist() == [1, 2, 2]
    assert ind.atr([2, 3, 4], [1, 1, 2], [1.5, 2.5, 3], 2)[1:].tolist() == [1.5, 1.75]
    # a gap counts: yesterday's close 10, today 12-12.5 -> true range 2.5
    assert ind.true_range([10.5, 12.5], [9.5, 12], [10, 12.2])[1] == 2.5


def test_stochastic_obv_roc_highest_lowest():
    k, d = ind.stochastic([10, 11, 12, 13], [8, 9, 10, 11], [9, 10, 11, 12.5], k=3, smooth_k=1, d=2)
    assert k[2:].tolist() == [75, 87.5] and d[3] == 81.25
    k, d = ind.stochastic([5] * 20, [5] * 20, [5] * 20)
    assert k[-1] == 50 and d[-1] == 50                     # flat range: middle
    assert ind.obv([10, 11, 11, 10, 12], [100, 200, 300, 400, 500]).tolist() == [0, 200, 200, -200, 300]
    assert ind.roc([100, 110, 99], 1)[1:].tolist() == pytest.approx([10, -10])
    assert ind.highest([1, 3, 2, 5, 4], 3)[2:].tolist() == [3, 5, 5]
    assert ind.lowest([1, 3, 2, 5, 4], 3)[2:].tolist() == [1, 2, 2]


def test_percent_and_atr_distance_helpers():
    assert ind.pct_diff(105, 100) == pytest.approx(5.0)
    assert math.isnan(ind.pct_diff(1, 0))
    assert ind.atr_distance(106, 100, 2) == 3.0
    assert ind.pct_diff(np.array([110, 90]), 100).tolist() == pytest.approx([10, -10])


# ---------------------------------------------------------------------------------------------- awkward input
@pytest.mark.parametrize("data", [[], [1.0], [1.0, 2.0], [5.0] * 40, [1.0, np.nan, 3.0] * 15, [np.inf, 2.0] * 20])
def test_short_flat_and_missing_values_never_crash(data):
    n = len(data)
    for out in (ind.sma(data, 5), ind.ema(data, 5), ind.rma(data, 5), ind.rsi(data), ind.roc(data, 3),
                ind.highest(data, 4), ind.lowest(data, 4), *ind.macd(data), *ind.bollinger(data),
                ind.atr(data, data, data), *ind.stochastic(data, data, data), ind.obv(data, [0.0] * n)):
        assert len(out) == n
    hi, lo = ind.pivots(data, data)
    assert len(hi) == len(lo) == n


def test_missing_value_is_skipped_not_spread():
    x = np.arange(40.0)
    x[20] = np.nan
    e = ind.ema(x, 5)
    assert np.isnan(e[20]) and np.isfinite(e[21:]).all()
    r = ind.rsi(x)
    assert np.isfinite(r[25:]).all()


def test_zero_volume_gives_nan_vwap_and_relative_volume_not_errors():
    bars = to_arrays(path([100, 101, 102, 101] * 10, ny(MONDAY, 10), volumes=[0] * 40))
    assert np.isnan(ind.vwap(bars)).all()
    assert np.isnan(ind.relative_volume(bars)).all()
    assert np.isnan(ind.day_relative_volume(bars)).all()
    assert ind.obv(bars.close, bars.volume).tolist() == [0.0] * 40


def test_to_arrays_cleans_and_sorts():
    rows = [
        {"t": "2026-10-07T15:02:00Z", "o": 10, "h": 11, "l": 9, "c": 10.5, "v": 100},
        {"t": "2026-10-07T15:00:00.000Z", "o": None, "h": None, "l": None, "c": "10", "v": None},
        {"t": "2026-10-07T11:01:00-04:00", "o": 10, "h": 9, "l": 11, "c": 10.2, "v": -5},  # same as 15:01Z
        {"t": "2026-10-07T15:02:00Z", "o": 10, "h": 12, "l": 9, "c": 11, "v": 200},          # repeat: keep last
        {"t": "garbage", "c": 10}, {"t": "2026-10-07T15:03:00Z", "c": 0}, {"t": None, "c": 5}, "not a bar",
        {"t": "NaTZ", "c": 5}, {"t": "2026-10-07T15:04:00Z", "c": float("nan")},
    ]
    b = to_arrays(rows)
    assert len(b) == 3
    assert b.iso(0) == "2026-10-07T15:00:00.000Z" and b.iso(1) == "2026-10-07T15:01:00.000Z"
    assert (b.open[0], b.high[0], b.low[0], b.volume[0]) == (10, 10, 10, 0)  # missing -> the close / 0
    assert b.high[1] >= 10.2 and b.low[1] <= 10 and b.volume[1] == 0         # high/low made consistent
    assert b.close[2] == 11 and b.volume[2] == 200
    # `now`: only bars already finished (a bar starting 15:02 finishes at 15:03)
    assert len(to_arrays(rows, now=datetime(2026, 10, 7, 15, 2, 59, tzinfo=UTC))) == 2
    assert len(to_arrays(rows, now="2026-10-07T15:03:00Z")) == 3
    assert len(to_arrays([])) == 0 and len(to_arrays(None)) == 0
    zulu = to_arrays([{"t": "2026-10-07T15:00:00Z", "c": 1}, {"t": "NaTZ", "c": 2}])     # the fast path
    assert len(zulu) == 1 and zulu.iso(0) == "2026-10-07T15:00:00.000Z"
    assert to_arrays([{"t": 1791385200000, "c": 1}, {"t": 1791385260, "c": 1}]).t.tolist() == [1791385200, 1791385260]


# ---------------------------------------------------------------------------------------------- VWAP
def _session(day: date, start: tuple[int, int], closes, volumes=None) -> list[dict]:
    return path(closes, ny(day, *start), volumes=volumes, wick=0.0)


def test_vwap_restarts_at_the_new_york_open_in_winter_and_summer():
    friday = _session(FRIDAY, (9, 28), [100, 100, 101, 102, 103, 104], [10, 10, 100, 100, 100, 100])
    monday = _session(MONDAY, (9, 27), [90, 90, 90, 95, 96], [10, 10, 10, 100, 300])
    bars = to_arrays(friday + monday)
    vw = ind.vwap(bars)
    t = [b["t"] for b in friday + monday]
    assert t[2] == "2026-03-06T14:30:00Z" and t[9] == "2026-03-09T13:30:00Z"  # 9:30 New York, before / after DST
    tp = (bars.high + bars.low + bars.close) / 3
    assert vw[2] == pytest.approx(tp[2])            # Friday 9:30: starts afresh
    assert vw[1] == pytest.approx((tp[0] + tp[1]) / 2)  # pre-market has its own VWAP
    assert vw[6] == pytest.approx(tp[6])            # Monday 9:27 pre-market: nothing carried over from Friday
    assert vw[8] == pytest.approx(tp[6:9].mean())
    assert vw[9] == pytest.approx(tp[9])            # Monday 9:30 (13:30 UTC in summer time): starts afresh
    assert vw[10] == pytest.approx((tp[9] * 100 + tp[10] * 300) / 400)


def test_vwap_carries_on_after_hours_and_rolls_24h_for_crypto():
    regular = _session(MONDAY, (15, 58), [100, 102, 104, 106], [100, 100, 100, 100])  # 15:58 .. 16:01
    bars = to_arrays(regular)
    vw = ind.vwap(bars)
    assert vw[3] == pytest.approx(((bars.high + bars.low + bars.close) / 3).mean())  # after-hours: same VWAP
    # crypto: one bar an hour for 30 hours - the rolling VWAP only looks back 24 hours
    start = datetime(2026, 10, 7, 0, 0, tzinfo=UTC)
    crypto = to_arrays(path(list(range(1, 31)), start, step=timedelta(hours=1), wick=0.0))
    roll = ind.vwap(crypto, crypto=True, rolling_hours=24)
    tp = (crypto.high + crypto.low + crypto.close) / 3
    assert roll[29] == pytest.approx(tp[6:30].mean())
    daily = ind.vwap(crypto, crypto=True)        # without rolling: restarts at midnight UTC
    assert daily[24] == pytest.approx(tp[24]) and daily[23] == pytest.approx(tp[:24].mean())


# ---------------------------------------------------------------------------------------------- volume
def test_relative_volume_compares_with_the_same_time_yesterday():
    yesterday = _session(FRIDAY, (10, 0), [100] * 30, [1000] * 30)
    today = _session(MONDAY, (10, 0), [100] * 30, [1000] * 10 + [3000] * 20)
    rv = ind.relative_volume(to_arrays(yesterday + today))
    assert np.isnan(rv[:5]).all()                 # first day: too few bars before to judge
    assert rv[29] == pytest.approx(1.0)           # first day after 20 bars: vs the 20 bars before
    assert rv[30] == pytest.approx(1.0)           # second day 10:00 vs Friday around 10:00
    assert rv[45] == pytest.approx(3.0)


def test_day_relative_volume_is_today_so_far_vs_a_normal_day_by_this_time():
    yesterday = _session(FRIDAY, (9, 30), [100] * 30, [1000] * 30)
    today = _session(MONDAY, (9, 30), [100] * 30, [1000] * 10 + [3000] * 20)
    drv = ind.day_relative_volume(to_arrays(yesterday + today))
    assert np.isnan(drv[:30]).all()
    assert drv[39] == pytest.approx(1.0)
    assert drv[59] == pytest.approx(70000 / 30000)


def test_quiet_premarket_doesnt_make_the_open_look_busy():
    pre = _session(MONDAY, (9, 0), [100] * 30, [50] * 30)
    regular = _session(MONDAY, (9, 30), [100] * 10, [5000] * 10)
    rv = ind.relative_volume(to_arrays(pre + regular), n=20)
    assert np.isnan(rv[30:35]).all()              # no earlier regular-session bars to compare with yet
    assert rv[36] == pytest.approx(1.0)


def test_relative_volume_for_crypto_runs_across_midnight():
    start = datetime(2026, 10, 7, 23, 30, tzinfo=UTC)
    rv = ind.relative_volume(to_arrays(path([1.0] * 60, start, volumes=[10] * 40 + [30] * 20)), crypto=True)
    assert rv[35] == pytest.approx(1.0) and rv[40] == pytest.approx(3.0)   # 00:05 / 00:10 UTC: no restart


def test_relative_volume_on_daily_bars():
    rows = daily_path([100] * 30, date(2026, 10, 7))
    rows[-1]["v"] = 3e6
    assert ind.relative_volume(to_arrays(rows))[-1] == pytest.approx(3.0)


def test_day_relative_volume_ignores_a_half_covered_earlier_day():
    half = _session(FRIDAY, (13, 0), [100] * 30, [1000] * 30)      # data starts mid-afternoon
    today = _session(MONDAY, (9, 30), [100] * 30, [1000] * 30)
    assert np.isnan(ind.day_relative_volume(to_arrays(half + today))).all()


# ---------------------------------------------------------------------------------------------- resampling
def test_resample_lines_up_with_the_open_and_drops_unfinished_bars():
    rows = path([float(x) for x in range(1, 15)], ny(MONDAY, 9, 28), volumes=list(range(1, 15)), wick=0.0)
    b5 = ind.resample(to_arrays(rows), 5)          # 9:28..9:41
    assert [datetime.fromtimestamp(t, NY).strftime("%H:%M") for t in b5.t] == ["09:25", "09:30", "09:35"]
    # 9:30 bar = minutes 9:30..9:34 = closes 3..7
    assert (b5.open[1], b5.high[1], b5.low[1], b5.close[1], b5.volume[1]) == (2, 7, 2, 7, 3 + 4 + 5 + 6 + 7)
    assert b5.close[0] == 2 and b5.volume[0] == 3   # pre-market 9:25 bucket: 9:28 and 9:29
    # the 9:40 bucket has only 9:40-9:41 -> not finished, not made
    with_now = ind.resample(to_arrays(rows), 5, now=ny(MONDAY, 9, 40))
    assert len(with_now) == 3
    early = ind.resample(to_arrays(rows), 5, now=ny(MONDAY, 9, 39))
    assert len(early) == 2                          # at 9:39 the 9:35 bar isn't finished yet
    b15 = ind.resample(to_arrays(path([100.0] * 40, ny(MONDAY, 9, 30))), 15)
    assert [datetime.fromtimestamp(t, NY).strftime("%H:%M") for t in b15.t] == ["09:30", "09:45"]


def test_resample_same_new_york_times_in_winter_and_summer():
    for day in (FRIDAY, MONDAY):  # 9:31..9:42: a 9:30 bar missing its first minute is fine, 9:40 isn't done
        b5 = ind.resample(to_arrays(path([100.0] * 12, ny(day, 9, 31))), 5)
        assert [datetime.fromtimestamp(t, NY).strftime("%H:%M") for t in b5.t] == ["09:30", "09:35"]


def test_resample_crypto_lines_up_with_utc():
    start = datetime(2026, 10, 7, 23, 57, tzinfo=UTC)
    b5 = ind.resample(to_arrays(path([1.0] * 10, start)), 5, crypto=True)
    assert [b5.iso(k) for k in range(len(b5))] == ["2026-10-07T23:55:00.000Z", "2026-10-08T00:00:00.000Z"]


def test_resample_empty_and_concat():
    assert len(ind.resample(Bars.empty(), 5)) == 0
    b = to_arrays(path([1.0, 2.0], ny(MONDAY, 10)))
    assert len(Bars.concat(b, b)) == 4


# ---------------------------------------------------------------------------------------------- swing points
def test_pivots_need_bars_on_both_sides():
    h = [1, 2, 5, 2, 1, 1, 1]
    hi, lo = ind.pivots(h, [x - 0.5 for x in h], left=2, right=2)
    assert hi.tolist() == [False, False, True, False, False, False, False]
    hi, _ = ind.pivots(h[:4], [x - 0.5 for x in h[:4]], left=2, right=2)
    assert not hi.any()                              # one bar after the peak isn't enough yet
    _, lo = ind.pivots([5, 4, 1, 4, 5], [4, 3, 0, 3, 4], left=2, right=2)
    assert lo.tolist() == [False, False, True, False, False]


def test_day_parts_and_early_close():
    t = np.array([ny(date(2026, 11, 27), 12, 59).timestamp(), ny(date(2026, 11, 27), 13, 0).timestamp(),
                  ny(MONDAY, 9, 29).timestamp(), ny(MONDAY, 15, 59).timestamp(), ny(MONDAY, 16, 0).timestamp()],
                 dtype=np.int64)
    _, _, seg = ind.day_parts(t)
    assert seg.tolist() == [ind.REGULAR, ind.AFTER, ind.PRE, ind.REGULAR, ind.AFTER]  # day after Thanksgiving
    assert ind.trading_day(ny(MONDAY, 23, 30).timestamp()) == (MONDAY - date(1970, 1, 1)).days
    assert ind.trading_day(ny(MONDAY, 23, 30).timestamp(), crypto=True) == (MONDAY - date(1970, 1, 1)).days + 1
    # daily bars stamped at midnight New York time or midnight UTC both land on their own date
    stamps = np.array([ny(MONDAY, 0).timestamp(), datetime(2026, 3, 9, tzinfo=UTC).timestamp()], dtype=np.int64)
    assert ind.daily_days(stamps).tolist() == [(MONDAY - date(1970, 1, 1)).days] * 2
