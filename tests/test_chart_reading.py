"""Chart engine reading: the snapshot, score and summary on made-up charts, confirm() and chart_signal(), crypto,
missing data, JSON for the UI, no look-ahead, and speed."""

from __future__ import annotations

import json
import time as clock
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pytest

from newstrader.chart import ChartReading, ChartSignal, Pattern, chart_signal, confirm, read_chart
from newstrader.chart.reading import SIGNAL_MAX_CONFIDENCE
from tests.test_chart_indicators import daily_path, iso, ny, path, walk

YESTERDAY, TODAY = date(2026, 10, 6), date(2026, 10, 7)


# ---------------------------------------------------------------------------------------------- made-up charts
def legs(start: float, *moves: tuple[int, float]) -> list[float]:
    """Closes going from `start` through each (minutes, price) leg in a straight line."""
    out, price = [], start
    for minutes, target in moves:
        out += list(np.linspace(price, target, minutes + 1)[1:])
        price = target
    return out


def wobble(closes: list[float], size: float = 0.02) -> list[float]:
    return [c + (size if k % 2 else -size) for k, c in enumerate(closes)]


def daily_history(start: float, seed: int = 4) -> list[dict]:
    """260 days drifting from `start` to a close of 100 yesterday, with day-to-day noise."""
    rng = np.random.default_rng(seed)
    closes = np.linspace(start, 100, 260) + rng.normal(0, 1.0, 260) * np.r_[np.ones(259), 0]
    return daily_path(list(closes), TODAY, wick=1.0)


def chart(today_closes: list[float], today_volumes: list[float], daily_start: float = 70.0):
    """(minute bars, daily bars, now): a quiet day yesterday at 100, then today from 9:30."""
    yesterday = path(wobble([100.0] * 390, 0.03), ny(YESTERDAY, 9, 30), volumes=[1000] * 390, wick=0.03)
    today = path(today_closes, ny(TODAY, 9, 30), volumes=today_volumes, wick=0.03, first_open=today_closes[0])
    now = ny(TODAY, 9, 30) + timedelta(minutes=len(today_closes))
    return yesterday + today, daily_history(daily_start), now


def breakout_chart():
    """Up to 102, three trips between 101.2 and 102, then a push through 102 on 3x volume."""
    closes = wobble(legs(100, (90, 102), *([(15, 101.2), (15, 102)] * 3), (5, 102.7)))
    return chart(closes, [1000] * (len(closes) - 5) + [3000] * 5)


def breakdown_chart():
    closes = wobble(legs(100, (90, 98.8), *([(15, 99.6), (15, 98.8)] * 3), (5, 98.1)))
    return chart(closes, [1000] * (len(closes) - 5) + [3000] * 5, daily_start=130.0)


def stretched_chart():
    """A straight run from 100.5 to 106 in 25 minutes on 4x volume."""
    closes = wobble(legs(100, (120, 100.5), (25, 106)))
    return chart(closes, [1000] * (len(closes) - 25) + [4000] * 25)


def quiet_chart():
    rng = np.random.default_rng(3)
    return chart(list(100 + np.cumsum(rng.normal(0, 0.01, 200))), [1000] * 200, daily_start=100.0)


# ---------------------------------------------------------------------------------------------- reading
def test_breakout_reading():
    r = read_chart("xyz", *breakout_chart())
    assert r.symbol == "XYZ" and r.asset_class == "stock" and not r.stale
    assert r.price == pytest.approx(102.68, abs=0.01) and r.bar_at == "2026-10-07T16:34:00.000Z"
    assert r.prev_close == 100 and r.day_change_pct == pytest.approx(2.68, abs=0.01)
    assert r.gap_pct == pytest.approx(0, abs=0.05)
    assert r.trend == "up" and r.trend_daily == "up" and r.atr_basis == "daily"
    assert r.vwap_pct > 0 and 0 < r.vwap_atr < 1 and 2 < r.atr_pct < 4
    assert 50 < r.rsi < 80 and r.bar_rel_volume == pytest.approx(3.0)
    assert r.rel_volume_basis.startswith("today so far")
    assert r.summary == "Uptrend, above VWAP, breaking out on heavy volume"
    assert r.score >= 0.7
    breakout = next(p for p in r.patterns if p.name == "breakout")
    assert breakout.timeframe == "5m" and breakout.level == pytest.approx(102.05, abs=0.05)
    assert "3 times" in breakout.explain and "3.0x normal volume" in breakout.explain
    assert [p.strength for p in r.patterns] == sorted((p.strength for p in r.patterns), reverse=True)
    sup = [lv for lv in r.levels if lv.kind == "support"]
    assert sup and all(lv.price < r.price for lv in sup)
    assert r.bars == {"1m": 390 + 185, "5m": 78 + 37, "1d": 261}   # 260 finished days + today so far


def test_stretched_reading():
    r = read_chart("XYZ", *stretched_chart())
    assert r.rsi > 90
    assert r.summary.startswith("Stretched: RSI 9") and "ATR above VWAP" in r.summary
    assert any(p.name == "rsi_overbought" and p.kind == "caution" for p in r.patterns)


def test_breakdown_reading():
    r = read_chart("XYZ", *breakdown_chart())
    assert r.trend == "down" and r.trend_daily == "down" and r.score <= -0.7
    assert r.summary == "Downtrend, below VWAP, breaking down on heavy volume"
    assert any(p.name == "breakdown" and p.direction == "bearish" for p in r.patterns)


# ---------------------------------------------------------------------------------------------- confirm()
def test_confirm_agrees_with_the_breakout():
    r = read_chart("XYZ", *breakout_chart())
    verdict, adjust, reason = confirm("bullish", r)
    assert verdict == "agrees" and 2 <= adjust <= 10 and reason.startswith("Chart agrees: uptrend")
    verdict, adjust, reason = confirm("bearish", r)
    assert verdict == "against" and -15 <= adjust <= -5 and "Broke above resistance" in reason


def test_confirm_says_stretched_when_the_move_already_happened():
    r = read_chart("XYZ", *stretched_chart())
    verdict, adjust, reason = confirm("bullish", r)
    assert verdict == "stretched" and adjust == -15            # RSI over 90: very stretched
    assert "RSI 9" in reason and "chasing" in reason
    assert confirm("bearish", r)[0] != "stretched"


def test_confirm_against_buying_into_a_breakdown():
    r = read_chart("XYZ", *breakdown_chart())
    verdict, adjust, reason = confirm("bullish", r)
    assert verdict == "against" and adjust == -15 and "Broke below support" in reason
    assert confirm("bearish", r)[0] == "agrees"


def test_confirm_quiet_chart_is_mild():
    r = read_chart("XYZ", *quiet_chart())
    for direction in ("bullish", "bearish"):
        verdict, adjust, _ = confirm(direction, r)
        assert -8 <= adjust <= 4


def _reading(**kw) -> ChartReading:
    base = dict(symbol="XYZ", price=100.0, stale=False, vwap=99.5, vwap_pct=0.5, vwap_atr=0.25, rsi=55.0,
                rsi_daily=55.0, macd_hist=0.1, rel_volume=2.0, bar_rel_volume=2.0, trend="up", trend_daily="up",
                score=0.5, summary="Uptrend, above VWAP", at="2026-10-07T15:00:00.000Z")
    return ChartReading(**(base | kw))


def _p(name: str, direction: str = "bullish", kind: str = "structure", strength: float = 0.6,
       timeframe: str = "5m") -> Pattern:
    return Pattern(name, direction, strength, "2026-10-07T14:55:00.000Z", 100.0, f"{name} explained", kind, timeframe)


@pytest.mark.parametrize("kw, verdict, adjust", [
    (dict(rsi=82.0), "stretched", -10),
    (dict(rsi=93.0), "stretched", -15),
    (dict(vwap_atr=2.2), "stretched", -10),
    (dict(vwap_atr=3.5), "stretched", -15),
    (dict(rsi=84.0, vwap_atr=2.1), "stretched", -15),
    (dict(rsi_daily=85.0), "stretched", -10),
    (dict(score=0.85), "agrees", 10),
    (dict(score=0.3), "agrees", 2),
    (dict(score=0.1), "neutral", 0),
    (dict(score=-0.35), "against", -5),
    (dict(score=-0.9), "against", -15),
])
def test_confirm_thresholds_bullish(kw, verdict, adjust):
    got = confirm("bullish", _reading(**kw))
    assert got[:2] == (verdict, adjust)
    assert got[2]


def test_confirm_mirrors_for_bearish_and_handles_missing_data():
    assert confirm("bearish", _reading(rsi=15.0))[0] == "stretched"
    assert confirm("bearish", _reading(vwap_atr=-2.5))[0] == "stretched"
    assert confirm("bearish", _reading(vwap_atr=2.5))[0] != "stretched"
    assert confirm("bearish", _reading(score=-0.6))[0] == "agrees"
    # an opposite breakdown on the chart makes it "against" even when the score is only mildly negative
    assert confirm("bullish", _reading(score=0.0, patterns=[_p("breakdown", "bearish")]))[0] == "against"
    assert confirm("bullish", None) == ("neutral", 0, "No chart data to check the signal against.")
    assert confirm("neutral", _reading())[:2] == ("neutral", 0)
    assert confirm("bullish", ChartReading(symbol="XYZ"))[:2] == ("neutral", 0)


# ---------------------------------------------------------------------------------------------- chart_signal()
def test_chart_signal_from_the_breakout():
    sig = chart_signal(read_chart("XYZ", *breakout_chart()))
    assert isinstance(sig, ChartSignal) and sig.direction == "bullish"
    assert sig.confidence <= SIGNAL_MAX_CONFIDENCE == 75 and "breakout" in sig.patterns
    assert sig.reason.startswith("Broke above resistance") and "heavy volume" in sig.reason
    assert json.loads(json.dumps(sig.as_dict()))["direction"] == "bullish"
    bear = chart_signal(read_chart("XYZ", *breakdown_chart()))
    assert bear is not None and bear.direction == "bearish" and "breakdown" in bear.patterns


def test_chart_signal_stays_quiet_when_it_should():
    assert chart_signal(read_chart("XYZ", *stretched_chart())) is None     # don't chase
    assert chart_signal(read_chart("XYZ", *quiet_chart())) is None
    assert chart_signal(None) is None and chart_signal(ChartReading(symbol="XYZ")) is None


def test_chart_signal_needs_more_than_one_candle():
    candle_only = _reading(patterns=[_p("hammer", kind="candle", strength=0.9), _p("bullish_engulfing", kind="candle")],
                           score=0.8)
    assert chart_signal(candle_only) is None
    # a trigger, but nothing else agrees (no volume, no trend, below VWAP, momentum down)
    alone = _reading(patterns=[_p("breakout")], rel_volume=1.0, bar_rel_volume=1.0, trend="sideways",
                     trend_daily="sideways", vwap_pct=-0.2, macd_hist=-0.1, score=0.3)
    assert chart_signal(alone) is None
    # trigger + heavy volume + one more piece: enough
    two_more = _reading(patterns=[_p("breakout")], trend="sideways", trend_daily="sideways", vwap_pct=0.2,
                        macd_hist=-0.1, score=0.3)
    sig = chart_signal(two_more)
    assert sig is not None and len(sig.evidence) == 3
    # ...but not without the volume
    assert chart_signal(_reading(patterns=[_p("breakout")], rel_volume=1.2, bar_rel_volume=1.2, score=0.6)) is None


def test_chart_signal_vetoes_and_cap():
    strong = dict(patterns=[_p("breakout", strength=1.0), _p("bull_flag", strength=1.0),
                            _p("hammer", kind="candle")], score=0.9)
    sig = chart_signal(_reading(**strong))
    assert sig is not None and sig.confidence == 75
    assert chart_signal(_reading(**strong | dict(rsi=85.0))) is None                  # stretched
    assert chart_signal(_reading(**strong | dict(score=0.1))) is None                 # chart doesn't lean that way
    assert chart_signal(_reading(**strong | dict(stale=True))) is None                # old data
    mixed = strong | dict(patterns=strong["patterns"] + [_p("vwap_lost", "bearish", "indicator")])
    assert chart_signal(_reading(**mixed)) is None                                   # a trigger the other way
    daily_only = dict(patterns=[_p("breakout", timeframe="1d")], score=0.9)
    assert chart_signal(_reading(**daily_only)) is None                              # daily patterns are context


# ---------------------------------------------------------------------------------------------- data edge cases
def test_no_data_at_all():
    r = read_chart("XYZ", [], [], now=ny(TODAY, 10))
    assert r.price is None and r.summary == "No chart data yet" and r.score == 0 and r.patterns == []
    assert json.loads(json.dumps(r.as_dict()))["price"] is None
    assert read_chart("XYZ", None, None).summary == "No chart data yet"


def test_only_daily_bars():
    r = read_chart("XYZ", [], daily_history(70), now=ny(TODAY, 10))
    assert r.price == 100 and r.trend_daily == "up" and r.stale and r.vwap is None and r.rsi_daily is not None
    assert r.trend == "unknown"


def test_only_a_few_minutes_of_data():
    bars = path([100.0, 100.1, 100.2], ny(TODAY, 9, 30), wick=0.05)
    r = read_chart("XYZ", bars, [], now=ny(TODAY, 9, 33))
    assert r.price == 100.2 and r.rsi is None and r.trend == "unknown" and r.atr_basis == "estimated"
    assert r.summary and json.dumps(r.as_dict())
    assert confirm("bullish", r)[0] in ("neutral", "agrees", "against")


def test_ignores_bars_after_now_and_todays_daily_bar():
    minutes, daily, now = breakout_chart()
    before = read_chart("XYZ", minutes, daily, now).as_dict()
    future_minutes = path(legs(102.7, (60, 90.0)), now, volumes=[50000] * 60, wick=0.5)
    future_daily = [{"t": iso(ny(TODAY, 0)), "o": 100, "h": 150, "l": 50, "c": 60, "v": 9e9},
                    {"t": iso(ny(TODAY + timedelta(days=1), 0)), "o": 60, "h": 61, "l": 59, "c": 60, "v": 9e9}]
    after = read_chart("XYZ", minutes + future_minutes, daily + future_daily, now).as_dict()
    assert after == before


@pytest.mark.parametrize("minutes_in", [45, 120, 185])
def test_reading_as_of_earlier_time_matches_truncated_data(minutes_in):
    minutes, daily, _ = breakout_chart()
    now = ny(TODAY, 9, 30) + timedelta(minutes=minutes_in)
    cut = [b for b in minutes if datetime.fromisoformat(b["t"].replace("Z", "+00:00")) + timedelta(minutes=1) <= now]
    assert read_chart("XYZ", minutes, daily, now).as_dict() == read_chart("XYZ", cut, daily, now).as_dict()


def test_random_charts_read_the_same_with_or_without_later_bars():
    rng = np.random.default_rng(21)
    minutes = []
    for k, day in enumerate((YESTERDAY, TODAY)):
        vols = rng.integers(100, 3000, 960).astype(float)
        vols[rng.integers(0, 960, 20)] *= 8
        minutes += path(list(walk(960, 30 + k, 100, 0.0015)), ny(day, 4, 0), volumes=list(vols), wick=0.03)
    daily = daily_path(list(walk(260, 40, 100, 0.02)), TODAY)
    seen = 0
    for hh, mm in ((9, 31), (9, 47), (10, 52), (13, 3), (15, 59), (17, 30)):
        now = ny(TODAY, hh, mm)
        cut = [b for b in minutes
               if datetime.fromisoformat(b["t"].replace("Z", "+00:00")) + timedelta(minutes=1) <= now]
        full = read_chart("XYZ", minutes, daily, now).as_dict()
        assert full == read_chart("XYZ", cut, daily, now).as_dict()
        seen += len(full["patterns"])
    assert seen > 10


def test_stale_when_the_last_bar_is_old():
    minutes, daily, now = quiet_chart()
    assert read_chart("XYZ", minutes, daily, now + timedelta(hours=2)).stale
    assert not read_chart("XYZ", minutes, daily, now).stale
    assert not read_chart("XYZ", minutes, daily).stale     # default "now": just after the last bar


def test_crypto_runs_around_the_clock():
    start = datetime(2026, 10, 6, 0, 0, tzinfo=UTC)
    rng = np.random.default_rng(11)
    minutes = path(list(walk(2880, 5, 60000, 0.0008)), start, volumes=list(rng.integers(1, 20, 2880)), wick=5)
    daily_rows = path(list(walk(365, 6, 60000, 0.02)), start - timedelta(days=365), step=timedelta(days=1), wick=500)
    r = read_chart("BTC/USD", minutes, daily_rows, now=start + timedelta(days=2), asset_class="crypto")
    assert r.asset_class == "crypto" and r.symbol == "BTC/USD" and not r.stale
    assert r.gap_pct is None and r.vwap is not None and r.rsi is not None and r.atr_basis == "daily"
    names = {p.name for p in r.patterns}
    assert not names & {"gap_up", "gap_down", "gap_fill", "opening_range_breakout", "opening_range_breakdown"}
    assert r.bars["1m"] == 2880 and r.bars["1d"] == 365          # today (UTC) has no bars yet -> nothing added
    json.dumps(r.as_dict())


def test_as_dict_is_plain_json():
    r = read_chart("XYZ", *breakout_chart())
    d = json.loads(json.dumps(r.as_dict()))
    assert set(d) >= {"symbol", "price", "rsi", "macd_hist", "vwap_pct", "atr_pct", "rel_volume", "trend",
                      "patterns", "levels", "score", "summary"}
    assert isinstance(d["stale"], bool) and isinstance(d["patterns"][0]["strength"], float)
    assert {"name", "direction", "strength", "at", "level", "explain", "kind", "timeframe"} == set(d["patterns"][0])


# ---------------------------------------------------------------------------------------------- speed
def test_fast_enough_for_two_days_of_minutes_and_a_year_of_days():
    minutes = []
    for k, day in enumerate((YESTERDAY, TODAY)):
        rng = np.random.default_rng(k)
        minutes += path(list(walk(960, k, 100, 0.001)), ny(day, 4, 0), volumes=list(rng.integers(100, 5000, 960)))
    daily = daily_path(list(walk(252, 9, 100, 0.015)), TODAY)
    now = ny(TODAY, 20, 0)
    read_chart("XYZ", minutes, daily, now)                    # warm-up (imports, caches)
    best = float("inf")
    for _ in range(5):
        t = clock.perf_counter()
        r = read_chart("XYZ", minutes, daily, now)
        best = min(best, clock.perf_counter() - t)
    assert r.bars["1m"] == 1920 and r.bars["1d"] == 253
    assert best < 0.05, f"read_chart took {best * 1000:.1f} ms"
