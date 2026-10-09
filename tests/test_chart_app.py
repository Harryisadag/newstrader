"""Charts in the app: the chart check on news signals (stretched -> review, the boost cap, sells never blocked,
off / soft / strict), chart signals (cooldown, storage, never traded, scoreboard), the ChartService's bar caching and
reuse of the market monitor's bars, the request pacer, migration 12 and the API. Never talks to Alpaca."""

from __future__ import annotations

import asyncio
import sqlite3
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from newstrader.ai.pipeline import Pipeline
from newstrader.chart import read_chart
from newstrader.chart.service import (
    COOLDOWN,
    NO_CHART,
    ChartService,
    Pacer,
    gate,
    history_start,
    worth_checking,
)
from newstrader.db import MIGRATIONS, Database, iso
from newstrader.performance.stats import compute_stats
from newstrader.performance.tracker import PerformanceTracker
from newstrader.trading.fake_broker import FakeBroker
from newstrader.trading.trader import Trader
from tests.helpers import make_tickers
from tests.test_chart_indicators import ny
from tests.test_chart_reading import TODAY, YESTERDAY, _reading, breakout_chart, quiet_chart, stretched_chart


class FakeAlerts:
    def __init__(self):
        self.sent: list[tuple[str, str]] = []

    async def send(self, kind, title, message="", level="info", force=False, fields=None):
        self.sent.append((kind, title))
        return {"toast": True}


def orders(broker) -> list[tuple]:
    return [c for c in broker.calls if c[0] == "submit_bracket"]


# ================================================================================ gate(): the rules
def test_stretched_goes_to_review_without_touching_the_confidence():
    c = gate("bullish", 90, "stretched", -15, "Chart says the move may already be done: RSI 93.", None, "soft", 80)
    assert c.review.startswith("Chart says the move may already be done") and "manual review" in c.review
    assert c.adjust == 0 and c.confidence == 90
    assert gate("bullish", 90, "stretched", -15, "x", None, "strict", 80).review


def test_against_lowers_in_soft_and_reviews_in_strict():
    soft = gate("bullish", 85, "against", -15, "Chart disagrees: y.", -0.9, "soft", 80)
    assert soft.adjust == -15 and soft.confidence == 70 and not soft.review
    strict = gate("bullish", 85, "against", -15, "Chart disagrees: y.", -0.9, "strict", 80)
    assert strict.adjust == 0 and strict.confidence == 85 and "strict" in strict.review
    assert gate("bullish", 4, "against", -15, "z", -0.9, "soft", 80).confidence == 0  # never below 0


@pytest.mark.parametrize("conf, adjust, expected", [
    (72, 10, 79),    # a boost never lifts a review-band signal to the buy threshold...
    (78, 2, 79),
    (79, 5, 79),     # ...not even by one point
    (60, 4, 64),     # a small boost below the cap is applied in full
    (55, 10, 65),    # (from "ignore" into the review band is fine - that's still a person deciding)
    (80, 5, 85),     # already an auto-trade: the boost applies
    (97, 10, 100),
])
def test_agrees_boost_is_capped_below_the_buy_threshold(conf, adjust, expected):
    c = gate("bullish", conf, "agrees", adjust, "Chart agrees: uptrend.", 0.8, "soft", 80)
    assert c.confidence == expected and c.adjust == expected - conf and not c.review


def test_closing_sells_are_never_held_back():
    for verdict, adjust in (("stretched", -15), ("against", -15)):
        for mode in ("soft", "strict"):
            c = gate("bearish", 85, verdict, adjust, "Chart says no.", 0.9, mode, 80, closing=True)
            assert not c.review and c.adjust == 0 and c.confidence == 85 and "never held back" in c.reason


def test_off_neutral_and_unavailable_change_nothing():
    assert gate("bullish", 90, "stretched", -15, "x", None, "off", 80).review == ""
    for verdict in ("neutral", "unavailable"):
        c = gate("bullish", 90, verdict, 0, "y", None, "strict", 80)
        assert not c.review and c.adjust == 0 and c.confidence == 90


def test_worth_checking_only_when_the_chart_could_change_something(ctx):
    t = ctx.config.settings.trading                       # review threshold 60, shorting off
    assert worth_checking("bullish", 85, False, t) and worth_checking("bullish", 50, False, t)
    assert not worth_checking("bullish", 49, False, t)    # even +10 can't reach manual review
    assert not worth_checking("neutral", 90, False, t)
    assert not worth_checking("bearish", 90, False, t)    # don't hold it, can't short it: ignored anyway
    assert not worth_checking("bearish", 90, True, t)     # a sell of a stock you hold: never held back, so never checked
    ctx.config.update({"trading": {"allow_shorting": True}})
    assert worth_checking("bearish", 90, False, ctx.config.settings.trading)
    assert not worth_checking("bearish", 90, True, ctx.config.settings.trading)


# ================================================================================ the check in the trader
NOW = datetime(2026, 10, 7, 15, 0, tzinfo=UTC)


@pytest.fixture
async def env(ctx, monkeypatch):
    monkeypatch.setattr("newstrader.chart.service.utcnow", lambda: NOW)
    broker = FakeBroker(prices={"AAPL": 100.0, "XYZ": 102.7})
    trader = Trader(ctx, broker_factory=lambda: broker)
    ctx.services["trader"] = trader
    await trader.connect()
    charts = ChartService(ctx)
    ctx.services["chart"] = charts
    alerts = FakeAlerts()
    ctx.services["alerts"] = alerts
    yield SimpleNamespace(ctx=ctx, trader=trader, broker=broker, charts=charts, alerts=alerts)
    await trader.stop()


def chart_says(env, symbol="AAPL", **kw):
    """Make the chart service's (cached) reading for `symbol` look like this."""
    env.charts._readings[symbol] = (NOW, _reading(symbol=symbol, **kw))


def news_signal(ctx, ticker="AAPL", direction="bullish", confidence=90) -> dict:
    sig = {"created_at": iso(NOW), "ticker": ticker, "direction": direction, "confidence": confidence,
           "reasoning": "Beat estimates.", "headline": "Apple beats", "engine": "local", "source_name": "Reuters"}
    sig["id"] = ctx.db.insert("signals", sig)
    return sig


def saved(ctx, sig) -> dict:
    return ctx.db.query_one("SELECT * FROM signals WHERE id = ?", (sig["id"],))


async def test_stretched_chart_sends_a_buy_to_review(env):
    chart_says(env, rsi=93.0)
    sig = news_signal(env.ctx)
    res = await env.trader.handle_signal(sig)
    assert res["action"] == "review" and not res["traded"] and not orders(env.broker)
    assert "RSI 93" in res["reason"] and "chasing" in res["reason"]
    row = saved(env.ctx, sig)
    assert row["chart_verdict"] == "stretched" and row["chart_adjust"] == 0 and row["chart_score"] == 0.5
    assert "RSI 93" in row["chart_reason"]
    # approving it by hand skips the chart check (like "don't chase")
    res = await env.trader.handle_signal(sig, manual=True)
    assert res["action"] == "bought" and res["traded"]


async def test_agreeing_chart_boosts_but_never_into_an_auto_trade(env):
    chart_says(env, score=0.85)                       # confirm(): agrees, +10
    sig = news_signal(env.ctx, confidence=75)          # review band (60-79)
    res = await env.trader.handle_signal(sig)
    assert res["action"] == "review" and not orders(env.broker)
    assert "(75 -> 79)" in res["reason"] and "Chart agrees" in res["reason"]
    row = saved(env.ctx, sig)
    assert row["chart_verdict"] == "agrees" and row["chart_adjust"] == 4 and row["confidence"] == 75
    sig = news_signal(env.ctx, confidence=82)          # already a buy: the full boost
    res = await env.trader.handle_signal(sig)
    assert res["action"] == "bought" and saved(env.ctx, sig)["chart_adjust"] == 10


async def test_disagreeing_chart_soft_lowers_strict_reviews(env):
    chart_says(env, score=-0.9)                       # confirm(): against, -15
    sig = news_signal(env.ctx, confidence=90)
    res = await env.trader.handle_signal(sig)
    assert res["action"] == "review" and "(90 -> 75)" in res["reason"] and "Chart disagrees" in res["reason"]
    assert saved(env.ctx, sig)["chart_adjust"] == -15
    sig = news_signal(env.ctx, confidence=99)          # still a buy after -15
    assert (await env.trader.handle_signal(sig))["action"] == "bought"

    env.ctx.config.update({"chart": {"confirm": "strict"}, "risk": {"ticker_cooldown_minutes": 0}})
    env.charts._readings.clear()
    chart_says(env, score=-0.9)
    sig = news_signal(env.ctx, confidence=99)
    res = await env.trader.handle_signal(sig)
    assert res["action"] == "review" and "strict" in res["reason"]
    assert saved(env.ctx, sig)["chart_adjust"] == 0
    assert len(orders(env.broker)) == 1


async def test_selling_a_stock_you_hold_is_never_blocked_by_the_chart(env):
    env.ctx.config.update({"risk": {"ticker_cooldown_minutes": 0}})
    assert (await env.trader.handle_signal(news_signal(env.ctx), manual=True))["action"] == "bought"
    for mode, reading in (("soft", {"rsi": 10.0}), ("strict", {"score": 0.9})):  # stretched down / chart says up
        env.ctx.config.update({"chart": {"confirm": mode}})
        env.charts._readings.clear()
        chart_says(env, **reading)
        sig = news_signal(env.ctx, direction="bearish", confidence=85)
        res = await env.trader.handle_signal(sig)
        assert res["action"] == "sold" and res["traded"], res
        assert saved(env.ctx, sig)["chart_verdict"] is None  # not even checked: a sell never waits for the chart
        if mode == "soft":
            assert (await env.trader.handle_signal(news_signal(env.ctx), manual=True))["traded"]  # buy it back


async def test_off_and_missing_chart_change_nothing(env):
    env.ctx.config.update({"chart": {"confirm": "off"}, "risk": {"ticker_cooldown_minutes": 0}})
    chart_says(env, rsi=95.0)
    sig = news_signal(env.ctx)
    assert (await env.trader.handle_signal(sig))["action"] == "bought"
    assert saved(env.ctx, sig)["chart_verdict"] is None
    env.ctx.config.update({"chart": {"confirm": "soft"}})
    chart_says(env, stale=True, rsi=95.0)             # no up-to-date chart: not checked
    sig = news_signal(env.ctx)
    assert (await env.trader.handle_signal(sig))["action"] == "bought"
    row = saved(env.ctx, sig)
    assert row["chart_verdict"] == "unavailable" and row["chart_reason"] == NO_CHART and row["chart_adjust"] == 0


async def test_chart_failure_or_no_service_never_stops_a_trade(env, monkeypatch):
    async def broken(*_a, **_k):
        raise RuntimeError("data down")

    monkeypatch.setattr(env.charts, "reading", broken)
    sig = news_signal(env.ctx)
    assert (await env.trader.handle_signal(sig))["action"] == "bought"
    assert saved(env.ctx, sig)["chart_verdict"] == "unavailable"
    env.ctx.services.pop("chart")
    env.ctx.config.update({"risk": {"ticker_cooldown_minutes": 0}})
    assert (await env.trader.handle_signal(news_signal(env.ctx)))["action"] == "bought"


async def test_a_slow_chart_is_skipped(env, monkeypatch):
    monkeypatch.setattr("newstrader.chart.service.CHECK_TIMEOUT", 0.05)

    async def slow(*_a, **_k):
        await asyncio.sleep(1)

    monkeypatch.setattr(env.charts, "reading", slow)
    sig = news_signal(env.ctx)
    assert (await env.trader.handle_signal(sig))["action"] == "bought"
    assert saved(env.ctx, sig)["chart_verdict"] == "unavailable"


async def test_selling_a_stock_you_hold_never_waits_for_the_chart(env, monkeypatch):
    monkeypatch.setattr("newstrader.chart.service.CHECK_TIMEOUT", 2.0)
    assert (await env.trader.handle_signal(news_signal(env.ctx), manual=True))["action"] == "bought"
    asked = []

    async def slow(*_a, **_k):
        asked.append(1)
        await asyncio.sleep(5)

    monkeypatch.setattr(env.charts, "reading", slow)
    sig = news_signal(env.ctx, direction="bearish", confidence=90)
    started = time.monotonic()
    res = await env.trader.handle_signal(sig)
    assert res["action"] == "sold" and time.monotonic() - started < 1 and not asked
    assert saved(env.ctx, sig)["chart_verdict"] is None


async def test_a_slow_chart_never_holds_up_other_trades_or_the_kill_switch(env, monkeypatch):
    monkeypatch.setattr("newstrader.chart.service.CHECK_TIMEOUT", 1.5)
    env.broker.prices["MSFT"] = 500.0

    async def slow(*_a, **_k):
        await asyncio.sleep(5)

    monkeypatch.setattr(env.charts, "reading", slow)
    started = time.monotonic()
    results = await asyncio.gather(*(env.trader.handle_signal(news_signal(env.ctx, t)) for t in ("AAPL", "XYZ")))
    assert [r["action"] for r in results] == ["bought", "bought"]
    assert time.monotonic() - started < 2.5  # the two chart waits overlap: no queueing behind each other
    waiting = [asyncio.create_task(env.trader.handle_signal(news_signal(env.ctx, "MSFT"))) for _ in range(3)]
    await asyncio.sleep(0.1)  # all three are waiting for their chart
    started = time.monotonic()
    out = await env.trader.kill(close_positions=True)
    assert time.monotonic() - started < 1 and out["closed"] == 2 and env.broker.positions() == []
    assert [(await t)["action"] for t in waiting] == ["blocked"] * 3
    assert len(orders(env.broker)) == 2


@pytest.mark.parametrize("make, action, verdict", [(breakout_chart, "bought", "agrees"),
                                                    (stretched_chart, "review", "stretched")])
async def test_real_charts_end_to_end(ctx, monkeypatch, make, action, verdict):
    """Real bars -> the chart service -> confirm() -> the trader, downloading each chart once."""
    minutes, daily, now = make()
    monkeypatch.setattr("newstrader.chart.service.utcnow", lambda: now)
    broker = FakeBroker(prices={"XYZ": 103.0})
    broker.bar_data["XYZ"], broker.daily_data["XYZ"] = minutes, daily
    trader = Trader(ctx, broker_factory=lambda: broker)
    ctx.services.update({"trader": trader, "chart": ChartService(ctx)})
    await trader.connect()
    try:
        sig = news_signal(ctx, "XYZ", confidence=85)
        res = await trader.handle_signal(sig)
        assert res["action"] == action, res
        assert saved(ctx, sig)["chart_verdict"] == verdict
        assert sorted(c[0] for c in broker.bar_calls) == ["1Day", "1Min"]   # prefetch and check share one download
        nothing = news_signal(ctx, "AAPL", direction="bearish", confidence=90)  # not held, no shorting
        assert (await trader.handle_signal(nothing))["action"] == "ignored"
        assert saved(ctx, nothing)["chart_verdict"] is None and len(broker.bar_calls) == 2
    finally:
        await trader.stop()


async def test_chart_signals_never_reach_the_trader_automatically(env):
    sig = news_signal(env.ctx)
    sig["engine"] = "chart"
    res = await env.trader.handle_signal(sig)
    assert res["action"] == "review" and not orders(env.broker)


# ================================================================================ chart signals
@pytest.fixture
async def scan_env(ctx):
    ctx.loop = asyncio.get_running_loop()
    ctx.bus.bind_loop(ctx.loop)
    minutes, daily, now = breakout_chart()
    broker = FakeBroker(prices={"XYZ": 102.7})
    broker.bar_data["XYZ"], broker.daily_data["XYZ"] = minutes, daily
    trader = SimpleNamespace(broker=broker, clock={"is_open": True}, positions=[], open_orders=[], account={})
    monitor = SimpleNamespace(watched_stocks=lambda: ["XYZ"], fresh_bars=lambda *a, **k: None)
    alerts = FakeAlerts()
    ctx.services.update({"trader": trader, "market": monitor, "alerts": alerts})
    charts = ChartService(ctx)
    ctx.services["chart"] = charts
    return SimpleNamespace(ctx=ctx, charts=charts, broker=broker, trader=trader, alerts=alerts, now=now,
                           minutes=minutes, daily=daily)


def chart_rows(ctx) -> list[dict]:
    return ctx.db.query("SELECT * FROM signals WHERE engine = 'chart' ORDER BY id")


async def test_breakout_becomes_a_watch_only_chart_signal(scan_env):
    q = scan_env.ctx.bus.subscribe()
    res = await scan_env.charts.scan(scan_env.now)
    assert res["status"] == "ok" and res["read"] == 1 and len(res["signals"]) == 1
    [row] = chart_rows(scan_env.ctx)
    assert row["action"] == "watch" and row["traded"] == 0 and row["review_status"] is None
    assert row["source_type"] == "chart" and row["source_name"] == "Chart patterns" and row["engine"] == "chart"
    assert row["direction"] == "bullish" and row["confidence"] <= 75 and row["event"] == "Chart: Breakout"
    assert row["headline"].startswith("Broke above resistance") and row["headline"] == row["reasoning"]
    assert row["chart_reason"] == "Uptrend, above VWAP, breaking out on heavy volume"
    assert "never traded" in row["action_reason"]
    assert not scan_env.alerts.sent                     # watch-only: not alerted
    assert not orders(scan_env.broker)
    events = []
    while not q.empty():
        events.append(q.get_nowait())
    assert any(e["type"] == "signal" and e["data"]["id"] == row["id"] for e in events)
    status = {c["component"]: c for c in scan_env.ctx.state.components()}["chart"]
    assert status["level"] == "ok" and "1 new signal" in status["detail"]
    # it can't be approved into a trade
    pipeline = Pipeline(scan_env.ctx, tickers=make_tickers(scan_env.ctx.db))
    with pytest.raises(PermissionError):
        await pipeline.approve(row["id"])


async def test_cooldown_one_signal_per_stock_and_direction(scan_env):
    await scan_env.charts.scan(scan_env.now)
    assert (await scan_env.charts.scan(scan_env.now + timedelta(seconds=30)))["status"] == "too_soon"
    res = await scan_env.charts.scan(scan_env.now + timedelta(minutes=1), force=True)
    assert res["status"] == "ok" and res["signals"] == []    # same breakout, inside the 30-minute cooldown
    assert len(chart_rows(scan_env.ctx)) == 1
    # the cooldown is read from the database, so it survives a restart - and it ends after 30 minutes
    scan_env.ctx.db.execute("UPDATE signals SET created_at = ?", (iso(scan_env.now - COOLDOWN - timedelta(minutes=1)),))
    fresh = ChartService(scan_env.ctx)
    res = await fresh.scan(scan_env.now + timedelta(minutes=1))
    assert len(res["signals"]) == 1 and len(chart_rows(scan_env.ctx)) == 2


async def test_review_mode_asks_you_and_only_trades_when_approved(scan_env, ctx):
    ctx.config.update({"chart": {"signals": "review"}})
    broker = FakeBroker(prices={"XYZ": 102.7})
    broker.bar_data["XYZ"], broker.daily_data["XYZ"] = scan_env.minutes, scan_env.daily
    trader = Trader(ctx, broker_factory=lambda: broker)
    ctx.services["trader"] = trader
    await trader.connect()
    try:
        ctx.config.update({"risk": {"market_hours_only": False}})
        trader.clock = {"is_open": True}
        trader.refresh = _no_refresh(trader.refresh)
        res = await scan_env.charts.scan(scan_env.now)
        [row] = chart_rows(ctx)
        assert res["signals"] == [row["id"]] and row["action"] == "review" and row["review_status"] == "pending"
        assert scan_env.alerts.sent and scan_env.alerts.sent[0][0] == "manual_review"
        assert not orders(broker)                       # never traded by itself
        pipeline = Pipeline(ctx, tickers=make_tickers(ctx.db))
        out = await pipeline.approve(row["id"])         # you approve it: then it is traded (risk checks apply)
        assert out["traded"] and len(orders(broker)) == 1
    finally:
        await trader.stop()


def _no_refresh(real):
    """The fake account's clock says closed outside a test's made-up market hours - keep the clock we set."""
    async def refresh(force=False):
        clock = real.__self__.clock
        await real(force)
        real.__self__.clock = clock
    return refresh


async def test_no_chart_signals_when_off_closed_or_nothing_watched(scan_env):
    scan_env.ctx.config.update({"chart": {"signals": "off"}})
    assert (await scan_env.charts.scan(scan_env.now))["status"] == "off"
    scan_env.ctx.config.update({"chart": {"signals": "watch"}})
    scan_env.trader.clock = {"is_open": False}
    assert (await scan_env.charts.scan(scan_env.now))["status"] == "closed"
    scan_env.trader.clock = {"is_open": True}
    scan_env.ctx.services["market"] = SimpleNamespace(watched_stocks=lambda: [])
    assert (await scan_env.charts.scan(scan_env.now))["status"] == "nothing"
    assert not chart_rows(scan_env.ctx) and not scan_env.broker.bar_calls


async def test_quiet_or_stretched_charts_make_no_signal(scan_env):
    for make in (quiet_chart, stretched_chart):
        minutes, daily, now = make()
        scan_env.broker.bar_data["XYZ"], scan_env.broker.daily_data["XYZ"] = minutes, daily
        charts = ChartService(scan_env.ctx)
        assert (await charts.scan(now))["signals"] == []
    assert not chart_rows(scan_env.ctx)


async def test_chart_signals_are_price_tracked_and_scored_on_their_own(scan_env):
    await scan_env.charts.scan(scan_env.now)
    [row] = chart_rows(scan_env.ctx)
    assert row["price_at_signal"] == pytest.approx(102.68, abs=0.01)
    await PerformanceTracker(scan_env.ctx).run_once()
    assert scan_env.ctx.db.query_one("SELECT * FROM signal_prices WHERE signal_id = ?", (row["id"],)) is not None

    def r(engine, ret, action, event):
        return {"direction": "bullish", "confidence": 70, "ret_1h": ret, "source_name": "x", "source_type": "x",
                "traded": 0, "engine": engine, "event": event, "action": action}

    rows = [r("local", 1.0, "bought", "Earnings beat"), r("chart", -1.0, "watch", "Chart: Breakout"),
            r("chart", 2.0, "review", "Chart: Breakout"), r("pro", 1.0, "watch", "Earnings beat")]
    s = compute_stats(rows, "1h")
    assert {g["key"]: g["count"] for g in s["by_engine"]} == {"Local machine learning": 1, "Chart patterns": 2,
                                                              "Pro AI": 1}
    assert {g["key"]: g["count"] for g in s["by_event"]} == {"Earnings beat": 1, "Chart: Breakout": 2}
    assert s["overall"]["count"] == 1 and sum(g["count"] for g in s["by_source"]) == 1   # news numbers untouched


async def test_chart_signals_stay_out_of_news_merges_and_the_market_monitor(scan_env):
    scan_env.ctx.config.update({"chart": {"signals": "review"}})   # (watch-only ones are left out anyway)
    await scan_env.charts.scan(scan_env.now)
    from newstrader.market.monitor import MarketMonitor

    mon = MarketMonitor(scan_env.ctx, tickers=make_tickers(scan_env.ctx.db))
    universe, _ = await mon._build_universe(scan_env.ctx.config.settings.market, scan_env.now)
    assert "XYZ" not in universe                        # a chart signal doesn't add its stock to the watch list
    assert mon._news_from_db("XYZ", scan_env.now) is None   # ...and isn't "the news behind a spike"


# ================================================================================ the service: bars and caching
def calls(broker, timeframe):
    return [c for c in broker.bar_calls if c[0] == timeframe]


async def test_readings_are_cached_and_bars_downloaded_once(scan_env):
    b, charts, now = scan_env.broker, scan_env.charts, scan_env.now
    r = await charts.reading("xyz", now)
    assert r.symbol == "XYZ" and r.summary == read_chart("XYZ", scan_env.minutes, scan_env.daily, now).summary
    [m1], [d1] = calls(b, "1Min"), calls(b, "1Day")
    assert m1[1] == ("XYZ",) and m1[2] == history_start(now) == ny(YESTERDAY, 4, 0)
    assert d1[2] < now - timedelta(days=360)
    assert await charts.reading("XYZ", now + timedelta(seconds=20)) is r    # cached reading
    assert len(b.bar_calls) == 2
    r2 = await charts.reading("XYZ", now + timedelta(seconds=45))           # new reading, bars still fresh
    assert r2 is not r and len(b.bar_calls) == 2
    await charts.reading("XYZ", now + timedelta(seconds=90))                # newest minutes only, no daily bars
    tail = calls(b, "1Min")[-1]
    assert len(calls(b, "1Min")) == 2 and tail[2] == now - timedelta(minutes=2) and len(calls(b, "1Day")) == 1
    next_day = datetime.combine(TODAY + timedelta(days=1), datetime.min.time(), UTC) + timedelta(hours=15)
    await charts.reading("XYZ", next_day)
    assert len(calls(b, "1Day")) == 2                                        # daily bars: once a day


async def test_bars_for_many_stocks_go_in_one_request(scan_env):
    b = scan_env.broker
    for sym in ("AAA", "BBB"):
        b.bar_data[sym], b.daily_data[sym] = scan_env.minutes, scan_env.daily
    out = await scan_env.charts.readings(["XYZ", "AAA", "BBB", "xyz"], scan_env.now)
    assert [r.symbol for r in out] == ["XYZ", "AAA", "BBB"]
    assert [c[1] for c in b.bar_calls] == [("XYZ", "AAA", "BBB")] * 2
    assert scan_env.charts.requests == 2


async def test_the_market_monitors_bars_are_reused(scan_env):
    b, charts, now = scan_env.broker, scan_env.charts, scan_env.now
    await charts.reading("XYZ", now)
    later = now + timedelta(minutes=3)
    new = [{"t": iso(now + timedelta(minutes=k)), "o": 103.0, "h": 103.6, "l": 102.9, "c": 103.5, "v": 900}
           for k in range(3)]
    seen = []

    def fresh_bars(symbol, when=None):
        seen.append(symbol)
        return new, now - timedelta(minutes=35), later + timedelta(seconds=5)

    scan_env.ctx.services["market"] = SimpleNamespace(watched_stocks=lambda: ["XYZ"], fresh_bars=fresh_bars)
    r = await charts.reading("XYZ", later + timedelta(seconds=10))
    assert seen == ["XYZ"] and len(calls(b, "1Min")) == 1          # no request: the monitor had them
    assert r.price == 103.5 and r.bar_at == iso(now + timedelta(minutes=2))
    # monitor bars that don't reach back to what we have would leave a gap: fetch instead
    scan_env.ctx.services["market"] = SimpleNamespace(
        fresh_bars=lambda s, w=None: ([], later + timedelta(minutes=30), later + timedelta(minutes=31)))
    await charts.reading("XYZ", later + timedelta(minutes=31))
    assert len(calls(b, "1Min")) == 2


async def test_a_failed_download_is_reported_and_retried(scan_env, monkeypatch):
    def down(*_a, **_k):
        raise RuntimeError("HTTP 500")

    monkeypatch.setattr(scan_env.broker, "bars_multi", down)
    r = await scan_env.charts.reading("XYZ", scan_env.now)
    assert r.price is None and "HTTP 500" in scan_env.charts._errors["XYZ"]
    from newstrader.chart.service import NoChartData

    with pytest.raises(NoChartData, match="HTTP 500"):
        await scan_env.charts.card("XYZ", scan_env.now + timedelta(minutes=1))


async def test_pacer_keeps_requests_under_the_limit():
    clock = {"t": 0.0}
    slept: list[float] = []

    async def sleep(s):
        slept.append(s)
        clock["t"] += s

    p = Pacer(3, clock=lambda: clock["t"], sleep=sleep)
    for _ in range(3):
        await p.room()
        p.take()
    assert not slept and p.used() == 3 and not p.fits()
    await p.room()                                          # the 4th waits until the first is a minute old
    p.take()
    assert slept == [60.0]
    p.charge(2)                                             # a big answer that took 2 more pages
    clock["t"] += 1
    await p.room()
    assert slept[-1] == pytest.approx(59.0) and p.used() <= 3
    q = Pacer(10, clock=lambda: clock["t"], sleep=sleep)
    q.charge(5)                                             # chart signals leave a reserve for news checks
    assert q.fits(3) and q.fits(3, reserve=2) and not q.fits(3, reserve=6)
    assert Pacer(3).fits(3, reserve=6) and not q.fits(20)


async def test_a_scan_waiting_for_the_pacer_never_holds_up_a_news_check(scan_env):
    b = scan_env.broker
    b.bar_data["AAA"], b.daily_data["AAA"] = scan_env.minutes, scan_env.daily
    charts = ChartService(scan_env.ctx, pacer=Pacer(10))
    scan_env.ctx.services["chart"] = charts
    charts.pacer.charge(5)  # half this minute's requests used: room for a news check, but a scan leaves it to them
    scan = asyncio.create_task(charts.readings(["XYZ"], scan_env.now))
    try:
        await asyncio.sleep(0.05)
        assert not scan.done() and not charts._fetch_lock.locked() and not b.bar_calls  # waits outside the lock
        await asyncio.wait_for(charts.reading("AAA", scan_env.now, urgent=True), 2)  # (what prefetch() asks for)
        assert [c[1] for c in b.bar_calls] == [("AAA",), ("AAA",)] and not scan.done()
        charts.pacer.charge(10)  # no room at all: a reading that needs no new bars still doesn't wait
        r = await asyncio.wait_for(charts.reading("AAA", scan_env.now + timedelta(seconds=45)), 1)
        assert r.price is not None and len(b.bar_calls) == 2
    finally:
        scan.cancel()


def test_history_start_is_two_trading_sessions():
    # Monday morning: from Friday's pre-market; a holiday is skipped (Thanksgiving 2026 is Thursday Nov 26)
    monday = datetime(2026, 10, 12, 14, 0, tzinfo=UTC)
    assert history_start(monday).date().isoformat() == "2026-10-09"
    assert history_start(datetime(2026, 11, 27, 15, 0, tzinfo=UTC)).date().isoformat() == "2026-11-25"
    assert history_start(datetime(2026, 10, 10, 15, 0, tzinfo=UTC)).date().isoformat() == "2026-10-08"  # Saturday


# ================================================================================ migration, settings, API
def test_migration_12_adds_the_chart_columns(tmp_path):
    path = tmp_path / "old.db"
    c = sqlite3.connect(path)
    for i, script in enumerate(MIGRATIONS[:11], start=1):  # a database from before charts
        c.executescript(script)
        c.execute(f"PRAGMA user_version={i}")
    c.execute("INSERT INTO signals (id, ticker, direction, confidence) VALUES (1, 'NVDA', 'bullish', 80)")
    c.commit()
    c.close()
    db = Database(path)
    try:
        assert db.scalar("PRAGMA user_version") == len(MIGRATIONS) >= 12
        row = db.query_one("SELECT * FROM signals WHERE id = 1")
        assert row["ticker"] == "NVDA"
        assert {"chart_verdict", "chart_reason", "chart_score", "chart_adjust"} <= set(row)
        assert row["chart_verdict"] is None
    finally:
        db.close()


def test_chart_settings_defaults_and_validation(ctx):
    s = ctx.config.settings.chart
    assert s.confirm == "soft" and s.signals == "watch"
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ctx.config.update({"chart": {"signals": "trade"}})   # there is no auto-trading mode
    with pytest.raises(ValidationError):
        ctx.config.update({"chart": {"confirm": "maybe"}})


def test_chart_api(client, ctx, monkeypatch):
    minutes, daily, now = breakout_chart()
    monkeypatch.setattr("newstrader.chart.service.utcnow", lambda: now)
    assert client.get("/api/chart/XYZ").status_code == 503               # no chart service
    broker = FakeBroker(prices={})
    broker.bar_data["XYZ"], broker.daily_data["XYZ"] = minutes, daily
    ctx.services["chart"] = ChartService(ctx)
    assert client.get("/api/chart/XYZ").status_code == 503               # not connected
    ctx.services["trader"] = SimpleNamespace(broker=broker, clock={"is_open": True})
    r = client.get("/api/chart/xyz")
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["symbol"] == "XYZ" and d["feed"] == "IEX" and d["confirm"] == "soft" and d["signals"] == "watch"
    rd = d["reading"]
    assert rd["summary"] == "Uptrend, above VWAP, breaking out on heavy volume" and rd["trend"] == "up"
    assert {"rsi", "macd_hist", "vwap_pct", "rel_volume", "trend_daily", "levels", "patterns", "score"} <= set(rd)
    assert all(p["explain"] for p in rd["patterns"])
    s = d["series"]
    assert len(s["t"]) == len(s["close"]) == len(s["vwap"]) == 185                # today's session only
    assert s["close"][-1] == pytest.approx(102.68, abs=0.01) and s["vwap"][-1] < s["close"][-1]
    assert d["signal"]["direction"] == "bullish"
    assert client.get("/api/chart/not a ticker!").status_code == 400
    assert client.get("/api/chart/NODATA").status_code == 503


def test_signals_api_chart_filter(client, ctx):
    common = {"created_at": iso(), "ticker": "NVDA", "direction": "bullish", "confidence": 70, "sources_seen": "[]"}
    news = ctx.db.insert("signals", {**common, "engine": "local", "action": "bought", "chart_verdict": "agrees",
                                     "chart_adjust": 4, "chart_reason": "Chart agrees: uptrend.", "chart_score": 0.6})
    watch = ctx.db.insert("signals", {**common, "engine": "chart", "action": "watch"})
    review = ctx.db.insert("signals", {**common, "engine": "chart", "action": "review", "review_status": "pending"})
    ids = lambda q: {s["id"] for s in client.get("/api/signals" + q).json()["signals"]}  # noqa: E731
    assert ids("?engine=chart") == {watch, review}
    assert ids("?engine=news") == {news}
    assert ids("?engine=main") == {news, review}
    row = next(s for s in client.get("/api/signals").json()["signals"] if s["id"] == news)
    assert row["chart_verdict"] == "agrees" and row["chart_adjust"] == 4 and row["chart_score"] == 0.6
    ctx.services["pipeline"] = Pipeline(ctx, tickers=make_tickers(ctx.db))
    r = client.post(f"/api/signals/{watch}/approve")
    assert r.status_code == 409 and "watch-only" in r.json()["detail"]


def test_the_review_queue_is_loaded_on_its_own(client, ctx):
    """A busy day of watch-only chart rows never pushes a signal waiting for review out of the Signals tab."""
    common = {"created_at": iso(), "ticker": "NVDA", "direction": "bullish", "confidence": 70, "sources_seen": "[]"}
    waiting = ctx.db.insert("signals", {**common, "engine": "claude", "action": "review", "review_status": "pending"})
    ctx.db.insert("signals", {**common, "engine": "claude", "action": "review", "review_status": "dismissed"})
    for _ in range(600):
        ctx.db.insert("signals", {**common, "engine": "chart", "action": "watch"})
    newest = [s["id"] for s in client.get("/api/signals?limit=500").json()["signals"]]
    assert waiting not in newest  # (the old way: the queue was cut from this list)
    queue = client.get("/api/signals?review_only=true&limit=1000").json()["signals"]
    assert [s["id"] for s in queue] == [waiting]
    js = (Path(__file__).resolve().parent.parent / "newstrader/web/js/signals.js").read_text(encoding="utf-8")
    assert "/signals?review_only=true" in js and 'engine: "main"' in js


async def test_monitor_hands_its_bars_to_the_chart_service(ctx):
    """The real market monitor: after a check it offers the bars it fetched and wakes the chart service."""
    from newstrader.market.monitor import MarketMonitor
    from tests.test_market import T0, make_bars, spike_bars

    ctx.loop = asyncio.get_running_loop()
    ctx.bus.bind_loop(ctx.loop)
    broker = FakeBroker(prices={})
    broker.bar_data["SPY"] = make_bars([600.0] * 35)
    broker.bar_data["NVDA"] = spike_bars()
    ctx.services["trader"] = SimpleNamespace(broker=broker, clock={"is_open": True}, positions=[], open_orders=[],
                                             account={})
    ctx.config.update({"market": {"watchlist": ["NVDA", "QUIET"], "index_symbols": ["SPY"], "world_symbols": [],
                                  "watch_signals_minutes": 0, "lookup_news": False}})
    m = MarketMonitor(ctx, tickers=make_tickers(ctx.db))
    woke = []
    ctx.services["chart"] = SimpleNamespace(market_checked=lambda now: woke.append(now))
    await m.run_once(now=T0)
    assert woke == [T0] and m.watched_stocks() == ["NVDA", "QUIET"]
    bars, start, end = m.fresh_bars("NVDA", T0 + timedelta(seconds=30))
    assert len(bars) == 35 and end == T0 and start <= T0 - timedelta(minutes=35)
    assert m.fresh_bars("QUIET", T0)[0] == []                        # asked for, no trades: still covered
    assert m.fresh_bars("AAPL", T0) is None                          # not asked for
    assert m.fresh_bars("NVDA", T0 + timedelta(minutes=5)) is None   # too old


async def test_a_news_signal_never_merges_into_a_chart_signal(ctx):
    from newstrader.ai.claude_client import ClaudeAnalyzer
    from tests.helpers import FakeClaude, sig, signal_json
    from tests.test_pro_ai import news, run

    ctx.config.update({"ai": {"engine": "claude"}, "chart": {"confirm": "off"}})
    broker = FakeBroker(prices={"NVDA": 180.0})
    trader = Trader(ctx, broker_factory=lambda: broker)
    ctx.services["trader"] = trader
    await trader.connect()
    try:
        chart_id = ctx.db.insert("signals", {"created_at": iso(), "ticker": "NVDA", "direction": "bullish",
                                             "confidence": 70, "engine": "chart", "action": "review",
                                             "review_status": "pending", "sources_seen": "[]"})
        claude = FakeClaude([signal_json(sig("NVDA", confidence=90))])
        p = Pipeline(ctx, analyzer=ClaudeAnalyzer(ctx, client_factory=lambda: claude), tickers=make_tickers(ctx.db))
        ctx.services["pipeline"] = p
        out = await run(p, news("Nvidia wins a giant government contract"))
        s = out["signals"][0]
        assert s["action"] == "bought" and s["merged_into"] is None and s["engine"] == "claude"
        assert ctx.db.query_one("SELECT corroborations FROM signals WHERE id = ?", (chart_id,))["corroborations"] == 1
    finally:
        await trader.stop()
