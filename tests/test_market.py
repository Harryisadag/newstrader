"""Market monitor: spike / market-move detection, the monitor service, the fake broker's market data, the API,
and the "don't chase" guard in the Trader. Never talks to Alpaca."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from newstrader.db import iso, utcnow
from newstrader.market import detect
from newstrader.market.detect import (
    DayLevels,
    EventMemory,
    detect_spike,
    market_phase,
    market_window_move,
    window_stats,
    world_alert_worthy,
)
from newstrader.market.monitor import MarketMonitor, build_universe, load_events
from newstrader.trading.fake_broker import DEMO_SPIKE, FakeBroker
from newstrader.trading.risk import chase_check
from newstrader.trading.trader import Trader
from tests.helpers import make_tickers

T0 = datetime(2026, 10, 7, 15, 0, tzinfo=UTC)  # Wednesday 11:00 New York time


def make_bars(prices: list[float], end: datetime = T0, volumes: list[float] | None = None) -> list[dict]:
    """1-minute bars; the last one starts one minute before `end`."""
    n = len(prices)
    return [{"t": iso(end - timedelta(minutes=n - i)), "o": p, "h": p, "l": p, "c": p,
             "v": (volumes[i] if volumes else 1000)} for i, p in enumerate(prices)]


def spike_bars(base: float = 100.0, tail: tuple = (101.5, 103.0, 104.5), tail_volume: float = 6000,
               flat: int = 32, end: datetime = T0) -> list[dict]:
    """Quiet trading at `base` (1,000 shares a minute), then a fast move on heavy volume."""
    prices = [base] * flat + list(tail)
    volumes = [1000] * flat + [tail_volume] * len(tail)
    return make_bars(prices, end, volumes)


# ====================================================================== detect.py
def test_spike_up_with_volume():
    st = window_stats("NVDA", spike_bars(), T0, 5)
    assert st.price == 104.5 and st.ref_price == 100.0 and not st.stale
    assert st.change_pct == pytest.approx(4.5)
    # window = last 5 bars: 2 quiet (1,000) + 3 busy (6,000); usual = median 1,000 x 5 minutes
    assert st.volume == 20000 and st.baseline_volume == 5000
    assert st.volume_ratio == pytest.approx(4.0)
    assert detect_spike(st, 3.0, 3.0) == "spike_up"
    assert detect_spike(st, 3.0, 5.0) is None  # not unusual enough, and 4.5% < 1.5 x 3%


def test_spike_down():
    st = window_stats("TSLA", spike_bars(tail=(98.5, 97.0, 95.8)), T0, 5)
    assert st.change_pct == pytest.approx(-4.2)
    assert detect_spike(st, 3.0, 3.0) == "spike_down"


def test_small_move_is_not_a_spike():
    st = window_stats("AAPL", spike_bars(tail=(100.5, 101.0, 101.5)), T0, 5)
    assert detect_spike(st, 3.0, 3.0) is None


def test_thin_data_needs_a_bigger_move():
    thin = make_bars([100, 100, 100, 100, 104])  # too few earlier bars to judge the volume
    st = window_stats("ABC", thin, T0, 3)
    assert st.volume_ratio is None and st.baseline_volume is None
    assert st.change_pct == pytest.approx(4.0)
    assert detect_spike(st, 3.0, 3.0) is None  # 4% < 1.5 x 3%
    st = window_stats("ABC", make_bars([100, 100, 100, 100, 105]), T0, 3)
    assert detect_spike(st, 3.0, 3.0) == "spike_up"


def test_stale_or_missing_data_never_spikes():
    st = window_stats("NVDA", spike_bars(), T0 + timedelta(minutes=5), 5)
    assert st.stale and detect_spike(st, 3.0, 3.0) is None
    empty = window_stats("NVDA", [], T0, 5)
    assert empty.price is None and empty.stale and detect_spike(empty, 3.0, 3.0) is None
    # no "before" price within reach -> no change
    st = window_stats("NVDA", make_bars([100, 104.5]), T0, 5)
    assert st.change_pct is None and detect_spike(st, 3.0, 3.0) is None


def test_min_price_filter():
    st = window_stats("PNNY", spike_bars(base=1.0, tail=(1.02, 1.04, 1.05)), T0, 5)
    assert detect_spike(st, 3.0, 3.0) == "spike_up"
    assert detect_spike(st, 3.0, 3.0, min_price=2.0) is None


def test_move_with_the_market_does_not_count():
    st = window_stats("NVDA", spike_bars(tail=(101, 102.5, 103.5)), T0, 5, market_change_pct=3.0)
    assert st.rel_change_pct == pytest.approx(0.5)
    assert detect_spike(st, 3.0, 3.0) is None  # the whole market rose 3%
    st = window_stats("NVDA", spike_bars(tail=(101, 102.5, 103.5)), T0, 5, market_change_pct=0.1)
    assert detect_spike(st, 3.0, 3.0) == "spike_up"
    # the market fell more than the stock: the stock didn't spike down on its own
    st = window_stats("NVDA", spike_bars(tail=(99, 98, 96.5)), T0, 5, market_change_pct=-4.0)
    assert detect_spike(st, 3.0, 3.0) is None


def test_volume_surge_with_a_smaller_move():
    st = window_stats("AMD", spike_bars(tail=(100.8, 101.4, 101.8), tail_volume=12000), T0, 5)
    assert st.volume_ratio >= 6
    assert detect_spike(st, 3.0, 3.0) == "volume_surge"


def test_market_window_move():
    falling = make_bars([100] * 10 + [99.8, 99.5, 99.3, 99.1, 98.9, 98.8, 98.8] + [98.7] * 8)
    st = window_stats("SPY", falling, T0, detect.MARKET_WINDOW_MINUTES)
    assert st.change_pct == pytest.approx(-1.3)
    assert market_window_move(st, 1.0)
    assert not market_window_move(st, 1.5)


def test_day_levels_once_per_level_per_day():
    lv = DayLevels()
    day = "2026-10-07"
    assert lv.new_level("SPY", day, -0.6, 1.0) is None
    assert lv.new_level("SPY", day, -1.2, 1.0) == -1.0
    assert lv.new_level("SPY", day, -1.6, 1.0) is None
    assert lv.new_level("SPY", day, -0.4, 1.0) is None
    assert lv.new_level("SPY", day, -1.1, 1.0) is None  # swung back through -1%: no repeat
    assert lv.new_level("SPY", day, -2.3, 1.0) == -2.0
    assert lv.new_level("SPY", day, 1.0, 1.0) == 1.0  # up levels are separate
    assert lv.new_level("QQQ", day, -3.4, 1.0) == -3.0  # straight to -3.4%: one event, not three
    assert lv.new_level("QQQ", day, -2.5, 1.0) is None
    assert lv.new_level("SPY", "2026-10-08", -1.2, 1.0) == -1.0  # a new day starts over
    lv.mark("IWM", day, -2.0, 1.0)
    assert lv.new_level("IWM", day, -1.5, 1.0) is None


def test_world_etf_alert_threshold():
    assert not world_alert_worthy(-1.5, 1.0)
    assert world_alert_worthy(-2.0, 1.0) and world_alert_worthy(2.4, 1.0)
    assert not world_alert_worthy(None, 1.0)


def test_event_memory_holds_one_move():
    mem = EventMemory()
    key = ("NVDA", "spike_up")
    assert mem.is_new(key, T0, 5, 4.0, 3.0)
    assert not mem.is_new(key, T0 + timedelta(minutes=2), 5, 4.5, 3.0)
    assert mem.is_new(key, T0 + timedelta(minutes=3), 5, 7.2, 3.0)  # a second leg
    assert mem.is_new(key, T0 + timedelta(minutes=9), 5, 4.0, 3.0)


def test_market_phase():
    session = {"open": "2026-10-07T13:30:00Z", "close": "2026-10-07T20:00:00Z"}
    assert market_phase(T0, {"is_open": True}, None, False) == "open"
    assert market_phase(T0, {"is_open": False}, session, False) == "closed"
    premarket = datetime(2026, 10, 7, 12, 0, tzinfo=UTC)  # 8:00 ET
    assert market_phase(premarket, {"is_open": False}, session, True) == "extended"
    assert market_phase(premarket, {"is_open": False}, session, False) == "closed"
    after = datetime(2026, 10, 7, 21, 0, tzinfo=UTC)  # 5pm ET
    assert market_phase(after, {"is_open": False}, session, True) == "extended"
    late = datetime(2026, 10, 8, 1, 0, tzinfo=UTC)  # 9pm ET
    assert market_phase(late, {"is_open": False}, session, True) == "closed"
    assert market_phase(premarket, {"is_open": False}, None, True) == "closed"  # holiday / weekend
    assert market_phase(T0, None, session, False) == "open"  # no clock yet: use the calendar


def test_build_universe_priority_and_cap():
    groups = [("position", ["AAPL"]), ("order", ["AAPL", "MSFT"]), ("watchlist", ["TSLA", "bad ticker!"]),
              ("signal", ["NVDA", "AMD"]), ("mover", ["PLTR", "AAPL"]), ("active", ["F"])]
    uni, dropped = build_universe(groups, 4, ["SPY", "QQQ"], ["EWJ"])
    assert list(uni)[:4] == ["AAPL", "MSFT", "TSLA", "NVDA"]
    assert uni["AAPL"] == ["position", "order", "mover"]
    assert dropped == 3  # AMD, PLTR, F
    assert uni["SPY"] == ["index"] and uni["EWJ"] == ["world"]  # always watched, on top of the cap


# ====================================================================== fake broker market data
def test_fake_broker_snapshots_and_movers():
    fb = FakeBroker(prices={"AAPL": 105.0, "NVDA": 90.0, "MSFT": 500.0})
    fb.prev_close = {"AAPL": 100.0, "NVDA": 100.0}
    snaps = fb.snapshots(["AAPL", "NVDA", "MSFT", "NOPE"])
    assert set(snaps) == {"AAPL", "NVDA", "MSFT"}
    assert snaps["AAPL"]["change_pct"] == pytest.approx(5.0) and snaps["AAPL"]["prev_close"] == 100.0
    assert snaps["MSFT"]["change_pct"] == 0
    mv = fb.movers(top=5)
    assert [m["symbol"] for m in mv["gainers"]] == ["AAPL"] and [m["symbol"] for m in mv["losers"]] == ["NVDA"]
    assert mv["losers"][0]["change_pct"] == pytest.approx(-10.0) and mv["updated"]
    fb.bar_data["NVDA"] = make_bars([90] * 5, end=utcnow(), volumes=[500] * 5)
    fb.bar_data["AAPL"] = make_bars([105] * 5, end=utcnow(), volumes=[100] * 5)
    act = fb.most_actives(top=5)
    assert [a["symbol"] for a in act["items"]] == ["NVDA", "AAPL"] and act["items"][0]["volume"] == 2500
    fb.screener_error = "screener not available"
    with pytest.raises(RuntimeError):
        fb.movers()
    with pytest.raises(RuntimeError):
        fb.most_actives()


def test_fake_broker_unseeded_bars_unchanged():
    fb = FakeBroker()
    assert fb.bars_multi(["AAPL"], T0 - timedelta(minutes=30), T0) == {}
    assert fb.bars("AAPL", T0 - timedelta(minutes=30), T0)[0]["c"] == 230.0  # one flat bar, as before
    assert fb.movers()["gainers"] == []


def test_fake_broker_demo_has_market_data_and_a_spike():
    fb = FakeBroker.demo()
    now = utcnow()
    assert {"SPY", "QQQ", "EWJ", DEMO_SPIKE} <= set(fb.bar_data)
    assert {p["symbol"] for p in fb.positions()} == {"AAPL", "NVDA"}
    st = window_stats(DEMO_SPIKE, fb.bar_data[DEMO_SPIKE], now, 5)
    assert detect_spike(st, 3.0, 3.0) == "spike_up"
    assert fb.snapshots(["SPY"])["SPY"]["change_pct"] == pytest.approx(-1.2, abs=0.05)
    again = FakeBroker.demo()
    assert again.bar_data["SPY"][0]["c"] == fb.bar_data["SPY"][0]["c"]  # fixed seed: repeatable


def test_real_broker_snapshot_conversion():
    from alpaca.data.models import Snapshot

    from newstrader.trading.broker import snapshot_to_dict

    raw = {"latestTrade": {"t": "2026-10-07T15:00:00Z", "p": 105.0, "s": 10, "x": "V", "i": 1, "c": [], "z": "C"},
           "minuteBar": {"t": "2026-10-07T14:59:00Z", "o": 104, "h": 105, "l": 104, "c": 105, "v": 1000, "n": 9,
                         "vw": 104.5},
           "dailyBar": {"t": "2026-10-07T04:00:00Z", "o": 100, "h": 106, "l": 99, "c": 105, "v": 100000, "n": 99,
                        "vw": 102},
           "prevDailyBar": {"t": "2026-10-06T04:00:00Z", "o": 98, "h": 101, "l": 97, "c": 100, "v": 90000, "n": 99,
                            "vw": 99}}
    d = snapshot_to_dict(Snapshot("X", raw))
    assert d["price"] == 105 and d["prev_close"] == 100 and d["change_pct"] == pytest.approx(5.0)
    assert d["day_volume"] == 100000 and d["minute_bar"]["c"] == 105
    # next morning before the open: today's daily bar doesn't exist yet -> compare with yesterday's close
    raw["latestTrade"] = dict(raw["latestTrade"], t="2026-10-08T11:00:00Z", p=103.95)
    d = snapshot_to_dict(Snapshot("X", raw))
    assert d["prev_close"] == 105 and d["change_pct"] == pytest.approx(-1.0) and d["day_volume"] is None


# ====================================================================== the monitor service
class FakeAlerts:
    def __init__(self):
        self.sent: list[dict] = []

    async def send(self, kind, title, message="", level="info", force=False, fields=None):
        self.sent.append({"kind": kind, "title": title, "message": message, "level": level, "fields": fields})
        return {"toast": True}


@pytest.fixture
async def mon(ctx):
    ctx.loop = asyncio.get_running_loop()
    ctx.bus.bind_loop(ctx.loop)
    broker = FakeBroker(prices={})
    broker.bar_data["SPY"] = make_bars([600.0] * 35)
    trader = SimpleNamespace(broker=broker, clock={"is_open": True}, positions=[], open_orders=[], account={})
    alerts = FakeAlerts()
    ctx.services.update({"trader": trader, "alerts": alerts})
    ctx.config.update({"market": {"watchlist": ["NVDA"], "index_symbols": ["SPY"], "world_symbols": [],
                                  "watch_signals_minutes": 0, "lookup_news": True}})
    m = MarketMonitor(ctx, tickers=make_tickers(ctx.db))
    ctx.services["market"] = m
    return SimpleNamespace(m=m, broker=broker, trader=trader, alerts=alerts, ctx=ctx)


def _events(ctx, kind=None):
    rows = ctx.db.query("SELECT * FROM market_events ORDER BY id")
    return [r for r in rows if kind is None or r["kind"] == kind]


async def test_monitor_stores_and_alerts_a_spike(mon):
    q = mon.ctx.bus.subscribe()
    mon.broker.bar_data["NVDA"] = spike_bars()
    res = await mon.m.run_once(now=T0)
    assert res["status"] == "ok" and res["events"] == 1 and res["alerts"] == 1
    [ev] = _events(mon.ctx)
    assert ev["kind"] == "spike_up" and ev["symbol"] == "NVDA" and ev["scope"] == "stock" and ev["alerted"] == 1
    assert ev["change_pct"] == pytest.approx(4.5) and ev["volume_ratio"] == pytest.approx(4.0)
    assert ev["window_min"] == 5 and ev["trading_day"] == "2026-10-07"
    [alert] = mon.alerts.sent
    assert alert["kind"] == "market_spike" and alert["level"] == "warn"
    assert alert["title"] == "NVDA jumped +4.5% in 5 min"
    assert "Now $104.50 (was $100.00) on 4.0x its usual volume" in alert["message"]
    assert "No news found for it yet" in alert["message"]
    assert alert["fields"]["Move"] == "+4.5% in 5 min"
    types = []
    while not q.empty():
        types.append(q.get_nowait()["type"])
    assert "market_event" in types and "market" in types
    status = {c["component"]: c for c in mon.ctx.state.components()}["market"]
    assert status["level"] == "ok" and "Watching 1 stock · IEX" in status["detail"]
    # the bars stay cached for the chase guard
    assert mon.m.price_near("NVDA", T0 - timedelta(minutes=10), now=T0) == 100.0
    assert mon.m.price_near("NVDA", T0 - timedelta(minutes=10), now=T0 + timedelta(minutes=10)) is None


async def test_same_move_is_stored_once_and_cooldown_blocks_second_alert(mon):
    mon.broker.bar_data["NVDA"] = spike_bars()
    await mon.m.run_once(now=T0)
    await mon.m.run_once(now=T0 + timedelta(seconds=60))  # still the same move
    assert len(_events(mon.ctx)) == 1
    # a second jump 10 minutes later: stored, but not alerted (30-minute cooldown)
    later = T0 + timedelta(minutes=10)
    mon.broker.bar_data["NVDA"] = spike_bars(base=104.5, tail=(106.0, 107.6, 109.2), end=later)
    res = await mon.m.run_once(now=later)
    assert res["events"] == 1 and res["alerts"] == 0
    rows = _events(mon.ctx, "spike_up")
    assert len(rows) == 2 and rows[1]["alerted"] == 0
    assert len(mon.alerts.sent) == 1


async def test_max_alerts_per_check_most_significant_first(mon):
    mon.ctx.config.update({"market": {"watchlist": ["NVDA", "AAPL", "TSLA"], "max_alerts_per_scan": 2}})
    mon.broker.bar_data["NVDA"] = spike_bars(tail=(101.5, 103.0, 104.5))
    mon.broker.bar_data["AAPL"] = spike_bars(tail=(102.0, 104.0, 106.0))
    mon.broker.bar_data["TSLA"] = spike_bars(tail=(97.0, 94.0, 92.0))
    res = await mon.m.run_once(now=T0)
    assert res["events"] == 3 and res["alerts"] == 2
    assert [a["title"].split()[0] for a in mon.alerts.sent] == ["TSLA", "AAPL"]
    assert {r["symbol"]: r["alerted"] for r in _events(mon.ctx)} == {"NVDA": 0, "AAPL": 1, "TSLA": 1}


async def test_turned_off_alert_type_still_stores_the_event(mon):
    mon.ctx.config.update({"alerts": {"on_market_spike": False}})
    mon.broker.bar_data["NVDA"] = spike_bars()
    await mon.m.run_once(now=T0)
    assert mon.alerts.sent == [] and _events(mon.ctx)[0]["alerted"] == 0


async def test_spike_linked_to_recent_signal(mon):
    mon.ctx.db.insert("signals", {"created_at": iso(T0 - timedelta(minutes=20)), "ticker": "NVDA",
                                  "direction": "bullish", "confidence": 85, "headline": "Nvidia wins huge contract",
                                  "url": "https://example.com/nv", "action": "bought"})
    mon.broker.bar_data["NVDA"] = spike_bars()
    await mon.m.run_once(now=T0)
    [ev] = _events(mon.ctx)
    assert ev["headline"] == "Nvidia wins huge contract" and ev["signal_id"] and ev["url"] == "https://example.com/nv"
    assert json.loads(ev["detail"])["news_source"] == "signal"
    assert mon.alerts.sent[0]["message"].endswith("News: Nvidia wins huge contract")


async def test_spike_linked_to_tagged_story_or_alpaca_news(mon):
    nid = mon.ctx.db.insert("news_items", {"source_id": "alpaca", "title": "Nvidia guidance raised",
                                           "url": "https://example.com/g", "received_at": iso(T0 - timedelta(minutes=5)),
                                           "symbols": json.dumps(["NVDA", "AMD"])})
    mon.broker.bar_data["NVDA"] = spike_bars()
    await mon.m.run_once(now=T0)
    assert _events(mon.ctx)[0]["news_id"] == nid
    # nothing in our own database: ask Alpaca's news
    mon.ctx.config.update({"market": {"watchlist": ["TSLA"]}})
    mon.broker.news_items = [{"created_at": T0 - timedelta(minutes=50), "symbols": ["TSLA"], "headline": "Old one"},
                             {"created_at": T0 - timedelta(minutes=8), "symbols": ["TSLA"],
                              "headline": "Tesla recalls cars", "url": "https://example.com/t"}]
    mon.broker.bar_data["TSLA"] = spike_bars(tail=(98.5, 97.0, 95.8))
    await mon.m.run_once(now=T0)
    ev = _events(mon.ctx, "spike_down")[0]
    assert ev["headline"] == "Tesla recalls cars" and json.loads(ev["detail"])["news_source"] == "alpaca"
    assert mon.alerts.sent[-1]["title"] == "TSLA dropped 4.2% in 5 min"


async def test_market_closed_no_detection_or_alerts(mon):
    mon.trader.clock = {"is_open": False}
    mon.broker.bar_data["NVDA"] = spike_bars()
    mon.broker.prev_close["SPY"] = 610.0
    res = await mon.m.run_once(now=T0)
    assert res["status"] == "closed"
    assert _events(mon.ctx) == [] and mon.alerts.sent == []
    assert not any(c[0] == "bars_multi" for c in mon.broker.calls)
    payload = mon.m.payload()
    assert payload["phase"] == "closed" and payload["indices"][0]["price"] == 600.0  # prices still refreshed
    n = len(mon.broker.calls)
    await mon.m.run_once(now=T0 + timedelta(minutes=2))  # closed: refresh only every 10 minutes
    assert len(mon.broker.calls) == n


async def test_extended_hours_scans_before_the_open(mon):
    premarket = datetime(2026, 10, 7, 12, 0, tzinfo=UTC)  # 8:00 New York time, a weekday
    mon.trader.clock = {"is_open": False}
    mon.broker.bar_data["NVDA"] = spike_bars(end=premarket)
    mon.broker.bar_data["SPY"] = make_bars([600.0] * 35, end=premarket)
    assert (await mon.m.run_once(now=premarket))["status"] == "closed"
    mon.ctx.config.update({"market": {"extended_hours": True}})
    res = await mon.m.run_once(now=premarket)
    assert res["phase"] == "extended" and res["events"] == 1


async def test_backs_off_while_training_or_backtesting(mon):
    assert mon.m.next_interval() == 60
    mon.ctx.services["ml_trainer"] = SimpleNamespace(running=True)
    assert mon.m.next_interval() == 180
    mon.ctx.services["ml_trainer"] = SimpleNamespace(running=False)
    mon.ctx.services["backtest"] = SimpleNamespace(running=True)
    assert mon.m.next_interval() == 180


async def test_market_wide_moves(mon):
    falling = [600.0] * 20 + [598.0, 596.0, 594.0, 592.5, 592.2] + [592.2] * 10
    mon.broker.bar_data["SPY"] = make_bars(falling)
    mon.broker.prev_close["SPY"] = 605.5  # down 2.2% on the day
    res = await mon.m.run_once(now=T0)
    moves = _events(mon.ctx, "market_move")
    assert len(moves) == 2 and res["alerts"] == 1  # one alert per symbol per check
    day = next(r for r in moves if r["window_min"] is None)
    window = next(r for r in moves if r["window_min"] == 15)
    assert json.loads(day["detail"])["level_pct"] == -2.0 and day["alerted"] == 1
    assert window["change_pct"] == pytest.approx(-1.3) and window["scope"] == "market"
    [alert] = mon.alerts.sent
    assert alert["kind"] == "market_move" and alert["title"] == "S&P 500 (SPY) is now down 2% today"
    assert json.loads(window["detail"])["title"] == "S&P 500 (SPY) fell 1.3% in 15 minutes"
    await mon.m.run_once(now=T0 + timedelta(minutes=1))
    assert len(_events(mon.ctx, "market_move")) == 2  # nothing new
    assert mon.m.summary()["spy_change_pct"] == pytest.approx(-2.2, abs=0.01)


async def test_world_etfs_alert_only_from_twice_the_step(mon):
    mon.ctx.config.update({"market": {"world_symbols": ["EWJ", "EWG"]}})
    mon.broker.bar_data["EWJ"] = make_bars([77.0] * 35)
    mon.broker.bar_data["EWG"] = make_bars([41.0] * 35)
    mon.broker.prev_close.update({"EWJ": 78.2, "EWG": 41.9})  # -1.5% and -2.1%
    await mon.m.run_once(now=T0)
    rows = {r["symbol"]: r for r in _events(mon.ctx, "market_move")}
    assert rows["EWJ"]["scope"] == "world" and rows["EWJ"]["level"] == "info" and rows["EWJ"]["alerted"] == 0
    assert rows["EWG"]["level"] == "warn" and rows["EWG"]["alerted"] == 1
    assert mon.alerts.sent[0]["title"] == "Germany (EWG) is now down 2% today"
    assert "holds stocks from Germany" in mon.alerts.sent[0]["message"]


async def test_restart_does_not_repeat_todays_alerts(mon):
    mon.broker.prev_close["SPY"] = 613.5  # down 2.2%
    mon.broker.bar_data["NVDA"] = spike_bars()
    await mon.m.run_once(now=T0)
    assert len(mon.alerts.sent) == 2
    fresh = MarketMonitor(mon.ctx, tickers=make_tickers(mon.ctx.db))
    fresh._load_state(now=T0 + timedelta(minutes=2))
    assert fresh.levels.new_level("SPY", "2026-10-07", -2.3, 1.0) is None  # -1% and -2% were reported
    assert not fresh.memory.is_new(("NVDA", "spike_up"), T0 + timedelta(minutes=2), 5, 4.5, 3.0)
    assert fresh._alerted_at[("NVDA", "spike_up")] == T0


async def test_movers_filtered_and_added_to_watch_list(mon):
    b = mon.broker
    for sym, price, prev in (("TSLA", 262.0, 245.0), ("HALT", 50.0, 40.0), ("OTCX", 9.0, 6.0), ("F", 1.5, 1.2),
                             ("AAPL", 220.0, 231.0)):
        b.prices[sym] = price
        b.prev_close[sym] = prev
    mon.ctx.config.update({"market": {"min_price": 2.0}})
    await mon.m.run_once(now=T0)
    gainers = [m["symbol"] for m in mon.m.movers["gainers"]]
    assert gainers == ["TSLA"]  # HALT isn't tradable, OTCX isn't US-listed, F is under $2
    assert mon.m.movers["gainers"][0]["name"] == "Tesla, Inc."
    assert [m["symbol"] for m in mon.m.movers["losers"]] == ["AAPL"]
    assert mon.m.universe["TSLA"] == ["mover"] and "index" in mon.m.universe["SPY"]
    watching = {w["symbol"]: w for w in mon.m.payload()["watching"]}
    assert watching["TSLA"]["change_pct"] == pytest.approx(6.94, abs=0.01)


async def test_screener_unavailable_is_not_an_error(mon):
    mon.broker.screener_error = "forbidden"
    mon.broker.bar_data["NVDA"] = spike_bars()
    res = await mon.m.run_once(now=T0)
    assert res["status"] == "ok" and res["events"] == 1
    assert mon.m.movers_error and mon.m.payload()["movers"]["error"]
    status = {c["component"]: c for c in mon.ctx.state.components()}["market"]
    assert status["level"] == "ok"


async def test_price_data_failure_sets_warning(mon, monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("rate limited")

    monkeypatch.setattr(mon.broker, "bars_multi", boom)
    res = await mon.m.run_once(now=T0)
    assert res["status"] == "error"
    status = {c["component"]: c for c in mon.ctx.state.components()}["market"]
    assert status["level"] == "warn" and "rate limited" in status["detail"]


async def test_turned_off_or_not_connected(mon):
    mon.ctx.config.update({"market": {"enabled": False}})
    assert (await mon.m.run_once(now=T0))["status"] == "off"
    status = {c["component"]: c for c in mon.ctx.state.components()}["market"]
    assert status["level"] == "off"
    mon.ctx.config.update({"market": {"enabled": True}})
    mon.trader.broker = None
    assert (await mon.m.run_once(now=T0))["status"] == "no_broker"
    assert mon.broker.calls == []


async def test_old_events_pruned_on_start(mon):
    db = mon.ctx.db
    db.insert("market_events", {"ts": iso(utcnow() - timedelta(days=40)), "kind": "spike_up", "symbol": "OLD"})
    db.insert("market_events", {"ts": iso(utcnow() - timedelta(days=2)), "kind": "spike_up", "symbol": "NEW"})
    await mon.m.start()
    await mon.m.stop()
    assert [r["symbol"] for r in load_events(db)] == ["NEW"]


async def test_status_summary_has_monitor(mon):
    from newstrader.api.routes.status import status_summary

    mon.broker.bar_data["NVDA"] = spike_bars()
    await mon.m.run_once(now=T0)
    out = await status_summary(mon.ctx, light=True)
    assert out["monitor"]["enabled"] and out["monitor"]["events_today"] == 1
    assert out["monitor"]["last_scan"] == iso(T0)


# ====================================================================== API
def test_market_api_without_the_service(client):
    r = client.get("/api/market")
    assert r.status_code == 200
    body = r.json()
    assert body["running"] is False and body["events"] == [] and body["connected"] is False
    assert [i["symbol"] for i in body["indices"]] == ["SPY", "QQQ", "IWM", "DIA"]
    assert body["world"][0] == {"symbol": "EWJ", "name": "Japan", "price": None, "prev_close": None,
                                "change_pct": None, "move_15m": None}
    assert client.post("/api/market/scan").status_code == 503
    assert client.get("/api/market/events").json() == {"events": []}


def test_market_api_with_the_service(client, ctx):
    broker = FakeBroker(prices={})
    broker.bar_data["SPY"] = make_bars([600.0] * 35, end=utcnow())
    broker.bar_data["NVDA"] = spike_bars(end=utcnow())
    ctx.services["trader"] = SimpleNamespace(broker=broker, clock={"is_open": True}, positions=[], open_orders=[],
                                             account={"equity": 1})
    ctx.config.update({"market": {"watchlist": ["NVDA"], "index_symbols": ["SPY"], "world_symbols": []}})
    ctx.services["market"] = MarketMonitor(ctx)
    r = client.post("/api/market/scan")
    assert r.status_code == 200 and r.json()["events"] == 1
    body = client.get("/api/market").json()
    assert body["running"] and body["connected"] and body["market"] == {"is_open": True}
    assert body["events"][0]["symbol"] == "NVDA" and body["events"][0]["detail"]["title"].startswith("NVDA jumped")
    assert body["watching"][0]["symbol"] == "NVDA" and body["watching"][0]["reasons"][0] == "watchlist"
    assert body["movers"]["gainers"][0]["symbol"] == "NVDA"  # +4.5% on the (fake) day
    assert body["counts"]["watching"] == 1 and body["thresholds"]["spike_pct"] == 3.0
    ev = client.get("/api/market/events?symbol=nvda&kind=spike&days=1").json()["events"]
    assert len(ev) == 1
    assert client.get("/api/market/events?kind=market_move").json()["events"] == []
    ctx.config.update({"market": {"enabled": False}})
    assert client.post("/api/market/scan").status_code == 409


# ====================================================================== don't chase
def test_chase_check():
    assert chase_check("bullish", 4.2, 3.0).startswith("Price already moved +4.2% toward the signal")
    assert "limit 3%" in chase_check("bullish", 4.2, 3.0)
    assert chase_check("bearish", 5.0, 3.0).startswith("Price already moved -5.0%")
    assert chase_check("bullish", 2.9, 3.0) == ""
    assert chase_check("bullish", -6.0, 3.0) == ""  # moved the other way: not chasing
    assert chase_check("bullish", 9.0, 0) == ""  # 0 = off
    assert chase_check("bullish", None, 3.0) == ""  # no data, no guard
    assert chase_check("neutral", 9.0, 3.0) == ""


@pytest.fixture
async def chase(ctx):
    broker = FakeBroker(prices={"AAPL": 105.0})
    t = Trader(ctx, broker_factory=lambda: broker)
    ctx.services["trader"] = t
    await t.connect()
    yield SimpleNamespace(t=t, broker=broker, ctx=ctx)
    await t.stop()


def _news_signal(ctx, minutes_ago: int = 20, confidence: int = 90, direction: str = "bullish") -> dict:
    now = utcnow()
    news_id = ctx.db.insert("news_items", {"source_id": "t", "title": "Apple beats", "status": "analysed",
                                           "published_at": iso(now - timedelta(minutes=minutes_ago)),
                                           "received_at": iso(now - timedelta(minutes=minutes_ago - 1))})
    analysis_id = ctx.db.insert("analyses", {"created_at": iso(now), "item_kind": "news", "item_id": news_id,
                                             "status": "ok"})
    sig = {"analysis_id": analysis_id, "created_at": iso(now), "ticker": "AAPL", "direction": direction,
           "confidence": confidence, "reasoning": "Beat estimates.", "headline": "Apple beats"}
    sig["id"] = ctx.db.insert("signals", sig)
    return sig


def _moved(broker, before: float, after: float, minutes_since: int = 10, total: int = 120) -> None:
    """Flat at `before`, then at `after` for the last few minutes."""
    now = utcnow()
    prices = [before] * (total - minutes_since) + [after] * minutes_since
    broker.bar_data["AAPL"] = make_bars(prices, end=now)
    broker.prices["AAPL"] = after


async def test_chase_guard_sends_to_review(chase):
    _moved(chase.broker, 100.0, 105.0)
    sig = _news_signal(chase.ctx)
    res = await chase.t.handle_signal(sig)
    assert res["action"] == "review" and not res["traded"]
    assert res["reason"].startswith("Price already moved +5.0% toward the signal since the news (limit 3%)")
    assert not any(c[0] == "submit_bracket" for c in chase.broker.calls)
    row = chase.ctx.db.query_one("SELECT pre_move_pct FROM signals WHERE id = ?", (sig["id"],))
    assert row["pre_move_pct"] == pytest.approx(5.0)


async def test_chase_guard_off_and_opposite_move(chase):
    _moved(chase.broker, 100.0, 105.0)
    chase.ctx.config.update({"trading": {"max_chase_pct": 0}})
    assert (await chase.t.handle_signal(_news_signal(chase.ctx)))["action"] == "bought"
    chase.ctx.config.update({"trading": {"max_chase_pct": 3.0}, "risk": {"ticker_cooldown_minutes": 0}})
    _moved(chase.broker, 100.0, 95.0)  # a bullish signal after the price fell: not chasing
    sig = _news_signal(chase.ctx)
    res = await chase.t.handle_signal(sig)
    assert res["action"] == "bought", res
    row = chase.ctx.db.query_one("SELECT pre_move_pct FROM signals WHERE id = ?", (sig["id"],))
    assert row["pre_move_pct"] == pytest.approx(-5.0)


async def test_chase_guard_bypassed_by_manual_approval(chase):
    _moved(chase.broker, 100.0, 105.0)
    res = await chase.t.handle_signal(_news_signal(chase.ctx), manual=True)
    assert res["action"] == "bought" and res["traded"]


async def test_chase_guard_looks_back_at_most_an_hour(chase):
    # 3-hour-old news, price jumped 35 minutes ago: inside the hour, so it counts
    _moved(chase.broker, 100.0, 105.0, minutes_since=35)
    assert (await chase.t.handle_signal(_news_signal(chase.ctx, minutes_ago=180)))["action"] == "review"
    # ...but a jump 90 minutes ago is older than the hour we look back: the price an hour ago was already 105
    _moved(chase.broker, 100.0, 105.0, minutes_since=90)
    assert (await chase.t.handle_signal(_news_signal(chase.ctx, minutes_ago=180)))["action"] == "bought"


async def test_chase_guard_uses_the_monitors_recent_bars(chase):
    chase.broker.prices["AAPL"] = 105.0  # the fake's own bars are flat at 105
    chase.ctx.services["market"] = SimpleNamespace(price_near=lambda sym, when, now=None: 100.0)
    res = await chase.t.handle_signal(_news_signal(chase.ctx))
    assert res["action"] == "review"


async def test_chase_guard_skipped_without_price_history(chase):
    chase.broker.bar_data["AAPL"] = []  # no bars at all -> no guard
    res = await chase.t.handle_signal(_news_signal(chase.ctx))
    assert res["action"] == "bought"
