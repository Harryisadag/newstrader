from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from newstrader.backtest.runner import BacktestParams, BacktestRunner, estimate, hold_until
from newstrader.backtest.simulator import simulate_bracket
from newstrader.db import iso
from newstrader.performance.prices import directional_return, next_session_same_time, price_at
from newstrader.performance.stats import bucket_of, compute_stats
from newstrader.performance.tracker import PerformanceTracker

T0 = datetime(2026, 10, 6, 14, 0, tzinfo=UTC)  # Tuesday 10:00 ET


def bar(minute, o, h, low, c, base=T0):
    return {"t": iso(base + timedelta(minutes=minute)), "o": o, "h": h, "l": low, "c": c, "v": 100}


# ---------------------------------------------------------------- simulator
def test_long_hits_take_profit():
    bars = [bar(1, 100, 100.5, 99.8, 100.2), bar(2, 100.2, 104.5, 100, 104), bar(3, 104, 105, 103, 104)]
    r = simulate_bracket("long", bars, T0, 2, 4, T0 + timedelta(hours=6))
    assert r.entered and r.entry_price == 100 and r.exit_reason == "take-profit"
    assert r.exit_price == pytest.approx(104) and r.ret_pct == pytest.approx(4.0)
    assert r.pnl(10) == pytest.approx(40.0)


def test_long_hits_stop():
    bars = [bar(1, 100, 100.2, 99.5, 99.8), bar(2, 99.8, 99.9, 97.5, 98)]
    r = simulate_bracket("long", bars, T0, 2, 4, T0 + timedelta(hours=6))
    assert r.exit_reason == "stop-loss" and r.ret_pct == pytest.approx(-2.0)


def test_both_in_same_bar_assumes_stop_first():
    bars = [bar(1, 100, 100.1, 99.9, 100), bar(2, 100, 105, 97, 101)]
    r = simulate_bracket("long", bars, T0, 2, 4, T0 + timedelta(hours=6))
    assert r.exit_reason == "stop-loss"


def test_gap_through_stop_exits_at_open():
    bars = [bar(1, 100, 100.1, 99.9, 100), bar(2, 95, 96, 94, 95)]
    r = simulate_bracket("long", bars, T0, 2, 4, T0 + timedelta(hours=6))
    assert r.exit_reason.startswith("stop-loss (gapped") and r.exit_price == 95


def test_time_limit_closes_at_last_price():
    bars = [bar(i, 100, 100.5, 99.5, 100 + i * 0.01) for i in range(1, 30)]
    r = simulate_bracket("long", bars, T0, 2, 4, T0 + timedelta(minutes=10))
    assert r.exit_reason.startswith("time limit") and r.exit_price == pytest.approx(100.09)


def test_short_trade():
    bars = [bar(1, 50, 50.1, 49.9, 50), bar(2, 50, 50.2, 47.9, 48)]
    r = simulate_bracket("short", bars, T0, 2, 4, T0 + timedelta(hours=6))
    assert r.exit_reason == "take-profit" and r.ret_pct == pytest.approx(4.0)


def test_no_bars_after_news():
    r = simulate_bracket("long", [bar(-5, 1, 1, 1, 1)], T0, 2, 4, T0 + timedelta(hours=1))
    assert not r.entered


def test_hold_until():
    close = T0 + timedelta(hours=6)
    assert hold_until("eod", T0, close) == close
    assert hold_until("2", T0, close) == T0 + timedelta(hours=2)
    assert hold_until("10", T0, close) == close


# ---------------------------------------------------------------- prices + stats
def test_price_at_and_directional_return():
    bars = [bar(0, 10, 10, 10, 10), bar(1, 10, 11, 10, 11), bar(5, 11, 12, 11, 12)]
    assert price_at(bars, T0 + timedelta(minutes=3)) == 11
    assert price_at(bars, T0 - timedelta(minutes=1)) is None
    assert directional_return("bullish", 10, 11) == pytest.approx(10)
    assert directional_return("bearish", 10, 11) == pytest.approx(-10)
    assert directional_return("neutral", 10, 11) is None


def test_next_trading_day_skips_weekend_and_clamps():
    sessions = [{"date": "2026-10-09", "open": "2026-10-09T13:30:00Z", "close": "2026-10-09T20:00:00Z"},
                {"date": "2026-10-12", "open": "2026-10-12T13:30:00Z", "close": "2026-10-12T20:00:00Z"}]
    friday_10am = datetime(2026, 10, 9, 14, 0, tzinfo=UTC)
    assert next_session_same_time(friday_10am, sessions) == datetime(2026, 10, 12, 14, 0, tzinfo=UTC)
    friday_7pm_et = datetime(2026, 10, 9, 23, 0, tzinfo=UTC)
    assert next_session_same_time(friday_7pm_et, sessions) == datetime(2026, 10, 12, 20, 0, tzinfo=UTC)


def test_stats_by_confidence_and_source():
    rows = [
        {"direction": "bullish", "confidence": 92, "ret_1h": 1.0, "source_name": "A", "source_type": "rss", "traded": 1},
        {"direction": "bullish", "confidence": 85, "ret_1h": -0.5, "source_name": "A", "source_type": "rss", "traded": 1},
        {"direction": "bearish", "confidence": 65, "ret_1h": 0.3, "source_name": "B", "source_type": "stream", "traded": 0},
        {"direction": "bullish", "confidence": 70, "ret_1h": None, "source_name": "B", "source_type": "stream", "traded": 0},
        {"direction": "neutral", "confidence": 99, "ret_1h": 5.0, "source_name": "C", "source_type": "rss", "traded": 0},
    ]
    s = compute_stats(rows, "1h")
    assert s["overall"]["count"] == 3 and s["overall"]["wins"] == 2
    b = {g["key"]: g for g in s["by_confidence"]}
    assert b["90-100"]["win_rate"] == 100 and b["80-89"]["win_rate"] == 0 and b["60-69"]["count"] == 1
    assert b["70-79"]["count"] == 0
    assert {g["key"]: g["count"] for g in s["by_source"]} == {"A": 2, "B": 1}
    assert s["pending"] == 1
    assert bucket_of(59) == "under 60" and bucket_of(100) == "90-100"


# ---------------------------------------------------------------- live tracker
async def test_tracker_records_checkpoints(ctx):
    from newstrader.trading.fake_broker import FakeBroker

    broker = FakeBroker(prices={"NVDA": 100.0})
    t0 = datetime.now(UTC) - timedelta(days=3)
    broker.bar_data["NVDA"] = [bar(m, 100 + m * 0.01, 100 + m * 0.01, 100 + m * 0.01, 100 + m * 0.01, base=t0)
                               for m in range(0, 60 * 24 * 2)]
    ctx.services["trader"] = type("T", (), {"broker": broker})()
    sid = ctx.db.insert("signals", {"created_at": iso(t0), "ticker": "NVDA", "direction": "bullish",
                                    "confidence": 88, "source_name": "X", "source_type": "rss"})
    ctx.db.insert("signals", {"created_at": iso(t0), "ticker": "NVDA", "direction": "neutral", "confidence": 50})
    tr = PerformanceTracker(ctx)
    await tr.run_once()
    row = ctx.db.query_one("SELECT * FROM signal_prices WHERE signal_id = ?", (sid,))
    assert row["price_t0"] == pytest.approx(100.0)
    assert row["price_5m"] == pytest.approx(100.05) and row["ret_5m"] == pytest.approx(0.05)
    assert row["price_1h"] == pytest.approx(100.6)
    assert row["price_1d"] is not None and row["status"] == "complete"
    assert ctx.db.scalar("SELECT COUNT(*) FROM signal_prices") == 1  # neutral signals aren't tracked
    assert ctx.db.query_one("SELECT price_at_signal FROM signals WHERE id = ?", (sid,))["price_at_signal"] == 100.0


async def test_performance_api(client, ctx):
    sid = ctx.db.insert("signals", {"created_at": iso(), "ticker": "NVDA", "direction": "bullish", "confidence": 91,
                                    "source_name": "CNBC", "source_type": "rss", "traded": 1})
    ctx.db.insert("signal_prices", {"signal_id": sid, "ticker": "NVDA", "direction": "bullish", "t0": iso(),
                                    "price_t0": 100, "price_1h": 101, "ret_1h": 1.0, "status": "pending"})
    r = client.get("/api/performance?horizon=1h&days=7").json()
    assert r["overall"]["count"] == 1 and r["overall"]["win_rate"] == 100
    assert client.get("/api/performance?horizon=2h").status_code == 400


# ---------------------------------------------------------------- backtest
async def test_backtest_end_to_end(ctx):
    from newstrader.ai.claude_client import ClaudeAnalyzer
    from newstrader.ai.pipeline import Pipeline
    from newstrader.trading.fake_broker import FakeBroker

    from .helpers import FakeClaude, make_tickers, sig, signal_json

    broker = FakeBroker(prices={"NVDA": 100.0, "TSLA": 200.0})
    news_t = datetime(2026, 10, 6, 14, 0, tzinfo=UTC)  # inside the fake session (13:30-20:00Z)
    broker.news_items = [
        {"id": 1, "headline": "Nvidia wins huge contract", "summary": "", "content": "", "symbols": ["NVDA"],
         "source": "benzinga", "created_at": news_t, "url": "https://x.com/a-long-path-1", "author": ""},
        {"id": 2, "headline": "Tesla recall announced", "summary": "", "content": "", "symbols": ["TSLA"],
         "source": "benzinga", "created_at": news_t + timedelta(minutes=30), "url": "https://x.com/a-long-path-2", "author": ""},
    ]
    broker.bar_data["NVDA"] = [bar(m, 100 + m * 0.1, 100 + m * 0.1 + 0.05, 100 + m * 0.1 - 0.05, 100 + m * 0.1, base=news_t)
                               for m in range(0, 120)]
    broker.bar_data["TSLA"] = [bar(m, 200, 200.5, 199.5, 200, base=news_t) for m in range(0, 400)]
    ctx.services["trader"] = type("T", (), {"broker": broker})()
    fake = FakeClaude([signal_json(sig("NVDA", confidence=90)),
                       signal_json(sig("TSLA", direction="bearish", confidence=85))])
    analyzer = ClaudeAnalyzer(ctx, client_factory=lambda: fake)
    pipeline = Pipeline(ctx, analyzer=analyzer, tickers=make_tickers(ctx.db))
    ctx.services["pipeline"] = pipeline
    runner = BacktestRunner(ctx)
    params = BacktestParams(start=date(2026, 10, 6), end=date(2026, 10, 6), max_articles=10, budget_usd=5,
                            engine="claude")
    run_id = ctx.db.insert("backtest_runs", {"created_at": iso(), "params": "{}", "status": "running"})
    summary = await runner.run(run_id, params)
    assert summary["analysed"] == 2 and summary["signals"] == 2
    assert summary["trades"] == 1 and summary["wins"] == 1  # NVDA climbs 0.1/min -> +4% target hit
    rows = {r["ticker"]: r for r in ctx.db.query("SELECT * FROM backtest_results WHERE run_id = ?", (run_id,))}
    assert rows["NVDA"]["action"] == "bought" and rows["NVDA"]["exit_reason"] == "take-profit"
    assert rows["TSLA"]["action"].startswith("bearish - shorting off")
    assert ctx.db.scalar("SELECT COUNT(*) FROM analyses WHERE is_backtest = 1") == 2
    # backtest spend doesn't count toward the live daily cap
    from newstrader.ai.costs import spend_today

    assert spend_today(ctx.db) == 0
    assert fake.requests[0]["messages"][0]["content"].find("2026-10-06") != -1  # "current time" = news time


def test_estimate():
    e = estimate(BacktestParams(start=date(2026, 1, 1), end=date(2026, 1, 2), max_articles=100), "claude-sonnet-5-5")
    assert 0 < e["per_article"] < 0.05 and e["max_cost"] == pytest.approx(e["per_article"] * 100, rel=0.01)


def test_backtest_api_validation(client):
    assert client.post("/api/backtest/estimate", json={"start": "2026-10-05", "end": "2026-10-01"}).status_code == 400
    assert client.post("/api/backtest/estimate", json={"start": "x", "end": "y"}).status_code == 400
    ok = client.post("/api/backtest/estimate", json={"start": "2026-09-01", "end": "2026-09-05", "max_articles": 20})
    assert ok.status_code == 200 and "warning" in ok.json()
    # running requires explicit confirmation of the cost
    assert client.post("/api/backtest/run", json={"start": "2026-09-01", "end": "2026-09-05"}).status_code == 400
