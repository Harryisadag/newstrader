"""End-to-end pipeline with a fake Claude and a fake broker: news in -> signal -> (paper) trade."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from newstrader.ai.claude_client import ClaudeAnalyzer
from newstrader.ai.costs import estimate_cost, price_for
from newstrader.ai.pipeline import Pipeline
from newstrader.sources.base import NewsItem
from newstrader.trading.fake_broker import FakeBroker
from newstrader.trading.trader import Trader

from .helpers import FakeClaude, make_tickers, sig, signal_json


@pytest.fixture
async def env(ctx):
    broker = FakeBroker(prices={"NVDA": 180.0, "AAPL": 230.0, "TSLA": 250.0, "JPM": 300.0})
    trader = Trader(ctx, broker_factory=lambda: broker)
    ctx.services["trader"] = trader
    await trader.connect()
    fake = FakeClaude()
    pipeline = Pipeline(ctx, analyzer=ClaudeAnalyzer(ctx, client_factory=lambda: fake), tickers=make_tickers(ctx.db))
    ctx.services["pipeline"] = pipeline
    yield ctx, pipeline, fake, broker
    await trader.stop()


def item(title="Nvidia wins $10 billion government AI contract", source="cnbc", ext=None, **kw):
    return NewsItem(source_id=source, source_type="rss", source_name=source.upper(), external_id=ext or title,
                    title=title, url=kw.pop("url", f"https://example.com/{source}/{abs(hash(title))}"),
                    published_at=datetime.now(UTC), **kw)


async def run(pipeline, it):
    res = await pipeline.submit(it)
    if res["status"] == "queued":
        _, queued_item, pre = pipeline.queue.get_nowait()
        pipeline.queue.task_done()
        return await pipeline.analyze_item(queued_item, pre)
    return res


async def test_bullish_news_trades(env):
    ctx, pipeline, fake, broker = env
    fake.responses = [signal_json(sig("NVDA", confidence=90))]
    out = await run(pipeline, item())
    assert out["status"] == "ok"
    s = out["signals"][0]
    assert s["traded"] == 1 and s["action"] == "bought"
    assert broker.calls[-1][:3] == ("submit_bracket", "NVDA", "buy")
    a = ctx.db.query_one("SELECT * FROM analyses")
    assert a["status"] == "ok" and a["cost_usd"] > 0


async def test_request_shape(env):
    ctx, pipeline, fake, _ = env
    fake.responses = [signal_json()]
    await run(pipeline, item())
    req = fake.requests[0]
    assert req["model"] == "claude-sonnet-5-5"
    assert req["output_config"]["format"]["type"] == "json_schema"
    assert req["output_config"]["effort"] == "low"
    assert req["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert req["betas"] == ["server-side-fallback-2026-07-01"] and req["fallbacks"] == "default"
    assert "thinking" not in req and "temperature" not in req
    assert "NVDA" in req["messages"][0]["content"]  # candidate tickers passed along


async def test_haiku_gets_no_effort_or_fallback(env):
    ctx, pipeline, fake, _ = env
    ctx.config.update({"ai": {"model": "claude-haiku-4-5-20251001"}})
    fake.responses = [signal_json()]
    await run(pipeline, item())
    req = fake.requests[0]
    assert "effort" not in req["output_config"] and "betas" not in req


async def test_review_band_alerts_instead_of_trading(env):
    ctx, pipeline, fake, broker = env
    fake.responses = [signal_json(sig("NVDA", confidence=70))]
    out = await run(pipeline, item())
    s = out["signals"][0]
    assert s["action"] == "review" and not s["traded"]
    assert not broker.calls
    res = await pipeline.approve(s["id"])
    assert res["traded"]


async def test_filtered_news_never_reaches_claude(env):
    _, pipeline, fake, _ = env
    out = await pipeline.submit(item("Local bakery wins croissant award"))
    assert out["status"] == "filtered"
    assert not fake.requests


async def test_same_item_twice_is_ignored(env):
    _, pipeline, _, _ = env
    await pipeline.submit(item(ext="x1"))
    assert (await pipeline.submit(item(ext="x1")))["status"] == "seen"


async def test_same_story_other_source_is_duplicate(env):
    ctx, pipeline, fake, _ = env
    fake.responses = [signal_json(sig("NVDA", confidence=90))]
    await run(pipeline, item("Nvidia wins $10 billion government AI contract", source="cnbc"))
    out = await pipeline.submit(item("NVIDIA wins $10 billion government AI contract", source="reuters"))
    assert out["status"] == "duplicate"
    assert len(fake.requests) == 1
    sig_row = ctx.db.query_one("SELECT * FROM signals")
    assert sig_row["corroborations"] == 2


async def test_same_signal_from_different_story_is_merged_not_retraded(env):
    ctx, pipeline, fake, broker = env
    fake.responses = [signal_json(sig("NVDA", confidence=90)), signal_json(sig("NVDA", confidence=95))]
    await run(pipeline, item("Nvidia wins huge contract", source="a"))
    out = await run(pipeline, item("Pentagon picks Nvidia chips for new program", source="b"))
    s = out["signals"][0]
    assert s["action"] == "merged"
    assert len([c for c in broker.calls if c[0] == "submit_bracket"]) == 1
    top = ctx.db.query_one("SELECT * FROM signals WHERE merged_into IS NULL")
    assert top["corroborations"] == 2 and top["confidence"] == 95


async def test_merged_confirmation_can_upgrade_review_to_trade(env):
    ctx, pipeline, fake, broker = env
    fake.responses = [signal_json(sig("NVDA", confidence=70)), signal_json(sig("NVDA", confidence=88))]
    await run(pipeline, item("Nvidia rumored to win contract", source="a"))
    assert not broker.calls
    await run(pipeline, item("Pentagon confirms Nvidia contract", source="b"))
    assert broker.calls and broker.calls[-1][1] == "NVDA"


async def test_invalid_ticker_rejected_and_logged(env):
    ctx, pipeline, fake, broker = env
    fake.responses = [signal_json(sig("FAKE", confidence=99))]
    out = await run(pipeline, item())
    assert out["status"] == "rejected" and not out["signals"]
    row = ctx.db.query_one("SELECT * FROM analyses")
    assert row["status"] == "rejected" and "FAKE" in row["error"]
    assert not broker.calls


async def test_malformed_json_rejected(env):
    ctx, pipeline, fake, broker = env
    fake.responses = ["I think NVDA goes up!"]
    out = await run(pipeline, item())
    assert out["status"] == "rejected"
    assert not broker.calls


async def test_refusal_is_recorded(env):
    ctx, pipeline, fake, broker = env
    fake.stop_reason = "refusal"
    out = await run(pipeline, item())
    assert out["status"] == "refusal"
    assert "declined" in ctx.db.query_one("SELECT error FROM analyses")["error"]


async def test_api_error_is_recorded(env):
    ctx, pipeline, fake, _ = env
    fake.error = RuntimeError("boom")
    out = await run(pipeline, item())
    assert out["status"] == "error"


async def test_spend_cap_stops_calls(env):
    ctx, pipeline, fake, _ = env
    ctx.config.update({"ai": {"daily_spend_cap_usd": 0.0}})
    out = await pipeline.submit(item())
    assert out["status"] == "skipped_cap"
    assert not fake.requests


async def test_neutral_signal_not_traded(env):
    ctx, pipeline, fake, broker = env
    fake.responses = [signal_json(sig("NVDA", direction="neutral", confidence=95))]
    out = await run(pipeline, item())
    assert out["signals"][0]["action"] == "ignored"
    assert not broker.calls


async def test_old_articles_are_not_analysed(env):
    _, pipeline, fake, _ = env
    it = item()
    it.published_at = datetime.now(UTC) - timedelta(hours=5)
    assert (await pipeline.submit(it))["status"] == "stale"


def test_cost_estimates():
    assert price_for("claude-haiku-4-5-20251001") == price_for("claude-haiku-4-5")
    cost = estimate_cost("claude-sonnet-5-5", input_tokens=1_000_000, output_tokens=100_000)
    assert cost == pytest.approx(2.0 + 1.0)
    assert estimate_cost("claude-unknown-9", 1_000_000) >= 4.0  # unknown -> priced high, never under-counted


async def test_no_claude_calls_without_ticker_list(ctx):
    from newstrader.ai.tickers import TickerTable

    fake = FakeClaude()
    p = Pipeline(ctx, analyzer=ClaudeAnalyzer(ctx, client_factory=lambda: fake), tickers=TickerTable(ctx.db))
    out = await p.submit(item("Fed signals a rate cut in December"))
    assert out["status"] == "no_tickers" and not fake.requests
