"""v0.4 speed pack: the news-to-order timer, faster feed checks (polite to websites) and faster TV analysis."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import UTC, datetime, timedelta, timezone
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from newstrader.ai.claude_client import ClaudeAnalyzer
from newstrader.ai.pipeline import Pipeline
from newstrader.audio.stream_manager import Line, StreamManager, StreamWorker, build_item
from newstrader.audio.transcriber import Job, Segment, Transcriber
from newstrader.config import AppSettings, ConfigStore, SourceConfig
from newstrader.db import MIGRATIONS, Database, iso
from newstrader.performance.speed import breakdown, gap_seconds, signal_speed, speed_summary, to_utc
from newstrader.sources import presets, rss
from newstrader.sources.base import NewsItem
from newstrader.sources.polling import MAX_RETRY_AFTER, SlowDown, next_wait, retry_after_seconds
from newstrader.sources.rss import SEC_USER_AGENT, USER_AGENT, FeedPoller, user_agent_for
from newstrader.sources.x_api import XAccountSource
from newstrader.trading.fake_broker import FakeBroker
from newstrader.trading.trader import Trader

from .helpers import FakeClaude, make_tickers, sig, signal_json

T0 = datetime(2026, 10, 8, 14, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------- migration 10
def test_migration_10_adds_the_timer_to_an_old_database(tmp_path):
    path = tmp_path / "old.db"
    c = sqlite3.connect(path)
    for i, script in enumerate(MIGRATIONS[:9], start=1):  # a v0.3 database
        c.executescript(script)
        c.execute(f"PRAGMA user_version={i}")
    c.execute("INSERT INTO news_items (id, title, published_at, received_at) VALUES "
              "(1, 'Nvidia wins', '2026-10-08T13:59:20.000Z', '2026-10-08T14:00:00.000Z'), "
              "(2, 'TV clip', '2026-10-08T14:00:00.000Z', '2026-10-08T14:00:30.000Z')")
    c.execute("INSERT INTO analyses (id, item_kind, item_id) VALUES (1, 'news', 1), (2, 'transcript', 2)")
    c.execute("INSERT INTO signals (id, analysis_id, created_at, ticker) VALUES "
              "(1, 1, '2026-10-08T14:00:02.000Z', 'NVDA'), (2, 2, '2026-10-08T14:00:31.000Z', 'AAPL'), "
              "(3, NULL, '2026-10-08T14:01:00.000Z', 'TSLA')")
    c.commit()
    c.close()

    db = Database(path)
    try:
        assert db.scalar("PRAGMA user_version") >= 10
        rows = {r["id"]: r for r in db.query("SELECT * FROM signals")}
        assert rows[1]["news_published_at"] == "2026-10-08T13:59:20.000Z"
        assert rows[1]["news_received_at"] == "2026-10-08T14:00:00.000Z"
        assert rows[1]["decided_at"] == rows[1]["created_at"]
        # an old TV clip only saved when it started, not when its newest words were spoken: left empty
        assert rows[2]["news_published_at"] is None and rows[2]["news_received_at"] == "2026-10-08T14:00:30.000Z"
        assert rows[3]["news_received_at"] is None and rows[3]["decided_at"] == "2026-10-08T14:01:00.000Z"
        assert signal_speed(rows[1])["feed_delay_s"] == 40
    finally:
        db.close()


# ---------------------------------------------------------------- the pure helper
def test_gap_handles_missing_naive_and_other_time_zones():
    assert gap_seconds("2026-10-08T14:00:05Z", "2026-10-08T14:00:00.000Z") == 5
    assert gap_seconds(T0, None) is None and gap_seconds(None, T0) is None
    assert gap_seconds("", T0) is None and gap_seconds("not a time", T0) is None
    naive = datetime(2026, 10, 8, 14, 0, 10)  # no time zone: taken as UTC
    assert gap_seconds(naive, T0) == 10
    assert gap_seconds("2026-10-08T14:00:10", T0) == 10
    new_york = datetime(2026, 10, 8, 10, 0, 30, tzinfo=timezone(timedelta(hours=-4)))
    assert gap_seconds(new_york, T0) == 30
    assert to_utc("2026-10-08T10:00:00-04:00") == T0


def test_small_clock_differences_count_as_zero_but_big_ones_are_dropped():
    assert gap_seconds(T0, T0 + timedelta(seconds=3)) == 0.0  # the website's clock is 3 s ahead
    assert gap_seconds(T0, T0 + timedelta(seconds=5)) == 0.0
    assert gap_seconds(T0, T0 + timedelta(seconds=6)) is None
    assert gap_seconds(T0, T0 + timedelta(hours=4)) is None  # "published" in the future: a wrong time zone


def test_breakdown_of_each_step():
    sp = breakdown(T0, T0 + timedelta(seconds=38), T0 + timedelta(seconds=39.2), T0 + timedelta(seconds=40.1))
    assert sp == {"feed_delay_s": 38, "thinking_s": 1.2, "order_s": 0.9, "news_to_order_s": 2.1}
    sp = breakdown(None, T0, T0 + timedelta(seconds=1))  # no published time, not traded
    assert sp["feed_delay_s"] is None and sp["thinking_s"] == 1 and sp["order_s"] is None
    assert sp["news_to_order_s"] is None
    # the broker's clock 2 s behind ours: still a sensible total
    sp = breakdown(T0, T0, T0 + timedelta(seconds=1), T0 - timedelta(seconds=1))
    assert sp["order_s"] == 0.0 and sp["news_to_order_s"] == 0.0


def test_signal_speed_marks_hand_approved_orders():
    row = {"news_published_at": iso(T0), "news_received_at": iso(T0 + timedelta(seconds=2)),
           "created_at": iso(T0 + timedelta(seconds=3)), "review_status": "approved"}
    sp = signal_speed(row, iso(T0 + timedelta(minutes=5)))
    assert sp["manual"] and sp["thinking_s"] == 1  # decided_at missing: created_at is used
    assert not signal_speed({**row, "review_status": None})["manual"]


def test_speed_summary():
    def traded(news_to_order, approved=False):
        return {"news_received_at": iso(T0), "decided_at": iso(T0 + timedelta(seconds=1)),
                "order_submitted_at": iso(T0 + timedelta(seconds=news_to_order)),
                "review_status": "approved" if approved else None}

    delays = [("Slow site", 600.0)] * 12 + [("Quick wire", 2.0)] * 12 + [("Odd clock", -2.0)] * 10
    delays += [("Rare", 9000.0)] * 2 + [("Quick wire", -60.0), ("Quick wire", None)]
    out = speed_summary([traded(2), traded(3), traded(10), traded(600, approved=True)], delays, min_count=10)
    assert out["orders"] == 3 and out["median_news_to_order_s"] == 3
    assert out["median_thinking_s"] == 1 and out["median_order_s"] == 2
    by = {s["source"]: s for s in out["by_source"]}
    assert by["Odd clock"]["median_feed_delay_s"] == 0  # a small clock difference counts as 0
    assert by["Quick wire"]["count"] == 12  # the clearly wrong times are left out
    assert by["Rare"]["few"] and by["Rare"]["median_feed_delay_s"] == 9000
    assert [s["source"] for s in out["slowest"]] == ["Slow site", "Quick wire", "Odd clock"]  # "Rare" has too few
    empty = speed_summary([], [])
    assert empty["median_news_to_order_s"] is None and empty["slowest"] == [] and empty["orders"] == 0


# ---------------------------------------------------------------- recorded by the pipeline, shown by the API
@pytest.fixture
async def trading(ctx):
    ctx.config.update({"ai": {"engine": "claude"}})
    broker = FakeBroker(prices={"NVDA": 180.0, "AAPL": 230.0})
    trader = Trader(ctx, broker_factory=lambda: broker)
    ctx.services["trader"] = trader
    await trader.connect()
    fake = FakeClaude()
    pipeline = Pipeline(ctx, analyzer=ClaudeAnalyzer(ctx, client_factory=lambda: fake), tickers=make_tickers(ctx.db))
    ctx.services["pipeline"] = pipeline
    yield pipeline, fake
    await trader.stop()


async def _run(pipeline, item):
    assert (await pipeline.submit(item))["status"] == "queued"
    _, queued, pre = pipeline.queue.get_nowait()
    pipeline.queue.task_done()
    return await pipeline.analyze_item(queued, pre)


async def test_pipeline_times_a_traded_signal(client, ctx, trading):
    pipeline, fake = trading
    ctx.bus.bind_loop(asyncio.get_running_loop())
    events = ctx.bus.subscribe()
    fake.responses = [signal_json(sig("NVDA", confidence=90))]
    published = datetime.now(UTC) - timedelta(seconds=40)
    await _run(pipeline, NewsItem(source_id="cnbc", source_type="rss", source_name="CNBC", external_id="n1",
                                  title="Nvidia wins $10 billion government AI contract", published_at=published))
    row = ctx.db.query_one("SELECT * FROM signals")
    assert row["traded"] and row["news_published_at"] == iso(published)
    assert 39 <= gap_seconds(row["news_received_at"], row["news_published_at"]) < 45
    assert gap_seconds(row["decided_at"], row["news_received_at"]) is not None

    msgs = []
    while not events.empty():
        msgs.append(events.get_nowait())
    live = next(m["data"] for m in msgs if m["type"] == "signal")
    assert live["speed"]["news_to_order_s"] is not None  # the Signals tab gets it straight away too

    sp = client.get("/api/signals").json()["signals"][0]["speed"]
    assert sp["feed_delay_s"] >= 39 and sp["thinking_s"] is not None and not sp["manual"]
    assert sp["news_to_order_s"] is not None and sp["order_s"] is not None
    assert client.get(f"/api/signals/{row['id']}").json()["signal"]["speed"]["feed_delay_s"] >= 39
    trades = client.get("/api/trades").json()["trades"]
    entry = next(t for t in trades if t["intent"] == "open_long")
    assert entry["speed"]["news_to_order_s"] == sp["news_to_order_s"]
    assert "news_received_at" not in entry
    assert all(t["speed"] is None for t in trades if t["intent"] in ("take_profit", "stop_loss"))


async def test_tv_news_time_is_when_the_words_were_spoken(ctx, trading):
    pipeline, fake = trading
    fake.responses = [signal_json(sig("AAPL", confidence=65))]
    src = SourceConfig(id="test-tv", type="stream", name="Test TV", url="https://www.youtube.com/@test/live")
    start = datetime.now(UTC).timestamp() - 30
    item = build_item(src, [Line(1, start - 20, start - 15, "earlier talk")],
                      [Line(2, start, start + 4, "Apple is talking about a buyout"), Line(3, start + 5, start + 9, "of a chip firm")])
    assert item.news_time == datetime.fromtimestamp(start + 9, UTC)  # the end of the newest line
    assert item.published_at == datetime.fromtimestamp(start, UTC)  # unchanged: what the AI is told
    await _run(pipeline, item)
    row = ctx.db.query_one("SELECT * FROM signals")
    assert row["news_published_at"] == iso(datetime.fromtimestamp(start + 9, UTC))
    assert 19 <= signal_speed(row)["feed_delay_s"] < 30


async def test_speed_api(client, ctx):
    now = datetime.now(UTC)
    for i in range(12):
        ctx.db.insert("news_items", {"source_id": "slow", "source_type": "rss", "source_name": "Slow Site",
                                     "external_id": f"s{i}", "published_at": iso(now - timedelta(minutes=10)),
                                     "received_at": iso(now)})
        ctx.db.insert("news_items", {"source_id": "wire", "source_type": "rss", "source_name": "Quick Wire",
                                     "external_id": f"w{i}", "published_at": iso(now - timedelta(seconds=4)),
                                     "received_at": iso(now)})
    ctx.db.insert("news_items", {"source_id": "old", "source_type": "rss", "source_name": "Last Month",
                                 "external_id": "o1", "published_at": iso(now - timedelta(days=40, minutes=5)),
                                 "received_at": iso(now - timedelta(days=40))})
    oid = ctx.db.insert("orders", {"alpaca_order_id": "a1", "symbol": "NVDA", "intent": "open_long",
                                   "submitted_at": iso(now - timedelta(seconds=7))})
    ctx.db.insert("signals", {"created_at": iso(now - timedelta(seconds=8)), "ticker": "NVDA", "traded": 1,
                              "order_id": oid, "news_published_at": iso(now - timedelta(seconds=40)),
                              "news_received_at": iso(now - timedelta(seconds=10)),
                              "decided_at": iso(now - timedelta(seconds=8)), "source_type": "stream",
                              "source_name": "Bloomberg TV"})
    r = client.get("/api/performance/speed").json()
    assert r["days"] == 30 and r["orders"] == 1
    assert r["median_news_to_order_s"] == pytest.approx(3, abs=0.01)
    assert r["median_thinking_s"] == pytest.approx(2, abs=0.01)
    by = {s["source"]: s for s in r["by_source"]}
    assert by["Slow Site"]["median_feed_delay_s"] == pytest.approx(600, abs=0.5)
    assert by["Quick Wire"]["median_feed_delay_s"] == pytest.approx(4, abs=0.5)
    assert by["Bloomberg TV"]["median_feed_delay_s"] == pytest.approx(30, abs=0.5) and by["Bloomberg TV"]["few"]
    assert "Last Month" not in by  # older than 30 days
    assert [s["source"] for s in r["slowest"]] == ["Slow Site", "Quick Wire"]


# ---------------------------------------------------------------- faster, polite feed checks
def test_retry_after_and_waits():
    assert retry_after_seconds({"Retry-After": "120"}) == 120
    assert retry_after_seconds({}) is None and retry_after_seconds({"Retry-After": "soon"}) is None
    at = datetime(2026, 10, 8, 14, 0, 0, tzinfo=UTC)
    assert retry_after_seconds({"Retry-After": "Thu, 08 Oct 2026 14:01:30 GMT"}, now=at) == 90
    assert retry_after_seconds({"Retry-After": "Thu, 08 Oct 2026 13:00:00 GMT"}, now=at) == 0
    for _ in range(200):
        w = next_wait(15)
        assert 15 <= w <= 15 * 1.15  # jitter only ever adds
    assert 60 <= next_wait(15, failures=2) <= 69
    assert 900 <= next_wait(60, failures=4) <= 900 * 1.15  # backoff is capped at 15 minutes
    assert 120 <= next_wait(15, failures=1, retry_after=120) <= 138
    assert next_wait(15, failures=1, retry_after=10 * 24 * 3600) <= MAX_RETRY_AFTER * 1.15


async def test_feed_check_is_conditional():
    seen = []
    body = (b'<?xml version="1.0"?><rss version="2.0"><channel><title>T</title>'
            b"<item><title>x</title><guid>g1</guid></item></channel></rss>")

    def handler(request):
        seen.append(request.headers)
        if len(seen) == 1:
            return httpx.Response(200, content=body, headers={"ETag": '"v1"', "Last-Modified": "Thu, 08 Oct 2026 14:00:00 GMT"})
        return httpx.Response(304)

    src = SourceConfig(id="feed", type="rss", name="Feed", url="https://example.com/rss")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        p = FeedPoller(src, None, lambda *a: None, client)
        await p.poll_once()
        assert await p.poll_once() == 0
    assert "if-none-match" not in seen[0]
    assert seen[1]["if-none-match"] == '"v1"' and seen[1]["if-modified-since"] == "Thu, 08 Oct 2026 14:00:00 GMT"
    assert seen[1]["user-agent"] == USER_AGENT


async def test_busy_website_is_left_alone_as_long_as_it_asks(monkeypatch):
    replies = [httpx.Response(429, headers={"Retry-After": "120"}), httpx.Response(503)]
    waits, statuses = [], []

    async def fake_sleep(seconds):
        waits.append(seconds)
        if len(waits) == 3:
            raise asyncio.CancelledError

    # only the feed reader's waits are replaced, not asyncio's own
    monkeypatch.setattr(rss, "asyncio", SimpleNamespace(sleep=fake_sleep, CancelledError=asyncio.CancelledError,
                                                        to_thread=asyncio.to_thread))
    src = SourceConfig(id="wire", type="rss", name="Wire", url="https://example.com/rss", poll_seconds=15)
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: replies.pop(0))) as client:
        p = FeedPoller(src, None, lambda lvl, d: statuses.append((lvl, d)), client)
        with pytest.raises(asyncio.CancelledError):
            await p.run()
    assert waits[0] <= 2.5  # the first check is spread out a little
    assert 120 <= waits[1] <= 138  # Retry-After honoured
    assert 60 <= waits[2] <= 69  # 503 without Retry-After: backing off
    assert statuses[-1][0] == "warn" and "HTTP 503" in statuses[-1][1]
    with pytest.raises(SlowDown):
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(429))) as client:
            await FeedPoller(src, None, lambda *a: None, client).poll_once()


def test_sec_gets_its_required_user_agent():
    assert user_agent_for("https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent") == SEC_USER_AGENT
    assert user_agent_for("https://sec.gov/news/pressreleases.rss") == SEC_USER_AGENT
    assert user_agent_for("https://www.sec.gov.example.com/feed") == USER_AGENT
    assert user_agent_for("https://example.com/rss") == USER_AGENT
    assert "NewsTrader/" in SEC_USER_AGENT and "github.com" in SEC_USER_AGENT


async def test_x_rate_limit_waits_until_the_reset():
    reset = int(datetime.now(UTC).timestamp()) + 300
    src = SourceConfig(id="x", type="x_account", name="X", url="@someone")

    def handler(request):
        return httpx.Response(429, headers={"x-rate-limit-reset": str(reset)})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        x = XAccountSource(src, None, lambda *a: None, "token", client)
        with pytest.raises(SlowDown) as exc:
            await x._get("/users/by/username/someone")
    assert 290 <= exc.value.wait <= 300


def test_breaking_news_wires_are_checked_every_15_seconds():
    srcs = {s.id: s for s in AppSettings().sources}
    for sid in ("prnewswire", "globenewswire", "sec-8k", "fed-press", "truth-social-trump"):
        assert srcs[sid].poll_seconds == 15, sid
    for s in srcs.values():
        if "sec.gov" in s.url:
            assert s.poll_seconds >= 15, s.id
        if "news.google.com" in s.url:
            assert s.poll_seconds == 300, s.id
    assert srcs["cnbc-top"].poll_seconds == 30 and srcs["bbc-business"].poll_seconds == 30
    assert SourceConfig(id="mine", type="rss", name="Mine", url="https://example.com/rss").poll_seconds == 30


# ---------------------------------------------------------------- upgrading saved settings
def _v03_config() -> dict:
    """Every built-in feed at its v0.3 check interval, as a v0.3 config.json had them."""
    data = AppSettings().model_dump(mode="json")
    found = {p["id"]: p for p in presets.default_sources()}
    for s in data["sources"]:
        if s["type"] in ("rss", "social_rss"):
            s["poll_seconds"] = presets.old_poll_seconds(found[s["id"]])
    data["schema_version"] = 2
    data["transcription"]["analysis_debounce_seconds"] = 10
    return data


def test_old_check_intervals_speed_up_but_user_choices_stay(tmp_path):
    data = _v03_config()
    changed = {"cnbc-top": 120, "prnewswire": 45, "truth-social-trump": 60}  # set by the user
    for s in data["sources"]:
        s["poll_seconds"] = changed.get(s["id"], s["poll_seconds"])
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    store = ConfigStore(path)
    poll = {s.id: s.poll_seconds for s in store.settings.sources}
    assert poll["cnbc-top"] == 120 and poll["prnewswire"] == 45 and poll["truth-social-trump"] == 60
    assert poll["fed-press"] == 15 and poll["sec-8k"] == 15 and poll["globenewswire"] == 15
    assert poll["bbc-business"] == 30 and poll["wsj-markets"] == 30
    assert poll["google-news-business"] == 300 and poll["trump-interviews"] == 300
    assert store.settings.transcription.analysis_debounce_seconds == 4
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["schema_version"] == 4
    assert {s["id"]: s["poll_seconds"] for s in saved["sources"]}["fed-press"] == 15  # saved straight away

    # after the upgrade, a value the user picks is kept - even the old default
    store.upsert_source({**store.source("fed-press").model_dump(mode="json"), "poll_seconds": 60})
    store.update({"transcription": {"analysis_debounce_seconds": 10}})
    again = ConfigStore(path)
    assert again.source("fed-press").poll_seconds == 60
    assert again.settings.transcription.analysis_debounce_seconds == 10


def test_truth_social_old_default_moves_and_custom_tv_pause_stays(tmp_path):
    data = _v03_config()
    data["transcription"]["analysis_debounce_seconds"] = 7  # changed by the user
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    store = ConfigStore(path)
    assert store.source("truth-social-trump").poll_seconds == 15  # was on the old 30 s default
    assert store.settings.transcription.analysis_debounce_seconds == 7
    assert AppSettings().transcription.analysis_debounce_seconds == 4
    assert AppSettings().transcription.chunk_seconds == 10


# ---------------------------------------------------------------- TV: analysed sooner, each line once
async def test_tv_hit_while_the_last_clip_is_still_being_sent_gets_its_own_analysis(ctx):
    submitted, release = [], asyncio.Event()

    class SlowPipeline:
        tickers = make_tickers(ctx.db)

        async def submit(self, item):
            submitted.append(item)
            if len(submitted) == 1:
                await release.wait()  # the first clip is still on its way when the next hit arrives
            return {"status": "queued"}

    ctx.services["pipeline"] = SlowPipeline()
    ctx.config.update({"transcription": {"analysis_debounce_seconds": 0}})
    worker = StreamWorker(StreamManager(ctx, transcriber=Transcriber(lambda: None, lambda *a: None)),
                          SourceConfig(id="tv", type="stream", name="TV", url="https://www.youtube.com/@tv/live"))
    now = datetime.now(UTC).timestamp()
    await worker.handle_segments(Job("tv", now, np.zeros(1)), [Segment(now, now + 3, "Nvidia wins a contract.")])
    await asyncio.sleep(0.05)
    await worker.handle_segments(Job("tv", now, np.zeros(1)), [Segment(now + 4, now + 7, "Tesla cuts prices.")])
    await asyncio.sleep(0.05)
    release.set()
    await asyncio.sleep(0.05)
    assert len(submitted) == 2
    assert submitted[0].title == "Nvidia wins a contract."
    assert submitted[1].title == "Tesla cuts prices."  # only the new line is "new"; the first is context
    assert "Earlier (context only" in submitted[1].body and "Nvidia" in submitted[1].body
    assert submitted[1].spoken_at == datetime.fromtimestamp(now + 7, UTC)


def test_alpaca_order_times_keep_milliseconds():
    from newstrader.trading.broker import order_to_dict

    t = datetime(2026, 10, 8, 14, 0, 1, 234567, tzinfo=UTC)
    o = SimpleNamespace(id="a1", client_order_id="c1", symbol="NVDA", side="buy", qty="1", filled_qty="1",
                        filled_avg_price="100", order_type="market", type=None, order_class="bracket", status="filled",
                        limit_price=None, stop_price=None, time_in_force="gtc", submitted_at=t, created_at=t,
                        filled_at=t, updated_at=t, legs=[])
    d = order_to_dict(o)
    assert d["submitted_at"] == "2026-10-08T14:00:01.234Z" and d["filled_at"] == "2026-10-08T14:00:01.234Z"
