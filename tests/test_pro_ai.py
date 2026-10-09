"""Pro AI wired into the app: which stories it reads, watch-only signals (never traded or alerted, but price-tracked),
judge mode (only ever manual reviews), the queue, its settings, the service's states, the API, diagnostics and the
eval script. Everything runs against fakes - no llama-server, no downloads."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from newstrader import paths
from newstrader.ai.claude_client import ClaudeAnalyzer
from newstrader.ai.engine import AnalysisResult
from newstrader.ai.pipeline import Pipeline
from newstrader.ai.prefilter import Candidate, PrefilterResult
from newstrader.ai.validator import ValidSignal
from newstrader.config import ConfigStore
from newstrader.db import iso
from newstrader.diagnostics import _check_pro_ai
from newstrader.llm import routing
from newstrader.llm.catalog import MODELS_BY_KEY, SERVER_BUILDS
from newstrader.llm.client import ChatResult
from newstrader.llm.download import DownloadCancelled
from newstrader.llm.hardware import HardwareInfo
from newstrader.llm.service import QUEUE_MAX, SELFTEST_KEY, ProAIService, ProJob
from newstrader.performance.stats import compute_stats
from newstrader.performance.tracker import PerformanceTracker
from newstrader.sources.base import NewsItem
from newstrader.trading.fake_broker import FakeBroker
from newstrader.trading.trader import Trader

from .helpers import FakeClaude, make_tickers, sig, signal_json

ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------------------------------------ fakes
class FakeProEngine:
    engine = "pro"

    def __init__(self, *answers: str, ok: bool = True, delay: float = 0.0):
        self.answers = list(answers)
        self.ok = ok
        self.delay = delay
        self.timeout_s = 20.0
        self.seen: list[str] = []
        self.server = SimpleNamespace(fully_on_gpu=True)

    async def analyze(self, item, pre, market_open, now, model=None, skip_rate_limit=False) -> AnalysisResult:
        self.seen.append(item.title)
        if self.delay:
            await asyncio.sleep(self.delay)
        if not self.ok:
            return AnalysisResult(ok=False, status="error", error="Pro AI took too long", engine="pro",
                                  model="fake-pro")
        text = self.answers.pop(0) if self.answers else signal_json()
        return AnalysisResult(ok=True, text=text, engine="pro", model="fake-pro", latency_ms=5)


class FakeAlerts:
    def __init__(self):
        self.sent: list[tuple[str, str]] = []

    async def send(self, kind, title, message="", level="info", force=False, fields=None):
        self.sent.append((kind, title))
        return {}


class FakeServer:
    """Behaves like LlamaServer (start / wait_ready / stop / restart-once) without a process."""

    def __init__(self, ready: bool = True):
        self.ready_ok = ready
        self.state, self.message, self.running = "off", "", False
        self.proc = None
        self.started: list[dict] = []
        self.stops = 0
        self.restarts = 0
        self.max_restarts = 1
        self.fully_on_gpu, self.gpu_layers, self.total_layers = True, 33, 33
        self.url, self.api_key = "http://127.0.0.1:1", "key"

    def start(self, model_path, server_exe, reserve_mib=1024, ctx=8192, slots=2, idle_seconds=0, build_key=""):
        self.started.append({"model": Path(model_path).name, "reserve_mib": reserve_mib, "slots": slots,
                             "build": build_key})
        self.restarts = 0
        self.state, self.message, self.running = "starting", "Loading the model...", True

    def wait_ready(self, timeout=180.0, poll=0.25, cancel_event=None):
        if self.ready_ok:
            self.state, self.message = "ready", "Ready - the whole model is on the graphics card (33 layers)"
            return True
        self.state, self.message, self.running = "error", "Pro AI ran out of graphics memory.", False
        return False

    def stop(self, timeout=10.0):
        self.stops += 1
        self.state, self.message, self.running = "off", "Stopped", False

    def crash(self):
        self.state, self.message, self.running = "error", "Pro AI's server stopped unexpectedly (exit code 1).", False

    def restart_if_crashed(self):
        if self.running or self.restarts >= self.max_restarts:
            return False
        self.restarts += 1
        self.state, self.running = "starting", True
        return True

    def health(self, client=None):
        return 200 if self.state == "ready" else None

    def snapshot(self):
        return {"state": self.state, "message": self.message, "fully_on_gpu": self.fully_on_gpu}


class FakeFiles:
    """The download module's interface over two sets of "downloaded" keys."""

    def __init__(self, folder: Path):
        self.folder = folder
        self.servers: set[str] = set()
        self.models: set[str] = set()
        self.hang = False  # the model download waits until it is cancelled

    def installed_server_exe(self, key):
        return self.folder / f"{key}-llama-server" if key in self.servers else None

    def model_path(self, model):
        key = model if isinstance(model, str) else model.key
        return self.folder / f"{key}.gguf" if key in self.models else None

    def install_server(self, build, progress_cb=None, cancel_event=None, client=None):
        size = SERVER_BUILDS[build].download_bytes
        for done in (size // 2, size):
            progress_cb(done, size)
        self.servers.add(build)
        return self.installed_server_exe(build)

    def fetch_model(self, model, progress_cb=None, cancel_event=None, client=None):
        if self.hang:
            while not cancel_event.is_set():
                time.sleep(0.01)
            raise DownloadCancelled("Download cancelled")
        size = MODELS_BY_KEY[model].size
        progress_cb(size, size)
        self.models.add(model)
        return self.model_path(model)

    def status(self):
        total = sum(MODELS_BY_KEY[k].size for k in self.models) + sum(SERVER_BUILDS[k].download_bytes
                                                                      for k in self.servers)
        return {"total_bytes": total, "folder": str(self.folder)}

    def delete_model(self, key):
        n = MODELS_BY_KEY[key].size if key in self.models else 0
        self.models.discard(key)
        return n

    def delete_server(self, key):
        n = SERVER_BUILDS[key].download_bytes if key in self.servers else 0
        self.servers.discard(key)
        return n

    def delete_old_servers(self):
        return 0


def rtx_5070_ti(_pids=()) -> HardwareInfo:
    return HardwareInfo("windows", "NVIDIA GeForce RTX 5070 Ti", 16303, 1100, "581.57", "12.0")


def make_service(ctx, tmp_path, engine=None, server=None):
    files = FakeFiles(tmp_path)
    made: list[tuple[str, dict]] = []

    def factory(srv, model_key, **kw):
        made.append((model_key, kw))
        return engine or FakeProEngine()

    pro = ProAIService(ctx, server=server or FakeServer(), engine_factory=factory, detect=rtx_5070_ti, files=files)
    return pro, files, made


def pro_ready(ctx, engine, **settings) -> ProAIService:
    """A running Pro AI (no server needed) with these settings."""
    ctx.config.update({"ai": {"pro_ai": {"enabled": True, **settings}}})
    pro = ProAIService(ctx, server=FakeServer())
    pro.engine, pro.state, pro.model_key = engine, "ready", "qwen3.5-9b"
    ctx.services["pro_ai"] = pro
    return pro


# ------------------------------------------------------------------------------------------------ items
def tv(text="Nvidia just won a ten billion dollar Pentagon contract, huge news", ext="tv-1") -> NewsItem:
    return NewsItem(source_id="cnbc-tv", source_type="stream", source_name="CNBC TV", external_id=ext,
                    title=text[:200], body="New:\n[10:00:00] " + text, url="https://www.youtube.com/@CNBC/live",
                    published_at=datetime.now(UTC), kind="transcript", language="en")


def news(title, source="reuters", ext=None, **kw) -> NewsItem:
    return NewsItem(source_id=source, source_type="rss", source_name=source.upper(), external_id=ext or title,
                    title=title, url=f"https://example.com/{abs(hash(title))}", published_at=datetime.now(UTC), **kw)


async def run(pipeline, item):
    res = await pipeline.submit(item)
    if res["status"] == "queued":
        _, queued, pre = pipeline.queue.get_nowait()
        pipeline.queue.task_done()
        return await pipeline.analyze_item(queued, pre)
    return res


def trades(broker) -> int:
    return len([c for c in broker.calls if c[0] == "submit_bracket"])


@pytest.fixture
async def env(ctx):
    ctx.config.update({"ai": {"engine": "claude"}})
    broker = FakeBroker(prices={"NVDA": 180.0, "AAPL": 230.0, "TSLA": 250.0, "SPY": 600.0, "JPM": 300.0})
    trader = Trader(ctx, broker_factory=lambda: broker)
    ctx.services["trader"] = trader
    await trader.connect()
    alerts = FakeAlerts()
    ctx.services["alerts"] = alerts
    claude = FakeClaude()
    pipeline = Pipeline(ctx, analyzer=ClaudeAnalyzer(ctx, client_factory=lambda: claude), tickers=make_tickers(ctx.db))
    ctx.services["pipeline"] = pipeline
    yield SimpleNamespace(ctx=ctx, pipeline=pipeline, claude=claude, broker=broker, alerts=alerts)
    await trader.stop()


# ------------------------------------------------------------------------------------------------ routing
def _pre(*symbols, keywords=(), why="name") -> PrefilterResult:
    cands = [Candidate(s, s, why) for s in symbols]
    return PrefilterResult(hit=bool(cands or keywords), candidates=cands, keywords=list(keywords))


def _signal(ticker="NVDA", direction="bullish", event="") -> ValidSignal:
    return ValidSignal(ticker, "", "", "", "", direction, 85, "hours", "why", event=event)


def test_hard_cases():
    story = news("Nvidia shares look great")
    assert routing.hard_case(tv(), _pre("NVDA"), "claude", [_signal()]) == routing.TV
    assert routing.hard_case(news("Nvidia gewinnt Auftrag", language="de"), _pre("NVDA"), None, []) == \
        routing.NOT_ENGLISH
    assert routing.hard_case(news("Fed cuts interest rates"), _pre(keywords=["rate cut"]), None, []) == routing.MACRO
    assert routing.hard_case(news("BOJ hikes"), _pre("EWJ", why="country: Japan"), "local", [_signal("EWJ")]) == \
        routing.MACRO
    # the local engine: wording only, or nothing although a company was named
    assert routing.hard_case(story, _pre("NVDA"), "local", [_signal()]) == routing.WORDING_ONLY
    assert routing.hard_case(story, _pre("NVDA"), "local", []) == routing.MAIN_NEUTRAL
    assert routing.hard_case(story, _pre("NVDA"), "local", [_signal(direction="neutral")]) == routing.MAIN_NEUTRAL
    assert routing.hard_case(story, _pre("NVDA"), "local", [_signal(event="Beat earnings")]) == ""
    assert routing.hard_case(story, _pre("NVDA"), "claude", []) == ""  # only the local engine's calls count


def test_route_and_scope():
    story = news("Nvidia beats estimates")
    with_event = [_signal(event="Beat earnings")]
    assert routing.route(story, _pre("NVDA"), "local", with_event, "hard") == ("", False)
    assert routing.route(story, _pre("NVDA"), "local", with_event, "all") == (routing.EVERY_STORY, False)
    assert routing.route(tv(), _pre("NVDA"), "local", with_event, "all") == (routing.TV, True)
    assert routing.route(news("Local bakery wins award"), _pre(), None, [], "all") == ("", False)  # nothing to name


def test_agreement():
    assert routing.compare({"NVDA": "bullish"}, {"NVDA": "bullish"}) == "agree"
    assert routing.compare({}, {}) == "agree"
    assert routing.compare({"NVDA": "neutral"}, {}) == "agree"
    assert routing.compare({"NVDA": "bullish"}, {"NVDA": "bearish"}) == "disagree"
    assert routing.compare({"NVDA": "bullish"}, {}) == "disagree"
    assert routing.compare({}, {"AAPL": "bullish"}) == "disagree"
    assert routing.compare(None, {"AAPL": "bullish"}) == "no_main"


# ------------------------------------------------------------------------------------------------ watch mode
async def test_watch_mode_stores_the_reading_but_never_trades_or_alerts(env):
    ctx, p = env.ctx, env.pipeline
    pro = pro_ready(ctx, FakeProEngine(signal_json(sig("NVDA", "bearish", 77), sig("AAPL", "bullish", 95))))
    env.claude.responses = [signal_json(sig("NVDA", confidence=90))]
    out = await run(p, tv())
    assert out["signals"][0]["traded"] == 1  # the main engine's call goes on exactly as before
    traded, alerted = trades(env.broker), len(env.alerts.sent)
    job = pro.take()
    assert job.reason == routing.TV and not job.judge and job.main_calls == {"NVDA": "bullish"}
    calls = await pro.run_job(job)
    assert set(calls) == {"NVDA", "AAPL"}
    a = ctx.db.query_one("SELECT * FROM analyses WHERE engine = 'pro'")
    assert a["cost_usd"] == 0 and a["agreement"] == "disagree" and a["pro_reason"] == routing.TV
    assert a["main_analysis_id"] == out["analysis_id"] and a["model"] == "fake-pro"
    rows = {r["ticker"]: r for r in ctx.db.query("SELECT * FROM signals WHERE engine = 'pro'")}
    assert {r["action"] for r in rows.values()} == {"watch"} and not any(r["traded"] for r in rows.values())
    assert rows["NVDA"]["main_direction"] == "bullish" and rows["AAPL"]["main_direction"] == "neutral"
    assert "never traded or alerted" in rows["AAPL"]["action_reason"]
    assert trades(env.broker) == traded  # AAPL at 95 would be a buy for the main engine - not for Pro AI
    assert len(env.alerts.sent) == alerted
    main = ctx.db.query_one("SELECT * FROM signals WHERE engine = 'claude'")
    assert main["pro_direction"] == "bearish"
    # watch signals are price-tracked like every other signal, so the scoreboard can compare the engines
    await PerformanceTracker(ctx).run_once()
    tracked = {r["signal_id"] for r in ctx.db.query("SELECT signal_id FROM signal_prices")}
    assert {rows["NVDA"]["id"], rows["AAPL"]["id"], main["id"]} <= tracked
    # and never approved into a trade by accident
    with pytest.raises(PermissionError):
        await p.approve(rows["AAPL"]["id"])
    assert trades(env.broker) == traded


async def test_watch_signals_never_merge_with_tradeable_ones(env):
    ctx, p = env.ctx, env.pipeline
    pro = pro_ready(ctx, FakeProEngine(signal_json(sig("AAPL", "bullish", 85)),
                                       signal_json(sig("AAPL", "bullish", 88))))
    env.claude.responses = [signal_json(), signal_json(), signal_json(sig("AAPL", confidence=90))]
    await run(p, tv("Apple is the talk of the floor this morning", ext="a"))
    await pro.run_job(pro.take())
    await run(p, tv("Apple again, people keep talking about Apple", ext="b"))
    await pro.run_job(pro.take())
    watch = ctx.db.query("SELECT * FROM signals WHERE engine = 'pro' ORDER BY id")
    assert [r["action"] for r in watch] == ["watch", "merged"]  # Pro AI's own repeats count once
    assert watch[1]["merged_into"] == watch[0]["id"]
    # a main-engine signal on the same stock isn't swallowed by the watch signal: it is traded on its own
    out = await run(p, news("Apple wins a giant government contract"))
    s = out["signals"][0]
    assert s["action"] == "bought" and s["merged_into"] is None and s["engine"] == "claude"


async def test_only_hard_stories_go_to_pro_ai_unless_scope_is_all(env):
    ctx, p = env.ctx, env.pipeline
    pro = pro_ready(ctx, FakeProEngine())
    env.claude.responses = [signal_json(sig("NVDA", confidence=90)), signal_json(sig("TSLA", confidence=90))]
    await run(p, news("Nvidia wins $10 billion government AI contract"))
    assert pro.take() is None  # an English story the main engine read: not a hard case
    ctx.config.update({"ai": {"pro_ai": {"scope": "all"}}})
    await run(p, news("Tesla wins a big robotaxi contract"))
    job = pro.take()
    assert job.reason == routing.EVERY_STORY and job.main_calls == {"TSLA": "bullish"}


async def test_stories_the_local_engine_skips_go_to_pro_ai(env):
    ctx, p = env.ctx, env.pipeline
    ctx.config.update({"ai": {"engine": "local"}})
    pro = pro_ready(ctx, FakeProEngine(signal_json(sig("NVDA", "bullish", 80))))
    out = await p.submit(news("Nvidia erhält Großauftrag vom Pentagon", language="de"))
    assert out["status"] == "filtered"  # the local engine reads English only...
    job = pro.take()
    assert job.reason == routing.NOT_ENGLISH and job.main_calls is None  # ...Pro AI reads any language
    await pro.run_job(job)
    a = ctx.db.query_one("SELECT * FROM analyses WHERE engine = 'pro'")
    assert a["agreement"] == "no_main" and a["main_analysis_id"] is None
    row = ctx.db.query_one("SELECT * FROM signals WHERE engine = 'pro'")
    assert row["action"] == "watch" and row["main_direction"] is None and "didn't read" in row["action_reason"]
    # market-wide news with no company named
    out = await p.submit(news("Fed signals a rate cut in December as inflation cools"))
    assert out["status"] == "filtered" and pro.take().reason == routing.MACRO
    # stories that are too old aren't sent
    old = news("Nvidia erhält noch einen Auftrag", language="de")
    old.published_at = datetime(2020, 1, 1, tzinfo=UTC)
    await p.submit(old)
    assert pro.take() is None


async def test_failed_or_not_running_pro_ai_changes_nothing(env):
    ctx, p = env.ctx, env.pipeline
    pro = pro_ready(ctx, FakeProEngine(ok=False))
    env.claude.responses = [signal_json(sig("NVDA", confidence=90)), signal_json(sig("NVDA", confidence=90))]
    await run(p, tv())
    assert await pro.run_job(pro.take()) is None
    a = ctx.db.query_one("SELECT * FROM analyses WHERE engine = 'pro'")
    assert a["status"] == "error" and a["agreement"] is None
    assert ctx.db.scalar("SELECT COUNT(*) FROM signals WHERE engine = 'pro'") == 0
    pro.state = "error"
    await run(p, tv(ext="tv-2"))
    assert not pro._jobs  # not running: nothing queued


# ------------------------------------------------------------------------------------------------ judge mode
async def test_judge_sends_a_disputed_signal_to_review_instead_of_trading(env):
    ctx, p = env.ctx, env.pipeline
    pro = pro_ready(ctx, FakeProEngine(signal_json(sig("NVDA", "bearish", 80))), mode="judge")
    pro._start_workers()
    env.claude.responses = [signal_json(sig("NVDA", confidence=92))]
    out = await run(p, tv())
    s = out["signals"][0]
    assert s["action"] == "review" and not s["traded"] and s["review_status"] == "pending"
    assert "Pro AI disagreed" in s["action_reason"] and s["pro_direction"] == "bearish"
    assert trades(env.broker) == 0
    assert any(kind == "manual_review" for kind, _ in env.alerts.sent)
    # Pro AI's own reading is a watch signal; the person can still approve the main engine's call
    assert ctx.db.query_one("SELECT action FROM signals WHERE engine = 'pro'")["action"] == "watch"
    res = await p.approve(s["id"])
    assert res["traded"] and trades(env.broker) == 1
    pro._stop_workers()


async def test_judge_agreeing_lets_the_trade_through_and_raises_missed_stocks_for_review(env):
    ctx, p = env.ctx, env.pipeline
    pro = pro_ready(ctx, FakeProEngine(signal_json(sig("NVDA", "bullish", 85), sig("AAPL", "bullish", 91),
                                                   sig("TSLA", "bullish", 30))), mode="judge")
    pro._start_workers()
    env.claude.responses = [signal_json(sig("NVDA", confidence=92))]
    out = await run(p, tv())
    assert out["signals"][0]["action"] == "bought" and out["signals"][0]["pro_direction"] == "bullish"
    rows = {r["ticker"]: r for r in ctx.db.query("SELECT * FROM signals WHERE engine = 'pro'")}
    # a stock the main engine missed becomes a manual review - never a trade, even at 91
    assert rows["AAPL"]["action"] == "review" and rows["AAPL"]["review_status"] == "pending"
    assert not rows["AAPL"]["traded"] and "never trades by itself" in rows["AAPL"]["action_reason"]
    assert rows["TSLA"]["action"] == "watch"  # below the review threshold: only watched
    assert rows["NVDA"]["action"] == "watch"
    assert trades(env.broker) == 1  # only the main engine's NVDA buy
    assert not any("TSLA" in title for _kind, title in env.alerts.sent)  # watch-only: no alert
    res = await p.approve(rows["AAPL"]["id"])  # the person decides
    assert res["traded"] and trades(env.broker) == 2
    pro._stop_workers()


async def test_judge_without_an_answer_keeps_the_main_engines_call(env):
    ctx, p = env.ctx, env.pipeline
    pro = pro_ready(ctx, FakeProEngine(ok=False), mode="judge")
    pro._start_workers()
    env.claude.responses = [signal_json(sig("NVDA", confidence=92))]
    out = await run(p, tv())
    assert out["signals"][0]["action"] == "bought" and out["signals"][0]["pro_direction"] is None
    pro._stop_workers()


async def test_judge_mode_on_an_easy_story_only_watches(env):
    ctx, p = env.ctx, env.pipeline
    pro = pro_ready(ctx, FakeProEngine(signal_json(sig("TSLA", "bearish", 95))), mode="judge", scope="all")
    env.claude.responses = [signal_json(sig("TSLA", confidence=92))]
    out = await run(p, news("Tesla wins a big robotaxi contract"))
    assert out["signals"][0]["action"] == "bought"  # not a hard case: never held back
    job = pro.take()
    assert not job.judge
    await pro.run_job(job)
    assert ctx.db.query_one("SELECT action FROM signals WHERE engine = 'pro'")["action"] == "watch"


# ------------------------------------------------------------------------------------------------ queue
async def test_queue_newest_first_drops_stale_and_full(ctx):
    pro = pro_ready(ctx, FakeProEngine(), max_wait_seconds=5)

    def job(n, age=0.0, urgent=False):
        return ProJob(item=news(f"story {n}"), pre=_pre("NVDA"), reason=routing.TV, urgent=urgent,
                      queued_at=time.monotonic() - age)

    assert pro.offer(job(1)) and pro.offer(job(2)) and pro.offer(job(3, urgent=True))
    assert pro.take().item.title == "story 3"  # a main-engine signal is waiting for this one
    assert pro.take().item.title == "story 2"  # then newest first
    stale = job(4, age=60)
    stale.future = asyncio.get_running_loop().create_future()
    pro._jobs.insert(0, stale)
    assert pro.take().item.title == "story 1"
    assert pro.take() is None and pro.stats["too_old"] == 1 and stale.future.result() is None
    for n in range(QUEUE_MAX + 2):
        pro.offer(job(n))
    assert len(pro._jobs) == QUEUE_MAX and pro.stats["queue_full"] == 2
    assert pro._jobs[0].item.title == "story 2"  # the oldest ones went
    pro.state = "off"
    assert not pro.offer(job(99))


# ------------------------------------------------------------------------------------------------ settings
def test_settings_defaults():
    from newstrader.config import AppSettings

    s = AppSettings().ai.pro_ai
    assert (s.enabled, s.mode, s.model, s.build, s.scope, s.max_wait_seconds, s.keep_tv_memory_free) == \
        (False, "watch", "auto", "auto", "hard", 20, True)
    assert AppSettings().schema_version == 4


def test_old_config_gets_pro_ai_switched_off(tmp_path):
    from newstrader.config import AppSettings

    data = AppSettings().model_dump(mode="json")
    data["schema_version"] = 3
    data["risk"]["max_dollars_per_trade"] = 250
    data["ai"].pop("pro_ai")
    data["ai"]["engine"] = "claude"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    store = ConfigStore(path)
    assert not store.settings.ai.pro_ai.enabled and store.settings.ai.pro_ai.mode == "watch"
    assert store.settings.risk.max_dollars_per_trade == 250 and store.settings.ai.engine == "claude"
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["schema_version"] == 4 and saved["ai"]["pro_ai"]["enabled"] is False  # saved straight away
    # an older file can't switch Pro AI on (or into judge mode) by itself
    data["ai"]["pro_ai"] = {"enabled": True, "mode": "judge"}
    path.write_text(json.dumps(data), encoding="utf-8")
    s = ConfigStore(path).settings.ai.pro_ai
    assert not s.enabled and s.mode == "watch"


def test_unknown_model_falls_back_to_auto_without_resetting_anything(tmp_path):
    store = ConfigStore(tmp_path / "config.json")
    store.update({"risk": {"max_dollars_per_trade": 300},
                  "ai": {"pro_ai": {"enabled": True, "model": "qwen3.5-9b", "mode": "judge"}}})
    raw = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    raw["ai"]["pro_ai"].update(model="a-model-an-update-removed", build="old-build")
    (tmp_path / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    s = ConfigStore(tmp_path / "config.json").settings
    assert s.ai.pro_ai.model == "auto" and s.ai.pro_ai.build == "auto"
    assert s.ai.pro_ai.enabled and s.ai.pro_ai.mode == "judge" and s.risk.max_dollars_per_trade == 300


@pytest.mark.parametrize("patch", [{"mode": "trade"}, {"scope": "some"}, {"max_wait_seconds": 1},
                                   {"max_wait_seconds": 1000}])
def test_bad_pro_ai_settings_rejected(patch):
    store = ConfigStore(paths.config_file())
    with pytest.raises(ValidationError):
        store.update({"ai": {"pro_ai": patch}})


# ------------------------------------------------------------------------------------------------ service
async def test_service_set_up_start_crash_and_stop(ctx, tmp_path):
    ctx.bus.bind_loop(asyncio.get_running_loop())
    events = ctx.bus.subscribe()
    pro, files, made = make_service(ctx, tmp_path)
    srv = pro.server
    await pro.start()
    assert pro.state == "off" and not srv.started
    ctx.config.update({"ai": {"pro_ai": {"enabled": True}}})
    await asyncio.sleep(0.05)
    await _settle(pro)
    assert pro.state == "not_set_up" and "Set up Pro AI" in pro.message

    plan = await asyncio.to_thread(pro.plan)
    assert "RTX 5070 Ti" in plan["card"] and plan["can_run"]
    assert plan["build"] == "cuda13" and plan["model"] == "qwen3.5-9b"  # TV is on: 5 GB kept for transcription
    fits = {m["key"]: m["fits"] for m in plan["models"]}
    assert fits["qwen3.5-9b"] and not fits["qwen3.6-35b-a3b"]
    assert plan["download_bytes"] == SERVER_BUILDS["cuda13"].download_bytes + MODELS_BY_KEY["qwen3.5-9b"].size
    assert {b["key"] for b in plan["builds"]} == {"cuda13", "cuda12", "vulkan", "cpu"}  # this PC's system only

    pro.setup("auto", "auto")
    with pytest.raises(RuntimeError):
        pro.setup()  # one download at a time
    await pro._setup_task
    await _settle(pro)
    assert pro.state == "ready" and pro.ready and files.models == {"qwen3.5-9b"} and files.servers == {"cuda13"}
    assert srv.started[-1]["model"] == "qwen3.5-9b.gguf" and srv.started[-1]["build"] == "cuda13"
    assert srv.started[-1]["reserve_mib"] >= 5 * 1024  # room kept for TV transcription
    assert made[-1][0] == "qwen3.5-9b" and made[-1][1]["timeout_s"] == 20
    assert ctx.db.kv_get("pro_ai_installed")["model"] == "qwen3.5-9b"
    seen = []
    while not events.empty():
        msg = events.get_nowait()
        if msg["type"] == "pro_ai":
            seen.append(msg["data"])
    assert any(e["state"] == "downloading" and e["progress"] for e in seen)  # progress reached the window
    assert seen[-1]["state"] == "ready"
    assert any(c["component"] == "pro_ai" and c["level"] == "ok" for c in ctx.state.components())

    srv.crash()  # restarted once...
    await pro.check_server()
    assert pro.state == "ready" and srv.restarts == 1
    srv.crash()  # ...but not twice
    await pro.check_server()
    assert pro.state == "error" and "already restarted once" in pro.message and not pro.ready

    await pro.start_server()  # starting again from Settings
    assert pro.state == "ready"
    ctx.config.update({"ai": {"pro_ai": {"enabled": False}}})  # turned off: the server stops, files stay
    await asyncio.sleep(0.05)
    await _settle(pro)
    assert pro.state == "off" and not srv.running and files.models
    assert not any(c["component"] == "pro_ai" for c in ctx.state.components())

    st = await asyncio.to_thread(pro.status)
    assert st["downloaded"] == {"server": True, "model": True} and st["disk_bytes"] > 5e9
    freed = await pro.delete_downloads()
    assert freed > 5e9 and not files.models and not files.servers and not ctx.config.settings.ai.pro_ai.enabled
    await pro.stop()
    assert not srv.running


async def _settle(pro):
    for _ in range(100):
        if not pro._tasks or all(t.done() for t in pro._tasks if t.get_name() != "pro-ai-watch"):
            return
        await asyncio.sleep(0.01)


async def test_service_start_failure_and_cancelled_download(ctx, tmp_path):
    pro, files, _ = make_service(ctx, tmp_path, server=FakeServer(ready=False))
    files.servers.add("cuda13")
    files.models.add("qwen3.5-9b")
    ctx.config.update({"ai": {"pro_ai": {"enabled": True, "model": "qwen3.5-9b", "build": "cuda13"}}})
    await pro.start()
    await _settle(pro)
    assert pro.state == "error" and "graphics memory" in pro.message
    # a model that isn't downloaded yet: "not set up"; its download can be cancelled
    ctx.config.update({"ai": {"pro_ai": {"model": "qwen3.5-4b"}}})
    await asyncio.sleep(0.05)
    await _settle(pro)
    assert pro.state == "not_set_up" and "isn't downloaded" in pro.message
    files.hang = True
    pro.setup("cuda13", "qwen3.5-4b")
    await asyncio.sleep(0.05)
    assert pro.downloading and pro.state == "downloading"
    pro.cancel_setup()
    await pro._setup_task
    assert pro.state == "not_set_up" and "cancelled" in pro.message and pro.progress is None
    await pro.stop()


async def test_intel_mac_cant_set_up(ctx, tmp_path):
    pro, files, _ = make_service(ctx, tmp_path)
    pro._detect = lambda _p: HardwareInfo("mac", "Intel Core i9", apple_silicon=False, ram_bytes=32 * 1024 ** 3)
    plan = await asyncio.to_thread(pro.plan)
    assert not plan["can_run"] and "Apple Silicon" in plan["build_why"]
    pro.setup()
    await pro._setup_task
    assert pro.state == "error" and "Apple Silicon" in pro.message and not files.models


async def test_selftest_result_is_kept(ctx, tmp_path):
    engine = FakeProEngine(ok=False)
    pro, _, _ = make_service(ctx, tmp_path, engine=engine)
    with pytest.raises(RuntimeError):
        pro.run_selftest()  # not running yet
    pro.engine, pro.state, pro.model_key = engine, "ready", "qwen3.5-9b"
    pro.run_selftest()
    with pytest.raises(RuntimeError):
        pro.run_selftest()  # already running
    await pro._selftest_task
    res = ctx.db.kv_get(SELFTEST_KEY)
    assert res["total"] == 20 and res["right"] == 0 and res["model"] == "qwen3.5-9b" and res["fully_on_gpu"]
    assert pro.selftest_progress is None and engine.seen


# ------------------------------------------------------------------------------------------------ API
def test_pro_ai_api(client, ctx, tmp_path):
    assert client.get("/api/pro-ai").status_code == 503  # not started yet
    pro, files, _ = make_service(ctx, tmp_path)
    ctx.services["pro_ai"] = pro
    st = client.get("/api/pro-ai").json()
    assert st["state"] == "off" and st["settings"]["mode"] == "watch" and not st["downloaded"]["model"]
    plan = client.get("/api/pro-ai/plan").json()
    assert plan["model"] == "qwen3.5-9b" and plan["disk_free_bytes"] > 0
    assert client.get("/api/pro-ai/plan?model=qwen3.5-4b").json()["model"] == "qwen3.5-4b"
    assert client.post("/api/pro-ai/setup", json={"model": "no-such-model"}).status_code == 400
    assert client.post("/api/pro-ai/start").status_code == 409  # turned off
    assert client.post("/api/pro-ai/selftest").status_code == 409  # not running
    assert client.post("/api/pro-ai/setup", json={}).json()["ok"]
    deadline = time.monotonic() + 10
    while client.get("/api/pro-ai").json()["state"] != "ready" and time.monotonic() < deadline:
        time.sleep(0.05)
    st = client.get("/api/pro-ai").json()
    assert st["state"] == "ready" and st["enabled"] and st["downloaded"] == {"server": True, "model": True}
    assert client.get("/api/status").json()["pro_ai"]["state"] == "ready"
    assert client.post("/api/pro-ai/stop").json()["state"] == "off"
    assert client.post("/api/pro-ai/start").status_code == 200
    r = client.delete("/api/pro-ai/downloads").json()
    assert r["freed_bytes"] > 5e9 and not files.models
    assert client.get("/api/pro-ai").json()["state"] == "off"


def test_settings_api_offers_models_and_rejects_bad_values(client):
    meta = client.get("/api/settings").json()["meta"]
    assert meta["pro_ai_models"]["auto"].startswith("Auto") and "qwen3.5-9b" in meta["pro_ai_models"]
    assert "GB download" in meta["pro_ai_models"]["qwen3.5-9b"] and "auto" in meta["pro_ai_builds"]
    assert client.put("/api/settings", json={"ai": {"pro_ai": {"mode": "trade"}}}).status_code == 422
    r = client.put("/api/settings", json={"ai": {"pro_ai": {"mode": "judge", "max_wait_seconds": 30}}})
    assert r.json()["settings"]["ai"]["pro_ai"]["mode"] == "judge"


def test_signals_api_engine_filter_and_watch_approve(client, ctx):
    common = {"created_at": iso(), "ticker": "NVDA", "direction": "bullish", "confidence": 90, "source_name": "TV",
              "sources_seen": "[]"}
    main = ctx.db.insert("signals", {**common, "engine": "local", "action": "review", "review_status": "pending"})
    watch = ctx.db.insert("signals", {**common, "engine": "pro", "action": "watch"})
    raised = ctx.db.insert("signals", {**common, "engine": "pro", "action": "review", "review_status": "pending"})
    ids = lambda q: {s["id"] for s in client.get("/api/signals" + q).json()["signals"]}  # noqa: E731
    assert ids("") == {main, watch, raised}
    assert ids("?engine=main") == {main, raised}
    assert ids("?engine=pro") == {watch, raised}
    assert ids("?action=watch") == {watch}
    ctx.services["pipeline"] = Pipeline(ctx, tickers=make_tickers(ctx.db))
    r = client.post(f"/api/signals/{watch}/approve")
    assert r.status_code == 409 and "watching" in r.json()["detail"]


def test_scoreboard_counts_watch_signals_only_by_engine():
    def row(engine, ret, action):
        return {"direction": "bullish", "confidence": 85, "ret_1h": ret, "source_name": "TV", "source_type": "stream",
                "traded": 0, "engine": engine, "event": None, "action": action}

    rows = [row("local", 1.0, "bought"), row("pro", -1.0, "watch"), row("pro", -1.0, "watch"),
            row("pro", 2.0, "review")]
    s = compute_stats(rows, "1h")
    assert {g["key"]: g["count"] for g in s["by_engine"]} == {"Local machine learning": 1, "Pro AI": 3}
    assert s["overall"]["count"] == 2 and s["overall"]["win_rate"] == 100  # watch-only calls left out
    assert sum(g["count"] for g in s["by_source"]) == 2


# ------------------------------------------------------------------------------------------------ diagnostics
async def test_diagnostics_check_pro_ai(ctx, tmp_path):
    assert await _check_pro_ai(ctx) == []  # off: nothing to check
    pro, files, _ = make_service(ctx, tmp_path)
    ctx.services["pro_ai"] = pro
    ctx.config.update({"ai": {"pro_ai": {"enabled": True, "model": "qwen3.5-9b", "build": "cuda13"}}})
    checks = await _check_pro_ai(ctx)
    assert checks[0]["name"] == "Pro AI files" and checks[0]["status"] == "error"
    files.servers.add("cuda13")
    files.models.add("qwen3.5-9b")
    await pro.start_server()
    checks = {c["name"]: c for c in await _check_pro_ai(ctx)}
    assert {c["status"] for c in checks.values()} == {"ok"} and len(checks) == 3
    assert "SHA-256" in checks["Pro AI files"]["detail"]
    pro.server.fully_on_gpu, pro.server.gpu_layers = False, 20
    checks = {c["name"]: c for c in await _check_pro_ai(ctx)}
    assert checks["Pro AI on the graphics card"]["status"] == "warn"
    assert "20 of 33" in checks["Pro AI on the graphics card"]["detail"]
    pro.server.crash()
    pro.state, pro.message = "error", pro.server.message
    checks = {c["name"]: c for c in await _check_pro_ai(ctx)}
    assert checks["Pro AI server"]["status"] == "error" and "Pro AI on the graphics card" not in checks


# ------------------------------------------------------------------------------------------------ eval script
def _eval_module():
    spec = importlib.util.spec_from_file_location("eval_detection", ROOT / "scripts" / "eval_detection.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FAKE_LLAMA = ROOT / "tests" / "test_llm_server.py"


def test_eval_scores_pro_ai_on_a_running_server(monkeypatch, capsys, tmp_path):
    ev = _eval_module()
    items = [json.loads(line) for line in (ROOT / "tests/fixtures/eval/dev.jsonl").read_text(
        encoding="utf-8").splitlines()[:4]]
    monkeypatch.setattr(ev, "load_items", lambda which: items)
    seen = {}

    async def answer(url, key, system, user, schema, **kw):
        enum = schema["properties"]["signals"]["items"]["properties"]["ticker"]["enum"]
        return ChatResult(text=signal_json(sig(enum[0], "bullish", 85)))

    def fake_engine(server, model_name, timeout):
        seen.update(url=server.url, key=server.api_key)
        return ev.ProAIEngine(server, model_name, ctx=ev._ctx(ev.AppSettings()), timeout_s=timeout, slots=1,
                              chat_fn=answer)

    monkeypatch.setattr(ev, "make_pro_engine", fake_engine)
    out_json = tmp_path / "out.json"
    assert ev.main(["--engine", "pro", "--pro-url", "http://127.0.0.1:9/", "--pro-key", "K", "--json",
                    str(out_json)]) == 0
    out = capsys.readouterr().out
    assert seen == {"url": "http://127.0.0.1:9", "key": "K"}
    assert "Pro AI (running server)" in out and "Pro AI answers that failed | 0" in out
    saved = json.loads(out_json.read_text(encoding="utf-8"))["Pro AI (running server)"]
    assert saved["items"] == 4 and saved["answer_errors"] == 0
    # the holdout set stays sealed for Pro AI too
    assert ev.main(["--set", "holdout", "--engine", "pro", "--pro-url", "http://x", "--compare",
                    "--mistakes", "5"]) == 0
    out = capsys.readouterr().out
    assert "Right answer" in out and "with event rules" in out and "Pro AI (running server)" in out
    assert "Companies found" not in out and "Seconds per story" not in out and "By kind of news" not in out
    assert ev.main(["--engine", "pro"]) == 2  # needs a server


def test_eval_starts_and_stops_its_own_server(monkeypatch, capsys, tmp_path):
    from newstrader.llm.server import LlamaServer

    from .test_llm_server import FAKE

    ev = _eval_module()
    items = [json.loads(line) for line in (ROOT / "tests/fixtures/eval/dev.jsonl").read_text(
        encoding="utf-8").splitlines()[:3]]
    monkeypatch.setattr(ev, "load_items", lambda which: items)
    exe = tmp_path / "fake-llama-server.py"
    exe.write_text(FAKE, encoding="utf-8")
    model = tmp_path / "tiny.gguf"
    model.write_bytes(b"GGUF")
    servers = []

    def make_server(pid_path=None):
        servers.append(LlamaServer(pid_path=pid_path, launcher=[sys.executable]))
        return servers[-1]

    monkeypatch.setattr(ev, "LlamaServer", make_server)
    assert ev.main(["--engine", "pro", "--pro-model", str(model), "--pro-server", str(exe)]) == 0
    out = capsys.readouterr().out
    assert "Pro AI (tiny)" in out and "Pro AI answers that failed | 0" in out
    assert servers and not servers[0].running and servers[0].state == "off"


# ------------------------------------------------------------------------------------------------ model names
def test_no_model_names_outside_the_catalog():
    """Model identifiers live in newstrader/llm/catalog.py only."""
    for path in (ROOT / "newstrader").rglob("*"):
        if path.suffix not in (".py", ".js", ".html") or path.name == "catalog.py" or "vendor" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        assert "qwen" not in text and "gemma" not in text, path

