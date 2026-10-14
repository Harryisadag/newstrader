"""Shared training: Pro AI labelling old news (resumable, pauses while the market is open), the student model (trains,
saves without pickle, tested on the newest labels), how the local engine uses it (wording-only stories stay manual
review, a disagreement only lowers confidence), the label file format and its checks, getting everyone's labels from
GitHub (fingerprint and ETag), and matching labels with the downloaded headlines. Everything runs offline."""

from __future__ import annotations

import gzip
import json
import random
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from newstrader.ai.engine import AnalysisResult
from newstrader.ai.pipeline import Pipeline
from newstrader.config import AppSettings
from newstrader.db import MIGRATIONS, Database, iso
from newstrader.llm import teacher as teacher_mod
from newstrader.llm.catalog import MODELS
from newstrader.llm.teacher import FAILS_TO_STOP, LIVE_FIRST, TeacherJob, market_busy, parse_answer
from newstrader.ml import shared_labels as sl
from newstrader.ml import sharing
from newstrader.ml import student as st
from newstrader.ml.engine import LocalMLEngine
from newstrader.ml.sentiment import SentimentScores
from newstrader.ml.sharing import SharedTraining, collect_labels, join_labels
from newstrader.ml.student import (
    CLASSES,
    N_HASH,
    NUMERIC,
    StudentFeatures,
    StudentModel,
    StudentSample,
    split_by_time,
    train_student,
)
from newstrader.trading.fake_broker import FakeBroker
from newstrader.trading.trader import Trader

from .helpers import make_tickers
from .test_ml import fake_service, news, run

ROOT = Path(__file__).resolve().parent.parent
MODEL_KEY = MODELS[2].key  # any Pro AI model in the catalog
SATURDAY = datetime(2026, 10, 10, 15, 0, tzinfo=UTC)
TUESDAY_11ET = datetime(2026, 10, 13, 15, 0, tzinfo=UTC)
TUESDAY_925ET = datetime(2026, 10, 13, 13, 25, tzinfo=UTC)
TUESDAY_5PM = datetime(2026, 10, 13, 21, 0, tzinfo=UTC)


# ------------------------------------------------------------------------------------------------ helpers
def add_story(db, nid: str, headline: str, symbols: list[str], at: datetime, summary: str = "") -> None:
    for sym in symbols:
        db.execute("INSERT INTO ml_samples (news_id, symbol, horizon_min, published_at, headline, summary, symbols, "
                   "label_status, created_at, text_version) VALUES (?,?,?,?,?,?,?,?,?,2)",
                   (nid, sym, 60, iso(at), headline, summary, json.dumps(symbols), "ok", iso()))


class FakeTeacherEngine:
    """Answers like Pro AI: bullish on "wins"/"beats", bearish on "recall"/"misses", nothing otherwise."""

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.asked: list[dict] = []

    async def analyze(self, item, pre, market_open, now, model=None, skip_rate_limit=False, *, max_signals=None,
                      timeout_s=None):
        self.asked.append({"id": item.external_id, "tickers": [c.symbol for c in pre.candidates],
                           "max_signals": max_signals, "timeout": timeout_s, "now": now,
                           "market_open": market_open})
        if self.fail:
            return AnalysisResult(ok=False, status="error", error="Pro AI took too long", engine="pro")
        low = item.title.lower()
        d = "bullish" if any(w in low for w in ("wins", "beats")) else "bearish" if any(
            w in low for w in ("recall", "misses")) else None
        first = pre.candidates[0].symbol
        signals = [] if d is None else [{"ticker": first, "direction": d, "confidence": 88, "company": "",
                                         "speaker": "", "bull_case": "", "bear_case": "",
                                         "time_sensitivity": "hours", "reasoning": "x"}]
        return AnalysisResult(ok=True, text=json.dumps({"signals": signals}), engine="pro")


class StubPro:
    """The parts of the Pro AI service "Label old news" uses."""

    def __init__(self, ctx, engine, ready: bool = True):
        self.ctx, self.engine, self.ready = ctx, engine, ready
        self.model_key = MODEL_KEY
        self.waiting = False
        self.testing = False

    @property
    def settings(self):
        return self.ctx.config.settings.ai.pro_ai

    def live_waiting(self) -> bool:
        return self.waiting


@pytest.fixture
def labelling(ctx):
    ctx.services["pipeline"] = SimpleNamespace(tickers=make_tickers(ctx.db))
    t0 = datetime(2026, 9, 1, 14, 0, tzinfo=UTC)
    add_story(ctx.db, "101", "Nvidia wins $10 billion government AI contract", ["NVDA"], t0)
    add_story(ctx.db, "102", "Tesla recalls 2 million vehicles to fix Autopilot", ["TSLA"], t0 + timedelta(hours=1))
    add_story(ctx.db, "103", "Apple to hold its annual shareholder meeting", ["AAPL"], t0 + timedelta(hours=2))
    add_story(ctx.db, "104", "Ford beats estimates; Tesla in focus", ["F", "TSLA"], t0 + timedelta(hours=3))
    return ctx


def stored(ctx) -> dict[tuple[str, str], dict]:
    return {(r["news_id"], r["ticker"]): r for r in ctx.db.query("SELECT * FROM pro_labels")}


def model_with_bias(bias, sentiment_model: str = "fake-finbert") -> StudentModel:
    """A student that always reads the same way (its words don't matter): bias in CLASSES order."""
    return StudentModel(np.zeros((3, N_HASH + len(NUMERIC))), np.array(bias, dtype=float), np.zeros(len(NUMERIC)),
                        np.ones(len(NUMERIC)), {"sentiment_model": sentiment_model, "trained_at": "2026-10-01",
                                                "n_labels": 500})


# ------------------------------------------------------------------------------------------------ settings + schema
def test_new_settings_and_their_defaults():
    s = AppSettings()
    assert s.ai.pro_ai.label_while_market_open is False  # labelling never slows trading unless you say so
    assert s.ml.use_student is True and s.ml.shared_labels_pull is True
    assert s.schema_version == 5  # new fields have defaults: no config migration needed


def test_pro_labels_migration(tmp_path):
    import sqlite3

    path = tmp_path / "old.db"
    c = sqlite3.connect(path)
    for i, script in enumerate(MIGRATIONS[:13], start=1):  # a database from before shared training
        for statement in script.split(";"):
            if statement.strip():
                c.execute(statement)
        c.execute(f"PRAGMA user_version={i}")
    c.commit()
    c.close()
    db = Database(path)
    try:
        assert db.scalar("PRAGMA user_version") == len(MIGRATIONS) == 14
        cols = {r["name"] for r in db.query("PRAGMA table_info(pro_labels)")}
        assert {"news_id", "ticker", "direction", "confidence", "event", "model", "labelled_at", "headline_hash",
                "day", "status"} <= cols
    finally:
        db.close()


# ------------------------------------------------------------------------------------------------ teacher job
def test_parse_answer_only_takes_the_storys_own_valid_calls():
    text = json.dumps({"signals": [
        {"ticker": "NVDA", "direction": "bullish", "confidence": 90},
        {"ticker": "NVDA", "direction": "bearish", "confidence": 90},  # second call for the same stock: ignored
        {"ticker": "AMD", "direction": "bearish", "confidence": 70},  # not one of the story's stocks
        {"ticker": "TSLA", "direction": "up", "confidence": 70},
        {"ticker": "F", "direction": "neutral", "confidence": True},
        "junk"]})
    assert parse_answer(text, ["NVDA", "TSLA", "F"]) == {"NVDA": ("bullish", 90)}
    assert parse_answer("```json\n{\"signals\": []}\n```", ["NVDA"]) == {}
    assert parse_answer("not json", ["NVDA"]) is None
    assert parse_answer(json.dumps({"answer": 1}), ["NVDA"]) is None


def test_market_busy_from_just_before_the_open_to_the_close():
    assert market_busy(TUESDAY_11ET) and market_busy(TUESDAY_925ET)
    assert not market_busy(SATURDAY) and not market_busy(TUESDAY_5PM)
    assert not market_busy(datetime(2026, 10, 13, 13, 0, tzinfo=UTC))  # 9:00 ET


async def test_teacher_labels_old_news_and_continues_where_it_stopped(labelling):
    ctx = labelling
    engine = FakeTeacherEngine()
    job = TeacherJob(ctx, StubPro(ctx, engine), now=lambda: SATURDAY)
    assert job.status()["total"] == 4 and job.status()["remaining"] == 4
    assert await job.step() == 0.0 and await job.step() == 0.0
    assert [a["id"] for a in engine.asked] == ["104", "103"]  # newest first
    first = engine.asked[0]
    assert first["tickers"] == ["F", "TSLA"] and first["max_signals"] == 2 and first["timeout"] == teacher_mod.TIMEOUT
    assert first["now"] == datetime(2026, 9, 1, 17, 1, tzinfo=UTC)  # read as fresh news, the way it arrived
    assert first["market_open"] is True
    rows = stored(ctx)
    assert rows[("104", "F")]["direction"] == "bullish" and rows[("104", "F")]["confidence"] == 88
    # a stock Pro AI didn't name counts as neutral (confidence 0)
    assert rows[("104", "TSLA")]["direction"] == "neutral" and rows[("104", "TSLA")]["confidence"] == 0
    assert rows[("103", "AAPL")]["direction"] == "neutral"
    r = rows[("104", "F")]
    assert r["model"] == MODEL_KEY and r["status"] == "ok" and r["day"] == "2026-09-01"
    assert r["headline_hash"] == sl.headline_hash("Ford beats estimates; Tesla in focus")
    assert r["event"] == "earnings_beat"

    # the app restarts: a new job only asks about what's left
    engine2 = FakeTeacherEngine()
    job2 = TeacherJob(ctx, StubPro(ctx, engine2), now=lambda: SATURDAY)
    for _ in range(3):
        await job2.step()
    assert [a["id"] for a in engine2.asked] == ["102", "101"]
    assert await job2.step() == teacher_mod.DONE_RECHECK and job2.state == "done"
    rows = stored(ctx)
    assert rows[("101", "NVDA")]["direction"] == "bullish" and rows[("101", "NVDA")]["event"] == "contract"
    assert rows[("102", "TSLA")]["direction"] == "bearish" and rows[("102", "TSLA")]["event"] == "recall"
    status = job2.status()
    assert status["done"] == 4 and status["remaining"] == 0 and status["label_count"] == 5
    assert status["labels"] == {"bullish": 2, "bearish": 1, "neutral": 2}
    assert status["seconds_per_story"] is not None and status["eta_seconds"] == 0


async def test_teacher_pauses_while_the_market_is_open(labelling):
    ctx = labelling
    engine = FakeTeacherEngine()
    clock = {"now": TUESDAY_11ET}
    job = TeacherJob(ctx, StubPro(ctx, engine), now=lambda: clock["now"], idle=7.0)
    assert await job.step() == 7.0
    assert engine.asked == [] and job.state == "waiting" and "market is open" in job.message
    clock["now"] = TUESDAY_925ET  # just before the open counts too
    assert await job.step() == 7.0 and engine.asked == []
    clock["now"] = TUESDAY_5PM  # after the close it carries on by itself
    assert await job.step() == 0.0 and len(engine.asked) == 1
    clock["now"] = TUESDAY_11ET
    ctx.config.update({"ai": {"pro_ai": {"label_while_market_open": True}}})
    assert await job.step() == 0.0 and len(engine.asked) == 2


async def test_teacher_waits_for_pro_ai_and_lets_live_stories_go_first(labelling):
    ctx = labelling
    engine = FakeTeacherEngine()
    pro = StubPro(ctx, engine, ready=False)
    job = TeacherJob(ctx, pro, now=lambda: SATURDAY, idle=5.0)
    assert await job.step() == 5.0 and "Waiting for Pro AI" in job.message
    pro.ready, pro.waiting = True, True
    assert await job.step() == LIVE_FIRST and engine.asked == []
    pro.waiting, pro.testing = False, True
    assert await job.step() == LIVE_FIRST and engine.asked == [] and "Test my PC" in job.message
    pro.testing = False
    assert await job.step() == 0.0 and len(engine.asked) == 1


async def test_teacher_stops_after_repeated_failures_and_start_retries_them(labelling):
    ctx = labelling
    add_story(ctx.db, "105", "Ford misses estimates", ["F"], datetime(2026, 9, 2, 14, tzinfo=UTC))
    engine = FakeTeacherEngine(fail=True)
    job = TeacherJob(ctx, StubPro(ctx, engine), now=lambda: SATURDAY)
    job._set_wanted(True)
    results = [await job.step() for _ in range(FAILS_TO_STOP)]
    assert results[-1] is None and job.state == "stopped" and "took too long" in job.message
    assert not job.wanted
    assert ctx.db.scalar("SELECT COUNT(DISTINCT news_id) FROM pro_labels WHERE status = 'failed'") == FAILS_TO_STOP
    assert job.status()["failed"] == FAILS_TO_STOP
    engine.fail = False
    job.start()  # failed stories get another try
    job.pause()
    assert ctx.db.scalar("SELECT COUNT(*) FROM pro_labels WHERE status = 'failed'") == 0


async def test_teacher_background_loop_start_pause_and_resume(labelling):
    import asyncio

    ctx = labelling
    engine = FakeTeacherEngine()
    job = TeacherJob(ctx, StubPro(ctx, engine), now=lambda: SATURDAY, idle=0.01)
    job.start()
    for _ in range(200):
        if job.state == "done":
            break
        await asyncio.sleep(0.01)
    assert job.state == "done" and len(engine.asked) == 4 and job.wanted
    job.pause()
    assert not job.running and not job.wanted and job.state == "off"
    job.resume_if_wanted()
    assert not job.running  # paused stays paused after a restart
    job._set_wanted(True)
    job.resume_if_wanted()
    assert job.running
    await job.stop()
    assert not job.running and job.wanted  # closing the app doesn't count as pausing


def test_labels_api(client, labelling):
    ctx = labelling
    job = TeacherJob(ctx, StubPro(ctx, FakeTeacherEngine()), now=lambda: SATURDAY)
    ctx.services["pro_ai"] = SimpleNamespace(teacher=job)
    r = client.get("/api/pro-ai/labels")
    assert r.status_code == 200 and r.json()["total"] == 4
    r = client.post("/api/pro-ai/labels/start")
    assert r.status_code == 200 and r.json()["wanted"] is True
    r = client.post("/api/pro-ai/labels/pause")
    assert r.status_code == 200 and r.json()["wanted"] is False
    ctx.db.execute("DELETE FROM ml_samples")
    r = client.post("/api/pro-ai/labels/start")
    assert r.status_code == 409 and "Train the price model first" in r.json()["detail"]


# ------------------------------------------------------------------------------------------------ student model
WORDS = {"bullish": ["soars", "wins", "upgrade", "beats", "record"],
         "bearish": ["plunges", "recall", "misses", "probe", "slashes"],
         "neutral": ["meeting", "schedules", "names", "hosts", "files"]}
FILL = ["the", "company", "said", "on", "monday", "shares", "investors", "quarter", "new", "plan", "with", "its"]


def synthetic(n: int = 300, seed: int = 3, start: datetime = datetime(2026, 1, 5, 15, tzinfo=UTC)):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        lab = rng.choice(["bullish", "bearish", "neutral", "neutral"])
        text = "companyx " + rng.choice(WORDS[lab]) + " " + " ".join(rng.sample(FILL, 4))
        out.append(StudentSample(StudentFeatures(text, SentimentScores(0.3, 0.3, 0.4)), lab,
                                 start + timedelta(hours=i), str(1000 + i)))
    return out


@pytest.fixture
def quick(monkeypatch):
    monkeypatch.setattr(st, "C_GRID", (1.0,))


def test_student_learns_tests_on_newest_and_saves_without_pickle(tmp_path, quick):
    samples = synthetic()
    model, report = train_student(samples, "fake-finbert")
    assert report["n_valid"] == 60 and report["n_train"] == 240
    assert report["valid_from"] >= samples[240].at.date().isoformat()  # the newest 20%
    assert report["accuracy"] >= 90 and report["accuracy"] > report["majority_accuracy"]
    assert report["current_accuracy"] is None and report["not_worse"] is True
    assert set(report["per_class"]) == set(CLASSES)
    assert model.meta["n_labels"] == 300 and model.meta["sentiment_model"] == "fake-finbert"
    probe = [StudentFeatures("companyx soars after the quarter", SentimentScores(0.3, 0.3, 0.4)),
             StudentFeatures("companyx recall probe", SentimentScores(0.3, 0.3, 0.4)),
             StudentFeatures("companyx schedules its meeting", SentimentScores(0.3, 0.3, 0.4))]
    assert model.predict(probe) == ["bullish", "bearish", "neutral"]

    folder = tmp_path / "student"
    model.save(folder)
    assert sorted(p.name for p in folder.iterdir()) == ["student.json", "student.npz"]
    with np.load(folder / "student.npz", allow_pickle=False) as data:  # plain numbers only
        assert set(data.files) == {"coef", "intercept", "mean", "std", "seen"}
        assert all(data[k].dtype != object for k in data.files)
    loaded = StudentModel.load(folder)
    assert np.allclose(loaded.predict_proba(probe), model.predict_proba(probe), atol=1e-4)
    assert loaded.saw([s.group for s in samples]).all() and not loaded.saw(["123456789"]).any()
    meta = json.loads((folder / "student.json").read_text())
    meta["version"] = 99
    (folder / "student.json").write_text(json.dumps(meta))
    assert StudentModel.load(folder) is None  # from another version: train again
    st.delete_student(folder)
    assert StudentModel.load(folder) is None


def test_student_split_never_leaks_and_needs_enough_labels(quick):
    samples = synthetic(100)
    for s in samples[50:52]:
        s.group = "same-story"
    train, valid = split_by_time(samples, 0.49)
    assert max(s.at for s in train) < min(s.at for s in valid)
    assert not ({s.group for s in train} & {s.group for s in valid})
    with pytest.raises(ValueError, match="at least 200"):
        train_student(synthetic(150), "fake")
    no_bearish = [s for s in synthetic(400) if s.label != "bearish"]
    with pytest.raises(ValueError, match="each kind"):
        train_student(no_bearish, "fake")


def test_student_is_compared_with_the_one_in_use_on_labels_it_never_saw(quick):
    samples = synthetic()
    good, good_report = train_student(samples, "fake")
    # newer stories the one in use never learned from, with a worse teacher for the older ones
    noisy = synthetic(seed=9, start=datetime(2026, 3, 1, 15, tzinfo=UTC))
    for s in noisy:
        s.group = str(int(s.group) + 1000)
    for s in noisy[:240]:
        s.label = random.Random(s.group).choice(CLASSES)
    _worse, report = train_student(noisy, "fake", current=good)
    assert report["compare_basis"] == "same_labels" and report["compared_on"] == 60
    assert report["current_accuracy"] > report["new_accuracy"] and report["not_worse"] is False
    # the same stories again: the one in use learned from all of them (it would score itself near 100%), so the
    # bar is what it got on its own test when it was trained
    _again, report = train_student(samples, "fake", current=good)
    assert report["compare_basis"] == "own_test" and report["current_accuracy"] == good_report["accuracy"]
    assert report["not_worse"] is True
    # a student that doesn't know which stories it learned from is never compared on labels it may have seen
    unknown = model_with_bias([0, 3, 0])
    unknown.meta["report"] = {"accuracy": 99.0}
    _x, report = train_student(noisy, "fake", current=unknown)
    assert report["compare_basis"] == "own_test" and report["not_worse"] is False


def test_student_scores_the_test_my_pc_headlines(ctx, quick):
    model = model_with_bias([0, 3, 0])  # always neutral

    def predict(texts):
        return [SentimentScores(0.1, 0.1, 0.8) for _ in texts]

    res = st.score_samples(model, predict, make_tickers(ctx.db))
    assert res["total"] == 23 and res["right"] == 6  # the six neutral answers


# ------------------------------------------------------------------------------------------------ engine
@pytest.fixture
async def student_env(ctx, tmp_path):
    broker = FakeBroker(prices={"NVDA": 180.0, "AAPL": 230.0, "TSLA": 250.0, "F": 11.0})
    trader = Trader(ctx, broker_factory=lambda: broker)
    ctx.services["trader"] = trader
    await trader.connect()
    svc, fake = fake_service(tmp_path)
    engine = LocalMLEngine(ctx, sentiment=svc, model_dir=tmp_path / "pm", student_dir=tmp_path / "student")
    pipeline = Pipeline(ctx, tickers=make_tickers(ctx.db), local=engine)
    ctx.services["pipeline"] = pipeline
    yield ctx, pipeline, broker
    await trader.stop()


NO_EVENT = "Apple shares in focus as investors weigh the outlook"


async def test_engine_without_a_student_is_unchanged(student_env):
    ctx, pipeline, broker = student_env
    assert pipeline.local.active_student() is None
    out = await run(pipeline, news(NO_EVENT))
    assert out["signals"] == []  # FinBERT reads neutral wording: no signal (as before)
    out = await run(pipeline, news("Nvidia wins $10 billion government AI contract"))
    assert out["signals"][0]["confidence"] == 89


async def test_student_replaces_wording_only_direction_but_stays_manual_review(student_env):
    ctx, pipeline, broker = student_env
    pipeline.local.set_student(model_with_bias([0, 0, 3]))  # reads everything bullish, 91% sure
    out = await run(pipeline, news(NO_EVENT))
    s = out["signals"][0]
    assert s["ticker"] == "AAPL" and s["direction"] == "bullish"
    assert s["confidence"] == 79 and s["action"] == "review" and s["traded"] == 0  # never auto-traded
    assert "Student model (learned from Pro AI's labels): 91% bullish" in s["reasoning"]
    assert "Sent for review" in s["reasoning"]
    assert not [c for c in broker.calls if c[0] == "submit_bracket"]
    assert ctx.db.query_one("SELECT model FROM analyses ORDER BY id DESC")["model"].startswith(
        "local: FinBERT + student model")
    # a student that reads it neutral: no signal
    from newstrader.ai.prefilter import prefilter

    pipeline.local.set_student(model_with_bias([0, 3, 0]))
    item = news(NO_EVENT)
    payload = pipeline.local._analyze_sync(item, prefilter(item.text, pipeline.tickers, [], False), datetime.now(UTC))
    assert payload["signals"] == [] and payload["details"][0]["student"]["neutral"] > 0.9


async def test_student_disagreeing_with_an_event_only_lowers_confidence(student_env):
    ctx, pipeline, broker = student_env
    headline = "Nvidia wins $10 billion government AI contract"
    item = news(headline)

    async def conf():
        from newstrader.ai.prefilter import prefilter

        await pipeline.local.ensure_ready()
        pre = prefilter(item.text, pipeline.tickers, [], False)
        payload = pipeline.local._analyze_sync(item, pre, datetime.now(UTC))
        return payload["signals"][0]["confidence"], payload["signals"][0]["direction"]

    assert await conf() == (89, "bullish")
    pipeline.local.set_student(model_with_bias([0, 0, 3]))  # agrees: never raises it
    assert await conf() == (89, "bullish")
    pipeline.local.set_student(model_with_bias([3, 0, 0]))  # confidently the other way: 8 off
    assert await conf() == (81, "bullish")
    pipeline.local.set_student(model_with_bias([0, 3, 0]))  # confidently neutral: 4 off
    assert await conf() == (85, "bullish")
    pipeline.local.set_student(model_with_bias([0.5, 0, 0]))  # not sure enough (45%): no change
    assert await conf() == (89, "bullish")


async def test_student_not_used_for_tv_when_off_or_trained_on_other_scores(student_env):
    ctx, pipeline, broker = student_env
    engine = pipeline.local
    await engine.ensure_ready()
    from newstrader.ai.prefilter import prefilter
    from newstrader.sources.base import NewsItem

    engine.set_student(model_with_bias([0, 0, 3]))
    tv = NewsItem(source_id="tv", source_type="stream", source_name="TV", external_id="x", title="",
                  body="New:\n[10:00:00] Apple shares in focus as investors weigh the outlook.", kind="transcript",
                  published_at=datetime.now(UTC))
    pre = prefilter(tv.text, pipeline.tickers, [], False)
    payload = engine._analyze_sync(tv, pre, datetime.now(UTC))
    assert payload["signals"] == [] and all(d["student"] is None for d in payload["details"])
    ctx.config.update({"ml": {"use_student": False}})
    assert engine.active_student() is None and engine.student_status()["why_inactive"] == "turned off in Settings"
    ctx.config.update({"ml": {"use_student": True}})
    engine.set_student(model_with_bias([0, 0, 3], sentiment_model="lexicon-v1"))
    assert engine.active_student() is None and "different sentiment model" in engine.student_status()["why_inactive"]
    ctx.config.update({"ml": {"event_rules": False}})  # v0.2 behaviour (wording only): no student either
    engine.set_student(model_with_bias([0, 0, 3]))
    out = await run(pipeline, news(NO_EVENT))
    assert out["signals"] == []


def test_engine_loads_the_saved_student(ctx, tmp_path, quick):
    svc, _fake = fake_service(tmp_path)
    model, _ = train_student(synthetic(), "fake-finbert")
    model.save(tmp_path / "student")
    engine = LocalMLEngine(ctx, sentiment=svc, model_dir=tmp_path / "pm", student_dir=tmp_path / "student")
    assert engine.current_student() is not None and engine.student_status()["n_labels"] == 300
    assert engine.model_status()["student"]["bundled"] is False


# ------------------------------------------------------------------------------------------------ the file format
TODAY = sl.utc_today()
DAY = (TODAY - timedelta(days=40)).isoformat()


def good_rows(n: int = 30) -> list[dict]:
    dirs = ["bullish", "bearish", "neutral"]
    return [{"v": 1, "id": 5000 + i, "h": f"{i:016x}", "t": "NVDA" if i % 2 else "BRK.B", "d": dirs[i % 3],
             "c": i % 101, "e": "earnings_beat" if i % 3 == 0 else "", "m": MODEL_KEY, "day": DAY}
            for i in range(n)]


NAME = sl.file_name(TODAY, "1a2b3c4d")


def check(rows, name=NAME, **kw):
    return sl.check_file(name, sl.dump(rows), TODAY, **kw)


def test_good_file_passes_and_round_trips(tmp_path):
    res = check(good_rows())
    assert res.ok, res.summary
    assert res.n_rows == 30 and res.code == "1a2b3c4d" and res.mix == {"bullish": 10, "bearish": 10, "neutral": 10}
    path = tmp_path / NAME
    path.write_bytes(sl.dump(good_rows()))
    assert sl.read_rows(path) == good_rows()
    assert sl.dump(good_rows()) == sl.dump(good_rows())  # same rows, same bytes
    assert sl.NAME_RE.match(NAME) and NAME.startswith(f"newstrader-labels-{TODAY.isoformat()}-")


def _bad(**change) -> list[dict]:
    rows = good_rows(5)
    rows[2] = {**rows[2], **change}
    return rows


@pytest.mark.parametrize("rows, why", [
    (_bad(v=2), "v must be 1"),
    (_bad(id="5002"), "id must be"),
    (_bad(id=True), "id must be"),
    (_bad(h="XYZ"), "h must be"),
    (_bad(t="nvda"), "t must be a ticker"),
    (_bad(t="TOOLONG"), "t must be a ticker"),
    (_bad(d="up"), "d must be"),
    (_bad(c=101), "c must be"),
    (_bad(c=50.5), "c must be"),
    (_bad(c=False), "c must be"),
    (_bad(e="made_up_event"), "e must be"),
    (_bad(m="unknown-model"), "m must be"),
    (_bad(day=(TODAY + timedelta(days=5)).isoformat()), "in the future"),
    (_bad(day=(TODAY - timedelta(days=4 * 365)).isoformat()), "more than 3 years"),
    (_bad(day="yesterday"), "day must be"),
])
def test_bad_rows_are_rejected(rows, why):
    res = check(rows)
    assert not res.ok and why in res.summary and "line 3" in res.summary


def test_missing_and_extra_keys_duplicates_and_bad_lines_are_rejected():
    rows = good_rows(5)
    del rows[0]["h"]
    rows[1]["headline"] = "Nvidia wins"  # no text may be shared
    rows[4] = dict(rows[3])
    res = check(rows)
    assert not res.ok
    assert "line 1: missing h" in res.summary and "line 2: unexpected headline" in res.summary
    assert "line 5: news 5003 / NVDA is in the file twice" in res.summary
    text = "\n".join(json.dumps(r) for r in good_rows(3)) + "\nnot json\n[1, 2]\n" + "x" * 2000 + "\n"
    res = sl.check_file(NAME, gzip.compress(text.encode()), TODAY)
    assert "line 4: not valid JSON" in res.summary and "line 5: not a JSON object" in res.summary
    assert "line 6: longer than" in res.summary


def test_bad_files_are_rejected(monkeypatch):
    assert "the name must look like" in check(good_rows(), name="labels.jsonl.gz").summary
    assert "the name must look like" in check(good_rows(), name=NAME.replace(".gz", "")).summary
    future = sl.file_name(TODAY + timedelta(days=9), "1a2b3c4d")
    assert "in the future" in check(good_rows(), name=future).summary
    assert "not a gzip" in sl.check_file(NAME, b"plain text", TODAY).summary
    assert "no labels" in sl.check_file(NAME, gzip.compress(b""), TODAY).summary
    monkeypatch.setattr(sl, "MAX_BYTES", 200)
    assert "bigger than" in check(good_rows(100)).summary
    monkeypatch.setattr(sl, "MAX_BYTES", 20 * 1024 * 1024)
    monkeypatch.setattr(sl, "MAX_ROWS", 10)
    assert "more than 10 labels" in check(good_rows(11)).summary
    monkeypatch.setattr(sl, "MAX_UNPACKED", 10_000)
    assert "unpacks to more than" in sl.check_file(NAME, gzip.compress(b"\n" * 20_000), TODAY).summary


def test_direction_mix_must_be_believable():
    rows = good_rows(600)
    for r in rows[:560]:
        r["d"] = "bullish"
    res = check(rows)
    assert not res.ok and "96% of the labels are bullish" in res.summary
    assert check(rows[:400]).ok  # all bullish, but small files have no limit
    trimmed, dropped = sl.trim_mix(rows, 0.85)
    assert dropped > 0 and check(trimmed).ok


def test_unknown_event_and_model_pass_the_apps_forward_compatible_check():
    rows = _bad(e="new_kind_from_v9", m="future-model-1")
    assert not check(rows).ok  # GitHub's check knows the current lists
    assert check(rows, known_only=False).ok  # an older app still uses a newer version's file


def test_check_folder_and_script(tmp_path):
    folder = tmp_path / "labels"
    folder.mkdir()
    (folder / "README.md").write_text("hi")
    (folder / NAME).write_bytes(sl.dump(good_rows()))
    script = ROOT / "scripts" / "check_shared_labels.py"
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import check_shared_labels
    finally:
        sys.path.pop(0)
    assert check_shared_labels.main([str(folder)]) == 0
    assert check_shared_labels.main([str(tmp_path / "missing")]) == 0
    (folder / "notes.txt").write_text("x")
    (folder / "sub").mkdir()
    (folder / "sub" / NAME).write_bytes(sl.dump(good_rows()))
    secret = tmp_path / "secret.txt"
    secret.write_text("not for the check")
    link = folder / sl.file_name(TODAY, "9f9f9f9f")
    try:
        link.symlink_to(secret)
    except OSError:  # (Windows without the right to make links)
        link = None
    results = {r.name: r for r in sl.check_folder(folder, TODAY)}
    assert results[NAME].ok and not results["notes.txt"].ok and not results[f"sub/{NAME}"].ok
    if link is not None:  # a link is never followed
        assert "links aren't allowed" in results[link.name].summary
    assert check_shared_labels.main([str(folder)]) == 1
    # GitHub runs it with no packages installed: only the standard library may be needed
    out = subprocess.run([sys.executable, "-I", "-S", str(script), str(folder)], capture_output=True, text=True,
                         timeout=60)
    assert out.returncode == 1 and "REJECTED" in out.stdout and "Traceback" not in out.stderr
    # the folder in the repo itself passes
    assert check_shared_labels.main([str(ROOT / "shared" / "labels")]) == 0


# ------------------------------------------------------------------------------------------------ share my labels
def add_label(db, nid, ticker, direction, conf=80, day=DAY, status="ok", model=MODEL_KEY, event="",
              headline=None):
    db.execute("INSERT INTO pro_labels (news_id, ticker, direction, confidence, event, model, labelled_at, "
               "headline_hash, day, status) VALUES (?,?,?,?,?,?,?,?,?,?)",
               (nid, ticker, direction, conf, event, model, iso(), sl.headline_hash(headline or f"headline {nid}"),
                day, status))


def test_share_my_labels_saves_a_checked_file_and_opens_github(ctx, tmp_path):
    seen = {}
    svc = SharedTraining(ctx, reveal=lambda p: seen.setdefault("revealed", p),
                         open_url=lambda u: seen.setdefault("url", u), downloads=tmp_path)
    with pytest.raises(ValueError, match="no labels to share"):
        svc.export()
    add_label(ctx.db, "24843171", "NKE", "bearish", 82, event="earnings_miss")
    add_label(ctx.db, "24843172", "AAPL", "neutral", 0)
    add_label(ctx.db, "24843173", "TSLA", None, None, status="failed")  # never shared
    add_label(ctx.db, "abc", "TSLA", "bullish")  # not an Alpaca news id
    add_label(ctx.db, "24843174", "F", "bullish", model="test-model")  # not a Pro AI model
    out = svc.export()
    path = Path(out["file"])
    assert path.parent == tmp_path and seen["revealed"] == path and seen["url"] == sharing.UPLOAD_URL
    assert sharing.UPLOAD_URL == ("https://github.com/Harryisadag/newstrader/upload/claude/zen-gauss-5ktkek/"
                                  "shared/labels")
    assert out["rows"] == 2 and out["skipped"] == 2 and len(out["steps"]) == 3
    assert "Propose changes" in out["steps"][1] and "Create pull request" in out["steps"][2]
    res = sl.check_file(path.name, path.read_bytes(), TODAY)
    assert res.ok, res.summary
    lines = gzip.decompress(path.read_bytes()).decode().splitlines()
    rows = [json.loads(line) for line in lines]
    assert all(set(r) == set(sl.KEYS) for r in rows)
    assert rows[0] == {"v": 1, "id": 24843172, "h": sl.headline_hash("headline 24843172"), "t": "AAPL",
                       "d": "neutral", "c": 0, "e": "", "m": MODEL_KEY, "day": DAY}
    assert rows[1]["e"] == "earnings_miss" and rows[1]["c"] == 82
    assert "Nike" not in path.read_bytes().decode("latin-1")  # no text
    assert svc.export()["name"] == out["name"]  # the same random code every time on this PC


def test_share_trims_a_lopsided_mix(ctx, tmp_path):
    svc = SharedTraining(ctx, reveal=lambda p: None, open_url=lambda u: None, downloads=tmp_path)
    for i in range(1200):
        add_label(ctx.db, str(70000 + i), "NVDA", "neutral" if i >= 100 else "bullish")
    out = svc.export()
    rows = sl.read_rows(Path(out["file"]))
    assert out["trimmed"] > 0 and sum(r["d"] == "neutral" for r in rows) <= 0.85 * len(rows) + 1
    assert sl.check_file(Path(out["file"]).name, Path(out["file"]).read_bytes(), TODAY).ok


# ------------------------------------------------------------------------------------------------ everyone's labels
class FakeGitHub:
    """The contents API and raw.githubusercontent.com, for a list of (name, bytes, sha-to-report)."""

    def __init__(self):
        self.files: list[tuple[str, bytes, str]] = []
        self.etag = '"v1"'
        self.requests: list[httpx.Request] = []
        self.status = 200

    def add(self, name: str, data: bytes, sha: str | None = None) -> None:
        self.files.append((name, data, sha or sl.blob_sha(data)))

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.host == "api.github.com":
            if self.status != 200:
                return httpx.Response(self.status)
            if request.headers.get("If-None-Match") == self.etag:
                return httpx.Response(304)
            listing = [{"name": n, "type": "file", "sha": sha, "size": len(d)} for n, d, sha in self.files]
            listing.append({"name": "README.md", "type": "file", "sha": "0" * 40, "size": 10})
            return httpx.Response(200, json=listing, headers={"ETag": self.etag})
        name = request.url.path.rsplit("/", 1)[-1]
        for n, d, _sha in self.files:
            if n == name:
                return httpx.Response(200, content=d)
        return httpx.Response(404)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler))


def raw_requests(gh: FakeGitHub) -> list[str]:
    return [str(r.url) for r in gh.requests if r.url.host == "raw.githubusercontent.com"]


def test_pull_checks_fingerprints_and_contents(ctx):
    gh = FakeGitHub()
    good = sl.dump(good_rows())
    gh.add(sl.file_name(TODAY, "aaaaaaaa"), good)
    gh.add(sl.file_name(TODAY, "bbbbbbbb"), good, sha="0" * 40)  # doesn't match GitHub's fingerprint
    bad = good_rows(5)
    bad[0]["d"] = "sideways"
    gh.add(sl.file_name(TODAY, "cccccccc"), sl.dump(bad))  # fails the checks
    svc = SharedTraining(ctx, client_factory=gh.client)
    res = svc.pull_sync()
    assert res["added"] == [sl.file_name(TODAY, "aaaaaaaa")] and res["error"] == ""
    assert res["files"] == 1 and res["labels"] == 30 and res["contributors"] == 1
    assert [r["name"] for r in res["rejected"]] == [sl.file_name(TODAY, "cccccccc")]
    assert "d must be" in res["rejected"][0]["why"]
    folder = sharing.cache_dir()
    assert (folder / sl.file_name(TODAY, "aaaaaaaa")).read_bytes() == good
    assert not (folder / sl.file_name(TODAY, "bbbbbbbb")).exists()
    listing = gh.requests[0]
    assert str(listing.url).startswith("https://api.github.com/repos/Harryisadag/newstrader/contents/shared/labels")
    assert listing.url.params["ref"] == "claude/zen-gauss-5ktkek" and "If-None-Match" not in listing.headers
    assert raw_requests(gh)[0].startswith("https://raw.githubusercontent.com/Harryisadag/newstrader/refs/heads/"
                                          "claude/zen-gauss-5ktkek/shared/labels/")
    # a fingerprint mismatch may be a file that changed meanwhile: listed again next time (no ETag kept)
    assert svc._state()["etag"] is None
    gh.requests.clear()
    svc.pull_sync()
    assert "If-None-Match" not in gh.requests[0].headers
    assert raw_requests(gh) == [sharing.RAW_URL.format(name=sl.file_name(TODAY, "bbbbbbbb"))]  # rejected: not again


def test_pull_honours_the_etag_and_follows_removals(ctx):
    gh = FakeGitHub()
    name = sl.file_name(TODAY, "aaaaaaaa")
    gh.add(name, sl.dump(good_rows()))
    svc = SharedTraining(ctx, client_factory=gh.client)
    assert svc.pull_sync()["added"] == [name] and svc._state()["etag"] == '"v1"'
    gh.requests.clear()
    res = svc.pull_sync()
    assert res["unchanged"] and gh.requests[0].headers["If-None-Match"] == '"v1"'
    assert raw_requests(gh) == []  # nothing downloaded again
    assert res["files"] == 1 and svc.pull_due() is False
    # the owner removed the file: it's gone here too
    gh.files.clear()
    gh.etag = '"v2"'
    res = svc.pull_sync()
    assert res["removed"] == [name] and res["files"] == 0 and not (sharing.cache_dir() / name).exists()
    gh.requests.clear()
    assert svc.pull_sync()["unchanged"] and gh.requests[0].headers["If-None-Match"] == '"v2"'  # still nothing
    # rate limited: a plain message, and the old ETag is kept for next time
    gh.status = 403
    res = svc.pull_sync()
    assert "slow down" in res["error"] and svc._state()["etag"] == '"v2"' and svc.pull_due() is False
    assert svc.pull_due(datetime.now(UTC) + timedelta(hours=4)) is True  # retried sooner after a problem
    gh.status = 404  # no shared/labels folder (yet)
    res = svc.pull_sync()
    assert res["error"] == "" and res["files"] == 0 and res["added"] == []


def test_pull_api_and_status(client, ctx):
    gh = FakeGitHub()
    gh.add(sl.file_name(TODAY, "aaaaaaaa"), sl.dump(good_rows()))
    ctx.services["shared_training"] = SharedTraining(ctx, client_factory=gh.client)
    r = client.post("/api/ml/shared/pull")
    assert r.status_code == 200 and r.json()["files"] == 1
    r = client.get("/api/ml/shared")
    body = r.json()
    assert r.status_code == 200 and body["shared"]["labels"] == 30 and body["own_labels"] == 0
    assert body["pull_daily"] is True and body["upload_url"] == sharing.UPLOAD_URL
    r = client.post("/api/ml/shared/export")
    assert r.status_code == 400 and "no labels to share" in r.json()["detail"]


# ------------------------------------------------------------------------------------------------ joining + training
def test_labels_are_matched_by_news_id_and_headline_fingerprint(ctx):
    t0 = datetime(2026, 9, 1, 14, tzinfo=UTC)
    add_story(ctx.db, "201", "Nvidia wins contract", ["NVDA"], t0)
    add_story(ctx.db, "202", "Tesla recalls cars", ["TSLA", "F"], t0)
    add_label(ctx.db, "201", "NVDA", "bullish", headline="Nvidia wins contract")  # own label
    folder = sharing.cache_dir()
    rows = [
        {"v": 1, "id": 201, "h": sl.headline_hash("Nvidia wins contract"), "t": "NVDA", "d": "bearish", "c": 60,
         "e": "", "m": MODEL_KEY, "day": DAY},  # own label wins over this one
        {"v": 1, "id": 202, "h": sl.headline_hash("Tesla recalls cars"), "t": "TSLA", "d": "bearish", "c": 90,
         "e": "recall", "m": MODEL_KEY, "day": DAY},
        {"v": 1, "id": 202, "h": sl.headline_hash("Tesla recalls trucks"), "t": "F", "d": "bullish", "c": 70,
         "e": "", "m": MODEL_KEY, "day": DAY},  # someone else's copy had another headline: skipped
        {"v": 1, "id": 999, "h": sl.headline_hash("x"), "t": "AAPL", "d": "bullish", "c": 70, "e": "",
         "m": MODEL_KEY, "day": DAY},  # a story this PC never downloaded
    ]
    other = [dict(rows[1], d="neutral", c=40)]  # a second contributor disagrees on TSLA (1 vs 1: the surer wins)
    files = [(TODAY, "aaaaaaaa", rows), (TODAY, "bbbbbbbb", other),
             (TODAY - timedelta(days=3), "bbbbbbbb", other)]  # its older file repeats the label: still one vote
    for day, code, rs in files:
        name = sl.file_name(day, code)
        (folder / name).write_bytes(sl.dump(rs))
        index = sharing.load_index(folder) | {name: {"sha": "x", "rows": len(rs), "code": code, "size": 1}}
        (folder / sharing.INDEX).write_text(json.dumps(index))
    labels, left_out = collect_labels(ctx.db, folder)
    assert len(labels) == 7 and left_out == 0
    assert collect_labels(ctx.db, folder, sharing.downloaded_ids(ctx.db))[1] == 1  # news 999: not downloaded here
    joined, counts = join_labels(ctx.db, labels)
    assert counts == {"labels": 7, "matched": 2, "not_downloaded": 1, "headline_differs": 1}
    picked = {(j["row"]["news_id"], j["label"]["t"]): j for j in joined}
    assert picked[("201", "NVDA")]["label"]["d"] == "bullish" and picked[("201", "NVDA")]["label"]["source"] == "own"
    assert picked[("202", "TSLA")]["label"]["d"] == "bearish"
    assert picked[("202", "TSLA")]["sources"] == {"aaaaaaaa", "bbbbbbbb"}


async def test_train_with_everyones_labels_end_to_end(student_env, quick, monkeypatch):
    ctx, pipeline, broker = student_env
    rng = random.Random(5)
    heads = {"bullish": ["{n} wins a big order", "{n} beats estimates", "{n} gets an upgrade"],
             "bearish": ["{n} recalls trucks", "{n} misses estimates", "{n} faces a probe"],
             "neutral": ["{n} schedules its meeting", "{n} names a director", "{n} hosts a call"]}
    t0 = datetime(2026, 3, 2, 14, tzinfo=UTC)
    for i in range(320):
        d = rng.choice(["bullish", "bearish", "neutral", "neutral"])
        sym, name = rng.choice([("NVDA", "Nvidia"), ("AAPL", "Apple"), ("TSLA", "Tesla"), ("F", "Ford")])
        headline = rng.choice(heads[d]).format(n=name) + f" ({i})"
        add_story(ctx.db, str(300000 + i), headline, [sym], t0 + timedelta(hours=3 * i))
        add_label(ctx.db, str(300000 + i), sym, d, headline=headline)
    svc = SharedTraining(ctx)
    ctx.services["shared_training"] = svc
    report = await svc.run_training()
    assert report["switched"] and report["used"]["own"] == 320 and report["used"]["contributors"] == 1
    assert report["accuracy"] >= 85 and report["samples_check"]["total"] == 23
    assert (pipeline.local.student_dir / "student.npz").exists()
    assert pipeline.local.active_student() is not None
    assert ctx.db.kv_get(sharing.REPORT_KEY)["switched"] is True
    assert any("student model active" in c["detail"] for c in ctx.state.components())

    # a new student that does worse than the one in use on the same labels is not switched in
    def worse(samples, sentiment_model, current=None, progress=None):
        model, rep = st.train_student(samples, sentiment_model, current, progress)
        rep.update(accuracy=10.0, new_accuracy=10.0, current_accuracy=90.0, compare_basis="same_labels",
                   compared_on=60, not_worse=False)
        return model, rep

    before = pipeline.local.current_student()
    monkeypatch.setattr(sharing, "train_student", worse)
    report = await svc.run_training()
    assert report["switched"] is False and pipeline.local.current_student() is before
    assert "new one agreed with Pro AI on 10.0% and the one in use on 90.0%" in svc.progress["message"]
    svc.delete_student()
    assert pipeline.local.current_student() is None


def test_student_api_guards(client, ctx):
    svc = SharedTraining(ctx)
    ctx.services["shared_training"] = svc
    r = client.post("/api/ml/student/train")
    assert r.status_code == 409 and "isn't running" in r.json()["detail"]
    ctx.services["ml_trainer"] = SimpleNamespace(running=True)
    ctx.services["pipeline"] = SimpleNamespace()
    r = client.post("/api/ml/student/train")
    assert r.status_code == 409 and "price model" in r.json()["detail"]
    assert client.post("/api/ml/student/cancel").status_code == 200


# ------------------------------------------------------------------------------------------------ eval script
def test_eval_script_scores_a_student_and_keeps_the_holdout_sealed(tmp_path, capsys, quick):
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import eval_detection
    finally:
        sys.path.pop(0)
    model = model_with_bias([0, 3, 0], sentiment_model="lexicon-v1")
    model.meta.update({"version": st.VERSION, "n_hash": N_HASH, "numeric": NUMERIC, "classes": list(CLASSES)})
    model.save(tmp_path / "student")
    assert eval_detection.main(["--set", "dev", "--student", str(tmp_path / "student")]) == 0
    out = capsys.readouterr().out
    assert "with student model" in out and "with event rules" in out
    assert eval_detection.main(["--set", "dev", "--student", str(tmp_path / "missing")]) == 1
    holdout = tmp_path / "holdout_x.jsonl"
    holdout.write_text((ROOT / "tests" / "fixtures" / "eval" / "dev.jsonl").read_text(encoding="utf-8")[:4000]
                       .rsplit("\n", 1)[0] + "\n", encoding="utf-8")
    assert eval_detection.main(["--set", str(holdout), "--student", str(tmp_path / "student"), "--mistakes", "5"]) == 0
    out = capsys.readouterr().out
    assert "Sealed set" in out and "expected" not in out
    finbert_student = model_with_bias([0, 3, 0], sentiment_model="finbert-x")
    finbert_student.meta.update(model.meta | {"sentiment_model": "finbert-x"})
    finbert_student.save(tmp_path / "fb")
    assert eval_detection.main(["--set", "dev", "--student", str(tmp_path / "fb")]) == 1
    assert "--sentiment finbert" in capsys.readouterr().out
