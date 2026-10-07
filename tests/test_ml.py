"""Local machine-learning engine: text targeting, sentiment models, labels, training and the live pipeline."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from newstrader.ai.pipeline import Pipeline
from newstrader.ai.prefilter import Candidate, prefilter
from newstrader.ml.dataset import Throttle, build_dataset, compute_label, load_samples
from newstrader.ml.engine import LocalMLEngine, time_of_day
from newstrader.ml.model import PriceModel, SampleFeatures, TrainingSample, time_split, train_price_model
from newstrader.ml.sentiment import FinbertOnnx, LexiconSentiment, SentimentScores, SentimentService
from newstrader.ml.text import mask_company, split_sentences, target_text
from newstrader.ml.trainer import ModelTrainer, TrainParams, default_range
from newstrader.sources.base import NewsItem
from newstrader.trading.fake_broker import FakeBroker
from newstrader.trading.trader import Trader

from .helpers import make_tickers

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_finbert"


class FakeSentiment:
    """Deterministic stand-in for FinBERT."""

    name = "finbert"
    model_id = "fake-finbert"

    def __init__(self):
        self.seen: list[str] = []

    def predict(self, texts):
        out = []
        for t in texts:
            self.seen.append(t)
            low = t.lower()
            if any(w in low for w in ("wins", "beats", "record")):
                out.append(SentimentScores(0.95, 0.02, 0.03))
            elif any(w in low for w in ("recall", "misses", "probe")):
                out.append(SentimentScores(0.03, 0.92, 0.05))
            else:
                out.append(SentimentScores(0.05, 0.05, 0.90))
        return out


def fake_service(tmp_path) -> tuple[SentimentService, FakeSentiment]:
    model = FakeSentiment()
    return SentimentService(tmp_path, loader=lambda _d: model), model


# ---------------------------------------------------------------- text
def test_target_text_finds_sentences_about_the_company(ctx):
    table = make_tickers(ctx.db)
    body = ("Markets were mixed on Tuesday. Nvidia said it won a $10 billion contract from the Pentagon. "
            "Separately, Ford recalled 5,000 trucks. Analysts called the Nvidia deal a milestone.")
    nv = Candidate("NVDA", "NVIDIA Corporation Common Stock", "nvidia")
    t = target_text("Chip stocks in focus", body, "text", nv, table)
    assert not t.in_headline and t.in_body
    assert "Pentagon" in t.snippet and "milestone" in t.snippet and "Ford" not in t.snippet
    f = target_text("Ford recalls trucks over brake issue", body, "text", Candidate("F", "Ford", "ford"), table)
    assert f.in_headline and "recalled" in f.snippet


def test_transcript_only_new_lines_count(ctx):
    table = make_tickers(ctx.db)
    body = ("Earlier (context only, already analysed):\n[10:00:01] Tesla shares fell sharply yesterday.\n\n"
            "New:\n[10:01:05] Turning to Apple, the company just announced a record buyback.")
    tsla = Candidate("TSLA", "Tesla", "tesla")
    assert target_text("", body, "transcript", tsla, table) is None
    aapl = target_text("", body, "transcript", Candidate("AAPL", "Apple", "apple"), table)
    assert aapl is not None and "buyback" in aapl.snippet and "Tesla" not in aapl.snippet


def test_mask_company_and_sentences(ctx):
    table = make_tickers(ctx.db)
    masked = mask_company("Apple (AAPL) beats estimates; Apple's shares rise", Candidate("AAPL", "Apple", "apple"),
                          table)
    assert "apple" not in masked and "aapl" not in masked and masked.count("companyx") == 3
    assert split_sentences("One. Two! [10:00] Three?\nFour") == ["One.", "Two!", "Three?", "Four"]


def test_time_of_day():
    assert time_of_day(datetime(2026, 10, 6, 13, 30, tzinfo=UTC)) == 0.0  # 9:30 ET
    assert time_of_day(datetime(2026, 10, 6, 20, 0, tzinfo=UTC)) == 1.0
    assert 0.4 < time_of_day(datetime(2026, 10, 6, 16, 45, tzinfo=UTC)) < 0.6


# ---------------------------------------------------------------- sentiment
def test_lexicon_never_reaches_default_buy_threshold():
    lex = LexiconSentiment()
    s = lex.predict(["Record profit, beats estimates, raises guidance, wins contract, upgrade, surge"])[0]
    assert s.label == "positive" and s.positive <= 0.79
    assert lex.predict(["Shares plunge after recall and fraud probe"])[0].label == "negative"
    assert lex.predict(["Company did not miss estimates"])[0].label == "positive"  # negation flips
    assert lex.predict(["The CEO will speak on Tuesday"])[0].label == "neutral"
    for s in lex.predict(["beats", "misses", "nothing", ""]):
        assert abs(s.positive + s.negative + s.neutral - 1) < 1e-9


def test_finbert_onnx_code_path_with_tiny_model():
    m = FinbertOnnx(FIXTURE)
    out = m.predict(["Shares of Apple jump after record profit", "Company files for bankruptcy",
                     "the results were in line", ""])
    assert [s.label for s in out] == ["positive", "negative", "neutral", "neutral"]
    for s in out:
        assert abs(s.positive + s.negative + s.neutral - 1) < 1e-6
    assert m.model_id.startswith("finbert-")


def test_sentiment_service_falls_back_to_word_list(tmp_path):
    def broken(_d):
        raise OSError("no internet")

    svc = SentimentService(tmp_path, loader=broken)
    svc.load("finbert")
    assert svc.ready and svc.model.name == "lexicon"
    assert "no internet" in svc.error and "FinBERT unavailable" in svc.describe()
    svc2, _ = fake_service(tmp_path)
    svc2.load("finbert")
    assert svc2.model_id == "fake-finbert" and svc2.error is None


def test_sentiment_service_without_loader_uses_real_download_path(tmp_path):
    svc = SentimentService(tmp_path)  # conftest blocks downloads -> must fall back, not crash
    svc.load("finbert")
    assert svc.model.name == "lexicon" and "no model downloads" in svc.error


# ---------------------------------------------------------------- live pipeline with the local engine
@pytest.fixture
async def local_env(ctx, tmp_path):
    broker = FakeBroker(prices={"NVDA": 180.0, "AAPL": 230.0, "TSLA": 250.0, "F": 11.0})
    trader = Trader(ctx, broker_factory=lambda: broker)
    ctx.services["trader"] = trader
    await trader.connect()
    svc, fake = fake_service(tmp_path)
    engine = LocalMLEngine(ctx, sentiment=svc, model_dir=tmp_path / "pm")
    pipeline = Pipeline(ctx, tickers=make_tickers(ctx.db), local=engine)
    ctx.services["pipeline"] = pipeline
    yield ctx, pipeline, fake, broker
    await trader.stop()


def news(title, body="", source="cnbc", symbols=None):
    return NewsItem(source_id=source, source_type="rss", source_name=source.upper(), external_id=title,
                    title=title, body=body, url=f"https://example.com/{abs(hash(title))}",
                    published_at=datetime.now(UTC), symbols=symbols or [])


async def run(pipeline, it):
    res = await pipeline.submit(it)
    if res["status"] == "queued":
        _, queued_item, pre = pipeline.queue.get_nowait()
        pipeline.queue.task_done()
        return await pipeline.analyze_item(queued_item, pre)
    return res


async def test_local_engine_is_the_default_and_trades_good_news(local_env):
    ctx, pipeline, fake, broker = local_env
    assert ctx.config.settings.ai.engine == "local"
    out = await run(pipeline, news("Nvidia wins $10 billion government AI contract"))
    assert out["status"] == "ok" and out["engine"] == "local"
    s = out["signals"][0]
    assert s["ticker"] == "NVDA" and s["direction"] == "bullish" and s["confidence"] == 95
    assert s["traded"] == 1 and broker.calls[-1][:3] == ("submit_bracket", "NVDA", "buy")
    a = ctx.db.query_one("SELECT * FROM analyses")
    assert a["engine"] == "local" and a["cost_usd"] == 0 and a["model"].startswith("local: FinBERT")
    assert ctx.db.query_one("SELECT engine FROM signals")["engine"] == "local"
    assert "wording reads 95% positive" in s["reasoning"]


async def test_local_engine_bearish_and_neutral(local_env):
    ctx, pipeline, fake, broker = local_env
    out = await run(pipeline, news("Tesla recall widens after safety probe"))
    s = out["signals"][0]
    assert s["direction"] == "bearish" and s["action"] != "bought"
    out = await run(pipeline, news("Apple to hold its annual shareholder meeting on Tuesday"))
    assert out["status"] == "ok" and out["signals"] == []  # neutral wording -> no signal
    assert not [c for c in broker.calls if c[0] == "submit_bracket"]


async def test_local_engine_skips_keyword_only_news_and_spend_cap(local_env):
    ctx, pipeline, fake, _ = local_env
    ctx.config.update({"ai": {"daily_spend_cap_usd": 0.0}})
    out = await pipeline.submit(news("Fed signals a rate cut in December"))
    assert out["status"] == "filtered"  # no company named; the local engine can't score that
    out = await run(pipeline, news("Nvidia wins another contract"))
    assert out["status"] == "ok"  # Claude's spend cap doesn't apply to the free engine


async def test_body_mentions_get_lower_confidence(local_env):
    ctx, pipeline, fake, _ = local_env
    out = await run(pipeline, news("Chipmakers in focus", "Nvidia wins a major cloud contract, sources said."))
    s = out["signals"][0]
    assert s["ticker"] == "NVDA" and s["confidence"] == round(95 * 0.85)
    assert "not the headline" in s["reasoning"]


async def test_test_the_ai_endpoint_uses_local_engine(local_env, client):
    ctx, pipeline, fake, broker = local_env
    r = client.post("/api/ai/test", json={"text": "Nvidia wins record order from Microsoft"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["engine"] == "local" and body["signals"][0]["action"] == "test"
    assert not broker.calls or broker.calls[-1][0] != "submit_bracket"


# ---------------------------------------------------------------- labels
def bars_around(t0: datetime, prices: list[tuple[int, float]]) -> list[dict]:
    return [{"t": (t0 + timedelta(minutes=m)).isoformat(), "o": p, "h": p, "l": p, "c": p, "v": 100}
            for m, p in prices]


def test_compute_label_vs_market():
    t = datetime(2026, 10, 6, 14, 0, tzinfo=UTC)
    close = datetime(2026, 10, 6, 20, 0, tzinfo=UTC)
    stock = bars_around(t, [(1, 100.0), (30, 101.0), (61, 102.0)])
    spy = bars_around(t, [(1, 500.0), (61, 505.0)])
    lab = compute_label(stock, spy, t, 60, close)
    assert lab.status == "ok" and lab.entry_price == 100 and lab.exit_price == 102
    assert lab.ret_pct == pytest.approx(2.0) and lab.spy_ret_pct == pytest.approx(1.0)
    assert lab.adj_ret_pct == pytest.approx(1.0)
    assert compute_label(stock, spy, close - timedelta(minutes=30), 60, close).status == "after_close"
    thin = bars_around(t, [(1, 100.0), (40, 102.0)])  # last trade 21 min before the exit time
    assert compute_label(thin, spy, t, 60, close).status == "no_bars"
    late = bars_around(t, [(9, 100.0), (61, 102.0)])  # first trade 8 min after we'd have bought
    assert compute_label(late, spy, t, 60, close).status == "no_bars"


# ---------------------------------------------------------------- synthetic history
SYMS = ["AAPL", "NVDA", "TSLA", "JPM", "MCD"]
NAMES = {"AAPL": "Apple", "NVDA": "Nvidia", "TSLA": "Tesla", "JPM": "JPMorgan", "MCD": "McDonald's"}
SLOTS = [(13, 45), (15, 0), (16, 15), (17, 30)]


def synthetic_broker(days: list[date], flip_every: int = 7) -> FakeBroker:
    """Each weekday: 5 stocks x 4 headlines. 'beats' headlines rise 1% vs the market, 'misses' fall 1%
    (every 7th one goes the 'wrong' way, so the model can't be perfect)."""
    fb = FakeBroker(prices={s: 100.0 for s in SYMS})
    bars = {s: [] for s in [*SYMS, "SPY"]}
    n = 0
    for d in days:
        for h, mi in SLOTS:
            t = datetime(d.year, d.month, d.day, h, mi, tzinfo=UTC)
            bars["SPY"] += bars_around(t, [(m, 500.0) for m in range(0, 71)])
            for k, sym in enumerate(SYMS):
                n += 1
                good = (n + k) % 2 == 0
                move = 1.0 if good else -1.0
                if n % flip_every == 0:
                    move = -move
                word = "beats estimates and raises its outlook" if good else "misses estimates and cuts its outlook"
                fb.news_items.append({"id": n, "headline": f"{NAMES[sym]} {word} ({d} {h}:{mi:02d})", "summary": "",
                                      "symbols": [sym], "created_at": t, "url": "", "source": "benzinga"})
                p1 = 100.0 * (1 + move / 100)
                bars[sym] += bars_around(t, [(m, 100.0 if m <= 30 else p1) for m in range(0, 71)])
    fb.news_items.sort(key=lambda x: x["created_at"])
    fb.bar_data = bars
    return fb


def weekdays(start: date, n: int) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def test_build_dataset_labels_and_caches(ctx):
    table = make_tickers(ctx.db)
    days = weekdays(date(2026, 3, 2), 3)
    fb = synthetic_broker(days)
    msgs = []
    stats = build_dataset(ctx.db, fb, table, days[0], days[-1], 60, 1000, lambda f, m: msgs.append(m), lambda: False,
                          throttle=Throttle(0))
    assert stats["days"] == 3 and stats["articles"] == 60 and stats["labelled"] == 60
    row = ctx.db.query_one("SELECT * FROM ml_samples WHERE news_id = '1'")
    assert row["label_status"] == "ok" and abs(row["adj_ret_pct"]) == pytest.approx(1.0)
    calls_before = len(fb.calls)
    again = build_dataset(ctx.db, fb, table, days[0], days[-1], 60, 1000, lambda f, m: None, lambda: False,
                          throttle=Throttle(0))
    assert again["days_cached"] == 3 and len(fb.calls) == calls_before  # nothing re-downloaded


def test_time_split_never_leaks_the_future():
    t0 = datetime(2026, 1, 5, 15, tzinfo=UTC)
    feats = SampleFeatures("x", SentimentScores(0.3, 0.3, 0.4), 1.0, True, True, 1, 0.5)
    samples = [TrainingSample(feats, i % 2, t0 + timedelta(minutes=30 * (i // 2)), 1.0, f"n{i // 2}")
               for i in range(400)]
    train, valid, test = time_split(samples, embargo_minutes=60)
    assert max(s.at for s in train) < min(s.at for s in valid) - timedelta(minutes=60)
    assert max(s.at for s in valid) < min(s.at for s in test) - timedelta(minutes=60)
    groups = [{s.group for s in part} for part in (train, valid, test)]
    assert not (groups[0] & groups[1]) and not (groups[1] & groups[2]) and not (groups[0] & groups[2])


def _samples_from(ctx, days, tmp_path):
    table = make_tickers(ctx.db)
    fb = synthetic_broker(days)
    build_dataset(ctx.db, fb, table, days[0], days[-1], 60, 5000, lambda f, m: None, lambda: False, Throttle(0))
    svc, _ = fake_service(tmp_path)
    svc.load("finbert")
    return load_samples(ctx.db, table, svc, 60, days[0], days[-1], 0.3, lambda f, m: None, lambda: False)


def test_train_save_load_roundtrip(ctx, tmp_path):
    days = weekdays(date(2026, 1, 5), 45)
    samples, counts = _samples_from(ctx, days, tmp_path)
    assert counts["used"] == 900 and counts["small_moves"] == 0
    model, report = train_price_model(samples, 60, 0.3, "fake-finbert", (days[0].isoformat(), days[-1].isoformat()))
    assert report["passed"], report
    assert report["auc"] > 0.75 and report["accuracy"] > 75
    assert report["n_test"] >= 150 and report["thresholds"][0]["threshold"] == 55
    model.save(tmp_path / "pm")
    loaded = PriceModel.load(tmp_path / "pm")
    probe = [s.feats for s in samples[:20]]
    assert np.allclose(model.predict_up(probe), loaded.predict_up(probe), atol=1e-5)
    good = [s.feats for s in samples if "beats" in s.feats.text][:5]
    assert (loaded.predict_up(good) > 0.6).all()


def test_too_little_history_is_refused(ctx, tmp_path):
    days = weekdays(date(2026, 1, 5), 3)
    samples, _ = _samples_from(ctx, days, tmp_path)
    with pytest.raises(ValueError, match="at least 200"):
        train_price_model(samples, 60, 0.3, "fake-finbert", ("a", "b"))


async def test_trainer_end_to_end_and_engine_uses_model(local_env, tmp_path):
    ctx, pipeline, fake, broker = local_env
    days = weekdays(date(2026, 1, 5), 45)
    synth = synthetic_broker(days)
    ctx.services["trader"].broker = synth
    trainer = ModelTrainer(ctx, throttle_factory=lambda: Throttle(0))
    ctx.services["ml_trainer"] = trainer
    run_id = ctx.db.insert("ml_runs", {"created_at": "x", "params": "{}", "status": "running"})
    report = await trainer.run(run_id, TrainParams(start=days[0], end=days[-1], max_articles=5000))
    assert report["passed"] and (tmp_path / "pm" / "price_model.npz").exists()
    row = ctx.db.query_one("SELECT * FROM ml_runs WHERE id = ?", (run_id,))
    assert row["status"] == "done" and row["progress"] == 1
    assert pipeline.local.active_price_model() is not None
    ml_status = next(c for c in ctx.state.components() if c["component"] == "ml")
    assert "price model active" in ml_status["detail"]
    ctx.services["trader"].broker = broker
    out = await run(pipeline, news("Nvidia beats estimates and raises its outlook"))
    s = out["signals"][0]
    assert s["direction"] == "bullish" and "Price model:" in s["reasoning"]
    assert out["model"].startswith("local: FinBERT + price model")
    # turning the trained model off falls back to sentiment only
    ctx.config.update({"ml": {"use_trained_model": "never"}})
    assert pipeline.local.active_price_model() is None


async def test_model_trained_on_other_sentiment_is_not_used(local_env, tmp_path):
    ctx, pipeline, fake, _ = local_env
    days = weekdays(date(2026, 1, 5), 45)
    samples, _ = _samples_from(ctx, days, tmp_path)
    model, _ = train_price_model(samples, 60, 0.3, "lexicon-v1", ("2026-01-05", "2026-03-06"))
    pipeline.local.set_price_model(model)
    await pipeline.local.ensure_ready()
    assert pipeline.local.active_price_model() is None
    assert "different sentiment model" in pipeline.local.model_status()["price_model"]["why_inactive"]


# ---------------------------------------------------------------- API
def test_ml_api(client, ctx):
    r = client.get("/api/ml/status")
    assert r.status_code == 503  # engine not started in this client fixture
    from newstrader.ai.pipeline import Pipeline as P

    ctx.services["pipeline"] = P(ctx, tickers=make_tickers(ctx.db))
    ctx.services["ml_trainer"] = ModelTrainer(ctx)
    r = client.get("/api/ml/status")
    assert r.status_code == 200 and r.json()["engine"] == "local" and r.json()["price_model"] is None
    today = date.today()
    r = client.post("/api/ml/train", json={"start": (today - timedelta(days=60)).isoformat(), "end": today.isoformat()})
    assert r.status_code == 400 and "before today" in r.json()["detail"]
    r = client.post("/api/ml/train", json={"start": "2026-01-01", "end": "2026-01-05"})
    assert r.status_code == 400
    start, end = default_range(180, today)
    assert end == today - timedelta(days=1) and (end - start).days == 179


def test_settings_engine_and_ml_section(client):
    r = client.put("/api/settings", json={"ai": {"engine": "claude"}, "ml": {"train_horizon_minutes": 30}})
    assert r.status_code == 200, r.text
    s = client.get("/api/settings").json()["settings"]
    assert s["ai"]["engine"] == "claude" and s["ml"]["train_horizon_minutes"] == 30
    assert client.put("/api/settings", json={"ai": {"engine": "gpt"}}).status_code in (400, 422)


def test_backtest_estimate_is_free_for_local(client, ctx):
    r = client.post("/api/backtest/estimate", json={"start": "2026-09-01", "end": "2026-09-05", "max_articles": 500})
    assert r.status_code == 200
    body = r.json()
    assert body["engine"] == "local" and body["max_cost"] == 0
    r = client.post("/api/backtest/estimate", json={"start": "2026-09-01", "end": "2026-09-05", "engine": "claude"})
    assert r.json()["max_cost"] > 0 and "cutoff" in r.json()["warning"]


def test_prefilter_keyword_only_flag():
    from newstrader.ai.tickers import TickerTable

    class _DB:
        def query(self, *a, **k):
            return []

    t = TickerTable(_DB())
    assert prefilter("Fed delivers a surprise rate cut", t, [], True).hit
    assert not prefilter("Fed delivers a surprise rate cut", t, [], False).hit
