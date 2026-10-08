"""Local machine-learning engine: text targeting, sentiment models, labels, training and the live pipeline."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from newstrader.ai.pipeline import Pipeline
from newstrader.ai.prefilter import Candidate, prefilter
from newstrader.ml import dataset as ds
from newstrader.ml.dataset import Throttle, build_dataset, compute_label, load_samples, sample_windows
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


@pytest.fixture(autouse=True)
def whole_hour_windows(monkeypatch):
    """The synthetic history has a few articles per hour; read whole hours so every one is fetched."""
    monkeypatch.setattr(ds, "SAMPLE_WINDOW", ds.SLICE)


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
    # contract win (0.72, +0.08 for $1bn+) + FinBERT agrees (+0.2 x 0.45) -> 89
    assert s["ticker"] == "NVDA" and s["direction"] == "bullish" and s["confidence"] == 89
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
    # contract win, unconfirmed ("sources said": x0.85), named only in the body (x0.85)
    assert s["ticker"] == "NVDA" and s["confidence"] == round((0.72 * 0.85 + 0.2 * 0.45) * 0.85 * 100)
    assert "not the headline" in s["reasoning"] and "Not confirmed yet" in s["reasoning"]


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


def test_build_dataset_falls_back_from_sip_and_survives_a_bad_day(ctx):
    table = make_tickers(ctx.db)
    days = weekdays(date(2026, 3, 2), 3)
    fb = synthetic_broker(days)
    fb.refuse_sip = True  # free account that can't read SIP history
    real_news = fb.news

    def flaky_news(start, end, *a, **k):
        if start.date() == days[1]:
            raise RuntimeError("429 too many requests")
        return real_news(start, end, *a, **k)

    fb.news = flaky_news
    stats = build_dataset(ctx.db, fb, table, days[0], days[-1], 60, 1000, lambda f, m: None, lambda: False,
                          throttle=Throttle(0))
    assert stats["price_feed"] == "settings" and stats["days_failed"] == 1 and stats["labelled"] == 40
    feeds = [c[2] for c in fb.calls if c[0] == "bars_multi"]
    assert feeds[0] == "sip" and set(feeds[1:]) == {None}
    fb.news = real_news  # next run only fetches the day that failed
    again = build_dataset(ctx.db, fb, table, days[0], days[-1], 60, 1000, lambda f, m: None, lambda: False,
                          throttle=Throttle(0))
    assert again["days_cached"] == 2 and again["labelled"] == 20


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
    days = weekdays(date(2026, 1, 5), 60)
    samples, counts = _samples_from(ctx, days, tmp_path)
    assert counts["used"] == 1200 and counts["small_moves"] == 0
    model, report = train_price_model(samples, 60, 0.3, "fake-finbert", (days[0].isoformat(), days[-1].isoformat()))
    assert report["passed"], report
    assert report["auc"] > 0.75 and report["accuracy"] > 75 and report["auc_low"] > 0.6
    assert report["test_days"] >= 10 and report["thresholds"][0]["threshold"] == 55
    assert report["thresholds"][0]["long_signals"] > 0 and report["thresholds"][0]["long_win_rate"] > 60
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
    days = weekdays(date(2026, 1, 5), 60)
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
    r = client.post("/api/ml/reload")
    assert r.status_code == 200 and r.json()["sentiment_model_id"]


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


# ---------------------------------------------------------------- review regressions
def test_other_companys_headline_doesnt_set_direction(ctx):
    table = make_tickers(ctx.db)
    t = target_text("Nvidia wins $10 billion Pentagon cloud contract",
                    "Tesla, which had also bid, lost out and said it would appeal the decision.", "text",
                    Candidate("TSLA", "Tesla", "tesla"), table)
    assert "Nvidia wins" not in t.snippet and "lost out" in t.snippet
    tagged_only = target_text("Chip stocks rally", "Shares rose across the sector today.", "text",
                              Candidate("NVDA", "NVIDIA", "tagged by source"), table)
    assert tagged_only.snippet.startswith("Chip stocks rally") and "sector" in tagged_only.snippet


def test_lexicon_counts_failures_as_negative():
    lex = LexiconSentiment()
    for text in ("Moderna phase 3 trial failed", "Boeing 737 fails FAA inspection"):
        assert lex.predict([text])[0].label == "negative", text


def test_sample_windows_cover_different_minutes(monkeypatch):
    monkeypatch.setattr(ds, "SAMPLE_WINDOW", timedelta(minutes=10))
    o = datetime(2026, 3, 2, 13, 30, tzinfo=UTC)
    last = o + timedelta(hours=5, minutes=28)
    offsets = set()
    for d in range(30):
        wins = sample_windows(o, last, f"2026-03-{d:02d}")
        assert len(wins) == 6 and all(w1 - w0 <= timedelta(minutes=10) for w0, w1 in wins)
        assert all(o <= w0 and w1 <= last for w0, w1 in wins)
        offsets.add(int((wins[0][0] - o).total_seconds() // 60))
    assert len(offsets) > 10  # not always the first minutes after the open


def test_dataset_skips_unfinished_session_and_keeps_delisted_stocks(ctx):
    table = make_tickers(ctx.db)
    days = weekdays(date(2026, 3, 2), 2)
    fb = synthetic_broker(days)
    t = datetime(days[0].year, days[0].month, days[0].day, 14, 0, tzinfo=UTC)
    fb.news_items.append({"id": 999, "headline": "Gone Corp agrees to be acquired at a 40% premium", "summary": "",
                          "symbols": ["GONE"], "created_at": t, "url": "", "source": "benzinga"})
    fb.bar_data["GONE"] = bars_around(t, [(m, 10.0 if m <= 30 else 14.0) for m in range(0, 71)])
    fb.news_items.sort(key=lambda x: x["created_at"])
    still_open = datetime(days[1].year, days[1].month, days[1].day, 19, 0, tzinfo=UTC)  # before day 2's close
    stats = build_dataset(ctx.db, fb, table, days[0], days[1], 60, 1000, lambda f, m: None, lambda: False,
                          throttle=Throttle(0), now=still_open)
    assert stats["days"] == 1  # the unfinished session is neither downloaded nor cached
    gone = ctx.db.query_one("SELECT * FROM ml_samples WHERE symbol = 'GONE'")
    assert gone and gone["label_status"] == "ok" and gone["ret_pct"] > 30  # not in today's ticker list, kept


def test_edited_article_is_timed_from_its_last_edit(ctx):
    table = make_tickers(ctx.db)
    days = weekdays(date(2026, 3, 2), 1)
    fb = synthetic_broker(days)
    first = fb.news_items[0]
    first["updated_at"] = first["created_at"] + timedelta(minutes=40)
    build_dataset(ctx.db, fb, table, days[0], days[0], 60, 1000, lambda f, m: None, lambda: False, Throttle(0))
    row = ctx.db.query_one("SELECT published_at FROM ml_samples WHERE news_id = ?", (str(first["id"]),))
    assert row["published_at"].startswith(iso_min(first["updated_at"]))


def iso_min(t):
    return t.strftime("%Y-%m-%dT%H:%M")


def test_test_set_is_not_filtered_on_the_outcome(ctx, tmp_path):
    days = weekdays(date(2026, 1, 5), 60)
    samples, _ = _samples_from(ctx, days, tmp_path)
    # make every 3rd headline a "small move": excluded from fitting, but the test must still include them
    for i, smp in enumerate(samples):
        if i % 3 == 0:
            smp.small = True
    _, report = train_price_model(samples, 60, 0.3, "fake-finbert", ("a", "b"))
    _, _, test = time_split(samples, 62)
    assert report["n_test"] == len(test) and any(x.small for x in test)
    assert "accuracy_moved_only" in report


def test_pure_noise_rarely_passes():
    import random

    rng = random.Random(3)
    t0 = datetime(2026, 1, 5, 15, tzinfo=UTC)
    words = ["alpha", "beta", "gamma", "delta", "omega", "sigma"]
    passed = 0
    for trial in range(5):
        samples = []
        for i in range(1200):
            s_ = SentimentScores(rng.random(), rng.random(), rng.random())
            f = SampleFeatures(" ".join(rng.choice(words) for _ in range(6)), s_, 1.0, True, True, 1, rng.random())
            day = t0 + timedelta(days=i // 25, minutes=i % 25 * 10)
            samples.append(TrainingSample(f, rng.randint(0, 1), day, rng.uniform(-1, 1), f"n{trial}-{i}"))
        _, report = train_price_model(samples, 60, 0.3, "x", ("a", "b"))
        passed += report["passed"]
    assert passed <= 1


async def test_failed_price_model_sends_sentiment_signals_to_review(local_env, tmp_path):
    ctx, pipeline, fake, broker = local_env
    await pipeline.local.ensure_ready()
    days = weekdays(date(2026, 1, 5), 45)
    samples, _ = _samples_from(ctx, days, tmp_path)
    model, _ = train_price_model(samples, 60, 0.3, "fake-finbert", ("2026-01-05", "2026-03-06"))
    model.meta["passed"] = False
    pipeline.local.set_price_model(model)
    out = await run(pipeline, news("Nvidia wins record contract"))
    s = out["signals"][0]
    assert s["confidence"] == 79 and s["action"] == "review" and "Sent for review" in s["reasoning"]
    ctx.config.update({"ml": {"sentiment_only_trading": "always"}})
    out = await run(pipeline, news("Tesla beats delivery estimates as demand surges in China"))
    assert out["signals"][0]["confidence"] == 87  # earnings beat 0.78 + FinBERT agrees 0.09


async def test_transcripts_use_sentiment_only(local_env, tmp_path):
    ctx, pipeline, fake, broker = local_env
    days = weekdays(date(2026, 1, 5), 60)
    samples, _ = _samples_from(ctx, days, tmp_path)
    model, _ = train_price_model(samples, 60, 0.3, "fake-finbert", ("2026-01-05", "2026-03-27"))
    pipeline.local.set_price_model(model)
    await pipeline.local.ensure_ready()
    item = NewsItem(source_id="tv", source_type="stream", source_name="TV", external_id="x", title="",
                    body="New:\n[10:00:00] Nvidia just won a record contract.", kind="transcript",
                    published_at=datetime.now(UTC))
    pre = prefilter(item.text, pipeline.tickers, [], False)
    res = await pipeline.local.analyze(item, pre, True, datetime.now(UTC))
    import json as _json

    assert all(d["p_up"] is None for d in _json.loads(res.text)["details"])


async def test_training_keeps_its_sentiment_model_if_settings_change(local_env, tmp_path):
    ctx, pipeline, fake, broker = local_env
    days = weekdays(date(2026, 1, 5), 60)
    ctx.services["trader"].broker = synthetic_broker(days)
    trainer = ModelTrainer(ctx, throttle_factory=lambda: Throttle(0))
    ctx.services["ml_trainer"] = trainer
    real_predict = fake.predict
    calls = {"n": 0}

    def predict_and_switch(texts):
        calls["n"] += 1
        if calls["n"] == 2:  # the user switches to the word list in the middle of training
            pipeline.local.sentiment.load("lexicon")
        return real_predict(texts)

    fake.predict = predict_and_switch
    run_id = ctx.db.insert("ml_runs", {"created_at": "x", "params": "{}", "status": "running"})
    await trainer.run(run_id, TrainParams(start=days[0], end=days[-1], max_articles=5000))
    assert pipeline.local.price_model.meta["sentiment_model"] == "fake-finbert"
    cached = {r["model_id"] for r in ctx.db.query("SELECT DISTINCT model_id FROM ml_sentiment_cache")}
    assert cached == {"fake-finbert"}


def test_lookahead_note_covers_dates_before_training():
    from newstrader.backtest.runner import lookahead_note

    pm = SimpleNamespace(meta={"trained_from": "2026-04-10", "trained_to": "2026-10-06"})
    assert "already knows" in lookahead_note(pm, date(2025, 10, 1))  # before the range: still look-ahead
    assert "already knows" in lookahead_note(pm, date(2026, 6, 1))  # overlap
    assert lookahead_note(pm, date(2026, 10, 7)) == ""  # strictly after: honest
    assert lookahead_note(None, date(2025, 1, 1)) == ""


def test_claude_call_count_ignores_local_engine(ctx):
    from newstrader.ai.costs import calls_today
    from newstrader.db import iso
    from newstrader.state import trading_day

    for engine in ("local", "local", "claude"):
        ctx.db.insert("analyses", {"created_at": iso(), "trading_day": trading_day(), "item_kind": "news",
                                   "status": "ok", "is_backtest": 0, "engine": engine, "cost_usd": 0})
    assert calls_today(ctx.db) == 1


def test_old_config_is_switched_to_local_with_a_notice(tmp_path):
    import json as _json

    from newstrader.config import ConfigStore

    path = tmp_path / "config.json"
    path.write_text(_json.dumps({"ai": {"model": "claude-opus-5-5"}}))
    store = ConfigStore(path)
    assert store.settings.ai.engine == "local" and store.engine_was_defaulted
    again = ConfigStore(path)  # saved with the engine now, so the notice is shown once
    assert not again.engine_was_defaulted


async def test_live_mode_sends_sentiment_only_signals_to_review(local_env):
    ctx, pipeline, fake, broker = local_env
    ctx.config.update({"ml": {"sentiment_only_trading": "always"}})
    ctx.state._set_live_armed(True)
    try:
        out = await run(pipeline, news("Nvidia wins record contract"))
    finally:
        ctx.state._set_live_armed(False)
    s = out["signals"][0]
    assert s["confidence"] == 79 and "LIVE" in s["reasoning"]
