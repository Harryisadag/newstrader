"""The local machine-learning engine: turns one news item into signals without any paid API.

For each company the pre-filter found:
  1. pick the headline + sentences about that company (text.py)
  2. FinBERT scores that wording positive / negative / neutral (sentiment.py)
  3. if a price model has been trained and passed its test, it turns those scores + the words into the
     probability the stock beats the S&P 500 over the next hour (model.py)
The answer is written in the same JSON shape Claude uses, so the same validator and trade rules apply.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .. import paths
from ..ai.engine import AnalysisResult
from ..ai.prefilter import Candidate, PrefilterResult
from ..ai.tickers import TickerTable
from ..sources.base import NewsItem
from .model import PriceModel, SampleFeatures
from .sentiment import SentimentScores, SentimentService
from .text import Target, mask_company, relevance, target_text

log = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
MIN_SENTIMENT_CONFIDENCE = 50  # weaker sentiment calls are treated as neutral (not emitted)
MODEL_NEUTRAL_BELOW = 55  # price-model calls this close to a coin flip are treated as neutral
MAX_CONFIDENCE = 99


def time_of_day(t: datetime | None) -> float | None:
    """0.0 at the 9:30 ET open, 1.0 at the 4:00 ET close (clamped)."""
    if t is None:
        return None
    local = t.astimezone(ET)
    mins = (local.hour * 60 + local.minute) - (9 * 60 + 30)
    return min(1.0, max(0.0, mins / 390))


def ordered_candidates(pre: PrefilterResult, symbols: list[str]) -> list[Candidate]:
    """Companies the source tagged come first, then the order they were found in the text."""
    tagged = {s.upper() for s in symbols or []}
    return sorted(pre.candidates, key=lambda c: (c.symbol not in tagged, pre.candidates.index(c)))


def build_features(item: NewsItem, cand: Candidate, target: Target, scores: SentimentScores, table: TickerTable,
                   at: datetime | None) -> SampleFeatures:
    return SampleFeatures(text=mask_company(target.snippet, cand, table), sentiment=scores,
                          relevance=relevance(target, len(item.symbols)), in_headline=target.in_headline,
                          tagged=target.tagged, n_symbols=len(item.symbols), time_of_day=time_of_day(at))


class LocalMLEngine:
    engine = "local"

    def __init__(self, ctx, sentiment: SentimentService | None = None, model_dir: Path | None = None):
        self.ctx = ctx
        self.sentiment = sentiment or SentimentService(paths.models_dir())
        self.model_dir = model_dir or (paths.models_dir() / "price_model")
        self.tickers: TickerTable | None = None
        self.price_model: PriceModel | None = None
        self._model_lock = threading.Lock()
        self._loaded_model_file = False

    # ---------------------------------------------------------------- models
    def load_price_model(self) -> PriceModel | None:
        with self._model_lock:
            try:
                self.price_model = PriceModel.load(self.model_dir)
            except Exception as exc:
                log.warning("Couldn't load the trained price model: %s", exc)
                self.price_model = None
            self._loaded_model_file = True
            return self.price_model

    def set_price_model(self, model: PriceModel | None) -> None:
        with self._model_lock:
            self.price_model = model
            self._loaded_model_file = True

    def active_price_model(self) -> PriceModel | None:
        """The trained model if settings allow it and it matches the sentiment model in use."""
        if not self._loaded_model_file:
            self.load_price_model()
        m = self.price_model
        mode = self.ctx.config.settings.ml.use_trained_model
        if m is None or mode == "never":
            return None
        if m.meta.get("sentiment_model") != self.sentiment.model_id:
            return None  # trained on different sentiment scores - its weights wouldn't mean the same thing
        if mode == "auto" and not m.meta.get("passed"):
            return None
        return m

    def model_status(self) -> dict:
        m = self.price_model if self._loaded_model_file else self.load_price_model()
        active = self.active_price_model() is not None
        info = {"sentiment": self.sentiment.describe(), "sentiment_model_id": self.sentiment.model_id,
                "sentiment_error": self.sentiment.error, "price_model": None, "price_model_active": active,
                "sentiment_cap": self.sentiment_cap_reason() if not active else ""}
        if m is not None:
            meta = m.meta
            why_inactive = ""
            if not active:
                mode = self.ctx.config.settings.ml.use_trained_model
                if mode == "never":
                    why_inactive = "turned off in Settings"
                elif meta.get("sentiment_model") != self.sentiment.model_id:
                    why_inactive = "trained with a different sentiment model - retrain"
                elif not meta.get("passed"):
                    why_inactive = "didn't pass its test on unseen news"
            info["price_model"] = {k: meta.get(k) for k in (
                "trained_at", "n_samples", "trained_from", "trained_to", "horizon_minutes", "min_move_pct",
                "sentiment_model", "passed", "report")}
            info["price_model"]["why_inactive"] = why_inactive
        return info

    def sentiment_cap_reason(self) -> str:
        """Why sentiment-only signals may not auto-buy right now ("" = they may)."""
        mode = self.ctx.config.settings.ml.sentiment_only_trading
        if mode == "always":
            return ""
        if mode == "review":
            return "sentiment-only signals are set to manual review"
        m = self.price_model if self._loaded_model_file else self.load_price_model()
        if (m is not None and not m.meta.get("passed")
                and m.meta.get("sentiment_model") == self.sentiment.model_id):
            return "the trained price model found headline wording didn't reliably predict moves"
        return ""

    def label(self) -> str:
        s = "FinBERT" if getattr(self.sentiment.model, "name", "") == "finbert" else "word list"
        m = self.active_price_model()
        if m is not None:
            return f"local: {s} + price model ({str(m.meta.get('trained_at', ''))[:10]})"
        return f"local: {s}"

    async def ensure_ready(self) -> None:
        which = self.ctx.config.settings.ml.sentiment_model
        if not self.sentiment.ready or self.sentiment.wanted != which:
            await asyncio.to_thread(self.sentiment.load, which)

    # ---------------------------------------------------------------- analysis
    async def analyze(self, item: NewsItem, pre: PrefilterResult, market_open: bool | None, now: datetime,
                      model: str | None = None, skip_rate_limit: bool = False) -> AnalysisResult:
        started = time.monotonic()
        try:
            await self.ensure_ready()
            payload = await asyncio.to_thread(self._analyze_sync, item, pre, now)
        except Exception as exc:
            log.exception("Local ML analysis failed")
            return AnalysisResult(ok=False, status="error", engine="local", model="local",
                                  error=f"Local ML error: {type(exc).__name__}: {exc}",
                                  latency_ms=int((time.monotonic() - started) * 1000))
        return AnalysisResult(ok=True, text=json.dumps(payload), model=self.label(), engine="local",
                              latency_ms=int((time.monotonic() - started) * 1000), stop_reason="end_turn")

    def _analyze_sync(self, item: NewsItem, pre: PrefilterResult, now: datetime) -> dict:
        table = self.tickers
        if table is None:
            raise RuntimeError("ticker table not attached")
        max_signals = self.ctx.config.settings.ai.max_signals_per_item
        targets: list[tuple[Candidate, Target]] = []
        for cand in ordered_candidates(pre, item.symbols):
            t = target_text(item.title, item.body, item.kind, cand, table)
            if t is not None:
                targets.append((cand, t))
        if not targets:
            return {"signals": [], "details": [], "note": "no company to score"}
        scores = self.sentiment.predict([t.snippet for _, t in targets])
        at = item.published_at if item.kind != "transcript" else now
        feats = [build_features(item, c, t, s, table, at or now) for (c, t), s in zip(targets, scores, strict=True)]
        # The price model learned from written news articles; speech from live TV is scored on sentiment only.
        pm = self.active_price_model() if item.kind != "transcript" else None
        p_up = pm.predict_up(feats) if pm is not None else None
        horizon = int(pm.meta.get("horizon_minutes", 60)) if pm is not None else 60
        cap_reason = self.sentiment_cap_reason() if pm is None else ""
        buy = self.ctx.config.settings.trading.buy_threshold

        signals, details = [], []
        for i, ((cand, target), s, f) in enumerate(zip(targets, scores, feats, strict=True)):
            info = table.get(cand.symbol)
            name = info.name if info else cand.symbol
            short = _short_name(name, cand.symbol)
            rel = f.relevance
            if p_up is not None:
                p = float(p_up[i])
                conf = round(100 * max(p, 1 - p))
                direction = "neutral" if conf < MODEL_NEUTRAL_BELOW else ("bullish" if p >= 0.5 else "bearish")
            else:
                p = None
                label = s.label
                p_dir = s.positive if label == "positive" else s.negative if label == "negative" else s.neutral
                conf = round(100 * p_dir * rel)
                direction = {"positive": "bullish", "negative": "bearish"}.get(label, "neutral")
                if conf < MIN_SENTIMENT_CONFIDENCE:
                    direction = "neutral"
            conf = int(min(MAX_CONFIDENCE, max(0, conf)))
            capped = bool(cap_reason) and direction != "neutral" and conf >= buy
            if capped:
                conf = max(0, buy - 1)  # stays a manual-review signal
            details.append({"ticker": cand.symbol, "matched": cand.why, "snippet": target.snippet[:300],
                            "sentiment": s.as_dict(), "relevance": rel, "p_up": None if p is None else round(p, 4),
                            "direction": direction, "confidence": conf})
            if direction == "neutral":
                continue
            signals.append({
                "ticker": cand.symbol, "company": name, "speaker": item.speaker or "",
                "bull_case": _bull(s, p, horizon), "bear_case": _bear(s, p, horizon),
                "direction": direction, "confidence": conf, "time_sensitivity": "hours",
                "reasoning": (_reasoning(short, s, p, horizon, target)
                              + (f" Sent for review: {cap_reason}." if capped else ""))[:400],
            })
        signals.sort(key=lambda x: x["confidence"], reverse=True)
        return {"signals": signals[:max_signals], "details": details}


# -------------------------------------------------------------------------------------------- wording
def _pct(x: float) -> int:
    return round(100 * x)


def _short_name(name: str, symbol: str) -> str:
    from ..ai.tickers import clean_company_name

    clean = clean_company_name(name)
    if not clean or clean.upper() == symbol:
        return symbol
    return clean.title()[:40]


def _reasoning(short: str, s: SentimentScores, p: float | None, horizon: int, t: Target) -> str:
    where = "" if t.in_headline else " (named in the article, not the headline)"
    text = (f"{short}{where}: wording reads {_pct(s.positive)}% positive / {_pct(s.negative)}% negative / "
            f"{_pct(s.neutral)}% neutral.")
    if p is not None:
        text += f" Price model: {_pct(p)}% chance it beats the S&P 500 over the next {horizon} min."
    else:
        text += " Sentiment only (no price model in use)."
    return text[:400]


def _bull(s: SentimentScores, p: float | None, horizon: int) -> str:
    text = f"Positive wording score {_pct(s.positive)}%."
    if p is not None:
        text += f" Headlines like this beat the market over {horizon} min {_pct(p)}% of the time in training."
    return text


def _bear(s: SentimentScores, p: float | None, horizon: int) -> str:
    text = f"Negative wording score {_pct(s.negative)}%."
    if p is not None:
        text += f" They lagged the market {_pct(1 - p)}% of the time."
    return text + " A word-based model can't tell whether the news was already expected or priced in."

