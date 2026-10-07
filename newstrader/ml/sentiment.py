"""Sentiment layer: is the wording about a company good, bad or neutral?

Default model: FinBERT (ProsusAI/finbert, a BERT model fine-tuned on financial news), in the ONNX format
published as Xenova/finbert on Hugging Face. It runs on the CPU with onnxruntime - no PyTorch, no GPU, no
API key. The ~110 MB file is downloaded once into the data folder's models/finbert.

Fallback: a small built-in word list (used if the download fails or you pick it in Settings). It is crude,
so its confidence is capped below the default buy threshold - it can flag news for review but never
auto-trades on default settings.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

FINBERT_REPO = "Xenova/finbert"
FINBERT_REVISION = "main"
FINBERT_ONNX = "onnx/model_quantized.onnx"
FINBERT_FILES = ("config.json", "tokenizer.json", FINBERT_ONNX)
DEFAULT_ID2LABEL = {0: "positive", 1: "negative", 2: "neutral"}
LEXICON_MAX_PROB = 0.79


@dataclass
class SentimentScores:
    positive: float
    negative: float
    neutral: float

    @property
    def label(self) -> str:
        return max(("positive", "negative", "neutral"), key=lambda k: getattr(self, k))

    @property
    def net(self) -> float:
        """One number from -1 (very negative) to +1 (very positive)."""
        return self.positive - self.negative

    def as_dict(self) -> dict:
        return {"positive": round(self.positive, 4), "negative": round(self.negative, 4),
                "neutral": round(self.neutral, 4)}


# ---------------------------------------------------------------------------------------------- word list
POSITIVE_WORDS = {
    "beat", "beats", "beating", "tops", "topped", "surpass", "surpasses", "surpassed", "exceed", "exceeds",
    "exceeded", "record", "records", "surge", "surges", "surged", "soar", "soars", "soared", "jump", "jumps",
    "jumped", "rally", "rallies", "rallied", "gain", "gains", "gained", "rise", "rises", "rose", "climb", "climbs",
    "climbed", "upgrade", "upgrades", "upgraded", "outperform", "overweight", "raise", "raises", "raised",
    "boost", "boosts", "boosted", "approve", "approves", "approved", "approval", "clearance", "cleared", "win",
    "wins", "won", "award", "awarded", "contract", "partnership", "partners", "expand", "expands", "expansion",
    "buyback", "repurchase", "dividend", "profit", "profitable", "profits", "growth", "grows", "grew", "strong",
    "stronger", "robust", "breakthrough", "launch", "launches", "milestone", "upbeat", "optimistic", "bullish",
    "higher", "accelerate", "accelerates", "momentum", "demand", "outpace", "outpaces", "acquire", "acquires",
    "takeover", "premium", "settle", "settles", "settled", "resolve", "resolved", "lifted", "lifts", "reinstated",
    "positive", "success", "successful", "successfully", "favorable", "best", "boom", "booming", "recovery",
}
NEGATIVE_WORDS = {
    "miss", "misses", "missed", "fall", "falls", "fell", "drop", "drops", "dropped", "plunge", "plunges",
    "plunged", "sink", "sinks", "sank", "tumble", "tumbles", "tumbled", "slump", "slumps", "slumped", "crash",
    "crashes", "crashed", "downgrade", "downgrades", "downgraded", "underperform", "underweight", "cut", "cuts",
    "lower", "lowers", "lowered", "reduce", "reduces", "reduced", "weak", "weaker", "weakness", "loss", "losses",
    "lose", "loses", "lost", "decline", "declines", "declined", "lawsuit", "sue", "sues", "sued", "probe",
    "investigation", "investigate", "subpoena", "fraud", "recall", "recalls", "recalled", "bankruptcy",
    "bankrupt", "default", "defaults", "layoff", "layoffs", "delay", "delays", "delayed", "halt", "halts",
    "halted", "suspend", "suspends", "suspended", "warning", "warns", "warned", "fine", "fined", "penalty",
    "ban", "bans", "banned", "reject", "rejects", "rejected", "rejection", "fail", "fails", "failed", "failure",
    "bearish", "concern", "concerns", "risk", "risks", "slowdown", "slows", "slowing", "shortfall", "dilution",
    "dilutive", "offering", "resign", "resigns", "resigned", "breach", "hack", "hacked", "outage", "tariff",
    "tariffs", "sanction", "sanctions", "negative", "worst", "disappointing", "disappoints", "plummet",
    "plummets", "plummeted", "shutdown", "closure", "closes", "withdraw", "withdraws", "terminated", "terminate",
}
NEGATORS = {"not", "no", "never", "without", "fails", "failed", "isn't", "wasn't", "won't", "didn't", "doesn't"}
_WORD = re.compile(r"[a-z']+")


class LexiconSentiment:
    name = "lexicon"
    model_id = "lexicon-v1"

    def predict(self, texts: list[str]) -> list[SentimentScores]:
        return [self._one(t) for t in texts]

    @staticmethod
    def _one(text: str) -> SentimentScores:
        words = _WORD.findall((text or "").lower())
        pos = neg = 0
        flip_until = -1
        for i, w in enumerate(words):
            if w in NEGATORS:
                flip_until = i + 3
                continue
            hit = 1 if w in POSITIVE_WORDS else -1 if w in NEGATIVE_WORDS else 0
            if hit and i <= flip_until:
                hit = -hit
            if hit > 0:
                pos += 1
            elif hit < 0:
                neg += 1
        total = pos + neg
        if total == 0 or pos == neg:
            return SentimentScores(0.15, 0.15, 0.70)
        net = abs(pos - neg) / total
        strength = min(1.0, total / 3)
        p_win = 0.45 + (LEXICON_MAX_PROB - 0.45) * net * strength
        p_lose = (1 - p_win) * 0.3
        p_neu = 1 - p_win - p_lose
        return SentimentScores(p_win, p_lose, p_neu) if pos > neg else SentimentScores(p_lose, p_win, p_neu)


# ---------------------------------------------------------------------------------------------- FinBERT
class FinbertOnnx:
    """FinBERT via onnxruntime + tokenizers. Thread-safe to call predict() from worker threads."""

    name = "finbert"

    def __init__(self, model_dir: Path, onnx_file: str = FINBERT_ONNX, max_length: int = 160, threads: int = 2):
        import numpy as np  # noqa: F401 - fail early if missing
        import onnxruntime as ort
        from tokenizers import Tokenizer

        d = Path(model_dir)
        self.model_dir = d
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.tok.enable_truncation(max_length=max_length)
        pad_token = next((t for t in ("[PAD]", "<pad>") if self.tok.token_to_id(t) is not None), "[PAD]")
        self.tok.enable_padding(pad_id=self.tok.token_to_id(pad_token) or 0, pad_token=pad_token)
        self.id2label = dict(DEFAULT_ID2LABEL)
        cfg = d / "config.json"
        if cfg.exists():
            raw = json.loads(cfg.read_text(encoding="utf-8")).get("id2label") or {}
            if raw:
                self.id2label = {int(k): str(v).lower() for k, v in raw.items()}
        if set(self.id2label.values()) != {"positive", "negative", "neutral"}:
            raise ValueError(f"unexpected FinBERT labels: {self.id2label}")
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = max(1, threads)
        self.sess = ort.InferenceSession(str(d / onnx_file), sess_options=so, providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.sess.get_inputs()}
        self.output_name = self.sess.get_outputs()[0].name
        self.model_id = f"finbert-{FINBERT_REPO.split('/')[0].lower()}-{Path(onnx_file).stem}"

    def predict(self, texts: list[str], batch_size: int = 16) -> list[SentimentScores]:
        import numpy as np

        out: list[SentimentScores] = []
        for i in range(0, len(texts), batch_size):
            batch = [t if t and t.strip() else "." for t in texts[i:i + batch_size]]
            enc = self.tok.encode_batch(batch)
            feeds = {
                "input_ids": np.array([e.ids for e in enc], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64),
                "token_type_ids": np.array([e.type_ids for e in enc], dtype=np.int64),
            }
            feeds = {k: v for k, v in feeds.items() if k in self.input_names}
            logits = self.sess.run([self.output_name], feeds)[0].astype(np.float64)
            logits -= logits.max(axis=-1, keepdims=True)
            probs = np.exp(logits)
            probs /= probs.sum(axis=-1, keepdims=True)
            for p in probs:
                scores = {self.id2label[j]: float(p[j]) for j in range(p.shape[0])}
                out.append(SentimentScores(scores["positive"], scores["negative"], scores["neutral"]))
        return out


def finbert_dir(models_dir: Path) -> Path:
    return models_dir / "finbert"


MIN_ONNX_BYTES = 50_000_000  # the int8 FinBERT file is ~110 MB; anything much smaller is a broken download


def finbert_present(models_dir: Path) -> bool:
    d = finbert_dir(models_dir)
    if not all((d / f).exists() and (d / f).stat().st_size > 0 for f in FINBERT_FILES):
        return False
    return (d / FINBERT_ONNX).stat().st_size >= MIN_ONNX_BYTES


def download_finbert(models_dir: Path) -> Path:
    """Download the three FinBERT files (~110 MB) from Hugging Face. Blocking."""
    from huggingface_hub import hf_hub_download

    dest = finbert_dir(models_dir)
    dest.mkdir(parents=True, exist_ok=True)
    for name in FINBERT_FILES:
        f = dest / name
        broken = name == FINBERT_ONNX and f.exists() and f.stat().st_size < MIN_ONNX_BYTES
        hf_hub_download(FINBERT_REPO, name, revision=FINBERT_REVISION, local_dir=str(dest), force_download=broken)
    if not finbert_present(models_dir):
        raise RuntimeError("FinBERT download looks incomplete - try again")
    return dest


class SentimentService:
    """Holds whichever sentiment model is in use and loads FinBERT on first need (download if missing)."""

    def __init__(self, models_dir: Path, loader=None):
        self.models_dir = models_dir
        self._loader = loader  # tests inject a fake
        self._lock = threading.Lock()
        self._predict_lock = threading.Lock()
        self.model = None
        self.wanted: str | None = None
        self.error: str | None = None
        self.loading = False

    @property
    def ready(self) -> bool:
        return self.model is not None

    @property
    def model_id(self) -> str:
        return getattr(self.model, "model_id", "none")

    def describe(self) -> str:
        if self.loading:
            return "downloading/loading FinBERT..."
        if self.model is None:
            return "not loaded"
        if getattr(self.model, "name", "") == "finbert":
            return "FinBERT"
        if self.wanted == "finbert":
            return "built-in word list (FinBERT unavailable)"
        return "built-in word list"

    def load(self, which: str) -> None:
        """Blocking. Loads `which` ("finbert" or "lexicon"); falls back to the word list if FinBERT fails."""
        with self._lock:
            if self.model is not None and self.wanted == which and (which == "lexicon" or self.error is None):
                return
            self.wanted = which
            self.loading = True
            try:
                if which == "lexicon":
                    self.model, self.error = LexiconSentiment(), None
                    return
                try:
                    if self._loader is not None:
                        model = self._loader(self.models_dir)
                    else:
                        if not finbert_present(self.models_dir):
                            log.info("Downloading FinBERT sentiment model (~110 MB, one time)...")
                            download_finbert(self.models_dir)
                        model = FinbertOnnx(finbert_dir(self.models_dir))
                    self.model, self.error = model, None
                    log.info("FinBERT sentiment model loaded")
                except Exception as exc:
                    self.error = _short(exc)
                    log.warning("FinBERT unavailable (%s) - using the built-in word list instead", self.error)
                    self.model = LexiconSentiment()
            finally:
                self.loading = False

    def predict(self, texts: list[str]) -> list[SentimentScores]:
        model = self.model
        if model is None:
            raise RuntimeError("sentiment model not loaded")
        with self._predict_lock:
            return model.predict(texts)


def _short(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text if len(text) < 200 else text[:197] + "..."
