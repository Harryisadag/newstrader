"""The price-trained layer: a logistic-regression model that learns, from past headlines, how often a stock
beat (or lagged) the S&P 500 in the hour after news like this.

Inputs per (headline, stock): the FinBERT scores, how clearly the item is about the stock, time of day, and
the words of the headline/sentences (company name masked, hashed into a fixed-size vector).
Output: probability that the stock out-performs the market over the horizon, given that it moves.

Saved as plain numbers (.npz + .json), never as a pickle, so a model file can't run code when loaded.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .sentiment import SentimentScores

log = logging.getLogger(__name__)

N_HASH = 2 ** 18
NUMERIC = ["pos", "neg", "neu", "net", "relevance", "in_headline", "tagged", "log_symbols", "log_words",
           "time_of_day"]
C_GRID = (0.03, 0.1, 0.3, 1.0, 3.0)
THRESHOLDS = (55, 60, 65, 70, 75, 80, 85, 90)
MODEL_VERSION = 1


@dataclass
class SampleFeatures:
    text: str  # headline + sentences about the company, company name masked
    sentiment: SentimentScores
    relevance: float
    in_headline: bool
    tagged: bool
    n_symbols: int
    time_of_day: float | None  # 0 = market open, 1 = close

    def numeric(self) -> list[float]:
        s = self.sentiment
        words = len(self.text.split())
        tod = 0.5 if self.time_of_day is None else min(1.0, max(0.0, self.time_of_day))
        return [s.positive, s.negative, s.neutral, s.net, self.relevance, float(self.in_headline),
                float(self.tagged), math.log1p(max(0, self.n_symbols)), math.log1p(words), tod]


def _vectorizer():
    from sklearn.feature_extraction.text import HashingVectorizer

    return HashingVectorizer(n_features=N_HASH, ngram_range=(1, 2), alternate_sign=False, norm="l2",
                             lowercase=True, token_pattern=r"(?u)\b\w+\b")


def _matrix(feats: list[SampleFeatures], mean: np.ndarray, std: np.ndarray):
    from scipy import sparse

    text = _vectorizer().transform([f.text for f in feats])
    num = (np.array([f.numeric() for f in feats], dtype=np.float64) - mean) / std
    return sparse.hstack([text, sparse.csr_matrix(num)], format="csr")


class PriceModel:
    def __init__(self, coef: np.ndarray, intercept: float, mean: np.ndarray, std: np.ndarray, meta: dict):
        self.coef = coef.astype(np.float64)
        self.intercept = float(intercept)
        self.mean = mean
        self.std = std
        self.meta = meta

    def predict_up(self, feats: list[SampleFeatures]) -> np.ndarray:
        if not feats:
            return np.zeros(0)
        x = _matrix(feats, self.mean, self.std)
        z = x @ self.coef + self.intercept
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    # ---------------------------------------------------------------- files
    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        tmp = folder / "price_model.tmp.npz"
        np.savez_compressed(tmp, coef=self.coef.astype(np.float32), intercept=np.array([self.intercept]),
                            mean=self.mean, std=self.std)
        os.replace(tmp, folder / "price_model.npz")
        meta_tmp = folder / "price_model.json.tmp"
        meta_tmp.write_text(json.dumps(self.meta, indent=2, default=str), encoding="utf-8")
        os.replace(meta_tmp, folder / "price_model.json")

    @classmethod
    def load(cls, folder: Path) -> PriceModel | None:
        npz, meta_file = folder / "price_model.npz", folder / "price_model.json"
        if not npz.exists() or not meta_file.exists():
            return None
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        if meta.get("version") != MODEL_VERSION or meta.get("n_hash") != N_HASH or meta.get("numeric") != NUMERIC:
            log.warning("Saved price model is from an older version of NewsTrader - retrain it in the Backtest tab")
            return None
        with np.load(npz, allow_pickle=False) as data:
            coef, intercept = data["coef"], float(data["intercept"][0])
            mean, std = data["mean"], data["std"]
        if coef.shape != (N_HASH + len(NUMERIC),) or mean.shape != (len(NUMERIC),):
            log.warning("Saved price model file has the wrong shape - retrain it")
            return None
        return cls(coef, intercept, mean, std, meta)


def delete_model(folder: Path) -> None:
    for name in ("price_model.npz", "price_model.json"):
        p = folder / name
        if p.exists():
            p.unlink()


# -------------------------------------------------------------------------------------------- training
@dataclass
class TrainingSample:
    feats: SampleFeatures
    up: int  # 1 = beat the market, 0 = lagged it
    at: datetime
    adj_ret: float  # stock return minus S&P 500 return over the horizon, in %
    group: str  # news id - samples from one article stay on the same side of a split


def _fit(x, y, c: float):
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(C=c, solver="liblinear", max_iter=2000)
    clf.fit(x, y)
    return clf


def _logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _auc(y: np.ndarray, p: np.ndarray) -> float | None:
    if len(set(y.tolist())) < 2:
        return None
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, p))


def time_split(samples: list[TrainingSample], embargo_minutes: int, test_frac: float = 0.2,
               valid_frac: float = 0.15) -> tuple[list[TrainingSample], list[TrainingSample], list[TrainingSample]]:
    """Oldest -> train, middle -> validation, newest -> test. Never shuffled (that would leak the future).

    Samples whose price window overlaps the next part's start are dropped (the "embargo"), and all samples
    from one article stay together.
    """
    ordered = sorted(samples, key=lambda s: (s.at, s.group))
    n = len(ordered)

    def cut_index(frac: float) -> int:
        i = int(n * (1 - frac))
        while 0 < i < n and ordered[i].group == ordered[i - 1].group:
            i += 1
        return i

    i_test = cut_index(test_frac)
    i_valid = cut_index(test_frac + valid_frac)
    train, valid, test = ordered[:i_valid], ordered[i_valid:i_test], ordered[i_test:]
    from datetime import timedelta

    gap = timedelta(minutes=embargo_minutes)
    if valid:
        train = [s for s in train if s.at < valid[0].at - gap]
    if test:
        valid = [s for s in valid if s.at < test[0].at - gap]
    return train, valid, test


def train_price_model(samples: list[TrainingSample], horizon_minutes: int, min_move_pct: float,
                      sentiment_model: str, trained_range: tuple[str, str], progress=None) -> tuple[PriceModel, dict]:
    """Fit, evaluate on the newest 20% (never seen while fitting), then refit on everything for live use."""
    if len(samples) < 200:
        raise ValueError(f"Only {len(samples)} usable headlines - need at least 200. Use a longer date range.")
    train, valid, test = time_split(samples, embargo_minutes=horizon_minutes)
    if len(train) < 100 or len(valid) < 30 or len(test) < 50:
        raise ValueError("Not enough headlines to split into train/validation/test - use a longer date range.")

    def xy(rows: list[TrainingSample], mean, std):
        return _matrix([r.feats for r in rows], mean, std), np.array([r.up for r in rows])

    def norm_stats(rows: list[TrainingSample]):
        num = np.array([r.feats.numeric() for r in rows], dtype=np.float64)
        std = num.std(axis=0)
        return num.mean(axis=0), np.where(std < 1e-9, 1.0, std)

    # 1) pick the regularisation strength on the validation part
    mean, std = norm_stats(train)
    x_tr, y_tr = xy(train, mean, std)
    x_va, y_va = xy(valid, mean, std)
    if len(set(y_tr.tolist())) < 2:
        raise ValueError("All training headlines had the same outcome - use a longer date range.")
    scores = {}
    for i, c in enumerate(C_GRID):
        clf = _fit(x_tr, y_tr, c)
        scores[c] = _logloss(y_va, clf.predict_proba(x_va)[:, 1])
        if progress:
            progress(0.2 + 0.4 * (i + 1) / len(C_GRID), f"tuning model ({i + 1}/{len(C_GRID)})")
    best_c = min(scores, key=scores.get)

    # 2) honest test: fit on train+validation, score the newest part
    fit_rows = train + valid
    mean, std = norm_stats(fit_rows)
    x_fit, y_fit = xy(fit_rows, mean, std)
    clf = _fit(x_fit, y_fit, best_c)
    x_te, y_te = xy(test, mean, std)
    p_te = clf.predict_proba(x_te)[:, 1]
    base_rate = float(y_fit.mean())
    report = evaluate(y_te, p_te, np.array([r.adj_ret for r in test]), base_rate)
    report.update({"best_c": best_c, "validation_logloss": {str(k): round(v, 5) for k, v in scores.items()},
                   "n_train": len(train), "n_valid": len(valid), "n_test": len(test),
                   "test_from": test[0].at.isoformat(), "test_to": test[-1].at.isoformat(),
                   "base_rate_up": round(base_rate, 4)})
    if progress:
        progress(0.8, "fitting final model on all headlines")

    # 3) final model on everything (most recent news included)
    mean, std = norm_stats(samples)
    x_all, y_all = xy(samples, mean, std)
    final = _fit(x_all, y_all, best_c)
    coef = np.asarray(final.coef_).ravel()
    meta = {
        "version": MODEL_VERSION, "n_hash": N_HASH, "numeric": NUMERIC, "trained_at": datetime.now().astimezone().isoformat(),
        "horizon_minutes": horizon_minutes, "min_move_pct": min_move_pct, "sentiment_model": sentiment_model,
        "n_samples": len(samples), "trained_from": trained_range[0], "trained_to": trained_range[1],
        "report": report, "passed": report["passed"],
    }
    return PriceModel(coef, float(np.asarray(final.intercept_).ravel()[0]), mean, std, meta), report


def evaluate(y: np.ndarray, p: np.ndarray, adj_ret: np.ndarray, base_rate: float) -> dict:
    """Test-set scores + what each confidence threshold would have done."""
    pred_up = p >= 0.5
    acc = float(np.mean(pred_up == (y == 1)))
    majority = max(float(np.mean(y)), 1 - float(np.mean(y)))
    ll = _logloss(y, p)
    base_ll = _logloss(y, np.full_like(p, min(max(base_rate, 1e-3), 1 - 1e-3)))
    auc = _auc(y, p)
    conf = np.round(100 * np.maximum(p, 1 - p))
    dir_ret = np.where(pred_up, adj_ret, -adj_ret)
    table = []
    for t in THRESHOLDS:
        sel = conf >= t
        n = int(sel.sum())
        table.append({"threshold": t, "signals": n,
                      "hit_rate": round(100 * float(np.mean(pred_up[sel] == (y[sel] == 1))), 1) if n else None,
                      "avg_return": round(float(np.mean(dir_ret[sel])), 3) if n else None})
    passed = len(y) >= 150 and auc is not None and auc >= 0.52 and ll <= base_ll
    why = ("" if passed else
           "too few test headlines" if len(y) < 150 else
           "it couldn't tell winners from losers better than chance on unseen news" if auc is None or auc < 0.52 else
           "its probabilities were worse than always guessing the average")
    return {"accuracy": round(100 * acc, 1), "majority_accuracy": round(100 * majority, 1),
            "auc": round(auc, 4) if auc is not None else None, "logloss": round(ll, 5),
            "baseline_logloss": round(base_ll, 5), "thresholds": table, "passed": passed, "why_not": why}
