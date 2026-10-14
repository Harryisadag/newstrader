"""The student model (shared training): a small, fast model that learned from Pro AI's labels how a careful reader
calls a headline for one company - bullish, bearish or neutral - so every NewsTrader gets some of Pro AI's reading
without a graphics card, and without any AI service at trade time.

Inputs per (headline, stock): the words of the headline and the sentences about the stock, with the stock's name
replaced by "companyx" and every other company by "companyy" (so it learns wording, not names), hashed into a
fixed-size vector; FinBERT's three scores; and what the event rules found (the kind of event as words, its direction
and strength). The model is a multinomial logistic regression: it reads a story in about a millisecond.

It is tested on the newest 20% of the labels (by date), which it never saw while learning - against Pro AI's answers
there, and on the 20 labelled headlines of Pro AI's "Test my PC" (llm/samples.json, labelled by people). A new student
replaces the one in use only if it does at least as well on the test labels the one in use never learned from (each
model keeps a fingerprint of the stories it learned from for that).

The engine (engine.py) uses it only where it would otherwise go by the wording alone (no news event recognised), and
those signals stay manual-review-only; when an event is recognised, a confident disagreement only lowers confidence.
Saved as plain numbers (.npz + .json), never as a pickle, so a model file can't run code when loaded.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from ..ai.prefilter import Candidate
from ..ai.tickers import TickerTable
from .events import Reading
from .sentiment import SentimentScores
from .text import Target, mask_companies

log = logging.getLogger(__name__)

N_HASH = 2 ** 18
CLASSES = ("bearish", "neutral", "bullish")
NUMERIC = ["pos", "neg", "neu", "event_dir", "event_score", "in_headline"]
VERSION = 1
C_GRID = (0.1, 0.3, 1.0, 3.0)
VALID_FRAC = 0.2  # the newest 20% of the labels (by date) test it
MIN_LABELS = 200
MIN_PER_CLASS = 10  # of each direction among the older 80%
MIN_VALID = 30
NPZ, META = "student.npz", "student.json"


@dataclass
class StudentFeatures:
    text: str  # masked headline + sentences about the company, plus the event words
    sentiment: SentimentScores
    event_dir: int = 0  # the event rules' direction (-1, 0, +1)
    event_strength: float = 0.0
    in_headline: bool = True

    def numeric(self) -> list[float]:
        s = self.sentiment
        return [s.positive, s.negative, s.neutral, float(self.event_dir), self.event_dir * self.event_strength,
                float(self.in_headline)]


def event_words(reading: Reading | None) -> str:
    """The event rules' findings as words ("evt_guidance_cut flag_unconfirmed evtdir_down")."""
    if reading is None:
        return ""
    words = [f"evt_{e.kind}" for e in reading.events if e.role != "actor"] + [f"flag_{f}" for f in reading.flags]
    if reading.direction:
        words.append("evtdir_up" if reading.direction > 0 else "evtdir_down")
    return " ".join(dict.fromkeys(words))


def student_features(cand: Candidate, target: Target, scores: SentimentScores, table: TickerTable,
                     reading: Reading | None = None, others=()) -> StudentFeatures:
    """The student's inputs - shared by live analysis, training and the tests on unseen labels."""
    text = mask_companies(target.snippet, cand, list(others), table)
    return StudentFeatures(text=f"{text} {event_words(reading)}".strip(), sentiment=scores,
                           event_dir=reading.direction if reading is not None else 0,
                           event_strength=reading.strength if reading is not None else 0.0,
                           in_headline=target.in_headline)


def _vectorizer():
    from sklearn.feature_extraction.text import HashingVectorizer

    return HashingVectorizer(n_features=N_HASH, ngram_range=(1, 2), alternate_sign=False, norm="l2",
                             lowercase=True, token_pattern=r"(?u)\b\w+\b")


def _matrix(feats: list[StudentFeatures], mean: np.ndarray, std: np.ndarray):
    from scipy import sparse

    text = _vectorizer().transform([f.text for f in feats])
    num = (np.array([f.numeric() for f in feats], dtype=np.float64) - mean) / std
    return sparse.hstack([text, sparse.csr_matrix(num)], format="csr")


def story_keys(groups) -> np.ndarray:
    """64-bit fingerprints of news ids (which stories a model learned from, without listing them)."""
    return np.array([int.from_bytes(hashlib.sha256(str(g).encode()).digest()[:8], "little") for g in groups],
                    dtype=np.uint64)


class StudentModel:
    def __init__(self, coef: np.ndarray, intercept: np.ndarray, mean: np.ndarray, std: np.ndarray, meta: dict,
                 seen: np.ndarray | None = None):
        self.coef = np.asarray(coef, dtype=np.float64)  # one row per class (CLASSES order)
        self._coef_t = np.ascontiguousarray(self.coef.T)  # (a transposed view would be copied on every call)
        self.intercept = np.asarray(intercept, dtype=np.float64)
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)
        self.meta = meta
        # fingerprints of the stories it learned from (None = not known: treated as if it saw every story)
        self.seen = None if seen is None else np.unique(np.asarray(seen, dtype=np.uint64))

    def saw(self, groups: list[str]) -> np.ndarray:
        """For each news id: did this model learn from it?"""
        if self.seen is None:
            return np.ones(len(groups), dtype=bool)
        return np.isin(story_keys(groups), self.seen)

    def predict_proba(self, feats: list[StudentFeatures]) -> np.ndarray:
        """(n, 3) probabilities in CLASSES order (bearish, neutral, bullish)."""
        if not feats:
            return np.zeros((0, len(CLASSES)))
        z = np.asarray(_matrix(feats, self.mean, self.std) @ self._coef_t) + self.intercept
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, feats: list[StudentFeatures]) -> list[str]:
        return [CLASSES[i] for i in self.predict_proba(feats).argmax(axis=1)]

    # ---------------------------------------------------------------- files
    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        tmp = folder / "student.tmp.npz"
        extra = {} if self.seen is None else {"seen": self.seen}
        np.savez_compressed(tmp, coef=self.coef.astype(np.float32), intercept=self.intercept, mean=self.mean,
                            std=self.std, **extra)
        os.replace(tmp, folder / NPZ)
        meta_tmp = folder / (META + ".tmp")
        meta_tmp.write_text(json.dumps(self.meta, indent=2, default=str), encoding="utf-8")
        os.replace(meta_tmp, folder / META)

    @classmethod
    def load(cls, folder: Path) -> StudentModel | None:
        npz, meta_file = Path(folder) / NPZ, Path(folder) / META
        if not npz.exists() or not meta_file.exists():
            return None
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        if (meta.get("version") != VERSION or meta.get("n_hash") != N_HASH or meta.get("numeric") != NUMERIC
                or meta.get("classes") != list(CLASSES)):
            log.warning("The saved student model is from another version of NewsTrader - train it again")
            return None
        with np.load(npz, allow_pickle=False) as data:
            coef, intercept, mean, std = data["coef"], data["intercept"], data["mean"], data["std"]
            seen = data["seen"] if "seen" in data.files else None
        width = N_HASH + len(NUMERIC)
        if (coef.shape != (len(CLASSES), width) or intercept.shape != (len(CLASSES),)
                or mean.shape != (len(NUMERIC),) or std.shape != (len(NUMERIC),)
                or (seen is not None and (seen.ndim != 1 or seen.dtype != np.uint64))):
            log.warning("The saved student model file has the wrong shape - train it again")
            return None
        return cls(coef, intercept, mean, std, meta, seen)


def delete_student(folder: Path) -> None:
    for name in (NPZ, META):
        p = Path(folder) / name
        if p.exists():
            p.unlink()


# -------------------------------------------------------------------------------------------- training
@dataclass
class StudentSample:
    feats: StudentFeatures
    label: str  # the teacher's direction
    at: datetime  # when the news came out
    group: str  # news id - labels of one story stay on the same side of the split
    source: str = "own"  # "own" or the contributor code of a shared file


def split_by_time(samples: list[StudentSample], frac: float = VALID_FRAC) -> tuple[list, list]:
    """Older (1 - frac) -> learning, newest `frac` -> testing. Never shuffled, and one story is never split."""
    ordered = sorted(samples, key=lambda s: (s.at, s.group))
    n = len(ordered)
    i = int(n * (1 - frac))
    while 0 < i < n and ordered[i].group == ordered[i - 1].group:
        i += 1
    return ordered[:i], ordered[i:]


def _y(rows: list[StudentSample]) -> np.ndarray:
    return np.array([CLASSES.index(r.label) for r in rows])


def _norm(rows: list[StudentSample]) -> tuple[np.ndarray, np.ndarray]:
    num = np.array([r.feats.numeric() for r in rows], dtype=np.float64)
    std = num.std(axis=0)
    return num.mean(axis=0), np.where(std < 1e-9, 1.0, std)


def _fit(rows: list[StudentSample], c: float, meta: dict) -> StudentModel:
    from sklearn.linear_model import LogisticRegression

    mean, std = _norm(rows)
    clf = LogisticRegression(C=c, solver="newton-cg", max_iter=200)  # (by far the fastest on hashed words)
    clf.fit(_matrix([r.feats for r in rows], mean, std), _y(rows))
    if list(clf.classes_) != list(range(len(CLASSES))):
        raise ValueError("Every direction (bullish, bearish, neutral) needs labels to learn from.")
    return StudentModel(clf.coef_, clf.intercept_, mean, std, dict(meta), story_keys({r.group for r in rows}))


def _logloss(y: np.ndarray, p: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1.0))))


def accuracy(model: StudentModel, rows: list[StudentSample]) -> float | None:
    """% of `rows` where the model's call is the teacher's."""
    if not rows:
        return None
    pred = model.predict_proba([r.feats for r in rows]).argmax(axis=1)
    return round(100 * float(np.mean(pred == _y(rows))), 1)


def compare_with(current: StudentModel, tested: StudentModel, valid: list[StudentSample], acc: float) -> dict:
    """Is the new student (`tested`: learned from the older 80%, `acc` on the newest 20%) at least as good as the one
    in use? Fair only on test labels the one in use never learned from (it learned from all of its own labels), so:
    on those when there are enough, else against the score the one in use got on its own test when it was trained."""
    unseen = [r for r, saw in zip(valid, current.saw([r.group for r in valid]), strict=True) if not saw]
    if len(unseen) >= MIN_VALID:
        cur, new, basis, n = accuracy(current, unseen), accuracy(tested, unseen), "same_labels", len(unseen)
    else:
        cur, new, basis, n = (current.meta.get("report") or {}).get("accuracy"), acc, "own_test", len(valid)
    if not isinstance(cur, (int, float)) or isinstance(cur, bool):
        cur = None
    return {"current_accuracy": cur, "new_accuracy": new, "compare_basis": basis, "compared_on": n,
            "not_worse": cur is None or new >= cur}


def train_student(samples: list[StudentSample], sentiment_model: str, current: StudentModel | None = None,
                  progress: Callable[[float, str], None] | None = None) -> tuple[StudentModel, dict]:
    """Learn from the older 80% of the labels, test on the newest 20% (and compare with `current`, the student in use,
    on those of them it never learned from - compare_with), then learn from all of them for the saved model."""
    if len(samples) < MIN_LABELS:
        raise ValueError(f"Only {len(samples):,} labels could be matched with your downloaded headlines - at least "
                         f"{MIN_LABELS} are needed. Let Pro AI label more old news, get everyone's labels, or train "
                         "the price model on more history (that downloads the headlines labels refer to).")
    train, valid = split_by_time(samples)
    counts = Counter(s.label for s in train)
    if any(counts[c] < MIN_PER_CLASS for c in CLASSES) or len(valid) < MIN_VALID:
        raise ValueError("Not enough labels of each kind (bullish, bearish and neutral) to learn from and test on "
                         f"yet ({', '.join(f'{c} {counts[c]}' for c in CLASSES)}) - label more old news first.")
    meta = {"version": VERSION, "n_hash": N_HASH, "numeric": NUMERIC, "classes": list(CLASSES),
            "sentiment_model": sentiment_model}

    # 1) how strongly to regularise: picked on the newest part of the learning labels
    inner_tr, inner_va = split_by_time(train)
    best_c, losses = 1.0, {}
    inner_counts = Counter(s.label for s in inner_tr)
    if inner_va and all(inner_counts[c] >= 2 for c in CLASSES):
        for i, c in enumerate(C_GRID):
            m = _fit(inner_tr, c, meta)
            losses[c] = _logloss(_y(inner_va), m.predict_proba([r.feats for r in inner_va]))
            if progress:
                progress(0.1 + 0.5 * (i + 1) / len(C_GRID), f"tuning ({i + 1}/{len(C_GRID)})")
        best_c = min(losses, key=losses.get)

    # 2) the honest test: learn from the older 80%, score the newest 20% it never saw
    tested = _fit(train, best_c, meta)
    p_va = tested.predict_proba([r.feats for r in valid])
    pred, y_va = p_va.argmax(axis=1), _y(valid)
    acc = round(100 * float(np.mean(pred == y_va)), 1)
    common = counts.most_common(1)[0][0]
    majority = round(100 * float(np.mean(y_va == CLASSES.index(common))), 1)
    compared = (compare_with(current, tested, valid, acc) if current is not None else
                {"current_accuracy": None, "new_accuracy": acc, "compare_basis": "", "compared_on": 0,
                 "not_worse": True})
    per_class = {c: {"labels": int(np.sum(y_va == k)), "right": int(np.sum((y_va == k) & (pred == k)))}
                 for k, c in enumerate(CLASSES)}
    confusion = {f"{CLASSES[a]}->{CLASSES[b]}": int(np.sum((y_va == a) & (pred == b)))
                 for a in range(len(CLASSES)) for b in range(len(CLASSES))}
    if progress:
        progress(0.8, "learning from all labels")

    # 3) the model to use: learn from every label (the newest included)
    report = {"accuracy": acc, "majority_accuracy": majority, "always_guess": common, **compared,
              "n_train": len(train), "n_valid": len(valid), "per_class": per_class, "confusion": confusion,
              "valid_from": valid[0].at.date().isoformat(), "valid_to": valid[-1].at.date().isoformat(),
              "best_c": best_c, "validation_logloss": {str(k): round(v, 5) for k, v in losses.items()},
              "label_mix": {c: sum(1 for s in samples if s.label == c) for c in CLASSES}}
    final = _fit(samples, best_c, meta)
    ordered = sorted(s.at for s in samples)
    final.meta.update({"trained_at": datetime.now().astimezone().isoformat(), "n_labels": len(samples),
                       "labels_from": ordered[0].date().isoformat(), "labels_to": ordered[-1].date().isoformat(),
                       "report": report})
    return final, report


# -------------------------------------------------------------------------------------------- "Test my PC" headlines
def score_samples(model: StudentModel, predict: Callable[[list[str]], list[SentimentScores]],
                  table: TickerTable) -> dict:
    """How the student reads the labelled headlines of Pro AI's "Test my PC" (llm/samples.json, labelled by people):
    {"right", "total"} over every (headline, stock) with an expected answer."""
    from ..llm.selftest import SAME_COMPANY, load_samples, news_item
    from .engine import readings_for
    from .text import target_text

    def norm(sym: str) -> str:
        return SAME_COMPANY.get(sym.upper(), sym.upper())

    right = total = 0
    for s in load_samples():
        expect = {norm(k): v for k, v in (s.get("expect") or {}).items()}
        if not expect:
            continue
        item = news_item(s)
        cands = [Candidate(c["symbol"], c.get("name", ""), c.get("why", "")) for c in s.get("candidates") or []]
        known = {norm(c.symbol) for c in cands}
        cands += [Candidate(sym, "", "tagged by source") for sym in expect if sym not in known]
        targets = []
        for c in cands:
            t = target_text(item.title, item.body, item.kind, c, table)
            if t is not None:
                targets.append((c, t))
        calls: dict[str, str] = {}
        if targets:
            readings = readings_for(item, targets, table)
            scores = predict([t.snippet for _, t in targets])
            feats = [student_features(c, t, sc, table, r, [o for o, _ in targets])
                     for (c, t), sc, r in zip(targets, scores, readings, strict=True)]
            calls = {norm(c.symbol): d for (c, _t), d in zip(targets, model.predict(feats), strict=True)}
        for sym, label in expect.items():
            total += 1
            right += calls.get(sym, "neutral") == label
    return {"right": right, "total": total}
