"""Builds the price model's training set from free Alpaca data (your paper keys are enough).

For every trading day in the range:
  1. download a spread of that day's Benzinga headlines (a few per hour, so the whole day is covered)
  2. keep headlines tagged with 1-4 US-listed stocks
  3. download 1-minute prices for those stocks + SPY, and measure what each stock did from 1 minute after the
     headline to `horizon` minutes later, minus what the S&P 500 (SPY) did over the same minutes
Labelled rows are cached in the database, so retraining later only downloads the new days.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from ..ai.prefilter import Candidate
from ..ai.tickers import TickerTable
from ..db import Database, iso, parse_iso
from ..performance.prices import first_bar_at_or_after
from .engine import time_of_day
from .model import SampleFeatures, TrainingSample
from .sentiment import SentimentScores, SentimentService
from .text import mask_company, relevance, target_text

log = logging.getLogger(__name__)

MARKET = "SPY"
LATENCY = timedelta(minutes=1)  # assume we'd trade one minute after the headline
TOLERANCE = timedelta(minutes=5)  # a price older/later than this doesn't count (thinly traded stock)
SLICE = timedelta(minutes=60)
MAX_SYMBOLS_PER_ARTICLE = 4  # round-ups tagging many stocks say little about each one
BARS_CHUNK = 40  # symbols per price request
MIN_SECONDS_BETWEEN_CALLS = 0.4  # stay well under Alpaca's free 200 requests/minute

Progress = Callable[[float, str], None]


class Cancelled(Exception):
    pass


class Throttle:
    def __init__(self, min_gap: float = MIN_SECONDS_BETWEEN_CALLS, sleep=time.sleep):
        self.min_gap = min_gap
        self._last = 0.0
        self._sleep = sleep

    def wait(self) -> None:
        gap = time.monotonic() - self._last
        if gap < self.min_gap:
            self._sleep(self.min_gap - gap)
        self._last = time.monotonic()


@dataclass
class Label:
    status: str  # ok | no_bars | after_close
    entry_price: float | None = None
    exit_price: float | None = None
    ret_pct: float | None = None
    spy_ret_pct: float | None = None
    adj_ret_pct: float | None = None


def _price_near(bars: list[dict], when: datetime, entry: bool) -> float | None:
    """Entry: open of the first bar at/after `when`. Exit: close of the last bar at/before `when`.
    Either must be within TOLERANCE of `when`, otherwise the stock wasn't trading enough to trust it."""
    if entry:
        b = first_bar_at_or_after(bars, when)
        if b is None or parse_iso(b["t"]) - when > TOLERANCE:
            return None
        return b.get("o")
    best = None
    for b in bars:
        t = parse_iso(b["t"])
        if t is None:
            continue
        if t <= when:
            best = (t, b)
        else:
            break
    if best is None or when - best[0] > TOLERANCE:
        return None
    return best[1].get("c")


def compute_label(bars: list[dict], spy: list[dict], published: datetime, horizon_min: int,
                  close: datetime) -> Label:
    entry_t = published + LATENCY
    exit_t = entry_t + timedelta(minutes=horizon_min)
    if exit_t > close:
        return Label("after_close")
    e, x = _price_near(bars, entry_t, True), _price_near(bars, exit_t, False)
    se, sx = _price_near(spy, entry_t, True), _price_near(spy, exit_t, False)
    if not e or not x or not se or not sx:
        return Label("no_bars")
    ret = (x / e - 1) * 100
    spy_ret = (sx / se - 1) * 100
    return Label("ok", e, x, round(ret, 5), round(spy_ret, 5), round(ret - spy_ret, 5))


def _slices(open_t: datetime, last_t: datetime) -> list[tuple[datetime, datetime]]:
    out, t = [], open_t
    while t < last_t:
        out.append((t, min(t + SLICE, last_t)))
        t += SLICE
    return out


def _done_key(horizon: int) -> str:
    return f"ml_done_days_h{horizon}"


def build_dataset(db: Database, broker, table: TickerTable, start: date, end: date, horizon_min: int,
                  max_articles: int, progress: Progress, cancelled: Callable[[], bool],
                  throttle: Throttle | None = None) -> dict:
    """Download + label every trading day in [start, end] that isn't cached yet. Blocking (run in a thread)."""
    throttle = throttle or Throttle()
    throttle.wait()
    sessions = broker.calendar(start, end)
    sessions = [s for s in sessions if s.get("open") and s.get("close")]
    if not sessions:
        raise ValueError("No trading days in that range.")
    done = set(db.kv_get(_done_key(horizon_min), []) or [])
    todo = [s for s in sessions if s["date"] not in done]
    per_day = max(10, math.ceil(max_articles / max(1, len(sessions))))
    stats = {"days": len(sessions), "days_cached": len(sessions) - len(todo), "articles": 0, "samples": 0,
             "labelled": 0}
    for i, sess in enumerate(todo):
        if cancelled():
            raise Cancelled()
        progress(i / max(1, len(todo)),
                 f"downloading day {i + 1} of {len(todo)} ({sess['date']}) - {stats['labelled']:,} labelled so far")
        open_t, close_t = parse_iso(sess["open"]), parse_iso(sess["close"])
        last_t = close_t - timedelta(minutes=horizon_min) - LATENCY - timedelta(minutes=1)
        if last_t <= open_t:
            done.add(sess["date"])
            continue
        slices = _slices(open_t, last_t)
        per_slice = max(3, math.ceil(per_day / len(slices)))
        articles, seen_ids, seen_titles = [], set(), set()
        for s0, s1 in slices:
            if cancelled():
                raise Cancelled()
            throttle.wait()
            for n in broker.news(s0, s1, None, limit=per_slice, include_content=False):
                nid = str(n.get("id"))
                title = (n.get("headline") or "").strip()
                if not title or nid in seen_ids or title.lower() in seen_titles:
                    continue
                syms = [str(x).upper() for x in (n.get("symbols") or [])]
                if not 1 <= len(syms) <= MAX_SYMBOLS_PER_ARTICLE:
                    continue
                created = n.get("created_at")
                created = parse_iso(created) if isinstance(created, str) else created
                if created is None or not (open_t <= created <= last_t):
                    continue
                seen_ids.add(nid)
                seen_titles.add(title.lower())
                articles.append({"id": nid, "headline": title, "summary": (n.get("summary") or "").strip(),
                                 "symbols": syms, "created_at": created})
        stats["articles"] += len(articles)
        pairs = [(a, sym) for a in articles for sym in a["symbols"] if table.is_valid(sym)]
        if pairs:
            symbols = sorted({sym for _, sym in pairs} | {MARKET})
            bars: dict[str, list[dict]] = {}
            for j in range(0, len(symbols), BARS_CHUNK):
                if cancelled():
                    raise Cancelled()
                throttle.wait()
                bars.update(broker.bars_multi(symbols[j:j + BARS_CHUNK], open_t - timedelta(minutes=1),
                                              close_t + timedelta(minutes=1)))
            spy = bars.get(MARKET, [])
            rows = []
            for a, sym in pairs:
                lab = compute_label(bars.get(sym, []), spy, a["created_at"], horizon_min, close_t)
                rows.append((a["id"], sym, horizon_min, iso(a["created_at"]), a["headline"][:500], a["summary"][:2000],
                             json.dumps(a["symbols"]), lab.status, lab.entry_price, lab.exit_price, lab.ret_pct,
                             lab.spy_ret_pct, lab.adj_ret_pct, iso()))
                stats["samples"] += 1
                stats["labelled"] += lab.status == "ok"
            db.executemany(
                "INSERT OR REPLACE INTO ml_samples (news_id, symbol, horizon_min, published_at, headline, summary, "
                "symbols, label_status, entry_price, exit_price, ret_pct, spy_ret_pct, adj_ret_pct, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        done.add(sess["date"])
        db.kv_set(_done_key(horizon_min), sorted(done))
    progress(1.0, "prices downloaded")
    return stats


def forget_cached_days(db: Database, horizon_min: int) -> None:
    db.kv_set(_done_key(horizon_min), [])


# -------------------------------------------------------------------------------------------- features
def _hash(model_id: str, text: str) -> str:
    return hashlib.sha1(f"{model_id}\n{text}".encode("utf-8", "ignore")).hexdigest()


def cached_sentiment(db: Database, sentiment: SentimentService, texts: list[str], progress: Progress,
                     cancelled: Callable[[], bool], batch: int = 64) -> list[SentimentScores]:
    """FinBERT scores for many texts, re-using scores computed in earlier training runs."""
    model_id = sentiment.model_id
    keys = [_hash(model_id, t) for t in texts]
    have: dict[str, SentimentScores] = {}
    uniq = list(dict.fromkeys(keys))
    for i in range(0, len(uniq), 500):
        chunk = uniq[i:i + 500]
        marks = ",".join("?" * len(chunk))
        for r in db.query(f"SELECT text_hash, positive, negative, neutral FROM ml_sentiment_cache "
                          f"WHERE model_id = ? AND text_hash IN ({marks})", [model_id, *chunk]):
            have[r["text_hash"]] = SentimentScores(r["positive"], r["negative"], r["neutral"])
    missing = [(k, t) for k, t in zip(keys, texts, strict=True) if k not in have]
    missing = list(dict((k, t) for k, t in missing).items())
    for i in range(0, len(missing), batch):
        if cancelled():
            raise Cancelled()
        part = missing[i:i + batch]
        scores = sentiment.predict([t for _, t in part])
        db.executemany("INSERT OR REPLACE INTO ml_sentiment_cache (model_id, text_hash, positive, negative, neutral) "
                       "VALUES (?,?,?,?,?)", [(model_id, k, s.positive, s.negative, s.neutral)
                                              for (k, _), s in zip(part, scores, strict=True)])
        for (k, _), s in zip(part, scores, strict=True):
            have[k] = s
        progress((i + len(part)) / len(missing), f"reading headlines with the sentiment model "
                                                 f"({i + len(part):,}/{len(missing):,})")
    return [have[k] for k in keys]


def load_samples(db: Database, table: TickerTable, sentiment: SentimentService, horizon_min: int, start: date,
                 end: date, min_move_pct: float, progress: Progress,
                 cancelled: Callable[[], bool]) -> tuple[list[TrainingSample], dict]:
    """Labelled rows in the date range -> model inputs. Moves smaller than min_move_pct are left out."""
    rows = db.query("SELECT * FROM ml_samples WHERE horizon_min = ? AND label_status = 'ok' AND published_at >= ? "
                    "AND published_at < ? ORDER BY published_at",
                    (horizon_min, start.isoformat(), (end + timedelta(days=1)).isoformat()))
    counts = {"labelled": len(rows), "small_moves": 0}
    picked, targets = [], []
    for r in rows:
        if abs(r["adj_ret_pct"]) < min_move_pct:
            counts["small_moves"] += 1
            continue
        info = table.get(r["symbol"])
        cand = Candidate(r["symbol"], info.name if info else r["symbol"], "tagged by source")
        t = target_text(r["headline"], r["summary"] or "", "text", cand, table)
        if t is None:
            continue
        picked.append((r, cand))
        targets.append(t)
    scores = cached_sentiment(db, sentiment, [t.snippet for t in targets], progress, cancelled)
    samples = []
    for (r, cand), t, s in zip(picked, targets, scores, strict=True):
        n_sym = len(json.loads(r["symbols"] or "[]"))
        at = parse_iso(r["published_at"])
        feats = SampleFeatures(text=mask_company(t.snippet, cand, table), sentiment=s, relevance=relevance(t, n_sym),
                               in_headline=t.in_headline, tagged=t.tagged, n_symbols=n_sym, time_of_day=time_of_day(at))
        samples.append(TrainingSample(feats=feats, up=int(r["adj_ret_pct"] > 0), at=at, adj_ret=r["adj_ret_pct"],
                                      group=r["news_id"]))
    counts["used"] = len(samples)
    return samples, counts
