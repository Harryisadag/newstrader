"""Builds the price model's training set from free Alpaca data (your paper keys are enough).

For every finished trading day in the range:
  1. download a spread of that day's Benzinga articles: one short window at a different minute of every hour,
     so over many days the whole session is covered evenly
  2. keep articles tagged with 1-4 stocks. Today's ticker list is NOT used here: companies that were later
     taken over or delisted stay in, otherwise their (often big) moves would quietly disappear
  3. download 1-minute prices for those stocks + SPY and measure what each stock did from 1 minute after the
     article's final version was published to `horizon` minutes later, minus what the S&P 500 did
Article text is built exactly like the live news feed does (headline + summary + article body).
Labelled rows are cached in the database, so retraining later only downloads the new days.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

from ..ai.prefilter import Candidate
from ..ai.tickers import TickerTable
from ..config import SourceConfig
from ..db import Database, iso, parse_iso
from ..performance.prices import first_bar_at_or_after
from ..sources.alpaca_news import news_to_item
from ..sources.base import NewsItem
from .engine import build_features, readings_for
from .model import TrainingSample
from .sentiment import SentimentScores
from .text import target_text

log = logging.getLogger(__name__)

MARKET = "SPY"
LATENCY = timedelta(minutes=1)  # assume we'd trade one minute after the headline
TOLERANCE = timedelta(minutes=5)  # a price older/later than this doesn't count (thinly traded stock)
SLICE = timedelta(minutes=60)  # one sample window per hour of the session...
SAMPLE_WINDOW = timedelta(minutes=10)  # ...this long, starting at a different minute each day
MAX_SYMBOLS_PER_ARTICLE = 4  # round-ups tagging many stocks say little about each one
BARS_CHUNK = 40  # symbols per price request
MIN_SECONDS_BETWEEN_CALLS = 0.4  # stay well under Alpaca's free 200 requests/minute
DONE_AFTER_CLOSE = timedelta(minutes=20)  # a session is cached as complete only this long after it closed
_SYMBOL_RE = re.compile(r"^[A-Z]{1,5}(\.[A-Z]{1,2})?$")
_SRC = SourceConfig(id="training", type="alpaca_news", name="Benzinga (history)")

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


def sample_windows(open_t: datetime, last_t: datetime, day: str) -> list[tuple[datetime, datetime]]:
    """One SAMPLE_WINDOW per hour of the session, at a minute that changes from day to day (and hour to hour)."""
    out, t, i = [], open_t, 0
    while t < last_t:
        span = min(SLICE, last_t - t)
        win = min(SAMPLE_WINDOW, span)
        free = int((span - win).total_seconds() // 60)
        offset = zlib.crc32(f"{day}:{i}".encode()) % (free + 1) if free > 0 else 0
        s0 = t + timedelta(minutes=offset)
        out.append((s0, s0 + win))
        t += SLICE
        i += 1
    return out


def _done_key(horizon: int) -> str:
    return f"ml_done_days_v2_h{horizon}"


def _is_permission_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(k in text for k in ("subscription", "permit", "forbidden", "403", "not authorized", "unauthorized"))


def _as_dt(value) -> datetime | None:
    if isinstance(value, str):
        value = parse_iso(value)
    if isinstance(value, datetime) and value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value if isinstance(value, datetime) else None


def build_dataset(db: Database, broker, table: TickerTable, start: date, end: date, horizon_min: int,
                  max_articles: int, progress: Progress, cancelled: Callable[[], bool],
                  throttle: Throttle | None = None, now: datetime | None = None) -> dict:
    """Download + label every finished trading day in [start, end] that isn't cached yet. Blocking (run in a
    thread)."""
    throttle = throttle or Throttle()
    now = now or datetime.now(UTC)
    throttle.wait()
    sessions = broker.calendar(start, end)
    sessions = [s for s in sessions if s.get("open") and s.get("close")
                and parse_iso(s["close"]) + DONE_AFTER_CLOSE <= now]  # never a session that's still going
    if not sessions:
        raise ValueError("No finished trading days in that range.")
    done = set(db.kv_get(_done_key(horizon_min), []) or [])
    todo = [s for s in sessions if s["date"] not in done]
    per_day = max(10, math.ceil(max_articles / max(1, len(sessions))))
    stats = {"days": len(sessions), "days_cached": len(sessions) - len(todo), "articles": 0, "samples": 0,
             "labelled": 0, "days_failed": 0}
    feed = ["sip"]  # full-market prices if the account allows it, else the feed chosen in Settings
    failures_in_a_row = 0
    for i, sess in enumerate(todo):
        if cancelled():
            raise Cancelled()
        progress(i / max(1, len(todo)),
                 f"downloading day {i + 1} of {len(todo)} ({sess['date']}) - {stats['labelled']:,} labelled so far")
        try:
            _one_day(db, broker, sess, horizon_min, per_day, throttle, cancelled, feed, stats)
        except Cancelled:
            raise
        except Exception as exc:  # one bad day (rate limit, network blip) shouldn't throw away the whole run
            stats["days_failed"] += 1
            failures_in_a_row += 1
            log.warning("Training data for %s failed (%s) - will retry on the next training run", sess["date"], exc)
            if failures_in_a_row >= 5:
                raise RuntimeError(f"Alpaca downloads keep failing ({exc}). Days fetched so far are saved - "
                                   "try again later.") from exc
            continue
        failures_in_a_row = 0
        done.add(sess["date"])
        db.kv_set(_done_key(horizon_min), sorted(done))
    stats["price_feed"] = feed[0] or "settings"
    progress(1.0, "prices downloaded")
    return stats


def _one_day(db: Database, broker, sess: dict, horizon_min: int, per_day: int, throttle: Throttle,
             cancelled: Callable[[], bool], feed: list, stats: dict) -> None:
    open_t, close_t = parse_iso(sess["open"]), parse_iso(sess["close"])
    last_t = close_t - timedelta(minutes=horizon_min) - LATENCY - timedelta(minutes=1)
    if last_t <= open_t:
        return
    windows = sample_windows(open_t, last_t, sess["date"])
    per_window = max(3, math.ceil(per_day / len(windows)))
    articles, seen_ids, seen_titles = [], set(), set()
    for s0, s1 in windows:
        if cancelled():
            raise Cancelled()
        throttle.wait()
        for n in broker.news(s0, s1, None, limit=per_window, include_content=True):
            item = news_to_item(n, _SRC)  # same text the live feed would have produced
            if not item.title or item.external_id in seen_ids or item.title.lower() in seen_titles:
                continue
            syms = [s for s in item.symbols if _SYMBOL_RE.match(s)]
            if not 1 <= len(syms) <= MAX_SYMBOLS_PER_ARTICLE or len(syms) != len(item.symbols):
                continue
            created, updated = item.published_at, _as_dt(n.get("updated_at") if isinstance(n, dict) else None)
            # If the story was edited later, its text may describe what happened after it first came out -
            # so measure the move from the time this final version existed.
            at = max(t for t in (created, updated) if t is not None) if created else None
            if at is None or not (open_t <= at <= last_t):
                continue
            seen_ids.add(item.external_id)
            seen_titles.add(item.title.lower())
            articles.append({"id": item.external_id, "headline": item.title[:500], "body": item.body[:7000],
                             "symbols": syms, "at": at})
    stats["articles"] += len(articles)
    pairs = [(a, sym) for a in articles for sym in a["symbols"]]
    if not pairs:
        return
    symbols = sorted({sym for _, sym in pairs} | {MARKET})
    bars = _day_bars(broker, symbols, open_t, close_t, throttle, cancelled, feed)
    spy = bars.get(MARKET, [])
    rows = []
    for a, sym in pairs:
        lab = compute_label(bars.get(sym, []), spy, a["at"], horizon_min, close_t)
        rows.append((a["id"], sym, horizon_min, iso(a["at"]), a["headline"], a["body"], json.dumps(a["symbols"]),
                     lab.status, lab.entry_price, lab.exit_price, lab.ret_pct, lab.spy_ret_pct, lab.adj_ret_pct,
                     iso(), 2))
        stats["samples"] += 1
        stats["labelled"] += lab.status == "ok"
    db.executemany(
        "INSERT OR REPLACE INTO ml_samples (news_id, symbol, horizon_min, published_at, headline, summary, "
        "symbols, label_status, entry_price, exit_price, ret_pct, spy_ret_pct, adj_ret_pct, created_at, "
        "text_version) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)


def _day_bars(broker, symbols: list[str], open_t: datetime, close_t: datetime, throttle: Throttle,
              cancelled: Callable[[], bool], feed: list) -> dict[str, list[dict]]:
    """All of a day's bars from ONE feed (so a stock and SPY are always measured the same way)."""
    start_b, end_b = open_t - timedelta(minutes=1), close_t + timedelta(minutes=1)
    while True:
        bars: dict[str, list[dict]] = {}
        try:
            for j in range(0, len(symbols), BARS_CHUNK):
                if cancelled():
                    raise Cancelled()
                throttle.wait()
                bars.update(broker.bars_multi(symbols[j:j + BARS_CHUNK], start_b, end_b, feed=feed[0]))
            return bars
        except Cancelled:
            raise
        except Exception as exc:
            if feed[0] is None or not _is_permission_error(exc):
                raise  # a real failure: the day is retried on the next training run
            log.info("Full-market (SIP) history isn't available on this account (%s) - using the feed from "
                     "Settings", exc)
            feed[0] = None  # and fetch the whole day again with it


def forget_cached_days(db: Database, horizon_min: int) -> None:
    db.kv_set(_done_key(horizon_min), [])


# -------------------------------------------------------------------------------------------- features
def _hash(model_id: str, text: str) -> str:
    return hashlib.sha1(f"{model_id}\n{text}".encode("utf-8", "ignore")).hexdigest()


def cached_sentiment(db: Database, model, texts: list[str], progress: Progress, cancelled: Callable[[], bool],
                     predict=None, batch: int = 64) -> list[SentimentScores]:
    """Sentiment scores for many texts from ONE model object (kept for the whole training run, even if the
    live engine switches models meanwhile), re-using scores computed in earlier runs."""
    predict = predict or model.predict
    model_id = model.model_id
    keys = [_hash(model_id, t) for t in texts]
    have: dict[str, SentimentScores] = {}
    uniq = list(dict.fromkeys(keys))
    for i in range(0, len(uniq), 500):
        chunk = uniq[i:i + 500]
        marks = ",".join("?" * len(chunk))
        for r in db.query(f"SELECT text_hash, positive, negative, neutral FROM ml_sentiment_cache "
                          f"WHERE model_id = ? AND text_hash IN ({marks})", [model_id, *chunk]):
            have[r["text_hash"]] = SentimentScores(r["positive"], r["negative"], r["neutral"])
    missing = list({k: t for k, t in zip(keys, texts, strict=True) if k not in have}.items())
    for i in range(0, len(missing), batch):
        if cancelled():
            raise Cancelled()
        part = missing[i:i + batch]
        scores = predict([t for _, t in part])
        db.executemany("INSERT OR REPLACE INTO ml_sentiment_cache (model_id, text_hash, positive, negative, neutral) "
                       "VALUES (?,?,?,?,?)", [(model_id, k, s.positive, s.negative, s.neutral)
                                              for (k, _), s in zip(part, scores, strict=True)])
        for (k, _), s in zip(part, scores, strict=True):
            have[k] = s
        progress((i + len(part)) / len(missing), f"reading headlines with the sentiment model "
                                                 f"({i + len(part):,}/{len(missing):,})")
    return [have[k] for k in keys]


def load_samples(db: Database, table: TickerTable, model, horizon_min: int, start: date, end: date,
                 min_move_pct: float, progress: Progress, cancelled: Callable[[], bool],
                 predict=None) -> tuple[list[TrainingSample], dict]:
    """Every labelled row in the date range -> model inputs. Rows with a move smaller than min_move_pct are
    marked `small`: they're left out of fitting but kept for the honest test on unseen news."""
    rows = db.query("SELECT * FROM ml_samples WHERE horizon_min = ? AND label_status = 'ok' AND text_version = 2 "
                    "AND published_at >= ? AND published_at < ? ORDER BY published_at",
                    (horizon_min, start.isoformat(), (end + timedelta(days=1)).isoformat()))
    counts = {"labelled": len(rows), "small_moves": 0}
    picked, targets = [], []
    for r in rows:
        info = table.get(r["symbol"])
        cand = Candidate(r["symbol"], info.name if info else r["symbol"], "tagged by source")
        t = target_text(r["headline"], r["summary"] or "", "text", cand, table)
        if t is None:
            continue
        item = NewsItem(source_id=_SRC.id, source_type=_SRC.type, source_name=_SRC.name,
                        external_id=str(r["news_id"]), title=r["headline"], body=r["summary"] or "",
                        published_at=parse_iso(r["published_at"]), symbols=json.loads(r["symbols"] or "[]"))
        picked.append((r, cand, item))
        targets.append(t)
    scores = cached_sentiment(db, model, [t.snippet for t in targets], progress, cancelled, predict)
    samples = []
    for (r, cand, item), t, s in zip(picked, targets, scores, strict=True):
        at = item.published_at
        small = abs(r["adj_ret_pct"]) < min_move_pct
        counts["small_moves"] += small
        # exactly the inputs live analysis builds (same text, same event rules)
        reading = readings_for(item, [(cand, t)], table)[0]
        feats = build_features(item, cand, t, s, table, at, reading)
        samples.append(TrainingSample(feats=feats, up=int(r["adj_ret_pct"] > 0), at=at, adj_ret=r["adj_ret_pct"],
                                      group=r["news_id"], ret=r["ret_pct"], small=small))
    counts["used"] = len(samples)
    return samples, counts
