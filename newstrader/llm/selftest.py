""""Test my PC": runs Pro AI on 20 labelled headlines (samples.json) and reports how fast it answers, whether every
answer was valid JSON, and how many it called right."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from ..ai.prefilter import Candidate, PrefilterResult, prefilter
from ..sources.base import NewsItem
from .engine import allowed_tickers

NOW = datetime(2026, 10, 7, 15, 0, tzinfo=UTC)
SAME_COMPANY = {"GOOG": "GOOGL", "FXI": "MCHI"}  # share classes / two big China funds: the same answer


def load_samples() -> list[dict]:
    return json.loads(Path(__file__).with_name("samples.json").read_text(encoding="utf-8"))["items"]


def news_item(sample: dict, now: datetime = NOW) -> NewsItem:
    transcript = sample.get("source_type") == "stream"
    item = NewsItem(source_id="selftest", source_type=sample.get("source_type") or "rss", source_name="test",
                    external_id=sample["id"], title=sample["title"], body=sample.get("body") or "",
                    published_at=now, symbols=list(sample.get("symbols") or []),
                    kind="transcript" if transcript else "text")
    if transcript:  # how the app hands over live TV speech
        item.body = "New:\n[10:00:00] " + (sample["title"] + " " + (sample.get("body") or "")).strip()
    return item


def sample_prefilter(sample: dict, item: NewsItem, tickers=None) -> PrefilterResult:
    """The app's own company finder when its ticker table is loaded, else the candidates stored with the sample."""
    if tickers is not None and getattr(tickers, "loaded", False):
        return prefilter(item.text, tickers, item.symbols, allow_keyword_only=True, country_etfs=True)
    cands = [Candidate(c["symbol"], c.get("name", ""), c.get("why", "")) for c in sample.get("candidates") or []]
    keywords = list(sample.get("keywords") or [])
    return PrefilterResult(hit=bool(cands or keywords), candidates=cands, keywords=keywords)


def _norm(sym: str) -> str:
    return SAME_COMPANY.get(sym.upper(), sym.upper())


def judge(sample: dict, text: str, allowed: list[str]) -> tuple[bool, bool, dict]:
    """(answer is valid JSON in the right shape, answer is right, {ticker: direction})."""
    try:
        data = json.loads(text)
        signals = data["signals"]
        got = {_norm(s["ticker"]): str(s["direction"]) for s in signals}
        ok = isinstance(signals, list) and all(s["ticker"] in allowed for s in signals)
    except (ValueError, KeyError, TypeError):
        return False, False, {}
    expect = {_norm(k): v for k, v in (sample.get("expect") or {}).items()}
    right = all(got.get(sym, "neutral") == label for sym, label in expect.items())
    right = right and all(got.get(_norm(a), "neutral") == "neutral" for a in sample.get("absent") or [])
    return ok, right, got


async def run_selftest(engine, tickers=None, samples: list[dict] | None = None,
                       progress_cb: Callable[[int, int], None] | None = None, warm_up: bool = True) -> dict:
    """Ask `engine` about each sample, one at a time. Returns {avg_seconds, max_seconds, json_ok, right, total,
    fully_on_gpu, items}."""
    samples = samples if samples is not None else load_samples()
    rows = []
    if warm_up and samples:  # the first question also reads the long instructions; real use has them cached
        first = news_item(samples[0])
        await engine.analyze(first, sample_prefilter(samples[0], first, tickers), True, NOW)
    for i, s in enumerate(samples):
        item = news_item(s)
        pre = sample_prefilter(s, item, tickers)
        started = time.monotonic()
        res = await engine.analyze(item, pre, True, NOW)
        seconds = time.monotonic() - started
        ok, right, got = judge(s, res.text, allowed_tickers(pre)) if res.ok else (False, False, {})
        rows.append({"id": s["id"], "title": s["title"], "seconds": round(seconds, 2), "json_ok": ok,
                     "right": right, "expect": s.get("expect") or {}, "got": got,
                     "error": None if res.ok else res.error})
        if progress_cb:
            progress_cb(i + 1, len(samples))
    secs = [r["seconds"] for r in rows]
    server = getattr(engine, "server", None)
    return {"avg_seconds": round(sum(secs) / len(secs), 2) if secs else 0.0,
            "max_seconds": round(max(secs), 2) if secs else 0.0,
            "json_ok": sum(r["json_ok"] for r in rows), "right": sum(r["right"] for r in rows), "total": len(rows),
            "fully_on_gpu": getattr(server, "fully_on_gpu", None), "items": rows}
