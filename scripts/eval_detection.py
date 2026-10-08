"""Measures how well the local engine reads news: which companies it finds, and good / bad / neutral for each.

    python scripts/eval_detection.py                         # dev set, built-in word list (works offline)
    python scripts/eval_detection.py --sentiment finbert      # downloads FinBERT (~110 MB) the first time
    python scripts/eval_detection.py --set set2 --compare     # also runs the v0.2 behaviour (no event rules)

The headline sets are described in tests/fixtures/eval/README.md. Labels say what a trader would expect the stock
to do in the next hour (bullish / bearish / neutral); each item was labelled twice, blind, and disagreements settled.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from newstrader.ai.prefilter import prefilter  # noqa: E402
from newstrader.ai.tickers import TickerTable  # noqa: E402
from newstrader.config import AppSettings  # noqa: E402
from newstrader.db import Database  # noqa: E402
from newstrader.ml.engine import LocalMLEngine  # noqa: E402
from newstrader.ml.sentiment import SentimentService  # noqa: E402
from newstrader.sources.base import NewsItem  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "eval"
SAME_COMPANY = {"GOOG": "GOOGL", "FXI": "MCHI"}  # share classes / two big China funds: the same answer
NOW = datetime(2026, 10, 7, 15, 0, tzinfo=UTC)


def load_items(which: str) -> list[dict]:
    path = Path(which) if Path(which).suffix == ".jsonl" else FIXTURES / f"{which}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def make_table(db_dir: Path) -> TickerTable:
    assets = json.loads((FIXTURES / "eval_tickers.json").read_text(encoding="utf-8"))
    table = TickerTable(Database(db_dir / "eval.db"))
    table.replace_all(assets)
    return table


def make_engine(table: TickerTable, sentiment: SentimentService, event_rules: bool) -> LocalMLEngine:
    settings = AppSettings()
    settings.ml.event_rules = event_rules
    settings.ml.sentiment_only_trading = "always"  # measure the engine's own confidence, not the review cap
    ctx = SimpleNamespace(config=SimpleNamespace(settings=settings), state=SimpleNamespace(mode="paper"))
    engine = LocalMLEngine(ctx, sentiment=sentiment, model_dir=Path(tempfile.mkdtemp()) / "none")
    engine.tickers = table
    engine.set_price_model(None)
    return engine


def _norm(sym: str) -> str:
    return SAME_COMPANY.get(sym, sym)


def evaluate(items: list[dict], engine: LocalMLEngine, table: TickerTable, buy_threshold: int = 80) -> dict:
    found = missed = absent_hits = extra = 0
    correct = total = 0
    conf = Counter()  # (label, predicted)
    by_cat: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    trades_right = trades_wrong = 0
    mistakes = []
    for it in items:
        item = NewsItem(source_id="eval", source_type=it.get("source_type") or "rss", source_name="eval",
                        external_id=it["id"], title=it["title"], body=it.get("body") or "", url=it.get("url") or "",
                        published_at=NOW, symbols=list(it.get("symbols") or []),
                        kind="transcript" if it.get("source_type") == "stream" else "text")
        if item.kind == "transcript":
            item.body = "New:\n[10:00:00] " + (it["title"] + " " + (it.get("body") or "")).strip()
        pre = prefilter(item.text, table, item.symbols, allow_keyword_only=False,
                        country_etfs=engine.ctx.config.settings.ml.country_etfs)
        payload = engine._analyze_sync(item, pre, NOW)
        got = {_norm(c.symbol) for c in pre.candidates}
        signals = {_norm(s["ticker"]): s for s in payload["signals"]}
        expect = {_norm(k): v for k, v in it["expect"].items()}
        absent = {_norm(a) for a in it.get("absent") or []}
        found += len(set(expect) & got)
        missed += len(set(expect) - got)
        absent_hits += len(absent & got)
        extra += len(got - set(expect) - absent)
        for sym in absent & got:  # a wrong company picked up: a signal on it is a mistake
            s = signals.get(sym)
            if s and s["confidence"] >= buy_threshold:
                trades_wrong += 1
        for sym, label in expect.items():
            s = signals.get(sym)
            pred = s["direction"] if s else "neutral"
            total += 1
            ok = pred == label
            correct += ok
            conf[(label, pred)] += 1
            by_cat[it.get("category", "?")][0] += ok
            by_cat[it.get("category", "?")][1] += 1
            if s and s["confidence"] >= buy_threshold:
                if ok:
                    trades_right += 1
                else:
                    trades_wrong += 1
            if not ok:
                mistakes.append({"id": it["id"], "ticker": sym, "label": label, "pred": pred,
                                 "conf": s["confidence"] if s else None, "found": sym in got, "title": it["title"]})
    directional = sum(v for (lab, _p), v in conf.items() if lab != "neutral")
    dir_right = sum(v for (lab, p), v in conf.items() if lab != "neutral" and p == lab)
    neutral = sum(v for (lab, _p), v in conf.items() if lab == "neutral")
    neutral_flagged = sum(v for (lab, p), v in conf.items() if lab == "neutral" and p != "neutral")
    flipped = sum(v for (lab, p), v in conf.items() if {lab, p} == {"bullish", "bearish"})
    return {
        "items": len(items),
        "companies_found_pct": round(100 * found / max(1, found + missed), 1),
        "wrong_company_hits": absent_hits,
        "unlabelled_extra_hits": extra,
        "accuracy_pct": round(100 * correct / max(1, total), 1),
        "good_bad_news_caught_pct": round(100 * dir_right / max(1, directional), 1),
        "neutral_news_flagged_pct": round(100 * neutral_flagged / max(1, neutral), 1),
        "opposite_direction": flipped,
        "auto_trades_right": trades_right,
        "auto_trades_wrong": trades_wrong,
        "by_category": {k: round(100 * v[0] / v[1]) for k, v in sorted(by_cat.items())},
        "confusion": {f"{lab}->{p}": v for (lab, p), v in sorted(conf.items())},
        "mistakes": mistakes,
    }


ROWS = [
    ("companies_found_pct", "Companies found (%)"),
    ("wrong_company_hits", "Wrong companies picked up"),
    ("accuracy_pct", "Right answer (good/bad/neutral, %)"),
    ("good_bad_news_caught_pct", "Good/bad news called correctly (%)"),
    ("neutral_news_flagged_pct", "Neutral news wrongly flagged (%)"),
    ("opposite_direction", "Called the opposite direction"),
    ("auto_trades_right", "Would auto-trade: right"),
    ("auto_trades_wrong", "Would auto-trade: wrong"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="dev", help="dev, set2, set3, holdout, or a path to a .jsonl file")
    ap.add_argument("--sentiment", default="lexicon", choices=["lexicon", "finbert"])
    ap.add_argument("--compare", action="store_true", help="also run without the event rules (v0.2 behaviour)")
    ap.add_argument("--mistakes", type=int, default=0, help="print this many wrong answers")
    ap.add_argument("--json", default="", help="write the full results to this file")
    args = ap.parse_args()

    items = load_items(args.set)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:  # Windows: the db is still open
        table = make_table(Path(tmp))
        sentiment = SentimentService(Path(tmp) / "models" if args.sentiment == "lexicon" else
                                     ROOT / "data" / "models")
        sentiment.load(args.sentiment)
        if args.sentiment == "finbert" and sentiment.model_id == "lexicon-v1":
            print(f"FinBERT didn't load: {sentiment.error}")
            return 1
        runs = {"with event rules": evaluate(items, make_engine(table, sentiment, True), table)}
        if args.compare:
            runs = {"v0.2 (wording only)": evaluate(items, make_engine(table, sentiment, False), table), **runs}

    head = f"## Detection on the {args.set} set ({len(items)} headlines, {sentiment.describe()})\n"
    lines = [head, "| | " + " | ".join(runs) + " |", "|---|" + "---|" * len(runs)]
    for key, label in ROWS:
        lines.append(f"| {label} | " + " | ".join(str(r[key]) for r in runs.values()) + " |")
    print("\n".join(lines))
    last = list(runs.values())[-1]
    print("\nBy kind of news (% right): " + ", ".join(f"{k} {v}" for k, v in last["by_category"].items()))
    for m in last["mistakes"][: args.mistakes]:
        print(f"  {m['id']} {m['ticker']}: expected {m['label']}, got {m['pred']} ({m['conf']})"
              f"{'' if m['found'] else ' [company not found]'} - {m['title'][:110]}")
    if args.json:
        Path(args.json).write_text(json.dumps(runs, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
