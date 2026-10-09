"""Measures how well the local engine (or Pro AI) reads news: which companies it finds, and good / bad / neutral
for each.

    python scripts/eval_detection.py                         # dev set, built-in word list (works offline)
    python scripts/eval_detection.py --sentiment finbert      # downloads FinBERT (~110 MB) the first time
    python scripts/eval_detection.py --set set2 --compare     # also runs the v0.2 behaviour (no event rules)
    python scripts/eval_detection.py --engine pro --pro-url http://127.0.0.1:8080 [--pro-key KEY]
                                                              # Pro AI on a llama-server that is already running
    python scripts/eval_detection.py --engine pro --pro-model model.gguf --pro-server path/to/llama-server
                                                              # starts its own llama-server and stops it after
    (with --engine pro, --compare also runs the local engine for a side-by-side table)

The headline sets are described in tests/fixtures/eval/README.md. Labels say what a trader would expect the stock
to do in the next hour (bullish / bearish / neutral); each item was labelled twice, blind, and disagreements settled.
The holdout set is sealed: only its overall accuracy and wrong-direction count are shown unless --unseal is passed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from newstrader.ai.prefilter import prefilter  # noqa: E402
from newstrader.ai.tickers import TickerTable  # noqa: E402
from newstrader.config import AppSettings  # noqa: E402
from newstrader.db import Database  # noqa: E402
from newstrader.llm.engine import ProAIEngine  # noqa: E402
from newstrader.llm.server import LlamaServer  # noqa: E402
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


def _ctx(settings: AppSettings) -> SimpleNamespace:
    return SimpleNamespace(config=SimpleNamespace(settings=settings), state=SimpleNamespace(mode="paper"))


def make_pro_engine(server, model_name: str, timeout: float) -> ProAIEngine:
    return ProAIEngine(server, model_name, ctx=_ctx(AppSettings()), timeout_s=timeout, slots=1)


@contextmanager
def pro_engine(args, tmp: Path):
    """Pro AI for --engine pro: a llama-server that is already running (--pro-url), or one started here for
    --pro-model with --pro-server (stopped again afterwards)."""
    if args.pro_url:
        server = SimpleNamespace(url=args.pro_url.rstrip("/"), api_key=args.pro_key, state="ready", message="")
        yield make_pro_engine(server, "running server", args.pro_timeout)
        return
    srv = LlamaServer(pid_path=tmp / "llama-server.pid")
    try:
        srv.start(Path(args.pro_model), Path(args.pro_server), ctx=8192, slots=1)
        if not srv.wait_ready(timeout=600):
            raise RuntimeError(srv.message)
        print(f"llama-server: {srv.message}", file=sys.stderr)
        yield make_pro_engine(srv, Path(args.pro_model).stem, args.pro_timeout)
    finally:
        srv.stop()


def _norm(sym: str) -> str:
    return SAME_COMPANY.get(sym, sym)


def read_item(engine, item: NewsItem, pre) -> tuple[list[dict], bool]:
    """(the engine's signals for one story, whether the engine failed to answer)."""
    if not isinstance(engine, ProAIEngine):
        return engine._analyze_sync(item, pre, NOW)["signals"], False
    res = asyncio.run(engine.analyze(item, pre, True, NOW))
    try:
        signals = json.loads(res.text)["signals"] if res.ok else None
    except (ValueError, KeyError, TypeError):
        signals = None
    if not isinstance(signals, list):
        return [], True
    return [s for s in signals if isinstance(s, dict) and s.get("ticker") and isinstance(s.get("confidence"), int)
            and s.get("direction") in ("bullish", "bearish", "neutral")], False


def evaluate(items: list[dict], engine, table: TickerTable, buy_threshold: int = 80) -> dict:
    found = missed = absent_hits = extra = 0
    correct = total = 0
    conf = Counter()  # (label, predicted)
    by_cat: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    trades_right = trades_wrong = 0
    mistakes = []
    pro = isinstance(engine, ProAIEngine)
    failed, seconds = 0, []
    for it in items:
        item = NewsItem(source_id="eval", source_type=it.get("source_type") or "rss", source_name="eval",
                        external_id=it["id"], title=it["title"], body=it.get("body") or "", url=it.get("url") or "",
                        published_at=NOW, symbols=list(it.get("symbols") or []),
                        kind="transcript" if it.get("source_type") == "stream" else "text")
        if item.kind == "transcript":
            item.body = "New:\n[10:00:00] " + (it["title"] + " " + (it.get("body") or "")).strip()
        # Pro AI also reads market-wide news with no company named (it may answer with index / sector / country funds)
        pre = prefilter(item.text, table, item.symbols, allow_keyword_only=pro,
                        country_etfs=engine.ctx.config.settings.ml.country_etfs)
        started = time.monotonic()
        raw, error = read_item(engine, item, pre)
        seconds.append(time.monotonic() - started)
        failed += error
        got = {_norm(c.symbol) for c in pre.candidates}
        signals = {_norm(s["ticker"]): s for s in raw}
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
        **({"answer_errors": failed, "avg_seconds": round(sum(seconds) / max(1, len(seconds)), 2)} if pro else {}),
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


PRO_ROWS = [("answer_errors", "Pro AI answers that failed"), ("avg_seconds", "Seconds per story")]

SEALED_ROWS = [("accuracy_pct", "Right answer (good/bad/neutral, %)"),
               ("opposite_direction", "Called the opposite direction")]


def is_sealed(which: str, unseal: bool = False) -> bool:
    """The holdout set (by name or as holdout*.jsonl) only shows its overall numbers unless --unseal is passed."""
    return not unseal and Path(which).stem.lower().startswith("holdout")


def report(runs: dict, which: str, n_items: int, describe: str, mistakes: int = 0, sealed: bool = False) -> str:
    head = f"## Detection on the {which} set ({n_items} headlines, {describe})\n"
    lines = [head, "| | " + " | ".join(runs) + " |", "|---|" + "---|" * len(runs)]
    rows = ROWS + [r for r in PRO_ROWS if any(r[0] in run for run in runs.values())]
    for key, label in SEALED_ROWS if sealed else rows:
        lines.append(f"| {label} | " + " | ".join(str(r.get(key, "-")) for r in runs.values()) + " |")
    if sealed:
        lines.append("\nSealed set: mistakes and the per-category table stay hidden (use --unseal to show them).")
        return "\n".join(lines)
    last = list(runs.values())[-1]
    lines.append("\nBy kind of news (% right): " + ", ".join(f"{k} {v}" for k, v in last["by_category"].items()))
    for m in last["mistakes"][:mistakes]:
        lines.append(f"  {m['id']} {m['ticker']}: expected {m['label']}, got {m['pred']} ({m['conf']})"
                     f"{'' if m['found'] else ' [company not found]'} - {m['title'][:110]}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="dev", help="dev, set2, set3, holdout, or a path to a .jsonl file")
    ap.add_argument("--sentiment", default="lexicon", choices=["lexicon", "finbert"])
    ap.add_argument("--compare", action="store_true", help="also run without the event rules (v0.2 behaviour)")
    ap.add_argument("--mistakes", type=int, default=0, help="print this many wrong answers")
    ap.add_argument("--json", default="", help="write the full results to this file")
    ap.add_argument("--unseal", action="store_true", help="show the holdout set's mistakes and categories too")
    ap.add_argument("--engine", default="local", choices=["local", "pro"], help="pro = Pro AI (a llama-server)")
    ap.add_argument("--pro-url", default="", help="Pro AI: a llama-server that is already running")
    ap.add_argument("--pro-key", default="", help="Pro AI: that server's API key (if it has one)")
    ap.add_argument("--pro-model", default="", help="Pro AI: a .gguf model file to start a llama-server with")
    ap.add_argument("--pro-server", default="", help="Pro AI: the llama-server program for --pro-model")
    ap.add_argument("--pro-timeout", type=float, default=60.0, help="Pro AI: seconds allowed per answer")
    args = ap.parse_args(argv)
    sealed = is_sealed(args.set, args.unseal)
    pro = args.engine == "pro"
    if pro and not args.pro_url and not (args.pro_model and args.pro_server):
        print("--engine pro needs --pro-url (a running llama-server) or --pro-model with --pro-server")
        return 2

    items = load_items(args.set)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:  # Windows: the db is still open
        table = make_table(Path(tmp))
        runs, describe = {}, "Pro AI"
        if not pro or args.compare:
            sentiment = SentimentService(Path(tmp) / "models" if args.sentiment == "lexicon" else
                                         ROOT / "data" / "models")
            sentiment.load(args.sentiment)
            if args.sentiment == "finbert" and sentiment.model_id == "lexicon-v1":
                print(f"FinBERT didn't load: {sentiment.error}")
                return 1
            describe = sentiment.describe() + (" vs Pro AI" if pro else "")
            runs = {"with event rules": evaluate(items, make_engine(table, sentiment, True), table)}
            if args.compare and not pro:
                runs = {"v0.2 (wording only)": evaluate(items, make_engine(table, sentiment, False), table), **runs}
        if pro:
            with pro_engine(args, Path(tmp)) as engine:
                runs[f"Pro AI ({engine.model_key})"] = evaluate(items, engine, table)

    print(report(runs, args.set, len(items), describe, args.mistakes, sealed))
    if args.json:
        if sealed:
            runs = {name: {k: r[k] for k, _label in SEALED_ROWS} for name, r in runs.items()}
        Path(args.json).write_text(json.dumps(runs, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
