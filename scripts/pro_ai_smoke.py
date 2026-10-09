"""Pro AI end to end on this computer's processor. Run by .github/workflows/tests.yml (job "pro-ai-smoke").

Downloads the pinned llama.cpp build and the smallest catalog model with the app's own download code (size and SHA-256
checked), starts llama-server, asks ProAIEngine about labelled headlines from tests/fixtures/eval/dev.jsonl, checks
each answer is valid JSON that names only the story's candidate tickers, runs the "Test my PC" self-test and stops
the server. Exit code 1 on any failure.

    python scripts/pro_ai_smoke.py                       # data in NEWSTRADER_DATA_DIR (default: a temp folder)
    python scripts/pro_ai_smoke.py --model-file x.gguf   # skip the model download and use this file
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("NEWSTRADER_DATA_DIR", str(Path(tempfile.gettempdir()) / "newstrader-pro-ai-smoke"))

from newstrader import paths  # noqa: E402
from newstrader.ai.prefilter import prefilter  # noqa: E402
from newstrader.ai.tickers import TickerTable  # noqa: E402
from newstrader.ai.validator import validate_response  # noqa: E402
from newstrader.db import Database  # noqa: E402
from newstrader.llm import catalog, download, selftest  # noqa: E402
from newstrader.llm.engine import ProAIEngine, allowed_tickers, is_macro_only  # noqa: E402
from newstrader.llm.server import LlamaServer  # noqa: E402
from newstrader.sources.base import NewsItem  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "eval"
NOW = datetime(2026, 10, 7, 15, 0, tzinfo=UTC)
MAX_SIGNALS = 3


def progress(label: str):
    state = {"last": -1}

    def cb(done: int, total: int) -> None:
        pct = int(done * 100 / total) if total else 100
        if pct // 25 != state["last"] // 25 or done == total:
            state["last"] = pct
            print(f"  {label}: {pct}% of {total / 1e6:.0f} MB", flush=True)

    return cb


def ticker_table() -> TickerTable:
    table = TickerTable(Database(paths.data_dir() / "smoke.db"))
    table.replace_all(json.loads((FIXTURES / "eval_tickers.json").read_text(encoding="utf-8")))
    return table


def pick_items(table: TickerTable, n: int) -> list[tuple[dict, NewsItem, object]]:
    """The first n dev headlines about named companies (market-wide ones may name funds outside the table)."""
    out = []
    for line in (FIXTURES / "dev.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sample = json.loads(line)
        item = selftest.news_item(sample, NOW)
        pre = prefilter(item.text, table, item.symbols, allow_keyword_only=True, country_etfs=True)
        if pre.candidates and not is_macro_only(pre):
            out.append((sample, item, pre))
        if len(out) == n:
            break
    return out


async def ask_headlines(engine: ProAIEngine, table: TickerTable, items) -> list[str]:
    failures, right, secs = [], 0, []
    for sample, item, pre in items:
        started = time.monotonic()
        res = await engine.analyze(item, pre, True, NOW)
        secs.append(time.monotonic() - started)
        allowed = allowed_tickers(pre)
        if not res.ok:
            failures.append(f"{sample['id']}: {res.status}: {res.error}")
            print(f"  {sample['id']} FAILED ({secs[-1]:.1f}s): {res.error}")
            continue
        v = validate_response(res.text, table, MAX_SIGNALS)
        raw = json.loads(res.text)["signals"] if v.fatal is None else []
        outside = sorted({str(s.get("ticker")) for s in raw} - set(allowed))
        if v.fatal:
            failures.append(f"{sample['id']}: {v.fatal}")
        if outside:
            failures.append(f"{sample['id']}: tickers outside the candidates {allowed}: {outside}")
        ok, is_right, got = selftest.judge(sample, res.text, allowed)
        right += is_right
        print(f"  {sample['id']} {secs[-1]:5.1f}s  {'right' if is_right else 'wrong'}  got {got or '{}'}  "
              f"expected {sample['expect']}  validator: {v.status}{' - ' + v.summary if v.summary else ''}")
    if secs:
        print(f"Headlines: {right}/{len(items)} right, {sum(secs) / len(secs):.1f}s average, {max(secs):.1f}s slowest")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="linux-cpu" if sys.platform.startswith("linux") else "")
    ap.add_argument("--model", default=min(catalog.MODELS, key=lambda m: m.size).key)
    ap.add_argument("--model-file", default="", help="use this GGUF file instead of downloading --model")
    ap.add_argument("--items", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=180.0, help="seconds per answer")
    args = ap.parse_args()
    if args.build not in catalog.SERVER_BUILDS:
        print(f"Pick a server build with --build ({', '.join(catalog.SERVER_BUILDS)})")
        return 1
    print(f"Data folder: {paths.data_dir()}")

    started = time.monotonic()
    exe = download.install_server(args.build, progress_cb=progress("server"))
    print(f"Server: {exe} ({time.monotonic() - started:.0f}s)")
    if args.model_file:
        model_path, model_key = Path(args.model_file), Path(args.model_file).stem
    else:
        model = catalog.MODELS_BY_KEY[args.model]
        started = time.monotonic()
        model_path, model_key = download.fetch_model(model, progress_cb=progress("model")), model.key
        if download.sha256_file(model_path) != model.sha256:  # also re-check a copy restored from the CI cache
            print(f"{model_path.name} doesn't match its pinned SHA-256")
            return 1
        print(f"Model: {model_path} ({model.size / 1e6:.0f} MB, SHA-256 checked, {time.monotonic() - started:.0f}s)")

    failures: list[str] = []
    srv = LlamaServer()
    try:
        srv.start(model_path, exe, ctx=8192, slots=1, build_key=args.build)
        print("Server flags: " + " ".join(a for a in srv.args[1:] if a != srv.api_key and a != str(model_path)))
        if srv.dropped_flags:
            failures.append(f"this llama.cpp build doesn't know {srv.dropped_flags} - check server.build_args")
        if not srv.wait_ready(timeout=300):
            print("\n".join(list(srv.logs)[-40:]))
            print(f"FAILED: {srv.message}")
            return 1
        print(f"{srv.message} after {srv.ready_seconds}s (layers on the GPU: {srv.gpu_layers}/{srv.total_layers})")
        table = ticker_table()
        engine = ProAIEngine(srv, model_key, max_signals=MAX_SIGNALS, timeout_s=args.timeout, slots=1)
        items = pick_items(table, args.items)
        if len(items) < args.items:
            failures.append(f"only {len(items)} usable headlines in dev.jsonl")
        failures += asyncio.run(ask_headlines(engine, table, items))

        print("Self-test (Test my PC):")
        out = asyncio.run(selftest.run_selftest(engine, tickers=table))
        print(f"  {out['right']}/{out['total']} right, {out['json_ok']}/{out['total']} valid answers, "
              f"{out['avg_seconds']}s average, {out['max_seconds']}s slowest")
        if out["json_ok"] != out["total"]:
            bad = [f"{r['id']}: {r['error'] or 'invalid answer'}" for r in out["items"] if not r["json_ok"]]
            failures.append(f"self-test answers not valid: {bad}")
        if srv.state != "ready":
            failures.append(f"server ended in state {srv.state}: {srv.message}")
    finally:
        srv.stop()
    if srv.running:
        failures.append("server still running after stop()")
    if failures:
        print("\nFAILED:\n- " + "\n- ".join(failures))
        return 1
    print("\nPro AI smoke test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
