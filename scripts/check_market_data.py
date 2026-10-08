"""Checks that the free market-data sources the built-in paper-trading simulator uses still answer, and shows a
sample of what they return. Run by .github/workflows/check-market-data.yml (weekly and on demand).

    python scripts/check_market_data.py            # prints a Markdown report
    python scripts/check_market_data.py --raw DIR  # also saves the raw responses (for test fixtures)

Standalone on purpose (only needs httpx): it shouldn't depend on the app importing cleanly.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0 Safari/537.36")
CHECKS = [
    ("yahoo_chart_1m", "https://query1.finance.yahoo.com/v8/finance/chart/AAPL",
     {"range": "1d", "interval": "1m", "includePrePost": "true"}),
    ("yahoo_chart_5d_1m", "https://query2.finance.yahoo.com/v8/finance/chart/SPY",
     {"range": "5d", "interval": "1m", "includePrePost": "false"}),
    ("yahoo_chart_daily", "https://query1.finance.yahoo.com/v8/finance/chart/MSFT", {"range": "1mo", "interval": "1d"}),
    ("yahoo_spark", "https://query1.finance.yahoo.com/v8/finance/spark",
     {"symbols": "AAPL,MSFT,NVDA,SPY", "range": "1d", "interval": "5m"}),
    ("yahoo_quote_nocrumb", "https://query1.finance.yahoo.com/v7/finance/quote", {"symbols": "AAPL,MSFT"}),
    ("yahoo_screener_gainers", "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved",
     {"scrIds": "day_gainers", "count": "10"}),
    ("yahoo_screener_actives", "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved",
     {"scrIds": "most_actives", "count": "10"}),
    ("yahoo_search_news", "https://query1.finance.yahoo.com/v1/finance/search",
     {"q": "NVDA", "newsCount": "5", "quotesCount": "0"}),
    ("nasdaq_listed", "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt", {}),
    ("nasdaq_other", "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt", {}),
    ("sec_tickers", "https://www.sec.gov/files/company_tickers_exchange.json", {}),
    ("stooq_quote", "https://stooq.com/q/l/", {"s": "aapl.us,msft.us", "f": "sd2t2ohlcv", "h": "", "e": "csv"}),
]


def shape(obj, depth: int = 0, max_depth: int = 5):
    """A short outline of a JSON value (keys and types, first list item only)."""
    if depth >= max_depth:
        return "..."
    if isinstance(obj, dict):
        return {k: shape(v, depth + 1, max_depth) for k, v in list(obj.items())[:40]}
    if isinstance(obj, list):
        return [shape(obj[0], depth + 1, max_depth), f"... {len(obj)} items"] if obj else []
    if isinstance(obj, str):
        return obj[:60]
    return obj


def trim(obj, keep: int = 3):
    """The full structure with every list cut to its first few items (for test fixtures)."""
    if isinstance(obj, dict):
        return {k: trim(v, keep) for k, v in obj.items()}
    if isinstance(obj, list):
        return [trim(v, keep) for v in obj[:keep]]
    return obj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="", help="save each raw response in this folder")
    args = ap.parse_args()
    raw = Path(args.raw) if args.raw else None
    if raw:
        raw.mkdir(parents=True, exist_ok=True)
    lines = ["## Free market data sources", "", "| Source | HTTP | Time | Size |", "|---|---|---|---|"]
    details = []
    failed = 0
    with httpx.Client(headers={"User-Agent": UA, "Accept": "*/*"}, timeout=20, follow_redirects=True) as client:
        for name, url, params in CHECKS:
            headers = {"User-Agent": "NewsTrader market-data check admin@example.com"} if "sec.gov" in url else {}
            t0 = time.monotonic()
            try:
                r = client.get(url, params=params, headers=headers)
                status, body = r.status_code, r.content
            except Exception as exc:  # network error
                status, body = f"error: {type(exc).__name__}", str(exc).encode()
            ms = (time.monotonic() - t0) * 1000
            ok = status == 200
            failed += not ok
            lines.append(f"| {name} | {status} | {ms:.0f} ms | {len(body):,} B |")
            if raw:
                (raw / f"{name}.txt").write_bytes(body)
            text = body.decode("utf-8", "replace")
            try:
                parsed = json.loads(text)
                outline = json.dumps(trim(parsed), separators=(",", ":"))[:6000]
            except Exception:
                outline = text[:1500]
            details += [f"### {name}", "", f"`{url}` {params}", "", "```", outline, "```", ""]
    print("\n".join(lines + [""] + details))
    return 1 if failed == len(CHECKS) else 0


if __name__ == "__main__":
    sys.exit(main())
