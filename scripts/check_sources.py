"""Checks every preset news source still works: RSS feeds parse and have recent items, YouTube channels exist.

    python scripts/check_sources.py             # report only
    python scripts/check_sources.py --strict    # exit 1 if a source that is ON by default is broken

Run by the "check-sources" GitHub workflow (weekly and on demand). It writes a Markdown table to the job summary.
"""

from __future__ import annotations

import argparse
import asyncio
import calendar
import json
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import feedparser
import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from newstrader import __version__  # noqa: E402
from newstrader.sources.presets import default_sources  # noqa: E402

# the same User-Agent the app's feed reader sends (newstrader/sources/rss.py), without importing the whole app
USER_AGENT = f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) NewsTrader/{__version__} (personal news monitor)"

STALE_DAYS = 14
BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0 Safari/537.36"


def _newest(entries) -> datetime | None:
    best = None
    for e in entries:
        for key in ("published_parsed", "updated_parsed"):
            t = e.get(key)
            if t:
                try:
                    dt = datetime.fromtimestamp(calendar.timegm(t), UTC)
                except (OverflowError, ValueError, TypeError):
                    continue
                best = dt if best is None or dt > best else best
                break
    return best


async def check_rss(client: httpx.AsyncClient, src: dict) -> dict:
    r = await client.get(src["url"], headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=25)
    if r.status_code >= 400:
        return {"status": "broken", "detail": f"HTTP {r.status_code}"}
    parsed = feedparser.parse(r.content, response_headers={"content-type": r.headers.get("content-type", "")})
    n = len(parsed.entries)
    if not n:
        return {"status": "broken", "detail": f"no items ({'not a feed' if parsed.bozo else 'empty feed'})"}
    newest = _newest(parsed.entries)
    if newest is None:
        return {"status": "ok", "detail": f"{n} items (no dates)"}
    age_days = (datetime.now(UTC) - newest).total_seconds() / 86400
    if age_days > STALE_DAYS:
        return {"status": "stale", "detail": f"{n} items, newest {age_days:.0f} days old"}
    return {"status": "ok", "detail": f"{n} items, newest {age_days * 24:.1f} h old"}


async def check_stream(client: httpx.AsyncClient, src: dict) -> dict:
    channel = src["url"].removesuffix("/live")
    r = await client.get(channel, headers={"User-Agent": BROWSER_UA, "Accept-Language": "en-US,en;q=0.9"},
                         follow_redirects=True, timeout=25)
    if r.status_code == 404:
        return {"status": "broken", "detail": "channel not found (404)"}
    if r.status_code >= 400:
        return {"status": "unknown", "detail": f"HTTP {r.status_code}"}
    m = re.search(r'"channelId":"(UC[\w-]{22})"', r.text) or re.search(r'channel/(UC[\w-]{22})', r.text)
    if not m:
        return {"status": "unknown", "detail": "page has no channel id (consent or bot page?)"}
    live = ""
    try:
        lr = await client.get(src["url"], headers={"User-Agent": BROWSER_UA, "Accept-Language": "en-US,en;q=0.9"},
                              follow_redirects=True, timeout=25)
        live = " · live now" if '"isLiveNow":true' in lr.text or '"isLive":true' in lr.text else " · not live now"
    except httpx.HTTPError:
        pass
    return {"status": "ok", "detail": f"channel {m.group(1)}{live}"}


async def check(src: dict, client: httpx.AsyncClient, sem: asyncio.Semaphore) -> dict:
    async with sem:
        started = time.monotonic()
        try:
            if src["type"] == "stream":
                res = await check_stream(client, src)
            elif src["type"] in ("rss", "social_rss"):
                res = await check_rss(client, src)
            else:
                res = {"status": "skipped", "detail": "not a URL source"}
        except Exception as exc:
            res = {"status": "broken", "detail": f"{type(exc).__name__}: {str(exc)[:120]}"}
        res["seconds"] = round(time.monotonic() - started, 1)
        return {"id": src["id"], "name": src["name"], "type": src["type"], "on": src.get("enabled", False),
                "url": src["url"], **res}


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="fail if a source that is on by default is broken")
    ap.add_argument("--json", default="", help="also write the results to this JSON file")
    args = ap.parse_args()
    sources = default_sources()
    sem = asyncio.Semaphore(8)
    async with httpx.AsyncClient(http2=False) as client:
        results = await asyncio.gather(*(check(s, client, sem) for s in sources))
    icon = {"ok": "✅", "stale": "🟡", "unknown": "❔", "broken": "❌", "skipped": "➖"}
    lines = ["| | Source | Type | On by default | Result |", "|---|---|---|---|---|"]
    for r in sorted(results, key=lambda x: (x["status"] == "ok", x["type"], x["id"])):
        lines.append(f"| {icon.get(r['status'], '')} | {r['name']} (`{r['id']}`) | {r['type']} | "
                     f"{'yes' if r['on'] else ''} | {r['detail']} |")
    counts = {k: sum(1 for r in results if r["status"] == k) for k in icon}
    summary = "## News sources check\n\n" + ", ".join(f"{v} {k}" for k, v in counts.items() if v) + "\n\n" + "\n".join(lines)
    print(summary)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(summary + "\n")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2), encoding="utf-8")
    bad_default = [r for r in results if r["on"] and r["status"] == "broken"]
    if bad_default:
        print("\nBroken sources that are ON by default: " + ", ".join(r["id"] for r in bad_default))
    return 1 if args.strict and bad_default else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
