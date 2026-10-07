"""Discord webhook alerts (rich embeds). Waits and retries once if Discord says we're sending too fast."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime

import httpx

log = logging.getLogger(__name__)

COLORS = {"trade": 0xA855F7, "success": 0x22C55E, "info": 0x4F8CFF, "warn": 0xF5A524, "error": 0xEF4444}


def build_payload(title: str, message: str, level: str, mode: str = "paper", fields: dict | None = None) -> dict:
    embed = {
        "title": title[:256],
        "description": (message or "")[:4000],
        "color": COLORS.get(level, COLORS["info"]),
        "timestamp": datetime.now(UTC).isoformat(),
        "footer": {"text": f"NewsTrader · {'LIVE - REAL MONEY' if mode == 'live' else 'paper trading'}"},
    }
    if fields:
        embed["fields"] = [{"name": str(k)[:256], "value": str(v)[:1024], "inline": True}
                           for k, v in list(fields.items())[:10]]
    return {"username": "NewsTrader", "embeds": [embed], "allowed_mentions": {"parse": []}}


async def send_discord(client: httpx.AsyncClient, webhook_url: str, payload: dict) -> tuple[bool, str]:
    for attempt in range(2):
        try:
            r = await client.post(webhook_url, json=payload, timeout=10)
        except httpx.HTTPError as exc:
            return False, f"network error: {exc}"
        if r.status_code in (200, 204):
            return True, "sent"
        if r.status_code == 429 and attempt == 0:
            try:
                wait = float(r.json().get("retry_after", 1))
            except Exception:
                wait = 1.0
            await asyncio.sleep(min(wait, 10))
            continue
        if r.status_code in (401, 404):
            return False, "webhook URL is invalid or was deleted - copy it again from Discord"
        return False, f"HTTP {r.status_code}: {r.text[:200]}"
    return False, "rate limited by Discord"
