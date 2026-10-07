"""Optional: read an X (Twitter) account's posts through the official X API v2.

X's free API tier can't read posts, so this only works if you buy API access (Basic tier or higher) and put
the bearer token in Settings -> API keys as X_BEARER_TOKEN. Without a token the source just shows a warning.
Polling is kept slow (every 60s+) because paid tiers have small monthly read limits.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

import httpx

from ..config import SourceConfig
from .base import NewsItem

log = logging.getLogger(__name__)

API = "https://api.x.com/2"


class XAccountSource:
    def __init__(self, src: SourceConfig, submit, set_status, bearer_token: str, client: httpx.AsyncClient):
        self.src = src
        self.submit = submit
        self.set_status = set_status
        self.token = bearer_token
        self.client = client
        self.user_id: str | None = None
        self.since_id: str | None = None
        self.username = src.url.strip().lstrip("@").split("/")[-1]

    async def _get(self, path: str, params: dict | None = None) -> dict:
        r = await self.client.get(f"{API}{path}", params=params, timeout=20,
                                  headers={"Authorization": f"Bearer {self.token}"})
        if r.status_code == 429:
            raise RuntimeError("X API rate limit - checking less often")
        if r.status_code in (401, 403):
            raise RuntimeError(f"X API refused the token (HTTP {r.status_code}) - reading posts needs a paid tier")
        r.raise_for_status()
        return r.json()

    async def run(self) -> None:
        if not self.token:
            self.set_status("warn", "Needs X_BEARER_TOKEN (paid X API) in Settings -> API keys")
            while True:
                await asyncio.sleep(3600)
        interval = max(60, self.src.poll_seconds)
        failures = 0
        while True:
            try:
                if self.user_id is None:
                    data = await self._get(f"/users/by/username/{self.username}")
                    self.user_id = data["data"]["id"]
                params = {"max_results": 5, "tweet.fields": "created_at", "exclude": "retweets,replies"}
                if self.since_id:
                    params["since_id"] = self.since_id
                data = await self._get(f"/users/{self.user_id}/tweets", params)
                posts = list(reversed(data.get("data") or []))
                first = self.since_id is None
                for p in posts:
                    self.since_id = p["id"] if (self.since_id is None or int(p["id"]) > int(self.since_id)) else self.since_id
                    if first:
                        continue  # don't analyse old posts on startup
                    created = None
                    if p.get("created_at"):
                        created = datetime.fromisoformat(p["created_at"].replace("Z", "+00:00"))
                    await self.submit(NewsItem(source_id=self.src.id, source_type="x_account", source_name=self.src.name,
                                               external_id=p["id"], title=p.get("text", "")[:1000],
                                               url=f"https://x.com/{self.username}/status/{p['id']}",
                                               published_at=created, speaker=self.src.speaker or f"@{self.username}"))
                self.set_status("ok", f"@{self.username} checked")
                failures = 0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                failures += 1
                self.set_status("error", str(exc)[:200])
            await asyncio.sleep(min(interval * (2 ** min(failures, 4)), 3600))
