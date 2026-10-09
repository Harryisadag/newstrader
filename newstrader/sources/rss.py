"""RSS / Atom feed poller (also used for social accounts exposed as RSS, e.g. Truth Social).

Every check asks "anything new since last time?" (ETag / If-Modified-Since), so an unchanged feed answers with a tiny
"304 Not Modified". A website that answers 429 / 503 is left alone for as long as it asks (Retry-After), and errors
back off. See polling.py.
"""

from __future__ import annotations

import asyncio
import calendar
import logging
import random
from datetime import UTC, datetime, timedelta
from urllib.parse import urlsplit

import feedparser
import httpx

from .. import __version__
from ..config import SourceConfig
from .base import NewsItem, stable_id, strip_html
from .lang import detect_language
from .polling import SlowDown, next_wait, retry_after_seconds

log = logging.getLogger(__name__)

USER_AGENT = f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) NewsTrader/{__version__} (personal news monitor)"
# sec.gov refuses automated readers that don't say who they are and where to find out more
SEC_USER_AGENT = f"NewsTrader/{__version__} personal news monitor (+https://github.com/Harryisadag/newstrader)"
# On the very first poll, only items newer than this are analysed (the rest are just marked as seen).
FIRST_POLL_MAX_AGE = timedelta(minutes=30)


def _entry_time(entry) -> datetime | None:
    for key in ("published_parsed", "updated_parsed", "created_parsed"):
        t = entry.get(key)
        if t:
            try:
                return datetime.fromtimestamp(calendar.timegm(t), UTC)
            except (OverflowError, ValueError, TypeError):
                continue
    return None


def user_agent_for(url: str) -> str:
    host = (urlsplit(url).hostname or "").lower()
    return SEC_USER_AGENT if host == "sec.gov" or host.endswith(".sec.gov") else USER_AGENT


def parse_feed(content: bytes, src: SourceConfig, content_type: str = "") -> list[NewsItem]:
    # the HTTP Content-Type can carry the character set (e.g. Shift_JIS) - without it some feeds decode wrongly
    parsed = feedparser.parse(content, response_headers={"content-type": content_type} if content_type else None)
    items: list[NewsItem] = []
    for e in parsed.entries[:100]:
        title = strip_html(e.get("title"), 1000)
        summary = strip_html(e.get("summary") or e.get("description"), 6000)
        if not title and summary:
            title, summary = summary[:200], summary
        if not title:
            continue
        link = e.get("link") or ""
        ext = e.get("id") or e.get("guid") or link or stable_id(title, summary[:200])
        kind_type = src.type if src.type in ("rss", "social_rss") else "rss"
        items.append(NewsItem(source_id=src.id, source_type=kind_type, source_name=src.name, external_id=str(ext)[:300],
                              title=title, body=summary, url=link, published_at=_entry_time(e),
                              speaker=src.speaker,
                              language=src.language if src.language not in ("", "auto")
                              else detect_language(f"{title} {summary[:500]}")))
    return items


class FeedPoller:
    """Polls one feed forever; hands new items to the pipeline."""

    def __init__(self, src: SourceConfig, submit, set_status, client: httpx.AsyncClient):
        self.src = src
        self.submit = submit
        self.set_status = set_status
        self.client = client
        self.etag: str | None = None
        self.modified: str | None = None
        self.seen: set[str] = set()
        self.first_poll = True
        self.new_count = 0
        self.failures = 0

    async def run(self) -> None:
        self.set_status("starting", "first check...")
        # spread the first checks out, so dozens of feeds don't all hit the network at the same moment
        await asyncio.sleep(random.uniform(0, min(10.0, self.src.poll_seconds / 6)))
        while True:
            wait = None
            try:
                await self.poll_once()
                self.failures = 0
            except asyncio.CancelledError:
                raise
            except SlowDown as exc:
                self.failures += 1
                wait = next_wait(self.src.poll_seconds, self.failures, exc.wait)
                self.set_status("warn", f"the website is busy or asked to be checked less often (HTTP {exc.status}) "
                                        f"- next check in {wait:.0f} s")
            except Exception as exc:
                self.failures += 1
                self.set_status("error", f"{type(exc).__name__}: {str(exc)[:160]}")
                log.debug("feed %s failed: %s", self.src.name, exc)
            # back off on repeated failures (max 15 min)
            await asyncio.sleep(wait if wait is not None else next_wait(self.src.poll_seconds, self.failures))

    async def poll_once(self) -> int:
        headers = {"User-Agent": user_agent_for(self.src.url),
                   "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml, */*"}
        if self.etag:
            headers["If-None-Match"] = self.etag
        if self.modified:
            headers["If-Modified-Since"] = self.modified
        r = await self.client.get(self.src.url, headers=headers, follow_redirects=True, timeout=20)
        if r.status_code == 304:
            self.set_status("ok", f"no new items · {self.new_count} total")
            return 0
        if r.status_code in (429, 503):
            raise SlowDown(r.status_code, retry_after_seconds(r.headers))
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}")
        self.etag = r.headers.get("ETag")
        self.modified = r.headers.get("Last-Modified")
        items = await asyncio.to_thread(parse_feed, r.content, self.src, r.headers.get("content-type", ""))
        if not items and b"<" not in r.content[:200]:
            raise RuntimeError("response is not a feed")
        now = datetime.now(UTC)
        fresh = 0
        for item in reversed(items):  # oldest first
            if item.external_id in self.seen:
                continue
            self.seen.add(item.external_id)
            if self.first_poll and (item.published_at is None or now - item.published_at > FIRST_POLL_MAX_AGE):
                continue
            fresh += 1
            await self.submit(item)
        self.first_poll = False
        if len(self.seen) > 5000:
            self.seen = set(list(self.seen)[-2000:])
        self.new_count += fresh
        self.set_status("ok", f"{len(items)} items, {fresh} new · {self.new_count} total")
        return fresh
