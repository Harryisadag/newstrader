"""Real-time Benzinga news through Alpaca's news websocket (included with Alpaca accounts)."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import UTC, datetime

from ..config import SourceConfig
from .base import NewsItem, strip_html

log = logging.getLogger(__name__)


def news_to_item(n, src: SourceConfig) -> NewsItem:
    def g(key, default=None):
        if isinstance(n, dict):
            return n.get(key, default)
        return getattr(n, key, default)

    created = g("created_at")
    if isinstance(created, str):
        try:
            created = datetime.fromisoformat(created.replace("Z", "+00:00"))
        except ValueError:
            created = None
    if isinstance(created, datetime) and created.tzinfo is None:
        created = created.replace(tzinfo=UTC)
    summary = strip_html(g("summary") or "", 2000)
    content = strip_html(g("content") or "", 5000)
    body = summary if not content else (summary + "\n" + content if summary and summary not in content else content)
    author = g("author") or ""
    origin = g("source") or "benzinga"
    return NewsItem(source_id=src.id, source_type="alpaca_news", source_name=f"{src.name} ({origin})",
                    external_id=str(g("id")), title=strip_html(g("headline") or "", 1000), body=body,
                    url=g("url") or "", published_at=created,
                    symbols=[s.upper() for s in (g("symbols") or []) if isinstance(s, str)], speaker=author,
                    language="en")


class AlpacaNewsSource:
    def __init__(self, src: SourceConfig, submit, set_status, api_key: str, secret_key: str):
        self.src = src
        self.submit = submit
        self.set_status = set_status
        self.api_key = api_key
        self.secret_key = secret_key
        self.stream = None
        self.count = 0

    async def run(self) -> None:
        from alpaca.data.live import NewsDataStream

        self.set_status("starting", "connecting to Alpaca news websocket...")
        self.stream = NewsDataStream(self.api_key, self.secret_key)

        async def handler(news) -> None:
            try:
                item = news_to_item(news, self.src)
                if not item.title:
                    return
                self.count += 1
                self.set_status("ok", f"connected · {self.count} stories received")
                await self.submit(item)
            except Exception:
                log.exception("Alpaca news handler failed")

        self.stream.subscribe_news(handler, "*")
        self.set_status("ok", "connected · waiting for news")
        try:
            await self.stream._run_forever()  # reconnects by itself
        except asyncio.CancelledError:
            with contextlib.suppress(Exception):
                await self.stream.stop_ws()
            raise
        except Exception as exc:
            self.set_status("error", f"news websocket stopped: {exc}")
            raise
