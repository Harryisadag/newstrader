from __future__ import annotations

from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from types import SimpleNamespace

import httpx

from newstrader.config import SourceConfig
from newstrader.sources.alpaca_news import news_to_item
from newstrader.sources.base import strip_html
from newstrader.sources.rss import FeedPoller, parse_feed

SRC = SourceConfig(id="test-feed", type="rss", name="Test Feed", url="https://example.com/rss", poll_seconds=60)


def rss(items: list[tuple[str, str, datetime]]) -> bytes:
    body = "".join(
        f"<item><title>{t}</title><link>https://example.com/{g}</link><guid>{g}</guid>"
        f"<description><![CDATA[<p>About <b>{t}</b></p>]]></description>"
        f"<pubDate>{format_datetime(d)}</pubDate></item>" for t, g, d in items)
    return f'<?xml version="1.0"?><rss version="2.0"><channel><title>T</title>{body}</channel></rss>'.encode()


def test_strip_html():
    assert strip_html("<p>Hi &amp; <b>there</b></p><script>x()</script>") == "Hi & there"


def test_parse_feed():
    now = datetime.now(UTC)
    items = parse_feed(rss([("Nvidia beats", "g1", now)]), SRC)
    assert len(items) == 1
    it = items[0]
    assert it.title == "Nvidia beats" and it.body == "About Nvidia beats"
    assert it.external_id == "g1" and it.source_type == "rss"
    assert abs((it.published_at - now).total_seconds()) < 2


async def test_poller_first_poll_skips_old_then_sends_new():
    now = datetime.now(UTC)
    feeds = [
        rss([("old story", "g0", now - timedelta(hours=2)), ("fresh story", "g1", now - timedelta(minutes=1))]),
        rss([("old story", "g0", now - timedelta(hours=2)), ("fresh story", "g1", now),
             ("newest story", "g2", now)]),
    ]

    def handler(request):
        return httpx.Response(200, content=feeds.pop(0), headers={"ETag": "abc"})

    got, statuses = [], []

    async def submit(item):
        got.append(item.title)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        p = FeedPoller(SRC, submit, lambda lvl, d: statuses.append((lvl, d)), client)
        assert await p.poll_once() == 1
        assert got == ["fresh story"]
        assert await p.poll_once() == 1
        assert got == ["fresh story", "newest story"]
    assert statuses[-1][0] == "ok"


async def test_poller_reports_http_errors():
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(403))) as client:
        p = FeedPoller(SRC, None, lambda *a: None, client)
        try:
            await p.poll_once()
            raise AssertionError("should fail")
        except RuntimeError as exc:
            assert "403" in str(exc)


def test_alpaca_news_conversion():
    src = SourceConfig(id="alpaca-news", type="alpaca_news", name="Alpaca / Benzinga news")
    n = SimpleNamespace(id=123, headline="Tesla recalls 10,000 vehicles", summary="Recall details",
                        content="<p>Full <b>story</b></p>", symbols=["TSLA"], source="benzinga",
                        created_at=datetime(2026, 10, 7, 14, 0, tzinfo=UTC), url="https://b.com/1", author="Jane")
    item = news_to_item(n, src)
    assert item.external_id == "123" and item.symbols == ["TSLA"]
    assert "Full story" in item.body and item.title.startswith("Tesla")
    assert item.source_name.endswith("(benzinga)")
