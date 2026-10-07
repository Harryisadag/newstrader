"""Common shape for everything that comes in from a news source."""

from __future__ import annotations

import hashlib
import html
import re
from dataclasses import dataclass, field
from datetime import datetime

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def strip_html(text: str | None, limit: int = 6000) -> str:
    if not text:
        return ""
    text = re.sub(r"(?is)<(script|style).*?</\1>", " ", text)
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = _WS_RE.sub(" ", text).strip()
    return text[:limit]


def safe_url(url: str | None) -> str:
    """Only plain web links are kept (a feed could otherwise send javascript: or file: links)."""
    url = (url or "").strip()
    return url if url.lower().startswith(("http://", "https://")) else ""


def stable_id(*parts: str) -> str:
    return hashlib.sha1("|".join(p or "" for p in parts).encode("utf-8", "ignore")).hexdigest()[:20]


@dataclass
class NewsItem:
    source_id: str
    source_type: str  # rss | social_rss | x_account | alpaca_news | stream | manual
    source_name: str
    external_id: str  # unique within the source
    title: str
    body: str = ""
    url: str = ""
    published_at: datetime | None = None
    symbols: list[str] = field(default_factory=list)  # tickers already tagged by the source
    speaker: str = ""
    kind: str = "text"  # text | transcript
    db_id: int | None = None

    @property
    def text(self) -> str:
        if self.body and self.body.strip() != self.title.strip():
            return f"{self.title}\n{self.body}"
        return self.title
