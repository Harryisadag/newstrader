"""Same story from several sources = one analysis (saves Claude calls).

Used for text items (headlines, posts). Live-transcript windows skip this step and rely on the
signal-level merge instead (same ticker + direction within a window = one signal, traded once),
which lives in pipeline.py.
"""

from __future__ import annotations

import re
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta

_STOP = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "at", "by", "with", "from", "as", "is", "are",
    "was", "were", "be", "been", "it", "its", "this", "that", "after", "over", "into", "amid", "says", "said",
    "say", "report", "reports", "reported", "news", "breaking", "update", "live", "new", "will", "would",
    "could", "may", "has", "have", "had", "about", "than", "more", "up", "down", "out", "but", "not", "no",
    "his", "her", "their", "they", "he", "she", "we", "you", "i", "our", "us", "what", "why", "how", "who",
    "video", "watch", "exclusive", "here", "s", "inc", "corp",
}
_WORD = re.compile(r"[a-z0-9$%&.]+")


def _article_url(url: str | None) -> str:
    """Only real article links count for URL matching (not a bare homepage shared by every item)."""
    if not url:
        return ""
    from urllib.parse import urlsplit

    parts = urlsplit(url.strip())
    if len(parts.path.strip("/")) < 8 and not parts.query:
        return ""
    return f"{parts.netloc.lower()}{parts.path.rstrip('/')}?{parts.query}"


def fingerprint(text: str) -> frozenset[str]:
    words = set()
    for w in _WORD.findall((text or "").lower()):
        w = w.strip(".")
        if not w or w in _STOP:
            continue
        if len(w) > 4 and w.endswith("s"):
            w = w[:-1]
        words.add(w)
    return frozenset(words)


def similarity(a: frozenset[str], b: frozenset[str]) -> float:
    """Jaccard similarity of two headline fingerprints (0 = nothing shared, 1 = same words).

    Deliberately conservative: a false "duplicate" would skip real news, while a missed duplicate
    only costs one extra Claude call (and the signal merge still prevents a double trade).
    """
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass
class _Seen:
    at: datetime
    fp: frozenset[str]
    item_id: int
    url: str
    source_id: str


class StoryDeduper:
    def __init__(self) -> None:
        self._recent: deque[_Seen] = deque()
        self._lock = threading.Lock()

    def check_and_add(self, *, item_id: int, text: str, url: str, source_id: str, at: datetime,
                      window_minutes: int, threshold: float) -> int | None:
        """Returns the id of an earlier item this one duplicates, else records it and returns None."""
        fp = fingerprint(text)
        url = _article_url(url)
        with self._lock:
            cutoff = at - timedelta(minutes=window_minutes)
            while self._recent and self._recent[0].at < cutoff:
                self._recent.popleft()
            if window_minutes > 0:
                for seen in reversed(self._recent):
                    if url and seen.url and url == seen.url:
                        return seen.item_id
                    if similarity(fp, seen.fp) >= threshold:
                        return seen.item_id
            self._recent.append(_Seen(at, fp, item_id, url or "", source_id))
            return None
