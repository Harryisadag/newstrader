"""Polite, quick feed checks: wait as long as a website asks after "too many requests", back off after errors, and
spread the checks out with a little randomness so dozens of feeds never fire at the same moment.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

MAX_RETRY_AFTER = 3600  # a website asking us to wait longer than an hour is checked again after an hour
JITTER = 0.15  # each wait is up to 15% longer, never shorter (so a feed is never checked faster than its setting)


class SlowDown(RuntimeError):
    """The website answered 429 (too many requests) or 503 (busy)."""

    def __init__(self, status: int, wait: float | None = None):
        super().__init__(f"HTTP {status}")
        self.status = status
        self.wait = wait


def retry_after_seconds(headers, now: datetime | None = None) -> float | None:
    """The Retry-After header (seconds, or an HTTP date) as seconds from now. None if missing or unreadable."""
    value = str(headers.get("Retry-After") or "").strip()
    if not value:
        return None
    if value.isdigit():
        return float(value)
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - (now or datetime.now(UTC))).total_seconds())


def next_wait(interval: float, failures: int = 0, retry_after: float | None = None, cap: float = 900) -> float:
    """Seconds until the next check: the interval, doubled for each failure in a row (up to 16x, at most `cap`),
    never sooner than the website's Retry-After, plus the jitter."""
    wait = min(interval * (2 ** min(failures, 4)), max(cap, interval)) if failures else interval
    if retry_after is not None:
        wait = max(wait, min(retry_after, MAX_RETRY_AFTER))
    return wait * random.uniform(1.0, 1.0 + JITTER)
