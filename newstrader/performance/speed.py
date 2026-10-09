"""The speed timer: how long each step from a piece of news to the order took.

The timestamps used (all stored in UTC):
- published (signals.news_published_at): when the news came out. For text news it is the time the website or feed
  put on the story - its own clock, often rounded to the minute. For live TV it is when the newest words in the
  transcribed clip were spoken (the end of the clip window), on this computer's clock; YouTube's own live delay
  can't be seen, so it isn't counted.
- received (news_received_at): when the story reached the app - a feed check found it, the news websocket sent it,
  or a TV clip was turned into text and the short pause after a company was mentioned had passed.
- decided (decided_at): when the AI engine finished reading the story.
- submitted (orders.submitted_at): when the broker took the order (the broker's own clock).

Clocks on different computers disagree a little, so a step that comes out slightly negative (up to 5 s) counts as
0 s. A bigger negative gap means a wrong clock or time zone somewhere, and that step is left out (None).
"""

from __future__ import annotations

from datetime import UTC, datetime
from statistics import median

SKEW_SECONDS = 5.0


def to_utc(value) -> datetime | None:
    """A datetime or ISO-8601 string as an aware UTC datetime. Times without a zone are taken as UTC."""
    if value is None or value == "":
        return None
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def gap_seconds(later, earlier) -> float | None:
    """Seconds from `earlier` to `later`; None if either is missing or the clocks clearly disagree."""
    a, b = to_utc(later), to_utc(earlier)
    if a is None or b is None:
        return None
    s = (a - b).total_seconds()
    if s < 0:
        return 0.0 if s >= -SKEW_SECONDS else None
    return round(s, 2)


def breakdown(published, received, decided, submitted=None) -> dict:
    """feed delay (how late the news was when it arrived), thinking time, order time and news -> order."""
    return {
        "feed_delay_s": gap_seconds(received, published),
        "thinking_s": gap_seconds(decided, received),
        "order_s": gap_seconds(submitted, decided),
        "news_to_order_s": gap_seconds(submitted, received),
    }


def signal_speed(signal: dict, submitted=None) -> dict:
    """The breakdown for a signals row (and its order's submitted_at, if it was traded)."""
    out = breakdown(signal.get("news_published_at"), signal.get("news_received_at"),
                    signal.get("decided_at") or signal.get("created_at"), submitted)
    out["manual"] = signal.get("review_status") == "approved"  # the order waited for you to approve it
    return out


def _median(values: list[float]) -> float | None:
    return round(median(values), 2) if values else None


def speed_summary(traded: list[dict], feed_delays: list[tuple[str, float]], min_count: int = 10,
                  slowest: int = 3) -> dict:
    """For the Performance tab.

    traded: signals rows (with order_submitted_at) that placed an order. Orders you approved by hand are left out
    of the timings, because they waited for you.
    feed_delays: (source name, seconds between published and received) for every headline.
    """
    auto = [signal_speed(r, r.get("order_submitted_at")) for r in traded if r.get("review_status") != "approved"]

    def med(key: str) -> float | None:
        return _median([s[key] for s in auto if s[key] is not None])

    by_source: dict[str, list[float]] = {}
    for name, secs in feed_delays:
        if secs is None or secs < -SKEW_SECONDS:
            continue
        by_source.setdefault(name or "?", []).append(max(0.0, secs))
    sources = [{"source": name, "count": len(v), "median_feed_delay_s": _median(v), "few": len(v) < min_count}
               for name, v in by_source.items()]
    sources.sort(key=lambda s: -s["median_feed_delay_s"])
    return {
        "orders": sum(1 for s in auto if s["news_to_order_s"] is not None),
        "median_news_to_order_s": med("news_to_order_s"),
        "median_thinking_s": med("thinking_s"),
        "median_order_s": med("order_s"),
        "median_feed_delay_s": _median([d for v in by_source.values() for d in v]),
        "by_source": sources,
        "slowest": [s for s in sources if not s["few"]][:slowest],
    }
