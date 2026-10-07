"""Helpers for "what was the price at time T" and "what is T + 1 trading day", from Alpaca minute bars."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from ..db import parse_iso

ET = ZoneInfo("America/New_York")


def price_at(bars: list[dict], when: datetime) -> float | None:
    """Close of the last 1-minute bar that started at or before `when` (bars sorted by time)."""
    best = None
    for b in bars:
        t = parse_iso(b["t"])
        if t is None:
            continue
        if t <= when:
            best = b
        else:
            break
    return float(best["c"]) if best and best.get("c") is not None else None


def first_bar_at_or_after(bars: list[dict], when: datetime) -> dict | None:
    for b in bars:
        t = parse_iso(b["t"])
        if t is not None and t >= when:
            return b
    return None


def next_session_same_time(t0: datetime, sessions: list[dict]) -> datetime | None:
    """Same clock time (New York) on the next trading day, clamped to that day's open..close.

    sessions: [{"date": "YYYY-MM-DD", "open": iso, "close": iso}, ...] from the Alpaca calendar.
    """
    local = t0.astimezone(ET)
    for s in sessions:
        d = date.fromisoformat(s["date"])
        if d <= local.date():
            continue
        target = datetime.combine(d, time(local.hour, local.minute, local.second), ET)
        open_t = parse_iso(s["open"]) if "T" in s["open"] else datetime.combine(d, time.fromisoformat(s["open"]), ET)
        close_t = parse_iso(s["close"]) if "T" in s["close"] else datetime.combine(d, time.fromisoformat(s["close"]), ET)
        if open_t and target < open_t:
            target = open_t
        if close_t and target > close_t:
            target = close_t
        return target
    return None


def fallback_next_day(t0: datetime) -> datetime:
    """When no calendar is available: next weekday, same time."""
    t = t0 + timedelta(days=1)
    while t.astimezone(ET).weekday() >= 5:
        t += timedelta(days=1)
    return t


def directional_return(direction: str, p0: float | None, p1: float | None) -> float | None:
    """% move in the direction the signal predicted (positive = the AI was right)."""
    if not p0 or p1 is None:
        return None
    raw = (p1 - p0) / p0 * 100
    return raw if direction == "bullish" else -raw if direction == "bearish" else None
