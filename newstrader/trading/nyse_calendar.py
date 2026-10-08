"""The New York Stock Exchange's trading calendar, worked out in code (no account or download needed).

Regular session 9:30-16:00 New York time; 13:00 close on the early-close days. Holidays follow NYSE rule 7.2:
a Saturday holiday is taken on the Friday before (except New Year's Day), a Sunday holiday on the Monday after.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
OPEN, CLOSE, EARLY_CLOSE = time(9, 30), time(16, 0), time(13, 0)
PRE_OPEN, AFTER_CLOSE = time(4, 0), time(20, 0)  # extended hours

# Closures that no rule predicts (national days of mourning, emergencies)
SPECIAL_CLOSURES: dict[date, str] = {
    date(2012, 10, 29): "Hurricane Sandy", date(2012, 10, 30): "Hurricane Sandy",
    date(2018, 12, 5): "President George H.W. Bush mourning", date(2025, 1, 9): "President Jimmy Carter mourning",
}


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """n-th (1-based) weekday (Mon=0) of a month; n=-1 is the last one."""
    if n > 0:
        d = date(year, month, 1)
        d += timedelta(days=(weekday - d.weekday()) % 7)
        return d + timedelta(weeks=n - 1)
    d = date(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def _easter(year: int) -> date:
    """Western Easter Sunday (anonymous Gregorian algorithm)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    month = (h + l_ - 7 * m + 114) // 31
    day = (h + l_ - 7 * m + 114) % 31 + 1
    return date(year, month, day)


def _observed(d: date, new_years: bool = False) -> date | None:
    if d.weekday() == 5:  # Saturday -> Friday before (New Year's Day isn't moved into the old year)
        return None if new_years else d - timedelta(days=1)
    if d.weekday() == 6:  # Sunday -> Monday after
        return d + timedelta(days=1)
    return d


@lru_cache(maxsize=64)
def holidays(year: int) -> dict[date, str]:
    days: dict[date, str] = {}

    def add(d: date | None, name: str) -> None:
        if d is not None and d.year == year:
            days[d] = name

    add(_observed(date(year, 1, 1), new_years=True), "New Year's Day")
    add(_nth_weekday(year, 1, 0, 3), "Martin Luther King Jr. Day")
    add(_nth_weekday(year, 2, 0, 3), "Washington's Birthday")
    add(_easter(year) - timedelta(days=2), "Good Friday")
    add(_nth_weekday(year, 5, 0, -1), "Memorial Day")
    if year >= 2022:
        add(_observed(date(year, 6, 19)), "Juneteenth")
    add(_observed(date(year, 7, 4)), "Independence Day")
    add(_nth_weekday(year, 9, 0, 1), "Labor Day")
    add(_nth_weekday(year, 11, 3, 4), "Thanksgiving Day")
    add(_observed(date(year, 12, 25)), "Christmas Day")
    for d, name in SPECIAL_CLOSURES.items():
        add(d, name)
    return days


@lru_cache(maxsize=64)
def early_closes(year: int) -> set[date]:
    """13:00 closes: July 3 (when July 4 is Tue-Fri), the day after Thanksgiving, Christmas Eve (Mon-Thu)."""
    out = {_nth_weekday(year, 11, 3, 4) + timedelta(days=1)}
    july3 = date(year, 7, 3)
    if july3.weekday() <= 3:
        out.add(july3)
    eve = date(year, 12, 24)
    if eve.weekday() <= 3:
        out.add(eve)
    return {d for d in out if d not in holidays(year)}


def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d not in holidays(d.year)


def session(d: date) -> tuple[datetime, datetime] | None:
    """(open, close) of the regular session on that New York date, or None if the market is closed all day."""
    if not is_trading_day(d):
        return None
    close = EARLY_CLOSE if d in early_closes(d.year) else CLOSE
    return datetime.combine(d, OPEN, ET), datetime.combine(d, close, ET)


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def next_session(after: datetime) -> tuple[datetime, datetime]:
    """The first session whose close is after `after` (today's if it's still open or hasn't opened yet)."""
    d = after.astimezone(ET).date()
    for _ in range(15):
        s = session(d)
        if s is not None and s[1] > after:
            return s
        d += timedelta(days=1)
    raise RuntimeError("no trading session in the next two weeks")  # can't happen with real calendars


def clock(now: datetime | None = None) -> dict:
    """Same shape as the broker's clock(): is_open, next_open, next_close, timestamp (UTC ISO strings)."""
    now = now or datetime.now(UTC)
    open_, close = next_session(now)
    is_open = open_ <= now < close
    if is_open:
        nxt_open = next_session(close + timedelta(seconds=1))[0]
    else:
        nxt_open = open_
    return {"is_open": is_open, "next_open": _iso(nxt_open), "next_close": _iso(close), "timestamp": _iso(now)}


def calendar(start: date, end: date) -> list[dict]:
    """Sessions between two dates, like the broker's calendar(): [{"date", "open", "close"}]."""
    out, d = [], start
    while d <= end:
        s = session(d)
        if s is not None:
            out.append({"date": d.isoformat(), "open": _iso(s[0]), "close": _iso(s[1])})
        d += timedelta(days=1)
    return out


def extended_hours(now: datetime | None = None) -> bool:
    """Pre-market (4:00-9:30) or after-hours (16:00-20:00) on a trading day."""
    now = (now or datetime.now(UTC)).astimezone(ET)
    s = session(now.date())
    if s is None:
        return False
    t = now.time()
    return PRE_OPEN <= t < s[0].time() or s[1].time() <= t < AFTER_CLOSE
