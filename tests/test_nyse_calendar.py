"""The built-in NYSE calendar, checked against the exchange's published holiday and early-close lists."""

from datetime import UTC, date, datetime

from newstrader.trading import nyse_calendar as cal

PUBLISHED = {
    2024: ["2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29", "2024-05-27", "2024-06-19", "2024-07-04",
           "2024-09-02", "2024-11-28", "2024-12-25"],
    2025: ["2025-01-01", "2025-01-09", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26", "2025-06-19",
           "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25"],
    2026: ["2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19", "2026-07-03",
           "2026-09-07", "2026-11-26", "2026-12-25"],
    2027: ["2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31", "2027-06-18", "2027-07-05",
           "2027-09-06", "2027-11-25", "2027-12-24"],
}
EARLY = {
    2024: ["2024-07-03", "2024-11-29", "2024-12-24"],
    2025: ["2025-07-03", "2025-11-28", "2025-12-24"],
    2026: ["2026-11-27", "2026-12-24"],
    2027: ["2027-11-26"],
}


def test_holidays_match_the_published_lists():
    for year, days in PUBLISHED.items():
        assert sorted(d.isoformat() for d in cal.holidays(year)) == days, year
    # New Year's Day on a Saturday isn't moved back into the old year; a Sunday Juneteenth / Christmas moves to Monday
    assert date(2021, 12, 31) not in cal.holidays(2021) and date(2022, 1, 1) not in cal.holidays(2022)
    assert date(2022, 6, 20) in cal.holidays(2022) and date(2022, 12, 26) in cal.holidays(2022)


def test_early_closes():
    for year, days in EARLY.items():
        assert sorted(d.isoformat() for d in cal.early_closes(year)) == days, year
    open_, close = cal.session(date(2025, 11, 28))
    assert close.hour == 13 and open_.hour == 9 and open_.minute == 30


def test_clock_open_closed_and_next_session():
    # Wednesday 8 Oct 2026, 15:00 UTC = 11:00 New York: open
    c = cal.clock(datetime(2026, 10, 8, 15, 0, tzinfo=UTC))
    assert c["is_open"] and c["next_close"] == "2026-10-08T20:00:00Z" and c["next_open"] == "2026-10-09T13:30:00Z"
    # Friday evening -> opens Monday
    c = cal.clock(datetime(2026, 10, 9, 22, 0, tzinfo=UTC))
    assert not c["is_open"] and c["next_open"] == "2026-10-12T13:30:00Z"
    # Before the open on Thanksgiving (closed) -> the Friday half day
    c = cal.clock(datetime(2026, 11, 26, 12, 0, tzinfo=UTC))
    assert not c["is_open"] and c["next_open"] == "2026-11-27T14:30:00Z" and c["next_close"] == "2026-11-27T18:00:00Z"
    # winter time: 9:30 New York = 14:30 UTC
    assert cal.calendar(date(2026, 12, 7), date(2026, 12, 7)) == [
        {"date": "2026-12-07", "open": "2026-12-07T14:30:00Z", "close": "2026-12-07T21:00:00Z"}]


def test_extended_hours():
    assert cal.extended_hours(datetime(2026, 10, 8, 12, 0, tzinfo=UTC))       # 8:00 New York
    assert not cal.extended_hours(datetime(2026, 10, 8, 15, 0, tzinfo=UTC))   # regular session
    assert cal.extended_hours(datetime(2026, 10, 8, 21, 0, tzinfo=UTC))       # 17:00
    assert not cal.extended_hours(datetime(2026, 10, 10, 15, 0, tzinfo=UTC))  # Saturday
