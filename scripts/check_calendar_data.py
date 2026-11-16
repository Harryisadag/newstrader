"""Checks the free data sources for the calendar features (earnings dates, Fed / inflation / jobs days), the SEC
filings watcher and phone alerts, and prints trimmed samples of what they return (for parsers and test fixtures).
Run by .github/workflows/check-calendar-data.yml (weekly and on demand).

    python scripts/check_calendar_data.py            # Markdown report with samples
    python scripts/check_calendar_data.py --raw DIR  # also saves each raw response

Standalone on purpose (only needs httpx). Read-only: public data, nothing is posted anywhere.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import httpx

BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/129.0 Safari/537.36")
# sec.gov refuses tools that give no contact address ("Undeclared Automated Tool"). This check uses a placeholder;
# the app itself asks the user for their own contact in Settings and sends it only to sec.gov.
SEC_UA = "NewsTrader data check admin@example.com"


def trim(obj, keep: int = 4):
    if isinstance(obj, dict):
        return {k: trim(v, keep) for k, v in obj.items()}
    if isinstance(obj, list):
        return [trim(v, keep) for v in obj[:keep]]
    return obj


def weekday(d: date) -> date:
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def digests(c: httpx.Client, nasdaq: dict, sec: dict, today: date) -> list[str]:
    """Samples that need a little digging: US economic events on coming days, the Fed's FOMC entries, and one
    insider-trade (Form 4) document."""
    out: list[str] = []
    # 1. US economic events for the next 12 weekdays (names only), to learn what CPI / jobs / Fed days look like
    d, lines = today, []
    for _ in range(12):
        d = weekday(d + timedelta(days=1))
        try:
            rows = c.get("https://api.nasdaq.com/api/calendar/economicevents", params={"date": d.isoformat()},
                         headers=nasdaq).json().get("data", {}).get("rows") or []
        except Exception as exc:
            lines.append(f"{d}: error {exc}")
            continue
        us = [f"{r.get('gmt')} {r.get('eventName')} (consensus {str(r.get('consensus')).strip() or '-'})"
              for r in rows if r.get("country") == "United States"]
        lines.append(f"{d}: {len(rows)} events, US: " + "; ".join(us[:40]))
        time.sleep(0.5)
    out += ["### digest_us_economic_events", "", "```", *lines, "```", ""]
    # 2. the Fed's calendar: which types exist, and every FOMC-looking entry
    try:
        text = c.get("https://www.federalreserve.gov/json/calendar.json",
                     headers={"User-Agent": BROWSER_UA}).content.decode("utf-8-sig")
        events = json.loads(text).get("events", [])
        types: dict[str, int] = {}
        for e in events:
            types[e.get("type", "?")] = types.get(e.get("type", "?"), 0) + 1
        fomc = [e for e in events if "FOMC" in (e.get("title", "") + e.get("type", "") + e.get("description", ""))]
        newest = sorted(fomc, key=lambda e: (e.get("month", ""), e.get("days", "")), reverse=True)[:14]
        out += ["### digest_fed_calendar", "", "```", f"{len(events)} events; types: {json.dumps(types)}",
                *[json.dumps(e, separators=(",", ":"))[:420] for e in newest], "```", ""]
    except Exception as exc:
        out += ["### digest_fed_calendar", "", f"error: {exc}", ""]
    # 3. one recent Form 4: the filing index, then its XML
    try:
        import re

        feed = c.get("https://www.sec.gov/cgi-bin/browse-edgar",
                     params={"action": "getcurrent", "type": "4", "company": "", "dateb": "", "owner": "only",
                             "count": "10", "output": "atom"}, headers=sec).text
        link = re.search(r'<link[^>]+href="(https://www\.sec\.gov/Archives/edgar/data/[^"]+-index\.htm)"', feed)
        out += ["### digest_form4", "", "```", feed[feed.find("<entry>"):][:1800]]
        if link:
            time.sleep(0.5)
            index = c.get(link.group(1), headers=sec).text
            xml = re.search(r'href="(/Archives/edgar/data/[^"]+\.xml)"', index)
            out.append(f"index page: {link.group(1)} -> xml: {xml.group(1) if xml else None}")
            if xml:
                time.sleep(0.5)
                path = re.sub(r"/xslF345X\d+/", "/", xml.group(1))  # the raw XML, not the styled view
                doc = c.get("https://www.sec.gov" + path, headers=sec).text
                out.append(f"raw xml: {path}")
                out.append(doc[:4500])
        out += ["```", ""]
    except Exception as exc:
        out += ["### digest_form4", "", f"error: {exc}", ""]
    # 4. one 8-K entry in full (items are in the summary)
    try:
        feed = c.get("https://www.sec.gov/cgi-bin/browse-edgar",
                     params={"action": "getcurrent", "type": "8-K", "company": "", "dateb": "", "owner": "include",
                             "count": "10", "output": "atom"}, headers=sec).text
        out += ["### digest_8k_entries", "", "```", feed[:600], "...", feed[feed.find("<entry>"):][:2600], "```", ""]
    except Exception as exc:
        out += ["### digest_8k_entries", "", f"error: {exc}", ""]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="")
    args = ap.parse_args()
    raw = Path(args.raw) if args.raw else None
    if raw:
        raw.mkdir(parents=True, exist_ok=True)
    today = date.today()
    day = weekday(today + timedelta(days=1)).isoformat()
    year = today.year
    nasdaq = {"User-Agent": BROWSER_UA, "Accept": "application/json, text/plain, */*",
              "Origin": "https://www.nasdaq.com", "Referer": "https://www.nasdaq.com/"}
    sec = {"User-Agent": SEC_UA, "Accept-Encoding": "gzip, deflate"}
    checks = [
        # earnings dates
        ("nasdaq_earnings", "https://api.nasdaq.com/api/calendar/earnings", {"date": day}, nasdaq,
         "Nasdaq: companies reporting earnings on a day (time of day, EPS forecast)"),
        ("nasdaq_earnings_today", "https://api.nasdaq.com/api/calendar/earnings", {"date": weekday(today).isoformat()},
         nasdaq, "Nasdaq: earnings today"),
        ("nasdaq_earnings_date_aapl", "https://api.nasdaq.com/api/analyst/AAPL/earnings-date", {}, nasdaq,
         "Nasdaq: one company's next earnings date"),
        ("yahoo_earnings_chart_meta", "https://query1.finance.yahoo.com/v8/finance/chart/AAPL",
         {"range": "1d", "interval": "1d", "events": "earnings"}, {"User-Agent": BROWSER_UA},
         "Yahoo chart: earnings events"),
        # macro days
        ("nasdaq_economic_events", "https://api.nasdaq.com/api/calendar/economicevents", {"date": day}, nasdaq,
         "Nasdaq: economic events on a day (CPI, jobs, Fed...)"),
        ("fed_calendar_json", "https://www.federalreserve.gov/json/calendar.json", {}, {"User-Agent": BROWSER_UA},
         "Federal Reserve: its own calendar (FOMC meetings, speeches)"),
        ("fed_fomc_page", "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm", {},
         {"User-Agent": BROWSER_UA}, "Federal Reserve: FOMC meeting calendar page (HTML)"),
        ("bls_schedule_ics", "https://www.bls.gov/schedule/news_release/bls.ics", {}, {"User-Agent": BROWSER_UA},
         "BLS: release schedule as a calendar file (CPI, jobs report, PPI)"),
        ("bls_schedule_cpi", "https://www.bls.gov/schedule/news_release/cpi.htm", {}, {"User-Agent": BROWSER_UA},
         "BLS: CPI release dates (HTML)"),
        ("bea_schedule", "https://www.bea.gov/news/schedule", {}, {"User-Agent": BROWSER_UA},
         "BEA: GDP / PCE release schedule (HTML)"),
        # SEC filings
        ("sec_current_8k_atom", "https://www.sec.gov/cgi-bin/browse-edgar",
         {"action": "getcurrent", "type": "8-K", "company": "", "dateb": "", "owner": "include", "count": "40",
          "output": "atom"}, sec, "SEC EDGAR: latest 8-K filings (Atom feed)"),
        ("sec_current_form4_atom", "https://www.sec.gov/cgi-bin/browse-edgar",
         {"action": "getcurrent", "type": "4", "company": "", "dateb": "", "owner": "only", "count": "40",
          "output": "atom"}, sec, "SEC EDGAR: latest Form 4 (insider trades) filings (Atom feed)"),
        ("sec_current_13d_atom", "https://www.sec.gov/cgi-bin/browse-edgar",
         {"action": "getcurrent", "type": "SC 13D", "company": "", "dateb": "", "owner": "include", "count": "40",
          "output": "atom"}, sec, "SEC EDGAR: latest Schedule 13D filings (Atom feed)"),
        ("sec_company_tickers", "https://www.sec.gov/files/company_tickers.json", {}, sec,
         "SEC: ticker -> CIK list"),
        ("sec_submissions_aapl", "https://data.sec.gov/submissions/CIK0000320193.json", {}, sec,
         "SEC: one company's recent filings (form, items, dates)"),
        ("sec_fulltext_search", "https://efts.sec.gov/LatestSearch/rest/search-index",
         {"q": "", "forms": "8-K", "dateRange": "custom", "startdt": (today - timedelta(days=1)).isoformat(),
          "enddt": today.isoformat()}, sec, "SEC full-text search: recent 8-Ks (items in the hits)"),
        ("sec_filing_index_json", "https://www.sec.gov/cgi-bin/browse-edgar",
         {"action": "getcompany", "CIK": "0000320193", "type": "4", "dateb": "", "owner": "only", "count": "10",
          "output": "atom"}, sec, "SEC EDGAR: one company's Form 4 filings (Atom)"),
        ("sec_company_tickers_exchange", "https://www.sec.gov/files/company_tickers_exchange.json", {}, sec,
         "SEC: ticker -> CIK -> exchange list"),
        # more calendar samples
        ("nasdaq_earnings_plus7", "https://api.nasdaq.com/api/calendar/earnings",
         {"date": weekday(today + timedelta(days=7)).isoformat()}, nasdaq, "Nasdaq: earnings a week ahead"),
        ("nasdaq_earnings_weekend", "https://api.nasdaq.com/api/calendar/earnings",
         {"date": (today + timedelta(days=(5 - today.weekday()) % 7 or 7)).isoformat()}, nasdaq,
         "Nasdaq: earnings on a Saturday (what an empty day looks like)"),
        ("nasdaq_dividends", "https://api.nasdaq.com/api/calendar/dividends", {"date": day}, nasdaq,
         "Nasdaq: ex-dividend dates"),
        ("nasdaq_splits", "https://api.nasdaq.com/api/calendar/splits", {"date": day}, nasdaq, "Nasdaq: stock splits"),
        # phone alerts
        ("ntfy_health", "https://ntfy.sh/v1/health", {}, {"User-Agent": SEC_UA}, "ntfy.sh: service health"),
        # market holidays / S&P for the report card
        ("yahoo_spy_daily", "https://query1.finance.yahoo.com/v8/finance/chart/SPY",
         {"range": "1mo", "interval": "1d"}, {"User-Agent": BROWSER_UA}, "Yahoo: SPY daily bars"),
        (f"fed_calendar_year_{year + 1}", "https://www.federalreserve.gov/newsevents/calendar.htm", {},
         {"User-Agent": BROWSER_UA}, "Federal Reserve: events calendar page (HTML)"),
    ]
    rows = ["## Free calendar / filings data sources", "", f"Sample day: {day}", "",
            "| Check | HTTP | Time | Size | Type |", "|---|---|---|---|---|"]
    details: list[str] = []
    failed = 0
    with httpx.Client(timeout=25, follow_redirects=True) as c:
        for name, url, params, headers, note in checks:
            t0 = time.monotonic()
            try:
                r = c.get(url, params=params, headers=headers)
                status, body, ctype = r.status_code, r.content, r.headers.get("content-type", "")
            except Exception as exc:
                status, body, ctype = f"error: {type(exc).__name__}", str(exc).encode(), ""
            ms = (time.monotonic() - t0) * 1000
            failed += status != 200
            rows.append(f"| {name} | {status} | {ms:.0f} ms | {len(body):,} B | {ctype.split(';')[0]} |")
            if raw:
                (raw / f"{name}.txt").write_bytes(body)
            text = body.decode("utf-8", "replace")
            try:
                outline = json.dumps(trim(json.loads(text)), separators=(",", ":"))[:6000]
            except Exception:
                # HTML / XML / ICS: the part most likely to hold the data
                marks = [text.find(m) for m in ("<entry>", "BEGIN:VEVENT", "fomc-meeting", "release-list", "<table")]
                start = min((m for m in marks if m >= 0), default=0)
                outline = text[start:start + 3500]
            details += [f"### {name}", "", f"{note} - `{url}` {params}", "", "```", outline, "```", ""]
            time.sleep(0.4)  # SEC asks for at most 10 requests a second; be gentle everywhere
        details += digests(c, nasdaq, sec, today)
    print("\n".join(rows + [""] + details))
    return 1 if failed == len(checks) else 0


if __name__ == "__main__":
    sys.exit(main())
