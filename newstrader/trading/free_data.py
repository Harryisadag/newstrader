"""Free market data with no account: prices from Yahoo Finance's public chart / screener endpoints and the list
of US-listed stocks from Nasdaq Trader. Used by the built-in paper-trading simulator (sim_broker.py).

Same return shapes as the Alpaca wrapper in broker.py, so the rest of the app doesn't care where prices come from.
Yahoo's endpoints are unofficial: they are rate-limited and may change, so every call is cached briefly, spaced
out, retried once on the other host, and failures return "no data" instead of crashing the app.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx

from . import nyse_calendar

log = logging.getLogger(__name__)

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0 Safari/537.36"
HOSTS = ("https://query1.finance.yahoo.com", "https://query2.finance.yahoo.com")
NASDAQ_LISTED = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
OTHER_EXCHANGES = {"N": "NYSE", "A": "AMEX", "P": "ARCA", "Z": "BATS", "V": "IEX"}
INTERVALS = {"1Min": ("1m", timedelta(days=7), timedelta(days=29)), "5Min": ("5m", timedelta(days=59),
             timedelta(days=59)), "1Hour": ("60m", timedelta(days=700), timedelta(days=700)),
             "1Day": ("1d", timedelta(days=36500), timedelta(days=36500))}  # (yahoo interval, per request, max back)
SPARK_CHUNK = 20  # symbols per spark request
MIN_GAP = 0.15  # seconds between requests (Yahoo throttles bursts)
WORKERS = 4


def yahoo_symbol(symbol: str) -> str:
    """'BRK.B' -> 'BRK-B' (Yahoo writes share classes with a dash)."""
    return symbol.upper().replace(".", "-").replace("/", "-")


def _iso(ts: float | int | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(int(ts), UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_chart(payload: dict) -> tuple[dict, list[dict]]:
    """Yahoo /v8/finance/chart JSON -> (meta, bars). Bars are {"t","o","h","l","c","v"} like Alpaca's; minutes
    with no trade (all None) are skipped."""
    result = ((payload or {}).get("chart") or {}).get("result") or []
    if not result:
        err = ((payload or {}).get("chart") or {}).get("error") or {}
        raise LookupError(err.get("description") or "no data")
    r = result[0]
    meta = r.get("meta") or {}
    quote = (((r.get("indicators") or {}).get("quote") or [{}])[0]) or {}
    o, h, lo, c, v = (quote.get(k) or [] for k in ("open", "high", "low", "close", "volume"))
    bars = []
    for i, ts in enumerate(r.get("timestamp") or []):
        close = c[i] if i < len(c) else None
        if close is None:
            continue
        bars.append({"t": _iso(ts), "o": _num(o, i, close), "h": _num(h, i, close), "l": _num(lo, i, close),
                     "c": round(float(close), 4), "v": float(v[i] or 0) if i < len(v) and v[i] is not None else 0.0})
    return meta, bars


def _num(values: list, i: int, default: float) -> float:
    x = values[i] if i < len(values) else None
    return round(float(default if x is None else x), 4)


def _screener_rows(payload: dict) -> list[dict]:
    result = ((payload or {}).get("finance") or {}).get("result") or []
    return (result[0].get("quotes") or []) if result else []


class FreeMarketData:
    """Market data without an account. All methods block (HTTP) - call them with asyncio.to_thread."""

    data_feed = "yahoo"

    def __init__(self, client: httpx.Client | None = None, min_gap: float = MIN_GAP):
        self.client = client or httpx.Client(headers={"User-Agent": UA, "Accept": "application/json,text/plain,*/*"},
                                             timeout=15, follow_redirects=True)
        self.min_gap = min_gap
        self._lock = threading.Lock()
        self._last = 0.0
        self._cache: dict[tuple, tuple[float, Any]] = {}
        self.last_error: str | None = None

    # ------------------------------------------------------------------ http
    def _wait(self) -> None:
        with self._lock:
            gap = time.monotonic() - self._last
            if gap < self.min_gap:
                time.sleep(self.min_gap - gap)
            self._last = time.monotonic()

    def _get_json(self, path: str, params: dict | None = None) -> Any:
        """GET a Yahoo endpoint; on a rate limit or server error, try the other host once."""
        last: Exception | None = None
        for host in HOSTS:
            self._wait()
            try:
                r = self.client.get(host + path, params=params)
            except httpx.HTTPError as exc:
                last = exc
                continue
            if r.status_code == 200:
                self.last_error = None
                return r.json()
            last = RuntimeError(f"Yahoo Finance answered HTTP {r.status_code}")
            if r.status_code not in (429, 500, 502, 503, 504):
                break
        self.last_error = str(last)
        raise RuntimeError(f"Free price data unavailable: {last}")

    def _cached(self, key: tuple, seconds: float, fetch):
        now = time.monotonic()
        hit = self._cache.get(key)
        if hit is not None and now - hit[0] < seconds:
            return hit[1]
        value = fetch()
        self._cache[key] = (now, value)
        if len(self._cache) > 2000:  # keep memory bounded
            for k in sorted(self._cache, key=lambda k: self._cache[k][0])[:500]:
                self._cache.pop(k, None)
        return value

    # ------------------------------------------------------------------ prices
    def chart(self, symbol: str, interval: str = "1m", range_: str | None = "1d", start: datetime | None = None,
              end: datetime | None = None, prepost: bool = True) -> tuple[dict, list[dict]]:
        params: dict[str, Any] = {"interval": interval, "includePrePost": "true" if prepost else "false"}
        if start is not None and end is not None:
            params["period1"], params["period2"] = int(start.timestamp()), int(end.timestamp())
        else:
            params["range"] = range_ or "1d"
        return parse_chart(self._get_json(f"/v8/finance/chart/{yahoo_symbol(symbol)}", params))

    def _today(self, symbol: str, max_age: float = 20.0) -> tuple[dict, list[dict]]:
        """Today's 1-minute bars (incl. pre/after-market) and the meta block, cached briefly."""
        return self._cached(("today", symbol.upper()), max_age, lambda: self.chart(symbol, "1m", "1d"))

    def latest_price(self, symbol: str) -> float | None:
        try:
            meta, bars = self._today(symbol)
        except Exception as exc:
            log.debug("price for %s unavailable: %s", symbol, exc)
            return None
        if bars:
            return bars[-1]["c"]
        price = meta.get("regularMarketPrice")
        return float(price) if price else None

    def bars(self, symbol: str, start: datetime, end: datetime, timeframe: str = "1Min") -> list[dict]:
        interval, per_request, max_back = INTERVALS[timeframe]
        now = datetime.now(UTC)
        start = max(start, now - max_back)
        if end <= start:
            return []
        out: list[dict] = []
        t = start
        while t < end:  # Yahoo serves at most `per_request` of 1-minute bars per call
            t2 = min(end, t + per_request)
            try:
                _meta, rows = self.chart(symbol, interval, start=t, end=t2, prepost=timeframe != "1Day")
            except LookupError:
                rows = []
            out.extend(rows)
            t = t2
        lo, hi = _utc_key(start), _utc_key(end)
        seen: set[str] = set()
        return [b for b in out if lo <= _ts_key(b["t"]) <= hi and not (b["t"] in seen or seen.add(b["t"]))]

    def bars_multi(self, symbols: list[str], start: datetime, end: datetime, timeframe: str = "1Min",
                   feed: str | None = None) -> dict[str, list[dict]]:
        recent = timeframe == "1Min" and datetime.now(UTC) - start <= timedelta(hours=6)

        def one(sym: str) -> tuple[str, list[dict]]:
            try:
                if recent:  # the market monitor asks every minute: reuse today's bars
                    _meta, rows = self._today(sym, max_age=45.0)
                    lo, hi = _utc_key(start), _utc_key(end)
                    return sym, [b for b in rows if lo <= _ts_key(b["t"]) <= hi]
                return sym, self.bars(sym, start, end, timeframe)
            except Exception as exc:
                log.debug("bars for %s unavailable: %s", sym, exc)
                return sym, []

        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            return {sym: rows for sym, rows in pool.map(one, dict.fromkeys(s.upper() for s in symbols)) if rows}

    def snapshots(self, symbols: list[str], feed: str | None = None) -> dict[str, dict]:
        """Latest price + day change for many symbols (Yahoo "spark": up to 20 symbols per request)."""
        wanted = list(dict.fromkeys(s.upper() for s in symbols if s))
        out: dict[str, dict] = {}
        for i in range(0, len(wanted), SPARK_CHUNK):
            chunk = wanted[i:i + SPARK_CHUNK]
            by_yahoo = {yahoo_symbol(s): s for s in chunk}
            try:
                data = self._cached(("spark", tuple(chunk)), 30.0, lambda c=chunk: self._get_json(
                    "/v8/finance/spark", {"symbols": ",".join(yahoo_symbol(s) for s in c), "range": "1d",
                                          "interval": "5m"}))
            except Exception as exc:
                log.debug("snapshots unavailable: %s", exc)
                continue
            for ysym, row in (data or {}).items():
                sym = by_yahoo.get(ysym, ysym)
                snap = _spark_snapshot(row)
                if snap is not None:
                    out[sym] = snap
        return out

    def _screener(self, scr_id: str, count: int) -> list[dict]:
        data = self._cached(("screener", scr_id, count), 120.0, lambda: self._get_json(
            "/v1/finance/screener/predefined/saved", {"scrIds": scr_id, "count": str(count)}))
        return _screener_rows(data)

    def movers(self, top: int = 20) -> dict:
        def row(q: dict) -> dict:
            return {"symbol": str(q.get("symbol") or "").replace("-", "."), "price": q.get("regularMarketPrice"),
                    "change": q.get("regularMarketChange"), "change_pct": q.get("regularMarketChangePercent")}

        gainers = [row(q) for q in self._screener("day_gainers", top) if q.get("quoteType") in (None, "EQUITY")]
        losers = [row(q) for q in self._screener("day_losers", top) if q.get("quoteType") in (None, "EQUITY")]
        return {"gainers": gainers, "losers": losers, "updated": _iso(time.time())}

    def most_actives(self, top: int = 20, by: str = "volume") -> dict:
        items = [{"symbol": str(q.get("symbol") or "").replace("-", "."), "volume": q.get("regularMarketVolume"),
                  "trade_count": None} for q in self._screener("most_actives", top)]
        return {"items": items, "updated": _iso(time.time())}

    def news(self, start: datetime, end: datetime, symbols: list[str] | None = None, limit: int = 200,
             include_content: bool = True, newest_first: bool = False) -> list[dict]:
        """Recent headlines about one stock (Yahoo search). Only the last few days - there's no history here."""
        if not symbols:
            return []
        data = self._get_json("/v1/finance/search", {"q": yahoo_symbol(symbols[0]), "newsCount": str(min(limit, 20)),
                                                     "quotesCount": "0"})
        out = []
        for n in (data or {}).get("news") or []:
            ts = n.get("providerPublishTime")
            when = datetime.fromtimestamp(int(ts), UTC) if ts else None
            if when is None or not (start <= when <= end):
                continue
            out.append({"id": n.get("uuid"), "headline": n.get("title"), "summary": "", "content": "",
                        "url": n.get("link"), "source": n.get("publisher"), "author": n.get("publisher"),
                        "symbols": [s.replace("-", ".") for s in n.get("relatedTickers") or []],
                        "created_at": when, "updated_at": when})
        out.sort(key=lambda n: n["created_at"], reverse=newest_first)
        return out[:limit]

    # ------------------------------------------------------------------ stock list
    def all_assets(self) -> list[dict]:
        """Every stock and ETF listed on a US exchange (Nasdaq Trader's daily symbol files), test issues left out."""
        listed = self._text(NASDAQ_LISTED)
        other = self._text(OTHER_LISTED)
        return parse_symbol_files(listed, other)

    def _text(self, url: str) -> str:
        self._wait()
        r = self.client.get(url)
        r.raise_for_status()
        return r.text

    # ------------------------------------------------------------------ calendar
    def clock(self) -> dict:
        return nyse_calendar.clock()

    def calendar(self, start: date, end: date) -> list[dict]:
        return nyse_calendar.calendar(start, end)


def _ts_key(t: str) -> str:
    return (t or "").replace("Z", "")[:19]


def _utc_key(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).isoformat()[:19]


def _spark_snapshot(row: dict) -> dict | None:
    """One symbol from Yahoo's spark endpoint -> the snapshot shape broker.snapshot_to_dict() returns."""
    if not isinstance(row, dict):
        return None
    closes = [c for c in row.get("close") or [] if c is not None]
    stamps = row.get("timestamp") or []
    price = row.get("fulldayPrice") or (closes[-1] if closes else None)
    if not price:
        return None
    prev = row.get("previousClose") or row.get("chartPreviousClose")
    change = (price - prev) / prev * 100 if prev else None
    last_t = _iso(stamps[-1]) if stamps else None
    return {"price": float(price), "prev_close": float(prev) if prev else None, "change_pct": change,
            "day_volume": None, "time": last_t, "minute_bar": None,
            "daily_bar": {"t": _iso(stamps[0]) if stamps else None, "o": closes[0] if closes else price,
                          "h": max(closes) if closes else price, "l": min(closes) if closes else price,
                          "c": float(price), "v": None},
            "prev_daily_bar": {"t": None, "o": prev, "h": prev, "l": prev, "c": prev, "v": None} if prev else None}


def parse_symbol_files(nasdaq_listed: str, other_listed: str) -> list[dict]:
    """Nasdaq Trader's nasdaqlisted.txt + otherlisted.txt -> asset dicts like broker.all_assets()."""
    out: dict[str, dict] = {}

    def rows(text: str):
        lines = [ln for ln in (text or "").splitlines() if ln.strip()]
        if not lines:
            return
        header = lines[0].split("|")
        for ln in lines[1:]:
            if ln.startswith("File Creation Time"):
                continue
            parts = ln.split("|")
            if len(parts) >= len(header) - 1:
                yield dict(zip(header, parts, strict=False))

    for r in rows(nasdaq_listed):
        sym = (r.get("Symbol") or "").strip()
        if sym and r.get("Test Issue") != "Y":
            out[sym] = _asset(sym, r.get("Security Name"), "NASDAQ", r.get("ETF") == "Y")
    for r in rows(other_listed):
        sym = (r.get("ACT Symbol") or "").strip()
        if sym and r.get("Test Issue") != "Y" and sym not in out:
            out[sym] = _asset(sym, r.get("Security Name"), OTHER_EXCHANGES.get(r.get("Exchange", ""), "NYSE"),
                              r.get("ETF") == "Y")
    return list(out.values())


def _asset(symbol: str, name: str | None, exchange: str, etf: bool) -> dict:
    name = (name or symbol).strip()
    # "Apple Inc. - Common Stock" (Nasdaq's style) -> "Apple Inc. Common Stock" (like Alpaca's names)
    name = name.replace(" - ", " ")
    return {"symbol": symbol, "name": name, "exchange": exchange, "tradable": True, "shortable": True,
            "easy_to_borrow": True, "fractionable": False, "etf": etf}
