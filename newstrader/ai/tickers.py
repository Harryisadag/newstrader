"""The lookup table of every US-listed stock (symbol + cleaned company name), from Alpaca's asset list.

Cached in SQLite and refreshed once a day, so the pre-filter works instantly at startup.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from datetime import timedelta

from ..db import Database, iso, parse_iso, utcnow
from .keywords import ALIASES

log = logging.getLogger(__name__)

# Exchanges we treat as "US-listed" (OTC / pink sheets are excluded)
LISTED_EXCHANGES = {"NYSE", "NASDAQ", "ARCA", "AMEX", "BATS", "NYSEARCA", "NYSEAMERICAN", "CBOE"}

_SUFFIXES = [
    r"american depositary shares?", r"american depository shares?", r"\bads\b", r"\badr\b",
    r"common stock", r"ordinary shares?", r"capital stock", r"common shares?", r"class [a-c]\b",
    r"series [a-z]\b", r"\bunits?\b", r"\bwarrants?\b", r"\brights?\b", r"\bpreferred\b",
    r"each representing.*$", r"\(the\)", r"new york registry shares?", r"subordinate voting shares?",
    r"\binc\b\.?", r"incorporated", r"\bcorp\b\.?", r"corporation", r"\bco\b\.?", r"company",
    r"\bltd\b\.?", r"limited", r"\bplc\b", r"\bn\.?v\.?\b", r"\bs\.?a\.?\b", r"\bag\b", r"\bse\b",
    r"\blp\b", r"\bl\.?p\.?\b", r"\bllc\b", r"holdings?", r"\bgroup\b", r"\bthe\b",
]
_SUFFIX_RE = re.compile("|".join(_SUFFIXES), re.IGNORECASE)
_PUNCT_RE = re.compile(r"[^a-z0-9&'\- ]+")
_SPACES = re.compile(r"\s+")


def clean_company_name(name: str) -> str:
    """'Apple Inc. Common Stock' -> 'apple'; 'Alphabet Inc. Class C Capital Stock' -> 'alphabet'."""
    s = (name or "").lower()
    s = s.replace(",", " ").replace("/", " ")
    for _ in range(2):
        s = _SUFFIX_RE.sub(" ", s)
    s = _PUNCT_RE.sub(" ", s)
    s = _SPACES.sub(" ", s).strip(" -'&")
    return s


def tokenize(text: str) -> list[str]:
    """Lower-case words; possessives dropped ("Apple's" -> "apple")."""
    s = _PUNCT_RE.sub(" ", text.lower().replace("’", "'"))
    out = []
    for t in s.split():
        if t.endswith("'s"):
            t = t[:-2]
        t = t.strip("-'")
        if t:
            out.append(t)
    return out


@dataclass
class TickerInfo:
    symbol: str
    name: str
    clean_name: str
    exchange: str
    tradable: bool
    shortable: bool
    easy_to_borrow: bool


class TickerTable:
    def __init__(self, db: Database):
        self.db = db
        self._lock = threading.RLock()
        self.by_symbol: dict[str, TickerInfo] = {}
        self.by_name: dict[tuple[str, ...], list[str]] = {}
        self.aliases: dict[tuple[str, ...], str] = {}
        self.max_name_len = 1
        self.updated_at: str | None = None

    # ---------------------------------------------------------------- loading
    @property
    def loaded(self) -> bool:
        return bool(self.by_symbol)

    def load_from_db(self) -> int:
        rows = self.db.query("SELECT * FROM tickers")
        self._build(rows)
        if rows:
            self.updated_at = max(r["updated_at"] or "" for r in rows) or None
        return len(rows)

    def needs_refresh(self, max_age_hours: float = 24) -> bool:
        if not self.loaded or not self.updated_at:
            return True
        ts = parse_iso(self.updated_at)
        return ts is None or utcnow() - ts > timedelta(hours=max_age_hours)

    def replace_all(self, assets: list[dict]) -> int:
        """Store a fresh asset list from Alpaca (only active, listed US equities are kept)."""
        now = iso()
        rows = []
        for a in assets:
            exch = (a.get("exchange") or "").upper().replace(" ", "")
            if exch not in LISTED_EXCHANGES:
                continue
            sym = (a.get("symbol") or "").upper()
            if not sym or "/" in sym:
                continue
            rows.append((sym, a.get("name") or "", clean_company_name(a.get("name") or ""), exch,
                         int(bool(a.get("tradable"))), int(bool(a.get("shortable"))),
                         int(bool(a.get("easy_to_borrow"))), int(bool(a.get("fractionable"))), now))
        if not rows:
            return 0
        with self.db._write_lock:
            c = self.db.conn()
            c.execute("BEGIN")
            try:
                c.execute("DELETE FROM tickers")
                c.executemany("INSERT INTO tickers (symbol, name, clean_name, exchange, tradable, shortable, "
                              "easy_to_borrow, fractionable, updated_at) VALUES (?,?,?,?,?,?,?,?,?)", rows)
                c.execute("COMMIT")
            except Exception:
                c.execute("ROLLBACK")
                raise
        self.load_from_db()
        log.info("Ticker table updated: %d US-listed symbols", len(rows))
        return len(rows)

    def _build(self, rows: list[dict]) -> None:
        by_symbol: dict[str, TickerInfo] = {}
        by_name: dict[tuple[str, ...], list[str]] = {}
        for r in rows:
            info = TickerInfo(r["symbol"], r["name"] or "", r["clean_name"] or "", r["exchange"] or "",
                              bool(r["tradable"]), bool(r["shortable"]), bool(r["easy_to_borrow"]))
            by_symbol[info.symbol] = info
            name_l = info.name.lower()
            # Fund names are long and generic ("SPDR S&P 500 ETF Trust"); match funds by ticker only.
            if not info.clean_name or " etf" in f" {name_l}" or " fund" in f" {name_l}" or "trust" in name_l.split():
                continue
            key = tuple(tokenize(info.clean_name))
            if 1 <= len(key) <= 6:
                by_name.setdefault(key, []).append(info.symbol)
        aliases = {}
        for phrase, sym in ALIASES.items():
            if not by_symbol or sym in by_symbol:
                aliases[tuple(tokenize(phrase))] = sym
        with self._lock:
            self.by_symbol = by_symbol
            self.by_name = by_name
            self.aliases = aliases
            lens = [len(k) for k in by_name] + [len(k) for k in aliases] + [1]
            self.max_name_len = min(max(lens), 6)

    # ---------------------------------------------------------------- lookups
    def get(self, symbol: str) -> TickerInfo | None:
        return self.by_symbol.get((symbol or "").upper())

    def is_valid(self, symbol: str, require_tradable: bool = True) -> bool:
        info = self.get(symbol)
        return info is not None and (info.tradable or not require_tradable)

    def name_of(self, symbol: str) -> str:
        info = self.get(symbol)
        return info.name if info else ""
