"""The lookup table of every US-listed stock (symbol + cleaned company name), from Alpaca's asset list.

Cached in SQLite and refreshed once a day, so the pre-filter works instantly at startup.
"""

from __future__ import annotations

import logging
import re
import threading
import unicodedata
from dataclasses import dataclass
from datetime import timedelta

from ..db import Database, iso, parse_iso, utcnow
from .keywords import ALIASES, COMMON_WORDS, CONTEXT_NAMES

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
    r"\bcompanies\b", r"registered (?:ordinary )?shares?", r"\bsponsored\b", r"\bp\.l\.c\.?", r"\bs\.p\.a\.?",
    r"\ba/s\b", r"\basa\b", r"\boyj\b", r"\bkgaa\b", r"\(publ\)", r"depositary receipts?", r"depository receipts?",
    r"\bsa/nv\b", r"\bs\.?e\.?\b", r"\bk\.?k\.?\b",
    r"\((?:[a-z]{2}|delaware|maryland|nevada|new|holding company)\)",  # "(DE)": state of incorporation
]
# Words that describe what a company does; "Sarepta Therapeutics" is called "Sarepta" in headlines
GENERIC_TAIL = {
    "therapeutics", "pharmaceuticals", "pharmaceutical", "pharma", "biosciences", "bioscience", "biotherapeutics",
    "biopharma", "biopharmaceuticals", "biologics", "biotech", "biotechnology", "medical", "technologies",
    "technology", "tech", "interactive", "financial", "holdings", "holding", "systems", "solutions", "software",
    "networks", "brands", "worldwide", "enterprises", "industries", "entertainment", "communications",
    "semiconductor", "semiconductors", "motors", "motor", "airlines", "athletica", "labs", "laboratories",
    "platforms", "health", "healthcare", "wellness", "oncology", "genomics", "sciences", "robotics", "devices",
    "electronics", "micro", "incorporated", "international", "global", "energy", "resources", "cruises", "outdoor",
}
_SUFFIX_RE = re.compile("|".join(_SUFFIXES), re.IGNORECASE)
_PUNCT_RE = re.compile(r"[^a-z0-9&'\- ]+")
_SPACES = re.compile(r"\s+")


def fold(text: str) -> str:
    """Accents off ('Nestlé' -> 'Nestle'), curly apostrophes straightened."""
    text = (text or "").replace("’", "'").replace("‘", "'")
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def clean_company_name(name: str) -> str:
    """'Apple Inc. Common Stock' -> 'apple'; 'Alphabet Inc. Class C Capital Stock' -> 'alphabet'."""
    s = fold(name).lower()
    s = s.replace(",", " ").replace("/", " ")
    for _ in range(2):
        s = _SUFFIX_RE.sub(" ", s)
    s = _PUNCT_RE.sub(" ", s)
    s = _SPACES.sub(" ", s).strip(" -'&")
    return s


def tokenize(text: str) -> list[str]:
    """Lower-case words; possessives dropped ("Apple's" -> "apple"), accents removed."""
    return [n for n, _o in token_pairs(text)]


_TOKEN = re.compile(r"[A-Za-z0-9&'\-]+")
_NUM_DASH = re.compile(r"(?<=\b\d)-(?=[A-Za-z])")


def token_pairs(text: str) -> list[tuple[str, str]]:
    """(lower-case token, the word as written) - the same tokens as tokenize(), lined up with the original text."""
    out = []
    # "UPDATE 1-Robinhood ..." (wire-service prefix): the number isn't part of the name
    for m in _TOKEN.finditer(_NUM_DASH.sub(" ", fold(text))):
        orig = m.group(0)
        t = orig.lower()
        if t.endswith("'s"):
            t = t[:-2]
        t = t.strip("-'")
        if t:
            out.append((t, orig))
    return out


def short_name(key: tuple[str, ...]) -> tuple[str, ...] | None:
    """('sarepta', 'therapeutics') -> ('sarepta',): the name headlines actually use. None if it would be generic."""
    short = list(key)
    while len(short) > 1 and short[-1] in GENERIC_TAIL:
        short.pop()
    if len(short) == len(key) or not short:
        return None
    if len(short) == 1:
        w = short[0]
        # everyday words like "ford" stay: the pre-filter only counts them next to company words ("Ford recalls")
        if len(w) < 4 or w in COMMON_WORDS or w in GENERIC_TAIL or w.isdigit():
            return None
    return tuple(short)


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
        shorts: dict[tuple[str, ...], list[str]] = {}
        for r in rows:
            info = TickerInfo(r["symbol"], r["name"] or "", r["clean_name"] or "", r["exchange"] or "",
                              bool(r["tradable"]), bool(r["shortable"]), bool(r["easy_to_borrow"]))
            by_symbol[info.symbol] = info
            name_l = info.name.lower()
            # Fund names are long and generic ("SPDR S&P 500 ETF Trust"); match funds by ticker only.
            if not info.clean_name or any(w in f" {name_l} " for w in (" etf", " fund ", " fund,", " ishares",
                                                                         " spdr", " index ", "trust units")):
                continue
            key = tuple(tokenize(info.clean_name))
            if 1 <= len(key) <= 6:
                by_name.setdefault(key, []).append(info.symbol)
                short = short_name(key)
                if short is not None:
                    shorts.setdefault(short, []).append(info.symbol)
        for key, syms in shorts.items():  # a full name always wins; a short name shared by two companies is unused
            if key not in by_name and len(syms) == 1:
                by_name[key] = syms
        for phrase, sym in CONTEXT_NAMES.items():
            key = tuple(tokenize(phrase))
            if (not by_symbol or sym in by_symbol) and key not in by_name:
                by_name[key] = [sym]
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
