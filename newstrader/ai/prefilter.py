"""Stage 1: a cheap, local check that decides whether text is worth sending to Claude.

Text passes if it mentions a listed company (by $cashtag, "(NASDAQ: XYZ)", a ticker in capitals, the
company name, or a known alias like "Google" or "Jensen Huang") or contains a market-moving keyword.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .keywords import (
    ALIAS_NOT_BEFORE,
    AMBIGUOUS_NAMES,
    COMMON_WORDS,
    COMPANY_CONTEXT_AFTER,
    COMPANY_CONTEXT_BEFORE,
    COUNTRY_ETFS,
    EUROZONE_COUNTRY_ETFS,
    EUROZONE_ETF,
    INDEX_CHANGE_WORDS_AFTER,
    INDEX_ETF_ALIASES,
    INDEX_MEMBERSHIP_WORDS,
    MACRO_WORDS,
    MARKET_KEYWORDS,
    MUSK_OTHER_COMPANIES,
    NOT_TICKERS,
)
from .tickers import TickerTable, token_pairs

_CASHTAG = re.compile(r"(?<![\w$])\$([A-Za-z]{1,5}(?:\.[A-Za-z])?)\b")
_EXCHANGE = re.compile(r"\b(?:NYSE(?:\s?American|\s?Arca)?|NASDAQ|Nasdaq|AMEX|Cboe|CBOE)\s*:\s*\$?([A-Z]{1,5}(?:\.[A-Z])?)\b")
_CAPS = re.compile(r"(?<![\w$.])([A-Z]{2,5}(?:\.[A-Z])?)(?![\w])")
_MARKERS = re.compile(r"New:|Earlier \(context only[^\n]*|\[\d{1,2}:\d{2}(?::\d{2})?\]")
_KEYWORD_RES = [(kw, re.compile(r"(?<![\w])" + re.escape(kw) + r"(?![\w])", re.IGNORECASE)) for kw in MARKET_KEYWORDS]


@dataclass
class Candidate:
    symbol: str
    name: str
    why: str


@dataclass
class PrefilterResult:
    hit: bool
    candidates: list[Candidate] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"hit": self.hit, "keywords": self.keywords,
                "candidates": [{"symbol": c.symbol, "name": c.name, "why": c.why} for c in self.candidates]}


def _shouty(text: str) -> bool:
    """Headlines written in ALL CAPS would make every word look like a ticker."""
    letters = [ch for ch in text if ch.isalpha()]
    return len(letters) > 20 and sum(ch.isupper() for ch in letters) / len(letters) > 0.6


def _starts_longer_name(sym: str, rest: str, table: TickerTable) -> bool:
    """'GE Vernova', 'GE HealthCare': the capitals start another company's name, so they aren't GE itself."""
    nxt = [t for t, _o in token_pairs(rest[:40])][:1]
    if not nxt:
        return False
    first_two = getattr(table, "_name_starts", None)
    if first_two is None or first_two[0] is not table.by_name:  # (built once per ticker-table refresh)
        starts: dict[tuple[str, str], set[str]] = {}
        for key, syms in list(table.by_name.items()) + [(k, [v]) for k, v in table.aliases.items()]:
            if len(key) >= 2:
                starts.setdefault(key[:2], set()).update(syms)
        first_two = (table.by_name, starts)
        table._name_starts = first_two
    return bool(first_two[1].get((sym.lower(), nxt[0]), set()) - {sym})


def _company_context(tokens: list[str], originals: list[str], i: int, n: int) -> bool:
    """Is an everyday-word name ("Target", "Ford", "Gap") used as the company here?"""
    after = tokens[i + n] if i + n < len(tokens) else ""
    before = tokens[i - 1] if i > 0 else ""
    possessive = originals[i + n - 1].lower().endswith(("'s", "s'"))
    return (after in COMPANY_CONTEXT_AFTER or before in COMPANY_CONTEXT_BEFORE
            or (possessive and before not in ("price", "the")))


def prefilter(text: str, table: TickerTable, source_symbols: list[str] | None = None,
              allow_keyword_only: bool = True, max_candidates: int = 10,
              country_etfs: bool = False) -> PrefilterResult:
    found: dict[str, Candidate] = {}

    def add(sym: str, why: str) -> None:
        sym = sym.upper()
        if sym in found or len(found) >= max_candidates:
            return
        if table.loaded and table.get(sym) is None:
            return
        found[sym] = Candidate(sym, table.name_of(sym), why)

    for sym in source_symbols or []:
        add(sym, "tagged by source")
    for m in _CASHTAG.finditer(text):
        add(m.group(1), f"${m.group(1).upper()}")
    for m in _EXCHANGE.finditer(text):
        add(m.group(1), m.group(0))

    if table.loaded and not _shouty(text):
        for m in _CAPS.finditer(text):
            sym = m.group(1)
            if sym in NOT_TICKERS or len(sym.replace(".", "")) < 2:
                continue
            if table.get(sym) is not None and not _starts_longer_name(sym, text[m.end():], table):
                add(sym, sym)

    # company names and aliases, longest phrase first
    pairs = token_pairs(text)
    tokens = [t for t, _o in pairs]
    originals = [o for _t, o in pairs]
    # e.g. a lower-case speech transcript (the app's own 'New:' / 'Earlier' markers don't count)
    caseless = not any(ch.isupper() for ch in _MARKERS.sub("", text))
    token_set = set(tokens)
    n_max = max(table.max_name_len, 3)
    i = 0
    while i < len(tokens):
        matched = False
        for n in range(min(n_max, len(tokens) - i), 0, -1):
            key = tuple(tokens[i:i + n])
            why = " ".join(key)
            first = originals[i]
            capital = first[:1].isupper() or first[:1].isdigit()
            nxt = tokens[i + n] if i + n < len(tokens) else ""
            sym = table.aliases.get(key)
            if sym is not None:
                if n == 1:
                    # single-word aliases ("Apple", "Shell") must be written as a name, and not "Amazon rainforest"
                    if not (capital or (caseless and len(key[0]) >= 4 and key[0] not in COMMON_WORDS)):
                        continue
                    if nxt in ALIAS_NOT_BEFORE.get(key[0], ()):
                        continue
                if sym == "TSLA" and "musk" in key and token_set & MUSK_OTHER_COMPANIES and "tesla" not in token_set:
                    continue  # "Elon Musk's SpaceX ..." is not Tesla news
                if sym in INDEX_ETF_ALIASES:
                    prev = tokens[i - 1] if i > 0 else ""
                    if prev == "the" and i > 1:
                        prev = tokens[i - 2]
                    if prev in INDEX_MEMBERSHIP_WORDS or nxt in INDEX_CHANGE_WORDS_AFTER:
                        continue  # "set to join the S&P 500" / "S&P 500 adds Affirm" is about the company, not SPY
            else:
                syms = table.by_name.get(key)
                if not syms:
                    continue
                if not (capital or caseless):
                    continue  # company names are written with a capital ("the best buy" is not Best Buy)
                if n == 1:
                    word = key[0]
                    if len(word) < 3:
                        continue
                    # everyday words ("Target", "Gap", "Ford") only count next to company words
                    if word in AMBIGUOUS_NAMES and not _company_context(tokens, originals, i, n):
                        continue
                    if nxt in ALIAS_NOT_BEFORE.get(word, ()):
                        continue
                sym = sorted(syms, key=lambda s: (len(s), s))[0]
                for extra in syms:
                    if extra != sym:
                        add(extra, why)
            add(sym, why)
            i += n
            matched = True
            break
        if not matched:
            i += 1

    # international macro news ("Bank of Japan raises rates") -> that country's US-listed fund
    if country_etfs and not found and table.loaded and token_set & MACRO_WORDS:
        for n in (4, 3, 2, 1):
            for j in range(len(tokens) - n + 1):
                phrase = " ".join(tokens[j:j + n])
                etf = COUNTRY_ETFS.get(phrase)
                if etf and (n > 1 or originals[j][:1].isupper() or caseless or phrase in ("uk", "boj", "ecb", "rba")):
                    name = phrase.title() if len(phrase) > 3 else phrase.upper()
                    add(etf, f"country: {name}")
                    if etf in EUROZONE_COUNTRY_ETFS:
                        add(EUROZONE_ETF, f"country: {name} (euro area)")

    keywords = [kw for kw, rx in _KEYWORD_RES if rx.search(text)]
    candidates = list(found.values())
    hit = bool(candidates) or (allow_keyword_only and bool(keywords))
    return PrefilterResult(hit=hit, candidates=candidates, keywords=keywords[:10])
