"""Stage 1: a cheap, local check that decides whether text is worth sending to Claude.

Text passes if it mentions a listed company (by $cashtag, "(NASDAQ: XYZ)", a ticker in capitals, the
company name, or a known alias like "Google" or "Jensen Huang") or contains a market-moving keyword.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .keywords import AMBIGUOUS_NAMES, MARKET_KEYWORDS, NOT_TICKERS
from .tickers import TickerTable, tokenize

_CASHTAG = re.compile(r"(?<![\w$])\$([A-Za-z]{1,5}(?:\.[A-Za-z])?)\b")
_EXCHANGE = re.compile(r"\b(?:NYSE(?:\s?American|\s?Arca)?|NASDAQ|Nasdaq|AMEX|Cboe|CBOE)\s*:\s*\$?([A-Z]{1,5}(?:\.[A-Z])?)\b")
_CAPS = re.compile(r"(?<![\w$.])([A-Z]{2,5}(?:\.[A-Z])?)(?![\w])")
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


def prefilter(text: str, table: TickerTable, source_symbols: list[str] | None = None,
              allow_keyword_only: bool = True, max_candidates: int = 10) -> PrefilterResult:
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
            if table.get(sym) is not None:
                add(sym, sym)

    # company names and aliases, longest phrase first
    raw_tokens = re.findall(r"[A-Za-z0-9&'’\-]+", text)
    tokens = tokenize(text)
    if len(tokens) == len(raw_tokens):
        originals = raw_tokens
    else:  # tokenizer split differently; fall back to lower-case only
        originals = tokens
    n_max = max(table.max_name_len, 3)
    i = 0
    while i < len(tokens):
        matched = False
        for n in range(min(n_max, len(tokens) - i), 0, -1):
            key = tuple(tokens[i:i + n])
            sym = table.aliases.get(key)
            why = " ".join(key)
            if sym is None:
                syms = table.by_name.get(key)
                if syms:
                    if n == 1:
                        word = key[0]
                        orig = originals[i] if i < len(originals) else word
                        # single-word names must be capitalised and not an everyday word
                        if word in AMBIGUOUS_NAMES or len(word) < 3 or not orig[:1].isupper():
                            continue
                    sym = sorted(syms, key=lambda s: (len(s), s))[0]
                    for extra in syms:
                        if extra != sym:
                            add(extra, why)
            if sym is not None:
                add(sym, why)
                i += n
                matched = True
                break
        if not matched:
            i += 1

    keywords = [kw for kw, rx in _KEYWORD_RES if rx.search(text)]
    candidates = list(found.values())
    hit = bool(candidates) or (allow_keyword_only and bool(keywords))
    return PrefilterResult(hit=hit, candidates=candidates, keywords=keywords[:10])
