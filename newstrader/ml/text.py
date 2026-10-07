"""Text helpers for the local ML engine: find the sentences about one company and tidy text for the models."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..ai.prefilter import Candidate
from ..ai.tickers import TickerTable, tokenize

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"'“(\[]?[A-Z0-9$])|\n+")
_TIMESTAMP = re.compile(r"^\[\d{1,2}:\d{2}(?::\d{2})?\]\s*")
_WS = re.compile(r"\s+")
NEW_MARKER = "New:"
CONTEXT_MARKER = "Earlier (context only"
MAX_SNIPPET_CHARS = 700


def split_sentences(text: str) -> list[str]:
    out = []
    for part in _SENTENCE_SPLIT.split(text or ""):
        part = _TIMESTAMP.sub("", part.strip())
        part = _WS.sub(" ", part).strip()
        if len(part) >= 3:
            out.append(part)
    return out


def analysable_body(body: str, kind: str) -> str:
    """For live-TV transcripts only the 'New:' part counts; earlier lines were already analysed."""
    if kind == "transcript" and NEW_MARKER in (body or ""):
        return body.split(NEW_MARKER, 1)[1]
    return body or ""


@dataclass
class Mention:
    terms: list[tuple[str, ...]]  # lower-case token phrases (company name, alias, matched words)
    symbol: str


def mention_for(cand: Candidate, table: TickerTable) -> Mention:
    terms: list[tuple[str, ...]] = []

    def add(text: str) -> None:
        key = tuple(tokenize(text))
        if key and key not in terms and not (len(key) == 1 and len(key[0]) < 3):
            terms.append(key)

    info = table.get(cand.symbol)
    if info is not None and info.clean_name:
        add(info.clean_name)
    if cand.why and cand.why != "tagged by source" and not cand.why.startswith("$") and ":" not in cand.why:
        add(cand.why)
    for phrase, sym in table.aliases.items():
        if sym == cand.symbol and phrase not in terms:
            terms.append(phrase)
    return Mention(terms=terms, symbol=cand.symbol)


def mentions(sentence: str, m: Mention) -> bool:
    sym = re.escape(m.symbol)
    if re.search(rf"(?<![\w$])\$?{sym}(?![\w])", sentence):  # ticker as written (case-sensitive)
        return True
    toks = tokenize(sentence)
    for term in m.terms:
        n = len(term)
        for i in range(len(toks) - n + 1):
            if tuple(toks[i:i + n]) == term:
                return True
    return False


@dataclass
class Target:
    snippet: str  # the text the sentiment model reads
    in_headline: bool
    in_body: bool
    tagged: bool


def target_text(title: str, body: str, kind: str, cand: Candidate, table: TickerTable,
                max_sentences: int = 3) -> Target | None:
    """The headline plus the sentences that talk about this company.

    Returns None when a live-TV transcript only mentioned the company in earlier, already-analysed lines.
    """
    m = mention_for(cand, table)
    body = analysable_body(body, kind)
    tagged = cand.why == "tagged by source"
    if kind == "transcript":
        sentences = split_sentences(body)
        hits = [s for s in sentences if mentions(s, m)]
        if not hits:
            return None
        # a sentence of context either side helps with "they" / "the company"
        idx = [i for i, s in enumerate(sentences) if s in hits][:max_sentences]
        keep: list[str] = []
        for i in idx:
            for j in (i - 1, i, i + 1):
                if 0 <= j < len(sentences) and sentences[j] not in keep:
                    keep.append(sentences[j])
        return Target(snippet=_clip(" ".join(keep)), in_headline=False, in_body=True, tagged=False)

    title = _WS.sub(" ", (title or "").strip())
    in_headline = mentions(title, m)
    body_sents = [s for s in split_sentences(body) if s != title]
    hits = [s for s in body_sents if mentions(s, m)][:max_sentences]
    parts = [title] if title else []
    if hits:
        parts += hits
    elif not in_headline and body_sents:
        parts.append(body_sents[0])  # tagged by the source but never named: read the lead sentence
    return Target(snippet=_clip(" ".join(parts)), in_headline=in_headline, in_body=bool(hits), tagged=tagged)


def relevance(t: Target, n_symbols: int) -> float:
    """How clearly the item is about this company (1.0 = it's in the headline)."""
    if t.in_headline:
        return 1.0
    if t.tagged and n_symbols <= 2:
        return 0.95
    if t.in_body:
        return 0.85
    return 0.8


def mask_company(snippet: str, cand: Candidate, table: TickerTable) -> str:
    """Replace the company's name/ticker with a placeholder so the trained model learns wording, not names."""
    m = mention_for(cand, table)
    text = re.sub(rf"(?<![\w])\$?{re.escape(cand.symbol)}(?![\w])", " companyx ", snippet)
    toks = tokenize(text)
    out: list[str] = []
    i = 0
    terms = sorted(m.terms, key=len, reverse=True)
    while i < len(toks):
        for term in terms:
            n = len(term)
            if tuple(toks[i:i + n]) == term:
                out.append("companyx")
                i += n
                break
        else:
            out.append(toks[i])
            i += 1
    return " ".join(out)


def _clip(text: str) -> str:
    text = _WS.sub(" ", text).strip()
    return text[:MAX_SNIPPET_CHARS]
