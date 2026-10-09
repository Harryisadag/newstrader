"""Which stories Pro AI reads, and how its reading compares with the main engine's.

Hard cases (scope "hard"): live TV transcripts, stories not in English, market-wide news with no company named, and
stories the local engine could only judge by their wording (no news event recognised) or called neutral although
companies were named. Scope "all" also sends every other story that names a company.
"""

from __future__ import annotations

from ..ai.prefilter import PrefilterResult
from ..sources.base import NewsItem
from .engine import allowed_tickers, is_macro_only

TV = "TV transcript"
NOT_ENGLISH = "not in English"
MACRO = "market-wide news (no company named)"
WORDING_ONLY = "wording only - no news event recognised"
MAIN_NEUTRAL = "the main engine called it neutral"
EVERY_STORY = "every story (scope: all)"


def hard_case(item: NewsItem, pre: PrefilterResult, main_engine: str | None, main_signals) -> str:
    """Why this story is hard for the main engine ("" = it isn't). `main_engine`: the engine that read it (None = it
    never got there, e.g. filtered out for not being in English); `main_signals`: its validated signals."""
    if item.kind == "transcript":
        return TV
    if item.language not in ("", "en"):
        return NOT_ENGLISH
    if is_macro_only(pre):
        return MACRO
    if main_engine == "local" and pre.candidates:
        calls = [s for s in main_signals or [] if s.direction != "neutral"]
        if not calls:
            return MAIN_NEUTRAL
        if any(not s.event for s in calls):
            return WORDING_ONLY
    return ""


def route(item: NewsItem, pre: PrefilterResult, main_engine: str | None, main_signals, scope: str) -> tuple[str, bool]:
    """(why Pro AI reads this story, whether it is a hard case). ("", False) = Pro AI skips it."""
    if not allowed_tickers(pre):
        return "", False
    reason = hard_case(item, pre, main_engine, main_signals)
    if reason:
        return reason, True
    if scope == "all" and pre.candidates:
        return EVERY_STORY, False
    return "", False


def calls_of(signals) -> dict[str, str]:
    """{ticker: direction} of validated signals."""
    return {s.ticker: s.direction for s in signals or []}


def compare(main: dict[str, str] | None, pro: dict[str, str]) -> str:
    """"agree" when both engines call every stock the same way (a stock one of them doesn't name counts as neutral),
    "disagree" otherwise, "no_main" when the main engine didn't read the story."""
    if main is None:
        return "no_main"
    same = all(main.get(t, "neutral") == pro.get(t, "neutral") for t in set(main) | set(pro))
    return "agree" if same else "disagree"
