"""Check every Claude response before anything can act on it.

A signal is only accepted if the JSON is valid, the ticker exists in the US-listed ticker table and is
tradable on Alpaca, confidence is a whole number from 0 to 100, and every field has an allowed value.
Anything else is rejected with a reason (shown in Logs -> Rejected AI responses).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .prompts import DIRECTIONS, TIME_SENSITIVITY
from .tickers import TickerTable

_TICKER_RE = re.compile(r"^[A-Z]{1,5}(\.[A-Z]{1,2})?$")
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)

TEXT_LIMITS = {"company": 120, "speaker": 160, "bull_case": 1200, "bear_case": 1200, "reasoning": 400}


@dataclass
class ValidSignal:
    ticker: str
    company: str
    speaker: str
    bull_case: str
    bear_case: str
    direction: str
    confidence: int
    time_sensitivity: str
    reasoning: str

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class ValidationResult:
    signals: list[ValidSignal] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    fatal: str | None = None
    raw_count: int = 0

    @property
    def status(self) -> str:
        if self.fatal:
            return "rejected"
        if self.raw_count and not self.signals:
            return "rejected"
        return "ok"

    @property
    def summary(self) -> str:
        if self.fatal:
            return self.fatal
        return "; ".join(self.problems)


def _parse_json(raw: str) -> Any:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = _FENCE_RE.sub("", text).strip()
    return json.loads(text)


def _confidence(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        n = value
    elif isinstance(value, float) and value.is_integer():
        n = int(value)
    else:  # strings like "85" or "high" are rejected - the schema says integer
        return None
    return n if 0 <= n <= 100 else None


def validate_response(raw: str, tickers: TickerTable, max_signals: int = 3) -> ValidationResult:
    res = ValidationResult()
    try:
        data = _parse_json(raw)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        res.fatal = f"Invalid JSON from Claude: {exc}"
        return res
    if not isinstance(data, dict):
        res.fatal = "Response is not a JSON object"
        return res
    items = data.get("signals")
    if not isinstance(items, list):
        res.fatal = "Response has no 'signals' list"
        return res
    res.raw_count = len(items)
    if not tickers.loaded:
        if items:
            res.fatal = "Ticker table isn't loaded yet (needs Alpaca keys), so tickers can't be verified"
        return res

    accepted: dict[str, ValidSignal] = {}
    for idx, item in enumerate(items, start=1):
        label = f"signal {idx}"
        if not isinstance(item, dict):
            res.problems.append(f"{label}: not an object")
            continue
        ticker = str(item.get("ticker") or "").strip().upper().lstrip("$")
        label = f"{label} ({ticker or 'no ticker'})"
        if not ticker:
            res.problems.append(f"{label}: missing ticker")
            continue
        if not _TICKER_RE.match(ticker):
            res.problems.append(f"{label}: '{ticker}' isn't a valid ticker format")
            continue
        info = tickers.get(ticker)
        if info is None:
            res.problems.append(f"{label}: {ticker} is not a US-listed stock in the ticker table")
            continue
        if not info.tradable:
            res.problems.append(f"{label}: {ticker} isn't tradable on Alpaca")
            continue
        direction = str(item.get("direction") or "").strip().lower()
        if direction not in DIRECTIONS:
            res.problems.append(f"{label}: direction '{item.get('direction')}' not one of {', '.join(DIRECTIONS)}")
            continue
        conf = _confidence(item.get("confidence"))
        if conf is None:
            res.problems.append(f"{label}: confidence '{item.get('confidence')}' is not a whole number 0-100")
            continue
        ts = str(item.get("time_sensitivity") or "").strip().lower()
        if ts not in TIME_SENSITIVITY:
            res.problems.append(f"{label}: time_sensitivity '{item.get('time_sensitivity')}' not allowed")
            continue
        texts = {}
        bad = False
        for key, limit in TEXT_LIMITS.items():
            value = item.get(key, "")
            if value is None:
                value = ""
            if not isinstance(value, str):
                res.problems.append(f"{label}: {key} must be text")
                bad = True
                break
            texts[key] = value.strip()[:limit]
        if bad:
            continue
        if not texts["reasoning"]:
            res.problems.append(f"{label}: missing reasoning")
            continue
        if ticker in accepted:
            res.problems.append(f"{label}: duplicate ticker in one response (kept the first)")
            continue
        accepted[ticker] = ValidSignal(ticker=ticker, company=texts["company"] or info.name,
                                       speaker=texts["speaker"], bull_case=texts["bull_case"],
                                       bear_case=texts["bear_case"], direction=direction, confidence=conf,
                                       time_sensitivity=ts, reasoning=texts["reasoning"])

    signals = sorted(accepted.values(), key=lambda s: s.confidence, reverse=True)
    if len(signals) > max_signals:
        for extra in signals[max_signals:]:
            res.problems.append(f"{extra.ticker}: more than {max_signals} signals in one response (dropped lowest)")
        signals = signals[:max_signals]
    res.signals = signals
    return res
