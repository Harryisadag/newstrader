"""Shared test helpers: a fake Claude client and a small ticker table."""

from __future__ import annotations

import json
from types import SimpleNamespace

from newstrader.ai.tickers import TickerTable

ASSETS = [
    {"symbol": "AAPL", "name": "Apple Inc. Common Stock", "exchange": "NASDAQ", "tradable": True, "shortable": True, "easy_to_borrow": True},
    {"symbol": "NVDA", "name": "NVIDIA Corporation Common Stock", "exchange": "NASDAQ", "tradable": True, "shortable": True, "easy_to_borrow": True},
    {"symbol": "TSLA", "name": "Tesla, Inc. Common Stock", "exchange": "NASDAQ", "tradable": True, "shortable": True, "easy_to_borrow": True},
    {"symbol": "GOOGL", "name": "Alphabet Inc. Class A Common Stock", "exchange": "NASDAQ", "tradable": True, "shortable": True, "easy_to_borrow": True},
    {"symbol": "GOOG", "name": "Alphabet Inc. Class C Capital Stock", "exchange": "NASDAQ", "tradable": True, "shortable": True, "easy_to_borrow": True},
    {"symbol": "F", "name": "Ford Motor Company Common Stock", "exchange": "NYSE", "tradable": True, "shortable": True, "easy_to_borrow": True},
    {"symbol": "TGT", "name": "Target Corporation Common Stock", "exchange": "NYSE", "tradable": True, "shortable": True, "easy_to_borrow": True},
    {"symbol": "JPM", "name": "JPMorgan Chase & Co. Common Stock", "exchange": "NYSE", "tradable": True, "shortable": True, "easy_to_borrow": True},
    {"symbol": "MCD", "name": "McDonald's Corporation Common Stock", "exchange": "NYSE", "tradable": True, "shortable": True, "easy_to_borrow": True},
    {"symbol": "BRK.B", "name": "Berkshire Hathaway Inc. Class B", "exchange": "NYSE", "tradable": True, "shortable": True, "easy_to_borrow": True},
    {"symbol": "SPY", "name": "SPDR S&P 500 ETF Trust", "exchange": "ARCA", "tradable": True, "shortable": True, "easy_to_borrow": True},
    {"symbol": "ALL", "name": "The Allstate Corporation Common Stock", "exchange": "NYSE", "tradable": True, "shortable": True, "easy_to_borrow": True},
    {"symbol": "HALT", "name": "Halted Example Corp", "exchange": "NYSE", "tradable": False, "shortable": False, "easy_to_borrow": False},
    {"symbol": "OTCX", "name": "Some OTC Company", "exchange": "OTC", "tradable": True, "shortable": False, "easy_to_borrow": False},
]


def make_tickers(db) -> TickerTable:
    t = TickerTable(db)
    t.replace_all(ASSETS)
    return t


def signal_json(*signals) -> str:
    return json.dumps({"signals": list(signals)})


def sig(ticker="NVDA", direction="bullish", confidence=88, **kw) -> dict:
    base = {"ticker": ticker, "company": "", "speaker": "Reuters", "bull_case": "Strong demand.",
            "bear_case": "Maybe priced in.", "direction": direction, "confidence": confidence,
            "time_sensitivity": "immediate", "reasoning": "Big contract win announced."}
    base.update(kw)
    return base


class FakeClaude:
    """Mimics anthropic.AsyncAnthropic just enough for ClaudeAnalyzer."""

    def __init__(self, responses: list[str] | None = None, stop_reason: str = "end_turn", error: Exception | None = None):
        self.responses = list(responses or [])
        self.stop_reason = stop_reason
        self.error = error
        self.requests: list[dict] = []
        self.messages = SimpleNamespace(create=self._create)
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    async def _create(self, **params):
        self.requests.append(params)
        if self.error:
            raise self.error
        text = self.responses.pop(0) if self.responses else signal_json()
        content = [SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)]
        usage = SimpleNamespace(input_tokens=1200, output_tokens=300, cache_read_input_tokens=900,
                                cache_creation_input_tokens=0)
        return SimpleNamespace(content=content, usage=usage, model=params["model"], stop_reason=self.stop_reason,
                               stop_details=SimpleNamespace(category="cyber") if self.stop_reason == "refusal" else None)
