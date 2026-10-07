"""Claude API cost estimates (USD per million tokens) and today's spend."""

from __future__ import annotations

from ..db import Database
from ..state import trading_day

# input, output, cache write (5-minute), cache read  - $ per 1M tokens
PRICES: dict[str, tuple[float, float, float, float]] = {
    "claude-sonnet-5-5": (2.00, 10.00, 2.50, 0.20),
    "claude-sonnet-5": (2.00, 10.00, 2.50, 0.20),
    "claude-haiku-4-5": (1.00, 5.00, 1.25, 0.10),
    "claude-opus-5-5": (4.00, 20.00, 5.00, 0.20),
    "claude-opus-5": (5.00, 25.00, 6.25, 0.50),
    "claude-fable-5-1": (10.00, 50.00, 12.50, 0.25),
}
DEFAULT_PRICE = PRICES["claude-opus-5-5"]  # unknown model -> assume an expensive one, never under-count


def price_for(model: str) -> tuple[float, float, float, float]:
    model = (model or "").lower()
    if model in PRICES:
        return PRICES[model]
    # dated snapshot ids, e.g. claude-haiku-4-5-20251001
    for key in sorted(PRICES, key=len, reverse=True):
        if model.startswith(key):
            return PRICES[key]
    return DEFAULT_PRICE


def estimate_cost(model: str, input_tokens: int = 0, output_tokens: int = 0, cache_write_tokens: int = 0,
                  cache_read_tokens: int = 0) -> float:
    p_in, p_out, p_cw, p_cr = price_for(model)
    total = (input_tokens * p_in + output_tokens * p_out + cache_write_tokens * p_cw + cache_read_tokens * p_cr)
    return round(total / 1_000_000, 6)


def spend_today(db: Database, include_backtests: bool = False) -> float:
    sql = "SELECT COALESCE(SUM(cost_usd), 0) FROM analyses WHERE trading_day = ?"
    if not include_backtests:
        sql += " AND is_backtest = 0"
    return float(db.scalar(sql, (trading_day(),)) or 0.0)


def calls_today(db: Database) -> int:
    return int(db.scalar("SELECT COUNT(*) FROM analyses WHERE trading_day = ? AND is_backtest = 0",
                         (trading_day(),)) or 0)
