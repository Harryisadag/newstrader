"""Simulate one bracket trade on 1-minute bars: enter at the next minute's open, exit at stop-loss,
take-profit, or the time limit - whichever comes first.

Conservative assumptions: if a single bar touches both the stop and the target, the STOP is assumed to
have hit first; if the price gaps past the stop, the exit is at the (worse) open price.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ..db import parse_iso


@dataclass
class SimResult:
    entered: bool
    entry_time: str | None = None
    entry_price: float | None = None
    exit_time: str | None = None
    exit_price: float | None = None
    exit_reason: str = ""
    ret_pct: float | None = None  # % gain for the position (positive = profit), long or short

    def pnl(self, qty: float) -> float:
        if not self.entered or self.ret_pct is None or self.entry_price is None:
            return 0.0
        return round(self.entry_price * qty * self.ret_pct / 100, 2)


def simulate_bracket(side: str, bars: list[dict], entry_after: datetime, stop_pct: float, target_pct: float,
                     exit_by: datetime) -> SimResult:
    """side: 'long' or 'short'. bars: sorted 1-minute bars with t/o/h/l/c."""
    entry_bar_idx = None
    for i, b in enumerate(bars):
        t = parse_iso(b["t"])
        if t is not None and t >= entry_after and t < exit_by:
            entry_bar_idx = i
            break
    if entry_bar_idx is None:
        return SimResult(entered=False, exit_reason="no trading after the news (market closed or no data)")

    eb = bars[entry_bar_idx]
    entry = float(eb["o"])
    if entry <= 0:
        return SimResult(entered=False, exit_reason="bad price data")
    if side == "long":
        stop, target = entry * (1 - stop_pct / 100), entry * (1 + target_pct / 100)
    else:
        stop, target = entry * (1 + stop_pct / 100), entry * (1 - target_pct / 100)

    def result(exit_price: float, when: str, reason: str) -> SimResult:
        move = (exit_price - entry) / entry * 100
        return SimResult(True, eb["t"], round(entry, 4), when, round(exit_price, 4), reason,
                         round(move if side == "long" else -move, 4))

    last = eb
    for i in range(entry_bar_idx, len(bars)):
        b = bars[i]
        t = parse_iso(b["t"])
        if t is None:
            continue
        if t >= exit_by:
            break
        o, h, low = float(b["o"]), float(b["h"]), float(b["l"])
        if side == "long":
            if i > entry_bar_idx and o <= stop:
                return result(o, b["t"], "stop-loss (gapped through)")
            if low <= stop:
                return result(stop, b["t"], "stop-loss")
            if i > entry_bar_idx and o >= target:
                return result(o, b["t"], "take-profit (gapped through)")
            if h >= target:
                return result(target, b["t"], "take-profit")
        else:
            if i > entry_bar_idx and o >= stop:
                return result(o, b["t"], "stop-loss (gapped through)")
            if h >= stop:
                return result(stop, b["t"], "stop-loss")
            if i > entry_bar_idx and o <= target:
                return result(o, b["t"], "take-profit (gapped through)")
            if low <= target:
                return result(target, b["t"], "take-profit")
        last = b
    return result(float(last["c"]), last["t"], "time limit (closed at market)")
