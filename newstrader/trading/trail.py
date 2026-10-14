"""The decision trail behind a signal, for the "Why did it trade?" card.

While the Trader decides what to do with a signal it writes down each step: the thresholds and trading style in use,
the confidence before and after each adjustment (the chart check), every check on the way (thresholds, risk checks,
don't chase, the chart) with whether it passed, sent the signal to manual review or stopped it, and the outcome. The
pipeline adds what the AI engine read (the news event, the wording scores, the price model's probability) and Pro
AI's view. It is saved as JSON on the signal (signals.decision_trail) at decision time, so the card shows what was
true then - not today's settings. A later decision on the same signal (you approved it, or a repeat of the news raised
its confidence) keeps the earlier ones under "earlier".
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ..db import iso
from . import presets

log = logging.getLogger(__name__)

VERSION = 1
KEEP_EARLIER = 3

# the risk checks a new position passes, in the order trading/risk.py runs them (README "How a trade happens")
STOCK_CHECKS = ["Kill switch off", "Auto-trade on", "Daily loss limit not reached", "Market open",
                "Not on the blacklist (and on the whitelist, if you use one)", "Cooldown for this stock over",
                "No order for it already waiting", "Room for another position", "Max $ per trade",
                "Max % of the account in one stock", "Enough buying power", "Share price above the minimum",
                "At least one whole share"]
CRYPTO_CHECKS = ["Kill switch off", "Auto-trade on", "Daily loss limit not reached", "Cooldown for this coin over",
                 "No order for it already waiting", "Not already holding it", "Room for another position",
                 "Max $ per crypto trade", "Max % of the account in one coin", "Enough cash (crypto can't use margin)"]
EXIT_CHECKS = ["Kill switch off", "Auto-trade on", "Sell on bearish news is on", "You hold it"]

# check results
PASS, REVIEW, STOP, INFO = "pass", "review", "stop", "info"


class Trail:
    """Collects one decision. Call finish() with the Trader's result to get the dict that is saved."""

    def __init__(self, settings: Any, signal: dict, manual: bool = False, broker: str = "", mode: str = "paper"):
        t, r = settings.trading, settings.risk
        self.data: dict[str, Any] = {
            "version": VERSION,
            "decided_at": iso(),
            "manual": bool(manual),
            "style": presets.detect(settings),
            "account": {"broker": broker or "", "mode": mode},
            "settings": {
                "buy_threshold": t.buy_threshold, "review_threshold": t.review_threshold,
                "stop_loss_pct": t.stop_loss_pct, "take_profit_pct": t.take_profit_pct,
                "max_chase_pct": t.max_chase_pct, "auto_trade": t.auto_trade, "allow_shorting": t.allow_shorting,
                "sell_on_bearish": t.sell_on_bearish, "chart_confirm": settings.chart.confirm,
                "max_dollars_per_trade": r.max_dollars_per_trade, "max_pct_per_stock": r.max_pct_per_stock,
                "max_open_positions": r.max_open_positions, "daily_loss_limit_usd": r.daily_loss_limit_usd,
                "ticker_cooldown_minutes": r.ticker_cooldown_minutes, "market_hours_only": r.market_hours_only,
            },
            "confidence": [{"label": "From the AI engine", "value": _int(signal.get("confidence"))}],
            "chart": None,
            "checks": [],
            "outcome": None,
        }

    def confidence(self, label: str, value: int, change: int | None = None) -> None:
        self.data["confidence"].append({"label": label, "value": int(value), "change": change})

    def chart(self, check: Any, skipped: str = "") -> None:
        """What the chart check said (chart/service.py ChartCheck), or why it didn't run."""
        if check is None:
            self.data["chart"] = {"verdict": "skipped", "reason": skipped}
            return
        self.data["chart"] = {"verdict": check.verdict, "reason": check.reason, "adjust": int(check.adjust or 0),
                              "score": check.columns().get("chart_score"), "review": check.review or ""}
        if check.adjust:
            word = "agrees" if check.adjust > 0 else "disagrees"
            self.confidence(f"Chart {word}", check.confidence, int(check.adjust))

    def check(self, name: str, result: str, detail: str = "", passed: list[str] | None = None) -> None:
        item: dict[str, Any] = {"name": name, "result": result, "detail": detail}
        if passed:
            item["passed"] = list(passed)
        self.data["checks"].append(item)

    def finish(self, result: dict) -> dict:
        self.data["outcome"] = {"action": result.get("action"), "reason": result.get("reason") or "",
                                "traded": bool(result.get("traded"))}
        return self.data


def engine_part(signal: dict, detail: dict | None = None, model: str | None = None) -> dict:
    """What the AI engine read, for the trail: its call and reasoning, plus (local engine) the news event, the wording
    scores, how much the text was about this company and the price model's / student model's reading."""
    d = detail or {}
    flags = signal.get("flags")
    if isinstance(flags, str):
        try:
            flags = json.loads(flags)
        except ValueError:
            flags = []
    return {"name": signal.get("engine") or "", "model": model or "", "direction": signal.get("direction"),
            "confidence": _int(signal.get("confidence")), "event": signal.get("event") or d.get("event") or "",
            "event_detail": d.get("event_detail") or "", "flags": flags or [], "reasoning": signal.get("reasoning") or "",
            "wording": d.get("sentiment"), "price_model": d.get("p_up"), "student": d.get("student"),
            "relevance": d.get("relevance"), "matched": d.get("matched") or ""}


def local_details(raw: str | None) -> dict[str, dict]:
    """The local engine's per-company details ({ticker: detail}) from its raw answer (empty for other engines)."""
    try:
        data = json.loads(raw or "")
    except (TypeError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(d.get("ticker")): d for d in data.get("details") or [] if isinstance(d, dict) and d.get("ticker")}


def store(db, signal_id: int | None, trail: dict | None) -> None:
    """Save a decision on the signal. An earlier decision on the same signal is kept under "earlier" (newest first)."""
    if not signal_id or not trail:
        return
    try:
        row = db.query_one("SELECT decision_trail FROM signals WHERE id = ?", (signal_id,))
        old = load(row["decision_trail"]) if row else None
        if old is not None and old is not trail:
            earlier = [{k: v for k, v in old.items() if k not in ("earlier", "engine")}] + list(old.get("earlier") or [])
            # what the engine read (and Pro AI's view) doesn't change when the same signal is decided again
            carried = {k: old[k] for k in ("engine", "pro_ai") if old.get(k) is not None and trail.get(k) is None}
            trail = {**trail, **carried, "earlier": earlier[:KEEP_EARLIER]}
        db.update("signals", signal_id, {"decision_trail": json.dumps(trail, default=str)})
    except Exception:  # the trail is an explanation: it must never stop a trade being recorded
        log.debug("couldn't save the decision trail of signal %s", signal_id, exc_info=True)


def load(text: str | None) -> dict | None:
    try:
        data = json.loads(text or "")
    except (TypeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
