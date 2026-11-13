"""The reject audit: what really happened to every coin the checklist looked at, 1 hour and 24 hours later.

Each follow-up asks GeckoTerminal for the pool again (up to 30 pools in one call, inside the desk's GeckoTerminal
budget). Two lists come out of it:
- "Rejected then ran": a coin a gate stopped whose price later went up 100% or more from when it was spotted - the
  gates may be too strict there (or it was luck: most of these still rug later).
- "Passed then rugged": a coin that passed every gate and then lost 80% of its price or its pool's money, or that
  RugCheck later marked as rugged - the gates missed something.
Plus, for each gate, how many coins it stopped and how many of those ran anyway.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from ..db import Database, iso
from .gates import GATES
from .parsers import Pool

FOLLOW_UPS = {"1h": timedelta(hours=1), "24h": timedelta(hours=24)}
RAN_PCT = 100.0  # "ran": the price went up at least this much (%) from when the coin was spotted
RUG_DROP_PCT = 80.0  # "rugged": the price or the pool's money fell at least this much (%)
AUDITED = ("rejected", "bought", "passed")  # the checklist reached a verdict (skipped / expired ones never ran)


def due(db: Database, now: datetime, limit: int = 60) -> list[tuple[dict, str]]:
    """Candidates whose 1-hour or 24-hour follow-up is due -> [(candidate, "1h" | "24h")], oldest first."""
    out: list[tuple[dict, str]] = []
    marks = ", ".join("?" for _ in AUDITED)
    for label, wait in FOLLOW_UPS.items():
        rows = db.query(f"SELECT * FROM sol_candidates WHERE status IN ({marks}) AND checked_{label}_at IS NULL AND "
                        f"created_at <= ? ORDER BY created_at LIMIT ?", (*AUDITED, iso(now - wait), limit))
        out += [(r, label) for r in rows]
    return out[:limit]


def _change(now_value: float | None, then_value: float | None) -> float | None:
    if now_value is None or not then_value:
        return None
    return (now_value / then_value - 1) * 100


def outcome(cand: dict) -> str | None:
    """"ran" (rejected, then +100%), "rugged" (passed, then -80% or marked rugged) or None."""
    prices = [_change(cand.get(f"price_{k}_usd"), cand.get("signal_price_usd")) for k in FOLLOW_UPS]
    pools = [_change(cand.get(f"liquidity_{k}_usd"), cand.get("signal_liquidity_usd")) for k in FOLLOW_UPS]
    if cand["status"] == "rejected":
        return "ran" if any(p is not None and p >= RAN_PCT for p in prices) else None
    if cand["status"] in ("bought", "passed"):
        drops = [x for x in prices + pools if x is not None]
        if cand.get("rugged_later") or any(x <= -RUG_DROP_PCT for x in drops):
            return "rugged"
    return None


def record(db: Database, cand: dict, label: str, pool: Pool | None, now: datetime, rugged: bool | None = None) -> dict:
    """Save one follow-up. A pool GeckoTerminal no longer knows counts as "no data" (a liquidity of 0 is not assumed)."""
    values: dict = {f"checked_{label}_at": iso(now)}
    if pool is not None:
        values[f"price_{label}_usd"] = pool.price_usd
        values[f"liquidity_{label}_usd"] = pool.reserve_usd
    if rugged is not None:
        values["rugged_later"] = int(rugged or bool(cand.get("rugged_later")))
    db.update("sol_candidates", cand["id"], values)
    fresh = db.query_one("SELECT * FROM sol_candidates WHERE id = ?", (cand["id"],))
    db.update("sol_candidates", cand["id"], {"outcome": outcome(fresh)})
    fresh["outcome"] = outcome(fresh)
    return fresh


def _row(c: dict) -> dict:
    best = [x for x in (_change(c.get("price_1h_usd"), c.get("signal_price_usd")),
                        _change(c.get("price_24h_usd"), c.get("signal_price_usd"))) if x is not None]
    pool = [x for x in (_change(c.get("liquidity_1h_usd"), c.get("signal_liquidity_usd")),
                        _change(c.get("liquidity_24h_usd"), c.get("signal_liquidity_usd"))) if x is not None]
    return {"id": c["id"], "created_at": c["created_at"], "symbol": c["symbol"], "mint": c["mint"], "pool": c["pool"],
            "status": c["status"], "killer_gate": c["killer_gate"],
            "killer_name": GATES.get(c["killer_gate"]) if c["killer_gate"] else None, "reason": c["reason"],
            "change_1h_pct": _change(c.get("price_1h_usd"), c.get("signal_price_usd")),
            "change_24h_pct": _change(c.get("price_24h_usd"), c.get("signal_price_usd")),
            "best_pct": max(best) if best else None, "worst_pct": min(best) if best else None,
            "pool_change_pct": min(pool) if pool else None, "rugged_later": bool(c.get("rugged_later"))}


def summary(db: Database, limit: int = 50) -> dict:
    """Both lists, and per gate: how many coins it stopped, how many of those were followed up, and how many ran."""
    counts = {g: {"gate": g, "name": name, "stopped": 0, "followed_up": 0, "ran": 0} for g, name in GATES.items()}
    for row in db.query("SELECT killer_gate, COUNT(*) AS n, SUM(checked_1h_at IS NOT NULL OR checked_24h_at IS NOT NULL) "
                        "AS f, SUM(outcome = 'ran') AS r FROM sol_candidates WHERE status = 'rejected' AND killer_gate "
                        "IS NOT NULL GROUP BY killer_gate"):
        if row["killer_gate"] in counts:
            counts[row["killer_gate"]].update(stopped=row["n"], followed_up=row["f"] or 0, ran=row["r"] or 0)
    totals = {r["status"]: r["n"] for r in db.query("SELECT status, COUNT(*) AS n FROM sol_candidates GROUP BY status")}
    passed = db.query_one("SELECT COUNT(*) AS n, SUM(outcome = 'rugged') AS r, SUM(checked_1h_at IS NOT NULL OR "
                          "checked_24h_at IS NOT NULL) AS f FROM sol_candidates WHERE status IN ('bought', 'passed')")
    ran = db.query("SELECT * FROM sol_candidates WHERE outcome = 'ran' ORDER BY created_at DESC LIMIT ?", (limit,))
    rugged = db.query("SELECT * FROM sol_candidates WHERE outcome = 'rugged' ORDER BY created_at DESC LIMIT ?", (limit,))
    marks = ", ".join("?" for _ in AUDITED)
    waiting = db.scalar(f"SELECT COUNT(*) FROM sol_candidates WHERE status IN ({marks}) AND checked_24h_at IS NULL",
                        AUDITED)
    return {"by_gate": list(counts.values()), "totals": totals,
            "passed": {"count": passed["n"] or 0, "followed_up": passed["f"] or 0, "rugged": passed["r"] or 0},
            "rejected_then_ran": [_row(c) for c in ran], "passed_then_rugged": [_row(c) for c in rugged],
            "waiting": waiting or 0}
