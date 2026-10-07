from __future__ import annotations

from datetime import timedelta

from fastapi import APIRouter, Depends, Query

from ...context import AppContext
from ...db import iso, utcnow
from ...performance.stats import compute_stats
from ..deps import bad_request, get_ctx

router = APIRouter(tags=["performance"])


@router.get("/performance")
async def performance(horizon: str = Query("1h"), days: int = Query(30, ge=0, le=3650),
                      ctx: AppContext = Depends(get_ctx)):
    if horizon not in ("5m", "1h", "1d"):
        raise bad_request("horizon must be 5m, 1h or 1d")
    sql = ("SELECT s.id, s.created_at, s.ticker, s.direction, s.confidence, s.source_name, s.source_type, s.traded, "
           "s.action, s.reasoning, p.price_t0, p.price_5m, p.price_1h, p.price_1d, p.ret_5m, p.ret_1h, p.ret_1d, p.status "
           "FROM signals s JOIN signal_prices p ON p.signal_id = s.id WHERE s.merged_into IS NULL")
    params: list = []
    if days:
        sql += " AND s.created_at >= ?"
        params.append(iso(utcnow() - timedelta(days=days)))
    sql += " ORDER BY s.id DESC"
    rows = ctx.db.query(sql, params)
    stats = compute_stats(rows, horizon)
    return {**stats, "recent": rows[:200], "measured_total": len(rows)}
