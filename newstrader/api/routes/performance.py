from __future__ import annotations

from datetime import timedelta

from fastapi import APIRouter, Depends, Query

from ...context import AppContext
from ...db import iso, utcnow
from ...performance.speed import speed_summary
from ...performance.stats import MIN_COUNT, compute_stats
from ..deps import bad_request, get_ctx

router = APIRouter(tags=["performance"])


@router.get("/performance")
async def performance(horizon: str = Query("1h"), days: int = Query(30, ge=0, le=3650),
                      ctx: AppContext = Depends(get_ctx)):
    if horizon not in ("5m", "1h", "1d"):
        raise bad_request("horizon must be 5m, 1h or 1d")
    sql = ("SELECT s.id, s.created_at, s.ticker, s.direction, s.confidence, s.source_name, s.source_type, s.traded, "
           "s.action, s.reasoning, s.engine, s.event, p.price_t0, p.price_5m, p.price_1h, p.price_1d, p.ret_5m, "
           "p.ret_1h, p.ret_1d, p.status FROM signals s JOIN signal_prices p ON p.signal_id = s.id "
           "WHERE s.merged_into IS NULL")
    params: list = []
    if days:
        sql += " AND s.created_at >= ?"
        params.append(iso(utcnow() - timedelta(days=days)))
    sql += " ORDER BY s.id DESC"
    rows = ctx.db.query(sql, params)
    stats = compute_stats(rows, horizon)
    return {**stats, "recent": rows[:200], "measured_total": len(rows)}


@router.get("/performance/speed")
async def speed(days: int = Query(30, ge=1, le=365), ctx: AppContext = Depends(get_ctx)):
    """How fast news turns into orders, and how late each source's news is when it arrives."""
    since = iso(utcnow() - timedelta(days=days))
    traded = ctx.db.query(
        "SELECT s.news_published_at, s.news_received_at, s.decided_at, s.created_at, s.review_status, "
        "o.submitted_at AS order_submitted_at FROM signals s JOIN orders o ON o.id = s.order_id "
        "WHERE s.created_at >= ? AND s.traded = 1", (since,))
    # every headline from the text sources (SQLite works out the seconds); for TV the moment the words were spoken
    # is only kept on signals
    delays = ctx.db.query(
        "SELECT source_name, (julianday(received_at) - julianday(published_at)) * 86400.0 AS secs FROM news_items "
        "WHERE received_at >= ? AND published_at IS NOT NULL AND source_type != 'stream'", (since,))
    delays += ctx.db.query(
        "SELECT source_name, (julianday(news_received_at) - julianday(news_published_at)) * 86400.0 AS secs "
        "FROM signals WHERE created_at >= ? AND source_type = 'stream' AND news_published_at IS NOT NULL "
        "AND merged_into IS NULL AND COALESCE(action, '') != 'watch'", (since,))
    summary = speed_summary(traded, [(r["source_name"], r["secs"]) for r in delays], min_count=MIN_COUNT)
    return {"days": days, **summary}
