from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from ...context import AppContext
from ..deps import get_ctx

router = APIRouter(tags=["logs"])

_LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}


@router.get("/logs")
async def get_logs(
    level: str = Query("INFO"),
    limit: int = Query(300, ge=1, le=2000),
    before: int | None = Query(None),
    q: str = Query(""),
    ctx: AppContext = Depends(get_ctx),
):
    min_level = _LEVELS.get(level.upper(), 20)
    allowed = [name for name, num in _LEVELS.items() if num >= min_level]
    sql = f"SELECT * FROM logs WHERE level IN ({','.join('?' for _ in allowed)})"
    params: list = list(allowed)
    if before:
        sql += " AND id < ?"
        params.append(before)
    if q:
        sql += " AND (message LIKE ? OR logger LIKE ?)"
        params += [f"%{q}%", f"%{q}%"]
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    return {"logs": ctx.db.query(sql, params), "components": ctx.state.components()}


@router.get("/rejections")
async def get_rejections(limit: int = Query(100, ge=1, le=1000), ctx: AppContext = Depends(get_ctx)):
    """Claude responses that failed validation (bad JSON, fake ticker, out-of-range confidence...)."""
    rows = ctx.db.query(
        "SELECT id, created_at, source_name, model, status, error, raw_response FROM analyses "
        "WHERE status IN ('rejected', 'refusal', 'error') ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    return {"rejections": rows}
