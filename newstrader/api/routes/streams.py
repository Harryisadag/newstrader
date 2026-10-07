from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from ...context import AppContext
from ..deps import get_ctx

router = APIRouter(tags=["streams"])


@router.get("/streams")
async def streams(ctx: AppContext = Depends(get_ctx)):
    comps = {c["component"]: c for c in ctx.state.components()}
    mgr = ctx.service("streams")
    running = set(mgr.tasks) if mgr else set()
    live = set(mgr.live) if mgr else set()
    out = []
    for s in ctx.config.settings.sources:
        if s.type != "stream":
            continue
        out.append({"id": s.id, "name": s.name, "url": s.url, "enabled": s.enabled, "running": s.id in running,
                    "live": s.id in live, "live_events": s.live_events, "language": s.language,
                    "translate": s.translate, "health": comps.get(f"source:{s.id}")})
    return {
        "streams": out,
        "transcription_enabled": ctx.config.settings.transcription.enabled,
        "max_streams": ctx.config.settings.transcription.max_concurrent_streams,
        "transcriber": comps.get("transcriber"),
        "engine": mgr.summary() if mgr else None,
    }


@router.get("/transcripts")
async def transcripts(source_id: str = Query(""), limit: int = Query(80, ge=1, le=1000),
                      ctx: AppContext = Depends(get_ctx)):
    if source_id:
        rows = ctx.db.query("SELECT * FROM transcripts WHERE source_id = ? ORDER BY id DESC LIMIT ?", (source_id, limit))
    else:
        rows = ctx.db.query("SELECT * FROM transcripts ORDER BY id DESC LIMIT ?", (limit,))
    rows.reverse()
    return {"lines": rows}
