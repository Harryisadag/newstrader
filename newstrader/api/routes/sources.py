from __future__ import annotations

import re
from typing import Any

from fastapi import APIRouter, Body, Depends

from ...config import WHISPER_LANGUAGES
from ...context import AppContext
from ..deps import bad_request, get_ctx

router = APIRouter(tags=["sources"])

TYPE_LABELS = {
    "stream": "Live stream (YouTube or any video/audio URL)",
    "rss": "RSS / Atom news feed",
    "social_rss": "Social account via RSS (e.g. Truth Social, rss.app feed for X)",
    "x_account": "X account via official X API (needs paid X_BEARER_TOKEN)",
    "alpaca_news": "Alpaca / Benzinga news websocket",
}


def _with_health(ctx: AppContext, src: dict) -> dict:
    comps = {c["component"]: c for c in ctx.state.components()}
    health = comps.get(f"source:{src['id']}")
    return {**src, "health": health}


def _notify(ctx: AppContext) -> None:
    ctx.bus.publish("sources_changed", {})


@router.get("/sources")
async def list_sources(ctx: AppContext = Depends(get_ctx)):
    s = ctx.config.settings
    sources = [_with_health(ctx, src.model_dump(mode="json")) for src in s.sources]
    return {"sources": sources, "types": TYPE_LABELS,
            "languages": {"": "Default (Settings -> Transcription)", "auto": "Detect automatically",
                          **WHISPER_LANGUAGES},
            "engine": s.ai.engine, "max_streams": s.transcription.max_concurrent_streams}


@router.post("/sources/toggle-many")
async def toggle_many(body: dict[str, Any] = Body(...), ctx: AppContext = Depends(get_ctx)):
    ids = body.get("ids")
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
        raise bad_request("ids must be a list of source ids")
    changed = ctx.config.set_sources_enabled(ids, bool(body.get("enabled")))
    _notify(ctx)
    return {"changed": changed}


@router.post("/sources")
async def add_source(body: dict[str, Any] = Body(...), ctx: AppContext = Depends(get_ctx)):
    name = str(body.get("name", "")).strip()
    if not name:
        raise bad_request("Give the source a name.")
    base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "source"
    existing = {s.id for s in ctx.config.settings.sources}
    new_id, n = base, 2
    while new_id in existing:
        new_id, n = f"{base}-{n}", n + 1
    data = {**body, "id": new_id, "builtin": False}
    if data.get("type") == "alpaca_news" and any(s.type == "alpaca_news" for s in ctx.config.settings.sources):
        raise bad_request("There is already an Alpaca news source.")
    src = ctx.config.upsert_source(data)
    _notify(ctx)
    return _with_health(ctx, src.model_dump(mode="json"))


@router.put("/sources/{source_id}")
async def update_source(source_id: str, body: dict[str, Any] = Body(...), ctx: AppContext = Depends(get_ctx)):
    current = ctx.config.source(source_id)
    if current is None:
        raise bad_request("Unknown source", 404)
    data = {**current.model_dump(mode="json"), **body, "id": source_id}
    src = ctx.config.upsert_source(data)
    _notify(ctx)
    return _with_health(ctx, src.model_dump(mode="json"))


@router.post("/sources/{source_id}/toggle")
async def toggle_source(source_id: str, body: dict[str, Any] = Body(...), ctx: AppContext = Depends(get_ctx)):
    try:
        src = ctx.config.set_source_enabled(source_id, bool(body.get("enabled")))
    except KeyError as exc:
        raise bad_request("Unknown source", 404) from exc
    _notify(ctx)
    return _with_health(ctx, src.model_dump(mode="json"))


@router.delete("/sources/{source_id}")
async def delete_source(source_id: str, ctx: AppContext = Depends(get_ctx)):
    try:
        ctx.config.delete_source(source_id)
    except KeyError as exc:
        raise bad_request("Unknown source", 404) from exc
    ctx.state.clear_status(f"source:{source_id}")
    _notify(ctx)
    return {"ok": True}
