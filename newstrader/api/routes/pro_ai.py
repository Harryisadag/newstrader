"""Pro AI (Settings -> Pro AI): status, what suits this PC, the one-time download, start / stop, "Test my PC" and
deleting the downloads. Progress is pushed to the window as "pro_ai" events."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Body, Depends, Query

from ...context import AppContext
from ..deps import bad_request, get_ctx

router = APIRouter(tags=["pro_ai"])


def _pro(ctx: AppContext):
    p = ctx.service("pro_ai")
    if p is None:
        raise bad_request("Pro AI isn't running (the engine is still starting).", 503)
    return p


@router.get("/pro-ai")
async def pro_status(ctx: AppContext = Depends(get_ctx)):
    return await asyncio.to_thread(_pro(ctx).status)


@router.get("/pro-ai/plan")
async def pro_plan(build: str = Query(""), model: str = Query(""), ctx: AppContext = Depends(get_ctx)):
    """This PC's graphics card, the recommended server build and model, download sizes and free disk space
    (`build` / `model`: what would be downloaded - "auto" or a catalog key; default: the saved settings)."""
    return await asyncio.to_thread(_pro(ctx).plan, build or None, model or None)


@router.post("/pro-ai/setup")
async def pro_setup(body: dict = Body(default={}), ctx: AppContext = Depends(get_ctx)):
    body = body or {}
    try:
        return _pro(ctx).setup(str(body.get("build") or "auto"), str(body.get("model") or "auto"))
    except ValueError as exc:
        raise bad_request(str(exc)) from exc
    except RuntimeError as exc:
        raise bad_request(str(exc), 409) from exc


@router.post("/pro-ai/setup/cancel")
async def pro_setup_cancel(ctx: AppContext = Depends(get_ctx)):
    _pro(ctx).cancel_setup()
    return {"ok": True}


@router.post("/pro-ai/start")
async def pro_start(ctx: AppContext = Depends(get_ctx)):
    pro = _pro(ctx)
    if not pro.settings.enabled:
        raise bad_request("Turn Pro AI on first (Settings -> Pro AI).", 409)
    if pro.downloading:
        raise bad_request("Pro AI is still downloading - it starts by itself when that's done.", 409)
    pro._spawn(pro.start_server(), "pro-ai-start")
    return pro.summary()


@router.post("/pro-ai/stop")
async def pro_stop(ctx: AppContext = Depends(get_ctx)):
    pro = _pro(ctx)
    await pro.stop_server()
    return pro.summary()


@router.post("/pro-ai/selftest")
async def pro_selftest(ctx: AppContext = Depends(get_ctx)):
    try:
        _pro(ctx).run_selftest()
    except RuntimeError as exc:
        raise bad_request(str(exc), 409) from exc
    return {"ok": True}


@router.delete("/pro-ai/downloads")
async def pro_delete(ctx: AppContext = Depends(get_ctx)):
    try:
        freed = await _pro(ctx).delete_downloads()
    except RuntimeError as exc:
        raise bad_request(str(exc), 409) from exc
    except Exception as exc:  # e.g. a file still in use on Windows
        raise bad_request(str(exc), 500) from exc
    return {"ok": True, "freed_bytes": freed}
