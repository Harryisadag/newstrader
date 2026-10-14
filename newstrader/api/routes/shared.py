"""Shared training: "Label old news" (Pro AI, Settings -> Pro AI), and everyone's labels from GitHub, "Share my labels"
and "Train with everyone's labels" (Backtest -> Local ML model). Progress is pushed to the window as "pro_ai_labels",
"shared_labels" and "student_training" events. Nothing here can place a trade."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends

from ...context import AppContext
from ..deps import bad_request, get_ctx

router = APIRouter(tags=["shared_training"])


def _pro(ctx: AppContext):
    p = ctx.service("pro_ai")
    if p is None:
        raise bad_request("Pro AI isn't running (the engine is still starting).", 503)
    return p


def _shared(ctx: AppContext):
    s = ctx.service("shared_training")
    if s is None:
        raise bad_request("Shared training isn't running (the engine is still starting).", 503)
    return s


@router.get("/pro-ai/labels")
async def labels_status(ctx: AppContext = Depends(get_ctx)):
    return await asyncio.to_thread(_pro(ctx).teacher.status)


@router.post("/pro-ai/labels/start")
async def labels_start(ctx: AppContext = Depends(get_ctx)):
    teacher = _pro(ctx).teacher
    if not ctx.db.scalar("SELECT 1 FROM ml_samples WHERE text_version = 2 LIMIT 1"):
        raise bad_request("There are no old headlines to label yet. Train the price model first (Backtest -> Local "
                          "ML model): that downloads the old headlines Pro AI labels.", 409)
    return teacher.start()


@router.post("/pro-ai/labels/pause")
async def labels_pause(ctx: AppContext = Depends(get_ctx)):
    return _pro(ctx).teacher.pause()


@router.get("/ml/shared")
async def shared_status(ctx: AppContext = Depends(get_ctx)):
    return await asyncio.to_thread(_shared(ctx).status)


@router.post("/ml/shared/pull")
async def shared_pull(ctx: AppContext = Depends(get_ctx)):
    """Get everyone's new label files from GitHub now."""
    try:
        return await _shared(ctx).pull()
    except RuntimeError as exc:
        raise bad_request(str(exc), 409) from exc


@router.post("/ml/shared/export")
async def shared_export(ctx: AppContext = Depends(get_ctx)):
    """Share my labels: save the file, show it and open GitHub's upload page."""
    try:
        return await asyncio.to_thread(_shared(ctx).export)
    except ValueError as exc:
        raise bad_request(str(exc)) from exc


@router.post("/ml/student/train")
async def student_train(ctx: AppContext = Depends(get_ctx)):
    """Train with everyone's labels (in the background)."""
    try:
        return _shared(ctx).train()
    except RuntimeError as exc:
        raise bad_request(str(exc), 409) from exc


@router.post("/ml/student/cancel")
async def student_cancel(ctx: AppContext = Depends(get_ctx)):
    _shared(ctx).cancel_training()
    return {"ok": True}


@router.delete("/ml/student")
async def student_delete(ctx: AppContext = Depends(get_ctx)):
    try:
        await asyncio.to_thread(_shared(ctx).delete_student)
    except RuntimeError as exc:
        raise bad_request(str(exc), 409) from exc
    return {"ok": True}
