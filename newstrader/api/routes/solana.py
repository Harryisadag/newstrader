"""The Solana memecoin desk's endpoints. PAPER ONLY: none of them can trade - they show the desk, pause or resume it,
check the feed now, or start a new paper wallet."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Body, Depends, Query

from ...context import AppContext
from ..deps import bad_request, get_ctx

router = APIRouter(tags=["solana"])


def _desk(ctx: AppContext):
    desk = ctx.service("solana")
    if desk is None:
        raise bad_request("The Solana paper desk isn't running", 503)
    return desk


@router.get("/solana")
async def solana_overview(limit: int = Query(60, ge=1, le=500), ctx: AppContext = Depends(get_ctx)):
    """Everything the Solana (paper) tab shows: status, wallet, the checklist feed, positions, trades and the audit."""
    return await asyncio.to_thread(_desk(ctx).snapshot, limit)


@router.get("/solana/candidates/{cand_id}")
async def solana_candidate(cand_id: int, ctx: AppContext = Depends(get_ctx)):
    """One coin: why the feed picked it, every gate result of every attempt, and its paper trade."""
    cand = await asyncio.to_thread(_desk(ctx).candidate, cand_id)
    if cand is None:
        raise bad_request("No such coin", 404)
    return cand


@router.post("/solana/pause")
async def solana_pause(ctx: AppContext = Depends(get_ctx)):
    """No new coins or paper buys (open paper positions are still watched)."""
    desk = _desk(ctx)
    desk.pause()
    return desk.summary()


@router.post("/solana/resume")
async def solana_resume(ctx: AppContext = Depends(get_ctx)):
    desk = _desk(ctx)
    desk.resume()
    return desk.summary()


@router.post("/solana/check-now")
async def solana_check_now(ctx: AppContext = Depends(get_ctx)):
    """Read the newest pools now instead of waiting for the next minute."""
    try:
        return await _desk(ctx).check_feed_now()
    except RuntimeError as exc:
        raise bad_request(str(exc), 409) from exc


@router.post("/solana/reset")
async def solana_reset(body: dict = Body(default={}), ctx: AppContext = Depends(get_ctx)):
    """A new paper wallet with the starting SOL. Needs the confirmation word typed exactly."""
    desk = _desk(ctx)
    try:
        await asyncio.to_thread(desk.reset_wallet, str(body.get("phrase", "")))
    except PermissionError as exc:
        raise bad_request(str(exc), 403) from exc
    return desk.summary()
