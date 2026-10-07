from __future__ import annotations

from fastapi import APIRouter, Depends

from ... import __version__
from ...context import AppContext
from ..deps import bad_request, get_ctx

router = APIRouter(tags=["updates"])


@router.get("/updates")
async def get_updates(ctx: AppContext = Depends(get_ctx)):
    checker = ctx.service("updates")
    if checker is None:
        return {"enabled": ctx.config.settings.ui.check_updates, "current": __version__, "newer": False}
    return checker.payload()


@router.post("/updates/check")
async def check_updates(ctx: AppContext = Depends(get_ctx)):
    checker = ctx.service("updates")
    if checker is None:
        raise bad_request("The update checker isn't running", 503)
    return await checker.check()
