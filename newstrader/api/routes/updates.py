from __future__ import annotations

from fastapi import APIRouter, Depends

from ... import __version__
from ...context import AppContext
from ...updater import UpdateError
from ..deps import bad_request, get_ctx

router = APIRouter(tags=["updates"])


@router.get("/updates")
async def get_updates(ctx: AppContext = Depends(get_ctx)):
    checker = ctx.service("updates")
    updater = ctx.service("updater")
    if checker is None:
        out = {"enabled": ctx.config.settings.ui.check_updates, "current": __version__, "newer": False}
    else:
        out = checker.payload()
    if updater is not None:
        out["install"] = updater.summary()
    return out


@router.post("/updates/check")
async def check_updates(ctx: AppContext = Depends(get_ctx)):
    checker = ctx.service("updates")
    if checker is None:
        raise bad_request("The update checker isn't running", 503)
    return await checker.check()


@router.post("/updates/install")
async def install_update(ctx: AppContext = Depends(get_ctx)):
    """Update now: download, check, swap in and restart (the ready-made app only; never while an order is placed)."""
    updater = ctx.service("updater")
    if updater is None:
        raise bad_request("The updater isn't running", 503)
    try:
        return await updater.install()
    except UpdateError as exc:
        raise bad_request(str(exc), exc.status) from exc


@router.post("/updates/cancel")
async def cancel_update(ctx: AppContext = Depends(get_ctx)):
    updater = ctx.service("updater")
    if updater is None:
        raise bad_request("The updater isn't running", 503)
    return updater.cancel()
