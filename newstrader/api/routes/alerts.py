from __future__ import annotations

from fastapi import APIRouter, Depends

from ...context import AppContext
from ..deps import bad_request, get_ctx

router = APIRouter(tags=["alerts"])


@router.post("/alerts/test")
async def test_alert(ctx: AppContext = Depends(get_ctx)):
    alerts = ctx.service("alerts")
    if alerts is None:
        raise bad_request("Alerts service isn't running", 503)
    return await alerts.test()


@router.get("/alerts/history")
async def history(ctx: AppContext = Depends(get_ctx)):
    alerts = ctx.service("alerts")
    return {"alerts": list(alerts.history) if alerts else []}
