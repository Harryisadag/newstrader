from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from ...context import AppContext
from ...market.monitor import RECENT_EVENTS, empty_payload, load_events
from ..deps import bad_request, get_ctx

router = APIRouter(tags=["market"])


def _monitor(ctx: AppContext):
    m = ctx.service("market")
    if m is None:
        raise bad_request("Market monitor isn't running", 503)
    return m


@router.get("/market")
async def market_overview(ctx: AppContext = Depends(get_ctx)):
    """Everything the Market tab shows: market clock, index and world ETFs, movers, what is being watched and
    the latest spikes / market moves."""
    m = ctx.service("market")
    trader = ctx.service("trader")
    out = m.payload() if m is not None else empty_payload(ctx.config.settings)
    out["connected"] = getattr(trader, "account", None) is not None
    out["market"] = getattr(trader, "clock", None)
    out["events"] = load_events(ctx.db, limit=RECENT_EVENTS)
    return out


@router.get("/market/events")
async def market_events(
    limit: int = Query(200, ge=1, le=2000),
    symbol: str = Query(""),
    kind: str = Query(""),
    days: int = Query(7, ge=1, le=60),
    ctx: AppContext = Depends(get_ctx),
):
    return {"events": load_events(ctx.db, limit=limit, symbol=symbol, kind=kind, days=days)}


@router.post("/market/scan")
async def market_scan(ctx: AppContext = Depends(get_ctx)):
    """Check now (also refreshes prices and movers while the market is closed)."""
    m = _monitor(ctx)
    if not ctx.config.settings.market.enabled:
        raise bad_request("The market monitor is turned off (Settings -> Market monitor).", 409)
    return await m.run_once(force=True)
