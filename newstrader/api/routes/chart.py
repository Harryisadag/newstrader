from __future__ import annotations

from fastapi import APIRouter, Depends

from ...chart.service import NoChartData
from ...context import AppContext
from ..deps import bad_request, get_ctx

router = APIRouter(tags=["chart"])


@router.get("/chart/{symbol}")
async def chart_card(symbol: str, ctx: AppContext = Depends(get_ctx)):
    """One stock's chart for the Market tab: the latest session's price and VWAP, the indicator values, the patterns
    showing now (each explained in plain English), nearby support / resistance, the summary and any chart signal."""
    charts = ctx.service("chart")
    if charts is None:
        raise bad_request("Charts aren't running", 503)
    try:
        return await charts.card(symbol)
    except ValueError as exc:
        raise bad_request(str(exc)) from exc
    except NoChartData as exc:
        raise bad_request(str(exc), 503) from exc
    except Exception as exc:  # the data provider failed: say so instead of a bare 500
        raise bad_request(f"Couldn't read the chart: {exc}", 502) from exc
