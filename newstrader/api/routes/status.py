from __future__ import annotations

from fastapi import APIRouter, Depends

from ... import __version__, paths
from ...context import AppContext
from ...state import trading_day
from ..deps import get_ctx

router = APIRouter(tags=["status"])


def spend_today(ctx: AppContext) -> float:
    value = ctx.db.scalar(
        "SELECT COALESCE(SUM(cost_usd), 0) FROM analyses WHERE trading_day = ? AND is_backtest = 0",
        (trading_day(),),
    )
    return round(float(value or 0), 4)


async def status_summary(ctx: AppContext, light: bool = False) -> dict:
    s = ctx.config.settings
    k = ctx.keys.keys
    out = {
        "version": __version__,
        "mode": ctx.state.mode,
        "kill_switch": ctx.state.kill_switch,
        "halted_today": ctx.state.halted_today,
        "auto_trade": s.trading.auto_trade,
        "spend_today": spend_today(ctx),
        "spend_cap": s.ai.daily_spend_cap_usd,
        "ai_model": s.ai.model,
        "ai_engine": s.ai.engine,
        "keys": {
            "alpaca_paper": k.has_alpaca_paper,
            "alpaca_live": k.has_alpaca_live,
            "anthropic": k.has_anthropic,
            "discord": k.has_discord,
        },
        "components": ctx.state.components(),
    }
    trader = ctx.service("trader")
    if trader is not None and hasattr(trader, "summary"):
        out.update(trader.summary())
    monitor = ctx.service("market")  # "market" in this dict is the market clock (from the trader)
    if monitor is not None and hasattr(monitor, "summary"):
        out["monitor"] = monitor.summary()
    if not light:
        out["data_dir"] = str(paths.data_dir())
        out["env_file"] = str(paths.env_file())
    return out


@router.get("/status")
async def get_status(ctx: AppContext = Depends(get_ctx)):
    return await status_summary(ctx)
