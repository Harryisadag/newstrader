from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends

from ...ai.engine import ENGINES
from ...config import CLAUDE_MODELS, WHISPER_LANGUAGES, WHISPER_MODELS
from ...context import AppContext
from ...llm.service import setting_options
from ...trading import crypto
from ..deps import get_ctx

router = APIRouter(tags=["settings"])


def crypto_info(ctx: AppContext) -> dict:
    """Can this account trade crypto, and which coins in the list Alpaca lists as tradable (for Settings -> Crypto)."""
    trader = ctx.service("trader")
    broker = getattr(trader, "broker", None)
    kind = getattr(trader, "broker_kind", "") if trader is not None else ""
    supported = bool(getattr(broker, "supports_crypto", False))
    table = getattr(ctx.service("pipeline"), "tickers", None)
    pairs = {s for s, info in (getattr(table, "by_symbol", None) or {}).items()
             if info.exchange == crypto.EXCHANGE and info.tradable}
    wanted = ctx.config.settings.crypto.symbols
    if kind == "simulator":
        why = ("The built-in simulator can't trade crypto (its free prices have no crypto). Crypto needs an Alpaca "
               "paper account - add your keys in API keys and pick Alpaca in Settings -> Trading.")
    elif broker is None:
        why = "Not connected to Alpaca yet - crypto needs an Alpaca paper account (API keys)."
    else:
        why = ""
    return {"supported": supported, "broker": kind, "why": why, "live": ctx.state.mode == "live",
            "listed": sorted(s for s in wanted if s in pairs),
            "not_listed": sorted(s for s in wanted if s not in pairs) if pairs else []}


def _payload(ctx: AppContext) -> dict:
    data = ctx.config.as_dict()
    data.pop("sources", None)  # sources have their own endpoints
    return {
        "settings": data,
        "meta": {
            "claude_models": CLAUDE_MODELS,
            "engines": ENGINES,
            "whisper_models": WHISPER_MODELS,
            "languages": {"auto": "Detect automatically", **WHISPER_LANGUAGES},
            "effort_note": "Effort only applies to Sonnet/Opus. Haiku 4.5 ignores it.",
            "crypto": crypto_info(ctx),
            **setting_options(),
        },
    }


@router.get("/settings")
async def get_settings(ctx: AppContext = Depends(get_ctx)):
    return _payload(ctx)


@router.put("/settings")
async def put_settings(patch: dict[str, Any] = Body(...), ctx: AppContext = Depends(get_ctx)):
    patch = dict(patch)
    patch.pop("sources", None)
    ctx.config.update(patch)  # raises ValidationError -> 422 with readable messages
    ctx.bus.publish("settings_changed", {})
    return _payload(ctx)


@router.post("/settings/reset")
async def reset_settings(ctx: AppContext = Depends(get_ctx)):
    ctx.config.reset(keep_sources=True)
    ctx.bus.publish("settings_changed", {})
    return _payload(ctx)
