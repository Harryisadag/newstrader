from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends

from ...ai.engine import ENGINES
from ...config import CLAUDE_MODELS, WHISPER_MODELS
from ...context import AppContext
from ..deps import get_ctx

router = APIRouter(tags=["settings"])


def _payload(ctx: AppContext) -> dict:
    data = ctx.config.as_dict()
    data.pop("sources", None)  # sources have their own endpoints
    return {
        "settings": data,
        "meta": {
            "claude_models": CLAUDE_MODELS,
            "engines": ENGINES,
            "whisper_models": WHISPER_MODELS,
            "effort_note": "Effort only applies to Sonnet/Opus. Haiku 4.5 ignores it.",
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
