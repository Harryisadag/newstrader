from __future__ import annotations

from fastapi import APIRouter, Body, Depends

from ... import paths
from ...context import AppContext
from ..deps import bad_request, get_ctx

router = APIRouter(tags=["keys"])


@router.get("/keys")
async def get_keys(ctx: AppContext = Depends(get_ctx)):
    return {"keys": ctx.keys.masked(), "env_file": str(paths.env_file())}


@router.put("/keys")
async def put_keys(values: dict[str, str | None] = Body(...), ctx: AppContext = Depends(get_ctx)):
    try:
        ctx.keys.update(values)
    except ValueError as exc:
        raise bad_request(str(exc)) from exc
    ctx.bus.publish("keys_changed", {})
    for svc in list(ctx.services.values()):
        hook = getattr(svc, "on_keys_changed", None)
        if hook is not None:
            await hook()
    return {"keys": ctx.keys.masked(), "env_file": str(paths.env_file())}
