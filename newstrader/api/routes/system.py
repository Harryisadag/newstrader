from __future__ import annotations

import os
import subprocess
import sys

from fastapi import APIRouter, Body, Depends

from ... import paths
from ...context import AppContext
from ...diagnostics import run_diagnostics
from ...tools import gpu_info
from ..deps import bad_request, get_ctx

router = APIRouter(tags=["system"])


@router.post("/diagnostics")
async def diagnostics(ctx: AppContext = Depends(get_ctx)):
    return await run_diagnostics(ctx)


@router.get("/system/gpu")
async def gpu(ctx: AppContext = Depends(get_ctx)):
    import asyncio

    return await asyncio.to_thread(gpu_info)


@router.get("/system/paths")
async def system_paths(ctx: AppContext = Depends(get_ctx)):
    return {
        "data": str(paths.data_dir()),
        "env": str(paths.env_file()),
        "logs": str(paths.logs_dir()),
        "exports": str(paths.exports_dir()),
        "models": str(paths.models_dir()),
    }


@router.post("/system/open-folder")
async def open_folder(body: dict = Body(...), ctx: AppContext = Depends(get_ctx)):
    which = body.get("which", "data")
    targets = {
        "data": paths.data_dir,
        "logs": paths.logs_dir,
        "exports": paths.exports_dir,
        "models": paths.models_dir,
    }
    if which not in targets:
        raise bad_request("unknown folder")
    folder = targets[which]()
    try:
        if sys.platform == "win32":
            os.startfile(str(folder))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(folder)])
        else:
            subprocess.Popen(["xdg-open", str(folder)])
    except Exception as exc:
        raise bad_request(f"Couldn't open {folder}: {exc}") from exc
    return {"ok": True, "path": str(folder)}
