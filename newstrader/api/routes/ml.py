"""Local ML engine: status, training the price model, retrying the FinBERT download."""

from __future__ import annotations

import asyncio
import json
from datetime import date, timedelta

from fastapi import APIRouter, Body, Depends

from ...context import AppContext
from ...ml.trainer import TrainParams, default_range, market_today
from ..deps import bad_request, get_ctx

router = APIRouter(tags=["ml"])


def _pipeline(ctx: AppContext):
    p = ctx.service("pipeline")
    if p is None:
        raise bad_request("AI engine isn't running", 503)
    return p


def _trainer(ctx: AppContext):
    t = ctx.service("ml_trainer")
    if t is None:
        raise bad_request("Model trainer isn't running", 503)
    return t


def _run_row(r: dict) -> dict:
    r["params"] = json.loads(r["params"] or "{}")
    r["report"] = json.loads(r["report"]) if r.get("report") else None
    return r


@router.get("/ml/status")
async def ml_status(ctx: AppContext = Depends(get_ctx)):
    p = _pipeline(ctx)
    t = ctx.service("ml_trainer")
    runs = [_run_row(r) for r in ctx.db.query("SELECT * FROM ml_runs ORDER BY id DESC LIMIT 10")]
    labelled = ctx.db.scalar("SELECT COUNT(*) FROM ml_samples WHERE label_status = 'ok'") or 0
    start, end = default_range(ctx.config.settings.ml.train_days)
    return {"engine": ctx.config.settings.ai.engine, **p.local.model_status(),
            "training": bool(t and t.running), "current_run": t.current_run if t else None, "runs": runs,
            "cached_samples": int(labelled), "default_start": start.isoformat(), "default_end": end.isoformat(),
            "settings": ctx.config.settings.ml.model_dump()}


def _train_params(body: dict, ctx: AppContext) -> TrainParams:
    ml = ctx.config.settings.ml
    start, end = default_range(ml.train_days)
    try:
        if body.get("start"):
            start = date.fromisoformat(str(body["start"]))
        if body.get("end"):
            end = date.fromisoformat(str(body["end"]))
    except ValueError as exc:
        raise bad_request("Dates must look like 2026-01-31.") from exc
    if end >= market_today():
        raise bad_request("The end date must be before today (today's trading isn't finished).")
    if end < start:
        raise bad_request("End date is before the start date.")
    if (end - start) < timedelta(days=20):
        raise bad_request("Use at least 3-4 weeks of history (more is better).")
    if (end - start) > timedelta(days=740):
        raise bad_request("Keep the range to about two years or less.")
    return TrainParams(start=start, end=end, horizon_minutes=ml.train_horizon_minutes,
                       min_move_pct=ml.train_min_move_pct, max_articles=ml.train_max_articles,
                       redownload=bool(body.get("redownload")))


@router.post("/ml/train")
async def ml_train(body: dict = Body(default={}), ctx: AppContext = Depends(get_ctx)):
    params = _train_params(body or {}, ctx)
    try:
        run_id = _trainer(ctx).launch(params)
    except RuntimeError as exc:
        raise bad_request(str(exc), 409) from exc
    return {"id": run_id}


@router.post("/ml/train/cancel")
async def ml_cancel(ctx: AppContext = Depends(get_ctx)):
    _trainer(ctx).cancel()
    return {"ok": True}


@router.delete("/ml/model")
async def ml_delete_model(ctx: AppContext = Depends(get_ctx)):
    try:
        _trainer(ctx).delete_model()
    except RuntimeError as exc:
        raise bad_request(str(exc), 409) from exc
    return {"ok": True}


@router.post("/ml/reload")
async def ml_reload(ctx: AppContext = Depends(get_ctx)):
    """Retry loading FinBERT (e.g. after the first download failed) and re-read the trained model."""
    p = _pipeline(ctx)
    t = ctx.service("ml_trainer")
    if t is not None and t.running:
        raise bad_request("Training is running - wait for it to finish (or cancel it) first.", 409)
    await asyncio.to_thread(p.local.sentiment.load, ctx.config.settings.ml.sentiment_model, True)
    await asyncio.to_thread(p.local.load_price_model)
    p.update_ml_status()
    return p.local.model_status()
