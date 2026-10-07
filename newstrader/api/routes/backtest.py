from __future__ import annotations

import json
from datetime import date, timedelta

from fastapi import APIRouter, Body, Depends

from ...backtest.runner import BacktestParams, estimate, lookahead_note
from ...context import AppContext
from ..deps import bad_request, get_ctx

router = APIRouter(tags=["backtest"])

LOOKAHEAD_WARNING = ("Claude was trained on data up to a cutoff date, so for older news it may already 'know' "
                     "what happened next. Treat backtest results on older dates as optimistic.")
LOCAL_WARNING = ("The local engine is free. If you trained its price model, only backtest dates AFTER its "
                 "training range give an honest result.")


async def _warning(ctx: AppContext, p: BacktestParams) -> str:
    if p.engine != "local":
        return LOOKAHEAD_WARNING
    pipeline = ctx.service("pipeline")
    pm = pipeline.local.price_model_for_run() if pipeline else None
    if pm is None:
        return "The local engine is free. No trained price model is in use, so this tests sentiment scoring only."
    return lookahead_note(pm, p.start) or LOCAL_WARNING


def _params(body: dict, default_engine: str) -> BacktestParams:
    try:
        start = date.fromisoformat(str(body.get("start")))
        end = date.fromisoformat(str(body.get("end")))
    except ValueError as exc:
        raise bad_request("Pick a start and end date.") from exc
    if end < start:
        raise bad_request("End date is before the start date.")
    if end - start > timedelta(days=366):
        raise bad_request("Keep the range to a year or less.")
    if end >= date.today() + timedelta(days=1):
        raise bad_request("The end date can't be in the future.")
    symbols = [s.strip().upper().lstrip("$") for s in str(body.get("symbols", "")).replace(";", ",").split(",") if s.strip()]
    try:
        max_articles = int(body.get("max_articles", 100))
        budget = float(body.get("budget_usd", 5))
    except (TypeError, ValueError) as exc:
        raise bad_request("Max stories and budget must be numbers.") from exc
    if not 1 <= max_articles <= 2000:
        raise bad_request("Max stories must be between 1 and 2000.")
    hold = str(body.get("hold", "eod"))
    engine = str(body.get("engine") or default_engine)
    if engine not in ("local", "claude"):
        raise bad_request("Engine must be 'local' or 'claude'.")
    return BacktestParams(start=start, end=end, symbols=symbols, max_articles=max_articles,
                          model=str(body.get("model") or ""), budget_usd=max(0.01, budget), hold=hold, engine=engine)


def _runner(ctx: AppContext):
    r = ctx.service("backtest")
    if r is None:
        raise bad_request("Backtest engine isn't running", 503)
    return r


@router.post("/backtest/estimate")
async def bt_estimate(body: dict = Body(...), ctx: AppContext = Depends(get_ctx)):
    p = _params(body, ctx.config.settings.ai.engine)
    return {**estimate(p, ctx.config.settings.ai.model, p.engine), "warning": await _warning(ctx, p)}


@router.post("/backtest/run")
async def bt_run(body: dict = Body(...), ctx: AppContext = Depends(get_ctx)):
    p = _params(body, ctx.config.settings.ai.engine)
    if body.get("confirm") is not True:
        raise bad_request("Confirm the cost estimate first.")
    try:
        run_id = _runner(ctx).launch(p)
    except RuntimeError as exc:
        raise bad_request(str(exc), 409) from exc
    return {"id": run_id}


@router.get("/backtest/runs")
async def bt_runs(ctx: AppContext = Depends(get_ctx)):
    rows = ctx.db.query("SELECT id, created_at, params, status, progress, message, cost_usd, summary FROM backtest_runs "
                        "ORDER BY id DESC LIMIT 50")
    for r in rows:
        r["params"] = json.loads(r["params"] or "{}")
        r["summary"] = json.loads(r["summary"]) if r["summary"] else None
    return {"runs": rows, "warning": LOOKAHEAD_WARNING}


@router.get("/backtest/runs/{run_id}")
async def bt_run_detail(run_id: int, ctx: AppContext = Depends(get_ctx)):
    run = ctx.db.query_one("SELECT * FROM backtest_runs WHERE id = ?", (run_id,))
    if run is None:
        raise bad_request("Unknown backtest", 404)
    run["params"] = json.loads(run["params"] or "{}")
    run["summary"] = json.loads(run["summary"]) if run["summary"] else None
    results = ctx.db.query("SELECT * FROM backtest_results WHERE run_id = ? ORDER BY news_time, id", (run_id,))
    return {"run": run, "results": results}


@router.post("/backtest/runs/{run_id}/cancel")
async def bt_cancel(run_id: int, ctx: AppContext = Depends(get_ctx)):
    r = _runner(ctx)
    if r.current_run == run_id:
        r.cancel()
    return {"ok": True}


@router.delete("/backtest/runs/{run_id}")
async def bt_delete(run_id: int, ctx: AppContext = Depends(get_ctx)):
    r = _runner(ctx)
    if r.current_run == run_id and r.running:
        raise bad_request("Cancel the run before deleting it.")
    ctx.db.execute("DELETE FROM backtest_results WHERE run_id = ?", (run_id,))
    ctx.db.execute("DELETE FROM backtest_runs WHERE id = ?", (run_id,))
    return {"ok": True}
