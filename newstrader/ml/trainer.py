"""Background service that trains the price model (Backtest tab -> Local ML model -> Train).

Steps: download + label history (dataset.py) -> FinBERT scores -> fit + test on the newest 20% (model.py)
-> save to the data folder -> the live engine starts using it if it passed its test.
Nothing here can place a trade.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict, dataclass
from datetime import date, timedelta

from .. import paths
from ..context import AppContext
from ..db import iso
from .dataset import Cancelled, Throttle, build_dataset, forget_cached_days, load_samples
from .model import delete_model, train_price_model

log = logging.getLogger(__name__)


@dataclass
class TrainParams:
    start: date
    end: date
    horizon_minutes: int = 60
    min_move_pct: float = 0.3
    max_articles: int = 20000
    redownload: bool = False


def default_range(days: int, today: date | None = None) -> tuple[date, date]:
    """`days` calendar days ending yesterday (today's session isn't finished, so it isn't used)."""
    end = (today or date.today()) - timedelta(days=1)
    return end - timedelta(days=days - 1), end


class ModelTrainer:
    name = "ml_trainer"

    def __init__(self, ctx: AppContext, throttle_factory=Throttle):
        self.ctx = ctx
        self.throttle_factory = throttle_factory  # tests pass one that never sleeps
        self._task: asyncio.Task | None = None
        self._cancel = False
        self.current_run: int | None = None

    async def start(self) -> None:
        self.ctx.db.execute("UPDATE ml_runs SET status = 'stopped', message = 'interrupted by restart' "
                            "WHERE status = 'running'")

    async def stop(self) -> None:
        self._cancel = True
        if self._task:
            self._task.cancel()

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def launch(self, params: TrainParams) -> int:
        if self.running:
            raise RuntimeError("Training is already running.")
        bt = self.ctx.service("backtest")
        if bt is not None and getattr(bt, "running", False):
            raise RuntimeError("Wait for the backtest to finish first.")
        run_id = self.ctx.db.insert("ml_runs", {"created_at": iso(), "params": json.dumps(asdict(params), default=str),
                                                "status": "running", "progress": 0, "message": "starting..."})
        self._cancel = False
        self.current_run = run_id
        self._task = asyncio.create_task(self._guarded(run_id, params), name=f"ml-train-{run_id}")
        return run_id

    def cancel(self) -> None:
        self._cancel = True

    def _progress(self, run_id: int, progress: float, message: str, **extra) -> None:
        self.ctx.db.update("ml_runs", run_id, {"progress": round(progress, 3), "message": message, **extra})
        self.ctx.bus.publish("ml_training", {"id": run_id, "progress": progress, "message": message,
                                             **{k: v for k, v in extra.items() if k != "report"}})

    async def _guarded(self, run_id: int, params: TrainParams) -> None:
        try:
            await self.run(run_id, params)
        except (asyncio.CancelledError, Cancelled):
            self._progress(run_id, 1, "cancelled", status="cancelled")
        except Exception as exc:
            log.warning("Model training %s failed: %s", run_id, exc)
            self._progress(run_id, 1, f"failed: {exc}", status="failed")

    async def run(self, run_id: int, params: TrainParams) -> dict:
        ctx = self.ctx
        trader = ctx.service("trader")
        broker = getattr(trader, "broker", None)
        pipeline = ctx.service("pipeline")
        if broker is None:
            raise RuntimeError("Not connected to Alpaca - training downloads history with your Alpaca keys.")
        if pipeline is None or not pipeline.tickers.loaded:
            raise RuntimeError("Ticker list not loaded yet - wait a minute after adding Alpaca keys.")
        engine = pipeline.local
        loop = asyncio.get_running_loop()

        def cancelled() -> bool:
            return self._cancel

        def stage(lo: float, hi: float):
            def report(frac: float, msg: str) -> None:
                p = lo + (hi - lo) * max(0.0, min(1.0, frac))
                loop.call_soon_threadsafe(self._progress, run_id, p, msg)
            return report

        self._progress(run_id, 0.01, "loading the sentiment model...")
        await engine.ensure_ready()
        if params.redownload:
            forget_cached_days(ctx.db, params.horizon_minutes)
        stats = await asyncio.to_thread(build_dataset, ctx.db, broker, pipeline.tickers, params.start, params.end,
                                        params.horizon_minutes, params.max_articles, stage(0.02, 0.6), cancelled,
                                        self.throttle_factory())
        samples, counts = await asyncio.to_thread(load_samples, ctx.db, pipeline.tickers, engine.sentiment,
                                                  params.horizon_minutes, params.start, params.end,
                                                  params.min_move_pct, stage(0.6, 0.8), cancelled)
        if cancelled():
            raise Cancelled()
        self._progress(run_id, 0.8, f"training on {len(samples):,} headlines...")
        model, report = await asyncio.to_thread(
            train_price_model, samples, params.horizon_minutes, params.min_move_pct, engine.sentiment.model_id,
            (params.start.isoformat(), params.end.isoformat()), stage(0.8, 0.97))
        if cancelled():
            raise Cancelled()
        report.update({"download": stats, "counts": counts})
        model.meta["report"] = report
        await asyncio.to_thread(model.save, engine.model_dir)
        engine.set_price_model(model)
        pipeline.update_ml_status()
        if report["passed"]:
            msg = (f"done: tested on {report['n_test']:,} unseen headlines - picked the right direction "
                   f"{report['accuracy']}% of the time. The live engine now uses it.")
        else:
            msg = (f"done, but the model isn't used: {report['why_not']}. "
                   "Try more days of history; sentiment-only scoring continues meanwhile.")
        self._progress(run_id, 1, msg, status="done", report=json.dumps(report, default=str))
        ctx.bus.publish("toast", {"kind": "success" if report["passed"] else "warn",
                                  "title": "Local ML model trained", "message": msg})
        return report

    def delete_model(self) -> None:
        if self.running:
            raise RuntimeError("Training is running - cancel it first.")
        pipeline = self.ctx.service("pipeline")
        folder = pipeline.local.model_dir if pipeline else paths.models_dir() / "price_model"
        delete_model(folder)
        if pipeline:
            pipeline.local.set_price_model(None)
            pipeline.update_ml_status()
