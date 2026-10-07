"""Starts and stops every background service, in order, inside the server's event loop."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Protocol

from .context import AppContext

log = logging.getLogger(__name__)


class Service(Protocol):
    name: str

    async def start(self) -> None: ...

    async def stop(self) -> None: ...


class Orchestrator:
    def __init__(self, ctx: AppContext):
        self.ctx = ctx
        self.services: list[Service] = []
        self._tasks: list[asyncio.Task] = []
        self.started = False

    def add(self, service: Service) -> Service:
        self.services.append(service)
        self.ctx.services[service.name] = service
        return service

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        self.ctx.loop = loop
        self.ctx.bus.bind_loop(loop)
        self.ctx.services["orchestrator"] = self
        self._build_services()
        for svc in self.services:
            try:
                await svc.start()
                log.info("Started %s", svc.name)
            except Exception:
                log.exception("Failed to start %s", svc.name)
                self.ctx.state.set_status(svc.name, "error", "failed to start - see Logs")
        self._tasks.append(asyncio.create_task(self._heartbeat(), name="heartbeat"))
        self.started = True
        log.info("NewsTrader engine started in %s mode", self.ctx.state.mode.upper())

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._tasks.clear()
        for svc in reversed(self.services):
            try:
                await asyncio.wait_for(svc.stop(), timeout=10)
            except Exception:
                log.exception("Error stopping %s", svc.name)
        self.started = False
        log.info("NewsTrader engine stopped")

    def _build_services(self) -> None:
        """Create every background service, in start order."""
        if self.services:
            return
        from .ai.pipeline import Pipeline
        from .alerts.manager import AlertManager
        from .audio.stream_manager import StreamManager
        from .backtest.runner import BacktestRunner
        from .ml.trainer import ModelTrainer
        from .performance.tracker import PerformanceTracker
        from .sources.manager import SourceManager
        from .system_monitor import SystemMonitor
        from .trading.trader import Trader

        self.add(SystemMonitor(self.ctx))
        self.add(AlertManager(self.ctx))
        self.add(Trader(self.ctx))
        self.add(Pipeline(self.ctx))
        self.add(SourceManager(self.ctx))
        self.add(StreamManager(self.ctx))
        self.add(PerformanceTracker(self.ctx))
        self.add(BacktestRunner(self.ctx))
        self.add(ModelTrainer(self.ctx))

    async def _heartbeat(self) -> None:
        from .api.routes.status import status_summary

        while True:
            try:
                self.ctx.bus.publish("heartbeat", await status_summary(self.ctx, light=True))
            except Exception:
                log.debug("heartbeat failed", exc_info=True)
            await asyncio.sleep(5)
