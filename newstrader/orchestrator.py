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
        from .trading.trader import Trader

        self.add(Trader(self.ctx))

    async def _heartbeat(self) -> None:
        from .api.routes.status import status_summary

        while True:
            try:
                self.ctx.bus.publish("heartbeat", await status_summary(self.ctx, light=True))
            except Exception:
                log.debug("heartbeat failed", exc_info=True)
            await asyncio.sleep(5)
