"""Starts/stops one background task per enabled text source (RSS, social, X, Alpaca news) and keeps them
in sync with Settings -> News sources. Live streams are handled by audio/stream_manager.py.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging

import httpx

from ..config import SourceConfig
from ..context import AppContext
from .alpaca_news import AlpacaNewsSource
from .rss import FeedPoller
from .x_api import XAccountSource

log = logging.getLogger(__name__)

TEXT_TYPES = ("rss", "social_rss", "x_account", "alpaca_news")


class SourceManager:
    name = "sources"

    def __init__(self, ctx: AppContext):
        self.ctx = ctx
        self.tasks: dict[str, asyncio.Task] = {}
        self.configs: dict[str, dict] = {}
        self.client: httpx.AsyncClient | None = None
        self._watch: asyncio.Task | None = None
        self._changed = asyncio.Event()

    async def start(self) -> None:
        self.client = httpx.AsyncClient(http2=False, follow_redirects=True,
                                        limits=httpx.Limits(max_connections=20))
        self.ctx.config.on_change(lambda _s: self._signal_change())
        await self.reconcile()
        self._watch = asyncio.create_task(self._watch_loop(), name="sources-watch")

    async def stop(self) -> None:
        if self._watch:
            self._watch.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._watch
        for sid in list(self.tasks):
            await self._stop_task(sid)
        if self.client:
            await self.client.aclose()

    async def on_keys_changed(self) -> None:
        # credentials changed: restart the sources that use them
        for sid, cfg in list(self.configs.items()):
            if cfg["type"] in ("alpaca_news", "x_account"):
                await self._stop_task(sid)
        await self.reconcile()

    def _signal_change(self) -> None:
        loop = self.ctx.loop
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(self._changed.set)

    async def _watch_loop(self) -> None:
        while True:
            await self._changed.wait()
            self._changed.clear()
            await asyncio.sleep(0.3)  # let rapid toggles settle
            try:
                await self.reconcile()
            except Exception:
                log.exception("source reconcile failed")

    async def reconcile(self) -> None:
        wanted = {s.id: s for s in self.ctx.config.settings.sources if s.type in TEXT_TYPES and s.enabled}
        for sid in list(self.tasks):
            if sid not in wanted or self.configs.get(sid) != wanted[sid].model_dump():
                await self._stop_task(sid)
        for sid, src in wanted.items():
            if sid not in self.tasks:
                self._start_task(src)
        for src in self.ctx.config.settings.sources:
            if src.type in TEXT_TYPES and not src.enabled:
                self.ctx.state.set_status(f"source:{src.id}", "off", "turned off")

    def _status_setter(self, src: SourceConfig):
        def set_status(level: str, detail: str) -> None:
            self.ctx.state.set_status(f"source:{src.id}", level, detail, name=src.name, type=src.type)
        return set_status

    def _start_task(self, src: SourceConfig) -> None:
        pipeline = self.ctx.service("pipeline")
        if pipeline is None:
            return
        set_status = self._status_setter(src)
        keys = self.ctx.keys.keys
        if src.type in ("rss", "social_rss"):
            runner = FeedPoller(src, pipeline.submit, set_status, self.client).run()
        elif src.type == "x_account":
            runner = XAccountSource(src, pipeline.submit, set_status, keys.x_bearer, self.client).run()
        elif src.type == "alpaca_news":
            # news data works with either paper or live keys; paper keys are always preferred
            key, secret = keys.alpaca_paper_key, keys.alpaca_paper_secret
            if not (key and secret):
                set_status("error", "needs Alpaca keys (Settings -> API keys)")
                return
            runner = AlpacaNewsSource(src, pipeline.submit, set_status, key, secret).run()
        else:
            return
        self.tasks[src.id] = asyncio.create_task(self._guard(src, runner), name=f"source-{src.id}")
        self.configs[src.id] = src.model_dump()

    async def _guard(self, src: SourceConfig, runner) -> None:
        try:
            await runner
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("Source %s stopped: %s", src.name, exc)
            self.ctx.state.set_status(f"source:{src.id}", "error", f"stopped: {exc}")

    async def _stop_task(self, sid: str) -> None:
        task = self.tasks.pop(sid, None)
        self.configs.pop(sid, None)
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
