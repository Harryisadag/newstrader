"""Sends alerts to the in-app toast area, Windows desktop pop-ups and Discord, following Settings -> Alerts.

Alert kinds: trade_placed, trade_filled, manual_review, error, daily_loss_limit, spend_cap, kill_switch.
Errors logged anywhere in the app are also turned into "error" alerts (throttled so you don't get spammed).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import deque

import httpx

from ..context import AppContext
from ..db import iso
from .desktop import desktop_supported, show_desktop
from .discord import build_payload, send_discord

log = logging.getLogger(__name__)

KIND_SETTING = {
    "trade_placed": "on_trade_placed",
    "trade_filled": "on_trade_filled",
    "manual_review": "on_manual_review",
    "error": "on_error",
    "daily_loss_limit": "on_daily_loss_limit",
    "spend_cap": "on_spend_cap",
    "kill_switch": "on_kill_switch",
}
TOAST_KIND = {"trade": "trade", "error": "error", "warn": "warn", "success": "success", "info": "info"}
DEDUPE_SECONDS = 60
ERROR_THROTTLE_SECONDS = 300


class _ErrorLogForwarder(logging.Handler):
    def __init__(self, manager: AlertManager):
        super().__init__(level=logging.ERROR)
        self.manager = manager
        self._last: dict[str, float] = {}

    def emit(self, record: logging.LogRecord) -> None:
        if not record.name.startswith("newstrader") or record.name.startswith("newstrader.alerts"):
            return
        try:
            msg = record.getMessage()
            key = f"{record.name}:{msg[:60]}"
            now = time.monotonic()
            if now - self._last.get(key, 0) < ERROR_THROTTLE_SECONDS:
                return
            self._last[key] = now
            loop = self.manager.ctx.loop
            if loop is not None and not loop.is_closed():
                loop.call_soon_threadsafe(lambda: asyncio.ensure_future(
                    self.manager.send("error", "NewsTrader error", f"{msg}\n({record.name})", "error")))
        except Exception:
            pass


class AlertManager:
    name = "alerts"

    def __init__(self, ctx: AppContext, http_client: httpx.AsyncClient | None = None, desktop=show_desktop):
        self.ctx = ctx
        self._client = http_client
        self._own_client = http_client is None
        self._desktop = desktop
        self._recent: dict[str, float] = {}
        self.history: deque[dict] = deque(maxlen=200)
        self._handler: _ErrorLogForwarder | None = None

    async def start(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient()
        self._handler = _ErrorLogForwarder(self)
        logging.getLogger().addHandler(self._handler)
        self._update_status()

    async def stop(self) -> None:
        if self._handler is not None:
            logging.getLogger().removeHandler(self._handler)
        if self._own_client and self._client is not None:
            await self._client.aclose()

    async def on_keys_changed(self) -> None:
        self._update_status()

    def _update_status(self) -> None:
        k = self.ctx.keys.keys
        detail = ["desktop pop-ups ready" if desktop_supported() else "desktop pop-ups: Windows only",
                  "Discord ready" if k.has_discord else "Discord: no webhook URL"]
        self.ctx.state.set_status("alerts", "ok" if k.has_discord or desktop_supported() else "warn", " · ".join(detail))

    def _enabled(self, kind: str) -> bool:
        a = self.ctx.config.settings.alerts
        attr = KIND_SETTING.get(kind)
        return True if attr is None else bool(getattr(a, attr))

    async def send(self, kind: str, title: str, message: str = "", level: str = "info", force: bool = False,
                   fields: dict | None = None) -> dict:
        """Returns what was delivered: {"toast": bool, "desktop": bool, "discord": bool|str}."""
        result = {"toast": True, "desktop": False, "discord": False, "skipped": None}
        self.ctx.bus.publish("toast", {"kind": TOAST_KIND.get(level, "info"), "title": title, "message": message})
        self.history.appendleft({"ts": iso(), "kind": kind, "title": title, "message": message, "level": level})

        if not force:
            if not self._enabled(kind):
                result["skipped"] = "this alert type is turned off"
                return result
            key = f"{kind}:{title}"
            now = time.monotonic()
            if now - self._recent.get(key, 0) < DEDUPE_SECONDS:
                result["skipped"] = "duplicate within 60s"
                return result
            self._recent[key] = now
            if len(self._recent) > 500:
                self._recent = {k: v for k, v in self._recent.items() if now - v < DEDUPE_SECONDS}

        settings = self.ctx.config.settings.alerts
        tasks = []
        if settings.desktop_enabled or force:
            tasks.append(("desktop", asyncio.to_thread(self._desktop, title, message, level)))
        k = self.ctx.keys.keys
        if (settings.discord_enabled or force) and k.has_discord and self._client is not None:
            payload = build_payload(title, message, level, self.ctx.state.mode, fields)
            tasks.append(("discord", send_discord(self._client, k.discord_webhook, payload)))
        if tasks:
            outs = await asyncio.gather(*(t for _, t in tasks), return_exceptions=True)
            for (name, _), out in zip(tasks, outs, strict=True):
                if isinstance(out, Exception):
                    result[name] = f"failed: {out}"
                elif name == "discord":
                    ok, why = out
                    result["discord"] = True if ok else why
                    if not ok:
                        self.ctx.state.set_status("alerts", "warn", f"Discord: {why}")
                else:
                    result[name] = bool(out)
        return result

    async def test(self) -> dict:
        res = await self.send("test", "Test alert from NewsTrader",
                              "If you can read this, alerts are working.", "success", force=True)
        with contextlib.suppress(Exception):
            if res.get("discord") is True:
                self.ctx.state.set_status("alerts", "ok", "Discord test sent")
        return res
