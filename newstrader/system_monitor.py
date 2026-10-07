"""Keeps a live "GPU" status card in Logs -> System status (name, load, memory - or the Apple chip on a Mac),
refreshed every 30 seconds."""

from __future__ import annotations

import asyncio
import contextlib
import logging

from .context import AppContext
from .tools import gpu_info

log = logging.getLogger(__name__)


def describe_gpu(info: dict) -> tuple[str, str]:
    mac = info.get("mac")
    if mac:
        chip = mac.get("chip") or ("Apple Silicon" if mac.get("apple_silicon") else "Intel Mac")
        if mac.get("apple_silicon") and mac.get("mlx"):
            return "ok", f"{chip} - speech-to-text runs on the Apple GPU (MLX)"
        if mac.get("apple_silicon"):
            return "warn", (f"{chip} - the Apple-GPU speech engine isn't installed, so speech-to-text uses the CPU "
                            "(slower). Run run.command again to install it.")
        return "warn", f"{chip} - Intel Macs transcribe on the CPU; use a small Whisper model (Settings)"
    if info.get("cuda_devices"):
        parts = [info.get("name") or "NVIDIA GPU"]
        if info.get("utilization_pct") is not None:
            parts.append(f"{info['utilization_pct']:.0f}% busy")
        if info.get("memory_total_mb"):
            parts.append(f"{(info.get('memory_used_mb') or 0) / 1024:.1f}/{info['memory_total_mb'] / 1024:.0f} GB memory")
        if info.get("driver"):
            parts.append(f"driver {info['driver']}")
        return "ok", " · ".join(parts)
    if info.get("name"):
        return "error", f"{info['name']} found, but CUDA isn't usable - run Logs -> Run diagnostics"
    return "warn", "No NVIDIA GPU detected - transcription will use the CPU (slow)"


class SystemMonitor:
    name = "system"

    def __init__(self, ctx: AppContext, interval: float = 30):
        self.ctx = ctx
        self.interval = interval
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._loop(), name="system-monitor")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task

    async def _loop(self) -> None:
        while True:
            try:
                level, detail = describe_gpu(await asyncio.to_thread(gpu_info))
                self.ctx.state.set_status("gpu", level, detail)
            except Exception:
                log.debug("gpu status failed", exc_info=True)
            await asyncio.sleep(self.interval)
