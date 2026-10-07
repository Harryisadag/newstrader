"""Locate external programs (ffmpeg, deno) - prefer ones already installed, else the pip-bundled copies."""

from __future__ import annotations

import functools
import logging
import shutil
import subprocess
import sys

log = logging.getLogger(__name__)

# Hide console windows for child processes on Windows (ffmpeg, nvidia-smi...)
CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


@functools.lru_cache(maxsize=1)
def find_ffmpeg() -> str | None:
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # package missing or binary not shipped for this platform
        log.debug("imageio-ffmpeg not usable: %s", exc)
        return None


@functools.lru_cache(maxsize=1)
def find_deno() -> str | None:
    path = shutil.which("deno")
    if path:
        return path
    try:
        import deno

        return deno.find_deno_bin()
    except Exception as exc:
        log.debug("deno package not usable: %s", exc)
        return None


def run_quiet(args: list[str], timeout: float = 10) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout,
                          creationflags=CREATE_NO_WINDOW, check=False)


def gpu_info() -> dict:
    """Best-effort GPU description using nvidia-smi and CTranslate2. Never raises."""
    info: dict = {"cuda_devices": 0, "name": None, "driver": None, "memory_total_mb": None,
                  "memory_used_mb": None, "utilization_pct": None, "error": None}
    try:
        import ctranslate2

        info["cuda_devices"] = ctranslate2.get_cuda_device_count()
    except Exception as exc:
        info["error"] = f"CTranslate2: {exc}"
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = run_quiet([smi, "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu",
                             "--format=csv,noheader,nounits"], timeout=5)
            line = out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                info.update(name=parts[0], driver=parts[1], memory_total_mb=_num(parts[2]),
                            memory_used_mb=_num(parts[3]), utilization_pct=_num(parts[4]))
        except Exception as exc:
            info["error"] = (info["error"] or "") + f" nvidia-smi: {exc}"
    return info


def _num(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None
