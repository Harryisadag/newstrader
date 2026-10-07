"""One-click health check: tests every connection and tool and explains how to fix anything red."""

from __future__ import annotations

import asyncio
import logging
import platform
import sys
import tempfile

import httpx

from . import __version__, paths
from .context import AppContext
from .tools import find_deno, find_ffmpeg, gpu_info, run_quiet

log = logging.getLogger(__name__)


def _check(name: str, status: str, detail: str, fix: str = "") -> dict:
    # status: ok | warn | error
    return {"name": name, "status": status, "detail": detail, "fix": fix}


async def _with_timeout(coro, seconds: float = 20):
    return await asyncio.wait_for(coro, timeout=seconds)


def _check_python() -> dict:
    v = sys.version_info
    text = f"Python {v.major}.{v.minor}.{v.micro} ({platform.architecture()[0]}) on {platform.system()} {platform.release()}"
    if v < (3, 11):
        return _check("Python", "error", text, "Install Python 3.12 (see README step 1).")
    if (v.major, v.minor) != (3, 12):
        return _check("Python", "warn", text, "Python 3.12 is the tested version; others may work.")
    return _check("Python", "ok", text)


def _check_data_dir() -> dict:
    d = paths.data_dir()
    try:
        with tempfile.NamedTemporaryFile(dir=d, delete=True):
            pass
        return _check("Data folder", "ok", str(d))
    except OSError as exc:
        return _check("Data folder", "error", f"{d}: {exc}", "Make sure the folder isn't read-only.")


def _check_keys(ctx: AppContext) -> list[dict]:
    k = ctx.keys.keys
    env = paths.env_file()
    out = [
        _check("Alpaca paper keys", "ok" if k.has_alpaca_paper else "error",
               "set" if k.has_alpaca_paper else "missing",
               "" if k.has_alpaca_paper else "Settings -> API Keys. Get them at app.alpaca.markets (Paper account)."),
        _check("Anthropic API key", "ok" if k.has_anthropic else "error",
               "set" if k.has_anthropic else "missing",
               "" if k.has_anthropic else "Settings -> API Keys. Create one at platform.claude.com."),
        _check("Discord webhook", "ok" if k.has_discord else "warn",
               "set" if k.has_discord else "not set (Discord alerts off)",
               "" if k.has_discord else "Optional. Settings -> API Keys."),
    ]
    out.append(_check(".env file", "ok" if env.exists() else "warn", str(env),
                      "" if env.exists() else "It will be created when you save keys in Settings -> API Keys."))
    return out


async def _check_alpaca(ctx: AppContext) -> dict:
    k = ctx.keys.keys
    if not k.has_alpaca_paper:
        return _check("Alpaca connection", "error", "skipped - no paper keys")

    def call():
        from alpaca.trading.client import TradingClient

        client = TradingClient(k.alpaca_paper_key, k.alpaca_paper_secret, paper=True)
        acct = client.get_account()
        clock = client.get_clock()
        return acct, clock

    try:
        acct, clock = await _with_timeout(asyncio.to_thread(call))
        market = "open" if clock.is_open else "closed"
        return _check("Alpaca connection", "ok",
                      f"Paper account {acct.account_number}: equity ${float(acct.equity):,.2f}, market {market}")
    except Exception as exc:
        return _check("Alpaca connection", "error", f"{type(exc).__name__}: {exc}",
                      "Check the paper keys (they're different from live keys) and your internet connection.")


async def _check_anthropic(ctx: AppContext) -> dict:
    k = ctx.keys.keys
    model = ctx.config.settings.ai.model
    if not k.has_anthropic:
        return _check("Claude API", "error", "skipped - no API key")
    try:
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=k.anthropic, max_retries=1, timeout=20)
        info = await _with_timeout(client.models.retrieve(model))
        return _check("Claude API", "ok", f"Key works; model {info.id} available")
    except Exception as exc:
        return _check("Claude API", "error", f"{type(exc).__name__}: {exc}",
                      "Check the key at platform.claude.com, that you have credits, and the model name in Settings -> AI.")


async def _check_discord(ctx: AppContext) -> dict:
    k = ctx.keys.keys
    if not k.has_discord:
        return _check("Discord webhook check", "warn", "skipped - no webhook URL")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(k.discord_webhook)
        if r.status_code == 200:
            name = r.json().get("name", "webhook")
            return _check("Discord webhook check", "ok", f"Webhook '{name}' found")
        return _check("Discord webhook check", "error", f"HTTP {r.status_code}",
                      "Copy the webhook URL again from Discord (Server Settings -> Integrations -> Webhooks).")
    except Exception as exc:
        return _check("Discord webhook check", "error", f"{type(exc).__name__}: {exc}")


def _check_gpu(ctx: AppContext) -> dict:
    from .audio.cuda_setup import setup_cuda_dll_paths

    setup_cuda_dll_paths()
    info = gpu_info()
    want_cuda = ctx.config.settings.transcription.device in ("cuda", "auto")
    if info["cuda_devices"] and info["cuda_devices"] > 0:
        detail = f"{info['name'] or 'NVIDIA GPU'} (driver {info['driver'] or '?'})"
        try:
            import ctranslate2

            types = sorted(ctranslate2.get_supported_compute_types("cuda"))
            detail += f", compute types: {', '.join(types)}"
        except Exception as exc:
            return _check("GPU (CUDA)", "error", f"{detail}; CUDA libraries failed to load: {exc}",
                          "Run update.bat to reinstall nvidia-cublas-cu12, and update your NVIDIA driver.")
        return _check("GPU (CUDA)", "ok", detail)
    status = "error" if want_cuda else "warn"
    detail = info.get("error") or "No CUDA GPU detected"
    if info.get("name"):
        detail = f"{info['name']} found by nvidia-smi, but CTranslate2 can't use it. {detail}"
    return _check("GPU (CUDA)", status, detail,
                  "Update the NVIDIA driver (RTX 50-series needs a 2025+ driver). Transcription falls back to CPU (slow).")


def _check_ffmpeg() -> dict:
    exe = find_ffmpeg()
    if not exe:
        return _check("ffmpeg", "error", "not found", "Run update.bat, or install with: winget install Gyan.FFmpeg")
    try:
        out = run_quiet([exe, "-hide_banner", "-version"], timeout=10)
        first = (out.stdout or out.stderr).splitlines()[0] if (out.stdout or out.stderr) else "unknown version"
        return _check("ffmpeg", "ok", f"{first} ({exe})")
    except Exception as exc:
        return _check("ffmpeg", "error", f"{exe}: {exc}")


def _check_ytdlp() -> list[dict]:
    out = []
    try:
        import yt_dlp

        out.append(_check("yt-dlp", "ok", f"version {yt_dlp.version.__version__}"))
    except Exception as exc:
        out.append(_check("yt-dlp", "error", str(exc), "Run update.bat."))
    deno = find_deno()
    out.append(_check("Deno (YouTube JavaScript runtime)", "ok" if deno else "warn", deno or "not found",
                      "" if deno else "Run update.bat. Without it some YouTube streams may fail."))
    return out


async def run_diagnostics(ctx: AppContext) -> dict:
    checks: list[dict] = [_check("NewsTrader", "ok", f"version {__version__}, {ctx.state.mode.upper()} mode"),
                          _check_python(), _check_data_dir()]
    checks += _check_keys(ctx)
    net = await asyncio.gather(_check_alpaca(ctx), _check_anthropic(ctx), _check_discord(ctx),
                               return_exceptions=True)
    for item in net:
        checks.append(item if isinstance(item, dict) else _check("Network check", "error", str(item)))
    checks.append(await asyncio.to_thread(_check_gpu, ctx))
    checks.append(await asyncio.to_thread(_check_ffmpeg))
    checks += await asyncio.to_thread(_check_ytdlp)
    summary = {s: sum(1 for c in checks if c["status"] == s) for s in ("ok", "warn", "error")}
    log.info("Diagnostics: %d ok, %d warnings, %d errors", summary["ok"], summary["warn"], summary["error"])
    return {"checks": checks, "summary": summary}
