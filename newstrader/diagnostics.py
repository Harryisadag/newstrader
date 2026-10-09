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

MAC = sys.platform == "darwin"
UPDATE = "update.command" if MAC else "update.bat"


def _update_hint() -> str:
    """How to get newer bundled tools: the downloaded app can only be replaced by a newer release."""
    if paths.is_frozen():
        return "Get the newest NewsTrader release (Update now in the banner when one is out, or see RELEASES.md on GitHub)"
    return f"Run {UPDATE}"


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
        _anthropic_key_check(ctx),
        _check("Discord webhook", "ok" if k.has_discord else "warn",
               "set" if k.has_discord else "not set (Discord alerts off)",
               "" if k.has_discord else "Optional. Settings -> API Keys."),
    ]
    out.append(_check(".env file", "ok" if env.exists() else "warn", str(env),
                      "" if env.exists() else "It will be created when you save keys in Settings -> API Keys."))
    return out


def _anthropic_key_check(ctx: AppContext) -> dict:
    if ctx.keys.keys.has_anthropic:
        return _check("Anthropic API key", "ok", "set")
    if ctx.config.settings.ai.engine == "local":
        return _check("Anthropic API key", "ok", "not set - not needed (the free local ML engine is selected)")
    return _check("Anthropic API key", "error", "missing",
                  "Settings -> API Keys (create one at platform.claude.com), or switch Settings -> AI engine "
                  "to the free local engine.")


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
        if ctx.config.settings.ai.engine == "local":
            return _check("Claude API", "ok", "skipped - not needed for the local ML engine")
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
    mac = info.get("mac")
    if mac:
        from .system_monitor import describe_gpu

        level, detail = describe_gpu(info)
        if level == "ok":
            fix = ""
        elif not mac.get("apple_silicon"):
            fix = "Pick 'small' or 'base' in Settings -> Transcription."
        elif paths.is_frozen():
            fix = f"{_update_hint()}."
        else:
            fix = "Run run.command again (it installs the Apple-GPU engine on macOS 14+)."
        return _check("Speech-to-text hardware", level, f"{detail} · macOS {mac.get('macos') or '?'}", fix)
    want_cuda = ctx.config.settings.transcription.device in ("cuda", "auto")
    if info["cuda_devices"] and info["cuda_devices"] > 0:
        detail = f"{info['name'] or 'NVIDIA GPU'} (driver {info['driver'] or '?'})"
        try:
            import ctranslate2

            types = sorted(ctranslate2.get_supported_compute_types("cuda"))
            detail += f", compute types: {', '.join(types)}"
        except Exception as exc:
            return _check("GPU (CUDA)", "error", f"{detail}; CUDA libraries failed to load: {exc}",
                          "Update your NVIDIA driver. If it still fails, download the newest NewsTrader release."
                          if paths.is_frozen() else
                          f"Run {UPDATE} to reinstall nvidia-cublas-cu12, and update your NVIDIA driver.")
        return _check("GPU (CUDA)", "ok", detail)
    if not info.get("name"):  # no NVIDIA card at all - the CPU is the expected path, not a fault
        return _check("GPU (CUDA)", "warn", info.get("error") or "No NVIDIA GPU found",
                      "TV transcription runs on the CPU with the 'small' model (slower). Nothing to fix "
                      "unless this PC has an NVIDIA card.")
    status = "error" if want_cuda else "warn"
    detail = info.get("error") or "No CUDA GPU detected"
    if info.get("name"):
        detail = f"{info['name']} found by nvidia-smi, but CTranslate2 can't use it. {detail}"
    return _check("GPU (CUDA)", status, detail,
                  "Update the NVIDIA driver (RTX 50-series needs a 2025+ driver). Transcription falls back to CPU (slow).")


def _check_ffmpeg() -> dict:
    exe = find_ffmpeg()
    if not exe:
        return _check("ffmpeg", "error", "not found",
                      f"{_update_hint()}, or install it with: " + ("brew install ffmpeg" if MAC else "winget install Gyan.FFmpeg"))
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
        out.append(_check("yt-dlp", "error", str(exc), f"{_update_hint()}."))
    deno = find_deno()
    out.append(_check("Deno (YouTube JavaScript runtime)", "ok" if deno else "warn", deno or "not found",
                      "" if deno else f"{_update_hint()}. Without it some YouTube streams may fail."))
    return out


def _check_ml(ctx: AppContext) -> dict:
    if ctx.config.settings.ai.engine != "local":
        return _check("Local ML engine", "ok", "not in use (Claude engine selected)")
    pipeline = ctx.service("pipeline")
    if pipeline is None:
        return _check("Local ML engine", "warn", "AI engine not running")
    st = pipeline.local.model_status()
    pm = st["price_model"]
    trained = ("price model active" if st["price_model_active"] else
               f"price model not used ({pm['why_inactive']})" if pm else "price model not trained yet")
    if st["sentiment_error"]:
        return _check("Local ML engine", "warn", f"FinBERT unavailable ({st['sentiment_error']}); using the built-in "
                      f"word list · {trained}",
                      "Check your internet connection (one-time ~110 MB download from huggingface.co), then click "
                      "'Retry FinBERT download' in Backtest -> Local ML model.")
    return _check("Local ML engine", "ok", f"Sentiment: {st['sentiment']} · {trained}",
                  "" if st["price_model_active"] else "Optional: Backtest tab -> Local ML model -> Train.")


async def _check_pro_ai(ctx: AppContext) -> list[dict]:
    """Only when Pro AI is turned on: its downloads, its server, and whether the whole model is on the graphics
    card."""
    if not ctx.config.settings.ai.pro_ai.enabled:
        return []
    from .llm.catalog import MODELS_BY_KEY, SERVER_BUILDS

    pro = ctx.service("pro_ai")
    if pro is None:
        return [_check("Pro AI", "warn", "turned on, but the engine isn't running yet")]
    setup = "Settings -> Pro AI -> Set up Pro AI."
    build, model, why = await asyncio.to_thread(pro.chosen)
    if build is None or model is None:
        return [_check("Pro AI files", "error", why or "no model chosen", setup)]
    exe = await asyncio.to_thread(pro.files.installed_server_exe, build)
    path = await asyncio.to_thread(pro.files.model_path, model)
    names = f"{MODELS_BY_KEY[model].label} on the {SERVER_BUILDS[build].label} server"
    if exe and path:
        out = [_check("Pro AI files", "ok", f"{names}: downloaded, size and SHA-256 checked")]
    else:
        missing = " and ".join(x for x, ok in (("the server", exe), (MODELS_BY_KEY[model].label, path)) if not ok)
        return [_check("Pro AI files", "error", f"{missing} not downloaded (or it failed its safety check)", setup)]
    srv = pro.server
    code = await asyncio.to_thread(srv.health) if pro.state == "ready" else None
    if pro.state == "ready" and code == 200:
        out.append(_check("Pro AI server", "ok", "running and answering (only this computer can reach it)"))
    elif pro.state in ("starting", "downloading"):
        out.append(_check("Pro AI server", "warn", pro.message, "Wait a minute and run diagnostics again."))
    else:
        detail = pro.message if pro.state != "ready" else f"not answering (HTTP {code})"
        out.append(_check("Pro AI server", "error", detail,
                          "Settings -> Pro AI -> Start. If it keeps stopping, pick a smaller model."))
        return out
    on_gpu = srv.fully_on_gpu if pro.state == "ready" else None
    if on_gpu is True:
        out.append(_check("Pro AI on the graphics card", "ok", f"the whole model is on the card ({srv.gpu_layers} "
                                                               "layers)"))
    elif on_gpu is False:
        out.append(_check("Pro AI on the graphics card", "warn",
                          f"only {srv.gpu_layers} of {srv.total_layers} layers fit - the rest runs on the processor, "
                          "which is much slower",
                          "Pick a smaller model in Settings -> Pro AI, close programs that use the graphics card, "
                          "or turn off TV transcription."))
    elif pro.state == "ready":
        out.append(_check("Pro AI on the graphics card", "warn", "can't tell (this server build runs on the "
                          "processor, or doesn't say)", "Settings -> Pro AI -> Test my PC shows how fast it is."))
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
    checks.append(_check_ml(ctx))
    try:
        checks += await _with_timeout(_check_pro_ai(ctx), 30)
    except Exception as exc:
        checks.append(_check("Pro AI", "error", f"{type(exc).__name__}: {exc}"))
    summary = {s: sum(1 for c in checks if c["status"] == s) for s in ("ok", "warn", "error")}
    log.info("Diagnostics: %d ok, %d warnings, %d errors", summary["ok"], summary["warn"], summary["error"])
    return {"checks": checks, "summary": summary}
