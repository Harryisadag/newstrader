"""Pro AI inside the app: setting it up (the one-time download), the llama-server it runs on, a small queue of
stories for it, and "Test my PC". Registered in the orchestrator as "pro_ai".

States: off (turned off) / not_set_up (files missing) / downloading / starting / ready / error, each with a plain
message. Every change is pushed to the window as a "pro_ai" event and shown on the Logs tab's status grid.

Nothing here places a trade: Pro AI's answers go to Pipeline.record_pro, which stores them as watch-only signals or,
in judge mode, sends signals to manual review.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import platform
import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

from ..ai.prefilter import PrefilterResult
from ..db import iso
from ..sources.base import NewsItem
from . import download, hardware, selftest
from .catalog import MODELS, MODELS_BY_KEY, NOT_AUTO, SERVER_BUILDS
from .engine import ProAIEngine
from .server import LlamaServer

log = logging.getLogger(__name__)

SLOTS = 2  # stories the server reads at once (the catalog's memory sizes allow for two)
CONTEXT = 8192  # shared by the slots
QUEUE_MAX = 50
READY_TIMEOUT = 300.0
WATCH_EVERY = 5.0  # seconds between checks that the server is still alive
INSTALLED_KEY = "pro_ai_installed"  # what the last set-up downloaded: {"build", "model"}
SELFTEST_KEY = "pro_ai_selftest"
LEVELS = {"ready": "ok", "starting": "starting", "downloading": "starting", "error": "error", "not_set_up": "warn",
          "off": "off"}


@dataclass
class ProJob:
    """One story waiting for Pro AI."""

    item: NewsItem
    pre: PrefilterResult
    reason: str  # why it was sent (routing.py)
    judge: bool = False  # judge mode on a hard case: may raise a manual-review signal the main engine missed
    main_engine: str | None = None
    main_analysis_id: int | None = None
    main_calls: dict[str, str] | None = None  # the main engine's {ticker: direction}; None = it didn't read it
    urgent: bool = False  # a main-engine signal waits for this answer (judge mode)
    queued_at: float = field(default_factory=time.monotonic)
    future: asyncio.Future | None = None


def _gb(n: float) -> str:
    return f"{n / 1e9:.1f} GB"


class ProAIService:
    name = "pro_ai"

    def __init__(self, ctx, server: LlamaServer | None = None, engine_factory=None, detect=None, files=download):
        """`server`, `engine_factory(server, model_key, ctx=, timeout_s=, slots=)`, `detect(our_pids)` and `files`
        (the download module) can be replaced in tests."""
        self.ctx = ctx
        self.server = server or LlamaServer()
        self.engine_factory = engine_factory or ProAIEngine
        self._detect = detect or (lambda pids: hardware.detect(pids))
        self.files = files
        self.engine = None
        self.state = "off"
        self.message = "Pro AI is off."
        self.model_key = ""
        self.build_key = ""
        self.progress: dict | None = None  # download: {"phase", "done", "total"}
        self.selftest_progress: dict | None = None  # {"done", "total"}
        self.stats = {"queued": 0, "read": 0, "failed": 0, "too_old": 0, "queue_full": 0}
        self._jobs: list[ProJob] = []
        self._wake = asyncio.Event()  # (binds to the event loop on first use)
        self._workers: list[asyncio.Task] = []
        self._tasks: set[asyncio.Task] = set()
        self._setup_task: asyncio.Task | None = None
        self._selftest_task: asyncio.Task | None = None
        self._cancel_download = threading.Event()
        self._cancel_start = threading.Event()
        self._lock = asyncio.Lock()  # one start / stop at a time
        self._loop: asyncio.AbstractEventLoop | None = None
        self._applied: tuple | None = None
        self._stopping = False

    # ------------------------------------------------------------------ settings
    @property
    def settings(self):
        return self.ctx.config.settings.ai.pro_ai

    @property
    def ready(self) -> bool:
        return self.state == "ready" and self.engine is not None

    @property
    def downloading(self) -> bool:
        return self._setup_task is not None and not self._setup_task.done()

    @staticmethod
    def _key(s) -> tuple:
        return s.enabled, s.model, s.build, s.keep_tv_memory_free

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stopping = False
        self._applied = self._key(self.settings)
        self.ctx.config.on_change(self._on_settings)
        self._spawn(self._watch(), "pro-ai-watch")
        if self.settings.enabled:
            self._spawn(self.start_server(), "pro-ai-start")
        else:
            self._set("off", "Pro AI is off.")

    async def stop(self) -> None:
        self._stopping = True
        self._cancel_download.set()
        self._cancel_start.set()
        for t in [*self._tasks, *self._workers, self._setup_task, self._selftest_task]:
            if t is not None:
                t.cancel()
        for t in [*self._tasks, self._setup_task, self._selftest_task]:
            if t is not None:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await t
        self._tasks.clear()
        self._stop_workers()
        with contextlib.suppress(Exception):
            await asyncio.to_thread(self.server.stop, 5.0)
        self.engine = None

    def _spawn(self, coro, name: str) -> asyncio.Task:
        task = asyncio.get_running_loop().create_task(coro, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    def _on_settings(self, settings) -> None:  # called from whichever thread saved the settings
        loop = self._loop
        key = self._key(settings.ai.pro_ai)
        if engine := self.engine:
            engine.timeout_s = float(settings.ai.pro_ai.max_wait_seconds)
        if key == self._applied or loop is None or loop.is_closed():
            return
        self._applied = key
        loop.call_soon_threadsafe(self._apply_settings)

    def _apply_settings(self) -> None:
        if self._stopping or self.downloading:
            return  # (a finished set-up starts the server itself)
        if self.settings.enabled:
            self._spawn(self.start_server(), "pro-ai-start")
        else:
            self._spawn(self.stop_server(), "pro-ai-stop")

    # ------------------------------------------------------------------ state
    def _set(self, state: str, message: str) -> None:
        self.state, self.message = state, message
        if state == "off" and not self.downloading:
            self.ctx.state.clear_status("pro_ai")
        else:
            self.ctx.state.set_status("pro_ai", LEVELS.get(state, "warn"), message, name="Pro AI")
        self.ctx.bus.publish("pro_ai", self.summary())

    def summary(self) -> dict:
        """The small status the header and the heartbeat carry."""
        s = self.settings
        return {"enabled": s.enabled, "mode": s.mode, "state": self.state, "message": self.message,
                "model": self.model_key, "progress": self.progress, "selftest_progress": self.selftest_progress}

    def _idle_state(self, message: str = "") -> None:
        """Off or "not set up" (when nothing is starting, running or downloading)."""
        if not self.settings.enabled:
            self._set("off", message or "Pro AI is off.")
            return
        build, model, _why = self._chosen_quick()
        if build and model and self.files.installed_server_exe(build) and self.files.model_path(model):
            self._set("off", message or "Pro AI is stopped.")
        else:
            self._set("not_set_up", message or "Pro AI isn't set up yet - click 'Set up Pro AI' to download it "
                                               "(one time).")

    # ------------------------------------------------------------------ choosing build + model
    def _installed(self) -> dict:
        info = self.ctx.db.kv_get(INSTALLED_KEY) or {}
        return info if isinstance(info, dict) else {}

    def _chosen_quick(self) -> tuple[str | None, str | None, str]:
        """The build and model the settings point at, without looking at the hardware: "auto" means what the last
        set-up downloaded (None when that isn't known)."""
        s = self.settings
        info = self._installed()
        build = s.build if s.build != "auto" else (info.get("build") if info.get("build") in SERVER_BUILDS else None)
        model = s.model if s.model != "auto" else (info.get("model") if info.get("model") in MODELS_BY_KEY else None)
        return build, model, ""

    def chosen(self) -> tuple[str | None, str | None, str]:
        """(server build, model, why there is none). "auto" = what the last set-up downloaded when it is still there,
        else what suits this PC (looks at the hardware - call from a worker thread)."""
        build, model, _ = self._chosen_quick()
        if build is not None and self.settings.build == "auto" and not self.files.installed_server_exe(build):
            build = None
        if model is not None and self.settings.model == "auto" and not self.files.model_path(model):
            model = None
        why = ""
        if build is None or model is None:
            rec = self.recommend()
            build = build or rec["build"]
            model = model or rec["model"]
            why = rec["build_why"] if rec["build"] is None else rec["model_why"]
        return build, model, why

    def _our_pids(self) -> set[int]:
        pids = {os.getpid()}
        proc = getattr(self.server, "proc", None)
        if proc is not None and getattr(self.server, "running", False):
            pids.add(proc.pid)
        return pids

    def _tv_on(self) -> bool:
        s = self.ctx.config.settings
        return s.transcription.enabled and any(src.type == "stream" and src.enabled for src in s.sources)

    def _whisper(self) -> dict:
        """What TV transcription holds or will hold on the graphics card (for hardware.memory_budget)."""
        t = self.ctx.config.settings.transcription
        streams = self.ctx.service("streams")
        tr = getattr(streams, "transcriber", None)
        device = getattr(tr, "device", "") or ""
        loaded = getattr(tr, "model", None) is not None and device in ("cuda", "mlx")
        cpu = t.device == "cpu" or device == "cpu"
        return {"tv_on": self._tv_on() and self.settings.keep_tv_memory_free, "whisper_compute": t.compute_type,
                "whisper_loaded": loaded, "whisper_device": "cpu" if cpu else "cuda"}

    def recommend(self, hw: hardware.HardwareInfo | None = None) -> dict:
        """What suits this PC (looks at the hardware - call from a worker thread)."""
        hw = hw or self._detect(self._our_pids())
        build, build_why = hardware.pick_build(hw)
        w = self._whisper()
        ours = None
        if getattr(self.server, "running", False) and self.model_key in MODELS_BY_KEY:
            # a running Pro AI holds its own memory (Windows can't say how much - assume what its model needs)
            whisper = hardware.whisper_reserve_gb(hw, True, w["whisper_compute"], w["whisper_device"])
            ours = max(hw.ours_mib, (MODELS_BY_KEY[self.model_key].vram_gb
                                     + (whisper if w["whisper_loaded"] else 0)) * 1024)
        args = (hw, w["tv_on"], w["whisper_compute"], ours, w["whisper_loaded"], w["whisper_device"])
        budget = hardware.memory_budget(*args)
        model, model_why = hardware.pick_model(*args)
        return {"hw": hw, "budget": budget, "build": build, "build_why": build_why,
                "model": model.key if model else None, "model_why": model_why}

    def plan(self, build: str | None = None, model: str | None = None) -> dict:
        """For "Set up Pro AI": this PC's graphics card, the recommended build and model with download sizes, free
        disk space and which models fit. `build` / `model`: what to download (default: the settings; "auto" = the
        recommendation). Looks at the hardware - call from a worker thread."""
        rec = self.recommend()
        hw, budget = rec["hw"], rec["budget"]
        s = self.settings
        build = build if build in SERVER_BUILDS else (s.build if build is None and s.build != "auto" else rec["build"])
        model = model if model in MODELS_BY_KEY else (s.model if model is None and s.model != "auto" else rec["model"])
        known = budget.kind != "unknown"
        models = [{"key": m.key, "label": m.label, "note": m.note, "size": m.size, "size_gb": m.size_gb,
                   "vram_gb": m.vram_gb, "fits": (m.vram_gb <= budget.free_gb + 1e-9) if known else None,
                   "downloaded": self.files.model_path(m) is not None, "recommended": m.key == rec["model"],
                   "auto_pick": m.key not in NOT_AUTO} for m in MODELS]
        builds = [{"key": b.key, "label": b.label, "download_bytes": b.download_bytes,
                   "installed": self.files.installed_server_exe(b.key) is not None, "recommended": b.key == rec["build"]}
                  for b in SERVER_BUILDS.values() if b.os == hw.os]
        need = fetch = 0  # disk space needed (server archives are unpacked next to themselves), bytes downloaded
        if build in SERVER_BUILDS and not self.files.installed_server_exe(build):
            fetch += SERVER_BUILDS[build].download_bytes
            need += int(SERVER_BUILDS[build].download_bytes * download.UNPACK_FACTOR)
        if model in MODELS_BY_KEY and not self.files.model_path(model):
            fetch += MODELS_BY_KEY[model].size
            need += MODELS_BY_KEY[model].size
        try:
            free = shutil.disk_usage(download.llm_dir()).free
        except OSError:
            free = None
        return {"card": describe(hw), "hardware": hw.as_dict(), "can_run": rec["build"] is not None,
                "build": build, "build_why": rec["build_why"], "model": model, "model_why": rec["model_why"],
                "recommended_build": rec["build"], "recommended_model": rec["model"],
                "budget": budget.as_dict(), "models": models, "builds": builds, "tv_on": self._tv_on(),
                "download_bytes": fetch, "need_bytes": need, "disk_free_bytes": free,
                "enough_disk": None if free is None else free >= need * download.DISK_SLACK}

    # ------------------------------------------------------------------ server
    async def start_server(self) -> None:
        """Start (or restart) llama-server with the chosen model, if it is turned on and downloaded."""
        self._cancel_start.clear()
        async with self._lock:
            await self._stop_locked()
            if self._stopping:
                return
            if not self.settings.enabled:
                self._idle_state()
                return
            try:
                build, model, why = await asyncio.to_thread(self.chosen)
            except Exception as exc:  # pragma: no cover - detection never raises
                self._set("error", f"Couldn't look at this computer's hardware: {exc}")
                return
            if build is None:
                self._set("error", why or "Pro AI can't run on this computer.")
                return
            if model is None:
                self._set("not_set_up", why or "Pick a Pro AI model in Settings -> Pro AI.")
                return
            exe, path = self.files.installed_server_exe(build), self.files.model_path(model)
            if exe is None or path is None:
                missing = MODELS_BY_KEY[model].label if exe is not None else "the Pro AI server"
                self._set("not_set_up", f"{missing} isn't downloaded yet - click 'Set up Pro AI' (one time).")
                return
            reserve = await asyncio.to_thread(self._reserve_mib)
            self.model_key, self.build_key = model, build
            self._set("starting", f"Loading {MODELS_BY_KEY[model].label} onto the graphics card...")
            try:
                await asyncio.to_thread(self.server.start, path, exe, reserve, CONTEXT, SLOTS, 0, build)
                ok = await asyncio.to_thread(self.server.wait_ready, READY_TIMEOUT, 0.25, self._cancel_start)
            except Exception as exc:
                log.warning("Pro AI server didn't start: %s", exc)
                self._set("error", getattr(self.server, "message", "") or f"Pro AI's server couldn't start: {exc}")
                return
            if self._cancel_start.is_set():
                return  # stop_server / a newer start takes it from here
            if not ok:
                self._set("error", getattr(self.server, "message", "") or "Pro AI's server didn't start.")
                return
            self.engine = self.engine_factory(self.server, model, ctx=self.ctx,
                                              timeout_s=float(self.settings.max_wait_seconds), slots=SLOTS)
            self._start_workers()
            self._set("ready", self._ready_message())
            log.info("Pro AI ready: %s (%s)", MODELS_BY_KEY[model].label, SERVER_BUILDS[build].label)

    async def stop_server(self) -> None:
        self._cancel_start.set()
        async with self._lock:
            await self._stop_locked()
            self._idle_state()

    async def _stop_locked(self) -> None:
        self._stop_workers()
        self.engine = None
        if getattr(self.server, "running", False) or getattr(self.server, "state", "off") != "off":
            await asyncio.to_thread(self.server.stop, 5.0)

    def _reserve_mib(self) -> int:
        """Graphics memory llama-server must leave free when it loads: room for TV transcription plus a margin."""
        try:
            w = self._whisper()
            hw = self._detect(self._our_pids())
            return hardware.memory_budget(hw, w["tv_on"], w["whisper_compute"], None, w["whisper_loaded"],
                                          w["whisper_device"]).reserve_mib
        except Exception:  # pragma: no cover - detection never raises
            return 1024

    def _ready_message(self) -> str:
        msg = getattr(self.server, "message", "") or "Ready"
        label = MODELS_BY_KEY[self.model_key].label if self.model_key in MODELS_BY_KEY else self.model_key
        return f"{msg} ({label})"

    async def _watch(self) -> None:
        """Restart a crashed server once (the server's own rule); a second crash stays an error."""
        while True:
            await asyncio.sleep(WATCH_EVERY)
            try:
                await self.check_server()
            except Exception:
                log.debug("Pro AI watch failed", exc_info=True)

    async def check_server(self) -> None:
        if self.state not in ("ready", "starting") or self._lock.locked():
            return
        srv = self.server
        if srv.state == "ready" and self.state != "ready" and self.engine is not None:
            self._set("ready", self._ready_message())
        elif srv.state == "starting" and self.state == "ready":
            self._set("starting", "Pro AI's server is restarting...")
        elif srv.state in ("error", "off") and not getattr(srv, "running", False):
            if await asyncio.to_thread(srv.restart_if_crashed):
                self._set("starting", "Pro AI's server stopped unexpectedly - starting it again...")
                ok = await asyncio.to_thread(srv.wait_ready, READY_TIMEOUT, 0.25, self._cancel_start)
                if ok and self.engine is not None:
                    self._set("ready", self._ready_message())
                    return
            if self._cancel_start.is_set() or self._stopping or self._lock.locked() or self.engine is None:
                return  # stopped or being restarted from Settings meanwhile
            self._stop_workers()
            self.engine = None
            self._set("error", (srv.message or "Pro AI's server stopped.") +
                      " It was already restarted once - start it again in Settings -> Pro AI.")

    # ------------------------------------------------------------------ queue
    def _start_workers(self) -> None:
        self._stop_workers()
        self._workers = [asyncio.get_running_loop().create_task(self._worker(), name=f"pro-ai-worker-{i}")
                         for i in range(SLOTS)]

    def _stop_workers(self) -> None:
        for w in self._workers:
            w.cancel()
        self._workers = []
        for job in self._jobs:
            self._finish(job, None)
        self._jobs.clear()

    @staticmethod
    def _finish(job: ProJob, result) -> None:
        if job.future is not None and not job.future.done():
            job.future.set_result(result)

    def offer(self, job: ProJob) -> bool:
        """Queue a story for Pro AI (call in the event loop). Newest first when busy; the oldest is dropped when the
        queue is full. False when Pro AI isn't running."""
        if not self.ready:
            self._finish(job, None)
            return False
        if len(self._jobs) >= QUEUE_MAX:
            oldest = next((j for j in self._jobs if not j.urgent), self._jobs[0])
            self._jobs.remove(oldest)
            self._finish(oldest, None)
            self.stats["queue_full"] += 1
        self._jobs.append(job)
        self.stats["queued"] += 1
        self._wake.set()
        return True

    async def ask(self, job: ProJob, timeout: float | None = None) -> dict | None:
        """Queue a story ahead of the others and wait for Pro AI's answer (judge mode). None if Pro AI isn't
        running, failed, or didn't answer within `timeout` seconds (it still finishes and is stored)."""
        job.urgent = True
        job.future = asyncio.get_running_loop().create_future()
        if not self.offer(job):
            return None
        try:
            return await asyncio.wait_for(asyncio.shield(job.future),
                                          timeout if timeout is not None else self.settings.max_wait_seconds)
        except TimeoutError:
            log.info("Pro AI didn't answer in time - the main engine's call stands: %s", job.item.title[:80])
            return None

    def take(self) -> ProJob | None:
        """The next story: urgent ones first, then the newest. Stories that waited too long are dropped."""
        limit = self.settings.max_wait_seconds
        now = time.monotonic()
        fresh = []
        for job in self._jobs:
            if now - job.queued_at > limit:
                self.stats["too_old"] += 1
                self._finish(job, None)
            else:
                fresh.append(job)
        self._jobs = fresh
        if not fresh:
            return None
        urgent = [j for j in fresh if j.urgent]
        job = (urgent or fresh)[-1]
        self._jobs.remove(job)
        return job

    async def _worker(self) -> None:
        while True:
            if not self.ready:  # the server is restarting: let the stories wait (or go stale) instead of failing
                await asyncio.sleep(0.5)
                continue
            job = self.take()
            if job is None:
                self._wake.clear()
                await self._wake.wait()
                continue
            await self.run_job(job)

    async def run_job(self, job: ProJob) -> dict | None:
        result = None
        engine = self.engine
        try:
            if engine is None:
                return None
            pipeline = self.ctx.service("pipeline")
            market_open = pipeline._market_open() if pipeline is not None else None
            res = await engine.analyze(job.item, job.pre, market_open, datetime.now(UTC))
            self.stats["read" if res.ok else "failed"] += 1
            if pipeline is not None:
                result = await pipeline.record_pro(job, res)
        except asyncio.CancelledError:
            raise
        except Exception:
            self.stats["failed"] += 1
            log.exception("Pro AI couldn't handle %s", job.item.title[:80])
        finally:
            self._finish(job, result)
        return result

    # ------------------------------------------------------------------ set-up (download)
    def setup(self, build: str = "auto", model: str = "auto") -> dict:
        """Start the one-time download in the background (call in the event loop). "auto" = what suits this PC.
        Raises ValueError (with a message for the user) or RuntimeError when a download is already running."""
        if self.downloading:
            raise RuntimeError("Pro AI is already downloading.")
        if build != "auto" and build not in SERVER_BUILDS:
            raise ValueError(f"Unknown server build '{build}'.")
        if model != "auto" and model not in MODELS_BY_KEY:
            raise ValueError(f"Unknown model '{model}'.")
        self._cancel_download.clear()
        self._setup_task = asyncio.get_running_loop().create_task(self._setup(build, model), name="pro-ai-setup")
        return {"ok": True}

    def cancel_setup(self) -> None:
        self._cancel_download.set()

    async def _setup(self, build: str, model: str) -> None:
        try:
            if build == "auto" or model == "auto":
                rec = await asyncio.to_thread(self.recommend)
                if build == "auto":
                    build = rec["build"]
                    if build is None:
                        raise download.DownloadError(rec["build_why"])
                if model == "auto":
                    model = rec["model"]
                    if model is None:
                        raise download.DownloadError(rec["model_why"] + " Pick a model in the list and try again.")
            b, m = SERVER_BUILDS[build], MODELS_BY_KEY[model]
            server_bytes = 0 if self.files.installed_server_exe(build) else b.download_bytes
            model_bytes = 0 if self.files.model_path(model) else m.size
            total = max(1, server_bytes + model_bytes)

            def report(phase: str, base: int):
                def cb(done: int, _total: int) -> None:
                    self.progress = {"phase": phase, "done": base + done, "total": total}
                    pct = int(100 * (base + done) / total)
                    what = "the Pro AI server" if phase == "server" else m.label
                    self.message = f"Downloading {what}: {_gb(base + done)} of {_gb(total)} ({pct}%)"
                    self.ctx.bus.publish("pro_ai", self.summary())
                return cb

            self.progress = {"phase": "server", "done": 0, "total": total}
            self._set("downloading", f"Downloading Pro AI ({_gb(total)})...")
            if server_bytes:
                await asyncio.to_thread(self.files.install_server, build, report("server", 0), self._cancel_download)
            await asyncio.to_thread(self.files.fetch_model, model, report("model", server_bytes),
                                    self._cancel_download)
            self.ctx.db.kv_set(INSTALLED_KEY, {"build": build, "model": model, "at": iso()})
            self.progress = None
            log.info("Pro AI downloaded: %s + %s", b.label, m.label)
            if not self.settings.enabled:  # set up = wanted: turn it on (watch-only unless the user changed that)
                self.ctx.config.update({"ai": {"pro_ai": {"enabled": True}}})
                self._applied = self._key(self.settings)
                self.ctx.bus.publish("settings_changed", {})
            self._spawn(self.start_server(), "pro-ai-start")
        except download.DownloadCancelled:
            self.progress = None
            self._idle_state("Download cancelled - setting up again continues where it stopped.")
        except download.DownloadError as exc:
            self.progress = None
            self._set("error", str(exc))
        except Exception as exc:
            log.exception("Pro AI set-up failed")
            self.progress = None
            self._set("error", f"Pro AI's download failed: {exc}")

    async def delete_downloads(self) -> int:
        """Stop Pro AI, delete every downloaded server and model and turn it off. Returns the bytes freed."""
        if self.downloading:
            raise RuntimeError("Cancel the download first.")
        if self.settings.enabled:
            self.ctx.config.update({"ai": {"pro_ai": {"enabled": False}}})
            self._applied = self._key(self.settings)
            self.ctx.bus.publish("settings_changed", {})
        await self.stop_server()

        def remove() -> int:
            freed = sum(self.files.delete_model(m.key) for m in MODELS)
            freed += sum(self.files.delete_server(k) for k in SERVER_BUILDS)
            return freed + self.files.delete_old_servers()

        freed = await asyncio.to_thread(remove)
        self.ctx.db.kv_set(INSTALLED_KEY, None)
        self.model_key = self.build_key = ""
        self._idle_state()
        log.info("Pro AI downloads deleted (%s freed)", _gb(freed))
        return freed

    # ------------------------------------------------------------------ Test my PC
    def run_selftest(self) -> None:
        """Start "Test my PC" in the background (20 labelled headlines). Raises RuntimeError if Pro AI isn't ready
        or a test is already running."""
        if not self.ready:
            raise RuntimeError("Start Pro AI first (it has to be set up and running).")
        if self._selftest_task is not None and not self._selftest_task.done():
            raise RuntimeError("The test is already running.")
        self._selftest_task = asyncio.get_running_loop().create_task(self._selftest(), name="pro-ai-selftest")

    async def _selftest(self) -> None:
        pipeline = self.ctx.service("pipeline")
        tickers = getattr(pipeline, "tickers", None)

        def progress(done: int, total: int) -> None:
            self.selftest_progress = {"done": done, "total": total}
            self.ctx.bus.publish("pro_ai", self.summary())

        try:
            self.selftest_progress = {"done": 0, "total": len(selftest.load_samples())}
            self.ctx.bus.publish("pro_ai", self.summary())
            res = await selftest.run_selftest(self.engine, tickers=tickers, progress_cb=progress)
            res.update(at=iso(), model=self.model_key, build=self.build_key)
            self.ctx.db.kv_set(SELFTEST_KEY, res)
            log.info("Pro AI test: %s/%s right, %.1f s per story, whole model on the graphics card: %s",
                     res["right"], res["total"], res["avg_seconds"], res["fully_on_gpu"])
        except Exception as exc:
            log.exception("Pro AI test failed")
            self.ctx.db.kv_set(SELFTEST_KEY, {"error": str(exc), "at": iso(), "model": self.model_key})
        finally:
            self.selftest_progress = None
            self.ctx.bus.publish("pro_ai", self.summary())

    # ------------------------------------------------------------------ status
    def status(self) -> dict:
        """Everything the Settings card shows (reads the download folder - call from a worker thread)."""
        s = self.settings
        build, model, _ = self._chosen_quick()
        files = self.files.status()
        test = self.ctx.db.kv_get(SELFTEST_KEY)
        agreement = {r["agreement"]: r["n"] for r in self.ctx.db.query(
            "SELECT agreement, COUNT(*) AS n FROM analyses WHERE engine = 'pro' AND agreement IS NOT NULL "
            "GROUP BY agreement")}
        server = self.server.snapshot() if hasattr(self.server, "snapshot") else {}
        return {**self.summary(), "settings": s.model_dump(), "build": build, "model": model or self.model_key,
                "running_model": self.model_key, "running_build": self.build_key,
                "downloaded": {"server": bool(build and self.files.installed_server_exe(build)),
                               "model": bool(model and self.files.model_path(model))},
                "disk_bytes": files.get("total_bytes", 0), "folder": files.get("folder", ""),
                "server": server, "queue": {"waiting": len(self._jobs), **self.stats},
                "agreement": agreement, "selftest": test,
                "selftest_running": self._selftest_task is not None and not self._selftest_task.done()}


def describe(hw: hardware.HardwareInfo) -> str:
    """This PC's graphics in a few words, e.g. "NVIDIA GeForce RTX 5070 Ti (16 GB, driver 581.57)"."""
    if hw.os == "mac":
        chip = hw.gpu_name or ("Apple Silicon" if hw.apple_silicon else "Intel Mac")
        return f"{chip} ({hw.ram_gb:.0f} GB memory)" if hw.ram_bytes else chip
    if hw.nvidia:
        return f"{hw.gpu_name} ({hw.vram_total_mib / 1024:.0f} GB, driver {hw.driver or '?'})"
    return "No NVIDIA graphics card found" + (f" ({hw.note})" if hw.note and "no NVIDIA" not in hw.note else "")


def setting_options(system: str | None = None) -> dict:
    """The choices Settings -> Pro AI offers: models (with sizes) and this operating system's server builds."""
    os_name = hardware._os_name(system or platform.system())
    models = {"auto": "Auto - the best one that fits this PC"}
    models.update({m.key: f"{m.label} - {m.size_gb:g} GB download, needs {m.vram_gb:g} GB graphics memory"
                   for m in MODELS})
    builds = {"auto": "Auto - matched to this PC's graphics"}
    builds.update({b.key: f"{b.label} - {b.download_bytes / 1e6:.0f} MB download" for b in SERVER_BUILDS.values()
                   if b.os == os_name})
    return {"pro_ai_models": models, "pro_ai_builds": builds}
