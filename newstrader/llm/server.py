"""Runs llama.cpp's llama-server as a separate program that only this computer can reach (127.0.0.1, a free port and
a random API key), watches its log, and stops it again. One server holds one model."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

import httpx

from . import download

log = logging.getLogger(__name__)

STATES = ("off", "starting", "ready", "error")
LOG_LINES = 300
CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_PROCESS_GROUP = 0x00000200
NO_GPU_BUILDS = {"cpu", "linux-cpu"}  # nothing to offload

_OFFLOAD = re.compile(r"offloaded (\d+)\s*/\s*(\d+) layers to GPU")
_FLAG = re.compile(r"(?<![\w-])(--?[A-Za-z][\w-]*)")
_LEVEL = re.compile(r"^\s*(?:[\d.:]+\s+)?([DITWE])\s")  # "0.00.712.350 W slot ..." (time, level, message)
_PROBLEM = re.compile(r"\b(error|failed|failure|out of memory|unable to|abort)", re.IGNORECASE)
_HARMLESS = ("need to evaluate at least 1 token", "n_past was set to")
_help_cache: dict[tuple[str, float], set[str] | None] = {}


def parse_help_flags(text: str) -> set[str]:
    """Every option name mentioned in `llama-server --help`."""
    return {m.group(1) for m in _FLAG.finditer(text or "")}


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def pid_file() -> Path:
    return download.llm_dir() / "llama-server.pid"


def _popen_kwargs() -> dict:
    if sys.platform == "win32":
        return {"creationflags": CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def explain_failure(lines: list[str], code: int | None = None) -> str:
    """Turn the end of the server's log into one plain sentence."""
    text = "\n".join(lines[-80:]).lower()
    if any(k in text for k in ("out of memory", "failed to allocate", "cudamalloc failed", "erroroutofdevicememory",
                                "unable to allocate", "insufficient memory")):
        return ("Pro AI ran out of graphics memory. Close other programs that use the graphics card, turn off TV "
                "transcription, or pick a smaller model.")
    if "failed to load model" in text or "error loading model" in text or "unable to load model" in text:
        return "Pro AI couldn't load the model file. Delete the model in Settings and download it again."
    if "couldn't bind" in text or "address already in use" in text or "failed to bind" in text:
        return "Pro AI couldn't open its local connection (port in use). Starting it again picks another one."
    if "invalid argument" in text or "unknown argument" in text or "error: unrecognized" in text:
        return "This Pro AI server build didn't accept its settings. Reinstall the server in Settings."
    if code is not None and sys.platform == "win32" and code in (-1073741515, 3221225781):
        return ("Pro AI's server is missing a system file (often the graphics driver or Microsoft Visual C++ "
                "runtime). Update the graphics driver and try again.")
    last = next((ln for ln in reversed(lines) if ln.strip()), "")
    tail = f" Last message: {last[:200]}" if last else ""
    return f"Pro AI's server stopped unexpectedly (exit code {code}).{tail}"


# --------------------------------------------------------------------------------------- stale servers
def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        out = _run_quiet(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"])
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _run_quiet(args: list[str]) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=10, check=False,
                              creationflags=CREATE_NO_WINDOW if sys.platform == "win32" else 0).stdout or ""
    except Exception:
        return ""


def _process_name(pid: int) -> str:
    if sys.platform == "win32":
        return _run_quiet(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"])
    proc = Path(f"/proc/{pid}/cmdline")
    if proc.exists():
        try:
            return proc.read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
        except OSError:
            return ""
    return _run_quiet(["ps", "-p", str(pid), "-o", "comm="])


def _kill_pid(pid: int) -> None:
    if sys.platform == "win32":
        _run_quiet(["taskkill", "/PID", str(pid), "/T", "/F"])
        return
    with contextlib.suppress(OSError):
        os.kill(pid, signal.SIGTERM)
    for _ in range(30):
        with contextlib.suppress(OSError):  # (only reaps it if it's our own child)
            os.waitpid(pid, os.WNOHANG)
        if not _pid_alive(pid):
            return
        time.sleep(0.1)
    with contextlib.suppress(OSError):
        os.kill(pid, signal.SIGKILL)


def kill_stale(path: Path | None = None) -> bool:
    """Stop a llama-server left running by an earlier run that crashed. Only kills a process that is still a
    llama-server (the pid may have been reused by something else)."""
    path = path or pid_file()
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
        pid = int(info.get("pid", 0))
    except (OSError, ValueError, TypeError, AttributeError):
        with contextlib.suppress(OSError):
            path.unlink()
        return False
    killed = False
    if pid != os.getpid() and _pid_alive(pid) and "llama-server" in _process_name(pid):
        log.info("Stopping a Pro AI server left over from the last run (pid %s)", pid)
        _kill_pid(pid)
        killed = True
    with contextlib.suppress(OSError):
        path.unlink()
    return killed


# --------------------------------------------------------------------------------------- the server
class LlamaServer:
    """One llama-server process. Not async itself: call start/wait_ready/stop from a worker thread
    (or use wait_ready_async)."""

    def __init__(self, pid_path: Path | None = None, launcher: list[str] | None = None, max_restarts: int = 1):
        self.pid_path = pid_path
        self.launcher = list(launcher or [])  # tests run a fake server script through the Python interpreter
        self.max_restarts = max_restarts
        self.state = "off"
        self.message = ""
        self.url = ""
        self.api_key = ""
        self.port = 0
        self.build_key = ""
        self.model_path: Path | None = None
        self.args: list[str] = []
        self.dropped_flags: list[str] = []
        self.supported: set[str] | None = None
        self.gpu_layers: int | None = None
        self.total_layers: int | None = None
        self.restarts = 0
        self.exit_code: int | None = None
        self.started_at = 0.0
        self.ready_seconds: float | None = None
        self.logs: deque[str] = deque(maxlen=LOG_LINES)
        self.proc: subprocess.Popen | None = None
        self._launch: dict | None = None
        self._stopping = False
        self._lock = threading.RLock()
        self._reader: threading.Thread | None = None
        self._logged: deque[float] = deque()
        self._hidden = 0

    # ------------------------------------------------------------------ status
    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    @property
    def fully_on_gpu(self) -> bool | None:
        if self.build_key in NO_GPU_BUILDS or self.gpu_layers is None or not self.total_layers:
            return None
        return self.gpu_layers >= self.total_layers

    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def snapshot(self) -> dict:
        return {"state": self.state, "message": self.message, "running": self.running, "build": self.build_key,
                "model_file": self.model_path.name if self.model_path else "", "port": self.port,
                "gpu_layers": self.gpu_layers, "total_layers": self.total_layers, "fully_on_gpu": self.fully_on_gpu,
                "restarts": self.restarts, "ready_seconds": self.ready_seconds,
                "dropped_flags": list(self.dropped_flags), "log_tail": list(self.logs)[-20:]}

    def _set(self, state: str, message: str) -> None:
        self.state, self.message = state, message

    # ------------------------------------------------------------------ arguments
    def _command(self, exe: Path) -> list[str]:
        return [*self.launcher, str(exe)]

    def supported_flags(self, exe: Path) -> set[str] | None:
        """Option names this build knows (from `llama-server --help`, asked once per program file)."""
        try:
            key = (str(exe), exe.stat().st_mtime)
        except OSError:
            return None
        if key in _help_cache:
            return _help_cache[key]
        flags = None
        try:
            out = subprocess.run([*self._command(exe), "--help"], capture_output=True, timeout=60, check=False,
                                 cwd=str(exe.parent), stdin=subprocess.DEVNULL, **_popen_kwargs())
            text = (out.stdout or b"").decode("utf-8", "replace") + (out.stderr or b"").decode("utf-8", "replace")
            found = parse_help_flags(text)
            flags = found if "--port" in found else None
        except Exception as exc:
            log.warning("Couldn't ask the Pro AI server for its options: %s", exc)
        if flags is None:
            log.warning("Pro AI server's --help didn't list its options; starting it with every option")
        _help_cache[key] = flags
        return flags

    @staticmethod
    def build_args(model_path: Path, exe_cmd: list[str], port: int, api_key: str, reserve_mib: int, ctx: int,
                   slots: int, idle_seconds: int, supported: set[str] | None) -> tuple[list[str], list[str]]:
        """(command line, optional flags left out because this build doesn't know them). `supported`: the
        option names from --help (None = unknown, use them all)."""
        args = [*exe_cmd, "-m", str(model_path), "--host", "127.0.0.1", "--port", str(port), "--api-key", api_key]
        optional: list[tuple[tuple[str, ...], str | None]] = [
            (("-c", "--ctx-size"), str(ctx)),  # shared by the slots
            (("-np", "--parallel"), str(slots)),
            (("--jinja",), None),
            (("--reasoning", "-rea"), "off"),
            (("--reasoning-budget",), "0"),
            (("--fit",), "on"),
            (("--fit-target", "-fitt"), str(max(0, int(reserve_mib)))),
            (("--cache-ram", "-cram"), "0"),  # no extra prompt cache in system memory (it grows to 8 GB)
            (("--no-cache-idle-slots",), None),
            (("--no-ui", "--no-webui"), None),
            (("--offline",), None),
            (("--log-prefix",), None),  # time and I/W/E in front of each line
            (("-lv", "--verbosity", "--log-verbosity"), "4"),  # 4 includes "offloaded X/Y layers to GPU"
        ]
        if idle_seconds > 0:
            optional.append((("--sleep-idle-seconds",), str(idle_seconds)))
        dropped = []
        for names, value in optional:
            name = names[0] if supported is None else next((n for n in names if n in supported), None)
            if name is None:
                dropped.append(names[0])
                continue
            args.append(name)
            if value is not None:
                args.append(value)
        if "--fit" in dropped:  # an older build: ask for every layer on the graphics card instead
            ngl = next((n for n in ("-ngl", "--n-gpu-layers", "--gpu-layers") if n in (supported or ())), None)
            if ngl:
                args += [ngl, "999"]
        return args, dropped

    # ------------------------------------------------------------------ start / stop
    def start(self, model_path: Path, server_exe: Path, reserve_mib: int = 1024, ctx: int = 8192, slots: int = 2,
              idle_seconds: int = 0, build_key: str = "") -> None:
        """Launch the server (returns at once; then call wait_ready). Stops an earlier one first. `reserve_mib`:
        graphics memory to leave free (MemoryBudget.reserve_mib); `idle_seconds` > 0 lets the server unload the
        model after that long without questions (the next question then waits for it to load again). Raises
        FileNotFoundError / OSError (with state "error" and a message) if it can't be launched."""
        with self._lock:
            self.stop()
            self.restarts = 0
            kill_stale(self.pid_path or pid_file())
            self._launch = {"model_path": Path(model_path), "server_exe": Path(server_exe),
                            "reserve_mib": reserve_mib, "ctx": ctx, "slots": slots, "idle_seconds": idle_seconds,
                            "build_key": build_key}
            self._spawn()

    def _spawn(self) -> None:
        p = self._launch or {}
        exe: Path = p["server_exe"]
        model: Path = p["model_path"]
        self.build_key = p["build_key"]
        self.model_path = model
        if not exe.is_file():
            self._set("error", "Pro AI's server program is missing - reinstall it in Settings.")
            raise FileNotFoundError(str(exe))
        if not model.is_file():
            self._set("error", "Pro AI's model file is missing - download it again in Settings.")
            raise FileNotFoundError(str(model))
        self.supported = self.supported_flags(exe)
        self.port = free_port()
        self.api_key = secrets.token_urlsafe(24)
        self.url = f"http://127.0.0.1:{self.port}"
        self.args, self.dropped_flags = self.build_args(model, self._command(exe), self.port, self.api_key,
                                                        p["reserve_mib"], p["ctx"], p["slots"], p["idle_seconds"],
                                                        self.supported)
        if self.dropped_flags:
            log.info("Pro AI server build doesn't know %s - left out", ", ".join(self.dropped_flags))
        self.logs.clear()
        self.gpu_layers = self.total_layers = None
        self.exit_code = None
        self.ready_seconds = None
        self._stopping = False
        self._set("starting", "Loading the model...")
        self.started_at = time.monotonic()
        try:
            proc = subprocess.Popen(self.args, cwd=str(exe.parent), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, **_popen_kwargs())
        except OSError as exc:
            self._set("error", f"Pro AI's server couldn't start: {exc}")
            raise
        self.proc = proc
        self._write_pid(proc.pid, exe)
        self._reader = threading.Thread(target=self._read_logs, args=(proc,), name="llama-server-log", daemon=True)
        self._reader.start()

    def _write_pid(self, pid: int, exe: Path) -> None:
        path = self.pid_path or pid_file()
        with contextlib.suppress(OSError):
            path.write_text(json.dumps({"pid": pid, "exe": str(exe), "port": self.port, "started": time.time()}),
                            encoding="utf-8")

    def stop(self, timeout: float = 10.0) -> None:
        """Stop the server (terminate, wait, then kill)."""
        with self._lock:
            proc = self.proc
            self._stopping = True
            if proc is not None and proc.poll() is None:
                with contextlib.suppress(OSError):
                    proc.terminate()
                try:
                    proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(OSError):
                        proc.kill()
                    with contextlib.suppress(subprocess.TimeoutExpired):
                        proc.wait(timeout=5)
            if proc is not None and self._reader is not None:
                self._reader.join(timeout=2)
            if proc is not None:
                with contextlib.suppress(OSError):
                    (self.pid_path or pid_file()).unlink()
            self.proc = None
            if self.state != "error" or proc is not None:
                self._set("off", "Stopped")

    def restart_if_crashed(self) -> bool:
        """The restart-once policy: if the server died on its own and hasn't been restarted max_restarts times yet,
        start it again (then call wait_ready). Returns True when it restarted."""
        with self._lock:
            if self._launch is None or self.running or self._stopping or self.state not in ("error", "starting",
                                                                                              "ready"):
                return False
            if self.restarts >= self.max_restarts:
                return False
            self.restarts += 1
            log.warning("Pro AI server stopped unexpectedly - restarting it (%s of %s)", self.restarts,
                        self.max_restarts)
            try:
                self._spawn()
            except Exception as exc:
                log.warning("Pro AI server restart failed: %s", exc)
                return False
            return True

    # ------------------------------------------------------------------ readiness
    def health(self, client: httpx.Client | None = None) -> int | None:
        """HTTP status of GET /health (200 ready, 503 loading) or None if nothing answers."""
        if not self.url:
            return None
        own = client is None
        client = client or httpx.Client(trust_env=False, timeout=5)
        try:
            return client.get(f"{self.url}/health", headers=self.headers()).status_code
        except httpx.HTTPError:
            return None
        finally:
            if own:
                client.close()

    def wait_ready(self, timeout: float = 180.0, poll: float = 0.25,
                   cancel_event: threading.Event | None = None) -> bool:
        """Poll /health until the model is loaded. False (with state "error" and a message) if the server exits,
        or it takes longer than `timeout` seconds (then it is stopped). A set cancel_event returns False at once and
        leaves the server to the caller."""
        deadline = time.monotonic() + timeout
        with httpx.Client(trust_env=False, timeout=5) as client:
            while time.monotonic() < deadline:
                if cancel_event is not None and cancel_event.is_set():
                    return False
                proc = self.proc
                if proc is not None and proc.poll() is not None:
                    if self._reader is not None:
                        self._reader.join(timeout=2)
                    if self.state != "error":
                        self._set("error", explain_failure(list(self.logs), proc.returncode))
                    return False
                if self.health(client) == 200:
                    self.ready_seconds = round(time.monotonic() - self.started_at, 1) if self.started_at else None
                    self._set("ready", self._ready_message())
                    return True
                time.sleep(poll)
        self.stop()
        self._set("error", f"Pro AI's server didn't finish loading within {timeout:.0f} seconds. A smaller model "
                           "loads faster.")
        return False

    async def wait_ready_async(self, timeout: float = 180.0) -> bool:
        return await asyncio.to_thread(self.wait_ready, timeout)

    def _ready_message(self) -> str:
        on_gpu = self.fully_on_gpu
        if on_gpu is True:
            return f"Ready - the whole model is on the graphics card ({self.gpu_layers} layers)"
        if on_gpu is False:
            return (f"Ready, but only {self.gpu_layers} of {self.total_layers} layers fit on the graphics card - the "
                    "rest runs on the processor, which is slower. A smaller model or turning off TV transcription "
                    "would make it faster.")
        return "Ready"

    # ------------------------------------------------------------------ log
    def _read_logs(self, proc: subprocess.Popen) -> None:
        stream = proc.stdout
        if stream is not None:
            for raw in iter(stream.readline, b""):
                self.handle_log_line(raw.decode("utf-8", "replace").rstrip())
            with contextlib.suppress(OSError):
                stream.close()
        code = proc.wait()
        if proc is not self.proc:  # (no lock: stop() holds it while waiting for this thread)
            return
        self.exit_code = code
        if not self._stopping:
            self._set("error", explain_failure(list(self.logs), code))
            log.warning("Pro AI server exited (code %s): %s", code, self.message)

    def handle_log_line(self, line: str) -> None:
        if self.api_key and self.api_key in line:
            line = line.replace(self.api_key, "***")
        self.logs.append(line)
        m = _OFFLOAD.search(line)
        if m:
            self.gpu_layers, self.total_layers = int(m.group(1)), int(m.group(2))
        level = _LEVEL.match(line)
        problem = level.group(1) in "WE" if level else bool(_PROBLEM.search(line))
        if problem and not any(h in line for h in _HARMLESS):
            self._log_rate_limited(line)

    def _log_rate_limited(self, line: str, per_minute: int = 5) -> None:
        now = time.monotonic()
        while self._logged and now - self._logged[0] > 60:
            self._logged.popleft()
        if len(self._logged) >= per_minute:
            self._hidden += 1
            return
        self._logged.append(now)
        extra = f" (+{self._hidden} similar messages hidden)" if self._hidden else ""
        self._hidden = 0
        log.warning("Pro AI server: %s%s", line[:300], extra)
