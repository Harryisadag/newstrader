"""Launches NewsTrader: the local web server in a background thread + the desktop window.

    python -m newstrader             desktop window (normal use)
    python -m newstrader --browser   open the dashboard in your normal web browser instead
    python -m newstrader --headless  server only, prints the dashboard URL (for debugging)
"""

from __future__ import annotations

import argparse
import logging
import os
import socket
import sys
import threading
import time
import webbrowser

from . import APP_NAME, __version__, paths

log = logging.getLogger("newstrader.app")


# ---------------------------------------------------------------- single instance
class SingleInstance:
    """Holds an exclusive lock on data/newstrader.lock while the app runs."""

    def __init__(self, path):
        self.path = path
        self._fh = None

    def acquire(self) -> bool:
        self._fh = open(self.path, "a+")  # noqa: SIM115 - must stay open for the life of the app
        try:
            if sys.platform == "win32":
                import msvcrt

                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            self._fh.close()
            self._fh = None
            return False


def _show_error(message: str) -> None:
    print(message, file=sys.stderr)
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(None, message, APP_NAME, 0x10)
        except Exception:
            pass
    elif sys.platform == "darwin":
        try:
            import subprocess

            # text goes in as an argument, never as script source
            subprocess.run(["/usr/bin/osascript", "-e", "on run argv",
                            "-e", "display alert (item 2 of argv) message (item 3 of argv) as critical",
                            "-e", "end run", "newstrader", APP_NAME, message], timeout=120, check=False,
                           capture_output=True)
        except Exception:
            pass


def _platform_fixes() -> None:
    """Mac: python.org Python doesn't use the system certificate store, so point TLS at certifi's bundle
    (Alpaca's live websockets need it). Intel Macs: the speech engine and scikit-learn each ship an OpenMP
    library; allow both to load instead of aborting."""
    if sys.platform != "darwin":
        return
    if not os.environ.get("SSL_CERT_FILE"):
        try:
            import certifi

            os.environ["SSL_CERT_FILE"] = certifi.where()
        except Exception:
            pass
    import platform

    if platform.machine() == "x86_64":
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _ensure_std_streams() -> None:
    """The windowed .exe has no console, so sys.stdout/stderr are None; some libraries expect real files."""
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")  # noqa: SIM115
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")  # noqa: SIM115


def main(argv: list[str] | None = None) -> int:
    _ensure_std_streams()
    _platform_fixes()
    parser = argparse.ArgumentParser(prog="newstrader", description=f"{APP_NAME} {__version__}")
    parser.add_argument("--browser", action="store_true", help="open the dashboard in your web browser")
    parser.add_argument("--headless", action="store_true", help="run the server only and print the URL")
    parser.add_argument("--port", type=int, default=0, help="port to listen on (default: random free port)")
    parser.add_argument("--token", default=None, help=argparse.SUPPRESS)  # for automated testing
    parser.add_argument("--debug", action="store_true", help="verbose logging")
    args = parser.parse_args(argv)

    from .logging_setup import attach_db_logging, setup_file_logging, shutdown_logging

    setup_file_logging(paths.logs_dir(), logging.DEBUG if args.debug else logging.INFO)

    lock = SingleInstance(paths.data_dir() / "newstrader.lock")
    if not lock.acquire():
        _show_error(f"{APP_NAME} is already running.\n\nOnly one copy can run at a time "
                    "(so two copies can't place the same trades).")
        return 1

    import uvicorn

    from .api.server import create_app
    from .audio.cuda_setup import setup_cuda_dll_paths
    from .context import build_context

    setup_cuda_dll_paths()  # must happen before anything loads the GPU libraries
    ctx = build_context(token=args.token)
    attach_db_logging(ctx.db, ctx.bus)
    log.info("%s %s starting (data folder: %s)", APP_NAME, __version__, paths.data_dir())

    port = args.port or _free_port()
    app = create_app(ctx)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", log_config=None,
                                           access_log=False, lifespan="on", ws="auto"))
    thread = threading.Thread(target=server.run, name="uvicorn", daemon=True)
    thread.start()

    deadline = time.time() + 60
    while not server.started and thread.is_alive() and time.time() < deadline:
        time.sleep(0.05)
    if not server.started:
        _show_error(f"{APP_NAME} couldn't start its local server. Check {paths.logs_dir() / 'newstrader.log'}.")
        return 1

    url = f"http://127.0.0.1:{port}/?token={ctx.token}"
    try:
        if args.headless:
            print(f"NewsTrader running at {url}", flush=True)
            while thread.is_alive():
                time.sleep(0.5)
        elif args.browser:
            webbrowser.open(url)
            print(f"Opened {url} - press Ctrl+C here to quit.", flush=True)
            while thread.is_alive():
                time.sleep(0.5)
        else:
            _run_window(url)
    except KeyboardInterrupt:
        pass
    finally:
        log.info("Shutting down")
        server.should_exit = True
        thread.join(timeout=20)
        shutdown_logging()
    return 0


def _run_window(url: str) -> None:
    import webview

    webview.settings["ALLOW_DOWNLOADS"] = True
    webview.settings["OPEN_EXTERNAL_LINKS_IN_BROWSER"] = True  # news links open in your normal browser
    webview.create_window(
        f"{APP_NAME} {__version__}",
        url,
        width=1440,
        height=920,
        min_size=(1024, 680),
        background_color="#0b0e14",
        text_select=True,
        confirm_close=True,  # closing the window stops the engine (and trading)
    )
    storage = paths.data_dir() / "webview"
    storage.mkdir(exist_ok=True)
    debug = os.environ.get("NEWSTRADER_DEVTOOLS") == "1"
    webview.start(private_mode=False, storage_path=str(storage), debug=debug)
