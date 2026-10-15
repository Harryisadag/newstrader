"""Tells the user plainly when the downloaded app's folder is incomplete, instead of showing a Python error.

That happens when NewsTrader.exe is opened before Windows' "Extract All" has finished (the exe is one of the first
files unpacked, the other ~7,800 follow over a minute or two), when extraction was cancelled, or when an antivirus
removed files. The build writes a list of the app's files (newstrader/app_files.txt); this module only runs after
start-up has already failed, so it costs nothing normally.
"""

from __future__ import annotations

import sys
from pathlib import Path
from zoneinfo import ZoneInfoNotFoundError

MANIFEST = Path("newstrader") / "app_files.txt"
# errors a missing file causes while starting up
_MISSING_KINDS = (FileNotFoundError, ImportError, ZoneInfoNotFoundError)

MESSAGE = ("NewsTrader can't start because its folder is incomplete{detail}.\n\n"
           "This usually happens when NewsTrader.exe is opened before \"Extract All\" has finished, or when the "
           "extraction was stopped.\n\n"
           "To fix it: close this, delete the NewsTrader folder you extracted (your settings and keys are kept "
           "elsewhere), right-click the downloaded zip -> Extract All..., wait until the progress window closes, "
           "then open NewsTrader.exe again.")


def app_dir() -> Path | None:
    """The folder with the app's files (PyInstaller's _internal), or None when running from source."""
    if not getattr(sys, "frozen", False):
        return None
    return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))


def missing_files(base: Path, manifest: Path) -> list[str]:
    names = [ln.strip() for ln in manifest.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [n for n in names if not (base / n).exists()]


def incomplete_message(exc: BaseException, base: Path | None = None) -> str | None:
    """The message to show if `exc` looks like it came from an incomplete app folder, else None."""
    base = base if base is not None else app_dir()
    if base is None:
        return None
    manifest = base / MANIFEST
    if manifest.is_file():
        try:
            missing = missing_files(base, manifest)
        except OSError:
            missing = []
        if missing:
            return MESSAGE.format(detail=f" ({len(missing)} of its files are missing)")
        return None
    # the list itself is missing: only blame the folder for errors a missing file causes
    if isinstance(exc, _MISSING_KINDS):
        return MESSAGE.format(detail="")
    return None


def show_error(text: str) -> None:
    print(text, file=sys.stderr)
    try:
        if sys.platform == "win32":
            import ctypes

            ctypes.windll.user32.MessageBoxW(None, text, "NewsTrader", 0x10)  # MB_ICONERROR
        elif sys.platform == "darwin":
            import subprocess

            script = f"display dialog {_applescript_text(text)} with title \"NewsTrader\" buttons {{\"OK\"}} " \
                     "with icon stop"
            subprocess.run(["osascript", "-e", script], check=False, timeout=600)
    except Exception:  # the message is already on stderr
        pass


def _applescript_text(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def explain_incomplete_install(exc: BaseException) -> bool:
    """Show the plain message and return True if start-up failed because the app folder is incomplete."""
    text = incomplete_message(exc)
    if text is None:
        return False
    show_error(text)
    return True
