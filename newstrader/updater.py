"""One-click updates for the ready-made app: "Update now" downloads the new version, checks it, swaps it in and restarts.

From the source code, update.bat / update.command do this instead. In the ready-made Windows / Mac app:
1. this computer's zip is downloaded from the newest GitHub release (into the data folder),
2. it is checked against the release's .sha256 file, and GitHub's own checksum when it lists one. A mismatch stops
   here, before anything is touched,
3. it is unpacked into a folder next to the app (same drive) and checked: the app is there, and it's that version,
4. a small helper script is started that waits for this app to close, swaps the old app for the new one (putting
   the old one back if any step fails) and opens it, then
5. the app closes the normal way, which also stops trading and the Pro AI server.

Nothing happens until you click Update now, and it refuses while an order is being placed. Settings, keys, history
and the kill switch live in the data folder (paths.py), so they carry over.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import os
import plistlib
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import httpx

from . import __version__, paths, updates
from .context import AppContext

log = logging.getLogger(__name__)

DOWNLOAD_PREFIX = f"https://github.com/{updates.REPO}/releases/download/"
CHUNK = 1 << 20
RETRIES = 4
UNPACKED_FACTOR = 3.0  # the unpacked app needs about this many times the zip's size
STATUS_FILE = "update-status.txt"  # the helper writes "ok <version>" or "failed: <why>" here
HELPER_LOG = "update-helper.log"
ORDER_WAIT = 15  # seconds to wait for an order that's being placed before giving up on the restart
QUIT_DELAY = 1.5  # seconds, so the window can show "Restarting..."
CLEANUP_DELAY = 30
WINDOWS_EXE = "NewsTrader.exe"
MAC_EXE = "NewsTrader"
BUSY_PHASES = ("checking", "downloading", "verifying", "unpacking", "restarting")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")

CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_BREAKAWAY_FROM_JOB = 0x01000000
CREATE_NO_WINDOW = 0x08000000


class UpdateError(Exception):
    """The update stopped; the message is meant for the user. Nothing was changed unless it says so."""

    status = 400


class UpdateBusy(UpdateError):
    status = 409


class UpdateCancelled(UpdateError):
    pass


@dataclass(frozen=True)
class ReleaseFiles:
    version: str
    name: str  # NewsTrader-<version>-<platform>.zip
    url: str
    size: int
    sha256_url: str
    digest: str = ""  # GitHub's own SHA-256 of the zip, when the API lists one


# ------------------------------------------------------------------------------------------------ the release
def platform_kind() -> str:
    """"windows", "mac", or "" where the app can't update itself."""
    return {"win32": "windows", "darwin": "mac"}.get(sys.platform, "")


def release_files(release: dict, asset: str | None = None) -> ReleaseFiles | None:
    """This computer's zip and its .sha256 file in a GitHub release (None when either is missing)."""
    tag = str(release.get("tag_name") or "")
    asset = updates.platform_asset() if asset is None else asset
    if not asset or updates.parse_version(tag) is None:
        return None
    version = tag.lstrip("v")
    name = f"NewsTrader-{version}-{asset}.zip"
    found = {str(a.get("name") or ""): a for a in release.get("assets") or []
             if a.get("state", "uploaded") == "uploaded"}
    zip_asset, sha_asset = found.get(name), found.get(name + ".sha256")
    if not zip_asset or not sha_asset:
        return None
    url, sha_url = (str(a.get("browser_download_url") or "") for a in (zip_asset, sha_asset))
    if not (url.startswith(DOWNLOAD_PREFIX) and sha_url.startswith(DOWNLOAD_PREFIX)):
        return None  # only ever from this project's own releases
    try:
        size = int(zip_asset.get("size") or 0)
    except (TypeError, ValueError):
        size = 0
    if size <= 0:
        return None
    digest = str(zip_asset.get("digest") or "").lower()
    digest = digest.split(":", 1)[1] if digest.startswith("sha256:") else ""
    return ReleaseFiles(version, name, url, size, sha_url, digest if _SHA_RE.match(digest) else "")


def parse_sha256_file(text: str, name: str) -> str:
    """The checksum in a "<sha256>  <file name>" line, as the release workflow writes it."""
    parts = text.strip().split()
    if not parts or not _SHA_RE.match(parts[0].lower()):
        raise UpdateError("The update's checksum file is damaged, so nothing was downloaded. Try again later.")
    if len(parts) > 1 and parts[1].lstrip("*") != name:
        raise UpdateError("The update's checksum file is for a different file, so nothing was downloaded.")
    return parts[0].lower()


def make_client() -> httpx.Client:
    return httpx.Client(follow_redirects=True, timeout=httpx.Timeout(30, read=60),
                        headers={"User-Agent": f"NewsTrader/{__version__}"})


def _http_problem(code: int) -> str:
    if code == 404:
        return "The update file isn't on GitHub any more (HTTP 404). Check again later."
    if code in (403, 429):
        return f"GitHub is limiting downloads right now (HTTP {code}). Try again in a few minutes."
    return f"GitHub answered with HTTP {code}. Try again later."


def check_space(folder: Path, needed: float) -> None:
    free = shutil.disk_usage(folder).free
    if free < needed * 1.1:
        raise UpdateError(f"Not enough free disk space: the update needs about {needed * 1.1 / 1e9:.1f} GB on the "
                          f"drive with {folder}, but only {free / 1e9:.1f} GB is free. Free up some space and try "
                          "again.")


def _cancelled(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise UpdateCancelled("Update cancelled. Nothing was changed.")


def sha256_file(path: Path, cancel: threading.Event | None = None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(CHUNK):
            _cancelled(cancel)
            h.update(chunk)
    return h.hexdigest()


def download_file(client: httpx.Client, url: str, dest: Path, size: int, sha256: str,
                  progress: Callable[[int, int], None] | None = None, cancel: threading.Event | None = None) -> Path:
    """Download url to dest, continuing after a dropped connection. The file only gets its real name once its size
    and SHA-256 match; otherwise it is deleted and UpdateError says why."""
    part = dest.with_name(dest.name + ".part")
    part.unlink(missing_ok=True)
    h, have, failures = hashlib.sha256(), 0, 0
    try:
        while have < size:
            _cancelled(cancel)
            headers = {"Accept-Encoding": "identity"}
            if have:
                headers["Range"] = f"bytes={have}-"
            before = have
            try:
                with client.stream("GET", url, headers=headers) as r:
                    if r.status_code == 429 or r.status_code >= 500:
                        raise httpx.TransportError(f"HTTP {r.status_code}")
                    if r.status_code >= 400:
                        raise UpdateError(_http_problem(r.status_code))
                    resumed = have and r.status_code == 206 and r.headers.get("content-range", "").startswith(
                        f"bytes {have}-")
                    if have and not resumed:  # the server can't continue where it stopped: start again
                        h, have, before = hashlib.sha256(), 0, 0
                    with open(part, "ab" if have else "wb") as fh:
                        for chunk in r.iter_bytes(CHUNK):
                            _cancelled(cancel)
                            if have + len(chunk) > size:
                                raise UpdateError("The download is bigger than GitHub said it would be, so it was "
                                                  "deleted and nothing was changed. Try again later.")
                            fh.write(chunk)
                            h.update(chunk)
                            have += len(chunk)
                            if progress:
                                progress(have, size)
            except httpx.TransportError as exc:
                failures = failures + 1 if have == before else 1
                if failures >= RETRIES:
                    raise UpdateError("Couldn't download the update - check the internet connection and try "
                                      f"again ({exc}).") from exc
                log.info("Update download interrupted (%s) - continuing", exc)
                if cancel is not None and cancel.wait(min(2.0 ** failures, 10.0)):
                    _cancelled(cancel)
                continue
            if have < size and have == before:
                failures += 1
                if failures >= RETRIES:
                    raise UpdateError("The download kept stopping early. Try again in a few minutes.")
        if h.hexdigest() != sha256:
            raise UpdateError("The download failed its safety check (its SHA-256 checksum doesn't match the "
                              "release's), so it was deleted and nothing was changed. It was probably damaged on "
                              "the way - try again.")
        os.replace(part, dest)
        return dest
    finally:
        part.unlink(missing_ok=True)


# ------------------------------------------------------------------------------------------------ the app on disk
def app_root(kind: str | None = None, executable: str | None = None) -> Path | None:
    """What gets replaced: the folder with NewsTrader.exe (Windows) or the NewsTrader.app bundle (Mac)."""
    kind = platform_kind() if kind is None else kind
    exe = Path(executable or sys.executable).absolute()
    if kind == "windows":
        return exe.parent
    if kind == "mac":
        bundle = exe.parent.parent.parent
        if exe.parent.name == "MacOS" and exe.parent.parent.name == "Contents" and bundle.suffix == ".app":
            return bundle
    return None


def staging_dir(root: Path, kind: str) -> Path:
    """Where the new version is unpacked: next to the app, so the swap is a quick rename on the same drive. (On a
    Mac it's a hidden folder, so Launchpad doesn't pick up the second copy.)"""
    return root.parent / (f".{root.stem}-update" if kind == "mac" else f"{root.name}-update")


def old_location(root: Path, kind: str) -> Path:
    """Where the helper moves the old version while it swaps."""
    return root.with_name(root.name + ".old") if kind == "windows" else staging_dir(root, kind) / f"{root.stem}-old.app"


def _inside(path: Path, root: Path) -> bool:
    try:
        path.absolute().relative_to(root.absolute())
        return True
    except ValueError:
        return False


def why_not_writable(root: Path, kind: str) -> str:
    """"" when this account can replace the app where it is, else why not (in plain English)."""
    if kind == "mac" and "/AppTranslocation/" in str(root):
        return ("macOS is running NewsTrader from a temporary read-only copy, because NewsTrader.app wasn't moved "
                "into the Applications folder.")
    probe = root.parent / f".newstrader-write-test-{os.getpid()}"
    try:
        probe.mkdir(exist_ok=True)
        probe.rmdir()
    except OSError:
        return f"Your account isn't allowed to change the folder NewsTrader is in ({root.parent})."
    if kind == "mac" and not os.access(root, os.W_OK):
        return f"Your account isn't allowed to replace {root.name} in {root.parent}."
    return ""


def _safe_parts(name: str) -> tuple[str, ...]:
    parts = tuple(p for p in name.replace("\\", "/").split("/") if p not in ("", "."))
    if name.startswith(("/", "\\")) or not parts or ".." in parts or any(":" in p for p in parts):
        raise UpdateError(f"The update file has an unsafe entry ({name!r}), so nothing was changed.")
    return parts


def unzip(archive: Path, dest: Path, kind: str) -> None:
    """Unpack the update into dest. Every entry must stay inside dest. On a Mac, ditto unpacks it (as macOS's own
    Archive Utility would), which keeps the app's links, permissions and signature intact."""
    try:
        with zipfile.ZipFile(archive) as zf:
            infos = zf.infolist()
            for info in infos:
                parts = _safe_parts(info.filename)
                if stat.S_ISLNK(info.external_attr >> 16):
                    target = zf.read(info).decode("utf-8", "replace")
                    if kind != "mac" or target.startswith("/") or ".." in PurePosixPath(
                            *parts[:-1], target).parts[: len(parts) - 1] or target.count("..") > len(parts) - 1:
                        raise UpdateError(f"The update file has an unsafe link ({info.filename!r}), so nothing was "
                                          "changed.")
            if kind == "mac" and sys.platform == "darwin" and Path("/usr/bin/ditto").exists():
                zf.close()
                r = subprocess.run(["/usr/bin/ditto", "-x", "-k", str(archive), str(dest)], capture_output=True,
                                   text=True, timeout=600, check=False)
                if r.returncode != 0:
                    raise UpdateError(f"The update couldn't be unpacked ({(r.stderr or '').strip()[:200]}). "
                                      "Nothing was changed.")
                return
            for info in infos:
                out = dest.joinpath(*_safe_parts(info.filename))
                mode = info.external_attr >> 16
                if stat.S_ISLNK(mode):
                    out.parent.mkdir(parents=True, exist_ok=True)
                    os.symlink(zf.read(info).decode("utf-8"), out)
                elif info.is_dir():
                    out.mkdir(parents=True, exist_ok=True)
                else:
                    out.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(info) as src, open(out, "wb") as dst:
                        shutil.copyfileobj(src, dst, CHUNK)
                    if os.name != "nt" and mode & 0o111:
                        os.chmod(out, 0o755)
    except (zipfile.BadZipFile, EOFError, OSError) as exc:
        raise UpdateError(f"The update couldn't be unpacked ({exc}). Nothing was changed.") from exc


def _read_version_file(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def check_layout(stage: Path, kind: str, version: str) -> Path:
    """The unpacked app, after checking it is a complete NewsTrader <version>. Raises UpdateError otherwise."""
    tops = [p for p in stage.iterdir() if p.name != "__MACOSX" and not p.name.startswith("._")]
    bad = UpdateError("The download doesn't contain a complete NewsTrader app, so nothing was changed. Use "
                      "Download to update by hand.")
    if len(tops) != 1 or not tops[0].is_dir():
        raise bad
    top = tops[0]
    if kind == "windows":
        exe = top / WINDOWS_EXE
        if not exe.is_file() or exe.stat().st_size == 0 or not (top / "_internal").is_dir():
            raise bad
        found = _read_version_file(top / "_internal" / "newstrader" / "version.txt")
    else:
        exe = top / "Contents" / "MacOS" / MAC_EXE
        if top.suffix != ".app" or not exe.is_file() or not os.access(exe, os.X_OK):
            raise bad
        try:
            with open(top / "Contents" / "Info.plist", "rb") as fh:
                found = str(plistlib.load(fh).get("CFBundleShortVersionString") or "") or None
        except (OSError, plistlib.InvalidFileException, ValueError) as exc:
            raise bad from exc
    if found is not None and found != version:
        raise UpdateError(f"The download contains NewsTrader {found}, not {version}, so nothing was changed.")
    return top


def carry_over(root: Path, new_app: Path) -> list[str]:
    """Copy things you keep inside the app's folder into the new version: a .env next to NewsTrader.exe."""
    copied = []
    env = paths.env_file()
    if env.is_file() and _inside(env, root):
        rel = env.absolute().relative_to(root.absolute())
        target = new_app / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(env, target)
        copied.append(str(rel))
    return copied


# ------------------------------------------------------------------------------------------------ helper scripts
WINDOWS_HELPER = r"""@echo off
rem NewsTrader update helper: waits for NewsTrader to close, swaps in the new version and starts it.
rem If a step fails it puts the old version back and starts that instead. The folders come in as NT_*
rem environment variables, so names with spaces, brackets or accents are safe, and are always quoted.
setlocal EnableExtensions DisableDelayedExpansion
call :log "Waiting for NewsTrader - process %NT_PID% - to close"
set /a waited=0
:wait
tasklist /FI "PID eq %NT_PID%" /FI "IMAGENAME eq %NT_IMAGE%" /FO CSV /NH 2>nul | find /I "%NT_FIND%" >nul
if errorlevel 1 goto closed
set /a waited+=1
if %waited% GEQ 180 goto still_open
ping -n 2 127.0.0.1 >nul
goto wait

:closed
call :log "NewsTrader has closed"
ping -n 3 127.0.0.1 >nul
if exist "%NT_OLD%\" rd /s /q "%NT_OLD%"
if exist "%NT_OLD%\" goto old_in_way
set /a tries=0
:move_old
move "%NT_APP%" "%NT_OLD%" >nul 2>&1
if exist "%NT_OLD%\" goto move_new
set /a tries+=1
if %tries% GEQ 15 goto cant_move_old
ping -n 3 127.0.0.1 >nul
goto move_old

:move_new
set /a tries=0
:move_new_again
move "%NT_NEW%" "%NT_APP%" >nul 2>&1
if exist "%NT_APP%\%NT_EXE%" goto swapped
if exist "%NT_APP%\" goto put_back
set /a tries+=1
if %tries% GEQ 5 goto put_back
ping -n 3 127.0.0.1 >nul
goto move_new_again

:put_back
call :log "The new version couldn't be moved into place - putting the old one back"
if exist "%NT_APP%\" rd /s /q "%NT_APP%"
set /a tries=0
:put_back_again
move "%NT_OLD%" "%NT_APP%" >nul 2>&1
if exist "%NT_APP%\%NT_IMAGE%" goto put_back_done
set /a tries+=1
if %tries% GEQ 10 goto put_back_failed
ping -n 3 127.0.0.1 >nul
goto put_back_again

:put_back_done
call :status "failed: the new version couldn't be moved into place, so the old one was put back"
goto start_old

:put_back_failed
call :status "failed: the old version couldn't be put back - it is in the folder ending in .old next to where NewsTrader was, rename it back"
goto done

:swapped
call :status "ok %NT_VERSION%"
call :log "Starting NewsTrader %NT_VERSION%"
start "" /D "%NT_APP%" "%NT_APP%\%NT_EXE%"
rd /s /q "%NT_OLD%" >nul 2>&1
rd /s /q "%NT_STAGE%" >nul 2>&1
goto done

:old_in_way
call :status "failed: a .old folder from an earlier update couldn't be removed, so nothing was changed"
goto start_old

:cant_move_old
call :status "failed: Windows wouldn't let the old version be moved - a file in it was still in use - so nothing was changed"
goto start_old

:still_open
call :status "failed: NewsTrader didn't close within 3 minutes, so nothing was changed"
goto done

:start_old
if exist "%NT_APP%\%NT_IMAGE%" start "" /D "%NT_APP%" "%NT_APP%\%NT_IMAGE%"
goto done

:done
call :log "Update helper finished"
(goto) 2>nul & rd /s /q "%NT_TEMP%"
exit /b 0

:log
>>"%NT_LOG%" echo %date% %time% %~1
exit /b 0

:status
>"%NT_STATUS%" echo %~1
call :log "%~1"
exit /b 0
"""

MAC_HELPER = """#!/bin/sh
# NewsTrader update helper: waits for NewsTrader to close, swaps in the new version and opens it.
# If a step fails it puts the old version back and opens that instead.
APP={app}
NEW={new}
OLD={old}
STAGE={stage}
HELPER_DIR={helper_dir}
LOG={log}
STATUS={status}
PID={pid}
VERSION={version}

log() {{ printf '%s %s\\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1" >> "$LOG" 2>/dev/null; }}
finish() {{ printf '%s\\n' "$1" > "$STATUS" 2>/dev/null; log "$1"; }}

log "Waiting for NewsTrader (process $PID) to close"
n=0
while kill -0 "$PID" 2>/dev/null; do
  n=$((n + 1))
  if [ "$n" -ge 360 ]; then
    finish "failed: NewsTrader didn't close within 3 minutes, so nothing was changed"
    exit 1
  fi
  sleep 0.5
done
log "NewsTrader has closed"
sleep 1
rm -rf "$OLD"
if ! mv "$APP" "$OLD"; then
  finish "failed: the old version couldn't be moved aside, so nothing was changed"
  open "$APP"
  exit 1
fi
if ! ditto "$NEW" "$APP" || [ ! -x "$APP/Contents/MacOS/{exe}" ]; then
  log "The new version couldn't be copied into place - putting the old one back"
  rm -rf "$APP"
  if mv "$OLD" "$APP"; then
    finish "failed: the new version couldn't be copied into place, so the old one was put back"
    open "$APP"
  else
    finish "failed: the old version couldn't be put back - it is in $OLD"
  fi
  exit 1
fi
xattr -dr com.apple.quarantine "$APP" 2>/dev/null
finish "ok $VERSION"
log "Opening NewsTrader $VERSION"
open "$APP"
rm -rf "$STAGE"
rm -rf "$HELPER_DIR"
"""


def windows_helper(root: Path, new_app: Path, version: str, pid: int, image: str, log_file: Path,
                   status_file: Path, helper_dir: Path) -> tuple[str, dict[str, str]]:
    """The .cmd helper and the environment variables that carry its paths. The script itself is plain ASCII: cmd
    reads .cmd files in the old DOS code page, which would mangle accented folder names written into it."""
    env = {
        "NT_PID": str(pid),
        "NT_IMAGE": image,
        "NT_FIND": image if image.isascii() else ".exe",
        "NT_EXE": WINDOWS_EXE,
        "NT_APP": str(root),
        "NT_OLD": str(old_location(root, "windows")),
        "NT_NEW": str(new_app),
        "NT_STAGE": str(staging_dir(root, "windows")),
        "NT_VERSION": version,
        "NT_LOG": str(log_file),
        "NT_STATUS": str(status_file),
        "NT_TEMP": str(helper_dir),
    }
    return WINDOWS_HELPER, env


def mac_helper(root: Path, new_app: Path, version: str, pid: int, log_file: Path, status_file: Path,
               helper_dir: Path) -> str:
    """The /bin/sh helper, with every path single-quoted for the shell."""
    q = shlex.quote
    return MAC_HELPER.format(app=q(str(root)), new=q(str(new_app)), old=q(str(old_location(root, "mac"))),
                             stage=q(str(staging_dir(root, "mac"))), helper_dir=q(str(helper_dir)),
                             log=q(str(log_file)), status=q(str(status_file)), pid=int(pid), version=q(version),
                             exe=MAC_EXE)


def helper_env(extra: dict[str, str]) -> dict[str, str]:
    """This app's environment without PyInstaller's own variables, so the new version starts as a fresh app."""
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith(("_PYI_", "_MEIPASS"))}
    env["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
    env.update(extra)
    return env


def launch_helper(kind: str, script: Path, env: dict[str, str], cwd: Path) -> None:
    """Start the helper on its own (no console window on Windows), so it outlives this app."""
    quiet = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL,
             "close_fds": True, "cwd": str(cwd), "env": env}
    if kind == "windows":
        comspec = os.environ.get("COMSPEC") or "cmd.exe"
        command = f'"{comspec}" /d /s /c ""{script}""'  # /s: cmd keeps the inner quotes around the script path
        flags = CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP
        try:
            subprocess.Popen(command, creationflags=flags | CREATE_BREAKAWAY_FROM_JOB, **quiet)
        except OSError:  # running inside a job that doesn't allow it: the helper still outlives a normal exit
            subprocess.Popen(command, creationflags=flags, **quiet)
    else:
        subprocess.Popen(["/bin/sh", str(script)], start_new_session=True, **quiet)


def reveal_file(path: Path) -> None:
    """Show a file in Finder / File Explorer."""
    if sys.platform == "darwin":
        subprocess.Popen(["/usr/bin/open", "-R", str(path)], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    elif sys.platform == "win32":
        subprocess.Popen(f'explorer /select,"{path}"')


def updates_folder() -> Path:
    path = paths.data_dir() / "updates"
    path.mkdir(parents=True, exist_ok=True)
    return path


# ------------------------------------------------------------------------------------------------ the service
class Updater:
    """Runs "Update now". Only ever starts when you click it."""

    name = "updater"

    def __init__(self, ctx: AppContext, client_factory: Callable[[], httpx.Client] | None = None,
                 launcher: Callable[[str, Path, dict, Path], None] | None = None,
                 reveal: Callable[[Path], None] | None = None, kind: str | None = None,
                 executable: str | None = None):
        self.ctx = ctx
        self._client_factory = client_factory or make_client
        self._launch = launcher or launch_helper
        self._reveal = reveal or reveal_file
        self._kind = kind
        self._executable = executable
        self.phase = "idle"
        self.message = ""
        self.done = 0
        self.total = 0
        self.version = ""
        self.last_result: dict | None = None
        self._task: asyncio.Task | None = None
        self._tasks: list[asyncio.Task] = []
        self._cancel = threading.Event()
        self._published = 0.0

    @property
    def kind(self) -> str:
        return platform_kind() if self._kind is None else self._kind

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def supported(self) -> bool:
        return updates.can_update_in_place() and self.kind in ("windows", "mac")

    async def start(self) -> None:
        self._read_last_result()
        if self.last_result:
            self._tasks.append(asyncio.create_task(self._announce_result(), name="update-result"))
        if paths.is_frozen():
            self._tasks.append(asyncio.create_task(self._cleanup_later(), name="update-cleanup"))

    async def stop(self) -> None:
        self._cancel.set()
        for t in [*self._tasks, self._task]:
            if t is not None and not t.done() and self.phase != "restarting":
                t.cancel()
        for t in [*self._tasks, self._task]:
            if t is not None:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await t

    def summary(self) -> dict:
        return {"phase": self.phase, "message": self.message, "done": self.done, "total": self.total,
                "version": self.version, "supported": self.supported(), "last_result": self.last_result}

    # ---- starting
    def _trader_busy(self) -> bool:
        trader = self.ctx.service("trader")
        return bool(trader is not None and getattr(trader, "busy", False))

    async def install(self) -> dict:
        """Start "Update now" in the background (the window follows it through the "update_install" event)."""
        if self.running or self.phase == "restarting":
            return self.summary()
        if not paths.is_frozen():
            raise UpdateError("You're running NewsTrader from the source code: double-click update.bat (Windows) "
                              "or update.command (Mac) to update it.")
        if not self.supported():
            raise UpdateError("This computer can't update NewsTrader by itself. Use Download to get the new "
                              "version from the Releases page.")
        if self._trader_busy():
            raise UpdateBusy("An order is being placed right now, so the update didn't start. Try again in a "
                             "minute - nothing was changed.")
        self._cancel.clear()
        self._set("checking", "Looking up the newest version...", publish=False)
        self._task = asyncio.create_task(self._run(), name="update-install")
        return self.summary()

    def cancel(self) -> dict:
        if self.running and self.phase in ("checking", "downloading", "verifying"):
            self._cancel.set()
        return self.summary()

    # ---- the steps
    async def _run(self) -> None:
        client = self._client_factory()
        try:
            files, sha = await asyncio.to_thread(self._lookup, client)
            self.version = files.version
            zip_path = await asyncio.to_thread(self._download, client, files, sha)
            self._set("unpacking", "Unpacking the new version...")
            prepared = await asyncio.to_thread(self._unpack, files, zip_path)
            if "manual" in prepared:
                self._set("manual", prepared["manual"])
                return
            await self._handoff(files, prepared["root"], prepared["new_app"])
        except UpdateCancelled as exc:
            self._set("idle", str(exc))
        except UpdateError as exc:
            log.warning("Update stopped: %s", exc)
            self._set("error", str(exc))
        except Exception as exc:
            log.exception("Update failed")
            self._set("error", f"The update stopped because of an unexpected problem ({type(exc).__name__}). Nothing "
                               "was changed - try again, or use Download to update by hand.")
        finally:
            with contextlib.suppress(Exception):
                client.close()

    def _lookup(self, client: httpx.Client) -> tuple[ReleaseFiles, str]:
        try:
            r = client.get(updates.LATEST_URL, headers={"Accept": "application/vnd.github+json"})
            release = r.json() if r.status_code == 200 else None
        except (httpx.HTTPError, ValueError) as exc:
            raise UpdateError("Couldn't reach GitHub to get the update - check the internet connection and try "
                              "again.") from exc
        if not isinstance(release, dict):
            raise UpdateError(f"GitHub didn't send the update details (HTTP {r.status_code}). Try again later.")
        latest, mine = updates.parse_version(release.get("tag_name")), updates.parse_version(__version__)
        if not latest or not mine or latest <= mine:
            raise UpdateError(f"You already have the newest version ({__version__}).")
        files = release_files(release)
        if files is None:
            raise UpdateError("The newest release has no ready-made download for this computer yet. Use Download "
                              "to get it from the Releases page.")
        _cancelled(self._cancel)
        try:
            r = client.get(files.sha256_url)
        except httpx.HTTPError as exc:
            raise UpdateError("Couldn't reach GitHub to get the update - check the internet connection and try "
                              "again.") from exc
        if r.status_code != 200 or len(r.content) > 4096:
            raise UpdateError(_http_problem(r.status_code) if r.status_code != 200 else
                              "The update's checksum file is damaged, so nothing was downloaded.")
        sha = parse_sha256_file(r.text, files.name)
        if files.digest and files.digest != sha:
            raise UpdateError("GitHub's checksum for the update doesn't match the release's own, so nothing was "
                              "downloaded. Try again later.")
        return files, sha

    def _download(self, client: httpx.Client, files: ReleaseFiles, sha: str) -> Path:
        folder = updates_folder()
        dest = folder / files.name
        if dest.is_file() and dest.stat().st_size == files.size:  # downloaded earlier: check it again
            self._set("verifying", "Checking the download is intact...")
            if sha256_file(dest, self._cancel) == sha:
                return dest
            dest.unlink()
        check_space(folder, files.size)
        self._set("downloading", f"Downloading NewsTrader {files.version}...")
        self.done, self.total = 0, files.size
        return download_file(client, files.url, dest, files.size, sha, self._progress, self._cancel)

    def _progress(self, done: int, total: int) -> None:
        self.done, self.total = done, total
        now = time.monotonic()
        if now - self._published >= 0.5 or done >= total:
            self._published = now
            self.ctx.bus.publish("update_install", self.summary())

    def _unpack(self, files: ReleaseFiles, zip_path: Path) -> dict:
        kind = self.kind
        root = app_root(kind, self._executable)
        if root is None or not root.exists():
            raise UpdateError("Couldn't find NewsTrader's own folder to update. Use Download to update by hand.")
        if _inside(paths.data_dir(), root):
            raise UpdateError(f"Your NewsTrader data folder is inside the app ({paths.data_dir()}), so replacing "
                              "the app would delete it. Use Download to update by hand.")
        why = why_not_writable(root, kind)
        if why:
            return {"manual": self._manual(zip_path, why, kind)}
        stage = staging_dir(root, kind)
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir()
        try:
            check_space(stage, files.size * UNPACKED_FACTOR)
            unzip(zip_path, stage, kind)
            new_app = check_layout(stage, kind, files.version)
        except BaseException:
            shutil.rmtree(stage, ignore_errors=True)
            raise
        return {"root": root, "new_app": new_app}

    def _manual(self, zip_path: Path, why: str, kind: str) -> str:
        """The app can't replace itself where it is: put the checked zip in Downloads and show it."""
        target = zip_path
        downloads = Path.home() / "Downloads"
        if downloads.is_dir():
            with contextlib.suppress(OSError):
                target = Path(shutil.copy2(zip_path, downloads / zip_path.name))
        with contextlib.suppress(Exception):
            self._reveal(target)
        where = "your Downloads folder" if target.parent == downloads else str(target.parent)
        if kind == "mac":
            steps = ("Close NewsTrader, double-click the zip, then drag the new NewsTrader.app into Applications and "
                     "choose Replace. Keep it in Applications so Update now works next time.")
        else:
            steps = ("Close NewsTrader, right-click the zip -> Extract All, and put the new NewsTrader folder in place "
                     "of the old one.")
        return (f"{why} The new version is downloaded and checked: {target.name} in {where} (a window just opened "
                f"on it). {steps} Your settings and keys stay.")

    async def _handoff(self, files: ReleaseFiles, root: Path, new_app: Path) -> None:
        """Start the helper and close the app - never while an order is being placed."""
        waited = 0.0
        while self._trader_busy() and waited < ORDER_WAIT:
            self._set("unpacking", "Waiting for an order that's being placed to finish...")
            await asyncio.sleep(0.5)
            waited += 0.5
        if self._trader_busy():
            shutil.rmtree(staging_dir(root, self.kind), ignore_errors=True)
            raise UpdateBusy("An order is being placed right now, so NewsTrader didn't restart. Click Update now "
                             "again in a minute (the download is saved) - nothing was changed.")
        trader = self.ctx.service("trader")
        pause = getattr(trader, "pause_for_update", None)
        if pause is not None:
            pause(True)  # straight after the check, so no new order can start in between
        try:
            await asyncio.to_thread(self._start_helper, root, new_app, files.version)
        except BaseException:
            if pause is not None:
                pause(False)
            shutil.rmtree(staging_dir(root, self.kind), ignore_errors=True)
            raise
        log.warning("Restarting to install NewsTrader %s", files.version)
        quit_app = getattr(self.ctx, "quit_app", None)
        if quit_app is None:
            self._set("restarting", "Close NewsTrader to finish the update - the new version then opens by itself.")
            return
        self._set("restarting", f"Restarting into NewsTrader {files.version}...")
        asyncio.get_running_loop().call_later(QUIT_DELAY, quit_app)

    def _start_helper(self, root: Path, new_app: Path, version: str) -> None:
        kind = self.kind
        for name in carry_over(root, new_app):
            log.info("Copied %s into the new version", name)
        helper_dir = Path(tempfile.mkdtemp(prefix="newstrader-update-"))
        log_file = paths.logs_dir() / HELPER_LOG
        status_file = paths.data_dir() / STATUS_FILE
        status_file.unlink(missing_ok=True)
        image = Path(self._executable or sys.executable).name
        if kind == "windows":
            script = helper_dir / "update.cmd"
            text, extra = windows_helper(root, new_app, version, os.getpid(), image, log_file, status_file,
                                         helper_dir)
            with open(script, "w", encoding="ascii", newline="\r\n") as fh:  # cmd wants Windows line endings
                fh.write(text)
        else:
            script = helper_dir / "update.sh"
            extra = {}
            script.write_text(mac_helper(root, new_app, version, os.getpid(), log_file, status_file, helper_dir),
                              encoding="utf-8")
            script.chmod(0o700)
        self._launch(kind, script, helper_env(extra), Path(tempfile.gettempdir()))

    def _set(self, phase: str, message: str, publish: bool = True) -> None:
        self.phase, self.message = phase, message
        if phase != "downloading":
            self.done = self.total = 0
        if publish:
            self.ctx.bus.publish("update_install", self.summary())

    # ---- after a restart
    def _read_last_result(self) -> None:
        f = paths.data_dir() / STATUS_FILE
        try:
            text = f.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            return
        with contextlib.suppress(OSError):
            f.unlink()
        word, _, rest = text.partition(" ")
        if word == "ok":
            if rest.strip() == __version__:
                self.last_result = {"ok": True, "message": f"Updated to NewsTrader {__version__}. Your settings, "
                                                           "keys and history came with it."}
        else:
            why = text.partition(":")[2].strip() or "something went wrong"
            self.last_result = {"ok": False, "message": f"The update didn't finish: {why}. You're on NewsTrader "
                                                        f"{__version__} - try Update now again, or use Download to "
                                                        "update by hand."}
        if self.last_result:
            log.log(logging.INFO if self.last_result["ok"] else logging.WARNING, self.last_result["message"])

    async def _announce_result(self) -> None:
        for _ in range(120):  # wait for the window to connect, so the message is seen
            if self.ctx.bus.subscriber_count:
                break
            await asyncio.sleep(0.5)
        await asyncio.sleep(1)
        res = self.last_result or {}
        self.ctx.bus.publish("toast", {"kind": "success" if res.get("ok") else "warn",
                                       "title": "Update installed" if res.get("ok") else "Update didn't finish",
                                       "message": res.get("message", "")})

    async def _cleanup_later(self) -> None:
        await asyncio.sleep(CLEANUP_DELAY)
        if not self.running and self.phase != "restarting":
            await asyncio.to_thread(self.cleanup)

    def cleanup(self) -> None:
        """Remove what an earlier update left behind (best effort)."""
        kind = self.kind
        root = app_root(kind, self._executable) if kind else None
        if root is not None:
            shutil.rmtree(staging_dir(root, kind), ignore_errors=True)
            old = old_location(root, kind)
            if kind == "windows" and (old / "_internal").is_dir():
                shutil.rmtree(old, ignore_errors=True)
        mine = updates.parse_version(__version__)
        for f in updates_folder().iterdir():
            m = re.match(r"^NewsTrader-(\d+\.\d+\.\d+)-", f.name)
            stale = f.name.endswith(".part") or (m and mine and updates.parse_version(m.group(1)) <= mine)
            if stale:
                with contextlib.suppress(OSError):
                    f.unlink()
