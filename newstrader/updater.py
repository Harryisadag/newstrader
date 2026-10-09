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
import posixpath
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
from pathlib import Path

import httpx

from . import __version__, paths, updates
from .context import AppContext

log = logging.getLogger(__name__)

DOWNLOAD_PREFIX = f"https://github.com/{updates.REPO}/releases/download/"
CHUNK = 1 << 20
RETRIES = 4
STATUS_FILE = "update-status.txt"  # the helper writes "ok <version>" or "failed: <why>" here
HELPER_LOG = "update-helper.log"
STAGE_MARKER = ".newstrader-update"  # marks the unpack folder as ours, so only ours is ever deleted
ORDER_WAIT = 15  # seconds to wait for an order that's being placed before giving up on the restart
QUIT_DELAY = 1.5  # seconds, so the window can show "Restarting..."
CLEANUP_DELAY = 30
WINDOWS_EXE = "NewsTrader.exe"
MAC_EXE = "NewsTrader"
DITTO = "/usr/bin/ditto"
# what may sit next to NewsTrader.exe; anything else means the folder holds more than the app, so it isn't replaced
WINDOWS_APP_FILES = {WINDOWS_EXE.lower(), "_internal", ".env", "desktop.ini", "thumbs.db"}
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
        root = exe.parent
        return root if root.name and root.parent != root else None  # never a whole drive
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
    if not os.access(root, os.W_OK):
        return f"Your account isn't allowed to replace {root.name} in {root.parent}."
    return ""


def other_files(root: Path, kind: str) -> list[str]:
    """Things in the Windows app folder that aren't part of NewsTrader (the whole folder gets replaced, so they'd be
    lost). A Mac .app bundle only ever holds the app."""
    if kind != "windows":
        return []
    with contextlib.suppress(OSError):
        return sorted(p.name for p in root.iterdir() if p.name.lower() not in WINDOWS_APP_FILES)
    return []


def _is_old_app(folder: Path) -> bool:
    return (folder / "_internal").is_dir() and (folder / WINDOWS_EXE).is_file()


def make_stage(stage: Path) -> None:
    """A fresh, empty unpack folder. An old one is only removed when it's ours."""
    if stage.exists():
        if not (stage / STAGE_MARKER).is_file():
            raise UpdateError(f"There's already a folder called {stage.name} next to NewsTrader ({stage.parent}). "
                              "Move or rename it, then try again. Nothing was changed.")
        shutil.rmtree(stage, ignore_errors=True)
        if stage.exists():
            raise UpdateError(f"The folder {stage} from an earlier update couldn't be removed. Delete it, then try "
                              "again. Nothing was changed.")
    stage.mkdir()
    (stage / STAGE_MARKER).write_text("NewsTrader unpacks updates here. Safe to delete.\n", encoding="utf-8")


def clear_old(root: Path, kind: str) -> None:
    """Windows: the helper renames the app to <name>.old, so nothing may be there yet."""
    old = old_location(root, kind)
    if kind != "windows" or not old.exists():
        return
    if not _is_old_app(old):
        raise UpdateError(f"There's already a folder called {old.name} next to NewsTrader ({old.parent}). Move or "
                          "rename it, then try again. Nothing was changed.")
    shutil.rmtree(old, ignore_errors=True)
    if old.exists():
        raise UpdateError(f"The folder {old} from an earlier update couldn't be removed. Delete it, then try again. "
                          "Nothing was changed.")


def _safe_parts(name: str) -> tuple[str, ...]:
    parts = tuple(p for p in name.replace("\\", "/").split("/") if p not in ("", "."))
    if name.startswith(("/", "\\")) or not parts or ".." in parts or any(":" in p for p in parts):
        raise UpdateError(f"The update file has an unsafe entry ({name!r}), so nothing was changed.")
    return parts


def _safe_link(parts: tuple[str, ...], target: str) -> bool:
    """A link inside the app may only point at something else inside it (Mac bundles use relative links)."""
    if not target or target.startswith("/") or "\\" in target:
        return False
    resolved = posixpath.normpath(posixpath.join(*parts[:-1], target))
    return resolved != ".." and not resolved.startswith("../") and not resolved.startswith("/")


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
                    if kind != "mac" or not _safe_link(parts, target):
                        raise UpdateError(f"The update file has an unsafe link ({info.filename!r}), so nothing was "
                                          "changed.")
            check_space(dest, sum(i.file_size for i in infos))
            if kind == "mac" and sys.platform == "darwin" and Path(DITTO).exists():
                r = subprocess.run([DITTO, "-x", "-k", str(archive), str(dest)], capture_output=True,
                                   text=True, timeout=900, check=False)
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
    except (zipfile.BadZipFile, EOFError, OSError, subprocess.SubprocessError) as exc:
        raise UpdateError(f"The update couldn't be unpacked ({exc}). Nothing was changed.") from exc


def _read_version_file(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def check_layout(stage: Path, kind: str, version: str) -> Path:
    """The unpacked app, after checking it is a complete NewsTrader <version>. Raises UpdateError otherwise."""
    tops = [p for p in stage.iterdir() if p.name != "__MACOSX" and not p.name.startswith(".")]
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
    if found is None:  # older builds had no version file: the checked zip's name is the version
        log.info("The unpacked update has no version file; going by the zip's name (%s)", version)
    elif found != version:
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
# Plain ASCII on purpose: cmd reads .cmd files in the old DOS code page, which would mangle accented folder names.
# Every path comes in as an NT_* environment variable and is only ever used inside double quotes, so spaces, accents,
# & and brackets in folder names are safe. Messages passed to :log / :status must not contain & | < > ^ or %.
WINDOWS_HELPER = r"""@echo off
rem NewsTrader update helper: waits for NewsTrader to close, swaps in the new version and starts it.
rem If a step fails it puts the old version back and starts that instead.
setlocal EnableExtensions DisableDelayedExpansion
call :log "Waiting for NewsTrader - process %NT_PID% - to close"
set /a waited=0
:wait
%SystemRoot%\System32\tasklist.exe /FI "PID eq %NT_PID%" /FI "IMAGENAME eq %NT_IMAGE%" /FO CSV /NH 2>nul | %SystemRoot%\System32\find.exe /I ".exe" >nul
if errorlevel 1 goto closed
set /a waited+=1
if %waited% GEQ 180 goto still_open
call :pause 2
goto wait

:closed
call :log "NewsTrader has closed"
call :pause 3
if exist "%NT_OLD%\" goto old_in_way
set /a tries=0
:move_old
move "%NT_APP%" "%NT_OLD%" >nul 2>&1
if exist "%NT_OLD%\" goto move_new
set /a tries+=1
if %tries% GEQ 15 goto cant_move_old
call :pause 3
goto move_old

:move_new
set /a tries=0
:move_new_again
if not exist "%NT_APP%\" move "%NT_NEW%" "%NT_APP%" >nul 2>&1
if exist "%NT_APP%\%NT_EXE%" goto swapped
set /a tries+=1
if %tries% GEQ 5 goto put_back
call :pause 3
goto move_new_again

:put_back
call :log "The new version couldn't be moved into place - putting the old one back"
set /a tries=0
:put_back_again
if exist "%NT_OLD%\" if exist "%NT_APP%\" rd /s /q "%NT_APP%" >nul 2>&1
if exist "%NT_OLD%\" if not exist "%NT_APP%\" move "%NT_OLD%" "%NT_APP%" >nul 2>&1
if not exist "%NT_OLD%\" if exist "%NT_APP%\" goto put_back_done
set /a tries+=1
if %tries% GEQ 10 goto put_back_failed
call :pause 3
goto put_back_again

:put_back_done
call :status "failed: the new version couldn't be moved into place, so the old one was put back"
goto start_old

:put_back_failed
call :status "failed: the old version couldn't be put back. It is in the folder ending in .old next to where NewsTrader was - rename it back"
goto done

:swapped
call :status "ok %NT_VERSION%"
call :log "Starting NewsTrader %NT_VERSION%"
start "" /D "%NT_APP%" "%NT_APP%\%NT_EXE%"
rd /s /q "%NT_OLD%" >nul 2>&1
rd /s /q "%NT_STAGE%" >nul 2>&1
goto done

:old_in_way
call :status "failed: a folder from an earlier update was in the way, so nothing was changed"
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

:pause
%SystemRoot%\System32\PING.EXE -n %~1 127.0.0.1 >nul
exit /b 0
"""

MAC_HELPER = """#!/bin/sh
# NewsTrader update helper: waits for NewsTrader to close, swaps in the new version and opens it.
# If a step fails it puts the old version back and opens that instead.
PATH=/usr/bin:/bin:/usr/sbin:/sbin
export PATH
APP={app}
NEW={new}
OLD={old}
STAGE={stage}
HELPER_DIR={helper_dir}
LOG={log}
STATUS={status}
PID={pid}
VERSION={version}
trap 'rm -rf "$HELPER_DIR"' EXIT

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
  finish "failed: macOS wouldn't let the old version be moved, so nothing was changed"
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
"""


def windows_helper(root: Path, new_app: Path, version: str, pid: int, image: str, log_file: Path,
                   status_file: Path, helper_dir: Path) -> tuple[str, dict[str, str]]:
    """The .cmd helper and the environment variables that carry its paths (they never go into the script)."""
    env = {
        "NT_PID": str(int(pid)),
        "NT_IMAGE": image,
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
                 executable: str | None = None, asset: str | None = None, quit_delay: float = QUIT_DELAY):
        self.ctx = ctx
        self._client_factory = client_factory or make_client
        self._launch = launcher or launch_helper
        self._reveal = reveal or reveal_file
        self._kind = kind  # these three are for tests; normally they come from this computer
        self._executable = executable
        self._asset = asset
        self.quit_delay = quit_delay
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
        """Only the ready-made Windows / Mac app updates itself; the source version has update.bat / .command."""
        return paths.is_frozen() and self.kind in ("windows", "mac")

    async def start(self) -> None:
        self._read_last_result()
        if self.last_result:
            self._tasks.append(asyncio.create_task(self._announce_result(), name="update-result"))
        if self.supported():
            self._tasks.append(asyncio.create_task(self._cleanup_later(), name="update-cleanup"))

    async def stop(self) -> None:
        self._cancel.set()  # a download in progress stops at its next chunk
        tasks = [t for t in (*self._tasks, self._task) if t is not None]
        for t in tasks:
            t.cancel()
        for t in tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t

    def summary(self) -> dict:
        return {"phase": self.phase, "busy": self.phase in BUSY_PHASES, "message": self.message, "done": self.done,
                "total": self.total, "version": self.version, "supported": self.supported(),
                "source_mode": not paths.is_frozen(), "last_result": self.last_result}

    # ---- starting
    def _trader(self):
        return self.ctx.service("trader")

    def _trader_busy(self) -> bool:
        trader = self._trader()
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
            self._set("unpacking", "Unpacking and checking the new version...")
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
        """The newest release's files for this computer and the zip's expected SHA-256 (nothing downloaded yet)."""
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
        files = release_files(release, self._asset)
        if files is None:
            raise UpdateError("The newest release has no ready-made download for this computer yet. Use Download "
                              "to get it from the Releases page.")
        _cancelled(self._cancel)
        try:
            r = client.get(files.sha256_url)
        except httpx.HTTPError as exc:
            raise UpdateError("Couldn't reach GitHub to get the update - check the internet connection and try "
                              "again.") from exc
        if r.status_code != 200:
            raise UpdateError(_http_problem(r.status_code))
        if len(r.content) > 4096:
            raise UpdateError("The update's checksum file is damaged, so nothing was downloaded.")
        sha = parse_sha256_file(r.text, files.name)
        if files.digest and files.digest != sha:
            raise UpdateError("GitHub's checksum for the update doesn't match the release's own, so nothing was "
                              "downloaded. Try again later.")
        return files, sha

    def _download(self, client: httpx.Client, files: ReleaseFiles, sha: str) -> Path:
        folder = updates_folder()
        dest = folder / files.name
        if dest.is_file() and dest.stat().st_size == files.size:  # downloaded earlier: check it again
            self._set("verifying", "Checking the earlier download is intact...")
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
        """Unpack the checked zip next to the app and check it. {"manual": message} when the app can't be replaced
        where it is."""
        kind = self.kind
        root = app_root(kind, self._executable)
        if root is None or not root.exists():
            return {"manual": self._manual(zip_path, "Couldn't find NewsTrader's own folder to replace.", kind)}
        if _inside(paths.data_dir(), root):
            return {"manual": self._manual(zip_path, f"Your NewsTrader data folder is inside the app "
                                                     f"({paths.data_dir()}), so replacing the app would delete it.",
                                           kind)}
        extra = other_files(root, kind)
        if extra:
            shown = ", ".join(extra[:3]) + (f" and {len(extra) - 3} more" if len(extra) > 3 else "")
            return {"manual": self._manual(zip_path, f"The folder NewsTrader is in ({root}) also has other things in "
                                                     f"it ({shown}), so Update now won't replace it.", kind)}
        why = why_not_writable(root, kind)
        if why:
            return {"manual": self._manual(zip_path, why, kind)}
        clear_old(root, kind)
        stage = staging_dir(root, kind)
        make_stage(stage)
        try:
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
            with contextlib.suppress(OSError, shutil.Error):
                target = Path(shutil.move(str(zip_path), str(downloads / zip_path.name)))
        with contextlib.suppress(Exception):
            self._reveal(target)
        where = "your Downloads folder" if target.parent == downloads else str(target.parent)
        if kind == "mac":
            steps = ("Close NewsTrader, double-click the zip, then drag the new NewsTrader.app into Applications and "
                     "choose Replace. Keep it in Applications so Update now works next time.")
        else:
            steps = ("Close NewsTrader, right-click the zip -> Extract All, and put the new NewsTrader folder in place "
                     "of the old one.")
        log.warning("Update now can't replace the app here: %s", why)
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
        hold = getattr(self._trader(), "hold_for_update", None)
        if hold is not None:
            hold(True)  # straight after the check, with no await in between, so no new order can start
        try:
            await asyncio.to_thread(self._start_helper, root, new_app, files.version)
        except BaseException:
            if hold is not None:
                hold(False)
            shutil.rmtree(staging_dir(root, self.kind), ignore_errors=True)
            raise
        log.warning("Restarting to install NewsTrader %s", files.version)
        if getattr(self.ctx, "quit_app", None) is None:
            self._set("restarting", "Close NewsTrader to finish the update - the new version then opens by itself.")
            return
        self._set("restarting", f"Restarting into NewsTrader {files.version}...")
        asyncio.get_running_loop().call_later(self.quit_delay, self._quit)

    def _quit(self) -> None:
        """Close the app the normal way (from its own thread, so a slow window can't stall the engine)."""
        loop = asyncio.get_running_loop()

        def run() -> None:
            try:
                self.ctx.quit_app()
            except Exception as exc:
                log.exception("Couldn't close NewsTrader for the update")
                loop.call_soon_threadsafe(self._quit_failed, exc)

        threading.Thread(target=run, name="update-quit", daemon=True).start()

    def _quit_failed(self, exc: Exception) -> None:
        hold = getattr(self._trader(), "hold_for_update", None)
        if hold is not None:
            hold(False)  # trading carries on until you close the app
        self._set("restarting", "NewsTrader couldn't close itself. Close it within 3 minutes to finish the update - "
                                f"the new version then opens by itself ({type(exc).__name__}).")

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
        elif text:
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
        """Remove what an earlier update left behind (best effort, and only things that are clearly ours)."""
        kind = self.kind
        root = app_root(kind, self._executable) if kind else None
        if root is not None:
            stage = staging_dir(root, kind)
            if (stage / STAGE_MARKER).is_file():
                shutil.rmtree(stage, ignore_errors=True)
            old = old_location(root, kind)
            if kind == "windows" and _is_old_app(old):
                shutil.rmtree(old, ignore_errors=True)
        mine = updates.parse_version(__version__)
        for f in updates_folder().iterdir():
            m = re.match(r"^NewsTrader-(\d+\.\d+\.\d+)-", f.name)
            stale = f.name.endswith(".part") or (m and mine and updates.parse_version(m.group(1)) <= mine)
            if stale:
                with contextlib.suppress(OSError):
                    f.unlink()
