"""Downloads Pro AI's files - the llama.cpp server and the model - over HTTPS.

Downloads resume after an interruption, are checked against the exact size and SHA-256 pinned in catalog.py (a
broken or tampered file is deleted), and only then renamed into place. A small ".verified" marker next to a checked
file means later starts don't re-read gigabytes. Server archives are unpacked safely (nothing may land outside the
server's own folder).

Layout under the data folder's models/llm/:
    models/<file>.gguf (+ .verified)          the model
    server/<build>-<key>/ (+ .installed)       an unpacked llama.cpp server build
    downloads/                                 server archives while they download (deleted once unpacked)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import posixpath
import shutil
import stat
import subprocess
import sys
import tarfile
import threading
import time
import zipfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

import httpx

from .. import __version__, paths
from .catalog import (
    LLAMA_CPP_BUILD,
    MODELS,
    MODELS_BY_KEY,
    SERVER_BUILDS,
    Asset,
    ModelChoice,
    ServerBuild,
    hf_url,
)

log = logging.getLogger(__name__)

Progress = Callable[[int, int], None]  # (bytes done, bytes total)
CHUNK = 1 << 20
DISK_SLACK = 1.10  # keep 10% more free than the download needs
UNPACK_FACTOR = 2.5  # archive + unpacked files, roughly
RETRIES = 4
MARKER = ".verified"
INSTALLED = ".installed"
CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0
_FILTER_ERROR = getattr(tarfile, "FilterError", type("_NoFilterError", (Exception,), {}))


class DownloadError(Exception):
    """A download or unpack failed; the message is meant for the user."""


class DownloadCancelled(DownloadError):
    pass


# ------------------------------------------------------------------------------------------------ folders
def llm_dir() -> Path:
    path = paths.models_dir() / "llm"
    path.mkdir(parents=True, exist_ok=True)
    return path


def models_folder() -> Path:
    return llm_dir() / "models"


def downloads_folder() -> Path:
    return llm_dir() / "downloads"


def server_folder(key: str) -> Path:
    return llm_dir() / "server" / f"{LLAMA_CPP_BUILD}-{key}"


def _gb(n: float) -> str:
    return f"{n / 1e9:.1f} GB"


def _spec(item: Asset | ModelChoice) -> tuple[str, str, int, str]:
    if isinstance(item, ModelChoice):
        return hf_url(item), item.file, item.size, item.sha256.lower()
    return item.url, item.name, item.size, item.sha256.lower()


# ------------------------------------------------------------------------------------------------ checks
def _marker(path: Path) -> Path:
    return path.with_name(path.name + MARKER)


def is_verified(path: Path, size: int, sha256: str) -> bool:
    """True when `path` was checked before (marker present) and still has the right size."""
    try:
        return (path.is_file() and path.stat().st_size == size
                and _marker(path).read_text(encoding="utf-8").strip().lower() == sha256.lower())
    except OSError:
        return False


def _mark(path: Path, sha256: str) -> None:
    _marker(path).write_text(sha256.lower() + "\n", encoding="utf-8")


def _hash_into(h, path: Path, cancel_event: threading.Event | None = None) -> int:
    n = 0
    with open(path, "rb") as fh:
        while chunk := fh.read(CHUNK):
            if cancel_event is not None and cancel_event.is_set():
                raise DownloadCancelled("Download cancelled")
            h.update(chunk)
            n += len(chunk)
    return n


def sha256_file(path: Path, cancel_event: threading.Event | None = None) -> str:
    h = hashlib.sha256()
    _hash_into(h, path, cancel_event)
    return h.hexdigest()


def check_space(folder: Path, needed: float) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(folder).free
    want = int(needed * DISK_SLACK)
    if free < want:
        raise DownloadError(f"Not enough free disk space: this needs about {_gb(want)}, but only {_gb(free)} is "
                            f"free on the drive with {folder}. Free up some space (or delete a Pro AI model you "
                            "don't use) and try again.")


def _http_message(status: int, url: str) -> str:
    host = urlparse(url).hostname or "the download server"
    if status in (401, 403):
        return f"{host} refused the download (HTTP {status})."
    if status == 404:
        return f"The file is no longer at its pinned address on {host} (HTTP 404). An app update will fix this."
    if status == 429:
        return f"{host} is busy (too many downloads, HTTP 429) - try again in a few minutes."
    if status >= 500:
        return f"{host} had a problem (HTTP {status}) - try again later."
    return f"{host} answered with HTTP {status}."


def make_client() -> httpx.Client:
    return httpx.Client(follow_redirects=True, timeout=httpx.Timeout(30, read=60),
                        headers={"User-Agent": f"NewsTrader/{__version__}"})


# ------------------------------------------------------------------------------------------------ fetch
def fetch(item: Asset | ModelChoice, dest_dir: Path, progress_cb: Progress | None = None,
          cancel_event: threading.Event | None = None, client: httpx.Client | None = None) -> Path:
    """Download one pinned file into dest_dir (resuming a .part file), check its size and SHA-256, and return its
    path. Raises DownloadError (DownloadCancelled when cancel_event is set) with a message for the user."""
    url, name, size, sha = _spec(item)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / name
    if is_verified(dest, size, sha):
        if progress_cb:
            progress_cb(size, size)
        return dest
    if dest.exists():  # e.g. copied in by hand: check it once
        if dest.stat().st_size == size and sha256_file(dest, cancel_event) == sha:
            _mark(dest, sha)
            return dest
        dest.unlink()
    part = dest_dir / (name + ".part")
    have = part.stat().st_size if part.exists() else 0
    if have > size:
        part.unlink()
        have = 0
    check_space(dest_dir, size - have)
    own = client is None
    client = client or make_client()
    try:
        h = _download(client, url, name, part, size, progress_cb, cancel_event)
    finally:
        if own:
            client.close()
    got = part.stat().st_size if part.exists() else 0
    if got != size:
        part.unlink(missing_ok=True)
        raise DownloadError(f"{name} downloaded with the wrong size ({got:,} bytes instead of {size:,}). "
                            "It was deleted - try again.")
    if h.hexdigest() != sha:
        part.unlink(missing_ok=True)
        raise DownloadError(f"{name} failed its safety check (SHA-256 doesn't match), so it was deleted. It was "
                            "probably damaged on the way - try again.")
    os.replace(part, dest)
    _mark(dest, sha)
    return dest


class _Transient(Exception):
    """A server error worth retrying (busy or temporarily broken)."""


def _expected_total(r: httpx.Response) -> int | None:
    """The whole file's size as the server states it (Content-Range, or Content-Length of a plain answer)."""
    if r.status_code == 206:
        total = r.headers.get("content-range", "").rpartition("/")[2]
        return int(total) if total.isdigit() else None
    if r.headers.get("content-encoding", "identity").lower() not in ("", "identity"):
        return None
    length = r.headers.get("content-length", "")
    return int(length) if length.isdigit() else None


def _range_start(r: httpx.Response) -> int | None:
    value = r.headers.get("content-range", "")  # "bytes 100-199/200"
    try:
        return int(value.split()[1].split("-")[0])
    except (IndexError, ValueError):
        return None


def _cancelled(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise DownloadCancelled("Download cancelled")


def _download(client: httpx.Client, url: str, name: str, part: Path, size: int, progress_cb: Progress | None,
              cancel_event: threading.Event | None):
    h = hashlib.sha256()
    have = _hash_into(h, part, cancel_event) if part.exists() else 0  # re-hash what's there from the start
    failures = restarts = 0
    last = 0.0
    host = urlparse(url).hostname or "the download server"
    while have < size:
        _cancelled(cancel_event)
        headers = {"Accept-Encoding": "identity"}
        if have:
            headers["Range"] = f"bytes={have}-"
        before = have
        try:
            with client.stream("GET", url, headers=headers) as r:
                code = r.status_code
                resumed = have and code == 206 and _range_start(r) == have
                if have and not resumed and code in (200, 206, 416):  # can't continue where it stopped: start over
                    restarts += 1
                    if restarts > 2:
                        raise DownloadError(f"Couldn't resume {name}. Delete it in Settings and try again.")
                    part.unlink(missing_ok=True)
                    h, have, before = hashlib.sha256(), 0, 0
                    if code != 200:
                        continue
                if code == 429 or code >= 500:
                    raise _Transient(_http_message(code, url))
                if code >= 400:
                    raise DownloadError(_http_message(code, url))
                total = _expected_total(r)
                if total is not None and total != size:
                    part.unlink(missing_ok=True)
                    raise DownloadError(f"{name} on {host} is {total:,} bytes, not the expected {size:,}, so it "
                                        "isn't the pinned file. Nothing was kept - an app update will fix this.")
                with open(part, "ab" if have else "wb") as fh:
                    for chunk in r.iter_bytes():  # as it arrives, so an interruption loses nothing
                        _cancelled(cancel_event)
                        if have + len(chunk) > size:
                            fh.close()
                            part.unlink(missing_ok=True)
                            raise DownloadError(f"{name} is bigger than expected, so it isn't the pinned file. "
                                                "It was deleted.")
                        fh.write(chunk)
                        h.update(chunk)
                        have += len(chunk)
                        now = time.monotonic()
                        if progress_cb and (now - last >= 0.25 or have == size):
                            last = now
                            progress_cb(have, size)
        except (httpx.TransportError, _Transient) as exc:
            failures = failures + 1 if have == before else 1
            if failures >= RETRIES:
                why = str(exc) if isinstance(exc, _Transient) else (
                    f"Couldn't download {name} from {host} - check the internet connection ({type(exc).__name__}).")
                raise DownloadError(f"{why} Starting again continues where it stopped.") from exc
            log.info("Download of %s interrupted (%s) - resuming", name, exc)
            _pause(failures, cancel_event)
            continue
        if have < size and have == before:  # the server ended the answer without sending anything
            failures += 1
            if failures >= RETRIES:
                raise DownloadError(f"{name} stopped downloading at {have:,} of {size:,} bytes - try again.")
            _pause(failures, cancel_event)
        elif have > before:
            failures = 0
    if progress_cb:
        progress_cb(have, size)
    return h


def _pause(failures: int, cancel_event: threading.Event | None) -> None:
    delay = min(2.0 ** failures, 10.0)
    if cancel_event is not None:
        if cancel_event.wait(delay):
            raise DownloadCancelled("Download cancelled")
    else:
        time.sleep(delay)


# ------------------------------------------------------------------------------------------------ models
def model_file(model: ModelChoice | str) -> Path:
    m = MODELS_BY_KEY[model] if isinstance(model, str) else model
    return models_folder() / m.file


def model_path(model: ModelChoice | str) -> Path | None:
    """The model's file when it is downloaded and verified, else None."""
    m = MODELS_BY_KEY[model] if isinstance(model, str) else model
    f = model_file(m)
    return f if is_verified(f, m.size, m.sha256) else None


def fetch_model(model: ModelChoice | str, progress_cb: Progress | None = None,
                cancel_event: threading.Event | None = None, client: httpx.Client | None = None) -> Path:
    m = MODELS_BY_KEY[model] if isinstance(model, str) else model
    return fetch(m, models_folder(), progress_cb, cancel_event, client)


# ------------------------------------------------------------------------------------------------ archives
def _unsafe(name: str, why: str) -> DownloadError:
    return DownloadError(f"The server download contains an unsafe entry ({name!r}: {why}), so it wasn't unpacked.")


def safe_relpath(name: str) -> PurePosixPath:
    """An archive entry's path, refusing absolute paths, drive letters and '..'."""
    n = name.replace("\\", "/")
    if n.startswith("/"):
        raise _unsafe(name, "absolute path")
    parts = [p for p in n.split("/") if p not in ("", ".")]
    if not parts:
        raise _unsafe(name, "empty path")
    if any(p == ".." for p in parts):
        raise _unsafe(name, "goes up a folder")
    if any(":" in p for p in parts):
        raise _unsafe(name, "drive letter or stream name")
    return PurePosixPath(*parts)


def _check_link(name: str, target: str, base: PurePosixPath | None) -> None:
    """A link may only point at something inside the archive: relative and without '..'."""
    t = target.replace("\\", "/")
    if not t or t.startswith("/") or ":" in t:
        raise _unsafe(name, f"link to {target!r}")
    if any(p == ".." for p in t.split("/")):
        raise _unsafe(name, f"link to {target!r} goes up a folder")
    joined = posixpath.normpath(posixpath.join(str(base) if base else "", t))
    if joined.startswith("..") or joined.startswith("/"):
        raise _unsafe(name, f"link to {target!r} leaves the folder")


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def extract_zip(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        infos = zf.infolist()
        for info in infos:
            rel = safe_relpath(info.filename)
            if stat.S_ISLNK(info.external_attr >> 16):
                _check_link(info.filename, zf.read(info).decode("utf-8", "replace"), rel.parent)
        for info in infos:
            rel = safe_relpath(info.filename)
            out = dest.joinpath(*rel.parts)
            if not _inside(out, dest):
                raise _unsafe(info.filename, "leaves the folder")
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                if os.name != "nt":
                    out.parent.mkdir(parents=True, exist_ok=True)
                    os.symlink(zf.read(info).decode("utf-8"), out)
                continue
            if info.is_dir():
                out.mkdir(parents=True, exist_ok=True)
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst, CHUNK)
            if os.name != "nt" and mode & 0o111:
                os.chmod(out, 0o755)


def extract_tar(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as tf:
        members = tf.getmembers()
        for m in members:
            rel = safe_relpath(m.name)
            if m.issym():
                _check_link(m.name, m.linkname, rel.parent)
            elif m.islnk():
                _check_link(m.name, m.linkname, None)
            elif not (m.isfile() or m.isdir()):
                raise _unsafe(m.name, "special file")
        if hasattr(tarfile, "data_filter"):  # the standard library's own safety filter on top
            tf.extractall(dest, members=members, filter="data")
        else:  # pragma: no cover - Python < 3.11.4
            tf.extractall(dest, members=members)


def extract_archive(archive: Path, dest: Path) -> None:
    name = archive.name.lower()
    try:
        if name.endswith(".zip"):
            extract_zip(archive, dest)
        elif name.endswith((".tar.gz", ".tgz", ".tar.xz", ".tar")):
            extract_tar(archive, dest)
        else:
            raise DownloadError(f"Don't know how to unpack {archive.name}")
    except _FILTER_ERROR as exc:  # (a TarError, so caught first)
        member = getattr(exc, "tarinfo", None)
        raise _unsafe(member.name if member else archive.name, str(exc)) from exc
    except (zipfile.BadZipFile, tarfile.TarError, EOFError) as exc:
        raise DownloadError(f"{archive.name} is damaged and couldn't be unpacked ({exc}).") from exc


# ------------------------------------------------------------------------------------------------ server
def server_exe_names() -> tuple[str, ...]:
    return ("llama-server.exe", "llama-server") if sys.platform == "win32" else ("llama-server", "llama-server.exe")


def find_server_exe(root: Path) -> Path | None:
    """The llama-server program inside an unpacked build (it may sit in a sub-folder)."""
    if not root.is_dir():
        return None
    for name in server_exe_names():
        found = sorted((p for p in root.rglob(name) if p.is_file()), key=lambda p: (len(p.parts), str(p)))
        if found:
            return found[0]
    return None


def _quiet(args: list[str]) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=60, check=False,
                              creationflags=CREATE_NO_WINDOW)
    except Exception as exc:
        log.debug("%s failed: %s", args[0], exc)
        return None


def prepare_binaries(root: Path, exe: Path, run: Callable[[list[str]], object] | None = None) -> None:
    """Make the server runnable: executable bit on Mac/Linux; on a Mac also drop the download quarantine flag and
    ad-hoc sign anything unsigned (Apple Silicon refuses unsigned code). Best effort."""
    if sys.platform == "win32":
        return
    run = run or _quiet
    bin_dir = exe.parent
    targets = [exe] + [p for p in bin_dir.iterdir() if p.is_file() and not p.is_symlink() and p != exe
                       and (p.suffix in (".so", ".dylib") or (p.name.startswith("llama-") and not p.suffix)
                            or ".so." in p.name)]
    for p in targets:
        try:
            p.chmod(p.stat().st_mode | 0o755)
        except OSError:
            pass
    if sys.platform == "darwin":
        run(["xattr", "-dr", "com.apple.quarantine", str(root)])
        for p in targets:
            res = run(["codesign", "-v", str(p)])
            if res is None or getattr(res, "returncode", 1) != 0:
                run(["codesign", "--force", "-s", "-", str(p)])


def _installed_info(key: str) -> dict | None:
    try:
        return json.loads((server_folder(key) / INSTALLED).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def installed_server_exe(key: str) -> Path | None:
    """The llama-server program of an installed build matching today's pins, else None."""
    build = SERVER_BUILDS.get(key)
    info = _installed_info(key)
    if build is None or info is None:
        return None
    if info.get("sha256") != [a.sha256 for a in build.assets]:
        return None
    exe = server_folder(key) / str(info.get("exe", ""))
    return exe if exe.is_file() and _inside(exe, server_folder(key)) else None


def install_server(build: ServerBuild | str, progress_cb: Progress | None = None,
                   cancel_event: threading.Event | None = None, client: httpx.Client | None = None) -> Path:
    """Download and unpack a llama.cpp server build (once); returns the llama-server program's path."""
    build = SERVER_BUILDS[build] if isinstance(build, str) else build
    exe = installed_server_exe(build.key)
    if exe is not None:
        if progress_cb:
            progress_cb(build.download_bytes, build.download_bytes)
        return exe
    check_space(llm_dir(), sum(a.size for a in build.assets
                               if not is_verified(downloads_folder() / a.name, a.size, a.sha256))
                + build.download_bytes * (UNPACK_FACTOR - 1))
    total, done = build.download_bytes, 0
    archives = []
    for asset in build.assets:
        cb = None
        if progress_cb:
            def cb(d: int, _t: int, base: int = done) -> None:
                progress_cb(base + d, total)
        archives.append(fetch(asset, downloads_folder(), cb, cancel_event, client))
        done += asset.size
    target = server_folder(build.key)
    tmp = target.with_name(target.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        extract_archive(archives[0], tmp)
        found = find_server_exe(tmp)
        if found is None:
            raise DownloadError("The llama.cpp download doesn't contain the llama-server program.")
        for extra in archives[1:]:  # runtime libraries go next to the program
            extract_archive(extra, found.parent)
        prepare_binaries(tmp, found)
        if target.exists():
            try:
                shutil.rmtree(target)
            except OSError as exc:
                raise DownloadError("Couldn't replace the old Pro AI server - is it still running? Stop Pro AI "
                                    f"and try again. ({exc})") from exc
        os.replace(tmp, target)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    exe = target / found.relative_to(tmp)
    (target / INSTALLED).write_text(json.dumps({
        "build": LLAMA_CPP_BUILD, "key": build.key, "sha256": [a.sha256 for a in build.assets],
        "exe": exe.relative_to(target).as_posix()}, indent=1), encoding="utf-8")
    for a in archives:  # the unpacked copy is what runs
        a.unlink(missing_ok=True)
        _marker(a).unlink(missing_ok=True)
    return exe


# ------------------------------------------------------------------------------------------------ status
def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if not path.is_dir():
        return 0
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def status() -> dict:
    """What is downloaded and how much disk it uses."""
    servers = {}
    for key, b in SERVER_BUILDS.items():
        d = server_folder(key)
        servers[key] = {"label": b.label, "installed": installed_server_exe(key) is not None,
                        "bytes_on_disk": _size(d), "download_bytes": b.download_bytes}
    old = [p for p in (llm_dir() / "server").glob("*")
           if p.is_dir() and not p.name.startswith(f"{LLAMA_CPP_BUILD}-")] if (llm_dir() / "server").is_dir() else []
    models = {}
    for m in MODELS:
        f = model_file(m)
        part = f.with_name(f.name + ".part")
        models[m.key] = {"label": m.label, "downloaded": is_verified(f, m.size, m.sha256), "size": m.size,
                         "bytes_on_disk": _size(f) + _size(part), "partial_bytes": _size(part)}
    pending = _size(downloads_folder())
    old_bytes = sum(_size(p) for p in old)
    total = (sum(s["bytes_on_disk"] for s in servers.values()) + sum(m["bytes_on_disk"] for m in models.values())
             + pending + old_bytes)
    return {"folder": str(llm_dir()), "build": LLAMA_CPP_BUILD, "servers": servers, "models": models,
            "downloads_in_progress_bytes": pending, "old_server_bytes": old_bytes, "total_bytes": total}


def _remove(path: Path) -> int:
    freed = _size(path)
    try:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        elif path.exists() or path.is_symlink():
            path.unlink()
    except OSError as exc:
        raise DownloadError(f"Couldn't delete {path.name} - is Pro AI still running? Stop it first. ({exc})") from exc
    return freed


def delete_model(key: str) -> int:
    """Delete a downloaded (or half-downloaded) model; returns the bytes freed."""
    f = model_file(key)
    return sum(_remove(p) for p in (f, _marker(f), f.with_name(f.name + ".part")))


def delete_server(key: str) -> int:
    """Delete an installed server build and its leftover downloads; returns the bytes freed."""
    freed = _remove(server_folder(key))
    freed += _remove(server_folder(key).with_name(server_folder(key).name + ".tmp"))
    build = SERVER_BUILDS.get(key)
    for a in build.assets if build else ():
        p = downloads_folder() / a.name
        freed += sum(_remove(x) for x in (p, _marker(p), p.with_name(p.name + ".part")))
    return freed


def delete_old_servers() -> int:
    """Remove server builds left over from an earlier pinned version."""
    base = llm_dir() / "server"
    if not base.is_dir():
        return 0
    return sum(_remove(p) for p in base.iterdir() if p.is_dir() and not p.name.startswith(f"{LLAMA_CPP_BUILD}-"))
