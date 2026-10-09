"""Pro AI downloads against a local web server: resume, size and SHA-256 checks, disk space, safe unpacking."""

from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import sys
import tarfile
import threading
import zipfile
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from newstrader.llm import catalog, download
from newstrader.llm.download import DownloadCancelled, DownloadError

POSIX = os.name != "nt"


@dataclass(frozen=True)
class LocalAsset:
    name: str
    size: int
    sha256: str
    url: str


class FileServer:
    """Serves byte strings with Range support, plus knobs to misbehave."""

    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.requests: list[dict] = []
        self.cut_after: dict[str, int] = {}  # send this many bytes of the next answer, then hang up
        self.cut_every: int | None = None  # ... of every answer
        self.no_length = False  # no Content-Length: the body ends when the connection closes
        self.fail_next: list[int] = []  # status codes to answer with before behaving
        self.ignore_range = False
        self.always_416 = False
        self.claimed_length: int | None = None
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                outer.requests.append({"path": self.path, "range": self.headers.get("Range"),
                                       "accept_encoding": self.headers.get("Accept-Encoding")})
                if outer.fail_next:
                    self.send_response(outer.fail_next.pop(0))
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                data = outer.files.get(self.path.lstrip("/"))
                if data is None:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                start = 0
                rng = self.headers.get("Range")
                if rng and outer.always_416:
                    self.send_response(416)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if rng and not outer.ignore_range:
                    start = int(rng.split("=")[1].split("-")[0])
                    if start >= len(data):
                        self.send_response(416)
                        self.send_header("Content-Range", f"bytes */{len(data)}")
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
                else:
                    self.send_response(200)
                body = data[start:]
                if not outer.no_length:
                    self.send_header("Content-Length", str(outer.claimed_length or len(body)))
                self.end_headers()
                cut = outer.cut_after.pop(self.path.lstrip("/"), outer.cut_every)
                self.wfile.write(body[:cut] if cut is not None else body)
                if cut is not None:
                    self.wfile.flush()
                    self.close_connection = True

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, args=(0.02,), daemon=True)
        self.thread.start()

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def add(self, name: str, data: bytes) -> LocalAsset:
        self.files[name] = data
        return LocalAsset(name, len(data), hashlib.sha256(data).hexdigest(), f"{self.base}/{name}")

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def srv():
    s = FileServer()
    yield s
    s.close()


@pytest.fixture
def client():
    with httpx.Client(trust_env=False, timeout=10) as c:  # never through a proxy
        yield c


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(download, "_pause", lambda failures, cancel_event: None)


def payload(n: int = 300_000, seed: int = 1) -> bytes:
    return bytes((i * 7 + seed) % 251 for i in range(n))


# ------------------------------------------------------------------------------------------------ fetch
def test_fetch_checks_and_marks(srv, client, tmp_path):
    asset = srv.add("model.gguf", payload())
    seen = []
    path = download.fetch(asset, tmp_path, progress_cb=lambda d, t: seen.append((d, t)), client=client)
    assert path.read_bytes() == srv.files["model.gguf"]
    assert download.is_verified(path, asset.size, asset.sha256)
    assert not (tmp_path / "model.gguf.part").exists()
    assert seen[-1] == (asset.size, asset.size) and all(d <= t for d, t in seen)
    assert srv.requests[0]["accept_encoding"] == "identity"
    n = len(srv.requests)
    assert download.fetch(asset, tmp_path, client=client) == path  # verified: no new request
    assert len(srv.requests) == n


def test_resume_after_dropped_connection(srv, client, tmp_path):
    asset = srv.add("model.gguf", payload())
    srv.cut_after["model.gguf"] = 100_000
    path = download.fetch(asset, tmp_path, client=client)
    assert path.read_bytes() == srv.files["model.gguf"]
    assert srv.requests[0]["range"] is None
    assert srv.requests[-1]["range"] == "bytes=100000-"


def test_resume_from_existing_part_file(srv, client, tmp_path):
    data = payload()
    asset = srv.add("model.gguf", data)
    (tmp_path / "model.gguf.part").write_bytes(data[:123_456])
    download.fetch(asset, tmp_path, client=client)
    assert [r["range"] for r in srv.requests] == ["bytes=123456-"]
    assert (tmp_path / "model.gguf").read_bytes() == data


def test_server_ignoring_range_starts_over(srv, client, tmp_path):
    data = payload()
    asset = srv.add("model.gguf", data)
    (tmp_path / "model.gguf.part").write_bytes(data[:5000])
    srv.ignore_range = True
    download.fetch(asset, tmp_path, client=client)
    assert (tmp_path / "model.gguf").read_bytes() == data


def test_range_refused_starts_over_then_gives_up(srv, client, tmp_path):
    data = payload()
    asset = srv.add("model.gguf", data)
    (tmp_path / "model.gguf.part").write_bytes(b"x" * 1000)
    srv.always_416 = True
    srv.cut_every = 2000  # every fresh start is cut too, and every resume refused
    with pytest.raises(DownloadError, match="Couldn't resume"):
        download.fetch(asset, tmp_path, client=client)


def test_checksum_mismatch_deletes_the_file(srv, client, tmp_path):
    good = payload()
    asset = srv.add("model.gguf", good)
    srv.files["model.gguf"] = payload(seed=2)  # same size, different bytes
    with pytest.raises(DownloadError, match="safety check"):
        download.fetch(asset, tmp_path, client=client)
    assert not (tmp_path / "model.gguf").exists()
    assert not (tmp_path / "model.gguf.part").exists()


@pytest.mark.parametrize("served", [250_000, 350_000])
def test_size_mismatch_is_refused_at_once(srv, client, tmp_path, served):
    asset = srv.add("model.gguf", payload())
    srv.files["model.gguf"] = payload(served)
    with pytest.raises(DownloadError, match="isn't the pinned file"):
        download.fetch(asset, tmp_path, client=client)
    assert len(srv.requests) == 1
    assert not (tmp_path / "model.gguf.part").exists()


def test_bigger_body_without_length_is_refused(srv, client, tmp_path):
    asset = srv.add("model.gguf", payload())
    srv.files["model.gguf"] = payload(350_000)
    srv.no_length = True
    with pytest.raises(DownloadError, match="bigger than expected"):
        download.fetch(asset, tmp_path, client=client)
    assert not (tmp_path / "model.gguf").exists() and not (tmp_path / "model.gguf.part").exists()


def test_wrong_length_header_is_refused(srv, client, tmp_path):
    asset = srv.add("model.gguf", payload())
    srv.claimed_length = asset.size + 1
    with pytest.raises(DownloadError, match="isn't the pinned file"):
        download.fetch(asset, tmp_path, client=client)


def test_busy_server_is_retried(srv, client, tmp_path):
    asset = srv.add("model.gguf", payload())
    srv.fail_next = [503, 429]
    download.fetch(asset, tmp_path, client=client)
    assert len(srv.requests) == 3


def test_server_errors_give_up_with_plain_message(srv, client, tmp_path):
    asset = srv.add("model.gguf", payload())
    srv.fail_next = [500] * 10
    with pytest.raises(DownloadError, match="had a problem"):
        download.fetch(asset, tmp_path, client=client)
    assert len(srv.requests) == download.RETRIES


def test_missing_file_message(srv, client, tmp_path):
    asset = LocalAsset("gone.gguf", 10, "0" * 64, f"{srv.base}/gone.gguf")
    with pytest.raises(DownloadError, match="404"):
        download.fetch(asset, tmp_path, client=client)


def test_unreachable_server(tmp_path):
    calls = []

    def refuse(request):
        calls.append(request)
        raise httpx.ConnectError("connection refused")

    asset = LocalAsset("x.gguf", 10, "0" * 64, "https://example.invalid/x.gguf")
    with httpx.Client(transport=httpx.MockTransport(refuse)) as c, \
            pytest.raises(DownloadError, match="internet connection"):
        download.fetch(asset, tmp_path, client=c)
    assert len(calls) == download.RETRIES


def test_not_enough_disk_space(srv, client, tmp_path, monkeypatch):
    asset = srv.add("model.gguf", payload())
    monkeypatch.setattr(download.shutil, "disk_usage", lambda p: _usage(1000))
    with pytest.raises(DownloadError, match="Not enough free disk space"):
        download.fetch(asset, tmp_path, client=client)
    assert srv.requests == []


def _usage(free: int):
    from collections import namedtuple

    return namedtuple("usage", "total used free")(10**12, 10**12 - free, free)


def test_cancel(srv, client, tmp_path):
    asset = srv.add("model.gguf", payload())
    cancel = threading.Event()

    def progress(done, total):
        if done:
            cancel.set()

    with pytest.raises(DownloadCancelled):
        download.fetch(asset, tmp_path, progress_cb=progress, cancel_event=cancel, client=client)
    assert not (tmp_path / "model.gguf").exists()


def test_file_copied_in_by_hand_is_checked_once(srv, client, tmp_path):
    data = payload()
    asset = srv.add("model.gguf", data)
    (tmp_path / "model.gguf").write_bytes(data)
    download.fetch(asset, tmp_path, client=client)
    assert srv.requests == []
    assert download.is_verified(tmp_path / "model.gguf", asset.size, asset.sha256)


def test_model_paths_and_delete(tmp_path):
    m = catalog.MODELS[0]
    f = download.model_file(m.key)
    assert f.parent == download.models_folder()
    assert download.model_path(m.key) is None
    f.parent.mkdir(parents=True, exist_ok=True)
    f.with_name(f.name + ".part").write_bytes(b"12345")
    assert download.status()["models"][m.key]["partial_bytes"] == 5
    assert download.delete_model(m.key) == 5


# ------------------------------------------------------------------------------------------------ archives
def _zip(path: Path, entries: list[tuple[str, bytes, int]]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data, mode in entries:
            info = zipfile.ZipInfo(name)
            info.external_attr = mode << 16
            zf.writestr(info, data)
    return path


def _tar(path: Path, members: list[tarfile.TarInfo], data: dict[str, bytes] | None = None) -> Path:
    with tarfile.open(path, "w:gz") as tf:
        for m in members:
            body = (data or {}).get(m.name)
            if body is not None:
                m.size = len(body)
                tf.addfile(m, io.BytesIO(body))
            else:
                tf.addfile(m)
    return path


def _member(name: str, kind=tarfile.REGTYPE, link: str = "", mode: int = 0o644) -> tarfile.TarInfo:
    m = tarfile.TarInfo(name)
    m.type, m.linkname, m.mode = kind, link, mode
    return m


@pytest.mark.parametrize("bad", ["../evil.txt", "/abs/evil.txt", "C:/evil.txt", "bin/../../evil.txt",
                                 "..\\evil.txt", "bin/file.txt:stream"])
def test_zip_traversal_is_refused(tmp_path, bad):
    archive = _zip(tmp_path / "a.zip", [("ok.txt", b"ok", 0o100644), (bad, b"evil", 0o100644)])
    dest = tmp_path / "out"
    with pytest.raises(DownloadError, match="unsafe"):
        download.extract_archive(archive, dest)
    assert not (tmp_path / "evil.txt").exists()


@pytest.mark.parametrize("target", ["/etc/passwd", "../outside", "a/../../outside", "C:\\Windows"])
def test_zip_symlink_out_is_refused(tmp_path, target):
    archive = _zip(tmp_path / "a.zip", [("bin/link", target.encode(), stat.S_IFLNK | 0o777)])
    with pytest.raises(DownloadError, match="unsafe"):
        download.extract_archive(archive, tmp_path / "out")


@pytest.mark.parametrize("member", [
    _member("../evil.txt"), _member("/abs.txt"), _member("bin/link", tarfile.SYMTYPE, "/etc/passwd"),
    _member("bin/link", tarfile.SYMTYPE, "../../outside"), _member("bin/hard", tarfile.LNKTYPE, "../outside"),
    _member("bin/dev", tarfile.CHRTYPE), _member("bin/fifo", tarfile.FIFOTYPE),
])
def test_tar_unsafe_members_are_refused(tmp_path, member):
    data = {member.name: b"x"} if member.type == tarfile.REGTYPE else None
    archive = _tar(tmp_path / "a.tar.gz", [_member("bin/ok.txt"), member], {"bin/ok.txt": b"ok", **(data or {})})
    with pytest.raises(DownloadError, match="unsafe"):
        download.extract_archive(archive, tmp_path / "out")
    assert not (tmp_path / "evil.txt").exists() and not (tmp_path / "abs.txt").exists()


def test_safe_archives_unpack(tmp_path):
    z = _zip(tmp_path / "a.zip", [("build/bin/", b"", stat.S_IFDIR | 0o755),
                                  ("build/bin/llama-server", b"#!/bin/sh\n", 0o100755),
                                  ("build/bin/readme.txt", b"hi", 0o100644)])
    download.extract_archive(z, tmp_path / "z")
    exe = tmp_path / "z" / "build" / "bin" / "llama-server"
    assert exe.read_bytes() == b"#!/bin/sh\n"
    if POSIX:
        assert os.access(exe, os.X_OK)
    members = [_member("llama-b1/llama-server", mode=0o755), _member("llama-b1/libllama.so.0.6.0")]
    if POSIX:
        members.append(_member("llama-b1/libllama.so", tarfile.SYMTYPE, "libllama.so.0.6.0"))
    t = _tar(tmp_path / "a.tar.gz", members, {"llama-b1/llama-server": b"exe", "llama-b1/libllama.so.0.6.0": b"so"})
    download.extract_archive(t, tmp_path / "t")
    assert download.find_server_exe(tmp_path / "t") == tmp_path / "t" / "llama-b1" / "llama-server"
    if POSIX:
        assert (tmp_path / "t" / "llama-b1" / "libllama.so").read_bytes() == b"so"


def test_damaged_archive(tmp_path):
    bad = tmp_path / "a.zip"
    bad.write_bytes(b"PK\x03\x04 not really a zip")
    with pytest.raises(DownloadError, match="damaged"):
        download.extract_archive(bad, tmp_path / "out")


# ------------------------------------------------------------------------------------------------ server install
def test_install_server_end_to_end(srv, client, tmp_path, monkeypatch):
    main = io.BytesIO()
    with zipfile.ZipFile(main, "w") as zf:
        zf.writestr("llama-server.exe", b"MZ fake server")
        zf.writestr("llama-server", b"#!/bin/sh\necho fake\n")
        zf.writestr("ggml-base.dll", b"dll")
    runtime = io.BytesIO()
    with zipfile.ZipFile(runtime, "w") as zf:
        zf.writestr("cudart64_13.dll", b"cuda runtime")
    a1 = srv.add("llama-test-bin-win-cuda-x64.zip", main.getvalue())
    a2 = srv.add("cudart-llama-bin-win-cuda-x64.zip", runtime.getvalue())
    monkeypatch.setattr(catalog, "_RELEASES", srv.base)
    build = catalog.ServerBuild("testbuild", "Test", sys.platform, (
        catalog.Asset(a1.name, a1.size, a1.sha256), catalog.Asset(a2.name, a2.size, a2.sha256)))
    monkeypatch.setitem(catalog.SERVER_BUILDS, "testbuild", build)
    monkeypatch.setattr(download, "prepare_binaries", lambda root, exe, run=None: None)
    seen = []
    exe = download.install_server("testbuild", progress_cb=lambda d, t: seen.append((d, t)), client=client)
    assert exe.is_file() and exe.parent == download.server_folder("testbuild")
    assert (exe.parent / "cudart64_13.dll").read_bytes() == b"cuda runtime"
    assert download.installed_server_exe("testbuild") == exe
    info = json.loads((download.server_folder("testbuild") / download.INSTALLED).read_text())
    assert info["sha256"] == [a1.sha256, a2.sha256]
    assert not list(download.downloads_folder().glob("*.zip"))  # archives removed once unpacked
    assert seen[-1] == (build.download_bytes, build.download_bytes)
    n = len(srv.requests)
    assert download.install_server("testbuild", client=client) == exe and len(srv.requests) == n
    assert download.status()["servers"]["testbuild"]["installed"]
    assert download.delete_server("testbuild") > 0
    assert download.installed_server_exe("testbuild") is None


def test_install_server_without_program_fails_cleanly(srv, client, monkeypatch):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("README.md", b"nothing here")
    a = srv.add("empty.zip", buf.getvalue())
    monkeypatch.setattr(catalog, "_RELEASES", srv.base)
    monkeypatch.setitem(catalog.SERVER_BUILDS, "emptybuild",
                        catalog.ServerBuild("emptybuild", "Empty", "windows", (catalog.Asset(a.name, a.size, a.sha256),)))
    with pytest.raises(DownloadError, match="llama-server"):
        download.install_server("emptybuild", client=client)
    assert not download.server_folder("emptybuild").exists()
    assert not download.server_folder("emptybuild").with_name(download.server_folder("emptybuild").name + ".tmp").exists()


def test_prepare_binaries_on_a_mac(tmp_path, monkeypatch):
    root = tmp_path / "server"
    (root / "bin").mkdir(parents=True)
    exe = root / "bin" / "llama-server"
    exe.write_bytes(b"x")
    (root / "bin" / "libggml-metal.dylib").write_bytes(b"x")
    (root / "bin" / "notes.txt").write_bytes(b"x")
    calls = []

    class Done:
        def __init__(self, code):
            self.returncode = code

    def run(args):
        calls.append(args)
        return Done(1 if args[:2] == ["codesign", "-v"] else 0)

    monkeypatch.setattr(download.sys, "platform", "darwin")
    download.prepare_binaries(root, exe, run=run)
    assert ["xattr", "-dr", "com.apple.quarantine", str(root)] in calls
    signed = [c[-1] for c in calls if c[:2] == ["codesign", "--force"]]
    assert sorted(Path(p).name for p in signed) == ["libggml-metal.dylib", "llama-server"]


def test_catalog_pins_look_right():
    names = set()
    for b in catalog.SERVER_BUILDS.values():
        assert b.assets and b.os in ("windows", "mac", "linux")
        for a in b.assets:
            assert len(a.sha256) == 64 and int(a.sha256, 16) >= 0 and a.size > 0
            assert a.url.startswith("https://github.com/ggml-org/llama.cpp/releases/download/")
            assert a.name.endswith((".zip", ".tar.gz"))
            names.add(a.name)
    for m in catalog.MODELS:
        assert len(m.sha256) == 64 and len(m.revision) == 40 and m.file.endswith(".gguf")
        assert catalog.hf_url(m) == f"https://huggingface.co/{m.repo}/resolve/{m.revision}/{m.file}"
    assert [m.vram_gb for m in catalog.MODELS] == sorted(m.vram_gb for m in catalog.MODELS)
