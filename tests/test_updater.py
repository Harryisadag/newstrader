"""Update now: picks this computer's zip, checks it before touching anything, swaps the app with a helper script
(never run here), and refuses from the source code or while an order is being placed."""

from __future__ import annotations

import asyncio
import hashlib
import io
import os
import plistlib
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath

import httpx
import pytest

from newstrader import __version__, paths
from newstrader import updater as upd
from newstrader import updates as up
from newstrader.trading.fake_broker import FakeBroker
from newstrader.trading.trader import UPDATE_HOLD, Trader

NEW = "9.1.0"
BASE = f"https://github.com/Harryisadag/newstrader/releases/download/v{NEW}/"
ASSETS = ("windows-x64", "mac-apple-silicon", "mac-intel")


def release(sizes: dict[str, int] | None = None, digests: dict[str, str] | None = None, tag: str = f"v{NEW}") -> dict:
    assets = []
    for a in ASSETS:
        name = f"NewsTrader-{NEW}-{a}.zip"
        entry = {"name": name, "state": "uploaded", "size": (sizes or {}).get(a, 1000),
                 "browser_download_url": BASE + name}
        if digests and a in digests:
            entry["digest"] = digests[a]
        assets += [entry, {"name": name + ".sha256", "state": "uploaded", "size": 90,
                           "browser_download_url": BASE + name + ".sha256"}]
    return {"tag_name": tag, "html_url": f"https://github.com/Harryisadag/newstrader/releases/tag/{tag}",
            "body": "notes", "assets": assets}


def windows_zip(version: str = NEW, extra: dict[str, bytes] | None = None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("NewsTrader/NewsTrader.exe", b"MZ the new app")
        zf.writestr("NewsTrader/_internal/newstrader/version.txt", version + "\n")
        zf.writestr("NewsTrader/_internal/python312.dll", b"dll")
        for name, data in (extra or {}).items():
            zf.writestr(name, data)
    return buf.getvalue()


def mac_zip(version: str = NEW, link: str | None = None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        exe = zipfile.ZipInfo("NewsTrader.app/Contents/MacOS/NewsTrader")
        exe.external_attr = 0o100755 << 16
        zf.writestr(exe, b"\xcf\xfa\xed\xfe new app")
        zf.writestr("NewsTrader.app/Contents/Info.plist",
                    plistlib.dumps({"CFBundleShortVersionString": version, "CFBundleExecutable": "NewsTrader"}))
        zf.writestr("NewsTrader.app/Contents/Resources/lib.dylib", b"lib")
        if link is not None:
            info = zipfile.ZipInfo("NewsTrader.app/Contents/Frameworks/lib.dylib")
            info.external_attr = 0o120777 << 16
            zf.writestr(info, link)
    return buf.getvalue()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class FakeGitHub:
    """Serves the latest release, its zips and their .sha256 files; remembers what was asked for."""

    def __init__(self, zips: dict[str, bytes], sha_override: dict[str, str] | None = None,
                 digests: dict[str, str] | None = None):
        self.zips = zips
        self.sha_override = sha_override or {}
        self.digests = digests
        self.requests: list[str] = []

    def release(self) -> dict:
        return release({a: len(z) for a, z in self.zips.items()}, self.digests)

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.requests.append(url)
        if url == up.LATEST_URL:
            return httpx.Response(200, json=self.release())
        for a, data in self.zips.items():
            name = f"NewsTrader-{NEW}-{a}.zip"
            if url == BASE + name:
                return httpx.Response(200, content=data)
            if url == BASE + name + ".sha256":
                return httpx.Response(200, text=f"{self.sha_override.get(a, sha(data))}  {name}\n")
        return httpx.Response(404)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler), follow_redirects=True)

    def zip_downloaded(self) -> bool:
        return any(u.endswith(".zip") for u in self.requests)


class FakeTrader:
    def __init__(self, busy: bool = False):
        self.busy = busy
        self.holds: list[bool] = []

    def hold_for_update(self, on: bool) -> None:
        self.holds.append(on)


@pytest.fixture(autouse=True)
def no_real_home(tmp_path, monkeypatch):
    """The fallback puts the zip in ~/Downloads: never the real one."""
    monkeypatch.setattr(upd.Path, "home", classmethod(lambda cls: tmp_path / "home"))


@pytest.fixture
def frozen(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)


@pytest.fixture
def windows_app(tmp_path, monkeypatch):
    """A fake installed Windows app in a folder with a space and accents in its name."""
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    root = tmp_path / "Apps" / "Jos\u00e9 M\u00fcller & Co" / "NewsTrader"
    (root / "_internal").mkdir(parents=True)
    (root / "NewsTrader.exe").write_bytes(b"MZ the old app")
    (root / "_internal" / "python312.dll").write_bytes(b"old dll")
    return root


@pytest.fixture
def mac_app(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path / "tmp"))
    monkeypatch.setattr(upd, "DITTO", str(tmp_path / "no-ditto"))  # the same unpacking on every test machine
    (tmp_path / "tmp").mkdir()
    bundle = tmp_path / "Applications" / "NewsTrader.app"
    (bundle / "Contents" / "MacOS").mkdir(parents=True)
    (bundle / "Contents" / "MacOS" / "NewsTrader").write_bytes(b"old")
    return bundle


def snapshot(folder: Path) -> dict[str, bytes]:
    return {str(p.relative_to(folder)): p.read_bytes() for p in sorted(folder.rglob("*")) if p.is_file()}


def make_updater(ctx, gh: FakeGitHub, kind: str, root: Path, asset: str, launches: list, reveals: list | None = None,
                 trader: FakeTrader | None = None) -> upd.Updater:
    exe = root / "NewsTrader.exe" if kind == "windows" else root / "Contents" / "MacOS" / "NewsTrader"
    if trader is not None:
        ctx.services["trader"] = trader
    u = upd.Updater(ctx, client_factory=gh.client, launcher=lambda *a: launches.append(a),
                    reveal=(reveals.append if reveals is not None else lambda p: None), kind=kind,
                    executable=str(exe), asset=asset, quit_delay=0)
    ctx.services["updater"] = u
    return u


async def run_update(u: upd.Updater) -> None:
    await u.install()
    await u._task


async def wait_for(check, timeout: float = 5) -> None:
    deadline = time.monotonic() + timeout
    while not check():
        assert time.monotonic() < deadline, "timed out"
        await asyncio.sleep(0.02)


# ------------------------------------------------------------------------------------------------ which zip
@pytest.mark.parametrize(("platform", "apple_silicon", "machine", "expected"), [
    ("win32", False, "AMD64", "windows-x64"),
    ("darwin", True, "arm64", "mac-apple-silicon"),
    ("darwin", True, "x86_64", "mac-apple-silicon"),  # an Intel build on an M-series Mac gets the native one
    ("darwin", False, "x86_64", "mac-intel"),
    ("linux", False, "x86_64", ""),
])
def test_asset_for_this_computer(monkeypatch, platform, apple_silicon, machine, expected):
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr("newstrader.tools.mac_info", lambda: {"apple_silicon": apple_silicon})
    monkeypatch.setattr("platform.machine", lambda: machine)
    assert up.platform_asset() == expected
    assert upd.platform_kind() == {"win32": "windows", "darwin": "mac"}.get(platform, "")


@pytest.mark.parametrize("asset", ASSETS)
def test_release_files_pick_the_zip_and_its_checksum(asset):
    digest = "ab" * 32
    files = upd.release_files(release({asset: 1234}, {asset: f"sha256:{digest}"}), asset)
    assert files.name == f"NewsTrader-{NEW}-{asset}.zip" and files.version == NEW and files.size == 1234
    assert files.url == BASE + files.name and files.sha256_url == BASE + files.name + ".sha256"
    assert files.digest == digest


def test_release_files_refuse_anything_odd():
    rel = release()
    assert upd.release_files(rel, "") is None  # Linux: no ready-made app
    no_sha = {**rel, "assets": [a for a in rel["assets"] if not a["name"].endswith(".sha256")]}
    assert upd.release_files(no_sha, "windows-x64") is None
    elsewhere = {**rel, "assets": [{**a, "browser_download_url": a["browser_download_url"].replace(
        "Harryisadag/newstrader", "someone/else")} for a in rel["assets"]]}
    assert upd.release_files(elsewhere, "windows-x64") is None
    half = {**rel, "assets": [{**a, "state": "starter"} for a in rel["assets"]]}
    assert upd.release_files(half, "windows-x64") is None
    assert upd.release_files({**rel, "tag_name": "nightly"}, "windows-x64") is None
    assert upd.release_files(release(digests={"mac-intel": "md5:abc"}), "mac-intel").digest == ""


def test_release_info_reports_the_zip_size():
    info = up.release_info(release({"windows-x64": 5_000_000}), current="0.3.0", asset="windows-x64")
    assert info["size"] == 5_000_000 and info["download"].endswith("windows-x64.zip")


def test_sha256_file_parsing():
    name = "NewsTrader-9.1.0-windows-x64.zip"
    assert upd.parse_sha256_file(f"{'A' * 64}  {name}\n", name) == "a" * 64
    assert upd.parse_sha256_file(f"{'b' * 64} *{name}", name) == "b" * 64
    with pytest.raises(upd.UpdateError, match="different file"):
        upd.parse_sha256_file(f"{'b' * 64}  other.zip", name)
    with pytest.raises(upd.UpdateError, match="damaged"):
        upd.parse_sha256_file("<html>rate limited</html>", name)


# ------------------------------------------------------------------------------------------------ frozen vs source
async def test_source_code_keeps_the_update_script(ctx, client):
    u = upd.Updater(ctx, kind="windows")
    ctx.services["updater"] = u
    assert not u.supported() and u.summary()["source_mode"]
    with pytest.raises(upd.UpdateError, match=r"update\.bat"):
        await u.install()
    r = client.post("/api/updates/install")
    assert r.status_code == 400 and "update.command" in r.json()["detail"]
    assert client.get("/api/updates").json()["install"]["supported"] is False
    assert u.phase == "idle" and u._task is None


async def test_packaged_app_can_update_itself(ctx, frozen):
    assert upd.Updater(ctx, kind="windows").supported() and upd.Updater(ctx, kind="mac").supported()
    linux = upd.Updater(ctx, kind="")
    assert not linux.supported()
    with pytest.raises(upd.UpdateError, match="Download"):
        await linux.install()


def test_update_endpoints_without_the_service(client):
    assert client.post("/api/updates/install").status_code == 503
    assert client.post("/api/updates/cancel").status_code == 503


# ------------------------------------------------------------------------------------------------ the whole flow
async def test_windows_update_downloads_checks_unpacks_and_hands_over(ctx, frozen, windows_app):
    data = windows_zip()
    gh = FakeGitHub({"windows-x64": data}, digests={"windows-x64": f"sha256:{sha(data)}"})
    launches, trader, quits = [], FakeTrader(), []
    ctx.quit_app = lambda: quits.append(True)
    u = make_updater(ctx, gh, "windows", windows_app, "windows-x64", launches, trader=trader)
    await run_update(u)
    assert u.phase == "restarting", u.message
    await wait_for(lambda: quits)
    assert trader.holds == [True]  # no new orders while it restarts

    stage = upd.staging_dir(windows_app, "windows")
    new_app = stage / "NewsTrader"
    assert stage.parent == windows_app.parent  # same drive, so the swap is a rename
    assert (new_app / "NewsTrader.exe").read_bytes() == b"MZ the new app"
    assert (windows_app / "NewsTrader.exe").read_bytes() == b"MZ the old app"  # the helper does the swap, later
    (kind, script, env, cwd), = launches
    assert kind == "windows" and script.name == "update.cmd" and script.parent.parent == Path(str(cwd))
    raw = script.read_bytes()
    assert raw.isascii() and b"\r\n" in raw and b"\n" not in raw.replace(b"\r\n", b"")
    assert env["NT_APP"] == str(windows_app) and env["NT_NEW"] == str(new_app)
    assert env["NT_OLD"] == str(windows_app) + ".old" and env["NT_PID"] == str(os.getpid())
    assert env["NT_VERSION"] == NEW and env["PYINSTALLER_RESET_ENVIRONMENT"] == "1"
    assert not any(k.startswith("_PYI_") for k in env)


async def test_checksum_mismatch_stops_before_anything_changes(ctx, frozen, windows_app):
    data = windows_zip()
    gh = FakeGitHub({"windows-x64": data}, sha_override={"windows-x64": "0" * 64})
    launches, trader, quits = [], FakeTrader(), []
    ctx.quit_app = lambda: quits.append(True)
    before = snapshot(windows_app.parent)
    u = make_updater(ctx, gh, "windows", windows_app, "windows-x64", launches, trader=trader)
    await run_update(u)
    assert u.phase == "error" and "safety check" in u.message and "nothing was changed" in u.message
    assert gh.zip_downloaded()
    assert snapshot(windows_app.parent) == before  # no unpack folder, the app untouched
    assert not upd.staging_dir(windows_app, "windows").exists()
    assert not any(upd.updates_folder().iterdir())  # the bad download is deleted
    assert not launches and not quits and not trader.holds


async def test_github_digest_mismatch_stops_before_downloading(ctx, frozen, windows_app):
    data = windows_zip()
    gh = FakeGitHub({"windows-x64": data}, digests={"windows-x64": f"sha256:{'f' * 64}"})
    launches = []
    u = make_updater(ctx, gh, "windows", windows_app, "windows-x64", launches)
    await run_update(u)
    assert u.phase == "error" and "doesn't match" in u.message
    assert not gh.zip_downloaded() and not launches


async def test_wrong_version_inside_the_zip_is_refused(ctx, frozen, windows_app):
    data = windows_zip(version="9.0.0")
    gh = FakeGitHub({"windows-x64": data})
    launches = []
    u = make_updater(ctx, gh, "windows", windows_app, "windows-x64", launches)
    await run_update(u)
    assert u.phase == "error" and "9.0.0, not 9.1.0" in u.message
    assert not upd.staging_dir(windows_app, "windows").exists() and not launches


async def test_unsafe_zip_entries_are_refused(ctx, frozen, windows_app):
    data = windows_zip(extra={"NewsTrader/../../evil.txt": b"x"})
    gh = FakeGitHub({"windows-x64": data})
    launches = []
    u = make_updater(ctx, gh, "windows", windows_app, "windows-x64", launches)
    await run_update(u)
    assert u.phase == "error" and "unsafe" in u.message
    assert not (windows_app.parent.parent / "evil.txt").exists() and not launches


def test_links_must_stay_inside_the_app():
    parts = ("NewsTrader.app", "Contents", "Frameworks", "lib.dylib")
    assert upd._safe_link(parts, "../Resources/lib.dylib")
    assert not upd._safe_link(parts, "../../../../outside")
    assert not upd._safe_link(parts, "/etc/passwd")
    assert not upd._safe_link(("top",), "..")


async def test_mac_update_unpacks_the_bundle_and_hands_over(ctx, frozen, mac_app):
    data = mac_zip(link=None if os.name == "nt" else "../Resources/lib.dylib")
    gh = FakeGitHub({"mac-apple-silicon": data})
    launches, quits = [], []
    ctx.quit_app = lambda: quits.append(True)
    u = make_updater(ctx, gh, "mac", mac_app, "mac-apple-silicon", launches)
    await run_update(u)
    assert u.phase == "restarting", u.message
    stage = upd.staging_dir(mac_app, "mac")
    assert stage.name.startswith(".") and stage.parent == mac_app.parent
    assert (stage / "NewsTrader.app" / "Contents" / "MacOS" / "NewsTrader").is_file()
    (kind, script, env, _cwd), = launches
    text = script.read_text(encoding="utf-8")
    assert kind == "mac" and script.name == "update.sh" and text.startswith("#!/bin/sh")
    assert f"PID={os.getpid()}" in text
    await wait_for(lambda: quits)


async def test_mac_link_out_of_the_bundle_is_refused(ctx, frozen, mac_app):
    gh = FakeGitHub({"mac-intel": mac_zip(link="../../../../../../etc/passwd")})
    launches = []
    u = make_updater(ctx, gh, "mac", mac_app, "mac-intel", launches)
    await run_update(u)
    assert u.phase == "error" and "unsafe link" in u.message and not launches


async def test_folder_you_cant_change_falls_back_to_the_zip(ctx, frozen, mac_app, tmp_path, monkeypatch):
    (tmp_path / "home" / "Downloads").mkdir(parents=True)
    monkeypatch.setattr(upd, "why_not_writable", lambda root, kind: "Your account isn't allowed to change it.")
    data = mac_zip()
    gh = FakeGitHub({"mac-apple-silicon": data})
    launches, reveals = [], []
    u = make_updater(ctx, gh, "mac", mac_app, "mac-apple-silicon", launches, reveals)
    await run_update(u)
    assert u.phase == "manual" and "Downloads" in u.message and "Applications" in u.message
    shown, = reveals
    assert shown == tmp_path / "home" / "Downloads" / f"NewsTrader-{NEW}-mac-apple-silicon.zip"
    assert sha(shown.read_bytes()) == sha(data)
    assert not launches and not upd.staging_dir(mac_app, "mac").exists()


async def test_windows_folder_with_other_files_is_not_replaced(ctx, frozen, windows_app):
    (windows_app / "my notes.txt").write_text("keep me")
    gh = FakeGitHub({"windows-x64": windows_zip()})
    launches, reveals = [], []
    u = make_updater(ctx, gh, "windows", windows_app, "windows-x64", launches, reveals)
    await run_update(u)
    assert u.phase == "manual" and "my notes.txt" in u.message and reveals and not launches
    assert (windows_app / "my notes.txt").read_text() == "keep me"


def test_folders_that_arent_ours_are_never_deleted(windows_app):
    stage = upd.staging_dir(windows_app, "windows")
    stage.mkdir()
    (stage / "precious.txt").write_text("x")
    with pytest.raises(upd.UpdateError, match="Move or rename"):
        upd.make_stage(stage)
    assert (stage / "precious.txt").exists()
    old = upd.old_location(windows_app, "windows")
    old.mkdir()
    with pytest.raises(upd.UpdateError, match="Move or rename"):
        upd.clear_old(windows_app, "windows")
    assert old.exists()
    assert upd.app_root("windows", str(PurePosixPath("/NewsTrader.exe"))) is None  # never a whole drive


# ------------------------------------------------------------------------------------------------ never mid-order
# (client comes before frozen, so the test app finds its web files the normal way)
async def test_refuses_while_an_order_is_being_placed(ctx, client, frozen, windows_app):
    gh = FakeGitHub({"windows-x64": windows_zip()})
    launches = []
    u = make_updater(ctx, gh, "windows", windows_app, "windows-x64", launches, trader=FakeTrader(busy=True))
    with pytest.raises(upd.UpdateBusy, match="order is being placed"):
        await u.install()
    r = client.post("/api/updates/install")
    assert r.status_code == 409 and "nothing was changed" in r.json()["detail"]
    assert not gh.requests and not launches


async def test_order_starting_during_the_download_stops_the_restart(ctx, frozen, windows_app, monkeypatch):
    monkeypatch.setattr(upd, "ORDER_WAIT", 0)
    gh = FakeGitHub({"windows-x64": windows_zip()})
    launches, trader, quits = [], FakeTrader(), []
    ctx.quit_app = lambda: quits.append(True)
    u = make_updater(ctx, gh, "windows", windows_app, "windows-x64", launches, trader=trader)
    await u.install()
    trader.busy = True
    await u._task
    assert u.phase == "error" and "didn't restart" in u.message
    assert not launches and not quits and not trader.holds
    assert not upd.staging_dir(windows_app, "windows").exists()
    assert (windows_app / "NewsTrader.exe").read_bytes() == b"MZ the old app"


async def test_trader_reports_busy_and_holds_orders_for_the_restart(ctx):
    t = Trader(ctx, broker_factory=FakeBroker)
    await t.connect()
    try:
        assert not t.busy
        async with t._lock:
            assert t.busy
        t.hold_for_update(True)
        res = await t.handle_signal({"id": 1, "ticker": "AAPL", "direction": "bullish", "confidence": 90})
        assert res["action"] == "blocked" and res["reason"] == UPDATE_HOLD and not t.broker.calls
        with pytest.raises(RuntimeError, match="restarting"):
            await t.manual_close("AAPL")
        t.hold_for_update(False)
        assert (await t.handle_signal({"id": 2, "ticker": "AAPL", "direction": "bullish", "confidence": 90}))["traded"]
    finally:
        await t.stop()


# ------------------------------------------------------------------------------------------------ helper scripts
PATH_VARS = ("NT_APP", "NT_OLD", "NT_NEW", "NT_STAGE", "NT_LOG", "NT_STATUS", "NT_TEMP")


def test_windows_helper_keeps_paths_out_of_the_script_and_quoted():
    root = PureWindowsPath("C:/Users/Jos\u00e9 M\u00fcller/My Apps & (old)/NewsTrader")
    new_app = PureWindowsPath(str(root) + "-update") / "NewsTrader"
    log_file = PureWindowsPath("C:/Users/Jos\u00e9 M\u00fcller/AppData/Local/NewsTrader/logs/update-helper.log")
    status = PureWindowsPath("C:/Users/Jos\u00e9 M\u00fcller/AppData/Local/NewsTrader/update-status.txt")
    helper_dir = PureWindowsPath("C:/Users/Jos\u00e9 M\u00fcller/AppData/Local/Temp/newstrader-update-x")
    script, env = upd.windows_helper(root, new_app, NEW, 4242, "NewsTrader.exe", log_file, status, helper_dir)
    assert script.isascii() and "M\u00fcller" not in script and "Apps" not in script
    assert env["NT_APP"] == "C:\\Users\\Jos\u00e9 M\u00fcller\\My Apps & (old)\\NewsTrader"
    assert env["NT_OLD"] == env["NT_APP"] + ".old" and env["NT_STAGE"] == env["NT_APP"] + "-update"
    assert env["NT_NEW"] == str(new_app) and env["NT_PID"] == "4242" and env["NT_TEMP"] == str(helper_dir)
    for line in script.splitlines():
        outside_quotes = re.sub(r'"[^"]*"', "", line)
        for var in PATH_VARS:
            assert f"%{var}%" not in outside_quotes, line
    for message in re.findall(r'call :(?:status|log) "([^"]*)"', script):
        assert not re.search(r"[&|<>^]", message) and message.replace("%NT_PID%", "").replace(
            "%NT_VERSION%", "").replace("%~1", "").count("%") == 0, message
    steps = ['tasklist.exe /FI "PID eq %NT_PID%"', 'move "%NT_APP%" "%NT_OLD%"', 'move "%NT_NEW%" "%NT_APP%"',
             'start "" /D "%NT_APP%" "%NT_APP%\\%NT_EXE%"', 'rd /s /q "%NT_OLD%"']
    assert [script.index(s) for s in steps] == sorted(script.index(s) for s in steps)  # in that order
    put_back = script[script.index(":put_back\n"):script.index(":put_back_done")]
    assert 'move "%NT_OLD%" "%NT_APP%"' in put_back
    assert 'start "" /D "%NT_APP%" "%NT_APP%\\%NT_IMAGE%"' in script[script.index(":start_old"):]


NASTY_MAC = "/Users/j\u00f6rg/Apps/It's \"mine\" $HOME `id` & more/NewsTrader.app"


def test_mac_helper_quotes_every_path():
    root = PurePosixPath(NASTY_MAC)
    new_app = upd.staging_dir(root, "mac") / "NewsTrader.app"
    text = upd.mac_helper(root, new_app, NEW, 4242, PurePosixPath("/tmp/log file.txt"),
                          PurePosixPath("/tmp/st\u00e4tus.txt"), PurePosixPath("/tmp/helper dir"))
    for line in text.splitlines():
        if re.match(r"^[A-Z_]+=", line):
            continue  # the assignments themselves: checked with a real shell below
        outside_quotes = re.sub(r"'[^']*'", "", re.sub(r'"[^"]*"', "", line))
        assert not re.search(r"\$(APP|NEW|OLD|STAGE|LOG|STATUS|HELPER_DIR)\b", outside_quotes), line
    for step in ('kill -0 "$PID"', 'mv "$APP" "$OLD"', 'ditto "$NEW" "$APP"', 'xattr -dr com.apple.quarantine "$APP"',
                 'open "$APP"', 'mv "$OLD" "$APP"'):
        assert step in text
    assert text.index('ditto "$NEW" "$APP"') < text.index("xattr -dr") < text.rindex('open "$APP"')


@pytest.mark.skipif(not Path("/bin/sh").exists(), reason="needs a POSIX shell")
def test_mac_helper_paths_survive_the_shell():
    root = PurePosixPath(NASTY_MAC)
    new_app = upd.staging_dir(root, "mac") / "NewsTrader.app"
    text = upd.mac_helper(root, new_app, NEW, 4242, PurePosixPath("/tmp/log file.txt"),
                          PurePosixPath("/tmp/st\u00e4tus.txt"), PurePosixPath("/tmp/helper dir"))
    assert subprocess.run(["/bin/sh", "-n"], input=text.encode(), check=False).returncode == 0  # syntax only
    assignments = "\n".join(line for line in text.splitlines() if re.match(r"^[A-Z_]+=", line) and
                            not line.startswith("PATH="))
    out = subprocess.run(["/bin/sh", "-c", assignments + '\nprintf "%s\\n" "$APP" "$NEW" "$OLD" "$STATUS" "$VERSION"'],
                         capture_output=True, check=True).stdout.decode()
    assert out.splitlines() == [NASTY_MAC, str(new_app), str(upd.old_location(root, "mac")),
                                "/tmp/st\u00e4tus.txt", NEW]


# ------------------------------------------------------------------------------------------------ after a restart
async def test_result_of_the_last_update_is_reported_once(ctx):
    status = paths.data_dir() / upd.STATUS_FILE
    status.write_text(f"ok {__version__}\n")
    u = upd.Updater(ctx, kind="windows")
    await u.start()
    try:
        assert u.last_result["ok"] and __version__ in u.last_result["message"] and not status.exists()
    finally:
        await u.stop()
    status.write_text("failed: Windows wouldn't let the old version be moved\n")
    u = upd.Updater(ctx, kind="windows")
    u._read_last_result()
    assert not u.last_result["ok"] and "wouldn't let the old version be moved" in u.last_result["message"]


def test_cleanup_removes_only_our_leftovers(ctx, windows_app):
    stage = upd.staging_dir(windows_app, "windows")
    upd.make_stage(stage)
    old = upd.old_location(windows_app, "windows")
    shutil.copytree(windows_app, old)
    folder = upd.updates_folder()
    (folder / "NewsTrader-0.0.1-windows-x64.zip").write_bytes(b"old")
    (folder / f"NewsTrader-{NEW}-windows-x64.zip.part").write_bytes(b"half")
    (folder / f"NewsTrader-{NEW}-windows-x64.zip").write_bytes(b"next")
    upd.Updater(ctx, kind="windows", executable=str(windows_app / "NewsTrader.exe")).cleanup()
    assert not stage.exists() and not old.exists() and windows_app.exists()
    assert sorted(p.name for p in folder.iterdir()) == [f"NewsTrader-{NEW}-windows-x64.zip"]


def test_quit_closes_the_window_without_asking():
    from types import SimpleNamespace

    from newstrader.app import _Quitter

    server = SimpleNamespace(should_exit=False)
    window = SimpleNamespace(confirm_close=True, destroyed=False)
    window.destroy = lambda: setattr(window, "destroyed", True)
    q = _Quitter(server)
    q.window = window
    q()
    assert window.destroyed and not window.confirm_close and q.called
    headless = _Quitter(server)
    headless()
    assert server.should_exit  # no window (--browser / --headless): stop the server the normal way


async def test_stopping_never_waits_for_the_leftover_cleanup(ctx, frozen):
    u = upd.Updater(ctx, kind="windows")
    await u.start()  # schedules the cleanup of an earlier update's leftovers for later
    u.phase = "restarting"
    await asyncio.wait_for(u.stop(), 2)  # the normal shutdown before the swap isn't held up
    assert all(t.done() for t in u._tasks)
