"""The update checker: tells you when a newer release is on GitHub, never installs anything."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from newstrader import __version__
from newstrader import updates as up

RELEASE = {
    "tag_name": "v9.1.0",
    "html_url": "https://github.com/Harryisadag/newstrader/releases/tag/v9.1.0",
    "published_at": "2026-12-01T10:00:00Z",
    "body": "Faster spikes.\n\n## Which file do I need?\n\ntable",
    "assets": [
        {"name": "NewsTrader-9.1.0-windows-x64.zip", "state": "uploaded",
         "browser_download_url": "https://github.com/x/win.zip"},
        {"name": "NewsTrader-9.1.0-windows-x64.zip.sha256", "state": "uploaded",
         "browser_download_url": "https://github.com/x/win.sha"},
        {"name": "NewsTrader-9.1.0-mac-apple-silicon.zip", "state": "uploaded",
         "browser_download_url": "https://github.com/x/arm.zip"},
    ],
}


def test_versions_compare_as_numbers():
    assert up.parse_version("v0.10.0") > up.parse_version("0.9.9")
    assert up.parse_version("0.3.0") == (0, 3, 0)
    assert up.parse_version("v1.0") is None and up.parse_version(None) is None


def test_release_info_picks_this_computers_zip():
    info = up.release_info(RELEASE, current="0.3.0", asset="windows-x64")
    assert info["newer"] and info["latest"] == "9.1.0" and info["download"] == "https://github.com/x/win.zip"
    assert info["notes"] == "Faster spikes."  # only the "what's new" part, not the install guide
    mac_intel = up.release_info(RELEASE, current="0.3.0", asset="mac-intel")
    assert mac_intel["download"] == RELEASE["html_url"]  # no Intel zip -> the release page
    same = up.release_info({**RELEASE, "tag_name": "v0.3.0"}, current="0.3.0", asset="windows-x64")
    assert not same["newer"]
    older = up.release_info({**RELEASE, "tag_name": "v0.2.1"}, current="0.3.0", asset="")
    assert not older["newer"]


def _checker(ctx, handler):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return up.UpdateChecker(ctx, client=client, first_delay=0), client


async def test_newer_release_is_announced_once(ctx):
    ctx.loop = asyncio.get_running_loop()
    ctx.bus.bind_loop(ctx.loop)
    q = ctx.bus.subscribe()
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=RELEASE)

    checker, client = _checker(ctx, handler)
    try:
        res = await checker.check()
        await checker.check()
    finally:
        await client.aclose()
    assert seen[0].url == httpx.URL(up.LATEST_URL) and seen[0].headers["user-agent"] == f"NewsTrader/{__version__}"
    assert res["newer"] and res["latest"] == "9.1.0" and checker.summary()["newer"]
    await asyncio.sleep(0.05)
    events = []
    while not q.empty():
        events.append(q.get_nowait())
    assert sum(1 for e in events if e["type"] == "update_available") == 1  # not repeated every check
    status = {c["component"]: c for c in ctx.state.components()}["updates"]
    assert status["level"] == "ok" and "9.1.0 is available" in status["detail"]


async def test_no_release_or_offline_is_quiet(ctx):
    checker, client = _checker(ctx, lambda r: httpx.Response(404, json={"message": "Not Found"}))
    try:
        res = await checker.check()
    finally:
        await client.aclose()
    assert not res.get("newer") and checker.summary() is None and not checker.error

    def boom(request):
        raise httpx.ConnectError("offline")

    checker, client = _checker(ctx, boom)
    try:
        res = await checker.check()
    finally:
        await client.aclose()
    assert "couldn't check" in res["error"] and checker.summary() is None
    status = {c["component"]: c for c in ctx.state.components()}["updates"]
    assert status["level"] == "warn"


@pytest.mark.parametrize("enabled", [True, False])
def test_updates_api_without_the_service(client, ctx, enabled):
    ctx.config.update({"ui": {"check_updates": enabled}})
    r = client.get("/api/updates")
    assert r.status_code == 200 and r.json()["current"] == __version__ and r.json()["enabled"] is enabled
    assert client.post("/api/updates/check").status_code == 503
