"""Tells you when a newer NewsTrader is out: checks GitHub once at start-up, then once a day.

Nothing is downloaded or installed by itself. The app shows a banner with an Update now button (the ready-made app,
see updater.py) and a link to the right download for this computer, or, when running from the source code, says to
run the update script. It is one small request to api.github.com a day; turn it off in Settings -> Display.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import platform
import re
import sys

import httpx

from . import __version__, paths
from .context import AppContext
from .db import iso

log = logging.getLogger(__name__)

REPO = "Harryisadag/newstrader"
LATEST_URL = f"https://api.github.com/repos/{REPO}/releases/latest"
RELEASES_PAGE = f"https://github.com/{REPO}/releases"
FIRST_CHECK_DELAY = 20  # seconds after start-up, so it doesn't compete with everything else starting
CHECK_EVERY = 24 * 3600
RETRY_AFTER_ERROR = 3600

_VERSION_RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")


def parse_version(value: str | None) -> tuple[int, int, int] | None:
    """"v0.3.1" / "0.3.1" -> (0, 3, 1); anything else -> None."""
    m = _VERSION_RE.match(str(value or "").strip())
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def platform_asset() -> str:
    """Which release zip fits this computer ("" when there is none, e.g. Linux)."""
    if sys.platform == "win32":
        return "windows-x64"
    if sys.platform == "darwin":
        # an Intel build running under Rosetta on an M-series Mac should still be pointed at the native build
        from .tools import mac_info

        return "mac-apple-silicon" if mac_info().get("apple_silicon") or platform.machine() == "arm64" else "mac-intel"
    return ""


def release_info(release: dict, current: str = __version__, asset: str | None = None) -> dict:
    """The parts of a GitHub release the app needs, and whether it is newer than this version."""
    tag = str(release.get("tag_name") or "")
    latest, mine = parse_version(tag), parse_version(current)
    asset = platform_asset() if asset is None else asset
    page = release.get("html_url") or RELEASES_PAGE
    download, size = page, 0
    if asset:
        for a in release.get("assets") or []:
            name = str(a.get("name") or "")
            if name.endswith(f"-{asset}.zip") and a.get("state", "uploaded") == "uploaded":
                download = a.get("browser_download_url") or page
                with contextlib.suppress(TypeError, ValueError):
                    size = int(a.get("size") or 0)
                break
    return {
        "current": current,
        "latest": tag.lstrip("v") if latest else tag,
        "newer": bool(latest and mine and latest > mine),
        "url": page,
        "download": download,
        "size": size,
        "published_at": release.get("published_at"),
        "notes": str(release.get("body") or "").split("\n## ")[0].strip()[:600],
    }


class UpdateChecker:
    name = "updates"

    def __init__(self, ctx: AppContext, client: httpx.AsyncClient | None = None,
                 first_delay: float = FIRST_CHECK_DELAY):
        self.ctx = ctx
        self._client = client
        self.first_delay = first_delay
        self.info: dict | None = None
        self.error = ""
        self.checked_at: str | None = None
        self._task: asyncio.Task | None = None
        self._announced: str | None = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._loop(), name="updates")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task

    def summary(self) -> dict | None:
        """Small dict for the 5-second heartbeat (None until a newer version is known)."""
        if not self.info or not self.info.get("newer"):
            return None
        return {k: self.info.get(k) for k in ("latest", "url", "download", "size")} | {
            "newer": True, "source_mode": not paths.is_frozen()}

    def payload(self) -> dict:
        return {"enabled": self.ctx.config.settings.ui.check_updates, "checked_at": self.checked_at,
                "error": self.error, "source_mode": not paths.is_frozen(), "current": __version__,
                **(self.info or {})}

    async def _loop(self) -> None:
        await asyncio.sleep(self.first_delay)
        while True:
            wait = CHECK_EVERY
            if self.ctx.config.settings.ui.check_updates:
                await self.check()
                if self.error:
                    wait = RETRY_AFTER_ERROR
            await asyncio.sleep(wait)

    async def check(self) -> dict:
        headers = {"Accept": "application/vnd.github+json", "User-Agent": f"NewsTrader/{__version__}"}
        try:
            if self._client is not None:
                r = await self._client.get(LATEST_URL, headers=headers, timeout=15)
            else:
                async with httpx.AsyncClient(follow_redirects=True) as client:
                    r = await client.get(LATEST_URL, headers=headers, timeout=15)
            if r.status_code == 404:
                self.info, self.error = None, ""  # no published release yet
            else:
                r.raise_for_status()
                self.info, self.error = release_info(r.json()), ""
        except Exception as exc:  # offline, rate-limited, GitHub down: try again later, quietly
            self.error = f"couldn't check for updates: {type(exc).__name__}"
            log.warning("Update check failed: %s", exc)
        self.checked_at = iso()
        self._report()
        return self.payload()

    def _report(self) -> None:
        info = self.info or {}
        if self.error:
            self.ctx.state.set_status("updates", "warn", self.error)
        elif info.get("newer"):
            self.ctx.state.set_status("updates", "ok", f"NewsTrader {info['latest']} is available (you have {__version__})")
            if self._announced != info["latest"]:
                self._announced = info["latest"]
                self.ctx.bus.publish("update_available", self.summary())
                updater = self.ctx.service("updater")
                if not paths.is_frozen():
                    how = "Run update.bat / update.command to get it."
                elif updater is not None and updater.supported():
                    how = "Click Update now in the banner at the top."
                else:
                    how = "Download it from the banner at the top."
                self.ctx.bus.publish("toast", {"kind": "info", "title": f"NewsTrader {info['latest']} is out",
                                               "message": how})
        else:
            self.ctx.state.set_status("updates", "ok", f"up to date ({__version__})")
