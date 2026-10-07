"""Windows desktop pop-up notifications (via winotify). Does nothing on other systems."""

from __future__ import annotations

import logging
import sys

from .. import APP_NAME, paths

log = logging.getLogger(__name__)


def desktop_supported() -> bool:
    if sys.platform != "win32":
        return False
    try:
        import winotify  # noqa: F401
    except Exception:
        return False
    return True


def show_desktop(title: str, message: str, level: str = "info") -> bool:
    """Blocking (spawns a hidden PowerShell); call from a worker thread. Returns True if shown."""
    if not desktop_supported():
        return False
    try:
        from winotify import Notification, audio

        icon = paths.package_dir() / "web" / "icon.png"
        toast = Notification(app_id=APP_NAME, title=title[:120], msg=(message or "")[:250],
                             icon=str(icon) if icon.exists() else "", duration="long" if level == "error" else "short")
        toast.set_audio(audio.Default if level in ("error", "trade", "warn") else audio.Silent, loop=False)
        toast.show()
        return True
    except Exception as exc:
        log.debug("desktop notification failed: %s", exc)
        return False
