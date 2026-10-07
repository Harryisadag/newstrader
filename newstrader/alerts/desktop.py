"""Desktop pop-up notifications: Windows (via winotify) and Mac (via the built-in osascript).
Does nothing on other systems.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys

from .. import APP_NAME, paths

log = logging.getLogger(__name__)


def safe_toast_text(text: str, limit: int) -> str:
    """winotify puts the text inside a PowerShell double-quoted here-string, where $(...) and backticks would be
    evaluated. News text is untrusted, so strip everything that PowerShell or the XML wrapper could interpret."""
    text = (text or "").replace("\r", " ").replace("\n", " ")
    text = text.replace("$", "\uff04").replace("`", "'").replace('"', "'").replace("]]>", "]] >")
    text = "".join(ch for ch in text if ch.isprintable())
    return text[:limit]


OSASCRIPT = "/usr/bin/osascript"
# The text is passed as command-line arguments to a fixed script, so news text is never parsed as AppleScript.
# The first argument is a fixed word so osascript never mistakes untrusted text for one of its own options.
_MAC_SCRIPT = ["-e", "on run argv", "-e", "display notification (item 3 of argv) with title (item 2 of argv)",
               "-e", "end run"]


def mac_notification_args(title: str, message: str) -> list[str]:
    def clean(text: str, limit: int) -> str:
        text = (text or "").replace("\r", " ").replace("\n", " ")
        return "".join(ch for ch in text if ch.isprintable())[:limit] or " "

    return [OSASCRIPT, *_MAC_SCRIPT, "newstrader", clean(title, 120), clean(message, 250)]


def desktop_supported() -> bool:
    if sys.platform == "darwin":
        return os.path.exists(OSASCRIPT)
    if sys.platform != "win32":
        return False
    try:
        import winotify  # noqa: F401
    except Exception:
        return False
    return True


def show_desktop(title: str, message: str, level: str = "info") -> bool:
    """Blocking (spawns a helper process); call from a worker thread. Returns True if shown."""
    if not desktop_supported():
        return False
    if sys.platform == "darwin":
        try:
            done = subprocess.run(mac_notification_args(title, message), capture_output=True, timeout=10,
                                  check=False)
            return done.returncode == 0
        except Exception as exc:
            log.debug("mac notification failed: %s", exc)
            return False
    try:
        from winotify import Notification, audio

        icon = paths.package_dir() / "web" / "icon.png"
        toast = Notification(app_id=APP_NAME, title=safe_toast_text(title, 120), msg=safe_toast_text(message, 250),
                             icon=str(icon) if icon.exists() else "", duration="long" if level == "error" else "short")
        toast.set_audio(audio.Default if level in ("error", "trade", "warn") else audio.Silent, loop=False)
        toast.show()
        return True
    except Exception as exc:
        log.debug("desktop notification failed: %s", exc)
        return False
