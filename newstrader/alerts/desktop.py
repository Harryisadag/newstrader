"""Windows desktop pop-up notifications (via winotify). Does nothing on other systems."""

from __future__ import annotations

import logging
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
        toast = Notification(app_id=APP_NAME, title=safe_toast_text(title, 120), msg=safe_toast_text(message, 250),
                             icon=str(icon) if icon.exists() else "", duration="long" if level == "error" else "short")
        toast.set_audio(audio.Default if level in ("error", "trade", "warn") else audio.Silent, loop=False)
        toast.show()
        return True
    except Exception as exc:
        log.debug("desktop notification failed: %s", exc)
        return False
