"""Turn a page URL (YouTube channel /live, a video, Twitch, any site yt-dlp supports) into a direct media
URL that ffmpeg can read. Plain .m3u8 / .mp3 / Icecast URLs are passed straight through.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..tools import find_deno

log = logging.getLogger(__name__)

DIRECT_SUFFIXES = (".m3u8", ".mp3", ".aac", ".m4a", ".ogg", ".opus", ".wav", ".flac", ".ts", ".mpd")


class StreamOffline(Exception):
    """The channel exists but isn't broadcasting right now."""


@dataclass
class ResolvedStream:
    media_url: str
    title: str = ""
    is_live: bool = True
    headers: dict[str, str] = field(default_factory=dict)


def _ydl_options(cookies_from_browser: str = "") -> dict:
    opts: dict = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
        # lowest-bandwidth format that still has audio; speech doesn't need HD video
        "format": "bestaudio/worst[acodec!=none]/best",
        "socket_timeout": 20,
    }
    deno = find_deno()
    if deno:
        opts["js_runtimes"] = {"deno": {"path": deno}}
    if cookies_from_browser:
        opts["cookiesfrombrowser"] = (cookies_from_browser,)
    return opts


def resolve_stream(url: str, cookies_from_browser: str = "") -> ResolvedStream:
    """Blocking. Raises StreamOffline if the channel isn't live, RuntimeError for other problems."""
    url = url.strip()
    if url.lower().split("?")[0].endswith(DIRECT_SUFFIXES):
        return ResolvedStream(media_url=url, title=url)
    import yt_dlp

    try:
        with yt_dlp.YoutubeDL(_ydl_options(cookies_from_browser)) as ydl:
            info = ydl.extract_info(url, download=False)
    except yt_dlp.utils.DownloadError as exc:
        msg = str(exc)
        low = msg.lower()
        offline_markers = ("not currently live", "is not live", "will begin", "premieres in", "is offline")
        if any(m in low for m in offline_markers):
            raise StreamOffline("not live right now") from exc
        if "sign in to confirm" in low:
            raise RuntimeError("YouTube wants a sign-in check. Set 'YouTube cookies from browser' in "
                               "Settings -> Transcription and make sure you're logged in to YouTube in that browser.") from exc
        raise RuntimeError(msg.replace("ERROR: ", "")[:300]) from exc
    if info is None:
        raise RuntimeError("yt-dlp returned nothing for this URL")
    if info.get("_type") == "playlist":
        entries = [e for e in (info.get("entries") or []) if e]
        if not entries:
            raise StreamOffline("no live video on this channel right now")
        info = entries[0]
    if info.get("live_status") == "is_upcoming":
        raise StreamOffline("scheduled but not live yet")
    media = info.get("url")
    headers = dict(info.get("http_headers") or {})
    if not media:
        fmts = [f for f in (info.get("requested_formats") or info.get("formats") or []) if f.get("url")]
        with_audio = [f for f in fmts if f.get("acodec") not in (None, "none")]
        pick = (with_audio or fmts or [None])[0]
        if pick is None:
            raise RuntimeError("no playable audio format found")
        media = pick["url"]
        headers = dict(pick.get("http_headers") or headers)
    # Recorded videos (not live) are allowed too - handy for testing; they're transcribed once.
    is_live = info.get("live_status") == "is_live" or bool(info.get("is_live"))
    return ResolvedStream(media_url=media, title=info.get("title") or url, is_live=is_live, headers=headers)
