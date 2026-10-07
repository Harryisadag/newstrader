"""Pull audio out of a stream with ffmpeg: 16 kHz mono 16-bit PCM on stdout, read by a background thread."""

from __future__ import annotations

import logging
import subprocess
import threading
from collections.abc import Callable

from ..tools import CREATE_NO_WINDOW, find_ffmpeg

log = logging.getLogger(__name__)

READ_BYTES = 16000 * 2 // 2  # half a second of audio per read


def ffmpeg_command(ffmpeg: str, media_url: str, headers: dict[str, str] | None = None, realtime: bool = False) -> list[str]:
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin"]
    if media_url.startswith(("http://", "https://")):
        cmd += ["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "10", "-rw_timeout", "30000000"]
        headers = dict(headers or {})
        ua = headers.pop("User-Agent", None)
        if ua:
            cmd += ["-user_agent", ua]
        extra = "".join(f"{k}: {v}\r\n" for k, v in headers.items() if k.lower() not in ("accept-encoding",))
        if extra:
            cmd += ["-headers", extra]
    if realtime:
        cmd += ["-re"]
    cmd += ["-i", media_url, "-vn", "-sn", "-dn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-acodec", "pcm_s16le", "pipe:1"]
    return cmd


class FfmpegCapture:
    """Runs ffmpeg and calls on_audio(bytes) from a reader thread until stopped or the stream ends."""

    def __init__(self, media_url: str, on_audio: Callable[[bytes], None], headers: dict[str, str] | None = None,
                 realtime: bool = False, name: str = "stream"):
        self.media_url = media_url
        self.on_audio = on_audio
        self.headers = headers or {}
        self.realtime = realtime
        self.name = name
        self.proc: subprocess.Popen | None = None
        self.thread: threading.Thread | None = None
        self.stderr_tail: list[str] = []
        self.bytes_read = 0
        self._stop = threading.Event()

    def start(self) -> None:
        ffmpeg = find_ffmpeg()
        if not ffmpeg:
            raise RuntimeError("ffmpeg not found (run update.command / update.bat, or install ffmpeg)")
        cmd = ffmpeg_command(ffmpeg, self.media_url, self.headers, self.realtime)
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
                                     creationflags=CREATE_NO_WINDOW, bufsize=0)
        self.thread = threading.Thread(target=self._read_loop, name=f"ffmpeg-{self.name}", daemon=True)
        self.thread.start()
        threading.Thread(target=self._stderr_loop, name=f"ffmpeg-err-{self.name}", daemon=True).start()

    def _read_loop(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        out = self.proc.stdout
        while not self._stop.is_set():
            data = out.read(READ_BYTES)
            if not data:
                break
            self.bytes_read += len(data)
            try:
                self.on_audio(data)
            except Exception:
                log.exception("audio handler failed for %s", self.name)

    def _stderr_loop(self) -> None:
        if self.proc is None or self.proc.stderr is None:
            return
        for raw in iter(self.proc.stderr.readline, b""):
            line = raw.decode("utf-8", "replace").strip()
            if line:
                self.stderr_tail = (self.stderr_tail + [line])[-10:]

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def wait(self, timeout: float | None = None) -> int | None:
        if self.thread is not None:
            self.thread.join(timeout)
        return self.proc.poll() if self.proc else None

    def stop(self) -> None:
        self._stop.set()
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    @property
    def last_error(self) -> str:
        return self.stderr_tail[-1] if self.stderr_tail else ""
