"""Speech-to-text with faster-whisper on the GPU.

One model is loaded once and shared by every stream (a single worker thread processes audio chunks in
order). Voice activity detection (VAD) skips silence and music so only speech is transcribed.
If CUDA can't be used, it falls back to the CPU with a smaller model and says so in the Logs tab.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .. import paths
from .chunker import rms

log = logging.getLogger(__name__)

# Phrases Whisper tends to invent on silence/music; dropped when they make up a whole segment.
HALLUCINATIONS = {
    "thank you.", "thank you", "thanks for watching!", "thanks for watching.", "thank you for watching.",
    "thank you for watching!", "please subscribe.", "subscribe to the channel.", "you", "bye.", "bye!",
    "subtitles by the amara.org community", "♪", "♪♪", "[music]", "(music)", "[applause]", "...",
}

CPU_FALLBACK_MODEL = "small"


@dataclass
class Job:
    source_id: str
    started_at: float  # wall-clock seconds of the chunk start
    audio: np.ndarray
    prompt: str = ""
    queued_at: float = field(default_factory=time.monotonic)


@dataclass
class Segment:
    start: float  # wall-clock seconds
    end: float
    text: str


def clean_segments(raw: list, chunk_start: float) -> list[Segment]:
    out = []
    for s in raw:
        text = (getattr(s, "text", "") or "").strip()
        if not text:
            continue
        if text.lower() in HALLUCINATIONS:
            continue
        if getattr(s, "no_speech_prob", 0) > 0.6 and getattr(s, "avg_logprob", 0) < -1.0:
            continue
        if getattr(s, "compression_ratio", 0) > 2.6:  # repeated-phrase loops
            continue
        out.append(Segment(chunk_start + float(s.start), chunk_start + float(s.end), text))
    return out


class Transcriber:
    name = "transcriber"

    def __init__(self, get_settings: Callable, set_status: Callable[[str, str], None], model_factory=None):
        self.get_settings = get_settings
        self.set_status = set_status
        self._factory = model_factory
        self.q: queue.Queue[Job | None] = queue.Queue()
        self.model = None
        self.model_desc = ""
        self.device = ""
        self._thread: threading.Thread | None = None
        self._loaded_key: tuple | None = None
        self._callbacks: dict[str, Callable[[Job, list[Segment]], None]] = {}
        self.processed = 0
        self.dropped = 0
        self.last_rtf = 0.0  # processing time / audio time

    # ---------------------------------------------------------------- public
    def start(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._loop, name="whisper", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self.q.put(None)

    def register(self, source_id: str, callback: Callable[[Job, list[Segment]], None]) -> None:
        self._callbacks[source_id] = callback

    def unregister(self, source_id: str) -> None:
        self._callbacks.pop(source_id, None)

    def submit(self, job: Job) -> None:
        # If the GPU can't keep up, drop the oldest waiting chunk rather than fall further and further behind.
        limit = max(4, 3 * len(self._callbacks))
        if self.q.qsize() >= limit:
            try:
                old = self.q.get_nowait()
                if old is not None:
                    self.dropped += 1
            except queue.Empty:
                pass
            self.set_status("warn", f"transcription falling behind - dropped {self.dropped} chunks "
                                    "(try fewer streams or a faster model)")
        self.q.put(job)

    @property
    def queue_size(self) -> int:
        return self.q.qsize()

    # ---------------------------------------------------------------- model
    def _load(self) -> None:
        s = self.get_settings()
        key = (s.model, s.device, s.compute_type)
        if self.model is not None and self._loaded_key == key:
            return
        self.model = None
        if self._factory is not None:
            self.model = self._factory(s)
            self.model_desc, self.device = f"{s.model} (test)", "test"
            self._loaded_key = key
            self.set_status("ok", f"ready - {self.model_desc}")
            return

        from .cuda_setup import setup_cuda_dll_paths

        setup_cuda_dll_paths()
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        from faster_whisper import WhisperModel

        attempts = []
        if s.device in ("cuda", "auto"):
            attempts.append((s.model, "cuda", s.compute_type))
            if s.compute_type != "float16":
                attempts.append((s.model, "cuda", "float16"))
        attempts.append((s.model if s.device == "cpu" else CPU_FALLBACK_MODEL, "cpu", "int8"))
        last_exc: Exception | None = None
        for model_name, device, compute in attempts:
            try:
                self.set_status("starting", f"loading {model_name} on {device.upper()} ({compute}) - the first time "
                                            "downloads the model, which can take a few minutes...")
                self.model = WhisperModel(model_name, device=device, compute_type=compute,
                                          download_root=str(paths.models_dir()))
                self.model_desc, self.device = f"{model_name} on {device.upper()} ({compute})", device
                self._loaded_key = key
                if device == "cpu" and s.device != "cpu":
                    self.set_status("warn", f"GPU unavailable ({last_exc}); using CPU with '{model_name}' - slower and "
                                            "less accurate. Check Logs -> Run diagnostics.")
                    log.warning("Whisper fell back to CPU: %s", last_exc)
                else:
                    self.set_status("ok", f"ready - {self.model_desc}")
                log.info("Whisper model loaded: %s", self.model_desc)
                return
            except Exception as exc:
                last_exc = exc
                log.warning("Couldn't load whisper %s on %s/%s: %s", model_name, device, compute, exc)
        self.set_status("error", f"couldn't load the speech model: {last_exc}")
        raise RuntimeError(f"Whisper failed to load: {last_exc}")

    def _transcribe(self, job: Job) -> list[Segment]:
        s = self.get_settings()
        if rms(job.audio) < 0.002:  # near silence - skip the GPU entirely
            return []
        segments, _info = self.model.transcribe(
            job.audio,
            language=None if s.language in ("", "auto") else s.language,
            beam_size=s.beam_size,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": s.vad_min_silence_ms},
            initial_prompt=job.prompt[-200:] or None,
            condition_on_previous_text=False,
        )
        return clean_segments(list(segments), job.started_at)

    # ---------------------------------------------------------------- worker thread
    def _loop(self) -> None:
        while True:
            job = self.q.get()
            if job is None:
                return
            cb = self._callbacks.get(job.source_id)
            if cb is None:
                continue
            try:
                self._load()
                t0 = time.monotonic()
                segs = self._transcribe(job)
                took = time.monotonic() - t0
                dur = max(len(job.audio) / 16000, 0.1)
                self.last_rtf = took / dur
                self.processed += 1
                if self.processed % 20 == 0 or self.last_rtf > 0.8:
                    level = "warn" if self.last_rtf > 0.8 else "ok"
                    self.set_status(level, f"{self.model_desc} · {self.last_rtf:.2f}s per second of audio · "
                                           f"queue {self.q.qsize()}")
                cb(job, segs)
            except Exception as exc:
                log.exception("transcription failed")
                self.set_status("error", f"transcription error: {exc}")
                time.sleep(5)
