"""Speech-to-text, on whatever fast hardware this computer has.

    Windows / Linux + NVIDIA GPU  -> faster-whisper on CUDA
    Mac with Apple Silicon        -> mlx-whisper on the Apple GPU (if installed - run.command does that)
    anything else                 -> faster-whisper on the CPU (slower; a smaller model is used)

One model is loaded once and shared by every stream (a single worker thread processes audio chunks in
order - MLX also requires all its work to stay on one thread). Voice activity detection (VAD) skips silence
and music so only speech is transcribed. If the preferred device can't be used, it falls back to the CPU and
says so in the Logs tab.
"""

from __future__ import annotations

import logging
import os
import platform
import queue
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from types import SimpleNamespace

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

# Whisper model name -> the MLX conversion of it on Hugging Face (Apple Silicon only)
MLX_REPOS = {
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "distil-large-v3": "mlx-community/distil-whisper-large-v3",
    "medium": "mlx-community/whisper-medium-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "tiny": "mlx-community/whisper-tiny",
}


def is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def release_mlx() -> None:
    """mlx-whisper keeps its last model in a class-level cache; drop it (and MLX's buffer cache) so switching
    to another device doesn't keep ~2-3 GB of unified memory busy. Call on the transcriber thread."""
    try:
        import importlib

        holder = getattr(importlib.import_module("mlx_whisper.transcribe"), "ModelHolder", None)
        if holder is not None:
            holder.model = None
            holder.model_path = None
        import mlx.core as mx

        clear = getattr(mx, "clear_cache", None) or getattr(getattr(mx, "metal", None), "clear_cache", None)
        if clear:
            clear()
    except Exception:
        pass


def mlx_available() -> bool:
    if not is_apple_silicon():
        return False
    try:
        import mlx_whisper  # noqa: F401
    except Exception:
        return False
    return True


class MlxWhisper:
    """mlx-whisper with the same Silero VAD faster-whisper uses. Call only from the transcriber's thread."""

    def __init__(self, model_name: str):
        try:
            import mlx_whisper  # noqa: F401
        except Exception as exc:
            raise RuntimeError("the Apple-GPU speech engine (mlx-whisper) isn't installed - run run.command "
                               f"again to install it ({type(exc).__name__})") from exc

        self.repo = MLX_REPOS.get(model_name, model_name if "/" in model_name else MLX_REPOS["large-v3-turbo"])
        self._warm = False

    def warm_up(self) -> None:
        """Downloads (first time) and loads the model by transcribing a short silent clip."""
        import mlx_whisper

        mlx_whisper.transcribe(np.zeros(16000, dtype=np.float32), path_or_hf_repo=self.repo, verbose=None,
                               language="en")
        self._warm = True

    def transcribe(self, audio: np.ndarray, language: str | None, initial_prompt: str | None,
                   min_silence_ms: int) -> list:
        import mlx_whisper
        from faster_whisper.vad import SpeechTimestampsMap, VadOptions, collect_chunks, get_speech_timestamps

        audio = np.ascontiguousarray(audio, dtype=np.float32)
        speech = get_speech_timestamps(audio, VadOptions(min_silence_duration_ms=min_silence_ms))
        if not speech:  # an empty clip list would make mlx-whisper transcribe everything
            return []
        chunks, _ = collect_chunks(audio, speech)
        result = mlx_whisper.transcribe(np.concatenate(chunks), path_or_hf_repo=self.repo, language=language,
                                        initial_prompt=initial_prompt, condition_on_previous_text=False,
                                        verbose=None)  # beam search isn't supported by mlx-whisper
        ts = SpeechTimestampsMap(speech, 16000)
        return [SimpleNamespace(start=ts.get_original_time(seg["start"]),
                                end=ts.get_original_time(seg["end"], is_end=True), text=seg.get("text", ""),
                                no_speech_prob=seg.get("no_speech_prob", 0.0),
                                avg_logprob=seg.get("avg_logprob", 0.0),
                                compression_ratio=seg.get("compression_ratio", 0.0))
                for seg in result.get("segments", [])]


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


def _cuda_devices() -> int | None:
    """How many CUDA GPUs CTranslate2 can see; None if it can't tell (then CUDA is still tried)."""
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count()
    except Exception:
        return None


class Transcriber:
    name = "transcriber"

    def __init__(self, get_settings: Callable, set_status: Callable[[str, str], None], model_factory=None):
        self.get_settings = get_settings
        self.set_status = set_status
        self._factory = model_factory
        self.q: queue.Queue[Job | None] = queue.Queue()
        self.model = None
        self.model_desc = ""
        self.device = ""  # cuda | cpu | mlx | test
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
        if self.device == "mlx":
            release_mlx()
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

        attempts = []
        mac = sys.platform == "darwin"
        if s.device == "mlx" or (s.device in ("auto", "cuda") and mac):
            if is_apple_silicon():
                attempts.append((s.model, "mlx", "float16"))
        elif s.device in ("cuda", "auto") and _cuda_devices() != 0:
            # (a failed CUDA attempt still downloads the big model first, so skip it when there's no NVIDIA GPU)
            attempts.append((s.model, "cuda", s.compute_type))
            if s.compute_type != "float16":
                attempts.append((s.model, "cuda", "float16"))
        attempts.append((s.model if s.device == "cpu" else CPU_FALLBACK_MODEL, "cpu", "int8"))
        last_exc: Exception | None = None
        if attempts[0][1] == "cpu" and s.device != "cpu":
            last_exc = RuntimeError("an Intel Mac has no GPU this app can use" if mac else "no NVIDIA GPU found")
        for model_name, device, compute in attempts:
            try:
                where = "Apple GPU (MLX)" if device == "mlx" else device.upper()
                self.set_status("starting", f"loading {model_name} on {where} ({compute}) - the first time "
                                            "downloads the model, which can take a few minutes...")
                if device == "mlx":
                    model = MlxWhisper(model_name)
                    try:
                        model.warm_up()
                    except Exception:
                        release_mlx()
                        raise
                    self.model = model
                else:
                    from faster_whisper import WhisperModel

                    self.model = WhisperModel(model_name, device=device, compute_type=compute,
                                              download_root=str(paths.models_dir()))
                self.model_desc, self.device = f"{model_name} on {where} ({compute})", device
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
        if self.device == "mlx":
            raw = self.model.transcribe(job.audio, None if s.language in ("", "auto") else s.language,
                                        job.prompt[-200:] or None, s.vad_min_silence_ms)
            return clean_segments(raw, job.started_at)
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
