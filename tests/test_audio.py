from __future__ import annotations

import asyncio
import inspect
import shutil
import subprocess
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from newstrader.audio.capture import FfmpegCapture, ffmpeg_command
from newstrader.audio.chunker import SAMPLE_RATE, AudioChunker
from newstrader.audio.resolver import resolve_stream
from newstrader.audio.stream_manager import Line, RollingTranscript, StreamManager, StreamWorker, build_item
from newstrader.audio.transcriber import Job, Transcriber, clean_segments
from newstrader.config import SourceConfig
from newstrader.tools import find_ffmpeg

STREAM = SourceConfig(id="test-tv", type="stream", name="Test TV", url="https://www.youtube.com/@test/live")


def pcm(seconds: float, freq: float | None = 440.0, amp: float = 0.3) -> bytes:
    n = int(seconds * SAMPLE_RATE)
    if freq is None:
        x = np.zeros(n)
    else:
        x = amp * np.sin(2 * np.pi * freq * np.arange(n) / SAMPLE_RATE)
    return (x * 32767).astype(np.int16).tobytes()


# ---------------------------------------------------------------- chunker
def test_chunker_cuts_at_quiet_point_and_keeps_all_audio():
    ch = AudioChunker(chunk_seconds=10)
    data = pcm(9.0) + pcm(0.5, None) + pcm(6.0)  # silence between 9.0s and 9.5s
    out = []
    # feed in odd-sized pieces to exercise the byte-boundary handling
    for i in range(0, len(data), 7777):
        out += ch.feed(data[i:i + 7777])
    assert len(out) == 1
    offset, chunk = out[0]
    assert offset == 0
    assert 9.0 <= len(chunk) / SAMPLE_RATE <= 9.55  # cut inside the quiet gap, not at exactly 10.0s
    assert chunk.dtype == np.float32 and np.abs(chunk).max() <= 1.0
    rest = ch.flush()
    total = len(chunk) + sum(len(c) for _, c in rest)
    assert total == len(data) // 2
    assert rest[0][0] == pytest.approx(len(chunk) / SAMPLE_RATE)


def test_chunker_drops_tiny_tail():
    ch = AudioChunker(chunk_seconds=10)
    ch.feed(pcm(0.2))
    assert ch.flush() == []


# ---------------------------------------------------------------- ffmpeg capture
@pytest.mark.skipif(find_ffmpeg() is None, reason="ffmpeg not available")
def test_ffmpeg_capture_from_local_file(tmp_path):
    src = tmp_path / "tone.mp3"
    subprocess.run([find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                    "sine=frequency=440:duration=3", "-ar", "44100", "-ac", "2", str(src)], check=True)
    got = bytearray()
    lock = threading.Lock()

    def on_audio(b):
        with lock:
            got.extend(b)

    cap = FfmpegCapture(str(src), on_audio, name="t")
    cap.start()
    cap.wait(timeout=30)
    cap.stop()
    seconds = len(got) / (SAMPLE_RATE * 2)
    assert 2.8 < seconds < 3.3


def test_ffmpeg_command_has_reconnect_and_headers():
    cmd = ffmpeg_command("ffmpeg", "https://x.com/a.m3u8", {"User-Agent": "UA", "Referer": "https://y"})
    assert "-reconnect" in cmd and cmd[cmd.index("-user_agent") + 1] == "UA"
    assert "Referer: https://y" in cmd[cmd.index("-headers") + 1]
    assert cmd[-6:] == ["-f", "s16le", "-acodec", "pcm_s16le", "pipe:1"][-6:] or cmd[-1] == "pipe:1"
    assert ["-ac", "1", "-ar", "16000"] == cmd[cmd.index("-ac"):cmd.index("-ac") + 4]


def test_direct_media_urls_skip_ytdlp():
    r = resolve_stream("https://radio.example.com/live/stream.mp3")
    assert r.media_url.endswith("stream.mp3") and r.is_live


# ---------------------------------------------------------------- transcription
def seg(start, end, text, **kw):
    return SimpleNamespace(start=start, end=end, text=text, no_speech_prob=kw.get("nsp", 0.1),
                           avg_logprob=kw.get("lp", -0.3), compression_ratio=kw.get("cr", 1.4))


def test_hallucinations_and_junk_removed():
    raw = [seg(0, 2, " Nvidia shares are up 5 percent."), seg(2, 3, " Thank you."), seg(3, 4, " ♪"),
           seg(4, 5, " hmm", nsp=0.9, lp=-1.5), seg(5, 6, " the the the the", cr=3.1), seg(6, 7, "  ")]
    out = clean_segments(raw, chunk_start=1000.0)
    assert [s.text for s in out] == ["Nvidia shares are up 5 percent."]
    assert out[0].start == 1000.0 and out[0].end == 1002.0


def test_faster_whisper_api_matches_our_calls():
    fw = pytest.importorskip("faster_whisper")
    init = inspect.signature(fw.WhisperModel.__init__).parameters
    for p in ("device", "compute_type", "download_root"):
        assert p in init
    tr = inspect.signature(fw.WhisperModel.transcribe).parameters
    for p in ("language", "beam_size", "vad_filter", "vad_parameters", "initial_prompt", "condition_on_previous_text"):
        assert p in tr


def test_transcriber_worker_with_stub_model():
    calls = []

    class StubModel:
        def transcribe(self, audio, **kw):
            calls.append(kw)
            return iter([seg(0.5, 2.0, " Apple is buying a startup.")]), None

    settings = SimpleNamespace(model="tiny", device="cpu", compute_type="int8", language="en", beam_size=1,
                               vad_min_silence_ms=500)
    statuses = []
    t = Transcriber(lambda: settings, lambda lvl, d: statuses.append((lvl, d)), model_factory=lambda s: StubModel())
    done = threading.Event()
    results = []
    t.register("tv", lambda job, segs: (results.append(segs), done.set()))
    t.start()
    audio = np.frombuffer(pcm(3.0), dtype=np.int16).astype(np.float32) / 32768
    t.submit(Job("tv", 500.0, audio, prompt="earlier words"))
    assert done.wait(5)
    t.stop()
    assert results[0][0].text == "Apple is buying a startup." and results[0][0].start == 500.5
    assert calls[0]["vad_filter"] is True and calls[0]["language"] == "en"
    assert calls[0]["initial_prompt"] == "earlier words"


def test_silent_chunks_skip_the_model():
    class Boom:
        def transcribe(self, *a, **k):
            raise AssertionError("should not be called for silence")

    settings = SimpleNamespace(model="tiny", device="cpu", compute_type="int8", language="en", beam_size=1,
                               vad_min_silence_ms=500)
    t = Transcriber(lambda: settings, lambda *a: None, model_factory=lambda s: Boom())
    t._load()
    assert t._transcribe(Job("tv", 0.0, np.zeros(16000, dtype=np.float32))) == []


class RecordingWhisper:
    loaded: list = []

    def __init__(self, name, device, compute_type, download_root):
        if device == "cuda":
            RecordingWhisper.loaded.append((name, device))
            raise RuntimeError("CUDA failed")
        RecordingWhisper.loaded.append((name, device))


@pytest.fixture
def pc_whisper(monkeypatch):
    import faster_whisper

    from newstrader.audio import transcriber as tr

    monkeypatch.setattr(tr.sys, "platform", "win32")
    monkeypatch.setattr(faster_whisper, "WhisperModel", RecordingWhisper)
    RecordingWhisper.loaded = []
    return tr


def _gpu_settings():
    return SimpleNamespace(model="large-v3", device="cuda", compute_type="float16", language="en", beam_size=1,
                           vad_min_silence_ms=500)


def test_no_nvidia_gpu_skips_the_big_cuda_model(pc_whisper, monkeypatch):
    monkeypatch.setattr(pc_whisper, "_cuda_devices", lambda: 0)
    statuses = []
    t = Transcriber(_gpu_settings, lambda lvl, d: statuses.append((lvl, d)))
    t._load()
    # straight to the small CPU model - large-v3 (3 GB) is never downloaded for a GPU that isn't there
    assert RecordingWhisper.loaded == [("small", "cpu")]
    assert statuses[-1][0] == "warn" and "no NVIDIA GPU" in statuses[-1][1]


@pytest.mark.parametrize("count", [1, None])  # a GPU, or CTranslate2 couldn't tell -> CUDA is still tried
def test_cuda_is_tried_when_a_gpu_may_exist(pc_whisper, monkeypatch, count):
    monkeypatch.setattr(pc_whisper, "_cuda_devices", lambda: count)
    t = Transcriber(_gpu_settings, lambda *a: None)
    t._load()
    assert RecordingWhisper.loaded == [("large-v3", "cuda"), ("small", "cpu")]


def test_diagnostics_gpu_check_without_nvidia(monkeypatch):
    from newstrader import diagnostics

    monkeypatch.setattr(diagnostics.sys, "platform", "win32")
    ctx = SimpleNamespace(config=SimpleNamespace(settings=SimpleNamespace(
        transcription=SimpleNamespace(device="cuda"))))
    monkeypatch.setattr(diagnostics, "gpu_info", lambda: {"cuda_devices": 0, "name": None, "error": None})
    assert diagnostics._check_gpu(ctx)["status"] == "warn"  # no NVIDIA card is normal, not a fault
    monkeypatch.setattr(diagnostics, "gpu_info", lambda: {"cuda_devices": 0, "name": "NVIDIA RTX 3060 Ti",
                                                          "error": None})
    assert diagnostics._check_gpu(ctx)["status"] == "error"  # card present but unusable -> driver problem


def test_diagnostics_update_hint_for_the_downloaded_app(monkeypatch):
    from newstrader import diagnostics

    monkeypatch.setattr(diagnostics.paths, "is_frozen", lambda: True)
    assert "newest NewsTrader release" in diagnostics._update_hint()
    monkeypatch.setattr(diagnostics.paths, "is_frozen", lambda: False)
    assert diagnostics._update_hint() == f"Run {diagnostics.UPDATE}"


# ---------------------------------------------------------------- rolling transcript -> AI
def test_rolling_transcript_marks_lines_once():
    r = RollingTranscript()
    for i, t in enumerate(["a", "b", "c"], start=1):
        r.add(Line(i, 100 + i * 5, 104 + i * 5, t))
    ctx, new = r.take_for_analysis(60)
    assert ctx == [] and [ln.text for ln in new] == ["a", "b", "c"]
    r.add(Line(4, 125, 129, "d"))
    ctx, new = r.take_for_analysis(60)
    assert [ln.text for ln in ctx] == ["a", "b", "c"] and [ln.text for ln in new] == ["d"]
    assert r.take_for_analysis(60) == ([], [])


def test_build_item_separates_context_and_new():
    item = build_item(STREAM, [Line(1, 100, 104, "old talk")], [Line(2, 105, 109, "Nvidia wins contract")])
    assert item.kind == "transcript" and item.source_type == "stream"
    assert item.external_id == "test-tv:2-2"
    assert "Earlier (context only" in item.body and item.body.index("old talk") < item.body.index("New:")
    assert item.title == "Nvidia wins contract"


async def test_transcript_hit_triggers_pipeline(ctx):
    from .helpers import make_tickers

    submitted = []

    class FakePipeline:
        tickers = make_tickers(ctx.db)

        async def submit(self, item):
            submitted.append(item)
            return {"status": "queued"}

    ctx.services["pipeline"] = FakePipeline()
    ctx.config.update({"transcription": {"analysis_debounce_seconds": 0}})
    mgr = StreamManager(ctx, transcriber=Transcriber(lambda: None, lambda *a: None))
    worker = StreamWorker(mgr, STREAM)
    now = time.time()
    await worker.handle_segments(Job("test-tv", now, np.zeros(1)), [
        transcriber_seg(now, "Welcome back to the show."),
    ])
    await asyncio.sleep(0.05)
    assert submitted == []  # nothing market-relevant
    await worker.handle_segments(Job("test-tv", now, np.zeros(1)), [
        transcriber_seg(now + 5, "Breaking: Nvidia just announced a huge acquisition."),
    ])
    await asyncio.sleep(0.05)
    assert len(submitted) == 1
    assert "Nvidia" in submitted[0].body and "Welcome back" in submitted[0].body
    rows = ctx.db.query("SELECT text FROM transcripts ORDER BY id")
    assert len(rows) == 2


def transcriber_seg(start, text):
    from newstrader.audio.transcriber import Segment

    return Segment(start, start + 3, text)


async def test_max_concurrent_streams(ctx, monkeypatch):
    from newstrader.audio import stream_manager as sm

    def offline(url, cookies=""):
        raise sm.StreamOffline("not live")

    monkeypatch.setattr(sm, "resolve_stream", offline)
    ctx.config.update({"transcription": {"max_concurrent_streams": 2}})
    mgr = StreamManager(ctx, transcriber=Transcriber(lambda: None, lambda *a: None))
    await mgr.reconcile()
    try:
        enabled = [s.id for s in ctx.config.settings.sources if s.type == "stream" and s.enabled]
        assert len(mgr.tasks) == 2 and list(mgr.tasks) == enabled[:2]
        comps = {c["component"]: c for c in ctx.state.components()}
        waiting = comps[f"source:{enabled[2]}"]
        assert "max 2" in waiting["detail"]
    finally:
        for sid in list(mgr.tasks):
            await mgr._stop(sid)


@pytest.mark.skipif(shutil.which("ffmpeg") is None and find_ffmpeg() is None, reason="ffmpeg not available")
def test_ffmpeg_found():
    assert find_ffmpeg()
