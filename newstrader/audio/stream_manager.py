"""Runs the live TV streams: yt-dlp -> ffmpeg -> chunks -> Whisper -> rolling transcript -> AI pipeline.

- Every turned-on stream is watched, but only `max_concurrent_streams` can be transcribed at once: a channel that
  isn't live (most "live events" channels, e.g. the White House) doesn't use a slot. When a live-events channel goes
  live and every slot is busy, it borrows the slot of the lowest always-on stream until the event ends.
- Each stream reconnects by itself (re-resolving the URL, since YouTube stream links expire).
- Non-English streams can be translated to English while they are transcribed (Settings -> News sources).
- When new transcript lines mention a company, ticker or market keyword, the last ~60 seconds of
  transcript are sent to the AI pipeline after a short pause (4 s by default) so the rest of the sentence - and any
  clip that finishes at the same moment - is included. Every line is analysed once as "new"; lines that were
  already analysed are only included as context.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from ..ai.prefilter import prefilter
from ..config import SourceConfig
from ..context import AppContext
from ..db import iso, utcnow
from ..sources.base import NewsItem
from .capture import FfmpegCapture
from .chunker import AudioChunker
from .resolver import StreamOffline, resolve_stream
from .transcriber import Job, Segment, Transcriber

log = logging.getLogger(__name__)

OFFLINE_RECHECK = 300  # seconds between checks when a channel isn't live
SLOT_RECHECK = 180  # a live stream waiting for a free slot checks again this often (or as soon as one frees up)
MAX_WATCHED = 30  # turned-on streams watched at once (each offline check is one small yt-dlp request)
ET = ZoneInfo("America/New_York")


@dataclass
class Line:
    id: int
    start: float
    end: float
    text: str


class RollingTranscript:
    def __init__(self, keep_seconds: int = 900):
        self.lines: deque[Line] = deque()
        self.keep_seconds = keep_seconds
        self.last_analyzed_id = 0

    def add(self, line: Line) -> None:
        self.lines.append(line)
        cutoff = line.end - self.keep_seconds
        while self.lines and self.lines[0].end < cutoff:
            self.lines.popleft()

    def window(self, seconds: float, now: float | None = None) -> list[Line]:
        if not self.lines:
            return []
        end = now if now is not None else self.lines[-1].end
        return [ln for ln in self.lines if ln.end >= end - seconds]

    def take_for_analysis(self, seconds: float) -> tuple[list[Line], list[Line]]:
        """(context_lines, new_lines) for the next analysis; marks the new lines as analysed."""
        lines = self.window(seconds)
        new = [ln for ln in lines if ln.id > self.last_analyzed_id]
        if not new:
            return [], []
        context = [ln for ln in lines if ln.id <= self.last_analyzed_id]
        self.last_analyzed_id = new[-1].id
        return context, new


def _clock(ts: float) -> str:
    # New York time, like every other time the AI is shown (the US market's clock)
    return datetime.fromtimestamp(ts, UTC).astimezone(ET).strftime("%H:%M:%S")


def build_item(src: SourceConfig, context: list[Line], new: list[Line]) -> NewsItem:
    new_text = " ".join(ln.text for ln in new)
    parts = []
    if context:
        parts.append("Earlier (context only, already analysed):")
        parts += [f"[{_clock(ln.start)}] {ln.text}" for ln in context]
        parts.append("")
    parts.append("New:")
    parts += [f"[{_clock(ln.start)}] {ln.text}" for ln in new]
    return NewsItem(source_id=src.id, source_type="stream", source_name=src.name,
                    external_id=f"{src.id}:{new[0].id}-{new[-1].id}", title=new_text[:200],
                    body="\n".join(parts), url=src.url, published_at=datetime.fromtimestamp(new[0].start, UTC),
                    spoken_at=datetime.fromtimestamp(new[-1].end, UTC),
                    kind="transcript", language="en" if src.translate else src.language.replace("auto", ""))


class StreamWorker:
    def __init__(self, manager: StreamManager, src: SourceConfig):
        self.m = manager
        self.src = src
        self.capture: FfmpegCapture | None = None
        self.prompt = ""
        self.title = ""
        self.rolling = RollingTranscript()
        self._analysis_timer: asyncio.Task | None = None
        self.paused_reason = ""  # set when this stream gives its slot to a live event (or the limit was lowered)

    def pause(self, reason: str) -> None:
        """Give up the transcription slot; the capture loop notices within ~2 seconds."""
        self.paused_reason = reason

    def set_status(self, level: str, detail: str) -> None:
        self.m.ctx.state.set_status(f"source:{self.src.id}", level, detail, name=self.src.name, type="stream")

    async def run(self) -> None:
        backoff = 5
        settings = self.m.ctx.config.settings.transcription
        while True:
            self.set_status("starting", "finding the stream...")
            try:
                resolved = await asyncio.to_thread(resolve_stream, self.src.url, settings.cookies_from_browser)
            except StreamOffline as exc:
                self.set_status("off", f"offline - {exc}; checking again in {OFFLINE_RECHECK // 60} min")
                await asyncio.sleep(OFFLINE_RECHECK)
                continue
            except Exception as exc:
                self.set_status("error", f"{str(exc)[:200]} (retrying in {backoff}s)")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 600)
                continue

            self.title = resolved.title
            if not self.m.acquire_slot(self):
                n = self.m.ctx.config.settings.transcription.max_concurrent_streams
                self.set_status("off", f"live, waiting for a free slot - {n} stream{'s' if n != 1 else ''} already "
                                       "being transcribed (Settings -> Transcription -> Max streams at once)")
                await self.m.wait_for_slot(SLOT_RECHECK)
                continue
            self.paused_reason = ""
            chunker = AudioChunker(self.m.ctx.config.settings.transcription.chunk_seconds)
            started_wall = time.time()
            transcriber = self.m.transcriber

            def on_audio(data: bytes, chunker=chunker, transcriber=transcriber, started_wall=started_wall) -> None:
                # runs on the ffmpeg reader thread
                for offset, audio in chunker.feed(data):
                    transcriber.submit(self._job(started_wall + offset, audio))

            self.capture = FfmpegCapture(resolved.media_url, on_audio, resolved.headers, name=self.src.id)
            try:
                await asyncio.to_thread(self.capture.start)
            except Exception as exc:
                self.m.release_slot(self)
                self.set_status("error", str(exc))
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 600)
                continue
            self.set_status("ok", f"live - {resolved.title[:80]}" + (" (translated to English)" if self.src.translate
                                                                      else ""))
            last_bytes, stalled_since = 0, time.monotonic()
            try:
                while self.capture.running and not self.paused_reason:
                    await asyncio.sleep(2)
                    if self.capture.bytes_read != last_bytes:
                        last_bytes, stalled_since = self.capture.bytes_read, time.monotonic()
                        backoff = 5
                    elif time.monotonic() - stalled_since > 60:
                        log.warning("%s: no audio for 60s, reconnecting", self.src.name)
                        break
            finally:
                await asyncio.to_thread(self.capture.stop)
                self.m.release_slot(self)
            for offset, audio in chunker.flush():
                transcriber.submit(self._job(started_wall + offset, audio))
            if self.paused_reason:
                self.set_status("off", f"paused - {self.paused_reason}; continues when a slot frees up")
                await self.m.wait_for_slot(SLOT_RECHECK)
                continue
            if not resolved.is_live:
                self.set_status("off", "finished (this was a recorded video, not a live stream)")
                return
            self.set_status("warn", f"stream dropped ({self.capture.last_error or 'ended'}); reconnecting in {backoff}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 300)

    def _job(self, started_at: float, audio) -> Job:
        return Job(self.src.id, started_at, audio, self.prompt, language=self.src.language,
                   task="translate" if self.src.translate else "transcribe")

    def stop(self) -> None:
        if self.capture is not None:
            self.capture.stop()
        if self._analysis_timer is not None:
            self._analysis_timer.cancel()

    # ---- called in the event loop when Whisper returns text ----
    async def handle_segments(self, job: Job, segs: list[Segment]) -> None:
        if not segs:
            return
        ctx = self.m.ctx
        texts = []
        for seg in segs:
            row = {"source_id": self.src.id, "source_name": self.src.name,
                   "start_ts": iso(datetime.fromtimestamp(seg.start, UTC)),
                   "end_ts": iso(datetime.fromtimestamp(seg.end, UTC)), "text": seg.text, "created_at": iso(),
                   "language": "en" if self.src.translate else (seg.language or self.src.language or None)}
            row["id"] = ctx.db.insert("transcripts", row)
            self.rolling.add(Line(row["id"], seg.start, seg.end, seg.text))
            texts.append(seg.text)
            pre = prefilter(seg.text, self.m.tickers(), [], ctx.config.settings.ai.analyze_keyword_only)
            ctx.bus.publish("transcript", {**row, "candidates": [c.symbol for c in pre.candidates],
                                           "keywords": pre.keywords, "hit": pre.hit})
            if pre.hit:
                self._schedule_analysis()
        self.prompt = (self.prompt + " " + " ".join(texts))[-300:]

    def _schedule_analysis(self) -> None:
        if self._analysis_timer is not None and not self._analysis_timer.done():
            return
        delay = self.m.ctx.config.settings.transcription.analysis_debounce_seconds
        self._analysis_timer = asyncio.create_task(self._analyse_after(delay))

    async def _analyse_after(self, delay: float) -> None:
        await asyncio.sleep(delay)
        seconds = self.m.ctx.config.settings.transcription.analysis_window_seconds
        context, new = self.rolling.take_for_analysis(seconds)
        # these lines are taken: a hit in the next lines starts a new wait, even while this clip is still being sent
        self._analysis_timer = None
        if not new:
            return
        pipeline = self.m.ctx.service("pipeline")
        if pipeline is not None:
            await pipeline.submit(build_item(self.src, context, new))


class StreamManager:
    name = "streams"

    def __init__(self, ctx: AppContext, transcriber: Transcriber | None = None):
        self.ctx = ctx
        self.transcriber = transcriber or Transcriber(lambda: ctx.config.settings.transcription,
                                                      lambda lvl, d: ctx.state.set_status("transcriber", lvl, d))
        self.workers: dict[str, StreamWorker] = {}
        self.tasks: dict[str, asyncio.Task] = {}
        self.configs: dict[str, dict] = {}
        self.live: dict[str, StreamWorker] = {}  # streams being transcribed right now (they hold a slot)
        self._waiting: set[str] = set()  # live streams waiting for a slot
        self._slot_freed = asyncio.Event()
        self._changed = asyncio.Event()
        self._watch: asyncio.Task | None = None

    def tickers(self):
        pipeline = self.ctx.service("pipeline")
        if pipeline is not None:
            return pipeline.tickers
        from ..ai.tickers import TickerTable

        return TickerTable(self.ctx.db)

    async def start(self) -> None:
        with contextlib.suppress(Exception):
            self.ctx.db.execute("DELETE FROM transcripts WHERE created_at < ?", (iso(utcnow() - timedelta(days=7)),))
        self.transcriber.start()
        self.ctx.config.on_change(lambda _s: self._signal_change())
        await self.reconcile()
        self._watch = asyncio.create_task(self._watch_loop(), name="streams-watch")

    async def stop(self) -> None:
        if self._watch:
            self._watch.cancel()
        for sid in list(self.tasks):
            await self._stop(sid)
        self.transcriber.stop()

    def _signal_change(self) -> None:
        loop = self.ctx.loop
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(self._changed.set)

    async def _watch_loop(self) -> None:
        while True:
            await self._changed.wait()
            self._changed.clear()
            await asyncio.sleep(0.3)
            try:
                await self.reconcile()
            except Exception:
                log.exception("stream reconcile failed")

    async def reconcile(self) -> None:
        t = self.ctx.config.settings.transcription
        streams = [s for s in self.ctx.config.settings.sources if s.type == "stream"]
        enabled = [s for s in streams if s.enabled] if t.enabled else []
        wanted = {s.id: s for s in enabled[:MAX_WATCHED]}
        for sid in list(self.tasks):
            if sid not in wanted or self.configs.get(sid) != wanted[sid].model_dump():
                await self._stop(sid)
        for sid, src in wanted.items():
            if sid not in self.tasks:
                self._start(src)
        for s in streams:
            if s.id in wanted:
                continue
            if not t.enabled:
                self.ctx.state.set_status(f"source:{s.id}", "off", "transcription is turned off in Settings",
                                          name=s.name, type="stream")
            elif not s.enabled:
                self.ctx.state.set_status(f"source:{s.id}", "off", "turned off", name=s.name, type="stream")
            else:
                self.ctx.state.set_status(f"source:{s.id}", "off",
                                          f"not watched - at most {MAX_WATCHED} streams can be turned on at once",
                                          name=s.name, type="stream")
        self._enforce_limit()
        if not wanted:
            self.ctx.state.set_status("transcriber", "off", "no live streams running")
        elif self.transcriber.model is None and not any(
                c["component"] == "transcriber" and c["level"] in ("starting", "error", "warn")
                for c in self.ctx.state.components()):
            self.ctx.state.set_status("transcriber", "starting",
                                      f"{t.model} loads when the first audio arrives ({t.device.upper()})")

    def _start(self, src: SourceConfig) -> None:
        worker = StreamWorker(self, src)
        loop = asyncio.get_running_loop()

        def on_segments(job: Job, segs: list[Segment]) -> None:  # transcriber thread
            loop.call_soon_threadsafe(lambda: asyncio.ensure_future(worker.handle_segments(job, segs)))

        self.transcriber.register(src.id, on_segments, translate=src.translate)
        self.workers[src.id] = worker
        self.tasks[src.id] = asyncio.create_task(self._guard(worker), name=f"stream-{src.id}")
        self.configs[src.id] = src.model_dump()

    async def _guard(self, worker: StreamWorker) -> None:
        try:
            await worker.run()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.exception("stream %s crashed", worker.src.name)
            worker.set_status("error", f"crashed: {exc}")

    # ---- transcription slots ----
    def _rank(self) -> dict[str, int]:
        """Lower = more important: the order of Settings -> News sources (then the order streams started)."""
        rank = {s.id: i for i, s in enumerate(self.ctx.config.settings.sources)}
        for i, sid in enumerate(self.workers):
            rank.setdefault(sid, 100_000 + i)
        return rank

    def acquire_slot(self, worker: StreamWorker) -> bool:
        """Give a live stream a transcription slot. A live-events channel may take one from an always-on stream."""
        sid = worker.src.id
        if self.live.get(sid) is worker:
            return True
        limit = self.ctx.config.settings.transcription.max_concurrent_streams
        free = limit - len(self.live)
        events_waiting = sum(1 for w in self._waiting if w != sid and w in self.workers
                             and self.workers[w].src.live_events)
        if free > 0 and (worker.src.live_events or free > events_waiting):
            self.live[sid] = worker
            self._waiting.discard(sid)
            return True
        if worker.src.live_events:
            order = self._rank()
            always_on = [w for w in self.live.values() if not w.src.live_events]
            if always_on:
                victim = max(always_on, key=lambda w: order.get(w.src.id, 0))
                victim.pause(f"gave its slot to {worker.src.name} (live event)")
                self.live.pop(victim.src.id, None)
                self.live[sid] = worker
                self._waiting.discard(sid)
                return True
        self._waiting.add(sid)
        return False

    def release_slot(self, worker: StreamWorker) -> None:
        if self.live.get(worker.src.id) is worker:
            del self.live[worker.src.id]
        self._slot_freed.set()
        self._slot_freed = asyncio.Event()

    async def wait_for_slot(self, timeout: float) -> None:
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._slot_freed.wait(), timeout)

    def _enforce_limit(self) -> None:
        """If Max streams at once was lowered, pause the lowest-priority streams (always-on ones first)."""
        limit = self.ctx.config.settings.transcription.max_concurrent_streams
        order = self._rank()
        while len(self.live) > limit:
            victim = max(self.live.values(), key=lambda w: (not w.src.live_events, order.get(w.src.id, 0)))
            victim.pause(f"max {limit} stream{'s' if limit != 1 else ''} at once")
            self.live.pop(victim.src.id, None)

    async def _stop(self, sid: str) -> None:
        worker = self.workers.pop(sid, None)
        task = self.tasks.pop(sid, None)
        self.configs.pop(sid, None)
        self._waiting.discard(sid)
        if worker is not None:
            self.release_slot(worker)
        self.transcriber.unregister(sid)
        if worker is not None:
            await asyncio.to_thread(worker.stop)
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    def summary(self) -> dict:
        tr = self.transcriber
        return {"running": list(self.live), "watching": list(self.tasks), "model": tr.model_desc, "device": tr.device, "queue": tr.queue_size,
                "processed": tr.processed, "dropped": tr.dropped, "rtf": round(tr.last_rtf, 3)}
