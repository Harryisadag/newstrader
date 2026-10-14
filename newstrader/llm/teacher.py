"""Shared training, step 1 - "Label old news": Pro AI reads the old Benzinga headlines the price model downloaded
(ml_samples) and writes down, for each company, what a careful trader would expect - bullish, bearish or neutral, and
how sure it is (table pro_labels). Those labels teach the small, fast student model (ml/student.py) and can be shared
with everyone through GitHub (ml/sharing.py).

It runs in the background, one story at a time, so Pro AI's other slot stays free for live news, and live stories go
first. By default it pauses while the US stock market is open, so trading is never slowed. Start and pause it any
time: what is labelled is kept, so it continues where it stopped, also after a restart. Nothing here can trade - the
answers only go to pro_labels.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import time
from collections import deque
from datetime import UTC, datetime, timedelta

from ..ai.prefilter import Candidate, PrefilterResult
from ..ai.prompts import DIRECTIONS
from ..db import iso, parse_iso
from ..ml.shared_labels import TICKER_RE, headline_hash
from ..sources.base import NewsItem
from ..trading import nyse_calendar

log = logging.getLogger(__name__)

WANTED_KEY = "pro_labels_job"  # {"wanted": bool} - started and not paused (survives a restart)
TIMEOUT = 90.0  # seconds Pro AI may take for one old story (nothing waits on it)
MAX_TICKERS = 4
FAILS_TO_STOP = 5  # this many stories in a row without an answer: stop and say why
IDLE = 30.0  # seconds between checks while waiting (market open, Pro AI not running)
DONE_RECHECK = 600.0  # everything labelled: look for newly downloaded headlines this often
LIVE_FIRST = 1.0  # live stories are waiting for Pro AI: look again this soon
OPEN_MARGIN = timedelta(minutes=10)  # it also pauses this long before the open
SPEED_WINDOW = 20  # stories the "time left" estimate averages over
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I)


def market_busy(now: datetime) -> bool:
    """From 10 minutes before the US open to the close, on trading days."""
    open_, close = nyse_calendar.next_session(now)
    return open_ - OPEN_MARGIN <= now < close


def parse_answer(text: str, tickers: list[str]) -> dict[str, tuple[str, int]] | None:
    """{ticker: (direction, confidence)} from Pro AI's answer; None when it isn't JSON with a signals list. Only the
    story's own stocks count, and the first call per stock wins."""
    try:
        data = json.loads(_FENCE.sub("", (text or "").strip()))
    except (ValueError, TypeError):
        return None
    signals = data.get("signals") if isinstance(data, dict) else None
    if not isinstance(signals, list):
        return None
    calls: dict[str, tuple[str, int]] = {}
    for s in signals:
        if not isinstance(s, dict):
            continue
        t = str(s.get("ticker") or "").strip().upper().lstrip("$")
        d = str(s.get("direction") or "").strip().lower()
        c = s.get("confidence")
        if (t in tickers and t not in calls and d in DIRECTIONS and isinstance(c, int) and not isinstance(c, bool)
                and 0 <= c <= 100):
            calls[t] = (d, c)
    return calls


def event_kinds(item: NewsItem, cands: list[Candidate], table) -> dict[str, str]:
    """{ticker: the kind of news event the rules recognise for it ("" = none)}."""
    from ..ml.engine import readings_for
    from ..ml.text import target_text

    if table is None:
        return {}
    targets = []
    for c in cands:
        t = target_text(item.title, item.body, item.kind, c, table)
        if t is not None:
            targets.append((c, t))
    out = {}
    for (c, _t), r in zip(targets, readings_for(item, targets, table) if targets else [], strict=True):
        ev = next((e for e in r.events if e.role != "actor"), None)
        out[c.symbol] = ev.kind if ev is not None else ""
    return out


class TeacherJob:
    """"Label old news" - one per Pro AI service (service.py)."""

    def __init__(self, ctx, service, now=None, idle: float = IDLE):
        """`service`: the Pro AI service (ready, engine, model_key, settings, live_waiting()). `now` and `idle` are
        for tests."""
        self.ctx = ctx
        self.service = service
        self._now = now or (lambda: datetime.now(UTC))
        self.idle = idle
        self.state = "off"  # off | running | waiting | done | stopped
        self.message = "Not started."
        self.last_error = ""
        self.labelled_now = 0  # stories labelled since the app started
        self._seconds: deque[float] = deque(maxlen=SPEED_WINDOW)
        self._fails = 0
        self._task: asyncio.Task | None = None

    # ------------------------------------------------------------------ control
    @property
    def wanted(self) -> bool:
        info = self.ctx.db.kv_get(WANTED_KEY) or {}
        return isinstance(info, dict) and bool(info.get("wanted"))

    def _set_wanted(self, value: bool) -> None:
        self.ctx.db.kv_set(WANTED_KEY, {"wanted": value, "at": iso()})

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> dict:
        """Start, or continue where it stopped (call in the event loop). Stories Pro AI couldn't answer before get one
        more try."""
        self._set_wanted(True)
        self.ctx.db.execute("DELETE FROM pro_labels WHERE status = 'failed'")
        self._fails = 0
        if not self.running:
            self._set("waiting", "Starting...")
            self._task = asyncio.get_running_loop().create_task(self._loop(), name="pro-ai-labels")
        return self.status()

    def pause(self) -> dict:
        self._set_wanted(False)
        if self._task is not None:
            self._task.cancel()
            self._task = None
        self._set("off", "Paused - click Start to continue where it stopped.")
        return self.status()

    def resume_if_wanted(self) -> None:
        """At start-up: carry on if it was running when the app closed."""
        if self.wanted and not self.running:
            self._task = asyncio.get_running_loop().create_task(self._loop(), name="pro-ai-labels")

    async def stop(self) -> None:
        """The app is closing (it carries on next time if it was running)."""
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    def _set(self, state: str, message: str) -> None:
        changed = (state, message) != (self.state, self.message)
        self.state, self.message = state, message
        if changed:
            with contextlib.suppress(Exception):
                self.ctx.bus.publish("pro_ai_labels", {"state": state, "message": message})

    # ------------------------------------------------------------------ the loop
    async def _loop(self) -> None:
        while True:
            try:
                wait = await self.step()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.exception("Label old news stopped")
                self._set_wanted(False)
                self._set("stopped", f"Stopped by an error: {exc}")
                return
            if wait is None:
                return
            if wait:
                await asyncio.sleep(wait)

    def waiting_for(self) -> tuple[str, float]:
        """(why it should wait right now, for how long); ("", 0) = go ahead."""
        svc = self.service
        if not svc.ready:
            return "Waiting for Pro AI to be running (Settings -> Pro AI).", self.idle
        if not svc.settings.label_while_market_open and market_busy(self._now()):
            return ("Paused while the US stock market is open (from 9:20 am ET), so live trading isn't slowed. It "
                    "carries on after the close.", self.idle)
        if svc.live_waiting():
            return "Live stories go first...", LIVE_FIRST
        if getattr(svc, "testing", False):  # (it would slow down the speed test)
            return "Waiting for Test my PC to finish...", LIVE_FIRST
        return "", 0.0

    async def step(self) -> float | None:
        """Label the next story if now is a good time. Returns how long to wait before the next step (None = stop)."""
        why, wait = self.waiting_for()
        if why:
            self._set("waiting", why)
            return wait
        row = self.next_story()
        if row is None:
            self._set("done", "Every downloaded headline is labelled. Training the price model on more history "
                              "downloads more headlines to label.")
            return DONE_RECHECK
        self._set("running", "Labelling old news...")
        result = await self.label_story(row)
        if result == "failed":
            self._fails += 1
            if self._fails >= FAILS_TO_STOP:
                self._set_wanted(False)
                self._set("stopped", f"Stopped: Pro AI couldn't answer {FAILS_TO_STOP} stories in a row "
                                     f"({self.last_error}). Check Settings -> Pro AI, then click Start again.")
                return None
        elif result == "ok":
            self._fails = 0
        else:  # Pro AI stopped meanwhile: the story is asked again later
            return self.idle
        return 0.0

    def next_story(self) -> dict | None:
        """The newest downloaded story nobody labelled yet."""
        return self.ctx.db.query_one(
            "SELECT news_id, MAX(published_at) AS published_at, headline, summary FROM ml_samples "
            "WHERE text_version = 2 AND news_id NOT IN (SELECT news_id FROM pro_labels) "
            "GROUP BY news_id ORDER BY published_at DESC LIMIT 1")

    async def label_story(self, row: dict) -> str:
        """Ask Pro AI about one old story and store a label for each of its stocks. "ok", "failed" (stored as failed,
        not asked again until Start is clicked) or "skipped" (Pro AI isn't running)."""
        engine = self.service.engine
        if engine is None:
            return "skipped"
        nid = str(row["news_id"])
        syms = [r["symbol"] for r in self.ctx.db.query(
            "SELECT DISTINCT symbol FROM ml_samples WHERE news_id = ? ORDER BY symbol", (nid,))
            if TICKER_RE.match(r["symbol"] or "")][:MAX_TICKERS]
        published = parse_iso(row["published_at"]) or self._now()
        item = NewsItem(source_id="training", source_type="alpaca_news", source_name="Benzinga (history)",
                        external_id=nid, title=row["headline"] or "", body=row["summary"] or "",
                        published_at=published, symbols=syms, language="en")
        table = getattr(self.ctx.service("pipeline"), "tickers", None)
        names = {s: getattr(table.get(s), "name", "") if table is not None else "" for s in syms}
        cands = [Candidate(s, names[s] or s, "tagged by source") for s in syms]
        pre = PrefilterResult(hit=bool(cands), candidates=cands, keywords=[])
        at = published + timedelta(minutes=1)  # read it as fresh news, the way it arrived
        calls = None
        if syms:
            started = time.monotonic()
            res = await engine.analyze(item, pre, nyse_calendar.clock(at)["is_open"], at,
                                       max_signals=len(syms), timeout_s=TIMEOUT)
            seconds = time.monotonic() - started
            calls = parse_answer(res.text, syms) if res.ok else None
            if calls is None:
                if not self.service.ready:
                    return "skipped"
                self.last_error = (res.error or "its answer wasn't valid")[:160]
            else:
                self._seconds.append(seconds)
        try:
            kinds = await asyncio.to_thread(event_kinds, item, cands, table) if calls is not None else {}
        except Exception:
            log.debug("Event rules failed on an old story", exc_info=True)
            kinds = {}
        model = self.service.model_key or str(getattr(engine, "model_key", ""))
        common = (model, iso(), headline_hash(item.title), published.astimezone(nyse_calendar.ET).date().isoformat())
        if calls is None:  # (a story without a usable ticker is stored as failed too, so it isn't picked again)
            rows = [(nid, s, None, None, "", *common, "failed") for s in syms or ["?"]]
        else:
            rows = [(nid, s, *calls.get(s, ("neutral", 0)), kinds.get(s, ""), *common, "ok") for s in syms]
            self.labelled_now += 1
        self.ctx.db.executemany(
            "INSERT OR REPLACE INTO pro_labels (news_id, ticker, direction, confidence, event, model, labelled_at, "
            "headline_hash, day, status) VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
        return "ok" if calls is not None else "failed"

    # ------------------------------------------------------------------ status
    def counts(self) -> dict:
        db = self.ctx.db
        total = db.scalar("SELECT COUNT(DISTINCT news_id) FROM ml_samples WHERE text_version = 2") or 0
        remaining = db.scalar("SELECT COUNT(DISTINCT news_id) FROM ml_samples WHERE text_version = 2 AND news_id "
                              "NOT IN (SELECT news_id FROM pro_labels)") or 0
        mix = {r["direction"]: r["n"] for r in db.query(
            "SELECT direction, COUNT(*) AS n FROM pro_labels WHERE status = 'ok' GROUP BY direction")}
        failed = db.scalar("SELECT COUNT(DISTINCT news_id) FROM pro_labels WHERE status = 'failed'") or 0
        return {"total": int(total), "remaining": int(remaining), "done": int(total - remaining),
                "failed": int(failed), "labels": {d: int(mix.get(d, 0)) for d in DIRECTIONS},
                "label_count": int(sum(mix.values()))}

    def status(self) -> dict:
        """For Settings -> Pro AI -> Label old news (reads the database - call from a worker thread when busy)."""
        c = self.counts()
        per = sum(self._seconds) / len(self._seconds) if self._seconds else None
        return {"state": self.state, "message": self.message, "wanted": self.wanted, "running": self.running,
                **c, "seconds_per_story": round(per, 1) if per is not None else None,
                "eta_seconds": int(per * c["remaining"]) if per is not None else None,
                "labelled_now": self.labelled_now,
                "while_market_open": bool(self.service.settings.label_while_market_open)}
