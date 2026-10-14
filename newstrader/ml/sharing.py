"""Shared training in the app: everyone's labels from GitHub, "Share my labels" and "Train with everyone's labels".

- Everyone's labels: once a day (Settings -> AI engine) and on a button, the list of files in the repo's shared/labels
  folder is read from GitHub's contents API (no account or key needed; when nothing changed GitHub answers "not
  modified", which doesn't count against its limit). Each new file is downloaded from raw.githubusercontent.com, must
  match GitHub's fingerprint for it (its git blob SHA-1) and pass the same checks GitHub runs on every upload
  (shared_labels.check_file), and is kept in the data folder (shared_labels/). Files removed from the repo go here too.
- Share my labels: your Pro AI labels (no text, no keys, no trades) go into one file in your Downloads folder, the
  folder is shown, and GitHub's upload page opens with three plain steps. The app never uploads anything itself.
- Train with everyone's labels: your labels and everyone's are matched with the headlines you downloaded (by news id,
  with the headline fingerprint as a check), the student model (student.py) learns from them and is tested on the
  newest 20%. It's only used if it's at least as good as the student in use on those same labels.
Nothing here can place a trade.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import secrets
import webbrowser
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

from .. import __version__, paths
from ..ai.prefilter import Candidate
from ..db import Database, iso, parse_iso
from ..sources.base import NewsItem
from .dataset import Cancelled, cached_sentiment
from .shared_labels import (
    MAX_BYTES,
    MAX_ROWS,
    NAME_RE,
    blob_sha,
    check_file,
    check_row,
    dump,
    file_name,
    headline_hash,
    read_rows,
    to_row,
    trim_mix,
    utc_today,
)
from .student import StudentSample, delete_student, score_samples, student_features, train_student

log = logging.getLogger(__name__)

REPO = "Harryisadag/newstrader"
BRANCH = "claude/zen-gauss-5ktkek"
FOLDER = "shared/labels"
LIST_URL = f"https://api.github.com/repos/{REPO}/contents/{FOLDER}"
RAW_URL = f"https://raw.githubusercontent.com/{REPO}/refs/heads/{BRANCH}/{FOLDER}/{{name}}"
UPLOAD_URL = f"https://github.com/{REPO}/upload/{BRANCH}/{FOLDER}"
FIRST_PULL_DELAY = 90  # seconds after start-up
LOOP_EVERY = 3600  # how often it looks whether a check is due
PULL_EVERY = 24 * 3600
RETRY_AFTER_ERROR = 3 * 3600
MAX_CACHE_BYTES = 500 * 1024 * 1024  # all downloaded label files together
STATE_KEY = "shared_labels_pull"  # {"etag", "checked_at", "error", "rejected": {name: {"sha", "why"}}, ...}
CODE_KEY = "shared_labels_code"  # this PC's random contributor code (8 hex characters)
EXPORT_KEY = "shared_labels_export"
REPORT_KEY = "student_training_last"
INDEX = "index.json"


class PullError(Exception):
    pass


def cache_dir() -> Path:
    path = paths.data_dir() / "shared_labels"
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_client() -> httpx.Client:
    return httpx.Client(follow_redirects=True, timeout=httpx.Timeout(30, read=60),
                        headers={"User-Agent": f"NewsTrader/{__version__}"})


def reveal_file(path: Path) -> None:
    from ..updater import reveal_file as reveal

    reveal(path)


def load_index(folder: Path) -> dict[str, dict]:
    """{file name: {"sha", "size", "rows", "code", "mix"}} of the label files kept in `folder` (missing files left
    out)."""
    try:
        index = json.loads((folder / INDEX).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(index, dict):
        return {}
    return {n: v for n, v in index.items() if isinstance(v, dict) and NAME_RE.match(n) and (folder / n).is_file()}


def _write(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


# ------------------------------------------------------------------------------------------------ joining labels
def downloaded_ids(db: Database) -> set[str]:
    return {str(r["news_id"]) for r in db.query("SELECT DISTINCT news_id FROM ml_samples WHERE text_version = 2")}


def collect_labels(db: Database, folder: Path, only: set[str] | None = None) -> tuple[list[dict], int]:
    """Your own labels and every downloaded file's, as {"id", "t", "h", "d", "c", "source"} ("own" or the file's
    contributor code). `only`: keep just these news ids (the headlines this PC downloaded). Returns (labels, how many
    shared labels were left out by `only`)."""
    out = [{"id": str(r["news_id"]), "t": r["ticker"], "h": r["headline_hash"], "d": r["direction"],
            "c": r["confidence"] or 0, "source": "own"}
           for r in db.query("SELECT news_id, ticker, headline_hash, direction, confidence FROM pro_labels "
                             "WHERE status = 'ok'")]
    left_out = 0
    for name, info in sorted(load_index(folder).items()):
        try:
            rows = read_rows(folder / name)
        except Exception as exc:
            log.warning("Couldn't read the shared label file %s: %s", name, exc)
            continue
        code = str(info.get("code") or name)
        for r in rows:
            nid = str(r["id"])
            if only is not None and nid not in only:
                left_out += 1
                continue
            out.append({"id": nid, "t": r["t"], "h": r["h"], "d": r["d"], "c": r["c"], "source": code})
    return out, left_out


def _vote(votes: list[dict]) -> dict:
    """The most common direction among shared labels for one (news, stock); a tie goes to the surer side."""
    tally = Counter(v["d"] for v in votes)
    sure = Counter()
    for v in votes:
        sure[v["d"]] += v["c"]
    best = max(tally, key=lambda d: (tally[d], sure[d]))
    return next(v for v in votes if v["d"] == best)


def join_labels(db: Database, labels: list[dict]) -> tuple[list[dict], dict]:
    """Match labels with the headlines downloaded on this PC (ml_samples) by news id and ticker, keeping only those
    whose headline fingerprint matches. One label per (news, stock): your own wins, otherwise the most common
    direction. Each PC gets one vote: a newer file from the same PC repeats its older labels, and the newest counts
    (`labels` come oldest file first). Returns ([{"label", "row", "sources"}], counts)."""
    by_source: dict[tuple[str, str], dict[str, dict]] = {}
    for lab in labels:
        by_source.setdefault((lab["id"], lab["t"]), {})[lab["source"]] = lab
    by_key = {k: list(v.values()) for k, v in by_source.items()}
    ids = sorted({k[0] for k in by_key})
    rows: dict[tuple[str, str], dict] = {}
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        marks = ",".join("?" * len(chunk))
        for r in db.query(f"SELECT news_id, symbol, published_at, headline, summary, symbols FROM ml_samples "
                          f"WHERE text_version = 2 AND news_id IN ({marks})", chunk):
            rows.setdefault((str(r["news_id"]), r["symbol"]), r)
    counts = {"labels": len(labels), "matched": 0, "not_downloaded": 0, "headline_differs": 0}
    joined = []
    for key, votes in by_key.items():
        row = rows.get(key)
        if row is None:
            counts["not_downloaded"] += 1
            continue
        h = headline_hash(row["headline"] or "")
        votes = [v for v in votes if v["h"] == h]
        if not votes:
            counts["headline_differs"] += 1
            continue
        own = [v for v in votes if v["source"] == "own"]
        joined.append({"label": own[0] if own else _vote(votes), "row": row, "sources": {v["source"] for v in votes}})
    counts["matched"] = len(joined)
    return joined, counts


def build_samples(db: Database, table, model, joined: list[dict], progress: Callable[[float, str], None],
                  cancelled: Callable[[], bool], predict=None) -> list[StudentSample]:
    """The student's inputs for every matched label - built exactly like live analysis builds them."""
    from .engine import readings_for
    from .text import target_text

    by_news: dict[str, list[dict]] = {}
    for j in joined:
        by_news.setdefault(str(j["row"]["news_id"]), []).append(j)
    pending = []  # (label, published, news id, cand, target, reading, cands)
    for nid, items in by_news.items():
        if cancelled():
            raise Cancelled()
        row = items[0]["row"]
        published = parse_iso(row["published_at"])
        if published is None:
            continue
        syms = list(dict.fromkeys([*json.loads(row["symbols"] or "[]"), *(j["label"]["t"] for j in items)]))
        item = NewsItem(source_id="training", source_type="alpaca_news", source_name="Benzinga (history)",
                        external_id=nid, title=row["headline"] or "", body=row["summary"] or "",
                        published_at=published, symbols=syms, language="en")
        cands = []
        for s in syms:
            info = table.get(s) if table is not None else None
            cands.append(Candidate(s, info.name if info else s, "tagged by source"))
        targets = []
        for c in cands:
            t = target_text(item.title, item.body, "text", c, table)
            if t is not None:
                targets.append((c, t))
        readings = dict(zip([c.symbol for c, _t in targets], readings_for(item, targets, table) if targets else [],
                            strict=True))
        found = {c.symbol: (c, t) for c, t in targets}
        for j in items:
            sym = j["label"]["t"]
            if sym in found:
                c, t = found[sym]
                pending.append((j, published, nid, c, t, readings.get(sym), cands))
    scores = cached_sentiment(db, model, [p[4].snippet for p in pending], progress, cancelled, predict)
    out = []
    for (j, published, nid, c, t, reading, cands), s in zip(pending, scores, strict=True):
        source = "own" if "own" in j["sources"] else j["label"]["source"]
        out.append(StudentSample(feats=student_features(c, t, s, table, reading, cands), label=j["label"]["d"],
                                 at=published, group=nid, source=source))
    return out


def comparison_text(report: dict) -> str:
    """How a new student did against the one in use (student.compare_with), in words."""
    cur, new = report.get("current_accuracy"), report.get("new_accuracy", report.get("accuracy"))
    if report.get("compare_basis") == "same_labels":
        return (f"on the {int(report.get('compared_on') or 0):,} newest labels the student in use never learned from, "
                f"the new one agreed with Pro AI on {new}% and the one in use on {cur}%")
    return (f"the new one agreed with Pro AI on {new}% of the newest labels; the student in use got {cur}% on its own "
            "test when it was trained (it already learned from these labels, so they can't test it)")


# ------------------------------------------------------------------------------------------------ the service
class SharedTraining:
    name = "shared_training"

    def __init__(self, ctx, client_factory: Callable[[], httpx.Client] | None = None,
                 reveal: Callable[[Path], None] | None = None, open_url: Callable[[str], object] | None = None,
                 downloads: Path | None = None, first_delay: float = FIRST_PULL_DELAY):
        """`client_factory`, `reveal`, `open_url` and `downloads` (the folder "Share my labels" saves to) are for
        tests."""
        self.ctx = ctx
        self._client_factory = client_factory or make_client
        self._reveal = reveal or reveal_file
        self._open_url = open_url or webbrowser.open
        self._downloads = downloads
        self.first_delay = first_delay
        self.pulling = False
        self.progress: dict | None = None  # while training: {"progress", "message"}
        self._loop_task: asyncio.Task | None = None
        self._train_task: asyncio.Task | None = None
        self._cancel = False

    async def start(self) -> None:
        self._loop_task = asyncio.create_task(self._loop(), name="shared-labels")

    async def stop(self) -> None:
        self._cancel = True
        for task in (self._loop_task, self._train_task):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task

    @property
    def training(self) -> bool:
        return self._train_task is not None and not self._train_task.done()

    def _state(self) -> dict:
        state = self.ctx.db.kv_get(STATE_KEY) or {}
        return state if isinstance(state, dict) else {}

    def contributor_code(self) -> str:
        """A random code made once on this PC (the end of a shared file's name). It isn't linked to you."""
        code = self.ctx.db.kv_get(CODE_KEY)
        if not (isinstance(code, str) and len(code) == 8):
            code = secrets.token_hex(4)
            self.ctx.db.kv_set(CODE_KEY, code)
        return code

    # ------------------------------------------------------------------ getting everyone's labels
    async def _loop(self) -> None:
        await asyncio.sleep(self.first_delay)
        while True:
            if self.ctx.config.settings.ml.shared_labels_pull and self.pull_due() and not self.pulling:
                try:
                    await self.pull()
                except Exception:
                    log.debug("Getting shared labels failed", exc_info=True)
            await asyncio.sleep(LOOP_EVERY)

    def pull_due(self, now: datetime | None = None) -> bool:
        """Once a day (after a problem: again 3 hours later), also across restarts."""
        state = self._state()
        last = parse_iso(state.get("checked_at"))
        if last is None:
            return True
        gap = RETRY_AFTER_ERROR if state.get("error") else PULL_EVERY
        return (now or datetime.now(UTC)) - last >= timedelta(seconds=gap)

    async def pull(self) -> dict:
        """Get any new shared label files (call in the event loop)."""
        if self.pulling:
            raise RuntimeError("Already getting everyone's labels.")
        self.pulling = True
        try:
            res = await asyncio.to_thread(self.pull_sync)
        finally:
            self.pulling = False
        self.ctx.bus.publish("shared_labels", {"pulled": res})
        return res

    def pull_sync(self) -> dict:
        """Blocking: list shared/labels on GitHub, download new files, check them, keep the good ones."""
        folder = cache_dir()
        index = load_index(folder)
        state = self._state()
        rejected = state.get("rejected") if isinstance(state.get("rejected"), dict) else {}
        headers = {"Accept": "application/vnd.github+json"}
        if state.get("etag") and (index or rejected or state.get("empty")):
            headers["If-None-Match"] = state["etag"]
        added, removed, error, etag, complete, empty = [], [], "", None, True, False
        try:
            with self._client_factory() as client:
                r = client.get(LIST_URL, params={"ref": BRANCH}, headers=headers)
                if r.status_code == 304:
                    self.ctx.db.kv_set(STATE_KEY, {**state, "checked_at": iso(), "error": ""})
                    return self._pull_result(added, removed, "", unchanged=True)
                if r.status_code in (403, 429):
                    raise PullError("GitHub asked NewsTrader to slow down (too many requests) - it tries again later.")
                if r.status_code == 404:
                    listing, empty = [], True  # nothing shared yet
                else:
                    r.raise_for_status()
                    listing = r.json()
                if not isinstance(listing, list):
                    raise PullError("GitHub's answer wasn't a list of files.")
                etag = r.headers.get("ETag")
                files = {str(e.get("name")): e for e in listing
                         if isinstance(e, dict) and e.get("type") == "file" and NAME_RE.match(str(e.get("name")))}
                empty = not files  # (so "nothing new" can be asked next time, even with no files kept here)
                for name in list(index):  # removed from the repo (or replaced): not used any more
                    if name not in files or str(files[name].get("sha")) != index[name].get("sha"):
                        with contextlib.suppress(OSError):
                            (folder / name).unlink()
                        del index[name]
                        removed.append(name)
                rejected = {n: v for n, v in rejected.items()
                            if n in files and isinstance(v, dict) and str(files[n].get("sha")) == v.get("sha")}
                used = sum(int(v.get("size") or 0) for v in index.values())
                for name in sorted(files):
                    if name in index or name in rejected:
                        continue
                    sha, size = str(files[name].get("sha") or ""), int(files[name].get("size") or 0)
                    if size > MAX_BYTES:
                        rejected[name] = {"sha": sha, "why": "bigger than 20 MB"}
                        continue
                    if used + size > MAX_CACHE_BYTES:
                        error, complete = "The shared labels already take 500 MB on this PC - no more are kept.", False
                        break
                    try:
                        data = self._download(client, name)
                    except Exception as exc:
                        log.warning("Couldn't download the shared label file %s: %s", name, exc)
                        complete = False
                        continue
                    if blob_sha(data) != sha:  # changed since it was listed, or damaged: tried again next time
                        log.warning("Shared label file %s doesn't match GitHub's fingerprint - not used", name)
                        complete = False
                        continue
                    res = check_file(name, data, known_only=False)
                    if not res.ok:
                        log.warning("Shared label file rejected: %s", res.summary)
                        rejected[name] = {"sha": sha, "why": res.summary[:300]}
                        continue
                    _write(folder / name, data)
                    index[name] = {"sha": sha, "size": len(data), "rows": res.n_rows, "code": res.code,
                                   "mix": res.mix, "added_at": iso()}
                    used += len(data)
                    added.append(name)
        except PullError as exc:
            error = str(exc)
        except (httpx.HTTPError, ValueError) as exc:
            error = f"Couldn't reach GitHub ({type(exc).__name__}) - it tries again later."
            log.warning("Getting shared labels failed: %s", exc)
        _write(folder / INDEX, json.dumps(index, indent=1).encode("utf-8"))
        new_state = {**state, "checked_at": iso(), "error": error, "rejected": rejected, "empty": empty,
                     "etag": etag if complete and etag else None}
        if error and etag is None:  # the listing itself failed: keep what we had
            new_state.update(etag=state.get("etag"), empty=state.get("empty", False))
        self.ctx.db.kv_set(STATE_KEY, new_state)
        if added or removed:
            log.info("Shared labels: %d new file(s), %d removed", len(added), len(removed))
        return self._pull_result(added, removed, error)

    def _download(self, client: httpx.Client, name: str) -> bytes:
        buf = bytearray()
        with client.stream("GET", RAW_URL.format(name=name)) as r:
            r.raise_for_status()
            for chunk in r.iter_bytes():
                buf += chunk
                if len(buf) > MAX_BYTES:
                    raise ValueError("bigger than 20 MB")
        return bytes(buf)

    def _pull_result(self, added: list[str], removed: list[str], error: str, unchanged: bool = False) -> dict:
        return {"added": added, "removed": removed, "error": error, "unchanged": unchanged,
                **self.shared_summary()}

    def shared_summary(self) -> dict:
        index = load_index(cache_dir())
        state = self._state()
        rejected = state.get("rejected") if isinstance(state.get("rejected"), dict) else {}
        return {"files": len(index), "labels": sum(int(v.get("rows") or 0) for v in index.values()),
                "contributors": len({v.get("code") for v in index.values()}),
                "checked_at": state.get("checked_at"), "error": state.get("error") or "",
                "rejected": [{"name": n, "why": v.get("why", "")} for n, v in sorted(rejected.items())]}

    # ------------------------------------------------------------------ "Share my labels"
    def downloads_folder(self) -> Path:
        if self._downloads is not None:
            return self._downloads
        downloads = Path.home() / "Downloads"
        return downloads if downloads.is_dir() else paths.data_dir()

    def export(self) -> dict:
        """Blocking: save your labels as one file (Downloads, else the data folder), show it and open GitHub's upload
        page. Raises ValueError (with a message for you) when there is nothing to share."""
        today = utc_today()
        labels = self.ctx.db.query("SELECT * FROM pro_labels WHERE status = 'ok' "
                                   "ORDER BY day DESC, CAST(news_id AS INTEGER) DESC, ticker")
        rows = [r for r in (to_row(lab) for lab in labels) if r is not None and not check_row(r, today)]
        skipped = len(labels) - len(rows)
        if not rows:
            raise ValueError("There are no labels to share yet - let Pro AI label old news first (Settings -> Pro AI "
                             "-> Label old news).")
        rows, trimmed = trim_mix(rows[:MAX_ROWS])
        name = file_name(today, self.contributor_code())
        data = dump(rows)
        check = check_file(name, data, today)
        if not check.ok:  # never hand over a file GitHub's check would turn down
            raise ValueError("The file didn't pass the check GitHub runs on uploads: " + check.summary)
        folder = self.downloads_folder()
        try:
            target = folder / name
            _write(target, data)
        except OSError:
            folder = paths.data_dir()
            target = folder / name
            _write(target, data)
        with contextlib.suppress(Exception):
            self._reveal(target)
        with contextlib.suppress(Exception):
            self._open_url(UPLOAD_URL)
        where = "your Downloads folder" if folder.name == "Downloads" else str(folder)
        out = {"file": str(target), "name": name, "folder": str(folder), "where": where, "rows": len(rows),
               "trimmed": trimmed, "skipped": skipped, "upload_url": UPLOAD_URL, "steps": [
                   f"On the GitHub page that just opened, sign in (a free GitHub account is enough), then drag {name} "
                   f"from {where} onto the page.",
                   "Scroll down and click the green button: \"Propose changes\" (it may say \"Commit changes\").",
                   "Click \"Create pull request\". The owner checks every upload before it is added for everyone.",
               ]}
        self.ctx.db.kv_set(EXPORT_KEY, {"at": iso(), "name": name, "rows": len(rows), "file": str(target)})
        log.info("Shared labels saved for upload: %s (%d labels)", target, len(rows))
        return out

    # ------------------------------------------------------------------ "Train with everyone's labels"
    def train(self) -> dict:
        """Start training the student in the background (call in the event loop)."""
        if self.training:
            raise RuntimeError("The student model is already training.")
        trainer = self.ctx.service("ml_trainer")
        if trainer is not None and getattr(trainer, "running", False):
            raise RuntimeError("Wait for the price model's training to finish first.")
        bt = self.ctx.service("backtest")
        if bt is not None and getattr(bt, "running", False):
            raise RuntimeError("Wait for the backtest to finish first.")
        if self.ctx.service("pipeline") is None:
            raise RuntimeError("The AI engine isn't running yet.")
        self._cancel = False
        self.progress = {"progress": 0.0, "message": "starting..."}
        self._train_task = asyncio.get_running_loop().create_task(self._guarded_train(), name="student-train")
        return {"ok": True}

    def cancel_training(self) -> None:
        self._cancel = True

    def _progress(self, progress: float, message: str, status: str = "running") -> None:
        self.progress = {"progress": round(progress, 3), "message": message, "status": status}
        self.ctx.bus.publish("student_training", self.progress)

    async def _guarded_train(self) -> None:
        try:
            await self.run_training()
        except (asyncio.CancelledError, Cancelled):
            self._progress(1, "cancelled", "cancelled")
        except Exception as exc:
            log.warning("Student training failed: %s", exc)
            self._progress(1, f"failed: {exc}", "failed")
            self.ctx.db.kv_set(REPORT_KEY, {"at": iso(), "error": str(exc)})

    async def run_training(self) -> dict:
        ctx = self.ctx
        pipeline = ctx.service("pipeline")
        engine, table = pipeline.local, pipeline.tickers
        loop = asyncio.get_running_loop()

        def cancelled() -> bool:
            return self._cancel

        def stage(lo: float, hi: float):
            def report(frac: float, msg: str) -> None:
                loop.call_soon_threadsafe(self._progress, lo + (hi - lo) * max(0.0, min(1.0, frac)), msg)
            return report

        self._progress(0.01, "loading the sentiment model...")
        await engine.ensure_ready()
        model = engine.sentiment.model  # kept for the whole run, even if the live engine switches models
        if model is None:
            raise RuntimeError("The sentiment model didn't load - see Logs.")

        def predict(texts):
            return engine.sentiment.predict_with(model, texts)

        self._progress(0.03, "matching labels with your downloaded headlines...")
        known = await asyncio.to_thread(downloaded_ids, ctx.db)
        labels, left_out = await asyncio.to_thread(collect_labels, ctx.db, cache_dir(), known)
        joined, counts = await asyncio.to_thread(join_labels, ctx.db, labels)
        counts["labels"] += left_out
        counts["not_downloaded"] += left_out
        samples = await asyncio.to_thread(build_samples, ctx.db, table, model, joined, stage(0.05, 0.4), cancelled,
                                          predict)
        if cancelled():
            raise Cancelled()
        current = engine.current_student()
        if current is not None and (current.meta.get("sentiment_model") != model.model_id
                                    or not ctx.config.settings.ml.use_student):
            current = None  # not the one in use: nothing to compare with
        self._progress(0.4, f"training on {len(samples):,} labels...")
        student, report = await asyncio.to_thread(train_student, samples, model.model_id, current, stage(0.4, 0.92))
        if cancelled():
            raise Cancelled()
        report["samples_check"] = await asyncio.to_thread(score_samples, student, predict, table)
        sources = Counter(s.source for s in samples)
        mine = ("own", self.contributor_code())  # (your own file may be among everyone's too)
        report["used"] = {**counts, "own": sources.get("own", 0), "shared": len(samples) - sources.get("own", 0),
                          "contributors": len({s for j in joined for s in j["sources"] if s not in mine})
                          + (1 if sources.get("own") else 0)}
        report["switched"] = bool(report["not_worse"])
        student.meta["report"] = report
        if report["switched"]:
            await asyncio.to_thread(student.save, engine.student_dir)
            engine.set_student(student)
            pipeline.update_ml_status()
            msg = (f"done: the student model learned from {len(samples):,} labels and agreed with Pro AI on "
                   f"{report['accuracy']}% of the newest {report['n_valid']:,} it never saw. The engine now uses it.")
            if report.get("current_accuracy") is not None:
                msg += f" ({comparison_text(report)}.)"
        else:
            msg = f"done, but not used: {comparison_text(report)} - so the one in use stays."
        ctx.db.kv_set(REPORT_KEY, {"at": iso(), **report})
        self._progress(1, msg, "done")
        ctx.bus.publish("toast", {"kind": "success" if report["switched"] else "info",
                                  "title": "Student model trained", "message": msg})
        return report

    def delete_student(self) -> None:
        if self.training:
            raise RuntimeError("The student model is training - cancel it first.")
        pipeline = self.ctx.service("pipeline")
        if pipeline is None:
            raise RuntimeError("The AI engine isn't running yet.")
        delete_student(pipeline.local.student_dir)
        pipeline.local.load_student()  # (one that came with the app, if any)
        pipeline.update_ml_status()

    # ------------------------------------------------------------------ status
    def status(self) -> dict:
        """For Backtest -> Local ML model -> Shared training (reads the database - call from a worker thread)."""
        db = self.ctx.db
        pipeline = self.ctx.service("pipeline")
        own = db.scalar("SELECT COUNT(*) FROM pro_labels WHERE status = 'ok'") or 0
        return {"own_labels": int(own), "shared": {**self.shared_summary(), "pulling": self.pulling},
                "pull_daily": self.ctx.config.settings.ml.shared_labels_pull,
                "use_student": self.ctx.config.settings.ml.use_student,
                "training": self.training, "progress": self.progress, "last": db.kv_get(REPORT_KEY),
                "last_export": db.kv_get(EXPORT_KEY), "upload_url": UPLOAD_URL,
                "student": pipeline.local.student_status() if pipeline is not None else None}
