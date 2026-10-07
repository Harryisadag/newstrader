"""Logging: a rotating log file + the Logs tab (SQLite) + live push to the UI.

Records are handed to a background thread through a queue, so logging never blocks the engine and a
database problem can't cause recursive logging.
"""

from __future__ import annotations

import logging
import logging.handlers
import queue
import threading
from datetime import timedelta

from .db import Database, iso, utcnow
from .events import EventBus

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


class _DbLogWriter(threading.Thread):
    def __init__(self, db: Database, bus: EventBus | None):
        super().__init__(name="log-writer", daemon=True)
        self.db = db
        self.bus = bus
        self.q: queue.Queue[logging.LogRecord | None] = queue.Queue(maxsize=10000)

    def run(self) -> None:
        while True:
            record = self.q.get()
            if record is None:
                return
            try:
                message = record.getMessage()
                if record.exc_info and record.exc_info[1] is not None:
                    message += f" | {type(record.exc_info[1]).__name__}: {record.exc_info[1]}"
                row = {"ts": iso(), "level": record.levelname, "logger": record.name, "message": message[:4000]}
                row["id"] = self.db.insert("logs", row)
                if self.bus is not None:
                    self.bus.publish("log", row)
            except Exception:  # never let logging kill the app
                pass

    def stop(self) -> None:
        try:
            self.q.put_nowait(None)
        except queue.Full:
            pass


class _QueueToDb(logging.Handler):
    def __init__(self, writer: _DbLogWriter):
        super().__init__()
        self.writer = writer

    def emit(self, record: logging.LogRecord) -> None:
        # Our own modules at INFO+, third-party libraries only at WARNING+ (they're chatty).
        if not record.name.startswith("newstrader") and record.levelno < logging.WARNING:
            return
        try:
            self.writer.q.put_nowait(record)
        except queue.Full:
            pass


_writer: _DbLogWriter | None = None


def setup_file_logging(log_dir, level: int = logging.INFO) -> None:
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        if getattr(h, "_newstrader", False):
            root.removeHandler(h)
    fh = logging.handlers.RotatingFileHandler(log_dir / "newstrader.log", maxBytes=5_000_000,
                                              backupCount=5, encoding="utf-8")
    fh.setFormatter(logging.Formatter(LOG_FORMAT))
    fh._newstrader = True  # type: ignore[attr-defined]
    root.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(LOG_FORMAT))
    sh._newstrader = True  # type: ignore[attr-defined]
    root.addHandler(sh)
    # Quieten noisy libraries
    for name in ("httpx", "httpcore", "httpx2", "urllib3", "websockets", "uvicorn.access", "faster_whisper",
                 "anthropic"):
        logging.getLogger(name).setLevel(logging.WARNING)


def attach_db_logging(db: Database, bus: EventBus | None, keep_days: int = 14) -> None:
    global _writer
    root = logging.getLogger()
    for h in list(root.handlers):
        if isinstance(h, _QueueToDb):
            root.removeHandler(h)
            h.writer.stop()
    try:
        cutoff = iso(utcnow() - timedelta(days=keep_days))
        db.execute("DELETE FROM logs WHERE ts < ?", (cutoff,))
    except Exception:
        pass
    _writer = _DbLogWriter(db, bus)
    _writer.start()
    handler = _QueueToDb(_writer)
    handler._newstrader = True  # type: ignore[attr-defined]
    handler.setLevel(logging.INFO)
    root.addHandler(handler)


def shutdown_logging() -> None:
    if _writer is not None:
        _writer.stop()
