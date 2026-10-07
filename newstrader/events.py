"""In-process event bus. Anything (any thread) can publish; the UI websocket subscribes.

Event shape sent to the browser: {"type": "...", "data": {...}, "ts": "..."}
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from typing import Any

from .db import iso

log = logging.getLogger(__name__)


class EventBus:
    def __init__(self, queue_size: int = 2000):
        self._queue_size = queue_size
        self._subscribers: set[asyncio.Queue] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=self._queue_size)
        with self._lock:
            self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._lock:
            self._subscribers.discard(q)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def publish(self, event_type: str, data: Any = None) -> None:
        """Safe to call from any thread."""
        msg = {"type": event_type, "data": data, "ts": iso()}
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            self._dispatch(msg)
        else:
            with contextlib.suppress(RuntimeError):  # loop shutting down
                loop.call_soon_threadsafe(self._dispatch, msg)

    def _dispatch(self, msg: dict) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for q in subscribers:
            if q.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    q.get_nowait()  # drop the oldest event for slow clients
            with contextlib.suppress(asyncio.QueueFull):
                q.put_nowait(msg)
