"""Rate limits for the desk's free data sources.

Every call to a host goes through that host's Limiter: calls start at least `interval` seconds apart, at most
`per_minute` start in any 60 seconds, and after a 429 ("too many requests") nobody calls the host again until it said
to (x-ratelimit-reset or Retry-After) - or, when it doesn't say, for a pause that doubles each time (2, 4, 8... s).
"""

from __future__ import annotations

import asyncio
import email.utils
import time
from collections import deque

from .parsers import RateLimited


def _header(headers, name: str) -> str | None:
    if not headers:
        return None
    for key, value in dict(headers).items():
        if str(key).lower() == name:
            return str(value).strip()
    return None


def told_wait(headers, wall_now: float) -> float | None:
    """How long the host asked us to wait, in seconds (None if it didn't say)."""
    reset = _header(headers, "x-ratelimit-reset")
    if reset:
        try:
            value = float(reset)
        except ValueError:
            value = None
        if value is not None:
            if value > 1e12:  # milliseconds since 1970
                value /= 1000
            if value > 1e9:  # seconds since 1970: when the window resets
                return max(1.0, value - wall_now)
            return max(1.0, value)  # seconds from now
    after = _header(headers, "retry-after")
    if after:
        try:
            return max(0.0, float(after))
        except ValueError:
            pass
        try:
            when = email.utils.parsedate_to_datetime(after)
            return max(0.0, when.timestamp() - wall_now)
        except (TypeError, ValueError, IndexError):
            return None
    return None


class Limiter:
    """Spaces out the calls to one host and pauses it after a 429."""

    def __init__(self, name: str, interval: float, per_minute: int | None = None, max_backoff: float = 120.0,
                 clock=time.monotonic, wall=time.time, sleep=asyncio.sleep):
        self.name = name
        self.interval = interval
        self.per_minute = per_minute
        self.max_backoff = max_backoff
        self._clock, self._wall, self._sleep = clock, wall, sleep
        self._next = 0.0
        self._blocked_until = 0.0
        self._starts: deque[float] = deque()
        self._strikes = 0
        self._lock = asyncio.Lock()
        self.calls = 0
        self.limited = 0

    def _slot(self, now: float) -> float:
        start = max(now, self._next, self._blocked_until)
        if self.per_minute:
            while self._starts and self._starts[0] <= now - 60:
                self._starts.popleft()
            if len(self._starts) >= self.per_minute:
                start = max(start, self._starts[len(self._starts) - self.per_minute] + 60)
        return start

    def wait_time(self) -> float:
        """Seconds until a call could start."""
        now = self._clock()
        return max(0.0, self._slot(now) - now)

    def cooling_down(self) -> float:
        """Seconds left of a pause after a 429 (0 = not paused)."""
        return max(0.0, self._blocked_until - self._clock())

    def free(self) -> int:
        """Calls still allowed in the current minute."""
        if not self.per_minute:
            return 1_000_000
        now = self._clock()
        self._slot(now)
        return max(0, self.per_minute - sum(1 for t in self._starts if t > now - 60))

    async def acquire(self, max_wait: float = 30.0) -> None:
        """Wait for this host's next free slot. Raises RateLimited when that is more than max_wait away."""
        async with self._lock:
            now = self._clock()
            start = self._slot(now)
            if start - now > max_wait:
                why = "is pausing after too many requests" if self._blocked_until > now else "is busy"
                raise RateLimited(f"{self.name} {why} - next call in {start - now:.0f} s")
            self._next = start + self.interval
            self._starts.append(start)
        delay = start - self._clock()
        if delay > 0:
            await self._sleep(delay)
        left = self._blocked_until - self._clock()
        if left > 0:  # a 429 came in while this call waited
            raise RateLimited(f"{self.name} is pausing after too many requests - next call in {left:.0f} s")
        self.calls += 1

    def hit_limit(self, headers=None) -> float:
        """The host answered 429: pause it. Returns the pause in seconds."""
        self.limited += 1
        pause = told_wait(headers, self._wall())
        if pause is None:
            pause = 2.0 * 2 ** self._strikes
            self._strikes += 1
        pause = min(max(pause, 0.5), self.max_backoff)
        self._blocked_until = max(self._blocked_until, self._clock() + pause)
        return pause

    def ok(self) -> None:
        """A normal answer: the next 429 starts the backoff from the beginning."""
        self._strikes = 0
