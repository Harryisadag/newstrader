"""Runtime state shared by the engine and the UI.

- Component health (GPU, Alpaca, Claude, each stream/feed) for the header and Logs tab.
- Kill switch and daily-loss halt: saved in the database, so they survive a restart.
- Live trading flag: memory only, on purpose - every restart goes back to paper.
"""

from __future__ import annotations

import threading
from datetime import datetime
from zoneinfo import ZoneInfo

from .db import Database, iso
from .events import EventBus

MARKET_TZ = ZoneInfo("America/New_York")

LEVELS = ("ok", "warn", "error", "off", "starting")


def trading_day(now: datetime | None = None) -> str:
    """The US market calendar date (New York time) as YYYY-MM-DD."""
    now = now or datetime.now(MARKET_TZ)
    return now.astimezone(MARKET_TZ).date().isoformat()


class AppState:
    def __init__(self, db: Database, bus: EventBus):
        self.db = db
        self.bus = bus
        self._lock = threading.RLock()
        self._components: dict[str, dict] = {}
        self._live_armed = False

    # ---- component health ----
    def set_status(self, component: str, level: str, detail: str = "", **extra) -> None:
        if level not in LEVELS:
            level = "warn"
        entry = {"component": component, "level": level, "detail": detail, "updated_at": iso(), **extra}
        with self._lock:
            prev = self._components.get(component)
            self._components[component] = entry
        if prev is None or prev.get("level") != level or prev.get("detail") != detail or extra:
            self.bus.publish("status", entry)

    def clear_status(self, component: str) -> None:
        with self._lock:
            self._components.pop(component, None)
        self.bus.publish("status_removed", {"component": component})

    def components(self) -> list[dict]:
        with self._lock:
            return sorted(self._components.values(), key=lambda c: c["component"])

    # ---- kill switch (persisted) ----
    @property
    def kill_switch(self) -> dict:
        return self.db.kv_get("kill_switch", {"engaged": False})

    @property
    def kill_engaged(self) -> bool:
        return bool(self.kill_switch.get("engaged"))

    def engage_kill_switch(self, reason: str = "manual") -> None:
        self.db.kv_set("kill_switch", {"engaged": True, "at": iso(), "reason": reason})
        self.bus.publish("kill_switch", self.kill_switch)

    def release_kill_switch(self) -> None:
        self.db.kv_set("kill_switch", {"engaged": False, "released_at": iso()})
        self.bus.publish("kill_switch", self.kill_switch)

    # ---- daily loss halt (persisted, auto-clears the next trading day) ----
    @property
    def halted_today(self) -> bool:
        return self.db.kv_get("halted_day") == trading_day()

    def halt_for_today(self, reason: str) -> None:
        self.db.kv_set("halted_day", trading_day())
        self.db.kv_set("halted_reason", reason)
        self.bus.publish("halted", {"day": trading_day(), "reason": reason})

    def clear_halt(self) -> None:
        self.db.kv_set("halted_day", None)
        self.bus.publish("halted", {"day": None, "reason": ""})

    # ---- live trading (memory only - never saved) ----
    @property
    def live_armed(self) -> bool:
        return self._live_armed

    def _set_live_armed(self, value: bool) -> None:
        """Only trading/live_guard.py should call this."""
        self._live_armed = bool(value)
        self.bus.publish("mode", {"mode": "live" if self._live_armed else "paper"})

    @property
    def mode(self) -> str:
        return "live" if self._live_armed else "paper"
