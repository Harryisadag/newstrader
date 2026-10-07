from __future__ import annotations

import threading

from newstrader.db import Database
from newstrader.events import EventBus
from newstrader.state import AppState, trading_day


def test_migrations_and_kv(tmp_path):
    db = Database(tmp_path / "t.db")
    assert db.scalar("PRAGMA user_version") >= 1
    db.kv_set("x", {"a": 1})
    assert db.kv_get("x") == {"a": 1}
    assert db.kv_get("missing", 5) == 5
    # migrating again is a no-op
    Database(tmp_path / "t.db")


def test_concurrent_writes(tmp_path):
    db = Database(tmp_path / "t.db")

    def worker(n):
        for i in range(50):
            db.insert("logs", {"ts": "t", "level": "INFO", "logger": f"w{n}", "message": str(i)})

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert db.scalar("SELECT COUNT(*) FROM logs") == 200


def test_kill_switch_and_halt_persist(tmp_path):
    db = Database(tmp_path / "t.db")
    state = AppState(db, EventBus())
    assert not state.kill_engaged
    state.engage_kill_switch("test")
    # a "restart" (new state object, same database) keeps the kill switch on
    state2 = AppState(Database(tmp_path / "t.db"), EventBus())
    assert state2.kill_engaged
    state2.release_kill_switch()
    assert not state2.kill_engaged

    state2.halt_for_today("loss limit")
    assert state2.halted_today
    db.kv_set("halted_day", "2000-01-01")  # yesterday's halt doesn't block today
    assert not state2.halted_today


def test_live_flag_never_persists(tmp_path):
    db = Database(tmp_path / "t.db")
    state = AppState(db, EventBus())
    state._set_live_armed(True)
    assert state.mode == "live"
    restarted = AppState(Database(tmp_path / "t.db"), EventBus())
    assert restarted.mode == "paper"


def test_trading_day_format():
    assert len(trading_day()) == 10
