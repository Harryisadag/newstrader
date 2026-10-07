from __future__ import annotations

import pytest

from newstrader.db import Database
from newstrader.events import EventBus
from newstrader.keys import Keys
from newstrader.state import AppState
from newstrader.trading import live_guard as lg

PAPER = dict(alpaca_paper_key="PK1", alpaca_paper_secret="ps1")
LIVE = dict(alpaca_live_key="AK1", alpaca_live_secret="ls1")


@pytest.fixture
def state(tmp_path):
    return AppState(Database(tmp_path / "t.db"), EventBus())


def test_starts_in_paper_and_uses_paper_keys(state):
    creds = lg.credentials_for_mode(state, Keys(**PAPER, **LIVE))
    assert creds.paper is True and creds.api_key == "PK1"


def test_no_paper_keys_means_no_credentials(state):
    assert lg.credentials_for_mode(state, Keys(**LIVE)) is None


@pytest.mark.parametrize("enable,phrase", [
    (False, lg.CONFIRM_PHRASE),
    (True, ""),
    (True, "i understand this uses real money"),
    (True, "I UNDERSTAND"),
    ("yes", lg.CONFIRM_PHRASE),
])
def test_arming_requires_explicit_enable_and_exact_phrase(state, enable, phrase):
    with pytest.raises(lg.LiveTradingLocked):
        lg.arm_live_trading(state, Keys(**PAPER, **LIVE), enable=enable, phrase=phrase)
    assert state.mode == "paper"


def test_arming_requires_live_keys(state):
    with pytest.raises(lg.LiveTradingLocked):
        lg.arm_live_trading(state, Keys(**PAPER), enable=True, phrase=lg.CONFIRM_PHRASE)
    assert state.mode == "paper"


def test_live_keys_must_differ_from_paper(state):
    keys = Keys(alpaca_paper_key="SAME", alpaca_paper_secret="x", alpaca_live_key="SAME", alpaca_live_secret="y")
    with pytest.raises(lg.LiveTradingLocked):
        lg.arm_live_trading(state, keys, enable=True, phrase=lg.CONFIRM_PHRASE)


def test_arm_then_disarm(state):
    keys = Keys(**PAPER, **LIVE)
    lg.arm_live_trading(state, keys, enable=True, phrase=lg.CONFIRM_PHRASE)
    assert state.mode == "live"
    creds = lg.credentials_for_mode(state, keys)
    assert creds.paper is False and creds.api_key == "AK1"
    lg.disarm_live_trading(state)
    assert lg.credentials_for_mode(state, keys).paper is True


def test_live_mode_does_not_survive_restart(tmp_path):
    db_path = tmp_path / "t.db"
    s1 = AppState(Database(db_path), EventBus())
    lg.arm_live_trading(s1, Keys(**PAPER, **LIVE), enable=True, phrase=lg.CONFIRM_PHRASE)
    s2 = AppState(Database(db_path), EventBus())
    assert s2.mode == "paper"


def test_live_client_refused_unless_armed(state):
    with pytest.raises(lg.LiveTradingLocked):
        lg.assert_mode_allowed(state, paper=False)
    lg.assert_mode_allowed(state, paper=True)


def test_broker_refuses_live_credentials_when_not_armed(state):
    from newstrader.trading.broker import Broker

    with pytest.raises(lg.LiveTradingLocked):
        Broker(lg.AlpacaCredentials("AK", "SK", paper=False), state)


def test_live_api_endpoint_requires_phrase(monkeypatch, client):
    client.put("/api/keys", json={"ALPACA_PAPER_API_KEY": "PK1", "ALPACA_PAPER_SECRET_KEY": "s",
                                  "ALPACA_LIVE_API_KEY": "AK1", "ALPACA_LIVE_SECRET_KEY": "l"})
    r = client.post("/api/live/arm", json={"enable": True, "phrase": "nope"})
    assert r.status_code == 403
    assert client.get("/api/status").json()["mode"] == "paper"
    r = client.post("/api/live/arm", json={"phrase": lg.CONFIRM_PHRASE})  # enable missing
    assert r.status_code == 403
    r = client.post("/api/live/arm", json={"enable": True, "phrase": lg.CONFIRM_PHRASE})
    assert r.status_code == 200 and r.json()["mode"] == "live"
    r = client.post("/api/live/disarm")
    assert r.json()["mode"] == "paper"
