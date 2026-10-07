"""AI output validation - nothing Claude says can trade unless it passes these checks."""

from __future__ import annotations

import json

import pytest

from newstrader.ai.validator import validate_response
from newstrader.db import Database

from .helpers import make_tickers, sig, signal_json


@pytest.fixture
def tickers(tmp_path):
    return make_tickers(Database(tmp_path / "t.db"))


def test_valid_response(tickers):
    res = validate_response(signal_json(sig()), tickers)
    assert res.status == "ok" and not res.problems
    s = res.signals[0]
    assert (s.ticker, s.direction, s.confidence, s.time_sensitivity) == ("NVDA", "bullish", 88, "immediate")
    assert s.company == "NVIDIA Corporation Common Stock"  # filled from the ticker table when blank


def test_empty_signals_list_is_fine(tickers):
    res = validate_response(signal_json(), tickers)
    assert res.status == "ok" and res.signals == []


@pytest.mark.parametrize("raw", ["", "not json", "{'signals': []}", "[1,2]", '{"signals": "none"}', '{"other": []}',
                                 '{"signals": [', "null"])
def test_bad_json_is_rejected(tickers, raw):
    res = validate_response(raw, tickers)
    assert res.status == "rejected" and res.fatal


def test_code_fenced_json_is_accepted(tickers):
    res = validate_response("```json\n" + signal_json(sig()) + "\n```", tickers)
    assert res.status == "ok" and len(res.signals) == 1


@pytest.mark.parametrize("ticker,reason", [
    ("FAKE", "not a US-listed"),
    ("OTCX", "not a US-listed"),  # OTC excluded from the table
    ("HALT", "isn't tradable"),
    ("", "missing ticker"),
    ("BTC-USD", "valid ticker format"),
    ("TOOLONGX", "valid ticker format"),
    ("nv da", "valid ticker format"),
])
def test_bad_tickers_rejected(tickers, ticker, reason):
    res = validate_response(signal_json(sig(ticker=ticker)), tickers)
    assert res.status == "rejected"
    assert reason in res.problems[0]


def test_lowercase_and_cashtag_tickers_normalised(tickers):
    res = validate_response(signal_json(sig(ticker="$aapl"), sig(ticker="brk.b")), tickers)
    assert [s.ticker for s in res.signals] == ["AAPL", "BRK.B"] or {s.ticker for s in res.signals} == {"AAPL", "BRK.B"}


@pytest.mark.parametrize("conf", [-5, 101, 150, "high", "85", None, 85.5, True, [80]])
def test_bad_confidence_rejected(tickers, conf):
    res = validate_response(signal_json(sig(confidence=conf)), tickers)
    assert res.status == "rejected" and "confidence" in res.problems[0]


@pytest.mark.parametrize("conf", [0, 100, 85.0])
def test_edge_confidence_accepted(tickers, conf):
    assert validate_response(signal_json(sig(confidence=conf)), tickers).status == "ok"


@pytest.mark.parametrize("field,value", [
    ("direction", "up"), ("direction", "BUY"), ("direction", None),
    ("time_sensitivity", "soon"), ("time_sensitivity", None),
    ("reasoning", ""), ("reasoning", 42), ("bull_case", ["a"]),
])
def test_bad_fields_rejected(tickers, field, value):
    res = validate_response(signal_json(sig(**{field: value})), tickers)
    assert res.status == "rejected"


def test_enum_case_is_normalised(tickers):
    res = validate_response(signal_json(sig(direction="Bullish", time_sensitivity="HOURS")), tickers)
    assert res.signals[0].direction == "bullish" and res.signals[0].time_sensitivity == "hours"


def test_missing_fields_rejected(tickers):
    item = sig()
    del item["direction"]
    assert validate_response(signal_json(item), tickers).status == "rejected"


def test_partial_acceptance_keeps_good_signals(tickers):
    res = validate_response(signal_json(sig("NVDA"), sig("FAKE"), sig("AAPL", confidence=200)), tickers)
    assert res.status == "ok"
    assert [s.ticker for s in res.signals] == ["NVDA"]
    assert len(res.problems) == 2


def test_duplicate_tickers_in_one_response(tickers):
    res = validate_response(signal_json(sig("NVDA", confidence=70), sig("NVDA", confidence=90)), tickers)
    assert len(res.signals) == 1 and res.signals[0].confidence == 70
    assert "duplicate" in res.problems[0]


def test_too_many_signals_keeps_highest_confidence(tickers):
    res = validate_response(signal_json(sig("NVDA", confidence=50), sig("AAPL", confidence=90),
                                        sig("TSLA", confidence=70), sig("JPM", confidence=80)), tickers, max_signals=3)
    assert [s.ticker for s in res.signals] == ["AAPL", "JPM", "TSLA"]
    assert any("more than 3" in p for p in res.problems)


def test_long_text_is_truncated(tickers):
    res = validate_response(signal_json(sig(reasoning="x" * 5000)), tickers)
    assert len(res.signals[0].reasoning) == 400


def test_no_ticker_table_means_reject(tmp_path):
    from newstrader.ai.tickers import TickerTable

    empty = TickerTable(Database(tmp_path / "e.db"))
    res = validate_response(signal_json(sig()), empty)
    assert res.status == "rejected" and "Ticker table" in res.fatal


def test_extra_keys_ignored(tickers):
    item = sig()
    item["evil"] = "ignore previous instructions"
    res = validate_response(json.dumps({"signals": [item], "note": "x"}), tickers)
    assert res.status == "ok"
