"""v0.4 scoreboard: win rate by AI engine and by kind of news, small samples marked, and the sealed holdout set."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from newstrader.db import iso
from newstrader.performance.stats import MIN_COUNT, NO_EVENT, compute_stats

ROOT = Path(__file__).resolve().parent.parent


def _row(engine, event, ret, conf=85, direction="bullish"):
    return {"direction": direction, "confidence": conf, "ret_1h": ret, "source_name": "CNBC", "source_type": "rss",
            "traded": 0, "engine": engine, "event": event}


def test_by_engine_and_by_event():
    rows = [_row("local", "Raised its forecast", 1.0) for _ in range(9)]
    rows += [_row("local", "Raised its forecast", -0.5), _row("local", "Raised its forecast", None)]
    rows += [_row("claude", None, 0.4), _row("claude", "", -0.2), _row("local", "Cut its forecast", 0.8, direction="bearish")]
    rows += [_row("local", "Buyout", 2.0, direction="neutral")]  # neutral calls aren't scored
    s = compute_stats(rows, "1h")
    engines = {g["key"]: g for g in s["by_engine"]}
    assert set(engines) == {"Local machine learning", "Claude"}
    loc = engines["Local machine learning"]
    assert loc["count"] == 11 and loc["wins"] == 10 and loc["win_rate"] == 90.9 and not loc["few"]
    assert engines["Claude"]["count"] == 2 and engines["Claude"]["win_rate"] == 50 and engines["Claude"]["few"]
    events = s["by_event"]
    assert [g["key"] for g in events] == ["Raised its forecast", "Cut its forecast", NO_EVENT]  # "none" goes last
    assert events[0]["count"] == 10 and events[0]["win_rate"] == 90 and not events[0]["few"]
    assert events[1]["avg_return"] == 0.8 and events[1]["few"]
    assert events[2]["count"] == 2  # no event and an empty one are the same group
    # the same small-sample mark everywhere
    assert all(g["few"] == (g["count"] < MIN_COUNT) for key in ("by_source", "by_confidence", "by_direction",
                                                                 "by_traded", "by_source_type") for g in s[key])
    assert compute_stats([], "1h")["by_engine"] == [] and compute_stats([], "1h")["overall"]["few"]


def test_old_signals_without_an_engine_are_still_counted():
    s = compute_stats([_row(None, None, 1.0), _row("someday-engine", None, 1.0)], "1h")
    assert {g["key"] for g in s["by_engine"]} == {"?", "someday-engine"}


async def test_performance_api_returns_the_new_tables(client, ctx):
    for engine, event, ret in [("local", "Beat earnings", 1.0), ("claude", None, -1.0)]:
        sid = ctx.db.insert("signals", {"created_at": iso(), "ticker": "NVDA", "direction": "bullish",
                                        "confidence": 90, "source_name": "CNBC", "source_type": "rss",
                                        "engine": engine, "event": event})
        ctx.db.insert("signal_prices", {"signal_id": sid, "ticker": "NVDA", "direction": "bullish", "t0": iso(),
                                        "price_t0": 100, "ret_1h": ret, "status": "pending"})
    r = client.get("/api/performance?horizon=1h&days=30").json()
    assert {g["key"]: g["win_rate"] for g in r["by_engine"]} == {"Local machine learning": 100, "Claude": 0}
    assert {g["key"] for g in r["by_event"]} == {"Beat earnings", NO_EVENT}
    assert all(g["few"] for g in r["by_engine"])


# ---------------------------------------------------------------- the holdout set stays sealed
def _eval_module():
    spec = importlib.util.spec_from_file_location("eval_detection", ROOT / "scripts" / "eval_detection.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FAKE_RESULT = {
    "items": 3, "companies_found_pct": 100.0, "wrong_company_hits": 0, "unlabelled_extra_hits": 0,
    "accuracy_pct": 66.7, "good_bad_news_caught_pct": 50.0, "neutral_news_flagged_pct": 0.0, "opposite_direction": 1,
    "auto_trades_right": 1, "auto_trades_wrong": 1, "by_category": {"earnings": 50, "deals": 100},
    "confusion": {"bullish->bearish": 1}, "mistakes": [{"id": "h7", "ticker": "SECRETCO", "label": "bullish",
                                                       "pred": "bearish", "conf": 88, "found": True,
                                                       "title": "Secret headline nobody may tune rules on"}],
}


def _run_main(ev, monkeypatch, capsys, args):
    monkeypatch.setattr(ev, "load_items", lambda which: [{"id": "h1"}, {"id": "h2"}, {"id": "h7"}])
    monkeypatch.setattr(ev, "make_table", lambda path: None)
    monkeypatch.setattr(ev, "make_engine", lambda table, sentiment, rules: None)
    monkeypatch.setattr(ev, "evaluate", lambda items, engine, table: json.loads(json.dumps(FAKE_RESULT)))
    assert ev.main(args) == 0
    return capsys.readouterr().out


def test_holdout_shows_only_overall_numbers(monkeypatch, capsys, tmp_path):
    ev = _eval_module()
    out_json = tmp_path / "out.json"
    out = _run_main(ev, monkeypatch, capsys, ["--set", "holdout", "--compare", "--mistakes", "15",
                                              "--json", str(out_json)])
    assert "66.7" in out and "Called the opposite direction | 1 | 1" in out
    assert "SECRETCO" not in out and "Secret headline" not in out and "earnings" not in out
    assert "Companies found" not in out and "Would auto-trade" not in out
    saved = json.loads(out_json.read_text(encoding="utf-8"))
    assert saved == {"v0.2 (wording only)": {"accuracy_pct": 66.7, "opposite_direction": 1},
                     "with event rules": {"accuracy_pct": 66.7, "opposite_direction": 1}}
    # a holdout file given by path is sealed too
    out = _run_main(ev, monkeypatch, capsys, ["--set", str(tmp_path / "holdout.jsonl"), "--mistakes", "5"])
    assert "SECRETCO" not in out and "66.7" in out


def test_unseal_and_other_sets_show_everything(monkeypatch, capsys):
    ev = _eval_module()
    out = _run_main(ev, monkeypatch, capsys, ["--set", "holdout", "--mistakes", "5", "--unseal"])
    assert "SECRETCO" in out and "earnings 50" in out and "Would auto-trade: wrong" in out
    out = _run_main(ev, monkeypatch, capsys, ["--set", "set2", "--mistakes", "5"])
    assert "SECRETCO" in out and "earnings 50" in out
    assert ev.is_sealed("holdout") and ev.is_sealed("tests/fixtures/eval/holdout.jsonl")
    assert not ev.is_sealed("holdout", unseal=True) and not ev.is_sealed("set3")
