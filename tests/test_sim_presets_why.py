"""Built-in simulator screens (where trades go, reset), trading styles (presets) and the "Why did it trade?" card."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from newstrader.ai.claude_client import ClaudeAnalyzer
from newstrader.ai.pipeline import Pipeline
from newstrader.config import AppSettings, ConfigStore
from newstrader.db import iso
from newstrader.trading import live_guard, presets
from newstrader.trading import trail as trails
from newstrader.trading.fake_broker import FakeBroker
from newstrader.trading.sim_broker import SimBroker
from newstrader.trading.trader import SIM_BOOK, Trader

from .helpers import FakeClaude, make_tickers, sig, signal_json
from .test_pipeline import item, run
from .test_sim_broker import OPEN_DAY, Clock, FakeData


class FakeAlpaca:
    """Stands in for trading/broker.Broker (no network): remembers the credentials it was made with."""
    kind = "alpaca"
    mode = "paper"

    def __init__(self, creds, state, data_feed="iex"):
        self.creds = creds
        self.mode = "paper" if creds.paper else "live"


@pytest.fixture
def fake_alpaca(monkeypatch):
    monkeypatch.setattr("newstrader.trading.broker.Broker", FakeAlpaca)
    return FakeAlpaca


def paper_keys(ctx):
    ctx.keys.update({"ALPACA_PAPER_API_KEY": "PKTEST", "ALPACA_PAPER_SECRET_KEY": "secret"})


# ================================================================================ where trades go
def test_default_is_automatic(ctx):
    assert AppSettings().trading.broker == "auto"
    assert ctx.config.settings.trading.broker == "auto"


def test_automatic_uses_the_simulator_without_keys_and_alpaca_with_them(ctx, fake_alpaca):
    t = Trader(ctx)
    broker = t._default_factory()
    assert isinstance(broker, SimBroker) and broker.mode == "paper" and broker.kind == "simulator"
    paper_keys(ctx)
    broker = t._default_factory()
    assert isinstance(broker, FakeAlpaca) and broker.creds.paper


def test_explicit_choices(ctx, fake_alpaca):
    t = Trader(ctx)
    ctx.config.update({"trading": {"broker": "alpaca"}})
    assert t._default_factory() is None  # Alpaca picked, no keys: not connected (as before)
    paper_keys(ctx)
    ctx.config.update({"trading": {"broker": "simulator"}})
    assert isinstance(t._default_factory(), SimBroker)  # keys or not, the simulator when you pick it


def test_live_mode_always_uses_alpaca(ctx, fake_alpaca):
    paper_keys(ctx)
    ctx.keys.update({"ALPACA_LIVE_API_KEY": "AKLIVE", "ALPACA_LIVE_SECRET_KEY": "live-secret"})
    ctx.config.update({"trading": {"broker": "simulator"}})
    live_guard.arm_live_trading(ctx.state, ctx.keys.keys, enable=True, phrase=live_guard.CONFIRM_PHRASE)
    try:
        broker = Trader(ctx)._default_factory()
        assert isinstance(broker, FakeAlpaca) and broker.creds.paper is False and broker.mode == "live"
    finally:
        live_guard.disarm_live_trading(ctx.state)


def _old_config(tmp_path, broker="alpaca") -> ConfigStore:
    data = AppSettings().model_dump(mode="json")
    data["schema_version"] = 5
    data["trading"]["broker"] = broker
    data["trading"]["buy_threshold"] = 85
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return ConfigStore(path)


def test_migration_keeps_key_holders_on_alpaca(tmp_path, ctx, fake_alpaca):
    store = _old_config(tmp_path)
    assert store.settings.trading.broker == "auto" and store.settings.trading.buy_threshold == 85
    saved = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert saved["schema_version"] == 6 and saved["trading"]["broker"] == "auto"
    ctx.config = store
    paper_keys(ctx)
    broker = Trader(ctx)._default_factory()
    assert isinstance(broker, FakeAlpaca) and broker.creds.paper  # keys present -> unchanged: the Alpaca paper account


def test_migration_without_keys_gives_the_simulator(tmp_path, ctx, fake_alpaca):
    ctx.config = _old_config(tmp_path)
    assert isinstance(Trader(ctx)._default_factory(), SimBroker)


def test_new_configs_keep_an_explicit_choice(tmp_path):
    data = AppSettings().model_dump(mode="json")
    data["trading"]["broker"] = "alpaca"  # chosen in Settings (schema 6)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert ConfigStore(path).settings.trading.broker == "alpaca"


async def test_switching_where_trades_go_reconnects_and_tells_the_services(ctx):
    calls = []
    brokers = {"auto": FakeBroker(), "simulator": FakeBroker(equity=50_000)}
    t = Trader(ctx, broker_factory=lambda: brokers[ctx.config.settings.trading.broker])
    ctx.services["trader"] = t

    class Svc:
        name = "svc"

        async def on_broker_changed(self):
            calls.append(t.broker)

    ctx.services["svc"] = Svc()
    await t.connect()
    ctx.config.update({"trading": {"broker": "simulator"}})
    await t.switch_broker()
    assert t.broker is brokers["simulator"] and t.account["equity"] == 50_000 and calls == [brokers["simulator"]]
    await t.stop()


async def test_simulator_orders_are_their_own_book(ctx):
    sim = FakeBroker()
    sim.kind = "simulator"
    t = Trader(ctx, broker_factory=lambda: sim)
    ctx.services["trader"] = t
    await t.connect()
    res = await t.handle_signal({"id": 1, "ticker": "AAPL", "direction": "bullish", "confidence": 90})
    assert res["traded"]
    assert {r["mode"] for r in ctx.db.query("SELECT mode FROM orders")} == {SIM_BOOK}
    sim.prices["AAPL"] = 240.0
    await t.manual_close("AAPL")
    closes = [r for r in t.trade_log() if r["intent"] == "close_long"]
    assert closes and closes[0]["realized_pl"] == pytest.approx(40.0) and closes[0]["mode"] == SIM_BOOK
    assert t.summary()["broker_kind"] == "simulator"
    await t.stop()


# ================================================================================ simulator reset
@pytest.fixture
def sim_trader(ctx, tmp_path):
    data = FakeData()
    data.prices["NVDA"] = 200.0
    sim = SimBroker(ctx.db, data=data, starting_cash=100_000, slippage_pct=0.05, clock=Clock(OPEN_DAY))
    t = Trader(ctx, broker_factory=lambda: sim)
    t.broker = sim
    ctx.services["trader"] = t
    return t, sim


def test_reset_needs_the_typed_word(client, ctx, sim_trader):
    t, sim = sim_trader
    sim.submit_bracket("NVDA", "buy", 10, 210, 190, "nt-1-a")
    ctx.db.insert("orders", {"alpaca_order_id": "sim-old", "symbol": "NVDA", "side": "buy", "intent": "open_long",
                             "qty": 10, "status": "filled", "filled_qty": 10, "filled_avg_price": 200.1,
                             "filled_at": iso(), "mode": SIM_BOOK})
    ctx.db.insert("orders", {"alpaca_order_id": "sim-old-sl", "symbol": "NVDA", "side": "sell", "intent": "stop_loss",
                             "qty": 10, "status": "held", "mode": SIM_BOOK, "parent_alpaca_id": "sim-old"})
    for body in ({}, {"phrase": "reset please"}, {"phrase": "yes"}):
        r = client.post("/api/simulator/reset", json=body)
        assert r.status_code == 403 and "RESET" in r.json()["detail"]
    assert "NVDA" in sim.state["positions"]  # nothing changed

    r = client.post("/api/simulator/reset", json={"phrase": "RESET", "starting_cash": 25000})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["in_use"] == "simulator" and body["simulator"]["starting_cash"] == 25000
    assert sim.state["positions"] == {} and sim.state["cash"] == 25000 and sim.state["orders"] == []
    assert ctx.config.settings.trading.sim_starting_cash == 25000
    old = ctx.db.query_one("SELECT mode FROM orders WHERE alpaca_order_id = 'sim-old'")
    assert old["mode"].startswith(f"{SIM_BOOK} (reset ")  # kept in the Trade Log, out of the new account's P/L
    leg = ctx.db.query_one("SELECT status FROM orders WHERE alpaca_order_id = 'sim-old-sl'")
    assert leg["status"] == "canceled"  # its stop-loss went with the old account


def test_reset_rejects_bad_starting_cash(client, sim_trader):
    r = client.post("/api/simulator/reset", json={"phrase": "RESET", "starting_cash": 5})
    assert r.status_code == 400


def test_reset_works_when_the_simulator_isnt_in_use(client, ctx):
    t = Trader(ctx, broker_factory=FakeBroker)
    t.broker = FakeBroker()
    ctx.services["trader"] = t
    ctx.db.execute("INSERT INTO sim_state (key, value, updated_at) VALUES ('account', ?, ?)",
                   (json.dumps({"account_number": "SIM-OLD", "cash": 1.0}), iso()))
    r = client.post("/api/simulator/reset", json={"phrase": "RESET"})
    assert r.status_code == 200 and r.json()["in_use"] == "alpaca"
    state = json.loads(ctx.db.query_one("SELECT value FROM sim_state")["value"])
    assert state["cash"] == 100_000 and state["account_number"] != "SIM-OLD"


def test_simulator_status_lists_what_needs_alpaca(client, sim_trader):
    body = client.get("/api/simulator").json()
    assert body["setting"] == "auto" and body["in_use"] == "simulator" and body["reset_phrase"] == "RESET"
    text = " ".join(body["needs_alpaca"]).lower()
    for word in ("benzinga", "history", "price model", "backtest", "crypto"):
        assert word in text
    meta = client.get("/api/settings").json()["meta"]
    assert meta["account"]["in_use"] == "simulator" and meta["style"]["current"] == "normal"


def test_training_and_backtests_say_they_need_alpaca(ctx, sim_trader):
    from newstrader.backtest.runner import BacktestParams, BacktestRunner
    from newstrader.ml.trainer import ModelTrainer, TrainParams

    with pytest.raises(RuntimeError, match="needs? a free Alpaca account"):
        ModelTrainer(ctx).launch(TrainParams(start=datetime(2026, 1, 1).date(), end=datetime(2026, 6, 1).date()))
    with pytest.raises(RuntimeError, match="need a free Alpaca account"):
        BacktestRunner(ctx).launch(BacktestParams(start=datetime(2026, 1, 1).date(), end=datetime(2026, 1, 2).date()))


# ================================================================================ trading styles
def test_normal_is_the_defaults():
    s = AppSettings()
    assert presets.detect(s) == "normal"
    for section, key, _label, _fmt in presets.FIELDS:
        assert presets.PRESETS["normal"]["values"][section][key] == getattr(getattr(s, section), key)


def test_presets_values():
    c = presets.PRESETS["cautious"]["values"]
    assert (c["trading"]["buy_threshold"], c["trading"]["review_threshold"], c["trading"]["stop_loss_pct"],
            c["trading"]["take_profit_pct"]) == (88, 70, 1.5, 3.0)
    assert (c["risk"]["max_dollars_per_trade"], c["risk"]["max_pct_per_stock"], c["risk"]["max_open_positions"],
            c["risk"]["daily_loss_limit_usd"], c["risk"]["ticker_cooldown_minutes"]) == (500, 5, 3, 200, 60)
    assert c["chart"]["confirm"] == "strict"
    a = presets.PRESETS["aggressive"]["values"]
    assert (a["trading"]["buy_threshold"], a["trading"]["review_threshold"], a["trading"]["stop_loss_pct"],
            a["trading"]["take_profit_pct"]) == (72, 55, 3.0, 6.0)
    assert (a["risk"]["max_dollars_per_trade"], a["risk"]["max_pct_per_stock"], a["risk"]["max_open_positions"],
            a["risk"]["daily_loss_limit_usd"], a["risk"]["ticker_cooldown_minutes"]) == (2000, 15, 8, 1000, 15)
    assert a["chart"]["confirm"] == "soft"
    assert "not more profit" in presets.PRESETS["aggressive"]["description"]


def test_apply_a_style_and_custom_detection(client, ctx):
    r = client.post("/api/presets/apply", json={"style": "cautious"})
    assert r.status_code == 200, r.text
    s = ctx.config.settings
    assert (s.trading.buy_threshold, s.risk.max_open_positions, s.chart.confirm) == (88, 3, "strict")
    assert r.json()["meta"]["style"]["current"] == "cautious"
    assert client.get("/api/presets").json()["current"] == "cautious"
    ctx.config.update({"trading": {"buy_threshold": 87}})  # edited by hand
    assert presets.detect(ctx.config.settings) == "custom"
    assert client.get("/api/presets").json()["current"] == "custom"
    ctx.config.update({"risk": {"blacklist": "GME"}})  # a setting no style sets doesn't matter
    client.post("/api/presets/apply", json={"style": "normal"})
    assert presets.detect(ctx.config.settings) == "normal"
    assert client.post("/api/presets/apply", json={"style": "yolo"}).status_code == 400


def test_styles_never_turn_shorting_on_or_touch_live(client, ctx):
    ctx.keys.update({"ALPACA_PAPER_API_KEY": "PK", "ALPACA_PAPER_SECRET_KEY": "s", "ALPACA_LIVE_API_KEY": "AK",
                     "ALPACA_LIVE_SECRET_KEY": "ls"})
    for name in presets.PRESETS:
        assert presets.PRESETS[name]["values"]["trading"]["allow_shorting"] is False
    ctx.config.update({"trading": {"allow_shorting": True}})
    assert presets.detect(ctx.config.settings) == "custom"
    for name in presets.PRESETS:
        client.post("/api/presets/apply", json={"style": name})
        assert ctx.config.settings.trading.allow_shorting is False and ctx.state.mode == "paper"
    live_guard.arm_live_trading(ctx.state, ctx.keys.keys, enable=True, phrase=live_guard.CONFIRM_PHRASE)
    try:
        client.post("/api/presets/apply", json={"style": "aggressive"})
        assert ctx.state.mode == "live"  # a style never changes live/paper
    finally:
        live_guard.disarm_live_trading(ctx.state)
    assert "live" not in json.dumps(presets.PRESETS).lower()


def test_style_table_rows(ctx):
    d = presets.describe(ctx.config.settings)
    labels = [r["label"] for r in d["rows"]]
    assert "Buy threshold" in labels and "Short selling" in labels
    short = next(r for r in d["rows"] if r["label"] == "Short selling")
    assert set(short["values"].values()) == {"Off"}
    buy = next(r for r in d["rows"] if r["label"] == "Buy threshold")
    assert buy["values"] == {"cautious": "88", "normal": "80", "aggressive": "72"} and buy["now"] == "80"


# ================================================================================ decision trail
@pytest.fixture
async def env(ctx):
    ctx.config.update({"ai": {"engine": "claude"}, "chart": {"confirm": "off"}})
    broker = FakeBroker(prices={"NVDA": 180.0, "AAPL": 230.0, "TSLA": 250.0, "JPM": 300.0})
    trader = Trader(ctx, broker_factory=lambda: broker)
    ctx.services["trader"] = trader
    await trader.connect()
    fake = FakeClaude()
    pipeline = Pipeline(ctx, analyzer=ClaudeAnalyzer(ctx, client_factory=lambda: fake), tickers=make_tickers(ctx.db))
    ctx.services["pipeline"] = pipeline
    yield ctx, pipeline, fake, broker
    await trader.stop()


def trail_of(ctx, signal_id) -> dict:
    return json.loads(ctx.db.query_one("SELECT decision_trail FROM signals WHERE id = ?", (signal_id,))["decision_trail"])


def check(trail: dict, name: str) -> dict:
    return next(c for c in trail["checks"] if c["name"] == name)


async def test_trail_on_a_buy(env):
    ctx, pipeline, fake, broker = env
    fake.responses = [signal_json(sig("NVDA", confidence=90))]
    out = await run(pipeline, item())
    s = out["signals"][0]
    assert s["action"] == "bought"
    t = trail_of(ctx, s["id"])
    assert t["version"] == 1 and t["account"]["broker"] == "alpaca"
    assert t["style"] == "custom"  # Normal, but with the chart check turned off (this test's setting)
    assert t["settings"]["buy_threshold"] == 80 and t["settings"]["review_threshold"] == 60
    assert t["confidence"][0] == {"label": "From the AI engine", "value": 90}
    assert check(t, "Thresholds")["result"] == "pass"
    risk = check(t, "Risk checks")
    assert risk["result"] == "pass" and "Kill switch off" in risk["passed"] and "BUY" in risk["detail"]
    assert check(t, "Order")["result"] == "pass"
    assert t["chart"]["verdict"] == "skipped" and "off" in t["chart"]["reason"]
    assert t["outcome"] == {"action": "bought", "reason": s["action_reason"], "traded": True}
    assert t["engine"]["name"] == "claude" and t["engine"]["confidence"] == 90


async def test_trail_on_review_and_approval(env):
    ctx, pipeline, fake, broker = env
    fake.responses = [signal_json(sig("NVDA", confidence=70))]
    s = (await run(pipeline, item()))["signals"][0]
    assert s["action"] == "review"
    t = trail_of(ctx, s["id"])
    assert check(t, "Thresholds")["result"] == "review" and t["outcome"]["action"] == "review"
    assert not any(c["name"] == "Risk checks" for c in t["checks"])  # never got that far
    await pipeline.approve(s["id"])
    t = trail_of(ctx, s["id"])
    assert t["manual"] is True and t["outcome"]["action"] == "bought"
    assert check(t, "Approved by you")["result"] == "pass"
    assert t["earlier"][0]["outcome"]["action"] == "review"
    assert t["engine"]["name"] == "claude"  # carried over from the first decision


async def test_trail_when_blocked(env):
    ctx, pipeline, fake, broker = env
    ctx.config.update({"risk": {"blacklist": ["NVDA"]}})
    fake.responses = [signal_json(sig("NVDA", confidence=92))]
    s = (await run(pipeline, item()))["signals"][0]
    assert s["action"] == "blocked"
    t = trail_of(ctx, s["id"])
    risk = check(t, "Risk checks")
    assert risk["result"] == "stop" and "blacklist" in risk["detail"] and "passed" not in risk
    assert t["outcome"]["action"] == "blocked" and not t["outcome"]["traded"]


async def test_trail_when_chasing_goes_to_review(env, monkeypatch):
    ctx, pipeline, fake, broker = env
    trader = ctx.services["trader"]

    async def moved(*_a, **_k):
        return 5.0

    monkeypatch.setattr(trader, "_pre_move", moved)
    fake.responses = [signal_json(sig("NVDA", confidence=90))]
    s = (await run(pipeline, item()))["signals"][0]
    assert s["action"] == "review"
    t = trail_of(ctx, s["id"])
    assert check(t, "Risk checks")["result"] == "pass" and check(t, "Don't chase")["result"] == "review"


async def test_trail_records_the_chart_change(ctx):
    from newstrader.chart.service import ChartCheck

    class Charts:
        def assess(self, signal, holding, reading):
            return ChartCheck("agrees", "Uptrend, above VWAP.", 0.6, 4, int(signal["confidence"]) + 4)

        def wanted(self, signal, holding):
            return True

        async def wait_reading(self, symbol):
            return None

    ctx.services["chart"] = Charts()
    broker = FakeBroker()
    t = Trader(ctx, broker_factory=lambda: broker)
    await t.connect()
    res = await t.handle_signal({"id": None, "ticker": "AAPL", "direction": "bullish", "confidence": 82})
    trail = res["trail"]
    assert trail["chart"]["verdict"] == "agrees" and trail["chart"]["adjust"] == 4
    assert [c["value"] for c in trail["confidence"]] == [82, 86] and trail["confidence"][1]["change"] == 4
    await t.stop()


def test_engine_part_reads_the_local_engines_details():
    raw = json.dumps({"signals": [], "details": [{"ticker": "NVDA", "sentiment": {"positive": 0.9, "negative": 0.05,
                                                                                    "neutral": 0.05},
                                                  "p_up": 0.63, "event": "Won a big contract", "relevance": 1.0}]})
    d = trails.local_details(raw)["NVDA"]
    part = trails.engine_part({"engine": "local", "direction": "bullish", "confidence": 81, "flags": '["unconfirmed"]'},
                              d, "local: FinBERT")
    assert part["wording"]["positive"] == 0.9 and part["price_model"] == 0.63 and part["flags"] == ["unconfirmed"]
    assert part["event"] == "Won a big contract"
    assert trails.local_details("not json") == {}


# ================================================================================ the "Why did it trade?" card
def test_why_card_with_missing_data(client, ctx):
    sid = ctx.db.insert("signals", {"created_at": iso(), "ticker": "AAPL", "direction": "bullish", "confidence": 70,
                                    "source_name": "Old feed", "headline": "Apple does something", "action": "review",
                                    "action_reason": "In the review band.", "engine": "claude",
                                    "reasoning": "Looks good."})
    r = client.get(f"/api/signals/{sid}/why")
    assert r.status_code == 200, r.text
    card = r.json()
    for key in ("signal", "news", "engine", "trail", "trail_note", "speed", "trade", "chart", "result"):
        assert key in card
    assert card["trail"] is None and "before NewsTrader saved" in card["trail_note"]
    assert card["news"]["headline"] == "Apple does something" and card["news"]["source"] == "Old feed"
    assert card["engine"]["label"] == "Claude" and "Looks good." in card["engine"]["lines"]
    assert card["trade"] is None and card["chart"]["t"] == [] and card["chart"]["error"]
    assert card["result"]["checkpoints"] == [] and card["result"]["note"]
    assert "decision_trail" not in card["signal"]
    assert client.get("/api/signals/99999/why").status_code == 404
    assert client.get("/api/trades/99999/why").status_code == 404


def test_why_card_for_an_order_placed_by_hand(client, ctx):
    oid = ctx.db.insert("orders", {"alpaca_order_id": "x1", "symbol": "AAPL", "side": "sell", "intent": "close_long",
                                   "qty": 3, "status": "filled", "filled_qty": 3, "filled_avg_price": 231.0,
                                   "filled_at": iso(), "mode": "paper", "reason": "Closed manually"})
    card = client.get(f"/api/trades/{oid}/why").json()
    assert card["signal"] is None and card["order"]["symbol"] == "AAPL" and "wasn't placed for a signal" in card["note"]


def _bars(start: datetime, minutes: int, p0: float, step: float) -> list[dict]:
    return [{"t": iso(start + timedelta(minutes=i)).replace(".000", ""), "o": p0 + i * step, "h": p0 + i * step,
             "l": p0 + i * step, "c": round(p0 + i * step, 4), "v": 1000} for i in range(minutes)]


def test_why_card_for_a_trade(client, ctx):
    now = datetime.now(UTC).replace(second=0, microsecond=0)
    t0 = now - timedelta(hours=4)
    broker = FakeBroker()
    broker.bar_data = {"NVDA": _bars(t0 - timedelta(minutes=40), 200, 180.0, 0.05),
                       "SPY": _bars(t0 - timedelta(minutes=40), 200, 600.0, 0.01)}
    t = Trader(ctx, broker_factory=lambda: broker)
    t.broker = broker
    ctx.services["trader"] = t
    trail = {"version": 1, "style": "normal", "checks": [{"name": "Risk checks", "result": "pass"}],
             "outcome": {"action": "bought", "traded": True}, "confidence": [{"label": "From the AI engine",
                                                                               "value": 88}]}
    sid = ctx.db.insert("signals", {"created_at": iso(t0 + timedelta(seconds=20)), "ticker": "NVDA",
                                    "direction": "bullish", "confidence": 88, "source_name": "CNBC",
                                    "headline": "Nvidia wins a contract", "news_published_at": iso(t0),
                                    "news_received_at": iso(t0 + timedelta(seconds=10)),
                                    "decided_at": iso(t0 + timedelta(seconds=15)), "action": "bought", "traded": 1,
                                    "engine": "claude", "decision_trail": json.dumps(trail)})
    entry = ctx.db.insert("orders", {"signal_id": sid, "alpaca_order_id": "o1", "symbol": "NVDA", "side": "buy",
                                     "intent": "open_long", "qty": 5, "status": "filled", "filled_qty": 5,
                                     "filled_avg_price": 180.1, "submitted_at": iso(t0 + timedelta(seconds=17)),
                                     "filled_at": iso(t0 + timedelta(minutes=1)), "stop_price": 176.5,
                                     "take_profit_price": 183.0, "mode": "paper"})
    ctx.db.update("signals", sid, {"order_id": entry})
    leg = ctx.db.insert("orders", {"signal_id": sid, "alpaca_order_id": "o1-tp", "symbol": "NVDA", "side": "sell",
                                   "intent": "take_profit", "qty": 5, "status": "filled", "filled_qty": 5,
                                   "filled_avg_price": 183.0, "filled_at": iso(t0 + timedelta(minutes=58)),
                                   "parent_alpaca_id": "o1", "take_profit_price": 183.0, "mode": "paper"})
    ctx.db.insert("signal_prices", {"signal_id": sid, "ticker": "NVDA", "direction": "bullish", "t0": iso(t0),
                                    "price_t0": 180.0, "due_5m": iso(t0 + timedelta(minutes=5)),
                                    "due_1h": iso(t0 + timedelta(hours=1)), "due_1d": iso(now + timedelta(hours=20)),
                                    "price_5m": 180.25, "price_1h": 183.0, "ret_5m": 0.139, "ret_1h": 1.667,
                                    "status": "pending"})
    for path in (f"/api/trades/{leg}/why", f"/api/trades/{entry}/why", f"/api/signals/{sid}/why"):
        card = client.get(path).json()
        assert card["signal"]["id"] == sid and card["trail"]["style"] == "normal" and not card["trail_note"]
        tr = card["trade"]
        assert tr["entry_price"] == 180.1 and tr["stop"] == 176.5 and tr["target"] == 183.0
        assert tr["exit"]["how"] == "Take-profit hit" and tr["realized_pl"] == pytest.approx(14.5)
        assert tr["return_pct"] == pytest.approx(1.61, abs=0.01)
        assert card["speed"]["news_to_order_s"] == pytest.approx(7.0)
        chart = card["chart"]
        assert chart["t"] and len(chart["t"]) == len(chart["close"]) and not chart["error"]
        assert {m["kind"] for m in chart["markers"]} == {"news", "entry", "exit"}
        cps = {c["key"]: c for c in card["result"]["checkpoints"]}
        assert cps["1h"]["stock_pct"] == pytest.approx(1.67) and cps["1h"]["spy_pct"] == pytest.approx(0.1, abs=0.01)
        assert cps["1h"]["vs_spy"] == pytest.approx(1.57, abs=0.02) and cps["1d"]["spy_pct"] is None
        assert card["result"]["trade"]["spy_pct"] is not None and card["result"]["trade"]["vs_spy"] > 0
