"""Crypto (paper) through Alpaca: symbols, settings, the pre-filter, the validator, the risk rules, the trader, the
exit manager (safety stop at Alpaca + take-profit watched by the app), the local engine's crypto rules, charts, the
market monitor, the scoreboard, migrations and the API. Never talks to Alpaca (FakeBroker)."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from newstrader.ai.pipeline import Pipeline
from newstrader.ai.prefilter import prefilter
from newstrader.ai.tickers import TickerTable
from newstrader.ai.validator import validate_response
from newstrader.chart.service import ChartService, history_start, price_series
from newstrader.config import AppSettings, ConfigStore, CryptoSettings, RiskSettings, TradingSettings
from newstrader.db import MIGRATIONS, Database, iso
from newstrader.market.monitor import MarketMonitor
from newstrader.ml.engine import LocalMLEngine
from newstrader.ml.events import crypto_reading
from newstrader.performance.tracker import PerformanceTracker
from newstrader.sources.alpaca_news import news_to_item
from newstrader.trading import crypto
from newstrader.trading.broker import order_to_dict, position_to_dict
from newstrader.trading.crypto_exits import CryptoExitManager, open_plans
from newstrader.trading.fake_broker import FakeBroker
from newstrader.trading.risk import (
    AccountSnapshot,
    AssetInfo,
    PositionInfo,
    RiskContext,
    check_crypto_entry,
    check_exit,
)
from newstrader.trading.trader import NO_CRYPTO_SHORT, Trader
from tests.helpers import ASSETS, signal_json
from tests.helpers import sig as ai_sig
from tests.test_market import T0, make_bars, spike_bars
from tests.test_ml import fake_service, news, run

PRICES = {"AAPL": 230.0, "BTC/USD": 60000.0, "ETH/USD": 3000.0, "SOL/USD": 150.0}
EXTRA = [
    {"symbol": "MARA", "name": "MARA Holdings, Inc. Common Stock", "exchange": "NASDAQ", "tradable": True},
    {"symbol": "MSTR", "name": "Strategy Inc Class A Common Stock", "exchange": "NASDAQ", "tradable": True},
    {"symbol": "COIN", "name": "Coinbase Global, Inc. Class A Common Stock", "exchange": "NASDAQ", "tradable": True},
    {"symbol": "BTC", "name": "Grayscale Bitcoin Mini Trust ETF", "exchange": "ARCA", "tradable": True},
    {"symbol": "SOL", "name": "Emeren Group Ltd American Depositary Shares", "exchange": "NYSE", "tradable": True},
]
COINS = [{"symbol": s, "name": f"{crypto.display_name(s)} / US Dollar", "tradable": True}
         for s in ("BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD")]


class FakeAlerts:
    def __init__(self):
        self.sent: list[dict] = []

    async def send(self, kind, title, message="", level="info", force=False, fields=None):
        self.sent.append({"kind": kind, "title": title, "message": message, "level": level})
        return {"toast": True}


def crypto_table(db, wanted=("BTC/USD", "ETH/USD", "SOL/USD")) -> TickerTable:
    t = TickerTable(db)
    t.replace_all(ASSETS + EXTRA)
    t.replace_crypto(COINS)
    t.crypto_wanted = list(wanted)
    return t


def sig(ticker="BTC/USD", direction="bullish", confidence=90, **kw):
    return {"id": kw.pop("id", 1), "ticker": ticker, "direction": direction, "confidence": confidence,
            "reasoning": "test", **kw}


def plan_row(ctx, plan_id=None) -> dict:
    sql = "SELECT * FROM crypto_exits" + (" WHERE id = ?" if plan_id else " ORDER BY id DESC LIMIT 1")
    return ctx.db.query_one(sql, (plan_id,) if plan_id else ())


def track(broker) -> list[tuple]:
    """Record cancels and sales in the order they happen."""
    events: list[tuple] = []
    cancel, close = broker.cancel_order, broker.close_position

    def cancel_order(order_id):
        events.append(("cancel", order_id))
        return cancel(order_id)

    def close_position(symbol, wait_seconds=0):
        events.append(("close", symbol))
        return close(symbol, wait_seconds)

    broker.cancel_order, broker.close_position = cancel_order, close_position
    return events


# ================================================================================ symbols
@pytest.mark.parametrize("raw,want", [
    ("BTCUSD", "BTC/USD"), ("btc/usd", "BTC/USD"), ("$BTC/USD", "BTC/USD"), ("ETHUSDT", "ETH/USDT"),
    ("SHIBUSD", "SHIB/USD"), ("AAPL", "AAPL"), ("BRK.B", "BRK.B"), ("ETHBTC", "ETHBTC"), ("BTC-USD", "BTC-USD"),
    ("XUSD", "XUSD"),  # 4 letters: could be a stock
])
def test_normalize(raw, want):
    assert crypto.normalize(raw) == want


def test_symbol_helpers():
    assert crypto.normalize("ETHBTC", "crypto") == "ETH/BTC"  # Alpaca says it's crypto
    assert crypto.is_crypto("BTCUSD") and crypto.is_crypto("SOL/USD") and not crypto.is_crypto("AAPL")
    assert crypto.api_symbol("BTC/USD") == "BTCUSD" and crypto.base("ETH/USD") == "ETH"
    assert crypto.display_name("BTC/USD") == "Bitcoin" and crypto.display_name("PEPE/USD", "Pepe / US Dollar") == "Pepe"
    assert crypto.parse_symbols("btc, ETHUSD; sol/usd  BTC") == ["BTC/USD", "ETH/USD", "SOL/USD"]
    for bad in ("ETH/BTC", "BTC-USD", "not a coin"):
        with pytest.raises(ValueError):
            crypto.parse_symbols(bad)
    assert crypto.exit_prices(60000.0, 3.0, 6.0, 0.01) == (58200.0, 57909.0, 63600.0)
    assert crypto.exit_prices(0.123456, 3.0, 6.0) == (0.1197, 0.1191, 0.1309)  # small coins: 4 decimals
    assert crypto.round_down(0.0083333337, 1e-9) == 0.008333333
    assert "bitcoin" in crypto.terms("BTC/USD") and "sol" in crypto.terms("SOL/USD")


def test_alpaca_objects_are_normalised():
    pos = SimpleNamespace(symbol="BTCUSD", asset_class="crypto", qty="0.0099", qty_available="0.0099", side="long",
                          avg_entry_price="60000", current_price="61000", market_value="603.9", cost_basis="594",
                          unrealized_pl="9.9", unrealized_plpc="0.0166", unrealized_intraday_pl="9.9",
                          change_today="0.01", lastday_price="60500")
    d = position_to_dict(pos)
    assert d["symbol"] == "BTC/USD" and d["asset_class"] == "crypto" and d["qty"] == pytest.approx(0.0099)
    order = SimpleNamespace(id="o1", client_order_id="c", symbol="BTC/USD", asset_class="crypto", side="buy",
                            qty=None, notional="500", filled_qty="0.0083", filled_avg_price="60000",
                            order_type="market", type="market", order_class="simple", status="filled",
                            limit_price=None, stop_price=None, time_in_force="gtc", submitted_at=None,
                            created_at=None, filled_at=None, updated_at=None, legs=None)
    o = order_to_dict(order)
    assert o["symbol"] == "BTC/USD" and o["notional"] == 500.0 and o["qty"] is None
    src = SimpleNamespace(id="alpaca-news", name="Alpaca")
    item = news_to_item({"id": 1, "headline": "Bitcoin ETF flows", "symbols": ["BTCUSD", "COIN"]}, src)
    assert item.symbols == ["BTC/USD", "COIN"]


# ================================================================================ settings and migrations
def test_crypto_settings_defaults_and_validation(ctx):
    c = CryptoSettings()
    assert (c.enabled, c.symbols, c.max_per_trade, c.stop_loss_pct, c.take_profit_pct, c.trade_on_news) == \
        (False, ["BTC/USD", "ETH/USD", "SOL/USD"], 500.0, 3.0, 6.0, True)
    assert AppSettings().schema_version == 5
    ctx.config.update({"crypto": {"symbols": "btc, dogeusd"}})  # the Settings page sends a comma list
    assert ctx.config.settings.crypto.symbols == ["BTC/USD", "DOGE/USD"]
    for bad in ({"symbols": "ETH/BTC"}, {"symbols": "AAPL!"}, {"max_per_trade": 0}, {"stop_loss_pct": 0}):
        with pytest.raises(ValidationError):
            ctx.config.update({"crypto": bad})


def test_old_config_gets_crypto_switched_off(tmp_path):
    data = AppSettings().model_dump(mode="json")
    data["schema_version"] = 4
    data["risk"]["max_dollars_per_trade"] = 250
    data["crypto"] = {"enabled": True, "max_per_trade": 9999}  # an older file can't switch crypto on
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    store = ConfigStore(path)
    assert not store.settings.crypto.enabled and store.settings.crypto.max_per_trade == 500
    assert store.settings.risk.max_dollars_per_trade == 250
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["schema_version"] == 5 and saved["crypto"]["enabled"] is False


def test_migration_13_adds_the_exit_plans(tmp_path):
    path = tmp_path / "old.db"
    c = sqlite3.connect(path)
    for i, script in enumerate(MIGRATIONS[:12], start=1):  # a database from before crypto
        c.executescript(script)
        c.execute(f"PRAGMA user_version={i}")
    c.commit()
    c.close()
    db = Database(path)
    try:
        assert db.scalar("PRAGMA user_version") == len(MIGRATIONS) >= 13
        cols = {r["name"] for r in db.query("PRAGMA table_info(crypto_exits)")}
        assert {"symbol", "mode", "entry_order_id", "stop_order_id", "stop_price", "stop_limit_price",
                "take_profit_price", "status", "exit_order_id"} <= cols
    finally:
        db.close()


# ================================================================================ pre-filter, validator
def syms(text, table, **kw):
    return [c.symbol for c in prefilter(text, table, **kw).candidates]


def test_prefilter_finds_coins_by_name_cashtag_and_news_tag(ctx):
    t = crypto_table(ctx.db)
    assert syms("Bitcoin jumps as spot ETF inflows hit a record", t) == ["BTC/USD"]
    assert syms("$BTC breaks out, $ETH follows", t) == ["BTC/USD", "ETH/USD"]  # the coins, not the BTC fund
    assert syms("Grayscale's fund sees inflows", t, source_symbols=["BTCUSD"]) == ["BTC/USD"]
    assert syms("Ether and Ethereum fees fall after the upgrade", t) == ["ETH/USD"]
    assert syms("SOL rallies as crypto traders pile in", t) == ["SOL/USD"]  # crypto story: SOL is Solana
    assert set(syms("Bitcoin miner MARA Holdings mined 700 bitcoin", t)) == {"MARA", "BTC/USD"}


def test_prefilter_crypto_false_positives(ctx):
    t = crypto_table(ctx.db)
    assert "SOL/USD" not in syms("Sol Gel Technologies wins FDA nod for acne cream", t)
    assert syms("ETH Zurich researchers publish a new battery study", t) == []
    assert syms("Emeren (SOL) signs a solar power deal", t) == ["SOL"]  # no crypto words: the stock
    assert syms("Dogecoin soars 20%", t) == []  # not in the crypto list
    assert syms("Bitcoin jumps", t, crypto=False) == []  # backtests replay stock news only
    t.crypto_wanted = []  # crypto off
    assert syms("Bitcoin jumps as $BTC breaks out", t) == ["BTC"]  # (only the fund's cashtag, as before)
    assert syms("Bitcoin jumps", t) == []


def test_validator_accepts_only_enabled_crypto(ctx):
    t = crypto_table(ctx.db)
    res = validate_response(signal_json(ai_sig("BTCUSD"), ai_sig("ETH/USD")), t)
    assert res.status == "ok" and {s.ticker for s in res.signals} == {"BTC/USD", "ETH/USD"}
    assert next(s for s in res.signals if s.ticker == "BTC/USD").company == "Bitcoin"
    res = validate_response(signal_json(ai_sig("DOGE/USD")), t)
    assert res.status == "rejected" and "isn't in your crypto list" in res.problems[0]
    t.crypto_wanted = []
    assert validate_response(signal_json(ai_sig("BTC/USD")), t).status == "rejected"


def test_claude_prompt_names_the_enabled_pairs_only():
    from newstrader.ai.prompts import system_prompt

    assert "Never crypto" in system_prompt(3)
    p = system_prompt(3, ["ETH/USD", "BTC/USD"])
    assert "plus these crypto pairs: BTC/USD, ETH/USD" in p and "bitcoin miner" in p


# ================================================================================ risk rules
def rctx(**kw) -> RiskContext:
    base = dict(trading=TradingSettings(), risk=RiskSettings(),
                account=AccountSnapshot(equity=100_000, last_equity=100_000, buying_power=200_000, cash=100_000,
                                        crypto_buying_power=100_000),
                positions=[], market_open=False, now=datetime.now(UTC),
                asset=AssetInfo(True, False, False, 0.0001, 0.01))
    base.update(kw)
    return RiskContext(**base)


def test_crypto_entry_trades_24_7_by_dollar_amount():
    c = CryptoSettings()
    d = check_crypto_entry("BTC/USD", 60000.0, rctx(), c)  # market closed, market-hours-only on: still fine
    assert d.allowed, d.reason
    assert d.notional == 500.0 and d.qty == pytest.approx(500 / 60000)  # a fraction of a coin
    assert (d.stop_price, d.stop_limit_price, d.take_profit_price) == (58200.0, 57909.0, 63600.0)
    cheap = check_crypto_entry("DOGE/USD", 0.15, rctx(asset=AssetInfo(True, False, False, 1, 0.00001)), c)
    assert cheap.allowed  # no minimum share price, no whole shares


def test_crypto_entry_limits():
    c = CryptoSettings()
    small = rctx(account=AccountSnapshot(equity=100_000, last_equity=100_000, buying_power=200_000, cash=300,
                                         crypto_buying_power=300))
    assert check_crypto_entry("BTC/USD", 60000.0, small, c).notional == 294.0  # cash (no margin), not $500
    assert not check_crypto_entry("BTC/USD", 60000.0, rctx(kill_engaged=True), c).allowed
    loss = rctx(account=AccountSnapshot(equity=99_000, last_equity=100_000, buying_power=1, cash=99_000))
    d = check_crypto_entry("BTC/USD", 60000.0, loss, c)
    assert not d.allowed and d.halt_for_day
    held = rctx(positions=[PositionInfo("BTC/USD", 0.01, 600.0)])
    assert "already hold" in check_crypto_entry("BTC/USD", 60000.0, held, c).reason
    full = rctx(positions=[PositionInfo(s, 1, 100.0) for s in ("A", "B", "C", "D", "E")])
    assert "Max open positions" in check_crypto_entry("BTC/USD", 60000.0, full, c).reason
    cool = rctx(last_entry_at={"BTC/USD": datetime.now(UTC) - timedelta(minutes=5)})
    assert "Cooldown" in check_crypto_entry("BTC/USD", 60000.0, cool, c).reason
    tiny = rctx(asset=AssetInfo(True, False, False, 1.0, 0.01))
    assert "smallest BTC/USD order" in check_crypto_entry("BTC/USD", 60000.0, tiny, c).reason
    pct = rctx(risk=RiskSettings(max_pct_per_stock=0.2))  # 0.2% of $100k = $200
    assert check_crypto_entry("BTC/USD", 60000.0, pct, c).notional == 200.0
    blocked = rctx(risk=RiskSettings(whitelist=["AAPL"]))  # the black / white lists are for stocks
    assert check_crypto_entry("BTC/USD", 60000.0, blocked, c).allowed


def test_crypto_exit_ignores_market_hours():
    d = check_exit("BTC/USD", rctx(positions=[PositionInfo("BTC/USD", 0.0083, 500.0)]))
    assert d.allowed and d.qty == pytest.approx(0.0083)
    assert not check_exit("AAPL", rctx(positions=[PositionInfo("AAPL", 3, 690.0)])).allowed  # stocks: closed


# ================================================================================ trader + exit manager
@pytest.fixture
async def env(ctx):
    ctx.config.update({"crypto": {"enabled": True}})
    broker = FakeBroker(prices=dict(PRICES), market_open=False)
    trader = Trader(ctx, broker_factory=lambda: broker)
    alerts = FakeAlerts()
    ctx.services.update({"trader": trader, "alerts": alerts})
    await trader.connect()
    exits = CryptoExitManager(ctx)
    ctx.services["crypto_exits"] = exits
    yield SimpleNamespace(ctx=ctx, broker=broker, trader=trader, exits=exits, alerts=alerts)
    await trader.stop()


async def bought(env) -> dict:
    res = await env.trader.handle_signal(sig())
    assert res["action"] == "bought" and res["traded"], res
    await env.exits.run_once()
    plan = plan_row(env.ctx)
    assert plan["status"] == "active" and plan["stop_order_id"], plan
    return plan


async def test_buy_fills_then_gets_a_safety_stop_at_alpaca(env):
    res = await env.trader.handle_signal(sig())  # the stock market is closed: crypto trades anyway
    assert res["action"] == "bought", res
    assert ("submit_crypto_buy", "BTC/USD", 500.0) in env.broker.calls
    assert plan_row(env.ctx)["status"] == "pending"
    await env.exits.run_once()
    plan = plan_row(env.ctx)
    assert plan["status"] == "active" and plan["entry_price"] == 60000.0
    assert (plan["stop_price"], plan["stop_limit_price"], plan["take_profit_price"]) == (58200.0, 57909.0, 63600.0)
    stop = env.broker.order(plan["stop_order_id"])
    assert stop["order_type"] == "stop_limit" and stop["side"] == "sell" and stop["time_in_force"] == "gtc"
    assert stop["qty"] == pytest.approx(500 / 60000) and stop["status"] == "new"
    rows = env.ctx.db.query("SELECT intent, parent_alpaca_id FROM orders WHERE symbol = 'BTC/USD' ORDER BY id")
    assert [r["intent"] for r in rows] == ["open_long", "stop_loss"]
    assert rows[1]["parent_alpaca_id"] == plan["entry_order_id"]
    assert any(a["kind"] == "trade_placed" and "Bought $500 of BTC/USD" in a["title"] for a in env.alerts.sent)
    assert (await env.exits.run_once())["plans"][plan["id"]] == "watching"
    assert sum(1 for c in env.broker.calls if c[0] == "submit_crypto_stop") == 1  # not placed twice


async def test_safety_stop_covers_what_alpaca_holds_after_fees(env):
    env.broker.crypto_fee_pct = 0.25  # Alpaca takes its fee in coins
    plan = await bought(env)
    stop = env.broker.order(plan["stop_order_id"])
    assert stop["qty"] == pytest.approx(500 / 60000 * 0.9975, rel=1e-6)


async def test_take_profit_cancels_the_stop_then_sells(env):
    plan = await bought(env)
    events = track(env.broker)
    env.broker.prices["BTC/USD"] = 63700.0  # past the $63,600 target
    await env.exits.run_once()
    assert events == [("cancel", plan["stop_order_id"]), ("close", "BTC/USD")]
    assert env.broker.order(plan["stop_order_id"])["status"] == "canceled"
    assert env.broker.positions() == []
    p = plan_row(env.ctx)
    assert p["status"] == "closing" and "take-profit" in p["reason"]
    rows = env.ctx.db.query("SELECT intent FROM orders WHERE symbol = 'BTC/USD' ORDER BY id")
    assert [r["intent"] for r in rows][-1] == "take_profit"
    assert any("Crypto take-profit: selling BTC/USD" == a["title"] for a in env.alerts.sent)
    await env.exits.run_once()
    assert plan_row(env.ctx)["status"] == "closed"


async def test_stop_that_filled_at_alpaca_closes_the_plan(env):
    plan = await bought(env)
    env.broker.prices["BTC/USD"] = 58100.0
    assert env.broker.check_stops() == 1  # Alpaca filled the safety stop (maybe while NewsTrader was closed)
    await env.exits.run_once()
    p = plan_row(env.ctx, plan["id"])
    assert p["status"] == "closed" and "safety stop sold it at $58,100.00" in p["reason"]


async def test_fall_through_the_stop_limit_sells_at_market(env):
    plan = await bought(env)
    env.broker.prices["BTC/USD"] = 57000.0  # below the stop's $57,909 limit: the stop-limit can't fill
    assert env.broker.check_stops() == 0
    events = track(env.broker)
    await env.exits.run_once()
    assert events == [("cancel", plan["stop_order_id"]), ("close", "BTC/USD")]
    assert env.broker.positions() == [] and "past the stop's limit" in plan_row(env.ctx)["reason"]


async def test_restart_resync_replaces_a_missing_stop_and_drops_gone_positions(env):
    plan = await bought(env)
    env.broker.cancel_order(plan["stop_order_id"])  # the stop was cancelled while NewsTrader was closed
    fresh = CryptoExitManager(env.ctx)  # NewsTrader starts again
    env.ctx.services["crypto_exits"] = fresh
    out = await fresh.run_once()
    assert out["plans"][plan["id"]] == "stop placed again"
    new = plan_row(env.ctx, plan["id"])
    assert new["stop_order_id"] != plan["stop_order_id"] and env.broker.order(new["stop_order_id"])["status"] == "new"
    env.broker.close_position("BTC/USD")  # sold outside NewsTrader
    await fresh.run_once()
    assert plan_row(env.ctx, plan["id"])["status"] == "closed" and open_plans(env.ctx.db) == []


async def test_a_buy_that_never_fills_ends_its_plan(env):
    env.broker.hold_crypto_fills = True
    await env.trader.handle_signal(sig())
    plan = plan_row(env.ctx)
    assert (await env.exits.run_once())["plans"][plan["id"]] == "waiting for the fill"
    env.broker.cancel_order(plan["entry_order_id"])
    await env.exits.run_once()
    p = plan_row(env.ctx, plan["id"])
    assert p["status"] == "closed" and "didn't fill" in p["reason"]
    assert not [c for c in env.broker.calls if c[0] == "submit_crypto_stop"]


async def test_kill_switch_cancels_the_stop_and_rearm_puts_it_back(env):
    plan = await bought(env)
    await env.trader.kill()
    assert env.broker.order(plan["stop_order_id"])["status"] == "canceled"
    assert "Crypto safety stops were cancelled too" in env.alerts.sent[-1]["message"]
    env.broker.prices["BTC/USD"] = 70000.0  # even past the target: nothing happens while killed
    assert (await env.exits.run_once())["status"] == "paused"
    assert env.broker.positions() and sum(1 for c in env.broker.calls if c[0] == "submit_crypto_stop") == 1
    env.broker.prices["BTC/USD"] = 60500.0
    await env.trader.rearm()
    await env.exits.run_once()
    p = plan_row(env.ctx, plan["id"])
    assert p["stop_order_id"] != plan["stop_order_id"] and env.broker.order(p["stop_order_id"])["status"] == "new"


async def test_bearish_news_sells_a_held_coin_after_cancelling_its_stop(env):
    plan = await bought(env)
    events = track(env.broker)
    res = await env.trader.handle_signal(sig(direction="bearish", id=2))
    assert res["action"] == "sold" and res["traded"], res
    assert events[0] == ("cancel", plan["stop_order_id"]) and ("close", "BTC/USD") in events
    p = plan_row(env.ctx, plan["id"])
    assert p["status"] == "closing" and p["reason"] == "Sold on a bearish signal"
    await env.exits.run_once()
    assert plan_row(env.ctx, plan["id"])["status"] == "closed"


async def test_crypto_is_never_shorted(env):
    env.ctx.config.update({"trading": {"allow_shorting": True}})
    for conf in (95, 70):
        res = await env.trader.handle_signal(sig("ETH/USD", "bearish", conf))
        assert res["action"] == "ignored" and res["reason"] == NO_CRYPTO_SHORT
    res = await env.trader.handle_signal(sig("ETH/USD", "bearish", 70), manual=True)  # not even by hand
    assert res["action"] == "ignored"
    assert not [c for c in env.broker.calls if c[0] in ("submit_bracket", "submit_crypto_buy")]


async def test_crypto_off_simulator_live_and_review_only(env):
    env.ctx.config.update({"crypto": {"trade_on_news": False}})
    res = await env.trader.handle_signal(sig())
    assert res["action"] == "review" and "wait for your approval" in res["reason"]
    assert (await env.trader.handle_signal(sig(), manual=True))["action"] == "bought"  # approving it trades it
    env.ctx.config.update({"crypto": {"enabled": False, "trade_on_news": True}})
    res = await env.trader.handle_signal(sig("ETH/USD", id=3))
    assert res["action"] == "blocked" and "Crypto trading is off" in res["reason"]
    env.ctx.config.update({"crypto": {"enabled": True}})
    env.ctx.state._set_live_armed(True)
    res = await env.trader.handle_signal(sig("ETH/USD", id=4))
    assert res["action"] == "blocked" and "paper only" in res["reason"]
    env.ctx.state._set_live_armed(False)
    env.broker.supports_crypto = False  # e.g. the built-in simulator
    res = await env.trader.handle_signal(sig("ETH/USD", id=5))
    assert res["action"] == "blocked" and "simulator can't trade crypto" in res["reason"]


async def test_stocks_still_respect_market_hours(env):
    res = await env.trader.handle_signal(sig("AAPL"))
    assert res["action"] == "blocked" and "Market is closed" in res["reason"]


async def test_manual_close_of_a_coin(env):
    plan = await bought(env)
    await env.trader.manual_close("btcusd")
    assert env.broker.positions() == [] and plan_row(env.ctx, plan["id"])["reason"] == "Sold by you"


# ================================================================================ the local engine
@pytest.fixture
async def local_crypto(ctx, tmp_path):
    ctx.config.update({"crypto": {"enabled": True}})
    broker = FakeBroker(prices={**PRICES, "MARA": 15.0, "MSTR": 300.0, "COIN": 250.0})
    trader = Trader(ctx, broker_factory=lambda: broker)
    ctx.services.update({"trader": trader, "alerts": FakeAlerts()})
    await trader.connect()
    svc, _fake = fake_service(tmp_path)
    table = crypto_table(ctx.db)
    pipeline = Pipeline(ctx, tickers=table, local=LocalMLEngine(ctx, sentiment=svc, model_dir=tmp_path / "pm"))
    ctx.services["pipeline"] = pipeline
    assert table.crypto_symbols == {"BTC/USD", "ETH/USD", "SOL/USD"}  # wanted by settings, listed by Alpaca
    yield SimpleNamespace(ctx=ctx, pipeline=pipeline, broker=broker)
    await trader.stop()


def crypto_signal(out, symbol="BTC/USD"):
    return next((s for s in out["signals"] if s["ticker"] == symbol), None)


async def test_engine_crypto_event_trades_the_coin(local_crypto):
    out = await run(local_crypto.pipeline, news("SEC approves spot ether ETFs in landmark decision"))
    s = crypto_signal(out, "ETH/USD")
    assert s["direction"] == "bullish" and s["event"] == "Spot ETF approved" and s["confidence"] == 85
    assert s["traded"] == 1 and ("submit_crypto_buy", "ETH/USD", 500.0) in local_crypto.broker.calls
    assert "price model is for stocks" in s["reasoning"]


async def test_miner_headline_does_not_trade_bitcoin(local_crypto):
    out = await run(local_crypto.pipeline, news("MARA Holdings mined a record 700 bitcoin in October"))
    assert crypto_signal(out) is None  # company news that mentions bitcoin
    detail = json.loads(local_crypto.ctx.db.query_one("SELECT raw_response FROM analyses")["raw_response"])
    btc = next(d for d in detail["details"] if d["ticker"] == "BTC/USD")
    assert btc["direction"] == "neutral" and "company's news" in btc["neutral_because"]
    assert not [c for c in local_crypto.broker.calls if c[0] == "submit_crypto_buy"]


async def test_wording_only_crypto_goes_to_manual_review(local_crypto):
    out = await run(local_crypto.pipeline, news("Bitcoin hits a record high"))
    s = crypto_signal(out)
    assert s["direction"] == "bullish" and s["action"] == "review" and s["confidence"] < 80 and not s["traded"]
    assert "no concrete event" in s["reasoning"]


async def test_engine_crypto_rules_treasury_and_hack(local_crypto):
    out = await run(local_crypto.pipeline, news("Strategy buys another 10,000 bitcoin for its treasury"))
    s = crypto_signal(out)
    assert s["direction"] == "bullish" and s["event"] == "Bought for a treasury / reserve"
    assert s["action"] == "review"  # 70: below the buy threshold
    out = await run(local_crypto.pipeline, news("Bybit hacked: $1.5 billion in ether stolen"))
    s = crypto_signal(out, "ETH/USD")
    assert s["direction"] == "bearish" and s["event"] == "Exchange hack / exploit"
    assert s["action"] == "ignored" and NO_CRYPTO_SHORT in s["action_reason"]  # not held: never shorted


@pytest.mark.parametrize("text,terms,companies,want", [
    ("SEC approves spot bitcoin ETFs", ["bitcoin", "btc"], [], (1, "Spot ETF approved")),
    ("Bitcoin jumps on hopes of spot ETF approval", ["bitcoin", "btc"], [], (0, "")),
    ("SEC could approve spot ether ETFs this week", ["ether", "ethereum", "eth"], [], (0, "")),
    ("Spot bitcoin ETFs see record $1.2 billion inflows", ["bitcoin", "btc"], [], (1, "Big ETF inflows")),
    ("Bitcoin ETFs post $900 million outflows", ["bitcoin", "btc"], [], (-1, "Big ETF outflows")),
    ("SEC sues Binance; bitcoin slides", ["bitcoin", "btc"], [], (-1, "Regulator lawsuit / crackdown")),
    ("China bans bitcoin mining and trading", ["bitcoin", "btc"], [], (-1, "Country bans it")),
    ("Robinhood lists SOL for US customers", ["solana", "sol"], [], (1, "Big exchange listing")),
    ("Binance to delist solana pairs", ["solana", "sol"], [], (-1, "Exchange delisting")),
    ("Coinbase acquires bitcoin options exchange Deribit", ["bitcoin", "btc"], [("COIN", [("coinbase",)])], (0, "")),
    ("Bitcoin clears $70,000 as ETF inflows grow", ["bitcoin", "btc"], [], (0, "")),
    ("Why bitcoin fell 5% today", ["bitcoin", "btc"], [], (0, "")),
    ("Exchange denies it was hacked; bitcoin steady", ["bitcoin", "btc"], [], (0, "")),
])
def test_crypto_rules(text, terms, companies, want):
    r = crypto_reading(text, terms, companies)
    assert (r.direction, r.label if r.direction else "") == want, (r.label, r.neutral_reason)


def test_pro_ai_and_backtests_stay_stock_only(ctx):
    from newstrader.llm.engine import allowed_tickers

    t = crypto_table(ctx.db)
    pre = prefilter("Coinbase lists solana; bitcoin rallies", t)
    assert "BTC/USD" in [c.symbol for c in pre.candidates]
    assert allowed_tickers(pre) == ["COIN"]


# ================================================================================ charts, market, scoreboard
def coin_bars(end: datetime, hours: int = 30, base: float = 60000.0) -> list[dict]:
    n = hours * 60
    return [{"t": iso(end - timedelta(minutes=n - i)), "o": base + i, "h": base + i + 5, "l": base + i - 5,
             "c": base + i, "v": 2.0} for i in range(n)]


async def test_crypto_chart_card_reads_24_7(ctx):
    now = datetime(2026, 10, 10, 3, 0, tzinfo=UTC)  # a Saturday night: no stock market
    assert history_start(now, True) == datetime(2026, 10, 9, tzinfo=UTC)
    broker = FakeBroker(prices={})
    broker.bar_data["BTC/USD"] = coin_bars(now)
    ctx.services["trader"] = SimpleNamespace(broker=broker, clock={"is_open": False})
    card = await ChartService(ctx).card("btc/usd", now=now)
    assert card["symbol"] == "BTC/USD" and card["asset_class"] == "crypto" and card["feed"] == "CRYPTO"
    assert card["reading"]["asset_class"] == "crypto" and not card["reading"]["stale"]
    assert len(card["series"]["t"]) == 24 * 60  # the last 24 hours
    assert price_series(broker.bar_data["BTC/USD"], now, coin=True)["vwap"][-1] < card["series"]["close"][-1]


async def test_crypto_chart_signals_run_while_the_stock_market_is_closed(ctx):
    ctx.config.update({"crypto": {"enabled": True}})
    now = datetime(2026, 10, 10, 3, 0, tzinfo=UTC)
    broker = FakeBroker(prices={"BTC/USD": 61800.0})
    broker.bar_data["BTC/USD"] = coin_bars(now)
    ctx.services["trader"] = SimpleNamespace(broker=broker, clock={"is_open": False})
    ctx.services["pipeline"] = SimpleNamespace(tickers=crypto_table(ctx.db, ["BTC/USD"]))
    out = await ChartService(ctx).scan(now=now)
    assert out["status"] == "ok" and out["read"] == 1


async def test_market_monitor_watches_crypto_around_the_clock(ctx):
    ctx.loop = asyncio.get_running_loop()
    ctx.bus.bind_loop(ctx.loop)
    broker = FakeBroker(prices={"BTC/USD": 62700.0})
    broker.bar_data["BTC/USD"] = spike_bars(base=60000.0, tail=(60900.0, 61800.0, 62700.0))
    broker.bar_data["SPY"] = make_bars([600.0] * 35)
    alerts = FakeAlerts()
    ctx.services.update({"trader": SimpleNamespace(broker=broker, clock={"is_open": False}, positions=[],
                                                   open_orders=[], account={}), "alerts": alerts})
    ctx.config.update({"market": {"index_symbols": ["SPY"], "world_symbols": [], "watch_signals_minutes": 0}})
    m = MarketMonitor(ctx, tickers=crypto_table(ctx.db, ["BTC/USD"]))
    res = await m.run_once(now=T0)
    assert res["status"] == "closed" and res["crypto"] == 1  # stock market closed, crypto still checked
    [ev] = [e for e in ctx.db.query("SELECT * FROM market_events") if e["symbol"] == "BTC/USD"]
    assert ev["kind"] == "spike_up" and ev["alerted"] == 1
    assert alerts.sent[0]["title"] == "BTC/USD jumped +4.5% in 5 min"
    [tile] = m.payload()["crypto"]
    assert tile["symbol"] == "BTC/USD" and tile["name"] == "Bitcoin" and tile["price"] == 62700.0


async def test_scoreboard_prices_crypto_24_hours_later(ctx):
    t0 = datetime(2026, 10, 10, 3, 0, tzinfo=UTC)  # Saturday
    broker = FakeBroker(prices={"BTC/USD": 60000.0})
    ctx.services["trader"] = SimpleNamespace(broker=broker)
    sid = ctx.db.insert("signals", {"ticker": "BTC/USD", "direction": "bullish", "confidence": 85,
                                    "created_at": iso(t0)})
    await PerformanceTracker(ctx).run_once(now=t0 + timedelta(seconds=30))
    row = ctx.db.query_one("SELECT * FROM signal_prices WHERE signal_id = ?", (sid,))
    assert row["price_t0"] == 60000.0 and row["due_1d"] == iso(t0 + timedelta(days=1))


# ================================================================================ API
def test_settings_api_explains_crypto(client, ctx):
    meta = client.get("/api/settings").json()["meta"]["crypto"]
    assert meta["supported"] is False and "Alpaca" in meta["why"]
    r = client.put("/api/settings", json={"crypto": {"enabled": True, "symbols": "btc, eth"}})
    assert r.status_code == 200 and r.json()["settings"]["crypto"]["symbols"] == ["BTC/USD", "ETH/USD"]
    assert client.put("/api/settings", json={"crypto": {"symbols": "ETH/BTC"}}).status_code == 422


def test_portfolio_and_close_api_for_a_coin(client, ctx):
    ctx.config.update({"crypto": {"enabled": True}})
    broker = FakeBroker(prices=dict(PRICES))
    trader = Trader(ctx, broker_factory=lambda: broker)
    ctx.services["trader"] = trader

    async def setup():
        await trader.connect()
        await trader.handle_signal(sig())
        await CryptoExitManager(ctx).run_once()

    asyncio.run(setup())
    pos = next(p for p in client.get("/api/portfolio").json()["positions"] if p["symbol"] == "BTC/USD")
    assert pos["crypto"] and pos["stop_loss"] == 58200.0 and pos["take_profit"] == 63600.0
    assert pos["exit_plan"]["status"] == "active" and pos["exit_plan"]["stop_live"]
    r = client.post("/api/positions/BTC%2FUSD/close")
    assert r.status_code == 200, r.text
    assert broker.positions() == []


def test_simulator_says_why_crypto_is_off(client, ctx):
    from newstrader.trading.sim_broker import SimBroker

    assert SimBroker.supports_crypto is False
    ctx.services["trader"] = SimpleNamespace(broker=SimpleNamespace(supports_crypto=False, kind="simulator"),
                                             broker_kind="simulator")
    meta = client.get("/api/settings").json()["meta"]["crypto"]
    assert meta["supported"] is False and "simulator can't trade crypto" in meta["why"]


def test_chart_api_takes_a_crypto_pair(client, ctx, monkeypatch):
    now = datetime(2026, 10, 10, 3, 0, tzinfo=UTC)
    monkeypatch.setattr("newstrader.chart.service.utcnow", lambda: now)
    broker = FakeBroker(prices={})
    broker.bar_data["BTC/USD"] = coin_bars(now)
    ctx.services.update({"trader": SimpleNamespace(broker=broker, clock={"is_open": False}),
                         "chart": ChartService(ctx)})
    r = client.get("/api/chart/BTC%2FUSD")
    assert r.status_code == 200, r.text
    assert r.json()["symbol"] == "BTC/USD" and r.json()["asset_class"] == "crypto"
