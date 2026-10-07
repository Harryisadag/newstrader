"""Trader service against the in-memory FakeBroker (never touches Alpaca)."""

from __future__ import annotations

import pytest

from newstrader.trading.fake_broker import FakeBroker
from newstrader.trading.trader import Trader, decide_action


@pytest.fixture
async def trader(ctx):
    broker = FakeBroker()
    t = Trader(ctx, broker_factory=lambda: broker)
    ctx.services["trader"] = t
    await t.connect()
    yield t
    await t.stop()


def sig(ticker="AAPL", direction="bullish", confidence=85, **kw):
    return {"id": kw.pop("id", 1), "ticker": ticker, "direction": direction, "confidence": confidence,
            "reasoning": "test", **kw}


# ---------------------------------------------------------------- decide_action
def test_decide_action_thresholds(ctx):
    t = ctx.config.settings.trading
    assert decide_action("bullish", 80, t, False)[0] == "buy"
    assert decide_action("bullish", 79, t, False)[0] == "review"
    assert decide_action("bullish", 60, t, False)[0] == "review"
    assert decide_action("bullish", 59, t, False)[0] == "ignore"
    assert decide_action("neutral", 99, t, False)[0] == "ignore"
    assert decide_action("bearish", 90, t, True)[0] == "sell"
    assert decide_action("bearish", 90, t, False)[0] == "ignore"  # shorting off by default
    assert decide_action("bearish", 70, t, True)[0] == "review"


# ---------------------------------------------------------------- trading
async def test_bullish_signal_places_bracket_buy(trader, ctx):
    res = await trader.handle_signal(sig())
    assert res["traded"] and res["action"] == "bought", res
    call = trader.broker.calls[-1]
    assert call[0] == "submit_bracket" and call[1] == "AAPL" and call[2] == "buy" and call[3] == 4
    rows = ctx.db.query("SELECT * FROM orders ORDER BY id")
    intents = [r["intent"] for r in rows]
    assert intents == ["open_long", "take_profit", "stop_loss"]
    assert rows[0]["signal_id"] == 1 and rows[0]["mode"] == "paper"


async def test_review_band_does_not_trade(trader):
    res = await trader.handle_signal(sig(confidence=70))
    assert res["action"] == "review" and not res["traded"]
    assert not trader.broker.calls


async def test_cooldown_blocks_second_buy(trader):
    assert (await trader.handle_signal(sig()))["traded"]
    res = await trader.handle_signal(sig(id=2))
    assert res["action"] == "blocked" and "Cooldown" in res["reason"]


async def test_kill_switch_blocks_and_cancels(trader, ctx):
    await trader.handle_signal(sig())
    out = await trader.kill(close_positions=False)
    assert ctx.state.kill_engaged
    assert out["cancelled"] >= 1
    res = await trader.handle_signal(sig("NVDA", id=3))
    assert res["action"] == "blocked" and "Kill switch" in res["reason"]
    await trader.rearm()
    assert (await trader.handle_signal(sig("NVDA", id=4)))["traded"]


async def test_kill_switch_can_flatten_positions(trader):
    await trader.handle_signal(sig())
    await trader.kill(close_positions=True)
    assert trader.broker.positions() == []


async def test_bearish_sells_existing_long(trader, ctx):
    await trader.handle_signal(sig())
    res = await trader.handle_signal(sig(direction="bearish", confidence=90, id=5))
    assert res["action"] == "sold" and res["traded"]
    assert trader.broker.positions() == []
    assert ctx.db.query_one("SELECT intent FROM orders WHERE signal_id = 5")["intent"] == "close_long"


async def test_bearish_without_position_does_nothing_by_default(trader):
    res = await trader.handle_signal(sig(direction="bearish", confidence=95))
    assert res["action"] == "ignored"


async def test_bearish_shorts_when_enabled(trader, ctx):
    ctx.config.update({"trading": {"allow_shorting": True}})
    res = await trader.handle_signal(sig("TSLA", direction="bearish", confidence=95))
    assert res["action"] == "shorted", res
    assert trader.broker.calls[-1][2] == "sell"


async def test_manual_approve_trades_review_signal(trader):
    res = await trader.handle_signal(sig(confidence=65), manual=True)
    assert res["traded"]


async def test_auto_trade_off(trader, ctx):
    ctx.config.update({"trading": {"auto_trade": False}})
    res = await trader.handle_signal(sig())
    assert res["action"] == "blocked" and "Auto-trade" in res["reason"]


async def test_daily_loss_limit_halts_trading(trader, ctx):
    trader.broker.last_equity = 101_000  # account is down $1000 today
    await trader.refresh(force=True)
    assert ctx.state.halted_today
    res = await trader.handle_signal(sig())
    assert res["action"] == "blocked"


async def test_rejected_order_reports_error(trader):
    trader.broker.fail_next_submit = "insufficient buying power"
    res = await trader.handle_signal(sig())
    assert res["action"] == "error" and "insufficient" in res["reason"]


async def test_unknown_price_blocks(trader):
    res = await trader.handle_signal(sig("ZZZZ"))
    assert res["action"] == "blocked"


async def test_trade_log_has_realized_pnl(trader):
    await trader.handle_signal(sig())
    trader.broker.prices["AAPL"] = 240.0
    await trader.manual_close("AAPL")
    log = trader.trade_log()
    closes = [r for r in log if r["intent"] == "close_long"]
    assert closes and closes[0]["realized_pl"] == pytest.approx(40.0)


async def test_no_broker_means_blocked(ctx):
    t = Trader(ctx, broker_factory=lambda: None)
    await t.connect()
    res = await t.handle_signal(sig())
    assert res["action"] == "blocked"
    assert any(c["component"] == "alpaca" and c["level"] == "error" for c in ctx.state.components())
