"""Risk control tests - every rule, plus edge cases."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from newstrader.config import RiskSettings, TradingSettings
from newstrader.trading.risk import (
    AccountSnapshot,
    AssetInfo,
    PositionInfo,
    RiskContext,
    bracket_prices,
    check_entry,
    check_exit,
)

NOW = datetime(2026, 10, 7, 15, 0, tzinfo=UTC)


def make_ctx(**over) -> RiskContext:
    trading = over.pop("trading", TradingSettings())
    risk = over.pop("risk", RiskSettings())
    account = over.pop("account", AccountSnapshot(equity=100_000, last_equity=100_000, buying_power=200_000,
                                                  cash=100_000, shorting_enabled=True))
    base = dict(trading=trading, risk=risk, account=account, positions=[], market_open=True, now=NOW,
                asset=AssetInfo(tradable=True, shortable=True, easy_to_borrow=True))
    base.update(over)
    return RiskContext(**base)


# ---------------------------------------------------------------- happy path + sizing
def test_buy_allowed_and_sized_by_max_dollars():
    d = check_entry("AAPL", "buy", 230.0, make_ctx())
    assert d.allowed, d.reason
    assert d.qty == 4  # floor(1000 / 230)
    assert d.notional == pytest.approx(920.0)
    assert d.stop_price == pytest.approx(225.40)  # -2%
    assert d.take_profit_price == pytest.approx(239.20)  # +4%


def test_symbol_is_uppercased():
    assert check_entry("aapl", "buy", 100.0, make_ctx()).allowed


def test_max_pct_per_stock_limits_size():
    ctx = make_ctx(risk=RiskSettings(max_dollars_per_trade=50_000, max_pct_per_stock=2))
    d = check_entry("AAPL", "buy", 100.0, ctx)
    assert d.allowed and d.qty == 20  # 2% of 100k = $2000


def test_max_pct_counts_existing_position():
    ctx = make_ctx(risk=RiskSettings(max_dollars_per_trade=50_000, max_pct_per_stock=10),
                   positions=[PositionInfo("AAPL", 40, 9_500.0)])
    d = check_entry("AAPL", "buy", 100.0, ctx)
    assert d.allowed and d.qty == 5  # only $500 of room left


def test_already_at_max_pct_blocks():
    ctx = make_ctx(positions=[PositionInfo("AAPL", 50, 10_000.0)])
    d = check_entry("AAPL", "buy", 100.0, ctx)
    assert not d.allowed and "max" in d.reason.lower()


def test_buying_power_limits_size():
    acct = AccountSnapshot(equity=100_000, last_equity=100_000, buying_power=500, cash=500)
    d = check_entry("AAPL", "buy", 100.0, make_ctx(account=acct))
    assert d.allowed and d.qty == 4  # 98% of $500


def test_no_buying_power_blocks():
    acct = AccountSnapshot(equity=100_000, last_equity=100_000, buying_power=0, cash=0)
    assert not check_entry("AAPL", "buy", 100.0, make_ctx(account=acct)).allowed


def test_share_more_expensive_than_budget_blocks():
    d = check_entry("NVR", "buy", 7_500.0, make_ctx())
    assert not d.allowed and "whole shares" in d.reason


# ---------------------------------------------------------------- gates
def test_kill_switch_blocks_everything():
    ctx = make_ctx(kill_engaged=True, positions=[PositionInfo("AAPL", 5, 1000.0)])
    assert "Kill switch" in check_entry("AAPL", "buy", 100.0, ctx).reason
    assert not check_entry("AAPL", "buy", 100.0, ctx, manual=True).allowed
    assert not check_exit("AAPL", ctx).allowed


def test_auto_trade_off_blocks_automatic_but_not_manual():
    ctx = make_ctx(trading=TradingSettings(auto_trade=False))
    assert not check_entry("AAPL", "buy", 100.0, ctx).allowed
    assert check_entry("AAPL", "buy", 100.0, ctx, manual=True).allowed


def test_daily_loss_limit_blocks_and_requests_halt():
    acct = AccountSnapshot(equity=99_400, last_equity=100_000, buying_power=100_000)
    d = check_entry("AAPL", "buy", 100.0, make_ctx(account=acct))
    assert not d.allowed and d.halt_for_day
    assert "Daily loss limit" in d.reason


def test_daily_loss_just_under_limit_allowed():
    acct = AccountSnapshot(equity=99_501, last_equity=100_000, buying_power=100_000)
    assert check_entry("AAPL", "buy", 100.0, make_ctx(account=acct)).allowed


def test_daily_loss_exactly_at_limit_blocks():
    acct = AccountSnapshot(equity=99_500, last_equity=100_000, buying_power=100_000)
    assert check_entry("AAPL", "buy", 100.0, make_ctx(account=acct)).halt_for_day


def test_halted_today_blocks_entries_but_not_exits():
    ctx = make_ctx(halted_today=True, positions=[PositionInfo("AAPL", 5, 1000.0)])
    assert not check_entry("MSFT", "buy", 100.0, ctx).allowed
    assert check_exit("AAPL", ctx).allowed


def test_account_blocked():
    acct = AccountSnapshot(equity=100_000, last_equity=100_000, buying_power=1e5, trading_blocked=True)
    assert "blocked" in check_entry("AAPL", "buy", 100.0, make_ctx(account=acct)).reason


def test_market_hours_only():
    closed = make_ctx(market_open=False)
    assert "closed" in check_entry("AAPL", "buy", 100.0, closed).reason
    anytime = make_ctx(market_open=False, risk=RiskSettings(market_hours_only=False))
    assert check_entry("AAPL", "buy", 100.0, anytime).allowed


def test_blacklist_and_whitelist():
    ctx = make_ctx(risk=RiskSettings(blacklist=["GME"]))
    assert "blacklist" in check_entry("GME", "buy", 20.0, ctx).reason
    assert check_entry("AAPL", "buy", 20.0, ctx).allowed
    ctx = make_ctx(risk=RiskSettings(whitelist=["AAPL", "MSFT"]))
    assert check_entry("MSFT", "buy", 20.0, ctx).allowed
    assert "whitelist" in check_entry("TSLA", "buy", 20.0, ctx).reason


def test_blacklist_wins_over_whitelist():
    ctx = make_ctx(risk=RiskSettings(whitelist=["GME"], blacklist=["GME"]))
    assert not check_entry("GME", "buy", 20.0, ctx).allowed


def test_untradable_asset():
    ctx = make_ctx(asset=AssetInfo(tradable=False))
    assert "tradable" in check_entry("XYZ", "buy", 20.0, ctx).reason


def test_cooldown():
    recent = make_ctx(last_entry_at={"AAPL": NOW - timedelta(minutes=10)})
    d = check_entry("AAPL", "buy", 100.0, recent)
    assert not d.allowed and "Cooldown" in d.reason
    old = make_ctx(last_entry_at={"AAPL": NOW - timedelta(minutes=31)})
    assert check_entry("AAPL", "buy", 100.0, old).allowed
    other = make_ctx(last_entry_at={"MSFT": NOW - timedelta(minutes=1)})
    assert check_entry("AAPL", "buy", 100.0, other).allowed
    off = make_ctx(risk=RiskSettings(ticker_cooldown_minutes=0), last_entry_at={"AAPL": NOW})
    assert check_entry("AAPL", "buy", 100.0, off).allowed


def test_pending_order_blocks_duplicate():
    ctx = make_ctx(pending_symbols={"AAPL"})
    assert "waiting to fill" in check_entry("AAPL", "buy", 100.0, ctx).reason


def test_max_open_positions():
    pos = [PositionInfo(s, 1, 100.0) for s in ("A", "B", "C", "D", "E")]
    ctx = make_ctx(positions=pos)
    assert "Max open positions" in check_entry("AAPL", "buy", 100.0, ctx).reason
    # adding to an existing position doesn't count as a new one
    assert check_entry("A", "buy", 100.0, ctx).allowed


def test_pending_orders_count_toward_max_positions():
    pos = [PositionInfo(s, 1, 100.0) for s in ("A", "B", "C", "D")]
    ctx = make_ctx(positions=pos, pending_symbols={"E"})
    assert not check_entry("AAPL", "buy", 100.0, ctx).allowed


def test_zero_qty_positions_ignored():
    pos = [PositionInfo(s, 0, 0.0) for s in ("A", "B", "C", "D", "E")]
    assert check_entry("AAPL", "buy", 100.0, make_ctx(positions=pos)).allowed


@pytest.mark.parametrize("price", [None, 0.0, -5.0, float("nan"), float("inf")])
def test_bad_prices_blocked(price):
    assert not check_entry("AAPL", "buy", price, make_ctx()).allowed


def test_min_share_price():
    assert "minimum" in check_entry("PENNY", "buy", 0.5, make_ctx()).reason
    ctx = make_ctx(risk=RiskSettings(min_share_price=0))
    assert check_entry("PENNY", "buy", 0.5, ctx).allowed


# ---------------------------------------------------------------- shorting
def test_short_blocked_by_default():
    assert "Short selling is off" in check_entry("TSLA", "short", 250.0, make_ctx()).reason


def test_short_allowed_when_enabled():
    ctx = make_ctx(trading=TradingSettings(allow_shorting=True))
    d = check_entry("TSLA", "short", 250.0, ctx)
    assert d.allowed and d.qty == 4
    assert d.stop_price > 250 > d.take_profit_price


def test_short_needs_borrowable_and_account_permission():
    t = TradingSettings(allow_shorting=True)
    assert not check_entry("TSLA", "short", 250.0, make_ctx(trading=t, asset=AssetInfo(True, False, False))).allowed
    acct = AccountSnapshot(equity=1e5, last_equity=1e5, buying_power=1e5, shorting_enabled=False)
    assert not check_entry("TSLA", "short", 250.0, make_ctx(trading=t, account=acct)).allowed


def test_no_short_on_top_of_long_and_no_long_on_top_of_short():
    t = TradingSettings(allow_shorting=True)
    long_ctx = make_ctx(trading=t, positions=[PositionInfo("TSLA", 2, 500.0)])
    assert not check_entry("TSLA", "short", 250.0, long_ctx).allowed
    short_ctx = make_ctx(positions=[PositionInfo("TSLA", -2, -500.0)])
    assert not check_entry("TSLA", "buy", 250.0, short_ctx).allowed


def test_unknown_side():
    assert not check_entry("TSLA", "yolo", 250.0, make_ctx()).allowed


# ---------------------------------------------------------------- exits
def test_exit_requires_long_position():
    assert not check_exit("AAPL", make_ctx()).allowed
    d = check_exit("AAPL", make_ctx(positions=[PositionInfo("AAPL", 7, 1610.0)]))
    assert d.allowed and d.qty == 7


def test_exit_respects_sell_on_bearish_setting():
    ctx = make_ctx(trading=TradingSettings(sell_on_bearish=False), positions=[PositionInfo("AAPL", 7, 1610.0)])
    assert not check_exit("AAPL", ctx).allowed
    assert check_exit("AAPL", ctx, manual=True).allowed


def test_exit_respects_market_hours():
    ctx = make_ctx(market_open=False, positions=[PositionInfo("AAPL", 7, 1610.0)])
    assert not check_exit("AAPL", ctx).allowed


# ---------------------------------------------------------------- bracket prices
def test_bracket_prices_round_and_stay_a_tick_away():
    stop, tp = bracket_prices("buy", 10.0, 0.01, 0.01)
    assert stop == 9.99 and tp == 10.01
    stop, tp = bracket_prices("buy", 123.456, 2, 4)
    assert stop == 120.99 and tp == 128.39
    stop, tp = bracket_prices("short", 50.0, 2, 4)
    assert stop == 51.0 and tp == 48.0
    stop, tp = bracket_prices("buy", 0.5, 10, 10)
    assert stop == 0.45 and tp == 0.55
