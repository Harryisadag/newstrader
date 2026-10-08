"""The built-in paper-trading simulator, driven by made-up prices and a controllable clock."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from newstrader.db import Database, parse_iso
from newstrader.trading.free_data import parse_chart, parse_symbol_files
from newstrader.trading.sim_broker import SimBroker

OPEN_DAY = datetime(2026, 10, 8, 14, 0, tzinfo=UTC)  # Thursday 10:00 New York - market open


class Clock:
    def __init__(self, t: datetime):
        self.t = t

    def __call__(self) -> datetime:
        return self.t


class FakeData:
    data_feed = "test"
    last_error = None

    def __init__(self):
        self.prices: dict[str, float] = {}
        self.prev: dict[str, float] = {}
        self.minute_bars: dict[str, list[dict]] = {}

    def latest_price(self, symbol):
        return self.prices.get(symbol)

    def bars_multi(self, symbols, start, end, timeframe="1Min", feed=None):
        return {s: [b for b in self.minute_bars.get(s, []) if start <= parse_iso(b["t"]) <= end] for s in symbols}

    def snapshots(self, symbols, feed=None):
        return {s: {"price": self.prices.get(s), "prev_close": self.prev.get(s)} for s in symbols}

    def bar(self, sym, t: datetime, o, h, lo, c):
        self.minute_bars.setdefault(sym, []).append(
            {"t": t.isoformat().replace("+00:00", "Z"), "o": o, "h": h, "l": lo, "c": c, "v": 1000})


@pytest.fixture
def sim(tmp_path):
    clock = Clock(OPEN_DAY)
    data = FakeData()
    data.prices["NVDA"] = 200.0
    broker = SimBroker(Database(tmp_path / "t.db"), data=data, starting_cash=100_000, slippage_pct=0.05, clock=clock)
    return broker, data, clock


def tick(broker, clock, minutes: int):
    clock.t += timedelta(minutes=minutes)
    broker.process(force=True)


def test_buy_fills_now_with_slippage_and_arms_the_exits(sim):
    b, data, clock = sim
    o = b.submit_bracket("NVDA", "buy", 10, 210.0, 195.0, "t1")
    assert o["status"] == "filled" and o["filled_avg_price"] == pytest.approx(200.1)
    assert {leg["order_type"]: leg["status"] for leg in o["legs"]} == {"limit": "new", "stop": "held"}
    acct = b.account()
    assert acct["cash"] == pytest.approx(100_000 - 2001.0)
    assert acct["equity"] == pytest.approx(100_000 - 1.0)  # marked at 200 after paying 200.10
    pos = b.positions()[0]
    assert pos["symbol"] == "NVDA" and pos["qty"] == 10 and pos["side"] == "long"
    assert len(b.orders("open")) == 1


def test_take_profit_hit_closes_the_trade_and_cancels_the_stop(sim):
    b, data, clock = sim
    b.submit_bracket("NVDA", "buy", 10, 210.0, 195.0, "t1")
    data.bar("NVDA", OPEN_DAY + timedelta(minutes=1), 200, 205, 199, 204)
    data.bar("NVDA", OPEN_DAY + timedelta(minutes=2), 204, 211, 203, 209)
    data.prices["NVDA"] = 209.0
    tick(b, clock, 3)
    assert b.positions() == []
    parent = b.orders("all")[0]
    legs = {leg["order_type"]: leg for leg in parent["legs"]}
    assert legs["limit"]["status"] == "filled" and legs["limit"]["filled_avg_price"] == 210.0
    assert legs["stop"]["status"] == "canceled"
    assert b.account()["cash"] == pytest.approx(100_000 - 2001.0 + 2100.0)
    assert b.orders("open") == []


def test_stop_first_when_one_minute_touches_both_and_gaps_fill_worse(sim):
    b, data, clock = sim
    b.submit_bracket("NVDA", "buy", 10, 210.0, 195.0, "t1")
    data.bar("NVDA", OPEN_DAY + timedelta(minutes=1), 200, 212, 194, 205)  # wild minute: both levels touched
    tick(b, clock, 2)
    legs = {leg["order_type"]: leg for leg in b.orders("all")[0]["legs"]}
    assert legs["stop"]["status"] == "filled" and legs["stop"]["filled_avg_price"] == 195.0
    # a gap down through the stop fills at the (worse) opening price
    b.submit_bracket("NVDA", "buy", 10, 230.0, 190.0, "t2")
    data.bar("NVDA", clock.t + timedelta(minutes=1), 180, 182, 179, 181)
    tick(b, clock, 2)
    second = b.orders("all")[-1]
    stop = next(leg for leg in second["legs"] if leg["order_type"] == "stop")
    assert stop["status"] == "filled" and stop["filled_avg_price"] == 180


def test_orders_sent_while_closed_fill_at_the_next_open(sim):
    b, data, clock = sim
    clock.t = datetime(2026, 10, 8, 23, 0, tzinfo=UTC)  # 19:00 New York - closed
    o = b.submit_bracket("NVDA", "buy", 5, 220.0, 190.0, "t1")
    assert o["status"] == "accepted" and b.positions() == []
    assert not b.clock()["is_open"]
    next_open = datetime(2026, 10, 9, 13, 30, tzinfo=UTC)
    data.bar("NVDA", next_open, 203.0, 204, 202, 203.5)
    data.prices["NVDA"] = 203.5
    clock.t = next_open + timedelta(minutes=2)
    b.process(force=True)
    filled = b.orders("all")[0]
    assert filled["status"] == "filled" and filled["filled_avg_price"] == pytest.approx(203.0 * 1.0005)


def test_day_profit_is_measured_from_the_previous_close(sim):
    b, data, clock = sim
    b.submit_bracket("NVDA", "buy", 10, 260.0, 150.0, "t1")
    clock.t = datetime(2026, 10, 8, 21, 0, tzinfo=UTC)  # 17:00 New York: after the close
    data.prices["NVDA"] = 202.0
    b.process(force=True)
    close_value = b.account()["equity"]
    assert b.account()["day_pl"] == pytest.approx(0.0)
    clock.t = datetime(2026, 10, 9, 15, 0, tzinfo=UTC)  # next day, open
    data.prices["NVDA"] = 205.0
    b.process(force=True)
    acct = b.account()
    assert acct["last_equity"] == pytest.approx(close_value)
    assert acct["day_pl"] == pytest.approx(30.0)


def test_shorting_and_buying_power(sim):
    b, data, clock = sim
    b.submit_bracket("NVDA", "sell", 10, 190.0, 210.0, "s1")
    pos = b.positions()[0]
    assert pos["qty"] == -10 and pos["side"] == "short"
    acct = b.account()
    assert acct["short_market_value"] == pytest.approx(-2000.0)
    assert acct["buying_power"] == pytest.approx(acct["equity"] - 2000.0)
    data.bar("NVDA", OPEN_DAY + timedelta(minutes=1), 199, 200, 189, 190)
    tick(b, clock, 2)
    assert b.positions() == []  # take-profit (buy back at 190) filled


def test_close_cancel_persist_and_reset(sim, tmp_path):
    b, data, clock = sim
    b.submit_bracket("NVDA", "buy", 10, 210.0, 195.0, "t1")
    assert b.cancel_all_orders() == 2  # the two exit legs: the position is now unprotected
    assert b.positions()[0]["qty"] == 10
    b.close_position("NVDA")
    assert b.positions() == []
    again = SimBroker(b.db, data=data, clock=clock)  # a restart keeps the account
    assert again.state["account_number"] == b.state["account_number"]
    assert again.account()["cash"] == pytest.approx(b.account()["cash"])
    again.reset(50_000)
    assert again.account()["cash"] == 50_000 and again.orders("all") == []
    with pytest.raises(RuntimeError, match="buying power"):
        again.submit_bracket("NVDA", "buy", 1000, 300, 100, "big")


def test_free_data_parsers():
    payload = {"chart": {"result": [{"meta": {"symbol": "AAPL", "regularMarketPrice": 337.43},
                                     "timestamp": [1791446400, 1791446460, 1791446520],
                                     "indicators": {"quote": [{"open": [336.21, None, 337.0],
                                                               "high": [337.03, None, 337.3],
                                                               "low": [335.8776, None, 337.0],
                                                               "close": [336.7, None, 337.3],
                                                               "volume": [0, None, 1200]}]}}], "error": None}}
    meta, bars = parse_chart(payload)
    assert meta["regularMarketPrice"] == 337.43 and len(bars) == 2  # the empty minute is skipped
    assert bars[0] == {"t": "2026-10-08T08:00:00Z", "o": 336.21, "h": 337.03, "l": 335.8776, "c": 336.7, "v": 0.0}
    listed = ("Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares\n"
              "AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N\nZXZZT|NASDAQ TEST STOCK|G|Y|N|100|N|N\n"
              "File Creation Time: 1008202616:01|||||||\n")
    other = ("ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol\n"
             "BRK.B|Berkshire Hathaway Inc. Class B|N|BRK.B|N|100|N|BRK=B\nSPY|SPDR S&P 500 ETF Trust|P|SPY|Y|100|N|SPY\n")
    assets = {a["symbol"]: a for a in parse_symbol_files(listed, other)}
    assert set(assets) == {"AAPL", "BRK.B", "SPY"}
    assert assets["AAPL"]["name"] == "Apple Inc. Common Stock" and assets["AAPL"]["exchange"] == "NASDAQ"
    assert assets["SPY"]["exchange"] == "ARCA" and assets["SPY"]["etf"] and assets["BRK.B"]["exchange"] == "NYSE"
