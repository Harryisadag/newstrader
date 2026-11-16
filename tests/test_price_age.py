"""When the latest price was printed, from each real price source (the "fresh data only" check before a trade).

Alpaca's SDK clients and Yahoo's chart endpoint are replaced with stubs - nothing goes over the network."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from alpaca.data.models import Quote, Trade

from newstrader.trading.broker import Broker
from newstrader.trading.free_data import FreeMarketData

T = datetime(2026, 11, 4, 15, 30, 5, tzinfo=UTC)


def _trade(sym, price, at):
    return Trade(sym, {"t": at, "p": price, "s": 100, "x": "V", "i": 1, "c": ["@"], "z": "A"})


def _quote(sym, bid, ask, at):
    return Quote(sym, {"t": at, "bp": bid, "bs": 1, "bx": "V", "ap": ask, "as": 1, "ax": "V", "c": ["R"], "z": "A"})


class Data:
    """Stands in for alpaca's StockHistoricalDataClient / CryptoHistoricalDataClient."""

    def __init__(self, trades=None, quotes=None, fail_trades=False):
        self.trades, self.quotes, self.fail_trades = trades or {}, quotes or {}, fail_trades

    def _trades(self, req):
        if self.fail_trades:
            raise TimeoutError("timed out")
        return {s: self.trades[s] for s in [req.symbol_or_symbols] if s in self.trades}

    def _quotes(self, req):
        return {s: self.quotes[s] for s in [req.symbol_or_symbols] if s in self.quotes}

    get_stock_latest_trade = get_crypto_latest_trade = _trades
    get_stock_latest_quote = get_crypto_latest_quote = _quotes


def broker(data: Data) -> Broker:
    b = Broker.__new__(Broker)  # no keys, no network
    b.data, b._crypto, b.data_feed = data, data, "iex"
    return b


def test_alpaca_last_trade_with_its_time():
    b = broker(Data(trades={"AAPL": _trade("AAPL", 230.5, T)}))
    assert b.latest_price_info("AAPL") == (230.5, T)
    assert b.latest_price("AAPL") == 230.5  # unchanged


def test_alpaca_quote_midpoint_when_there_is_no_trade():
    b = broker(Data(quotes={"AAPL": _quote("AAPL", 230.0, 231.0, T - timedelta(seconds=3))}))
    assert b.latest_price_info("AAPL") == (230.5, T - timedelta(seconds=3))
    # the trade lookup failing (a timeout) falls back to the quote the same way
    b = broker(Data(trades={"AAPL": _trade("AAPL", 1.0, T)}, quotes={"AAPL": _quote("AAPL", 230.0, 231.0, T)},
                    fail_trades=True))
    assert b.latest_price_info("AAPL") == (230.5, T)


def test_alpaca_crypto_and_no_price():
    b = broker(Data(trades={"BTC/USD": _trade("BTC/USD", 101_000.0, T)}))
    assert b.latest_price_info("BTCUSD") == (101_000.0, T)
    assert broker(Data()).latest_price_info("AAPL") == (None, None)


def _yahoo(bars: list[tuple[datetime, float]], meta: dict) -> FreeMarketData:
    d = FreeMarketData.__new__(FreeMarketData)
    rows = [{"t": t.isoformat().replace("+00:00", "Z"), "o": c, "h": c, "l": c, "c": c, "v": 10} for t, c in bars]
    d._today = lambda symbol, max_age=20.0: (meta, rows)
    return d


def test_yahoo_last_bar_or_last_trade_time():
    bar_start = datetime(2026, 11, 4, 15, 30, tzinfo=UTC)
    # the last trade's time from Yahoo's meta, when it's in that bar (same price)
    d = _yahoo([(bar_start - timedelta(minutes=1), 99.0), (bar_start, 100.0)],
               {"regularMarketPrice": 100.0, "regularMarketTime": int(T.timestamp())})
    assert d.latest_price_info("AAPL") == (100.0, T)
    # otherwise the bar's start (never claims to be fresher than it is)
    d = _yahoo([(bar_start, 100.0)], {"regularMarketPrice": 98.0, "regularMarketTime": int(T.timestamp())})
    assert d.latest_price_info("AAPL") == (100.0, bar_start)
    # no bars today: Yahoo's last price and its time
    d = _yahoo([], {"regularMarketPrice": 97.5, "regularMarketTime": int(T.timestamp())})
    assert d.latest_price_info("AAPL") == (97.5, T)
    assert _yahoo([], {}).latest_price_info("AAPL") == (None, None)


def test_simulator_without_times_reports_unknown():
    from newstrader.trading.sim_broker import SimBroker

    sim = SimBroker.__new__(SimBroker)
    sim.data = SimpleNamespace(latest_price=lambda s: 50.0)  # a price source that can't say when
    assert sim.latest_price_info("AAPL") == (50.0, None)
