from __future__ import annotations

from newstrader.trading.pnl import realized_pnl


def f(i, side, qty, price, t, sym="AAPL"):
    return {"id": i, "symbol": sym, "side": side, "filled_qty": qty, "filled_avg_price": price, "filled_at": t}


def test_long_round_trip():
    out = realized_pnl([f("b", "buy", 10, 100, "1"), f("s", "sell", 10, 104, "2")])
    assert out == {"s": 40.0}


def test_fifo_partial_lots():
    out = realized_pnl([f("b1", "buy", 5, 100, "1"), f("b2", "buy", 5, 110, "2"), f("s", "sell", 7, 120, "3")])
    assert out["s"] == 5 * 20 + 2 * 10


def test_short_round_trip():
    out = realized_pnl([f("ss", "sell", 4, 50, "1"), f("c", "buy", 4, 45, "2")])
    assert out == {"c": 20.0}


def test_symbols_are_separate_and_unfilled_ignored():
    out = realized_pnl([f("b", "buy", 1, 10, "1", "A"), f("s", "sell", 1, 5, "2", "B"),
                        {"id": "x", "symbol": "A", "side": "sell", "filled_qty": 0, "filled_avg_price": None, "filled_at": None}])
    assert out == {}


def test_flip_long_to_short():
    out = realized_pnl([f("b", "buy", 2, 10, "1"), f("s", "sell", 5, 12, "2"), f("c", "buy", 3, 11, "3")])
    assert out["s"] == 4.0  # closed 2 long for +2 each
    assert out["c"] == 3.0  # covered 3 short opened at 12, bought at 11
