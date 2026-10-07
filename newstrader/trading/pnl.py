"""Realized profit/loss per filled order, matched first-in-first-out per symbol (longs and shorts)."""

from __future__ import annotations

from collections import defaultdict, deque


def realized_pnl(fills: list[dict]) -> dict:
    """fills: dicts with id, symbol, side ('buy'/'sell'), filled_qty, filled_avg_price, filled_at.

    Returns {order_id: realized_pl} for every order that closed (part of) a position.
    """
    out: dict = {}
    # lots[symbol] = deque of [signed_qty, price]  (+ long lots, - short lots)
    lots: dict[str, deque] = defaultdict(deque)
    for f in sorted(fills, key=lambda x: (x.get("filled_at") or "", str(x.get("id")))):
        qty = float(f.get("filled_qty") or 0)
        price = f.get("filled_avg_price")
        if qty <= 0 or price is None:
            continue
        price = float(price)
        signed = qty if f["side"] == "buy" else -qty
        book = lots[f["symbol"]]
        pnl = 0.0
        closed_any = False
        while signed != 0 and book and (book[0][0] > 0) != (signed > 0):
            lot_qty, lot_price = book[0]
            match = min(abs(lot_qty), abs(signed))
            if lot_qty > 0:  # closing a long with a sell
                pnl += (price - lot_price) * match
                book[0][0] -= match
                signed += match
            else:  # covering a short with a buy
                pnl += (lot_price - price) * match
                book[0][0] += match
                signed -= match
            closed_any = True
            if abs(book[0][0]) < 1e-9:
                book.popleft()
        if abs(signed) > 1e-9:
            book.append([signed, price])
        if closed_any:
            out[f["id"]] = round(pnl, 2)
    return out
