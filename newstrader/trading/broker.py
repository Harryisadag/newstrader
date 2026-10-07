"""Thin wrapper around alpaca-py. Returns plain dicts so the rest of the app never touches SDK objects.

All methods are blocking (HTTP calls) - call them with asyncio.to_thread from async code.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, date, datetime, timedelta
from typing import Any

from ..state import AppState
from .live_guard import AlpacaCredentials, assert_mode_allowed

log = logging.getLogger(__name__)


def _f(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    return str(value)


def _enum(value: Any) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "value", value))


def order_to_dict(o: Any) -> dict:
    legs = [order_to_dict(leg) for leg in (getattr(o, "legs", None) or [])]
    return {
        "id": str(o.id),
        "client_order_id": o.client_order_id,
        "symbol": o.symbol,
        "side": _enum(o.side),
        "qty": _f(o.qty),
        "filled_qty": _f(o.filled_qty),
        "filled_avg_price": _f(o.filled_avg_price),
        "order_type": _enum(o.order_type or o.type),
        "order_class": _enum(o.order_class),
        "status": _enum(o.status),
        "limit_price": _f(o.limit_price),
        "stop_price": _f(o.stop_price),
        "time_in_force": _enum(o.time_in_force),
        "submitted_at": _iso(o.submitted_at or o.created_at),
        "filled_at": _iso(o.filled_at),
        "updated_at": _iso(o.updated_at),
        "legs": legs,
    }


def position_to_dict(p: Any) -> dict:
    return {
        "symbol": p.symbol,
        "qty": _f(p.qty),
        "qty_available": _f(getattr(p, "qty_available", None)),
        "side": _enum(p.side),
        "avg_entry_price": _f(p.avg_entry_price),
        "current_price": _f(p.current_price),
        "market_value": _f(p.market_value),
        "cost_basis": _f(p.cost_basis),
        "unrealized_pl": _f(p.unrealized_pl),
        "unrealized_plpc": (_f(p.unrealized_plpc) or 0.0) * 100,
        "unrealized_intraday_pl": _f(p.unrealized_intraday_pl),
        "change_today": (_f(p.change_today) or 0.0) * 100,
        "lastday_price": _f(p.lastday_price),
    }


class Broker:
    """Real Alpaca connection (paper by default; live only through live_guard)."""

    def __init__(self, creds: AlpacaCredentials, state: AppState, data_feed: str = "iex"):
        assert_mode_allowed(state, creds.paper)
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.trading.client import TradingClient

        self.paper = creds.paper
        self.creds = creds
        self.data_feed = data_feed
        self.trading = TradingClient(creds.api_key, creds.secret_key, paper=creds.paper)
        self.data = StockHistoricalDataClient(creds.api_key, creds.secret_key)

    @property
    def mode(self) -> str:
        return "paper" if self.paper else "live"

    # ---------------------------------------------------------------- account
    def account(self) -> dict:
        a = self.trading.get_account()
        equity = _f(a.equity) or 0.0
        last_equity = _f(a.last_equity) or equity
        day_pl = equity - last_equity
        return {
            "account_number": a.account_number,
            "status": _enum(a.status),
            "equity": equity,
            "last_equity": last_equity,
            "day_pl": day_pl,
            "day_pl_pct": (day_pl / last_equity * 100) if last_equity else 0.0,
            "cash": _f(a.cash),
            "buying_power": _f(a.buying_power),
            "long_market_value": _f(a.long_market_value),
            "short_market_value": _f(a.short_market_value),
            "shorting_enabled": bool(a.shorting_enabled),
            "trading_blocked": bool(a.trading_blocked or a.account_blocked or a.trade_suspended_by_user),
            "pattern_day_trader": bool(a.pattern_day_trader),
            "daytrade_count": a.daytrade_count,
        }

    def clock(self) -> dict:
        c = self.trading.get_clock()
        return {"is_open": bool(c.is_open), "next_open": _iso(c.next_open), "next_close": _iso(c.next_close),
                "timestamp": _iso(c.timestamp)}

    def calendar(self, start: date, end: date) -> list[dict]:
        """Trading sessions. Alpaca gives open/close as New York wall-clock times without a timezone."""
        from zoneinfo import ZoneInfo

        from alpaca.trading.requests import GetCalendarRequest

        et = ZoneInfo("America/New_York")

        def stamp(dt: datetime) -> str | None:
            return _iso(dt if dt.tzinfo else dt.replace(tzinfo=et))

        days = self.trading.get_calendar(GetCalendarRequest(start=start, end=end))
        return [{"date": d.date.isoformat(), "open": stamp(d.open), "close": stamp(d.close)} for d in days]

    def positions(self) -> list[dict]:
        return [position_to_dict(p) for p in self.trading.get_all_positions()]

    def orders(self, status: str = "open", limit: int = 100, after: datetime | None = None,
               symbols: list[str] | None = None) -> list[dict]:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        req = GetOrdersRequest(status=QueryOrderStatus(status), limit=limit, after=after, nested=True,
                               symbols=symbols)
        return [order_to_dict(o) for o in self.trading.get_orders(req)]

    def asset(self, symbol: str) -> dict | None:
        try:
            a = self.trading.get_asset(symbol)
        except Exception:
            return None
        return {"symbol": a.symbol, "name": a.name, "exchange": _enum(a.exchange), "tradable": bool(a.tradable),
                "shortable": bool(a.shortable), "easy_to_borrow": bool(a.easy_to_borrow),
                "fractionable": bool(a.fractionable), "status": _enum(a.status)}

    def all_assets(self) -> list[dict]:
        from alpaca.trading.enums import AssetClass, AssetStatus
        from alpaca.trading.requests import GetAssetsRequest

        assets = self.trading.get_all_assets(GetAssetsRequest(status=AssetStatus.ACTIVE,
                                                              asset_class=AssetClass.US_EQUITY))
        return [{"symbol": a.symbol, "name": a.name, "exchange": _enum(a.exchange), "tradable": bool(a.tradable),
                 "shortable": bool(a.shortable), "easy_to_borrow": bool(a.easy_to_borrow),
                 "fractionable": bool(a.fractionable)} for a in assets]

    def portfolio_history(self, period: str = "1M", timeframe: str | None = None) -> dict:
        from alpaca.trading.requests import GetPortfolioHistoryRequest

        tf = timeframe or {"1D": "5Min", "1W": "1H", "1M": "1D", "3M": "1D", "1A": "1D"}.get(period, "1D")
        h = self.trading.get_portfolio_history(GetPortfolioHistoryRequest(period=period, timeframe=tf))
        return {"timestamp": [datetime.fromtimestamp(t, UTC).isoformat() for t in (h.timestamp or [])],
                "equity": [_f(v) for v in (h.equity or [])],
                "profit_loss": [_f(v) for v in (h.profit_loss or [])],
                "base_value": _f(h.base_value), "timeframe": tf}

    # ---------------------------------------------------------------- prices
    def _feed(self):
        from alpaca.data.enums import DataFeed

        return DataFeed.SIP if self.data_feed == "sip" else DataFeed.IEX

    def latest_price(self, symbol: str) -> float | None:
        from alpaca.data.requests import StockLatestQuoteRequest, StockLatestTradeRequest

        try:
            trades = self.data.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=symbol, feed=self._feed()))
            t = trades.get(symbol) if isinstance(trades, dict) else None
            if t is not None and _f(t.price):
                return _f(t.price)
        except Exception as exc:
            log.debug("latest trade failed for %s: %s", symbol, exc)
        try:
            quotes = self.data.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=symbol, feed=self._feed()))
            q = quotes.get(symbol) if isinstance(quotes, dict) else None
            if q is not None:
                bid, ask = _f(q.bid_price) or 0, _f(q.ask_price) or 0
                if bid and ask:
                    return round((bid + ask) / 2, 4)
                return ask or bid or None
        except Exception as exc:
            log.debug("latest quote failed for %s: %s", symbol, exc)
        return None

    def bars(self, symbol: str, start: datetime, end: datetime, timeframe: str = "1Min") -> list[dict]:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        tf = {"1Min": TimeFrame(1, TimeFrameUnit.Minute), "5Min": TimeFrame(5, TimeFrameUnit.Minute),
              "1Hour": TimeFrame(1, TimeFrameUnit.Hour), "1Day": TimeFrame(1, TimeFrameUnit.Day)}[timeframe]
        res = self.data.get_stock_bars(StockBarsRequest(symbol_or_symbols=symbol, start=start, end=end,
                                                        timeframe=tf, feed=self._feed()))
        rows = res.data.get(symbol, []) if hasattr(res, "data") else []
        return [{"t": _iso(b.timestamp), "o": _f(b.open), "h": _f(b.high), "l": _f(b.low), "c": _f(b.close),
                 "v": _f(b.volume)} for b in rows]

    def news(self, start: datetime, end: datetime, symbols: list[str] | None = None, limit: int = 200) -> list[dict]:
        """Historical Benzinga news (oldest first), used by backtests."""
        from alpaca.data.historical.news import NewsClient
        from alpaca.data.requests import NewsRequest

        client = NewsClient(self.creds.api_key, self.creds.secret_key)
        req = NewsRequest(start=start, end=end, symbols=",".join(symbols) if symbols else None, limit=limit,
                          sort="asc", include_content=True)
        res = client.get_news(req)
        return [n.model_dump() for n in res.data.get("news", [])]

    # ---------------------------------------------------------------- orders
    def submit_bracket(self, symbol: str, side: str, qty: int, take_profit: float, stop_loss: float,
                       client_order_id: str) -> dict:
        from alpaca.trading.enums import OrderClass, OrderSide, OrderType, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest, StopLossRequest, TakeProfitRequest

        req = MarketOrderRequest(
            symbol=symbol, qty=qty, side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            type=OrderType.MARKET,
            # GTC so the stop-loss / take-profit legs stay active overnight (DAY legs would expire at close)
            time_in_force=TimeInForce.GTC,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=take_profit),
            stop_loss=StopLossRequest(stop_price=stop_loss),
            client_order_id=client_order_id,
        )
        return order_to_dict(self.trading.submit_order(req))

    def cancel_order(self, order_id: str) -> None:
        self.trading.cancel_order_by_id(order_id)

    def cancel_orders_for(self, symbol: str) -> int:
        open_orders = self.orders("open", symbols=[symbol])
        n = 0
        for o in open_orders:
            try:
                self.trading.cancel_order_by_id(o["id"])
                n += 1
            except Exception as exc:
                log.warning("Couldn't cancel order %s for %s: %s", o["id"], symbol, exc)
        return n

    def close_position(self, symbol: str, wait_seconds: float = 8.0) -> dict:
        """Cancel the symbol's open orders (bracket legs hold the shares), then close at market."""
        self.cancel_orders_for(symbol)
        deadline = time.time() + wait_seconds
        last_exc: Exception | None = None
        while time.time() < deadline:
            try:
                return order_to_dict(self.trading.close_position(symbol))
            except Exception as exc:  # legs may still be cancelling
                last_exc = exc
                time.sleep(0.5)
        raise RuntimeError(f"Couldn't close {symbol}: {last_exc}")

    def cancel_all_orders(self) -> int:
        res = self.trading.cancel_orders()
        return len(res or [])

    def close_all_positions(self) -> int:
        res = self.trading.close_all_positions(cancel_orders=True)
        return len(res or [])

    def recent_orders(self, days: int = 7) -> list[dict]:
        return self.orders("all", limit=500, after=datetime.now(UTC) - timedelta(days=days))
