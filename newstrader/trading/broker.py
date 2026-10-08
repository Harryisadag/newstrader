"""Thin wrapper around alpaca-py. Returns plain dicts so the rest of the app never touches SDK objects.

All methods are blocking (HTTP calls) - call them with asyncio.to_thread from async code.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, date, datetime, timedelta
from typing import Any

from ..state import MARKET_TZ, AppState
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


def _patient(client: Any) -> None:
    """alpaca-py retries a rate-limited (429) request only 3 times, 3 s apart; long training downloads need more."""
    try:
        client._retry = max(int(getattr(client, "_retry", 3)), 6)
        client._retry_wait = max(int(getattr(client, "_retry_wait", 3)), 5)
    except Exception:
        pass


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


SNAPSHOT_CHUNK = 100  # symbols per snapshot request


def _bar(b: Any) -> dict | None:
    if b is None:
        return None
    return {"t": _iso(b.timestamp), "o": _f(b.open), "h": _f(b.high), "l": _f(b.low), "c": _f(b.close),
            "v": _f(b.volume)}


def snapshot_to_dict(snap: Any) -> dict:
    """One Alpaca snapshot as a plain dict, with the day change worked out.

    Before the open (pre-market) the latest daily bar is still the previous session's, so the change is measured
    against that bar's close; once today's daily bar exists it is measured against the previous day's close."""
    trade = getattr(snap, "latest_trade", None)
    minute, daily, prev = (_bar(getattr(snap, "minute_bar", None)), _bar(getattr(snap, "daily_bar", None)),
                           _bar(getattr(snap, "previous_daily_bar", None)))
    price = _f(getattr(trade, "price", None)) if trade is not None else None
    trade_time = getattr(trade, "timestamp", None) if trade is not None else None
    if price is None:
        price = (minute or {}).get("c") or (daily or {}).get("c")
    daily_day = getattr(getattr(snap, "daily_bar", None), "timestamp", None)
    new_session = (trade_time is not None and daily_day is not None
                   and _market_date(trade_time) > _market_date(daily_day))
    if new_session:
        prev_close, day_volume = (daily or {}).get("c"), None
    else:
        prev_close, day_volume = (prev or {}).get("c"), (daily or {}).get("v")
    change = (price - prev_close) / prev_close * 100 if price and prev_close else None
    return {"price": price, "prev_close": prev_close, "change_pct": change, "day_volume": day_volume,
            "time": _iso(trade_time), "minute_bar": minute, "daily_bar": daily, "prev_daily_bar": prev}


def _market_date(value: datetime) -> date:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(MARKET_TZ).date()


def blocked_reason(a) -> str:
    """Which of Alpaca's three block flags is set, in words, with what to do about it ("" if none)."""
    if a.trade_suspended_by_user:
        return ("Trading is paused in your Alpaca account settings ('suspend trading' is on). "
                "Turn it off in the Alpaca dashboard (Account -> Configure), then try again.")
    if a.account_blocked:
        return "Alpaca has blocked this account. Check the Alpaca dashboard for a notice, or contact Alpaca support."
    if a.trading_blocked:
        return ("Alpaca has blocked trading on this account. For a paper account, reset it in the Alpaca dashboard "
                "(or make a new paper account) and paste its new keys in Settings -> API keys.")
    return ""

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
        _patient(self.data)
        self._news = None
        self._screener = None

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
            "blocked_reason": blocked_reason(a),
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

    def bars_multi(self, symbols: list[str], start: datetime, end: datetime, timeframe: str = "1Min",
                   feed: str | None = None) -> dict[str, list[dict]]:
        """Bars for several symbols in one request (the SDK follows the pages). Used to label training data and
        by the market monitor.

        feed="sip" asks for the full consolidated tape, which free accounts may use for history that ended more
        than 15 minutes ago (denser than IEX for smaller stocks)."""
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        tf = {"1Min": TimeFrame(1, TimeFrameUnit.Minute), "5Min": TimeFrame(5, TimeFrameUnit.Minute),
              "1Hour": TimeFrame(1, TimeFrameUnit.Hour), "1Day": TimeFrame(1, TimeFrameUnit.Day)}[timeframe]
        chosen = {"sip": DataFeed.SIP, "iex": DataFeed.IEX}.get(feed or "", None) or self._feed()
        res = self.data.get_stock_bars(StockBarsRequest(symbol_or_symbols=list(symbols), start=start, end=end,
                                                        timeframe=tf, feed=chosen))
        data = res.data if hasattr(res, "data") else {}
        return {sym: [{"t": _iso(b.timestamp), "o": _f(b.open), "h": _f(b.high), "l": _f(b.low), "c": _f(b.close),
                       "v": _f(b.volume)} for b in rows] for sym, rows in data.items()}

    def snapshots(self, symbols: list[str], feed: str | None = None) -> dict[str, dict]:
        """Latest price and today's / the previous day's bars for many symbols (one request per 100 symbols).

        Returns {symbol: {"price", "prev_close", "change_pct", "day_volume", "time", "minute_bar", "daily_bar",
        "prev_daily_bar"}}. Symbols Alpaca has no data for are left out."""
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockSnapshotRequest

        chosen = {"sip": DataFeed.SIP, "iex": DataFeed.IEX}.get(feed or "", None) or self._feed()
        wanted = list(dict.fromkeys(s.upper() for s in symbols if s))
        out: dict[str, dict] = {}
        for i in range(0, len(wanted), SNAPSHOT_CHUNK):
            res = self.data.get_stock_snapshot(StockSnapshotRequest(symbol_or_symbols=wanted[i:i + SNAPSHOT_CHUNK],
                                                                    feed=chosen))
            if isinstance(res, dict):
                out.update({sym: snapshot_to_dict(snap) for sym, snap in res.items() if snap is not None})
        return out

    def _screener_client(self):
        from alpaca.data.historical.screener import ScreenerClient

        if self._screener is None:
            self._screener = ScreenerClient(self.creds.api_key, self.creds.secret_key)
            _patient(self._screener)
        return self._screener

    def movers(self, top: int = 20) -> dict:
        """Today's biggest gainers and losers (Alpaca screener; may be unavailable on some accounts - raises)."""
        from alpaca.data.requests import MarketMoversRequest

        res = self._screener_client().get_market_movers(MarketMoversRequest(top=top))

        def row(m: Any) -> dict:
            return {"symbol": m.symbol, "price": _f(m.price), "change": _f(m.change),
                    "change_pct": _f(m.percent_change)}

        return {"gainers": [row(m) for m in res.gainers or []], "losers": [row(m) for m in res.losers or []],
                "updated": _iso(res.last_updated)}

    def most_actives(self, top: int = 20, by: str = "volume") -> dict:
        """Today's most traded stocks, by share volume or by number of trades (Alpaca screener; may raise)."""
        from alpaca.data.enums import MostActivesBy
        from alpaca.data.requests import MostActivesRequest

        res = self._screener_client().get_most_actives(MostActivesRequest(top=top, by=MostActivesBy(by)))
        return {"items": [{"symbol": a.symbol, "volume": _f(a.volume), "trade_count": _f(a.trade_count)}
                          for a in res.most_actives or []],
                "updated": _iso(res.last_updated)}

    def _news_client(self):
        from alpaca.data.historical.news import NewsClient

        if self._news is None:
            self._news = NewsClient(self.creds.api_key, self.creds.secret_key)
            _patient(self._news)
        return self._news

    def news(self, start: datetime, end: datetime, symbols: list[str] | None = None, limit: int = 200,
             include_content: bool = True, newest_first: bool = False) -> list[dict]:
        """Historical Benzinga news (oldest first, or newest first), used by backtests, model training and to find
        the story behind a price spike."""
        from alpaca.data.requests import NewsRequest

        req = NewsRequest(start=start, end=end, symbols=",".join(symbols) if symbols else None, limit=limit,
                          sort="desc" if newest_first else "asc", include_content=include_content)
        res = self._news_client().get_news(req)
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
