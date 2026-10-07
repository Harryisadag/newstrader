"""Risk controls - pure functions, no network, fully unit-tested.

`check_entry` decides whether a new position (buy, or short if enabled) may be opened and how big.
`check_exit` decides whether an existing long position may be closed on a bearish signal.
Each returns a RiskDecision with a plain-English reason when something is blocked.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

from ..config import RiskSettings, TradingSettings


@dataclass
class AccountSnapshot:
    equity: float
    last_equity: float
    buying_power: float
    cash: float = 0.0
    shorting_enabled: bool = False
    trading_blocked: bool = False

    @property
    def day_pl(self) -> float:
        return self.equity - self.last_equity


@dataclass
class PositionInfo:
    symbol: str
    qty: float  # negative for shorts
    market_value: float  # negative for shorts

    @property
    def is_long(self) -> bool:
        return self.qty > 0

    @property
    def is_short(self) -> bool:
        return self.qty < 0


@dataclass
class AssetInfo:
    tradable: bool = True
    shortable: bool = False
    easy_to_borrow: bool = False


@dataclass
class RiskContext:
    trading: TradingSettings
    risk: RiskSettings
    account: AccountSnapshot
    positions: list[PositionInfo]
    market_open: bool
    now: datetime
    kill_engaged: bool = False
    halted_today: bool = False
    pending_symbols: set[str] = field(default_factory=set)  # symbols with an unfilled entry order
    last_entry_at: dict[str, datetime] = field(default_factory=dict)
    asset: AssetInfo | None = None

    def position(self, symbol: str) -> PositionInfo | None:
        return next((p for p in self.positions if p.symbol == symbol and p.qty != 0), None)


@dataclass
class RiskDecision:
    allowed: bool
    reason: str = ""
    qty: int = 0
    price: float = 0.0
    notional: float = 0.0
    stop_price: float | None = None
    take_profit_price: float | None = None
    halt_for_day: bool = False  # daily loss limit hit - caller should halt trading today

    @classmethod
    def block(cls, reason: str, **kw) -> RiskDecision:
        return cls(allowed=False, reason=reason, **kw)


def _round_price(value: float) -> float:
    # Alpaca: 2 decimals for prices >= $1, up to 4 below $1
    return round(value, 2) if value >= 1 else round(value, 4)


def bracket_prices(side: str, price: float, stop_loss_pct: float, take_profit_pct: float) -> tuple[float, float]:
    """(stop_price, take_profit_price) for a bracket order, at least one tick away from the entry."""
    tick = 0.01 if price >= 1 else 0.0001
    if side == "buy":
        stop = _round_price(price * (1 - stop_loss_pct / 100))
        tp = _round_price(price * (1 + take_profit_pct / 100))
        stop = min(stop, _round_price(price - tick))
        tp = max(tp, _round_price(price + tick))
    else:  # short
        stop = _round_price(price * (1 + stop_loss_pct / 100))
        tp = _round_price(price * (1 - take_profit_pct / 100))
        stop = max(stop, _round_price(price + tick))
        tp = min(tp, _round_price(price - tick))
    return stop, tp


def _common_gates(ctx: RiskContext, symbol: str, manual: bool, is_exit: bool) -> RiskDecision | None:
    if ctx.kill_engaged:
        return RiskDecision.block("Kill switch is engaged - re-arm trading to resume.")
    if not manual and not ctx.trading.auto_trade:
        return RiskDecision.block("Auto-trade is off (Settings -> Trading).")
    if ctx.account.trading_blocked:
        return RiskDecision.block("Alpaca reports this account is blocked from trading.")
    if not is_exit:
        if ctx.halted_today:
            return RiskDecision.block("Trading is halted for today (daily loss limit was hit).")
        loss = -ctx.account.day_pl
        if loss >= ctx.risk.daily_loss_limit_usd:
            return RiskDecision.block(
                f"Daily loss limit hit: down ${loss:,.2f} today (limit ${ctx.risk.daily_loss_limit_usd:,.2f}).",
                halt_for_day=True)
    if ctx.risk.market_hours_only and not ctx.market_open:
        return RiskDecision.block("Market is closed (market-hours-only is on).")
    if not is_exit:
        if ctx.risk.whitelist and symbol not in ctx.risk.whitelist:
            return RiskDecision.block(f"{symbol} is not on your whitelist.")
        if symbol in ctx.risk.blacklist:
            return RiskDecision.block(f"{symbol} is on your blacklist.")
    return None


def check_entry(symbol: str, side: str, price: float | None, ctx: RiskContext, manual: bool = False) -> RiskDecision:
    """side: 'buy' (open/add long) or 'short' (open short)."""
    symbol = symbol.upper()
    blocked = _common_gates(ctx, symbol, manual, is_exit=False)
    if blocked:
        return blocked

    if ctx.asset is not None and not ctx.asset.tradable:
        return RiskDecision.block(f"Alpaca says {symbol} isn't tradable.")

    cooldown = ctx.risk.ticker_cooldown_minutes
    last = ctx.last_entry_at.get(symbol)
    if cooldown and last is not None:
        mins = (ctx.now - last).total_seconds() / 60
        if mins < cooldown:
            return RiskDecision.block(f"Cooldown: {symbol} was traded {mins:.0f} min ago (cooldown {cooldown} min).")

    if symbol in ctx.pending_symbols:
        return RiskDecision.block(f"An order for {symbol} is already waiting to fill.")

    existing = ctx.position(symbol)
    if side == "buy":
        if existing is not None and existing.is_short:
            return RiskDecision.block(f"You're short {symbol}; not opening a long on top of a short.")
    elif side == "short":
        if not ctx.trading.allow_shorting:
            return RiskDecision.block("Short selling is off (Settings -> Trading).")
        if not ctx.account.shorting_enabled:
            return RiskDecision.block("Your Alpaca account doesn't have shorting enabled.")
        if ctx.asset is not None and not (ctx.asset.shortable and ctx.asset.easy_to_borrow):
            return RiskDecision.block(f"{symbol} isn't easy to borrow, so it can't be shorted.")
        if existing is not None and existing.is_long:
            return RiskDecision.block(f"You hold {symbol} long; sell it instead of shorting.")
    else:
        return RiskDecision.block(f"Unknown order side '{side}'.")

    held = {p.symbol for p in ctx.positions if p.qty != 0} | set(ctx.pending_symbols)
    if symbol not in held and len(held) >= ctx.risk.max_open_positions:
        return RiskDecision.block(f"Max open positions reached ({ctx.risk.max_open_positions}).")

    if price is None or not math.isfinite(price) or price <= 0:
        return RiskDecision.block(f"No current price for {symbol}.")
    if price < ctx.risk.min_share_price:
        return RiskDecision.block(f"{symbol} trades at ${price:.4f}, below the ${ctx.risk.min_share_price:.2f} minimum.")

    equity = max(ctx.account.equity, 0.0)
    exposure = abs(existing.market_value) if existing is not None else 0.0
    room_in_stock = equity * ctx.risk.max_pct_per_stock / 100 - exposure
    if room_in_stock <= 0:
        return RiskDecision.block(
            f"Already at the {ctx.risk.max_pct_per_stock:g}% max for {symbol} (${exposure:,.0f} held).")
    budget = min(ctx.risk.max_dollars_per_trade, room_in_stock, max(ctx.account.buying_power, 0.0) * 0.98)
    if budget <= 0:
        return RiskDecision.block("Not enough buying power.")
    qty = math.floor(budget / price)
    if qty < 1:
        limiter = ("max $ per trade" if budget == ctx.risk.max_dollars_per_trade
                   else "max % per stock" if budget == room_in_stock else "buying power")
        return RiskDecision.block(f"One share of {symbol} (${price:,.2f}) costs more than your {limiter} allows "
                                  f"(${budget:,.2f}). Bracket orders need whole shares.")

    stop, tp = bracket_prices(side, price, ctx.trading.stop_loss_pct, ctx.trading.take_profit_pct)
    return RiskDecision(allowed=True, qty=qty, price=price, notional=round(qty * price, 2),
                        stop_price=stop, take_profit_price=tp,
                        reason=f"{side.upper()} {qty} {symbol} @ ~${price:,.2f} (${qty * price:,.2f}); "
                               f"stop ${stop:,.2f}, target ${tp:,.2f}")


def check_exit(symbol: str, ctx: RiskContext, manual: bool = False) -> RiskDecision:
    """Close an existing long position (used for bearish signals)."""
    symbol = symbol.upper()
    blocked = _common_gates(ctx, symbol, manual, is_exit=True)
    if blocked:
        return blocked
    if not manual and not ctx.trading.sell_on_bearish:
        return RiskDecision.block("Sell-on-bearish is off (Settings -> Trading).")
    pos = ctx.position(symbol)
    if pos is None or not pos.is_long:
        return RiskDecision.block(f"No long {symbol} position to sell.")
    return RiskDecision(allowed=True, qty=math.floor(pos.qty), notional=pos.market_value,
                        reason=f"SELL all {pos.qty:g} {symbol} (bearish signal)")
