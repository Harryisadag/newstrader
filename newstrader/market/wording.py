"""Plain-English names and alert wording for the market monitor (written for someone who isn't a trader)."""

from __future__ import annotations

import re

# US-listed ETFs that track a whole market. Alpaca only trades US stocks, so these stand in for the indexes.
INDEX_NAMES = {
    "SPY": "S&P 500", "VOO": "S&P 500", "IVV": "S&P 500", "QQQ": "Nasdaq 100", "IWM": "Russell 2000",
    "DIA": "Dow Jones", "VTI": "Total US market",
}
WORLD_NAMES = {
    "EWJ": "Japan", "FXI": "China", "MCHI": "China", "EWG": "Germany", "EWU": "UK", "INDA": "India",
    "EWZ": "Brazil", "EZU": "Eurozone", "EWY": "South Korea", "EWC": "Canada", "EWA": "Australia",
    "EWT": "Taiwan", "EWH": "Hong Kong", "EWW": "Mexico", "EWQ": "France", "EWI": "Italy", "EWP": "Spain",
    "EWL": "Switzerland", "EWS": "Singapore", "EZA": "South Africa", "KSA": "Saudi Arabia", "TUR": "Turkey",
    "VGK": "Europe", "EEM": "Emerging markets", "EFA": "Developed markets outside the US",
}
NO_NEWS = "No news found for it yet - it could be a rumour, a big order, or news we haven't seen."
_NAME_TAIL = re.compile(r"\s+(class [a-z]\b.*|common stock.*|ordinary shares?.*|capital stock.*|common shares?.*|"
                        r"american depositary.*|american depository.*|units?\b.*)$", re.IGNORECASE)


def market_name(symbol: str) -> str:
    """'S&P 500' for SPY, 'Japan' for EWJ, '' for anything else."""
    return INDEX_NAMES.get(symbol) or WORLD_NAMES.get(symbol) or ""


def label(symbol: str) -> str:
    """'S&P 500 (SPY)', 'Japan (EWJ)', or just the ticker."""
    name = market_name(symbol)
    return f"{name} ({symbol})" if name else symbol


def company_name(name: str) -> str:
    """'Apple Inc. Common Stock' -> 'Apple Inc.' (for the movers list)."""
    return _NAME_TAIL.sub("", name or "").strip()[:60]


def _money(v: float | None) -> str:
    return "?" if v is None else f"${v:,.2f}"


def spike_title(symbol: str, kind: str, change_pct: float, window_min: int, volume_ratio: float | None) -> str:
    if kind == "spike_up":
        return f"{symbol} jumped {change_pct:+.1f}% in {window_min} min"
    if kind == "spike_down":
        return f"{symbol} dropped {abs(change_pct):.1f}% in {window_min} min"
    ratio = f"{volume_ratio:.0f}x" if volume_ratio else "much more than"
    return f"Unusual trading in {symbol}: {ratio} its usual volume"


def spike_message(price: float | None, ref_price: float | None, volume_ratio: float | None,
                  headline: str | None) -> str:
    volume = (f" on {volume_ratio:.1f}x its usual volume." if volume_ratio is not None
              else ". There isn't enough trading data yet to compare its volume.")
    news = f"News: {headline}" if headline else NO_NEWS
    return f"Now {_money(price)} (was {_money(ref_price)}){volume} {news}"


def market_window_title(symbol: str, change_pct: float, window_min: int) -> str:
    verb = "rose" if change_pct > 0 else "fell"
    return f"{label(symbol)} {verb} {abs(change_pct):.1f}% in {window_min} minutes"


def market_window_message(price: float | None, ref_price: float | None) -> str:
    return (f"Now {_money(price)} (was {_money(ref_price)}). When the whole market moves this fast, most stocks "
            "tend to move with it.")


def day_level_title(symbol: str, level: float) -> str:
    direction = "up" if level > 0 else "down"
    return f"{label(symbol)} is now {direction} {abs(level):g}% today"


def day_level_message(symbol: str, price: float | None, prev_close: float | None, change_pct: float | None,
                      world: bool) -> str:
    change = f"{change_pct:+.1f}%" if change_pct is not None else "?"
    msg = f"Now {_money(price)}, {change} from yesterday's close ({_money(prev_close)})."
    if world and market_name(symbol):
        msg += (f" {symbol} is a US-listed fund that holds stocks from {market_name(symbol)}, so it shows how that "
                "market is doing.")
    return msg
