"""Crypto (paper) through Alpaca: symbols, names, and the words news uses for each coin.

Alpaca writes a crypto pair two ways: "BTC/USD" (orders, market data, the asset list) and "BTCUSD" (positions and
news tags). NewsTrader always uses "BTC/USD": normalize() turns either form into it, and api_symbol() gives the
"BTCUSD" form the positions endpoints need.

Crypto trades 24/7, can be bought in fractions or by dollar amount, can't be shorted on Alpaca and has no bracket
orders. So each buy gets a separate safety stop held at Alpaca and a take-profit NewsTrader watches itself
(crypto_exits.py).
"""

from __future__ import annotations

import math
import re

DEFAULT_SYMBOLS = ["BTC/USD", "ETH/USD", "SOL/USD"]
QUOTE = "USD"  # only coins priced in US dollars are traded (the risk limits are in dollars)
EXCHANGE = "CRYPTO"  # how crypto rows are marked in the ticker table
STOP_LIMIT_GAP_PCT = 0.5  # the safety stop's limit price sits this far below its stop price

_PAIR = re.compile(r"^([A-Z0-9]{2,10})/(USDT|USDC|USD|BTC)$")
_JOINED = re.compile(r"^([A-Z0-9]{2,10}?)(USDT|USDC|USD)$")
_JOINED_ANY = re.compile(r"^([A-Z0-9]{2,10}?)(USDT|USDC|USD|BTC)$")

NAMES = {
    "BTC": "Bitcoin", "ETH": "Ethereum", "SOL": "Solana", "DOGE": "Dogecoin", "LTC": "Litecoin", "XRP": "XRP",
    "AVAX": "Avalanche", "LINK": "Chainlink", "DOT": "Polkadot", "ADA": "Cardano", "SHIB": "Shiba Inu",
    "BCH": "Bitcoin Cash", "UNI": "Uniswap", "AAVE": "Aave", "USDC": "USD Coin", "USDT": "Tether",
}

# Words news uses for a coin -> its symbol (without "/USD"). Always counted wherever they appear...
ALIASES: dict[str, str] = {
    "bitcoin": "BTC", "btc": "BTC", "ether": "ETH", "ethereum": "ETH", "solana": "SOL", "dogecoin": "DOGE",
    "litecoin": "LTC", "cardano": "ADA", "polkadot": "DOT", "chainlink": "LINK", "shiba inu": "SHIB",
    "bitcoin cash": "BCH", "uniswap": "UNI",
}
# ...and short forms that are also everyday words or other things ("Sol", "ETH Zurich", "link", "dot"), only counted
# when the story is about crypto (CONTEXT words).
CONTEXT_ALIASES: dict[str, str] = {
    "eth": "ETH", "sol": "SOL", "doge": "DOGE", "xrp": "XRP", "ltc": "LTC", "avax": "AVAX", "avalanche": "AVAX",
    "link": "LINK", "ada": "ADA", "dot": "DOT", "shib": "SHIB", "bch": "BCH", "uni": "UNI", "aave": "AAVE",
}
CONTEXT: set[str] = {
    "crypto", "cryptocurrency", "cryptocurrencies", "coin", "coins", "token", "tokens", "blockchain", "defi",
    "altcoin", "altcoins", "stablecoin", "stablecoins", "memecoin", "memecoins", "bitcoin", "btc", "ethereum", "ether",
    "solana", "staking", "staked", "web3", "on-chain", "onchain", "nft", "nfts", "satoshi", "halving", "mainnet",
    "coinbase", "binance", "kraken", "bybit", "okx", "bitstamp", "ftx", "crypto-friendly",
}


def normalize(symbol, asset_class=None) -> str:
    """'BTCUSD' / 'btc/usd' / '$BTC/USD' -> 'BTC/USD'; any stock ticker comes back upper-case.

    A joined pair is only split when it can't be a stock (US tickers have at most 5 letters) or Alpaca says it's
    crypto (asset_class)."""
    s = str(symbol or "").strip().upper().lstrip("$")
    if not s or "/" in s:
        return s
    if "crypto" in str(getattr(asset_class, "value", asset_class) or "").lower():
        m = _JOINED_ANY.match(s)
        return f"{m.group(1)}/{m.group(2)}" if m else s
    m = _JOINED.match(s) if len(s) >= 6 else None
    return f"{m.group(1)}/{m.group(2)}" if m else s


def is_crypto(symbol) -> bool:
    return bool(_PAIR.match(normalize(symbol)))


def base(symbol) -> str:
    """'BTC/USD' -> 'BTC'."""
    return normalize(symbol).split("/", 1)[0]


def api_symbol(symbol) -> str:
    """'BTC/USD' -> 'BTCUSD' (Alpaca's positions endpoints want it without the slash)."""
    return normalize(symbol).replace("/", "")


def pair(base_symbol: str) -> str:
    return f"{base_symbol.upper()}/{QUOTE}"


def display_name(symbol, fallback: str = "") -> str:
    """'Bitcoin' for BTC/USD; Alpaca's own name ('Bitcoin / US Dollar' -> 'Bitcoin') for coins not listed here."""
    known = NAMES.get(base(symbol))
    if known:
        return known
    name = re.sub(r"\s*/\s*(?:US Dollar|USD|Tether|USD Coin|Bitcoin)\s*$", "", str(fallback or "")).strip()
    return name or base(symbol)


def terms(symbol) -> list[str]:
    """Every word or phrase news uses for this coin (lower-case), longest first."""
    b = base(symbol)
    found = {p for p, s in {**ALIASES, **CONTEXT_ALIASES}.items() if s == b} | {b.lower()}
    name = NAMES.get(b)
    if name:
        found.add(name.lower())
    return sorted(found, key=lambda t: (-len(t), t))


def parse_symbols(value) -> list[str]:
    """'btc, ETHUSD; SOL/USD' -> ['BTC/USD', 'ETH/USD', 'SOL/USD'] (a bare 'BTC' means BTC/USD). Raises ValueError for
    anything that isn't a crypto pair priced in US dollars."""
    if isinstance(value, str):
        value = re.split(r"[\s,;]+", value)
    out: list[str] = []
    for item in value or []:
        raw = str(item).strip()
        if not raw:
            continue
        s = normalize(raw)
        if "/" not in s and re.fullmatch(r"[A-Z0-9]{2,10}", s):
            s = pair(s)
        m = _PAIR.match(s)
        if not m:
            raise ValueError(f"'{raw}' doesn't look like a crypto pair (e.g. BTC/USD)")
        if m.group(2) != QUOTE:
            raise ValueError(f"'{raw}': only crypto priced in US dollars (like BTC/USD) can be traded")
        if s not in out:
            out.append(s)
    return out


def round_down(value: float, increment: float | None, decimals: int = 9) -> float:
    """value rounded DOWN to a multiple of increment (to `decimals` places when there is none)."""
    if value is None or not math.isfinite(value) or value <= 0:
        return 0.0
    if increment and increment > 0:
        steps = math.floor(value / increment + 1e-9)
        out = steps * increment
        places = max(0, -int(math.floor(math.log10(increment)))) if increment < 1 else 0
        return round(out, min(places + 1, 12))
    factor = 10 ** decimals
    return math.floor(value * factor + 1e-6) / factor


def price_decimals(price: float) -> int:
    """Decimals for an order price when Alpaca doesn't say: cents above $1, more below (tiny coins)."""
    if price >= 1:
        return 2
    if price >= 0.01:
        return 4
    return 8


def round_price(value: float, increment: float | None = None) -> float:
    """A stop / limit price rounded down to the coin's price step (cents when unknown)."""
    if increment and increment > 0:
        return round_down(value, increment)
    d = price_decimals(value)
    return math.floor(value * 10 ** d + 1e-6) / 10 ** d


def exit_prices(entry: float, stop_loss_pct: float, take_profit_pct: float,
                increment: float | None = None) -> tuple[float, float, float]:
    """(stop price, the stop's limit price, take-profit price) for a crypto position bought at `entry`."""
    stop = round_price(entry * (1 - stop_loss_pct / 100), increment)
    limit = round_price(stop * (1 - STOP_LIMIT_GAP_PCT / 100), increment)
    target = entry * (1 + take_profit_pct / 100)
    d = price_decimals(target)
    target = math.ceil(target * 10 ** d - 1e-6) / 10 ** d
    return stop, limit, target
