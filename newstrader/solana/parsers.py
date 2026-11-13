"""What each data source's answer means. Pure functions: JSON in, plain values out.

A source that answers with an error, a 429 or something unusable raises DataError (RateLimited, NoRoute), so the
gates can fail closed on it. A field that is missing stays "missing" - it is never guessed.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from urllib.parse import urlsplit

SOL_MINT = "So11111111111111111111111111111111111111112"
SYSTEM_PROGRAM = "11111111111111111111111111111111"  # "nobody" / burned
LAMPORTS = 1_000_000_000  # 1 SOL
ROUTE_ERRORS = {"NO_ROUTES_FOUND", "TOKEN_NOT_TRADABLE", "COULD_NOT_FIND_ANY_ROUTE", "ROUTE_PLAN_DOES_NOT_CONSUME_ALL_THE_AMOUNT",
                "CIRCULAR_ARBITRAGE_IS_DISABLED"}
_ADDRESS = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{32,44}$")


class DataError(Exception):
    """A data source didn't give a usable answer."""


class RateLimited(DataError):
    """The source said "too many requests" (or is still pausing after saying it)."""


class NoRoute(DataError):
    """Jupiter found no way to make that swap right now (NO_ROUTES_FOUND, TOKEN_NOT_TRADABLE...)."""

    def __init__(self, code: str, message: str = ""):
        super().__init__(message or code)
        self.code = code


def is_address(value) -> bool:
    """A Solana address (base58, 32-44 characters). Addresses from the feeds are checked before they go into a URL."""
    return isinstance(value, str) and bool(_ADDRESS.match(value))


def num(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out and abs(out) != float("inf") else None


def integer(value) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        f = num(value)
        return int(f) if f is not None else None


def when(value) -> datetime | None:
    """An ISO time or a unix time (seconds or milliseconds) -> an aware UTC datetime."""
    if value is None or value == "" or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        ts = float(value) / (1000 if value > 1e12 else 1)
        return datetime.fromtimestamp(ts, UTC) if ts > 0 else None
    text = str(value).strip()
    if re.fullmatch(r"\d+(\.\d+)?", text):
        return when(float(text))
    text = re.sub(r"(\.\d{6})\d+", r"\1", text.replace("Z", "+00:00"))
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


# ---------------------------------------------------------------------------------------------- GeckoTerminal
@dataclass
class Pool:
    """One pool from GeckoTerminal (prices in US dollars, numbers that weren't there are None)."""

    address: str
    name: str = ""
    mint: str = ""
    quote_mint: str = ""
    dex: str = ""
    symbol: str = ""
    decimals: int | None = None
    created_at: datetime | None = None
    price_usd: float | None = None
    quote_price_usd: float | None = None
    price_native: float | None = None
    fdv_usd: float | None = None
    market_cap_usd: float | None = None
    reserve_usd: float | None = None
    change: dict = field(default_factory=dict)
    txns: dict = field(default_factory=dict)
    volume: dict = field(default_factory=dict)
    graduation_pct: float | None = None
    graduated: bool | None = None

    @property
    def price_sol(self) -> float | None:
        """The coin's price in SOL (only for pools paired with SOL)."""
        if self.quote_mint != SOL_MINT:
            return None
        if self.price_native:
            return self.price_native
        if self.price_usd and self.quote_price_usd:
            return self.price_usd / self.quote_price_usd
        return None

    @property
    def sol_usd(self) -> float | None:
        return self.quote_price_usd if self.quote_mint == SOL_MINT else None

    def age_minutes(self, now: datetime) -> float | None:
        return None if self.created_at is None else (now - self.created_at).total_seconds() / 60

    def window(self, name: str = "m5") -> dict:
        return self.txns.get(name) or {}

    def to_dict(self) -> dict:
        return {"address": self.address, "name": self.name, "mint": self.mint, "symbol": self.symbol,
                "dex": self.dex, "created_at": self.created_at.isoformat() if self.created_at else None,
                "price_usd": self.price_usd, "price_sol": self.price_sol, "sol_usd": self.sol_usd,
                "fdv_usd": self.fdv_usd, "reserve_usd": self.reserve_usd, "change": self.change,
                "txns_m5": self.window("m5"), "volume_m5": self.volume.get("m5")}


def gecko_id(value) -> str:
    """'solana_<address>' -> '<address>'."""
    text = str(value or "")
    return text.split("_", 1)[1] if text.startswith("solana_") else text


def check_gecko(payload, status: int = 200) -> None:
    """Raise on GeckoTerminal's error answers ({"status": {"error_code": 429...}} or {"errors": [...]})."""
    if not isinstance(payload, dict):
        raise DataError("GeckoTerminal sent something that isn't JSON data")
    err = payload.get("status") if isinstance(payload.get("status"), dict) else None
    code = integer((err or {}).get("error_code")) or (status if status != 200 else None)
    if code == 429:
        raise RateLimited("GeckoTerminal says too many requests")
    if err and code:
        raise DataError(f"GeckoTerminal error {code}: {str(err.get('error_message') or '')[:120]}")
    if payload.get("errors"):
        first = payload["errors"][0] if isinstance(payload["errors"], list) and payload["errors"] else {}
        raise DataError(f"GeckoTerminal error: {str((first or {}).get('title') or first)[:120]}")
    if status != 200:
        raise DataError(f"GeckoTerminal answered HTTP {status}")


def _pool(item: dict, tokens: dict) -> Pool | None:
    if not isinstance(item, dict):
        return None
    a = item.get("attributes") or {}
    rel = item.get("relationships") or {}
    address = a.get("address") or gecko_id(item.get("id"))
    if not is_address(address):
        return None
    mint = gecko_id(((rel.get("base_token") or {}).get("data") or {}).get("id"))
    quote = gecko_id(((rel.get("quote_token") or {}).get("data") or {}).get("id"))
    token = tokens.get(mint) or {}
    txns = {}
    for win, t in (a.get("transactions") or {}).items():
        if isinstance(t, dict):
            txns[win] = {k: integer(t.get(k)) for k in ("buys", "sells", "buyers", "sellers")}
    lp = a.get("launchpad_details") or {}
    return Pool(
        address=address, name=html.unescape(str(a.get("name") or "")), mint=mint if is_address(mint) else "",
        quote_mint=quote, dex=gecko_id(((rel.get("dex") or {}).get("data") or {}).get("id")),
        symbol=html.unescape(str(token.get("symbol") or "")), decimals=integer(token.get("decimals")),
        created_at=when(a.get("pool_created_at")), price_usd=num(a.get("base_token_price_usd")),
        quote_price_usd=num(a.get("quote_token_price_usd")),
        price_native=num(a.get("base_token_price_quote_token")) if quote == SOL_MINT else None,
        fdv_usd=num(a.get("fdv_usd")), market_cap_usd=num(a.get("market_cap_usd")),
        reserve_usd=num(a.get("reserve_in_usd")),
        change={k: num(v) for k, v in (a.get("price_change_percentage") or {}).items()},
        txns=txns, volume={k: num(v) for k, v in (a.get("volume_usd") or {}).items()},
        graduation_pct=num(lp.get("graduation_percentage")),
        graduated=lp.get("completed") if isinstance(lp.get("completed"), bool) else None)


def _included_tokens(payload: dict) -> dict:
    out = {}
    for inc in payload.get("included") or []:
        if isinstance(inc, dict) and inc.get("type") == "token":
            attrs = inc.get("attributes") or {}
            out[attrs.get("address") or gecko_id(inc.get("id"))] = attrs
    return out


def parse_pools(payload, status: int = 200) -> list[Pool]:
    """GeckoTerminal's new_pools / pools/multi answer (a list), or one pool (a dict) -> pools."""
    check_gecko(payload, status)
    data = payload.get("data")
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise DataError("GeckoTerminal's answer had no pool list")
    tokens = _included_tokens(payload)
    return [p for p in (_pool(item, tokens) for item in data) if p is not None]


def parse_pool(payload, status: int = 200) -> Pool:
    pools = parse_pools(payload, status)
    if not pools:
        raise DataError("GeckoTerminal doesn't know this pool")
    return pools[0]


# ---------------------------------------------------------------------------------------------- Solana network (RPC)
def check_rpc(payload) -> dict:
    """The "result" of a JSON-RPC answer, or raise (a 429 inside the answer counts as one)."""
    if not isinstance(payload, dict):
        raise DataError("The Solana network sent something that isn't JSON data")
    err = payload.get("error")
    if err:
        code = integer((err or {}).get("code")) if isinstance(err, dict) else None
        message = str((err or {}).get("message") if isinstance(err, dict) else err)[:120]
        if code == 429:
            raise RateLimited(f"The Solana network says too many requests ({message})")
        raise DataError(f"The Solana network said: {message}")
    result = payload.get("result")
    if not isinstance(result, dict):
        raise DataError("The Solana network's answer had no result")
    return result


@dataclass
class MintInfo:
    """A coin's mint account: who can still make more of it or freeze it, and its Token-2022 extensions."""

    mint: str
    program: str = ""
    owner: str = ""
    decimals: int | None = None
    supply: int | None = None
    mint_authority: str | None = None
    freeze_authority: str | None = None
    extensions: list = field(default_factory=list)
    missing: tuple = ()  # fields the answer didn't have

    def to_dict(self) -> dict:
        return {"mint": self.mint, "program": self.program, "decimals": self.decimals, "supply": self.supply,
                "mint_authority": self.mint_authority, "freeze_authority": self.freeze_authority,
                "extensions": [e.get("extension") for e in self.extensions], "missing": list(self.missing)}


def parse_mint_account(payload, mint: str) -> MintInfo:
    """getAccountInfo(mint, jsonParsed) -> MintInfo."""
    value = check_rpc(payload).get("value")
    if value is None:
        raise DataError("The Solana network has no coin at that address")
    data = value.get("data") if isinstance(value, dict) else None
    parsed = data.get("parsed") if isinstance(data, dict) else None
    if not isinstance(parsed, dict) or parsed.get("type") != "mint":
        raise DataError("That address isn't a coin (no readable mint account)")
    info = parsed.get("info") or {}
    missing = tuple(k for k in ("mintAuthority", "freezeAuthority", "decimals", "supply") if k not in info)
    program = str(data.get("program") or "")
    extensions = info.get("extensions")
    if extensions is None:
        extensions = []  # a mint without extensions simply has no list
    elif not isinstance(extensions, list):
        missing += ("extensions",)
        extensions = []
    return MintInfo(mint=mint, program=program, owner=str(value.get("owner") or ""),
                    decimals=integer(info.get("decimals")), supply=integer(info.get("supply")),
                    mint_authority=info.get("mintAuthority") or None, freeze_authority=info.get("freezeAuthority") or None,
                    extensions=[e for e in extensions if isinstance(e, dict)], missing=missing)


def parse_largest_accounts(payload) -> list[dict]:
    """getTokenLargestAccounts -> [{"address", "amount"}] (raw amounts)."""
    value = check_rpc(payload).get("value")
    if not isinstance(value, list):
        raise DataError("The Solana network's holder list was empty")
    out = []
    for row in value:
        if isinstance(row, dict) and is_address(row.get("address")) and integer(row.get("amount")) is not None:
            out.append({"address": row["address"], "amount": integer(row["amount"])})
    return out


# ---------------------------------------------------------------------------------------------- Jupiter
@dataclass
class Quote:
    """A Jupiter price quote (nothing is swapped: it only says what a swap would give right now)."""

    input_mint: str
    output_mint: str
    in_amount: int
    out_amount: int
    price_impact: float | None  # a fraction: 0.031 = 3.1%
    labels: list = field(default_factory=list)
    slippage_bps: int | None = None

    def to_dict(self) -> dict:
        return {"in": self.input_mint, "out": self.output_mint, "in_amount": str(self.in_amount),
                "out_amount": str(self.out_amount), "price_impact_pct": None if self.price_impact is None
                else round(self.price_impact * 100, 4), "route": self.labels}


def parse_quote(payload, status: int = 200) -> Quote:
    """Jupiter's quote answer -> Quote. "No route" answers raise NoRoute, other errors DataError."""
    if not isinstance(payload, dict):
        raise DataError("Jupiter sent something that isn't JSON data")
    code = payload.get("errorCode")
    if code or payload.get("error"):
        code = str(code or "")
        message = str(payload.get("error") or code)[:160]
        if code in ROUTE_ERRORS or "route" in message.lower() or "not tradable" in message.lower():
            raise NoRoute(code or "NO_ROUTES_FOUND", message)
        raise DataError(f"Jupiter said: {message}")
    if status != 200:
        raise DataError(f"Jupiter answered HTTP {status}")
    in_amount, out_amount = integer(payload.get("inAmount")), integer(payload.get("outAmount"))
    if in_amount is None or out_amount is None or in_amount <= 0:
        raise DataError("Jupiter's quote had no amounts")
    if out_amount <= 0:
        raise NoRoute("ZERO_OUT", "Jupiter's quote gives nothing back")
    labels = []
    for step in payload.get("routePlan") or []:
        label = ((step or {}).get("swapInfo") or {}).get("label")
        if label and label not in labels:
            labels.append(str(label))
    return Quote(input_mint=str(payload.get("inputMint") or ""), output_mint=str(payload.get("outputMint") or ""),
                 in_amount=in_amount, out_amount=out_amount, price_impact=num(payload.get("priceImpactPct")),
                 labels=labels, slippage_bps=integer(payload.get("slippageBps")))


def parse_prices(payload) -> dict[str, dict]:
    """Jupiter's price answer -> {mint: {"usd", "liquidity", "decimals", "launchpad"}} (unknown coins are left out)."""
    if not isinstance(payload, dict):
        raise DataError("Jupiter's price answer isn't JSON data")
    out = {}
    for mint, row in payload.items():
        if isinstance(row, dict) and num(row.get("usdPrice")):
            out[mint] = {"usd": num(row.get("usdPrice")), "liquidity": num(row.get("liquidity")),
                         "decimals": integer(row.get("decimals")), "launchpad": row.get("launchpad")}
    return out


def parse_token_search(payload, mint: str) -> dict | None:
    """Jupiter's token search -> its audit of this coin, or None when Jupiter doesn't list it (yet)."""
    if not isinstance(payload, list):
        raise DataError("Jupiter's token list answer isn't a list")
    row = next((r for r in payload if isinstance(r, dict) and r.get("id") == mint), None)
    if row is None:
        return None
    audit = row.get("audit") if isinstance(row.get("audit"), dict) else {}
    return {"mint": mint, "dev": row.get("dev"), "launchpad": row.get("launchpad"),
            "holders": integer(row.get("holderCount")), "liquidity": num(row.get("liquidity")),
            "fdv": num(row.get("fdv")), "has_audit": bool(audit),
            "mint_authority_disabled": audit.get("mintAuthorityDisabled"),
            "freeze_authority_disabled": audit.get("freezeAuthorityDisabled"),
            "top_holders_pct": num(audit.get("topHoldersPercentage")), "dev_mints": integer(audit.get("devMints")),
            "organic_score": num(row.get("organicScore")), "organic_label": row.get("organicScoreLabel"),
            "tags": [str(t) for t in row.get("tags") or []]}


# ---------------------------------------------------------------------------------------------- RugCheck
def parse_rugcheck(payload, mint: str) -> dict:
    """RugCheck's full report, checked to be about this coin. The gates read its fields (and fail on missing ones)."""
    if not isinstance(payload, dict):
        raise DataError("RugCheck sent something that isn't JSON data")
    if payload.get("error") or (payload.get("message") and "mint" not in payload):
        raise DataError(f"RugCheck said: {str(payload.get('error') or payload.get('message'))[:120]}")
    if payload.get("mint") != mint:
        raise DataError("RugCheck's report is about a different coin")
    return payload


def no_one(value) -> bool:
    """An authority that is nobody: missing (None), empty, or the system program (burned)."""
    return value in (None, "", SYSTEM_PROGRAM)


# ---------------------------------------------------------------------------------------------- your own RPC address
def rpc_host(url: str) -> str:
    """Only the host of an RPC address is ever shown or logged - the rest of it may hold an access code."""
    try:
        return urlsplit(str(url or "")).hostname or ""
    except ValueError:
        return ""


def check_rpc_url(url) -> str:
    """'' (use the public Solana RPC) or an https:// address of your own; raises ValueError otherwise."""
    text = str(url or "").strip()
    if not text:
        return ""
    if any(c.isspace() for c in text) or not text.lower().startswith("https://") or not rpc_host(text):
        raise ValueError("the RPC address must start with https:// (or leave it empty for the free public one)")
    return text
