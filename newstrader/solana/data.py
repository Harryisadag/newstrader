"""The desk's data sources: free, public and read-only. Nothing here can trade - it only asks questions.

- GeckoTerminal: the newest pools, one pool, or up to 30 pools in one call. About 30 calls a minute for everyone,
  so the desk stays under 25 (one shared limiter for the feed, the checklist and the follow-ups).
- The Solana network: a coin's mint account, from the free public RPC or your own (Settings -> Solana). Holder
  lists only come from your own RPC - the public one always answers 429 to that question.
- Jupiter: price quotes, prices and token info. Its free allowance is tiny (about 5 calls per window), so every
  Jupiter call shares one limiter: one call every 1.5 s, and after a 429 none until Jupiter says.
- RugCheck: the full risk report, at most one call a second.

Every problem (no answer, a timeout, a 429, an answer that makes no sense) raises DataError, so the gates fail closed.
"""

from __future__ import annotations

import asyncio
import logging
import time

import httpx

from .limiter import Limiter
from .parsers import (
    SOL_MINT,
    DataError,
    MintInfo,
    NoRoute,
    Pool,
    Quote,
    RateLimited,
    is_address,
    parse_largest_accounts,
    parse_mint_account,
    parse_pool,
    parse_pools,
    parse_prices,
    parse_quote,
    parse_rugcheck,
    parse_token_search,
    rpc_host,
)

log = logging.getLogger(__name__)

GECKO = "https://api.geckoterminal.com/api/v2/networks/solana"
JUPITER_QUOTE = "https://lite-api.jup.ag/swap/v1/quote"
JUPITER_PRICE = "https://lite-api.jup.ag/price/v3"
JUPITER_TOKENS = "https://lite-api.jup.ag/tokens/v2/search"
RUGCHECK = "https://api.rugcheck.xyz/v1/tokens/{}/report"
PUBLIC_RPC = "https://api.mainnet-beta.solana.com"
UA = "NewsTrader paper desk"
TIMEOUT = 10.0
MULTI_MAX = 30  # pools per GeckoTerminal multi-pool call
SOURCES = {"gecko": "GeckoTerminal", "rpc": "The Solana network", "jupiter": "Jupiter", "rugcheck": "RugCheck"}


def _check_address(value: str, what: str = "address") -> str:
    if not is_address(value):
        raise DataError(f"'{str(value)[:50]}' isn't a Solana {what}")
    return value


class SolanaData:
    """All the desk's questions to the outside world. Every call waits for its host's limiter."""

    def __init__(self, client: httpx.AsyncClient | None = None, rpc_url: str = "", clock=time.monotonic,
                 wall=time.time, sleep=asyncio.sleep):
        self._client = client
        self._own_client = client is None
        self.rpc_url = rpc_url
        kw = {"clock": clock, "wall": wall, "sleep": sleep}
        self.gecko = Limiter("GeckoTerminal", 2.0, per_minute=25, **kw)
        self.jupiter = Limiter("Jupiter", 1.5, **kw)
        self.rugcheck_limit = Limiter("RugCheck", 1.0, **kw)
        self.rpc = Limiter("The Solana network", 0.5, **kw)
        self.health: dict[str, dict] = {}
        self._wall = wall

    @property
    def rpc_label(self) -> str:
        return f"your Solana RPC ({rpc_host(self.rpc_url)})" if self.rpc_url else "The Solana network (public RPC)"

    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(headers={"User-Agent": UA, "Accept": "application/json"},
                                             timeout=TIMEOUT, follow_redirects=False)
        return self._client

    async def close(self) -> None:
        if self._own_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    def _note(self, source: str, error: str | None = None, limited: bool = False) -> None:
        h = self.health.setdefault(source, {"calls": 0, "limited": 0, "ok_at": None, "error": None, "error_at": None})
        h["calls"] += 1
        h["limited"] += int(limited)
        if error is None:
            h["ok_at"], h["error"] = self._wall(), None
        else:
            h["error"], h["error_at"] = error, self._wall()

    async def _call(self, source: str, limiter: Limiter, method: str, url: str, name: str, *, params=None,
                    body=None, max_wait: float = 20.0):
        """One request through the host's limiter -> (HTTP status, JSON answer)."""
        try:
            await limiter.acquire(max_wait)
        except RateLimited as exc:
            self._note(source, str(exc), limited=True)
            raise
        try:
            r = await self.client().request(method, url, params=params, json=body, timeout=TIMEOUT)
        except httpx.TimeoutException as exc:
            self._note(source, f"{name} didn't answer within {TIMEOUT:.0f} s")
            raise DataError(f"{name} didn't answer within {TIMEOUT:.0f} s") from exc
        except httpx.HTTPError as exc:
            self._note(source, f"{name} couldn't be reached ({type(exc).__name__})")
            raise DataError(f"{name} couldn't be reached ({type(exc).__name__})") from exc
        if r.status_code == 429:
            pause = limiter.hit_limit(r.headers)
            why = f"{name} says too many requests - waiting {pause:.0f} s before asking again"
            self._note(source, why, limited=True)
            log.info(why)
            raise RateLimited(why)
        try:
            payload = r.json()
        except ValueError as exc:
            self._note(source, f"{name} answered HTTP {r.status_code} without data")
            raise DataError(f"{name} answered HTTP {r.status_code} without data") from exc
        if r.status_code < 400:
            limiter.ok()
        return r.status_code, payload

    def _parsed(self, source: str, fn, *args):
        try:
            out = fn(*args)
        except NoRoute:
            self._note(source)  # the source answered fine: there is just no way to swap
            raise
        except RateLimited as exc:
            self._note(source, str(exc), limited=True)
            raise
        except DataError as exc:
            self._note(source, str(exc))
            raise
        self._note(source)
        return out

    # ------------------------------------------------------------------ GeckoTerminal
    async def new_pools(self, max_wait: float = 30.0) -> list[Pool]:
        status, payload = await self._call("gecko", self.gecko, "GET", f"{GECKO}/new_pools", "GeckoTerminal",
                                           params={"page": 1, "include": "base_token,dex"}, max_wait=max_wait)
        return self._parsed("gecko", parse_pools, payload, status)

    async def pool(self, address: str, max_wait: float = 20.0) -> Pool:
        _check_address(address, "pool")
        status, payload = await self._call("gecko", self.gecko, "GET", f"{GECKO}/pools/{address}", "GeckoTerminal",
                                           params={"include": "base_token"}, max_wait=max_wait)
        return self._parsed("gecko", parse_pool, payload, status)

    async def pools(self, addresses: list[str], max_wait: float = 20.0) -> dict[str, Pool]:
        """Up to 30 pools in one call -> {pool address: Pool} (pools GeckoTerminal no longer knows are left out)."""
        wanted = [_check_address(a, "pool") for a in dict.fromkeys(addresses)]
        if not wanted or len(wanted) > MULTI_MAX:
            raise DataError(f"ask for 1 to {MULTI_MAX} pools at a time")
        status, payload = await self._call("gecko", self.gecko, "GET", f"{GECKO}/pools/multi/{','.join(wanted)}",
                                           "GeckoTerminal", params={"include": "base_token"}, max_wait=max_wait)
        return {p.address: p for p in self._parsed("gecko", parse_pools, payload, status)}

    # ------------------------------------------------------------------ the Solana network
    async def _rpc(self, method: str, params: list, max_wait: float):
        url = self.rpc_url or PUBLIC_RPC
        status, payload = await self._call("rpc", self.rpc, "POST", url, self.rpc_label, max_wait=max_wait,
                                           body={"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
        err = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(err, dict) and err.get("code") == 429:  # "too many requests" inside the answer
            self.rpc.hit_limit()
        if status >= 400 and not err:
            self._note("rpc", f"{self.rpc_label} answered HTTP {status}")
            raise DataError(f"{self.rpc_label} answered HTTP {status}")
        return payload

    async def mint_account(self, mint: str, max_wait: float = 20.0) -> MintInfo:
        _check_address(mint, "coin address")
        payload = await self._rpc("getAccountInfo", [mint, {"encoding": "jsonParsed"}], max_wait)
        return self._parsed("rpc", parse_mint_account, payload, mint)

    async def largest_accounts(self, mint: str, max_wait: float = 20.0) -> list[dict]:
        """The 20 biggest holder accounts (only with your own RPC - the public one refuses)."""
        _check_address(mint, "coin address")
        if not self.rpc_url:
            raise DataError("Holder lists from the network need your own RPC (Settings -> Solana)")
        payload = await self._rpc("getTokenLargestAccounts", [mint], max_wait)
        return self._parsed("rpc", parse_largest_accounts, payload)

    # ------------------------------------------------------------------ Jupiter (quotes, prices, token info)
    async def quote(self, input_mint: str, output_mint: str, amount: int, slippage_bps: int = 500,
                    max_wait: float = 20.0) -> Quote:
        """What swapping `amount` (raw units) would give right now. Only a quote: nothing is swapped."""
        _check_address(input_mint, "coin address")
        _check_address(output_mint, "coin address")
        if int(amount) <= 0:
            raise DataError("the amount to quote must be more than 0")
        status, payload = await self._call(
            "jupiter", self.jupiter, "GET", JUPITER_QUOTE, "Jupiter", max_wait=max_wait,
            params={"inputMint": input_mint, "outputMint": output_mint, "amount": str(int(amount)),
                    "slippageBps": int(slippage_bps)})
        return self._parsed("jupiter", parse_quote, payload, status)

    async def prices(self, mints: list[str], max_wait: float = 20.0) -> dict[str, dict]:
        wanted = [_check_address(m, "coin address") for m in dict.fromkeys(mints)]
        status, payload = await self._call("jupiter", self.jupiter, "GET", JUPITER_PRICE, "Jupiter",
                                           params={"ids": ",".join(wanted)}, max_wait=max_wait)
        if status != 200:
            self._note("jupiter", f"Jupiter's prices answered HTTP {status}")
            raise DataError(f"Jupiter's prices answered HTTP {status}")
        return self._parsed("jupiter", parse_prices, payload)

    async def token_info(self, mint: str, max_wait: float = 20.0) -> dict | None:
        _check_address(mint, "coin address")
        status, payload = await self._call("jupiter", self.jupiter, "GET", JUPITER_TOKENS, "Jupiter",
                                           params={"query": mint}, max_wait=max_wait)
        if status != 200:
            self._note("jupiter", f"Jupiter's token list answered HTTP {status}")
            raise DataError(f"Jupiter's token list answered HTTP {status}")
        return self._parsed("jupiter", parse_token_search, payload, mint)

    async def sol_usd(self, max_wait: float = 20.0) -> float:
        price = (await self.prices([SOL_MINT], max_wait)).get(SOL_MINT)
        if not price:
            raise DataError("Jupiter had no SOL price")
        return price["usd"]

    # ------------------------------------------------------------------ RugCheck
    async def rugcheck(self, mint: str, max_wait: float = 20.0) -> dict:
        _check_address(mint, "coin address")
        status, payload = await self._call("rugcheck", self.rugcheck_limit, "GET", RUGCHECK.format(mint), "RugCheck",
                                           max_wait=max_wait)
        if status != 200:
            self._note("rugcheck", f"RugCheck answered HTTP {status}")
            raise DataError(f"RugCheck answered HTTP {status} (it may not know this coin yet)")
        return self._parsed("rugcheck", parse_rugcheck, payload, mint)
