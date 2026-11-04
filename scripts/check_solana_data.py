"""Checks the free Solana data sources the paper memecoin desk would use, and prints trimmed samples of what they
return (for building parsers and test fixtures). Run by .github/workflows/check-solana-data.yml (weekly and on demand).

    python scripts/check_solana_data.py            # Markdown report with samples
    python scripts/check_solana_data.py --raw DIR  # also saves each raw response

Standalone on purpose (only needs httpx). Read-only: it only asks for public data, never signs or sends anything.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

UA = "NewsTrader solana-data check (+https://github.com/Harryisadag/newstrader)"
SOL = "So11111111111111111111111111111111111111112"
USDC = "EPjFWJd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
RPC = "https://api.mainnet-beta.solana.com"


def trim(obj, keep: int = 3):
    if isinstance(obj, dict):
        return {k: trim(v, keep) for k, v in obj.items()}
    if isinstance(obj, list):
        return [trim(v, keep) for v in obj[:keep]]
    return obj


class Report:
    def __init__(self, raw: Path | None):
        self.rows = ["## Free Solana data sources", "", "| Check | HTTP | Time | Size |", "|---|---|---|---|"]
        self.details: list[str] = []
        self.failed = 0
        self.total = 0
        self.raw = raw

    def add(self, name: str, status, ms: float, body: bytes, note: str = "") -> None:
        self.total += 1
        self.failed += status != 200
        self.rows.append(f"| {name} | {status} | {ms:.0f} ms | {len(body):,} B |")
        if self.raw:
            (self.raw / f"{name}.json").write_bytes(body)
        try:
            outline = json.dumps(trim(json.loads(body)), separators=(",", ":"))[:7000]
        except Exception:
            outline = body.decode("utf-8", "replace")[:1500]
        self.details += [f"### {name}", "", note, "", "```", outline, "```", ""]

    def text(self) -> str:
        return "\n".join(self.rows + [""] + self.details)


def call(client: httpx.Client, rep: Report, name: str, method: str, url: str, note: str = "", **kw):
    t0 = time.monotonic()
    try:
        r = client.request(method, url, **kw)
        status, body = r.status_code, r.content
    except Exception as exc:
        status, body = f"error: {type(exc).__name__}", str(exc).encode()
    rep.add(name, status, (time.monotonic() - t0) * 1000, body, note)
    try:
        return json.loads(body) if status == 200 else None
    except Exception:
        return None


def rpc(client, rep, name, method, params, note=""):
    return call(client, rep, name, "POST", RPC, note,
                json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="")
    args = ap.parse_args()
    raw = Path(args.raw) if args.raw else None
    if raw:
        raw.mkdir(parents=True, exist_ok=True)
    rep = Report(raw)
    with httpx.Client(headers={"User-Agent": UA, "Accept": "application/json"}, timeout=20,
                      follow_redirects=True) as c:
        # 1. new pools (the candidate feed)
        pools = call(c, rep, "gecko_new_pools", "GET",
                     "https://api.geckoterminal.com/api/v2/networks/solana/new_pools",
                     "GeckoTerminal: newest Solana pools (free, ~30 calls/min)", params={"page": 1,
                                                                                         "include": "base_token,dex"})
        call(c, rep, "gecko_trending", "GET", "https://api.geckoterminal.com/api/v2/networks/solana/trending_pools",
             "GeckoTerminal: trending pools", params={"page": 1})
        call(c, rep, "dexscreener_profiles", "GET", "https://api.dexscreener.com/token-profiles/latest/v1",
             "DexScreener: latest token profiles")
        # pick a young memecoin from the feed for the per-token checks
        mint, pool = None, None
        for p in (pools or {}).get("data", []):
            rel = p.get("relationships", {})
            base = (rel.get("base_token", {}).get("data") or {}).get("id", "")
            if base.startswith("solana_") and base != f"solana_{SOL}":
                mint, pool = base.removeprefix("solana_"), p.get("attributes", {}).get("address")
                break
        if not mint:
            mint = "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"  # BONK, a fallback that always exists
        rep.details.insert(0, f"Sample token: `{mint}` pool `{pool}`\n")
        if pool:
            call(c, rep, "gecko_pool", "GET", f"https://api.geckoterminal.com/api/v2/networks/solana/pools/{pool}",
                 "GeckoTerminal: one pool (liquidity, volume, transactions)")
            call(c, rep, "gecko_pool_ohlcv", "GET",
                 f"https://api.geckoterminal.com/api/v2/networks/solana/pools/{pool}/ohlcv/minute",
                 "GeckoTerminal: 1-minute candles", params={"aggregate": 1, "limit": 30})
        call(c, rep, "dexscreener_token_pairs", "GET", f"https://api.dexscreener.com/token-pairs/v1/solana/{mint}",
             "DexScreener: pairs for the token")
        # 2. authority + Token-2022 extensions
        rpc(c, rep, "rpc_mint_account", "getAccountInfo", [mint, {"encoding": "jsonParsed"}],
            "Solana RPC: the mint account (mintAuthority, freezeAuthority, extensions)")
        rpc(c, rep, "rpc_largest_accounts", "getTokenLargestAccounts", [mint], "Solana RPC: top 20 token accounts")
        rpc(c, rep, "rpc_token_supply", "getTokenSupply", [mint], "Solana RPC: supply")
        # 3. buy then sell quotes (the sell simulation without a wallet)
        buy = call(c, rep, "jupiter_quote_buy", "GET", "https://lite-api.jup.ag/swap/v1/quote",
                   "Jupiter: quote 0.1 SOL -> token", params={"inputMint": SOL, "outputMint": mint,
                                                             "amount": 100_000_000, "slippageBps": 500})
        out = (buy or {}).get("outAmount")
        if out:
            call(c, rep, "jupiter_quote_sell", "GET", "https://lite-api.jup.ag/swap/v1/quote",
                 "Jupiter: quote all of it back -> SOL", params={"inputMint": mint, "outputMint": SOL,
                                                                 "amount": out, "slippageBps": 500})
        call(c, rep, "jupiter_price", "GET", "https://lite-api.jup.ag/price/v3", "Jupiter: price",
             params={"ids": f"{mint},{SOL}"})
        call(c, rep, "jupiter_quote_usdc", "GET", "https://lite-api.jup.ag/swap/v1/quote",
             "Jupiter: a liquid pair for comparison", params={"inputMint": SOL, "outputMint": USDC,
                                                              "amount": 100_000_000, "slippageBps": 50})
        # 3b. more quote / data variants (which ones still work without a key)
        BONK = "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"
        bonk = call(c, rep, "jupiter_quote_buy_bonk", "GET", "https://lite-api.jup.ag/swap/v1/quote",
                    "Jupiter: 0.1 SOL -> BONK (established token)",
                    params={"inputMint": SOL, "outputMint": BONK, "amount": 100_000_000, "slippageBps": 100})
        if (bonk or {}).get("outAmount"):
            call(c, rep, "jupiter_quote_sell_bonk", "GET", "https://lite-api.jup.ag/swap/v1/quote",
                 "Jupiter: BONK -> SOL", params={"inputMint": BONK, "outputMint": SOL, "amount": bonk["outAmount"],
                                                 "slippageBps": 100})
        if out:
            call(c, rep, "jupiter_quote_sell_half", "GET", "https://lite-api.jup.ag/swap/v1/quote",
                 "Jupiter: half of it back -> SOL", params={"inputMint": mint, "outputMint": SOL,
                                                            "amount": str(int(out) // 2), "slippageBps": 1000})
            call(c, rep, "jupiter_ultra_sell", "GET", "https://lite-api.jup.ag/ultra/v1/order",
                 "Jupiter Ultra: order quote (no taker) token -> SOL",
                 params={"inputMint": mint, "outputMint": SOL, "amount": out})
        call(c, rep, "jupiter_ultra_buy", "GET", "https://lite-api.jup.ag/ultra/v1/order",
             "Jupiter Ultra: order quote (no taker) SOL -> token",
             params={"inputMint": SOL, "outputMint": mint, "amount": 100_000_000})
        call(c, rep, "jupiter_api_quote_nokey", "GET", "https://api.jup.ag/swap/v1/quote",
             "Jupiter main API without a key", params={"inputMint": SOL, "outputMint": BONK, "amount": 100_000_000})
        call(c, rep, "jupiter_token_search", "GET", "https://lite-api.jup.ag/tokens/v2/search",
             "Jupiter token info (audit: authorities, holders, organic score)", params={"query": mint})
        call(c, rep, "jupiter_tokens_recent", "GET", "https://lite-api.jup.ag/tokens/v2/recent",
             "Jupiter: recently created tokens")
        call(c, rep, "pumpfun_amm_sell_quote_dexscreener", "GET",
             f"https://api.dexscreener.com/latest/dex/tokens/{mint}", "DexScreener: token (all pairs)")
        for name, url in (("publicnode", "https://solana-rpc.publicnode.com"),):
            call(c, rep, f"rpc_{name}_largest_accounts", "POST", url, f"{name}: getTokenLargestAccounts",
                 json={"jsonrpc": "2.0", "id": 1, "method": "getTokenLargestAccounts", "params": [mint]})
        time.sleep(3)
        rpc(c, rep, "rpc_largest_accounts_retry", "getTokenLargestAccounts", [mint], "Solana RPC: retry after 3 s")
        # 4. third-party risk report (LP lock, creator, insiders)
        call(c, rep, "rugcheck_report", "GET", f"https://api.rugcheck.xyz/v1/tokens/{mint}/report/summary",
             "RugCheck: summary report")
        call(c, rep, "rugcheck_report_full", "GET", f"https://api.rugcheck.xyz/v1/tokens/{mint}/report",
             "RugCheck: full report (markets/LP, creator, top holders, insider graph)")
        call(c, rep, "rugcheck_new_tokens", "GET", "https://api.rugcheck.xyz/v1/stats/new_tokens",
             "RugCheck: newest tokens")
        # 5. pump.fun (unofficial, may block)
        call(c, rep, "pumpfun_coin", "GET", f"https://frontend-api-v3.pump.fun/coins/{mint}",
             "pump.fun (unofficial API): coin + creator")
    print(rep.text())
    return 1 if rep.failed == rep.total else 0


if __name__ == "__main__":
    sys.exit(main())
