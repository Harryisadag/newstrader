"""The pre-flight checklist for one coin: fetch the evidence, run the seven gates and stop at the first failure.

1. The cheap evidence, all at once: the coin on the Solana network, RugCheck's report, the pool again on
   GeckoTerminal, Jupiter's token info (and the holder list from your own RPC, if you set one). Gates 1, 3, 4, 5.
2. Jupiter's round trip: a buy quote for the order size, then a quote to sell ALL of it back. Gate 2.
3. The desk limits again, then the FINAL Jupiter buy quote right before the paper buy. Gates 6 and 7.

If the only problem at the end is stale evidence (Gate 7) and the run took less than 25 s, it runs once more from the
start. Nothing here buys anything: the service does the paper buy when every gate passed.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from ..db import utcnow
from .gates import GATE_FUNCS, Evidence, Fetched, GateResult, Rules
from .parsers import SOL_MINT, DataError, NoRoute, RateLimited

RERUN_UNDER = 25.0  # seconds: a stale-only failure in a run shorter than this is tried once more
JUPITER_WAIT = 10.0  # seconds a checklist quote waits for Jupiter's allowance before it counts as an error
FETCH_TIMEOUT = 30.0


@dataclass
class Attempt:
    number: int
    results: list[GateResult] = field(default_factory=list)
    evidence: Evidence | None = None
    seconds: float = 0.0
    blocked: str = ""  # a desk limit stopped it just before the paper buy
    order_sol: float = 0.0

    @property
    def killer(self) -> GateResult | None:
        return next((r for r in self.results if not r.passed), None)

    @property
    def passed(self) -> bool:
        return not self.blocked and len(self.results) == len(GATE_FUNCS) and self.killer is None


@dataclass
class Outcome:
    attempts: list[Attempt]

    @property
    def last(self) -> Attempt:
        return self.attempts[-1]

    @property
    def passed(self) -> bool:
        return self.last.passed

    @property
    def killer(self) -> GateResult | None:
        return self.last.killer

    @property
    def size_factor(self) -> float:
        g5 = next((r for r in self.last.results if r.gate == 5), None)
        return float((g5.data if g5 else {}).get("size_factor", 1.0))


async def fetch(source: str, coro, now=utcnow) -> Fetched:
    """Run one question to a data source; any problem becomes an error in the evidence (the gates fail closed)."""
    try:
        value = await asyncio.wait_for(coro, FETCH_TIMEOUT)
    except NoRoute as exc:
        return Fetched(source, None, f"Jupiter found no route ({exc.code})", now(), "no_route")
    except RateLimited as exc:
        return Fetched(source, None, str(exc), now(), "rate_limited")
    except DataError as exc:
        return Fetched(source, None, str(exc), now(), "error")
    except TimeoutError:
        return Fetched(source, None, f"no answer within {FETCH_TIMEOUT:.0f} s", now(), "error")
    except Exception as exc:  # never let a strange answer crash the desk: it's just a failed gate
        return Fetched(source, None, f"{type(exc).__name__}: {exc}"[:200], now(), "error")
    return Fetched(source, value, "", now())


async def _attempt(number: int, data, cand: dict, s, rules: Rules, blocked: Callable[[float], str], now,
                   mono) -> Attempt:
    t0 = mono()
    mint, pool = cand["mint"], cand["pool"]
    ev = Evidence(mint=mint, pool=pool, order_sol=s.order_sol, signal_price_sol=cand.get("signal_price_sol"))
    att = Attempt(number, evidence=ev, order_sol=s.order_sol)

    def gate(g: int) -> bool:
        res = GATE_FUNCS[g](ev, rules, now())
        att.results.append(res)
        return res.passed

    def done() -> Attempt:
        att.seconds = mono() - t0
        return att

    jobs = [fetch("rpc", data.mint_account(mint), now), fetch("rugcheck", data.rugcheck(mint), now),
            fetch("gecko", data.pool(pool), now), fetch("jupiter", data.token_info(mint, max_wait=JUPITER_WAIT), now)]
    if getattr(data, "rpc_url", ""):
        jobs.append(fetch("rpc", data.largest_accounts(mint), now))
    got = await asyncio.gather(*jobs)
    ev.mint_info, ev.rug, ev.pool_now, ev.token = got[:4]
    if len(got) > 4:
        ev.holders_rpc = got[4]
    for g in (1, 3, 4, 5):
        if not gate(g):
            return done()
    g5 = att.results[-1]
    att.order_sol = s.order_sol * float(g5.data.get("size_factor", 1.0))
    amount = int(round(att.order_sol * 1_000_000_000))
    ev.buy = await fetch("jupiter", data.quote(SOL_MINT, mint, amount, max_wait=JUPITER_WAIT), now)
    if ev.buy.ok:
        ev.sell = await fetch("jupiter", data.quote(mint, SOL_MINT, ev.buy.value.out_amount, max_wait=JUPITER_WAIT),
                              now)
    if not gate(2):
        return done()
    att.blocked = blocked(att.order_sol)
    if att.blocked:
        return done()
    ev.final = await fetch("jupiter", data.quote(SOL_MINT, mint, amount, max_wait=JUPITER_WAIT), now)
    for g in (6, 7):
        if not gate(g):
            return done()
    return done()


async def run_checklist(data, cand: dict, s, blocked: Callable[[float], str], now=utcnow,
                        mono=time.monotonic) -> Outcome:
    """The whole checklist for one candidate (a dict with mint, pool and signal_price_sol)."""
    rules = Rules.from_settings(s)
    first = await _attempt(1, data, cand, s, rules, blocked, now, mono)
    killer = first.killer
    if killer is not None and killer.gate == 7 and killer.data.get("stale_only") and first.seconds < RERUN_UNDER:
        return Outcome([first, await _attempt(2, data, cand, s, rules, blocked, now, mono)])
    return Outcome([first])
