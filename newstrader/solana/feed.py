"""Which brand-new pools are worth a look: the entry trigger, and the watchlist that waits for young pools to grow up.

The trigger is a conservative MOMENTUM GUESS, nothing more: a pool 3-120 minutes old, with $8k+ in it, 10+ buys from
8+ different buyers in the last 5 minutes, at least 1.5 buys per sell, the price up 5-40% in those 5 minutes and a
fully diluted value of $20k-$5M. Nobody knows whether that predicts anything - the reject audit measures it.

GeckoTerminal's newest-pools list only covers the last minute or so (dozens of coins launch every minute), so a pool
seen there is put on a watchlist and looked at again, up to 30 pools per call, once it is old enough: every 3 minutes
while it is busy, less and less often while it is quiet, and never after 2 hours.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from .parsers import SOL_MINT, Pool

QUICK = timedelta(minutes=3)  # how soon a busy pool is looked at again
MAX_WATCH = 3000


def trigger(pool: Pool, s, now: datetime) -> tuple[bool, str]:
    """(fires?, why) for one pool. Anything missing means no."""
    if pool.quote_mint != SOL_MINT or not pool.mint:
        return False, "not a coin paired with SOL"
    age = pool.age_minutes(now)
    if age is None:
        return False, "age unknown"
    if age < s.min_age_minutes:
        return False, f"too new ({age:.0f} min)"
    if age > s.max_age_minutes:
        return False, f"too old ({age:.0f} min)"
    if pool.reserve_usd is None or pool.reserve_usd < s.min_liquidity_usd:
        return False, "too little money in the pool"
    m5 = pool.window("m5")
    buys, sells, buyers = m5.get("buys"), m5.get("sells"), m5.get("buyers")
    if buys is None or sells is None or buyers is None:
        return False, "no trade counts"
    if buys < s.min_buys_5m or buyers < s.min_buyers_5m:
        return False, "too few buys"
    if sells and buys / sells < s.min_buy_sell_ratio:
        return False, "too many sells"
    change = pool.change.get("m5")
    if change is None or not s.min_change_5m_pct <= change <= s.max_change_5m_pct:
        return False, "price move out of range"
    if pool.fdv_usd is None or not s.min_fdv_usd <= pool.fdv_usd <= s.max_fdv_usd:
        return False, "value out of range"
    ratio = "no sells" if not sells else f"{buys / sells:.1f} buys per sell"
    return True, (f"{buys} buys from {buyers} buyers in 5 minutes ({ratio}), price {change:+.0f}%, "
                  f"${pool.reserve_usd:,.0f} in the pool, {age:.0f} minutes old")


@dataclass
class Watched:
    address: str
    created_at: datetime
    next_check: datetime
    quiet: int = 0


class Watchlist:
    """Young pools waiting to be old enough for the trigger (memory only - a restart starts again)."""

    def __init__(self, max_size: int = MAX_WATCH):
        self.max_size = max_size
        self.items: dict[str, Watched] = {}
        self.done: dict[str, datetime] = {}  # pools already decided (triggered or too old): never again

    def __len__(self) -> int:
        return len(self.items)

    def known(self, address: str) -> bool:
        return address in self.items or address in self.done

    def add(self, pool: Pool, s, now: datetime) -> bool:
        if self.known(pool.address) or pool.created_at is None or pool.quote_mint != SOL_MINT:
            return False
        if pool.age_minutes(now) > s.max_age_minutes:
            return False
        if len(self.items) >= self.max_size:
            oldest = min(self.items.values(), key=lambda w: w.created_at)
            self.items.pop(oldest.address)
        first = max(now + timedelta(seconds=30), pool.created_at + timedelta(minutes=s.min_age_minutes))
        self.items[pool.address] = Watched(pool.address, pool.created_at, first)
        return True

    def due(self, now: datetime, limit: int) -> list[str]:
        ready = sorted((w for w in self.items.values() if w.next_check <= now), key=lambda w: w.next_check)
        return [w.address for w in ready[:limit]]

    def finish(self, address: str, now: datetime) -> None:
        self.items.pop(address, None)
        self.done[address] = now
        if len(self.done) > 4 * self.max_size:
            cutoff = now - timedelta(days=1)
            self.done = {k: v for k, v in self.done.items() if v > cutoff}

    def checked(self, address: str, pool: Pool | None, s, now: datetime) -> None:
        """After a look that didn't fire: when to look again (or drop it - too old, gone, or dead)."""
        w = self.items.get(address)
        if w is None:
            return
        age = (now - w.created_at).total_seconds() / 60
        if pool is None or age >= s.max_age_minutes:
            self.finish(address, now)
            return
        m5, m15 = pool.window("m5"), pool.window("m15")
        if (m5.get("buys") or 0) >= 3:
            w.quiet = 0
            w.next_check = now + QUICK
            return
        if age >= 30 and not (m15.get("buys") or 0):
            self.finish(address, now)  # no buys for 15 minutes: dead
            return
        w.quiet += 1
        w.next_check = now + QUICK * 2 ** min(w.quiet, 4)
