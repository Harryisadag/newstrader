"""The Solana paper desk: the entry trigger and watchlist, the checklist runner (cheapest gates first, the freshness
rerun), desk limits and the kill switch, paper fills and exits (stop, trailing, time, can't-sell, estimates), the
reject audit's follow-ups, the database migration and the settings. Offline: a fake data source built from the real
samples in tests/fixtures/solana. PAPER ONLY - there is nothing to trade with."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from newstrader.config import AppSettings, ConfigStore, SolanaSettings
from newstrader.db import MIGRATIONS, Database, iso
from newstrader.solana import audit
from newstrader.solana import parsers as P
from newstrader.solana.desk import exit_decision, exit_levels
from newstrader.solana.feed import Watchlist, trigger
from newstrader.solana.service import SolanaDesk

FIX = Path(__file__).parent / "fixtures" / "solana"
MINT = "5N3r46YNyCuCsTsVZ4LXnzZuNqgJQ15pWm2cvemcrw5d"
POOL = "9Lvrh9mvYnKfgbWZ1fzHGyusqJZrSi1f5cfTNDdxDWno"
NOW = datetime(2026, 11, 13, 20, 0, 0, tzinfo=UTC)
BUY_OUT = 3514996623441  # coins for 0.1 SOL in the real buy quote
SELL_OUT = 96893417  # lamports for all of them in the real sell quote


def fixture(name: str):
    return json.loads((FIX / f"{name}.json").read_text())


def addr(n: int, letter: str = "p") -> str:
    return f"{letter.upper()}{n:03d}" + letter * 39


class Clock:
    def __init__(self):
        self.t = NOW
        self.m = 1000.0

    def now(self) -> datetime:
        return self.t

    def mono(self) -> float:
        return self.m

    def advance(self, seconds: float) -> None:
        self.t += timedelta(seconds=seconds)
        self.m += seconds


def hot_pool(address: str = POOL, mint: str = MINT, age_min: float = 10.0, **attrs) -> P.Pool:
    """The real pool sample, grown up: 10 minutes old, busy and climbing - the trigger fires on it."""
    p = P.parse_pool(fixture("gecko_pool"))
    p.address, p.mint = address, mint
    p.created_at = NOW - timedelta(minutes=age_min)
    p.reserve_usd, p.fdv_usd = 25_000.0, 60_000.0
    p.change = {"m5": 12.0, "m15": 20.0}
    p.txns = {"m5": {"buys": 20, "sells": 6, "buyers": 15, "sellers": 5},
              "m15": {"buys": 50, "sells": 20, "buyers": 30, "sellers": 12}}
    for k, v in attrs.items():
        setattr(p, k, v)
    return p


class FakeData:
    """Answers like the real sources (the samples), without any network. Every call is written down."""

    def __init__(self, clock: Clock):
        self.clock = clock
        self.rpc_url = ""
        self.gecko = SimpleNamespace(free=lambda: 25)
        self.health: dict = {}
        self.calls: list = []
        self.new: list[P.Pool] = []
        self.pool_map: dict[str, P.Pool] = {POOL: hot_pool()}
        self.mint_info = P.parse_mint_account(fixture("rpc_mint_account"), MINT)
        self.report = fixture("rugcheck_report_full")
        self.token = P.parse_token_search(fixture("jupiter_token_search"), MINT)
        self.sell_value: int | None = None
        self.price_map: dict = {}
        self.fail: dict = {}
        self.on_quote = None

    def _fail(self, name: str) -> None:
        exc = self.fail.get(name)
        if isinstance(exc, list):
            exc = exc.pop(0) if exc else None
        if exc is not None:
            raise exc

    async def new_pools(self, max_wait=30.0):
        self.calls.append("new_pools")
        self._fail("new_pools")
        return list(self.new)

    async def pool(self, address, max_wait=20.0):
        self.calls.append("pool")
        self._fail("pool")
        if address not in self.pool_map:
            raise P.DataError("GeckoTerminal doesn't know this pool")
        return self.pool_map[address]

    async def pools(self, addresses, max_wait=20.0):
        self.calls.append(("pools", len(addresses)))
        self._fail("pools")
        return {a: self.pool_map[a] for a in addresses if a in self.pool_map}

    async def mint_account(self, mint, max_wait=20.0):
        self.calls.append("rpc")
        self._fail("rpc")
        return self.mint_info

    async def largest_accounts(self, mint, max_wait=20.0):
        self.calls.append("largest")
        return []

    async def rugcheck(self, mint, max_wait=20.0):
        self.calls.append("rugcheck")
        self._fail("rugcheck")
        return {**self.report, "mint": mint}

    async def token_info(self, mint, max_wait=20.0):
        self.calls.append("token")
        return self.token

    async def quote(self, input_mint, output_mint, amount, slippage_bps=500, max_wait=20.0):
        buying = input_mint == P.SOL_MINT
        self.calls.append(("buy" if buying else "sell", amount))
        if self.on_quote:
            self.on_quote(buying, amount)
        self._fail("buy" if buying else "sell")
        if buying:
            return P.Quote(P.SOL_MINT, output_mint, amount, BUY_OUT * amount // 100_000_000, 0.0000254, ["Pump.fun"])
        out = self.sell_value if self.sell_value is not None else SELL_OUT * amount // BUY_OUT
        return P.Quote(input_mint, P.SOL_MINT, amount, out, 0.031, ["Pump.fun"])

    async def prices(self, mints, max_wait=20.0):
        self.calls.append("prices")
        return self.price_map

    async def close(self):
        pass


class FakeAlerts:
    def __init__(self):
        self.sent: list[dict] = []

    async def send(self, kind, title, message="", level="info", force=False, fields=None):
        self.sent.append({"kind": kind, "title": title, "message": message, "level": level})
        return {"toast": True}


@pytest.fixture
def env(ctx):
    clock = Clock()
    data = FakeData(clock)
    alerts = FakeAlerts()
    ctx.services["alerts"] = alerts
    ctx.config.update({"solana": {"enabled": True}})
    desk = SolanaDesk(ctx, data=data, now=clock.now, mono=clock.mono)
    ctx.services["solana"] = desk
    return SimpleNamespace(ctx=ctx, clock=clock, data=data, alerts=alerts, desk=desk, db=ctx.db)


def cand_row(env, cid: int) -> dict:
    return env.db.query_one("SELECT * FROM sol_candidates WHERE id = ?", (cid,))


async def buy_one(env) -> dict:
    env.data.new = [hot_pool()]
    stats = await env.desk.feed_tick()
    assert stats["triggered"] == 1
    cand = await env.desk.work_once()
    assert cand["status"] == "bought", cand["reason"]
    return env.desk.desk.position(cand["position_id"])


# ---------------------------------------------------------------------------------------------- the entry trigger
def test_trigger_fires_on_a_busy_young_pool():
    fired, why = trigger(hot_pool(), SolanaSettings(), NOW)
    assert fired and "20 buys from 15 buyers" in why and "3.3 buys per sell" in why


@pytest.mark.parametrize("change,words", [
    ({"age_min": 2}, "too new"), ({"age_min": 121}, "too old"), ({"reserve_usd": 7999.0}, "too little"),
    ({"reserve_usd": None}, "too little"), ({"fdv_usd": 19_000.0}, "value"), ({"fdv_usd": 6_000_000.0}, "value"),
    ({"fdv_usd": None}, "value"), ({"change": {"m5": 4.9}}, "price move"), ({"change": {"m5": 41.0}}, "price move"),
    ({"change": {}}, "price move"), ({"quote_mint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"}, "SOL"),
    ({"txns": {"m5": {"buys": 9, "sells": 1, "buyers": 9, "sellers": 1}}}, "too few"),
    ({"txns": {"m5": {"buys": 12, "sells": 1, "buyers": 7, "sellers": 1}}}, "too few"),
    ({"txns": {"m5": {"buys": 12, "sells": 9, "buyers": 10, "sellers": 1}}}, "too many sells"),
    ({"txns": {}}, "no trade counts"),
])
def test_trigger_says_no(change, words):
    age = change.pop("age_min", 10.0)
    fired, why = trigger(hot_pool(age_min=age, **change), SolanaSettings(), NOW)
    assert not fired and words in why


def test_trigger_counts_no_sells_as_fine():
    p = hot_pool(txns={"m5": {"buys": 10, "sells": 0, "buyers": 8, "sellers": 0}})
    assert trigger(p, SolanaSettings(), NOW) == (True, trigger(p, SolanaSettings(), NOW)[1])
    assert "no sells" in trigger(p, SolanaSettings(), NOW)[1]


def test_watchlist_schedule():
    s = SolanaSettings()
    w = Watchlist()
    young = hot_pool(age_min=0.5)
    assert w.add(young, s, NOW) and not w.add(young, s, NOW) and w.known(POOL)
    assert w.due(NOW, 30) == [] and w.due(NOW + timedelta(minutes=3), 30) == [POOL]  # old enough at 3 minutes
    later = NOW + timedelta(minutes=3)
    w.checked(POOL, hot_pool(), s, later)  # busy: again in 3 minutes
    assert w.items[POOL].next_check == later + timedelta(minutes=3)
    quiet = hot_pool(txns={"m5": {"buys": 0, "sells": 0, "buyers": 0, "sellers": 0}, "m15": {"buys": 2}})
    w.checked(POOL, quiet, s, later)
    w.checked(POOL, quiet, s, later)
    assert w.items[POOL].next_check == later + timedelta(minutes=12)  # quiet: less and less often
    dead = hot_pool(txns={"m5": {"buys": 0}, "m15": {"buys": 0}})
    w.checked(POOL, dead, s, NOW + timedelta(minutes=31))
    assert POOL not in w.items and w.known(POOL)  # dead after 30 minutes without a buy: never again
    w2 = Watchlist()
    w2.add(hot_pool(age_min=1), s, NOW)
    w2.checked(POOL, None, s, NOW + timedelta(minutes=4))  # GeckoTerminal no longer knows it
    assert len(w2) == 0
    assert not Watchlist().add(hot_pool(age_min=200), s, NOW)
    small = Watchlist(max_size=2)
    for i in range(3):
        small.add(hot_pool(address=addr(i), age_min=1 + i), s, NOW)
    assert len(small) == 2 and addr(1) not in small.items  # full: the oldest pool already watched makes room


# ---------------------------------------------------------------------------------------------- the feed and queue
async def test_feed_queues_each_pool_once_and_each_coin_once_a_day(env):
    env.data.new = [hot_pool()]
    assert (await env.desk.feed_tick())["triggered"] == 1
    cand = env.db.query_one("SELECT * FROM sol_candidates")
    assert cand["status"] == "queued" and cand["mint"] == MINT and cand["signal_price_sol"] == pytest.approx(2.8468e-8, rel=1e-3)
    assert json.loads(cand["trigger"])["m5"]["buys"] == 20 and cand["signal_liquidity_usd"] == 25_000
    assert (await env.desk.feed_tick())["triggered"] == 0
    env.data.new = [hot_pool(address=addr(1))]  # another pool of the same coin
    assert (await env.desk.feed_tick())["triggered"] == 0
    assert env.db.scalar("SELECT COUNT(*) FROM sol_candidates") == 1 and len(env.desk.queue) == 1
    env.clock.advance(25 * 3600)
    env.data.new = [hot_pool(address=addr(2), created_at=env.clock.t - timedelta(minutes=10))]
    assert (await env.desk.feed_tick())["triggered"] == 1  # the next day it may come back


async def test_feed_looks_again_at_young_pools(env):
    env.data.new = [hot_pool(age_min=1)]
    stats = await env.desk.feed_tick()
    assert stats["triggered"] == 0 and stats["new"] == 1 and stats["watching"] == 1
    env.clock.advance(150)
    env.data.new = []
    stats = await env.desk.feed_tick()
    assert stats["looked_at"] == 1 and stats["triggered"] == 1  # 3.5 minutes old now, and busy
    assert ("pools", 1) in env.data.calls


async def test_feed_errors_and_budget(env):
    env.data.fail["new_pools"] = P.RateLimited("GeckoTerminal says too many requests")
    stats = await env.desk.feed_tick()
    assert "too many requests" in stats["error"] and env.desk.summary()["state"] == "problem"
    env.data.fail.clear()
    env.data.new = [hot_pool(address=addr(i), mint=addr(i, "m"), age_min=1) for i in range(5)]
    await env.desk.feed_tick()
    env.clock.advance(150)
    env.data.gecko = SimpleNamespace(free=lambda: 6)  # the budget is nearly used: the watchlist waits
    env.data.calls.clear()
    await env.desk.feed_tick()
    assert not any(isinstance(c, tuple) and c[0] == "pools" for c in env.data.calls)


async def test_queue_limit_and_waiting_too_long(env):
    env.data.new = [hot_pool(address=addr(i), mint=addr(i, "m")) for i in range(22)]
    stats = await env.desk.feed_tick()
    assert stats["triggered"] == 20 and env.desk.dropped == 2
    env.clock.advance(11 * 60)
    assert await env.desk.work_once() is None
    assert env.db.scalar("SELECT COUNT(*) FROM sol_candidates WHERE status = 'expired'") == 20
    assert "too old" in env.db.query_one("SELECT reason FROM sol_candidates")["reason"]


# ---------------------------------------------------------------------------------------------- the checklist
async def test_a_coin_that_passes_every_gate_gets_a_paper_buy(env):
    pos = await buy_one(env)
    cand = env.db.query_one("SELECT * FROM sol_candidates")
    gates = env.db.query("SELECT * FROM sol_gate_results WHERE candidate_id = ? ORDER BY step", (cand["id"],))
    assert [g["gate"] for g in gates] == [1, 3, 4, 5, 2, 6, 7] and all(g["passed"] for g in gates)
    assert cand["size_factor"] == 0.5 and "half size" in cand["reason"]  # the creator has no known history
    assert pos["sol_in"] == pytest.approx(0.0505) and int(pos["tokens"]) == int(BUY_OUT // 2 * 0.99)
    assert env.desk.desk.wallet()["sol"] == pytest.approx(10 - 0.0505)
    fill = env.desk.desk.fills(pos["id"])[0]
    assert fill["side"] == "buy" and fill["fee_sol"] == 0.0005 and fill["haircut_sol"] == pytest.approx(0.0005)
    assert env.alerts.sent[-1]["kind"] == "trade_placed" and "PAPER ONLY" in env.alerts.sent[-1]["message"]
    quotes = [c for c in env.data.calls if isinstance(c, tuple) and c[0] in ("buy", "sell")]
    assert quotes == [("buy", 50_000_000), ("sell", BUY_OUT // 2), ("buy", 50_000_000)]  # 3 Jupiter quotes
    row = env.desk.feed_rows()[0]
    assert [x["state"] for x in row["lights"]] == ["pass"] * 7


async def test_a_failed_gate_stops_the_coin_and_spares_jupiter(env):
    env.data.report = {**env.data.report, "token": {**env.data.report["token"], "mintAuthority": addr(5, "a")}}
    env.data.new = [hot_pool()]
    await env.desk.feed_tick()
    cand = await env.desk.work_once()
    assert cand["status"] == "rejected" and cand["killer_gate"] == 1 and "Gate 1 (Authority)" in cand["reason"]
    assert not any(isinstance(c, tuple) and c[0] in ("buy", "sell") for c in env.data.calls)
    lights = env.desk.feed_rows()[0]["lights"]
    assert lights[0]["state"] == "fail" and {x["state"] for x in lights[1:]} == {"not_run"}
    assert env.desk.desk.open_positions() == []


async def test_no_sell_route_fails_gate_2(env):
    env.data.fail["sell"] = P.NoRoute("NO_ROUTES_FOUND")
    env.data.new = [hot_pool()]
    await env.desk.feed_tick()
    cand = await env.desk.work_once()
    assert cand["killer_gate"] == 2 and "classic rug" in cand["reason"]
    assert ("buy", 50_000_000) in env.data.calls and env.data.calls.count(("buy", 50_000_000)) == 1  # no final quote


async def test_errors_fail_closed(env):
    env.data.fail["rugcheck"] = P.DataError("RugCheck didn't answer within 10 s")
    env.data.new = [hot_pool()]
    await env.desk.feed_tick()
    cand = await env.desk.work_once()
    assert cand["status"] == "rejected" and cand["killer_gate"] == 1 and "RugCheck" in cand["reason"]


async def test_stale_evidence_reruns_the_checklist_once(env):
    seen = []

    def slow_first_quote(buying, amount):
        if not seen:
            env.clock.advance(12)  # Jupiter made us wait: the RugCheck report is 12 s old at the buy
        seen.append(amount)

    env.data.on_quote = slow_first_quote
    pos = await buy_one(env)
    cand = env.db.query_one("SELECT * FROM sol_candidates")
    assert cand["attempts"] == 2 and pos is not None
    first = env.db.query("SELECT * FROM sol_gate_results WHERE candidate_id = ? AND attempt = 1", (cand["id"],))
    assert first[-1]["gate"] == 7 and not first[-1]["passed"] and json.loads(first[-1]["data"])["stale_only"]


async def test_no_rerun_when_the_run_was_slow(env):
    seen = []

    def very_slow(buying, amount):
        if not seen:
            env.clock.advance(30)
        seen.append(amount)

    env.data.on_quote = very_slow
    env.data.new = [hot_pool()]
    await env.desk.feed_tick()
    cand = await env.desk.work_once()
    assert cand["status"] == "rejected" and cand["killer_gate"] == 7 and cand["attempts"] == 1


async def test_chase_on_the_final_quote(env):
    calls = []

    def ran_up(buying, amount):
        calls.append(buying)

    env.data.on_quote = ran_up
    real = env.data.quote

    async def pricier(input_mint, output_mint, amount, slippage_bps=500, max_wait=20.0):
        q = await real(input_mint, output_mint, amount, slippage_bps, max_wait)
        if len(calls) == 3:  # the final quote: 20% fewer coins for the same SOL
            q.out_amount = int(q.out_amount / 1.2)
        return q

    env.data.quote = pricier
    env.data.new = [hot_pool()]
    await env.desk.feed_tick()
    cand = await env.desk.work_once()
    assert cand["killer_gate"] == 6 and "don't chase" in cand["reason"]


# ---------------------------------------------------------------------------------------------- desk limits
async def test_kill_switch_stops_new_paper_buys_but_not_exits(env):
    pos = await buy_one(env)
    env.ctx.state.engage_kill_switch("test")
    env.data.new = [hot_pool(address=addr(1), mint=addr(1, "m"))]
    env.data.pool_map[addr(1)] = hot_pool(address=addr(1), mint=addr(1, "m"))
    await env.desk.feed_tick()
    env.data.calls.clear()
    cand = await env.desk.work_once()
    assert cand["status"] == "skipped" and "kill switch" in cand["reason"] and env.data.calls == []
    env.data.sell_value = int(pos["sol_in"] * 0.85 * 1e9)
    done = await env.desk.check_exits()
    assert done[pos["id"]] == "sold"  # exits keep simulating
    assert env.desk.summary()["state"] == "killed"


async def test_desk_limits(env):
    d = env.desk.desk
    assert d.blocked(False, MINT) == ""
    d.set_paused(True)
    assert "paused" in d.blocked(False)
    d.set_paused(False)
    pos = await buy_one(env)
    assert "one position per coin" in d.blocked(False, MINT)
    env.ctx.config.update({"solana": {"max_open_positions": 1}})
    assert "Already 1 open paper position" in d.blocked(False, addr(1, "m"))
    env.ctx.config.update({"solana": {"max_open_positions": 5}})
    w = d.wallet()
    w["sol"] = 0.05
    d.set_state("wallet", w)
    assert "Not enough paper SOL" in d.blocked(False, addr(1, "m"))
    assert d.blocked(False, addr(1, "m"), order_sol=0.04) == ""
    assert pos["status"] == "open"


async def test_daily_paper_loss_limit_halts_new_buys(env):
    env.ctx.config.update({"solana": {"daily_loss_limit_sol": 0.005}})
    pos = await buy_one(env)
    env.data.sell_value = int(pos["sol_in"] * 0.85 * 1e9)
    await env.desk.check_exits()
    assert env.desk.desk.halted_today()
    assert "daily paper loss limit" in env.desk.desk.blocked(False, addr(1, "m"))
    assert any(a["kind"] == "daily_loss_limit" for a in env.alerts.sent)
    assert env.desk.summary()["state"] == "halted"
    env.clock.advance(24 * 3600)
    assert not env.desk.desk.halted_today()  # a new day (UTC)


# ---------------------------------------------------------------------------------------------- fills and exits
async def test_watching_then_stop_loss_counts_every_fee(env):
    pos = await buy_one(env)
    done = await env.desk.check_exits()
    assert done[pos["id"]] == "watching"  # the real round trip loses ~3%, plus fees: about -5%
    view = env.desk.desk.view(env.desk.desk.position(pos["id"]))
    assert view["pnl_pct"] == pytest.approx(-6.9, abs=0.3) and view["levels"]["stop_value"] == pytest.approx(0.04646)
    env.data.sell_value = int(0.0505 * 0.9 * 1e9)
    done = await env.desk.check_exits()
    assert done[pos["id"]] == "sold"
    closed = env.desk.desk.position(pos["id"])
    gross = 0.0505 * 0.9
    assert closed["status"] == "closed" and closed["exit_reason"].startswith("stop_loss")
    assert closed["sol_out"] == pytest.approx(gross * 0.99 - 0.0005)
    assert closed["pnl_sol"] == pytest.approx(gross * 0.99 - 0.0005 - 0.0505)
    assert env.desk.desk.wallet()["sol"] == pytest.approx(10 - 0.0505 + gross * 0.99 - 0.0005)
    assert env.alerts.sent[-1]["kind"] == "trade_filled" and "Paper sale" in env.alerts.sent[-1]["title"]


async def test_trailing_stop(env):
    pos = await buy_one(env)
    for factor, want in ((1.20, "watching"), (1.10, "watching"), (1.30, "watching"), (1.18, "watching"),
                         (1.16, "sold")):  # 10% below the best (1.30) is 1.17
        env.data.sell_value = int(0.0505 * factor * 1e9)
        assert (await env.desk.check_exits())[pos["id"]] == want, factor
    closed = env.desk.desk.position(pos["id"])
    assert closed["exit_reason"].startswith("trailing_stop") and closed["pnl_sol"] > 0


def test_exit_rules_directly():
    s = SolanaSettings()
    pos = {"sol_in": 0.1005, "peak_value_sol": 0.1, "opened_at": iso(NOW)}
    assert exit_decision(pos, 0.0924, NOW, s)[0] == "stop_loss"  # -8.06%
    assert exit_decision(pos, 0.0930, NOW, s) is None
    up = {**pos, "peak_value_sol": 0.12}
    assert exit_decision(up, 0.110, NOW, s) is None and exit_decision(up, 0.108, NOW, s)[0] == "trailing_stop"
    assert exit_decision(pos, 0.1, NOW + timedelta(hours=4), s)[0] == "time_stop"
    lv = exit_levels(up, s)
    assert lv["trailing"] and lv["trail_value"] == pytest.approx(0.108) and lv["time_stop_at"].startswith("2026-11-14")
    assert exit_levels(pos, s)["trail_value"] is None


async def test_time_stop(env):
    pos = await buy_one(env)
    env.clock.advance(4 * 3600 + 1)
    assert (await env.desk.check_exits())[pos["id"]] == "sold"
    assert env.desk.desk.position(pos["id"])["exit_reason"].startswith("time_stop")


async def test_a_429_just_skips_the_exit_check(env):
    pos = await buy_one(env)
    env.data.fail["sell"] = P.RateLimited("Jupiter is pausing after too many requests")
    assert (await env.desk.check_exits())[pos["id"]] == "skipped (Jupiter busy)"
    assert env.desk.desk.position(pos["id"])["status"] == "open"


async def test_cant_sell_after_three_no_routes_and_estimates(env):
    pos = await buy_one(env)
    env.data.fail["sell"] = [P.NoRoute("NO_ROUTES_FOUND")] * 3
    env.data.price_map = {MINT: {"usd": 1e-6}, P.SOL_MINT: {"usd": 80.0}}  # far below the stop - but only a guess
    for n in (1, 2, 3):
        assert (await env.desk.check_exits())[pos["id"]] == "no route"
        row = env.desk.desk.position(pos["id"])
        assert row["no_route_count"] == n and row["status"] == "open" and row["cant_sell"] == int(n >= 3)
    tokens = int(pos["tokens"])
    assert row["value_kind"] == "estimate" and row["value_sol"] == pytest.approx(1e-6 * tokens / 1e6 / 80 * 0.95)
    errors = [a for a in env.alerts.sent if a["kind"] == "error"]
    assert len(errors) == 1 and "can't sell" in errors[0]["title"]
    assert env.desk.summary()["state"] == "problem" and env.desk.summary()["cant_sell"] == 1
    assert (await env.desk.check_exits())[pos["id"]] == "watching"  # a real quote is back
    row = env.desk.desk.position(pos["id"])
    assert row["cant_sell"] == 0 and row["no_route_count"] == 0 and row["value_kind"] == "quote"


async def test_reset_needs_the_word_and_takes_positions_off_the_books(env):
    pos = await buy_one(env)
    with pytest.raises(PermissionError):
        env.desk.reset_wallet("reset please")
    w = env.desk.reset_wallet("RESET")
    assert w["id"] == 2 and w["sol"] == 10.0 and env.desk.desk.open_positions() == []
    old = env.desk.desk.position(pos["id"])
    assert old["status"] == "old_wallet" and old["pnl_sol"] is None
    view = env.desk.snapshot()["closed"][0]
    assert view["old_wallet"] is True


# ---------------------------------------------------------------------------------------------- the reject audit
def add_cand(env, status: str, n: int, minutes_ago: float = 61, killer: int | None = None) -> int:
    return env.db.insert("sol_candidates", {
        "created_at": iso(NOW - timedelta(minutes=minutes_ago)), "wallet": 1, "pool": addr(n), "mint": addr(n, "m"),
        "symbol": f"C{n}", "status": status, "killer_gate": killer, "signal_price_usd": 1e-6,
        "signal_liquidity_usd": 20_000.0, "reason": "test"})


async def test_follow_ups_find_rejected_then_ran_and_passed_then_rugged(env):
    ran = add_cand(env, "rejected", 1, killer=4)
    calm = add_cand(env, "rejected", 2, killer=4)
    rugged = add_cand(env, "bought", 3)
    fine = add_cand(env, "bought", 4)
    add_cand(env, "skipped", 5)  # never checked: no follow-up
    add_cand(env, "rejected", 6, minutes_ago=30, killer=1)  # not due yet
    env.data.pool_map = {
        addr(1): hot_pool(address=addr(1), price_usd=2.5e-6), addr(2): hot_pool(address=addr(2), price_usd=1.2e-6),
        addr(3): hot_pool(address=addr(3), price_usd=0.9e-6, reserve_usd=2000.0),
        addr(4): hot_pool(address=addr(4), price_usd=1.5e-6, reserve_usd=30_000.0)}
    assert await env.desk.follow_up() == 4
    assert ("pools", 4) in env.data.calls and env.data.calls.count("rugcheck") == 2  # RugCheck for the bought ones
    assert cand_row(env, ran)["outcome"] == "ran" and cand_row(env, calm)["outcome"] is None
    assert cand_row(env, rugged)["outcome"] == "rugged" and cand_row(env, fine)["outcome"] is None
    s = audit.summary(env.db)
    assert [r["id"] for r in s["rejected_then_ran"]] == [ran] and s["rejected_then_ran"][0]["best_pct"] == pytest.approx(150)
    assert [r["id"] for r in s["passed_then_rugged"]] == [rugged]
    g4 = next(g for g in s["by_gate"] if g["gate"] == 4)
    assert g4 == {"gate": 4, "name": "Holders", "stopped": 2, "followed_up": 2, "ran": 1}
    assert s["passed"] == {"count": 2, "followed_up": 2, "rugged": 1}
    assert await env.desk.follow_up() == 0  # the 24-hour ones aren't due yet
    env.clock.advance(23 * 3600)
    env.report_rugged = True
    env.data.report = {**env.data.report, "rugged": True}
    assert await env.desk.follow_up() == 5  # four 24-hour follow-ups + the late one's 1-hour follow-up
    assert cand_row(env, fine)["outcome"] == "rugged" and cand_row(env, fine)["rugged_later"] == 1


async def test_a_very_late_1h_follow_up_is_skipped(env):
    cid = add_cand(env, "rejected", 1, minutes_ago=6 * 60, killer=2)
    env.data.pool_map = {addr(1): hot_pool(address=addr(1), price_usd=5e-6)}
    assert await env.desk.follow_up() == 1
    row = cand_row(env, cid)
    assert row["checked_1h_at"] and row["price_1h_usd"] is None and row["outcome"] is None
    assert env.data.calls == []


def test_outcome_rules():
    base = {"status": "rejected", "signal_price_usd": 1.0, "signal_liquidity_usd": 100.0}
    assert audit.outcome({**base, "price_24h_usd": 2.0}) == "ran"
    assert audit.outcome({**base, "price_1h_usd": 1.99}) is None
    assert audit.outcome({**base, "status": "bought", "liquidity_1h_usd": 19.0}) == "rugged"
    assert audit.outcome({**base, "status": "bought", "price_24h_usd": 0.2}) == "rugged"
    assert audit.outcome({**base, "status": "passed", "price_24h_usd": 0.5}) is None
    assert audit.outcome({**base, "status": "bought", "rugged_later": 1}) == "rugged"


# ---------------------------------------------------------------------------------------------- status and service
async def test_status_states(env):
    s = env.desk.summary()
    assert s["state"] == "watching" and s["enabled"] and s["sol"] == 10.0
    env.desk.pause()
    assert env.desk.summary()["state"] == "paused"
    env.data.calls.clear()
    with pytest.raises(RuntimeError, match="paused"):
        await env.desk.check_feed_now()
    env.desk.resume()
    env.ctx.config.update({"solana": {"enabled": False}})
    assert env.desk.summary()["state"] == "off"
    with pytest.raises(RuntimeError, match="off"):
        await env.desk.check_feed_now()


async def test_the_desk_makes_no_calls_while_off(ctx):
    clock = Clock()
    data = FakeData(clock)
    data.new = [hot_pool()]
    desk = SolanaDesk(ctx, data=data, now=clock.now, mono=clock.mono)
    await desk.start()
    await asyncio.sleep(0.05)
    await desk.stop()
    assert data.calls == [] and not ctx.config.settings.solana.enabled
    assert next(c for c in ctx.state.components() if c["component"] == "solana")["level"] == "off"


async def test_the_desk_runs_by_itself_when_on(env):
    env.data.new = [hot_pool()]
    await env.desk.start()
    try:
        for _ in range(100):
            await asyncio.sleep(0.01)
            if env.desk.desk.open_positions():
                break
    finally:
        await env.desk.stop()
    assert len(env.desk.desk.open_positions()) == 1


# ---------------------------------------------------------------------------------------------- settings and database
def test_settings_defaults_and_validation(ctx):
    s = SolanaSettings()
    assert (s.enabled, s.order_sol, s.starting_sol, s.fee_sol, s.haircut_pct, s.max_open_positions) == \
        (False, 0.1, 10, 0.0005, 1, 5)
    assert (s.stop_loss_pct, s.trail_start_pct, s.trail_pct, s.time_stop_hours, s.poll_seconds) == (8, 15, 10, 4, 60)
    assert (s.max_chase_pct, s.max_price_impact_pct, s.max_evidence_age_seconds, s.min_liquidity_usd) == (15, 3, 10, 8000)
    assert AppSettings().schema_version == 7
    ctx.config.update({"solana": {"rpc_url": "https://rpc.example.com/?api-key=abc"}})
    assert ctx.config.settings.solana.rpc_url.startswith("https://")
    for bad in ({"rpc_url": "http://rpc.example.com"}, {"min_age_minutes": 200}, {"min_fdv_usd": 1e7},
                {"order_sol": 0}, {"max_change_5m_pct": -200}):
        with pytest.raises(ValidationError):
            ctx.config.update({"solana": bad})


def test_old_config_gets_the_desk_switched_off(tmp_path):
    data = AppSettings().model_dump(mode="json")
    data["schema_version"] = 6
    data["solana"] = {"enabled": True, "order_sol": 50}  # an older file can't switch the desk on
    data["risk"]["max_open_positions"] = 7
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    store = ConfigStore(path)
    assert store.settings.solana.enabled is False and store.settings.solana.order_sol == 0.1
    assert store.settings.risk.max_open_positions == 7
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["schema_version"] == 7 and saved["solana"]["enabled"] is False


def test_migration_17_adds_the_desk_tables(tmp_path):
    path = tmp_path / "old.db"
    c = sqlite3.connect(path)
    for i, script in enumerate(MIGRATIONS[:16], start=1):  # a database from before the Solana desk
        c.executescript(script)
        c.execute(f"PRAGMA user_version={i}")
    c.commit()
    c.close()
    db = Database(path)
    try:
        assert db.scalar("PRAGMA user_version") == len(MIGRATIONS) >= 17
        tables = {r["name"] for r in db.query("SELECT name FROM sqlite_master WHERE type = 'table'")}
        assert {"sol_candidates", "sol_gate_results", "sol_positions", "sol_fills", "sol_state"} <= tables
        cols = {r["name"] for r in db.query("PRAGMA table_info(sol_candidates)")}
        assert {"killer_gate", "outcome", "price_1h_usd", "price_24h_usd", "rugged_later"} <= cols
        db.insert("sol_candidates", {"created_at": "x", "pool": "p", "mint": "m", "status": "queued"})
        with pytest.raises(sqlite3.IntegrityError):  # each pool once
            db.insert("sol_candidates", {"created_at": "y", "pool": "p", "mint": "m2", "status": "queued"})
    finally:
        db.close()
