"""The seven gates of the Solana paper desk's pre-flight checklist: each one's pass and fail, built from the real
answers saved in tests/fixtures/solana. Fail closed: errors, timeouts and missing fields must fail."""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from newstrader.solana import parsers as P
from newstrader.solana.gates import (
    GATES,
    RUN_ORDER,
    Evidence,
    Fetched,
    Rules,
    gate_authority,
    gate_chase,
    gate_creator,
    gate_freshness,
    gate_holders,
    gate_liquidity,
    gate_sell,
    real_holders,
    run_gates,
)

FIX = Path(__file__).parent / "fixtures" / "solana"
MINT = "5N3r46YNyCuCsTsVZ4LXnzZuNqgJQ15pWm2cvemcrw5d"
POOL = "9Lvrh9mvYnKfgbWZ1fzHGyusqJZrSi1f5cfTNDdxDWno"
WHALE = "BwWK17cbHxwWBKZkUYvzxLcNQ1YVyaFezduWbtm2de6s"  # a plain wallet that held ~50% of a new coin
NOW = datetime(2026, 11, 13, 20, 0, 0, tzinfo=UTC)
RULES = Rules()


def fixture(name: str):
    return json.loads((FIX / f"{name}.json").read_text())


def wallet(n: int) -> str:
    return f"W{n:02d}" + "x" * 38  # 41 characters, address-shaped


def ok(source: str, value, age: float = 1.0) -> Fetched:
    return Fetched(source, value, at=NOW - timedelta(seconds=age))


def failed(source: str, error: str = "RugCheck didn't answer within 10 s", kind: str = "error") -> Fetched:
    return Fetched(source, None, error=error, at=NOW - timedelta(seconds=1), kind=kind)


def mint_raw(**info) -> dict:
    raw = fixture("rpc_mint_account")
    raw["result"]["value"]["data"]["parsed"]["info"].update(info)
    return raw


def mint_info(**info) -> P.MintInfo:
    return P.parse_mint_account(mint_raw(**info), MINT)


def rug(**changes) -> dict:
    report = copy.deepcopy(fixture("rugcheck_report_full"))
    report.update(changes)
    return report


def pool_now(reserve: float | None = 25_000.0) -> P.Pool:
    p = P.parse_pool(fixture("gecko_pool"))
    p.reserve_usd = reserve
    return p


def quote(name: str, **changes) -> P.Quote:
    raw = fixture(name)
    raw.update(changes)
    return P.parse_quote(raw)


def evidence(**over) -> Evidence:
    ev = Evidence(mint=MINT, pool=POOL, order_sol=0.1, signal_price_sol=P.parse_pool(fixture("gecko_pool")).price_sol,
                  mint_info=ok("rpc", mint_info()), rug=ok("rugcheck", rug()),
                  token=ok("jupiter", P.parse_token_search(fixture("jupiter_token_search"), MINT)),
                  pool_now=ok("gecko", pool_now()), buy=ok("jupiter", quote("jupiter_quote_buy")),
                  sell=ok("jupiter", quote("jupiter_quote_sell")), final=ok("jupiter", quote("jupiter_quote_buy")))
    for k, v in over.items():
        setattr(ev, k, v)
    return ev


def test_the_real_samples_pass_every_gate_but_liquidity():
    """The sampled coin was 1 second old: clean, but with $1,810 in its pool - Gate 3 stops it."""
    ev = evidence(pool_now=ok("gecko", P.parse_pool(fixture("gecko_pool"))))
    results = run_gates(ev, RULES, NOW)
    assert [r.gate for r in results] == [1, 3]
    assert results[0].passed and not results[1].passed and "$1,810" in results[1].reason
    full = run_gates(evidence(), RULES, NOW)
    assert [r.gate for r in full] == list(RUN_ORDER) == [1, 3, 4, 5, 2, 6, 7]
    assert all(r.passed for r in full), [r.reason for r in full if not r.passed]
    assert {r.name for r in full} == set(GATES.values())
    assert full[3].data["size_factor"] == 0.5  # Gate 5: the creator has no known history


# ---------------------------------------------------------------------------------------------- Gate 1: Authority
def test_authority_passes_and_names_its_sources():
    r = gate_authority(evidence(), RULES, NOW)
    assert r.passed and r.gate == 1 and "RugCheck and Jupiter" in r.reason and r.data["jupiter_audit"] is True


@pytest.mark.parametrize("info,words", [
    ({"mintAuthority": "TSLvdd1pWpHVjahSpsvCXUbgwsL3JAcvokwaKt1eokM"}, "make more"),
    ({"freezeAuthority": "TSLvdd1pWpHVjahSpsvCXUbgwsL3JAcvokwaKt1eokM"}, "freeze"),
])
def test_authority_still_held_fails(info, words):
    r = gate_authority(evidence(mint_info=ok("rpc", mint_info(**info))), RULES, NOW)
    assert not r.passed and words in r.reason


def test_authority_fails_closed():
    raw = fixture("rpc_mint_account")
    del raw["result"]["value"]["data"]["parsed"]["info"]["mintAuthority"]
    unknown = ok("rpc", P.parse_mint_account(raw, MINT))
    assert "unknown" in gate_authority(evidence(mint_info=unknown), RULES, NOW).reason
    assert not gate_authority(evidence(mint_info=failed("rpc", "timed out")), RULES, NOW).passed
    assert not gate_authority(evidence(mint_info=None), RULES, NOW).passed
    assert not gate_authority(evidence(rug=failed("rugcheck")), RULES, NOW).passed
    no_token = rug()
    del no_token["token"]["freezeAuthority"]
    assert "double-checked" in gate_authority(evidence(rug=ok("rugcheck", no_token)), RULES, NOW).reason
    assert not gate_authority(evidence(token=failed("jupiter", "Jupiter is busy", "rate_limited")), RULES, NOW).passed


def test_authority_sources_must_agree():
    says_held = rug()
    says_held["token"]["mintAuthority"] = "TSLvdd1pWpHVjahSpsvCXUbgwsL3JAcvokwaKt1eokM"
    r = gate_authority(evidence(rug=ok("rugcheck", says_held)), RULES, NOW)
    assert not r.passed and "disagree" in r.reason
    jup = P.parse_token_search(fixture("jupiter_token_search"), MINT)
    jup["freeze_authority_disabled"] = False
    r = gate_authority(evidence(token=ok("jupiter", jup)), RULES, NOW)
    assert not r.passed and "Jupiter" in r.reason
    r = gate_authority(evidence(token=ok("jupiter", None)), RULES, NOW)  # Jupiter doesn't list it yet: noted
    assert r.passed and "no audit" in r.reason
    burned = rug()
    burned["token"]["mintAuthority"] = P.SYSTEM_PROGRAM  # "burned" counts as nobody
    assert gate_authority(evidence(rug=ok("rugcheck", burned)), RULES, NOW).passed


# ---------------------------------------------------------------------------------------------- Gate 2: Sell check
def test_sell_check_passes_on_the_real_round_trip():
    r = gate_sell(evidence(), RULES, NOW)
    assert r.passed and r.data["round_trip_loss_pct"] == pytest.approx(3.11, abs=0.01)
    assert r.data["extensions"] == ["metadataPointer", "tokenMetadata"] and r.data["sell_route"] == ["Pump.fun"]


def with_extension(name: str, state=None) -> Evidence:
    raw = mint_raw()
    raw["result"]["value"]["data"]["parsed"]["info"]["extensions"].append({"extension": name, "state": state or {}})
    return evidence(mint_info=ok("rpc", P.parse_mint_account(raw, MINT)))


@pytest.mark.parametrize("name,state,words", [
    ("transferHook", {"programId": "Hook111111111111111111111111111111111111111"}, "transfer hook"),
    ("pausableConfig", {"authority": "x", "paused": False}, "pause"),
    ("permanentDelegate", {"delegate": "x"}, "permanent delegate"),
    ("nonTransferable", {}, "can't be transferred"),
    ("defaultAccountState", {"accountState": "frozen"}, "frozen"),
    ("confidentialTransferMint", {}, "confidential"),
    ("somethingNew", {}, "unusual feature (somethingNew)"),
])
def test_dangerous_extensions_fail(name, state, words):
    r = gate_sell(with_extension(name, state), RULES, NOW)
    assert not r.passed and words in r.reason


@pytest.mark.parametrize("name,state", [("groupPointer", {}), ("tokenGroupMember", {}), ("immutableOwner", {}),
                                        ("defaultAccountState", {"accountState": "initialized"})])
def test_harmless_extensions_pass(name, state):
    assert gate_sell(with_extension(name, state), RULES, NOW).passed


def fee(bps: int, authority=None) -> dict:
    return {"transferFeeConfigAuthority": authority, "withdrawWithheldAuthority": None, "withheldAmount": 0,
            "olderTransferFee": {"epoch": "1", "maximumFee": "1000", "transferFeeBasisPoints": bps},
            "newerTransferFee": {"epoch": "2", "maximumFee": "1000", "transferFeeBasisPoints": bps}}


def test_transfer_fees():
    assert gate_sell(with_extension("transferFeeConfig", fee(500)), RULES, NOW).passed  # 5%, nobody can change it
    r = gate_sell(with_extension("transferFeeConfig", fee(1200)), RULES, NOW)
    assert not r.passed and "12% fee" in r.reason
    r = gate_sell(with_extension("transferFeeConfig", fee(0, "FeeAuth11111111111111111111111111111111111")), RULES, NOW)
    assert not r.passed and "can still change" in r.reason  # a changeable fee fails even at 0%
    no_rate = fee(0)
    del no_rate["olderTransferFee"], no_rate["newerTransferFee"]
    assert "unknown size" in gate_sell(with_extension("transferFeeConfig", no_rate), RULES, NOW).reason


def test_sell_check_cross_checks_rugcheck():
    r1 = rug()
    r1["token_extensions"]["permanentDelegate"] = {"delegate": "x"}
    r = gate_sell(evidence(rug=ok("rugcheck", r1)), RULES, NOW)
    assert not r.passed and "RugCheck" in r.reason and "disagree" in r.reason
    r2 = rug(transferFee={"pct": 0, "maxAmount": 0, "authority": "FeeAuth11111111111111111111111111111111111"})
    assert "change the coin's transfer fee" in gate_sell(evidence(rug=ok("rugcheck", r2)), RULES, NOW).reason
    r3 = rug(token_extensions=None)
    assert "couldn't be double-checked" in gate_sell(evidence(rug=ok("rugcheck", r3)), RULES, NOW).reason
    classic = P.parse_mint_account(mint_raw(), MINT)
    classic.program, classic.extensions = "spl-token", []
    assert gate_sell(evidence(mint_info=ok("rpc", classic), rug=ok("rugcheck", r3)), RULES, NOW).passed
    classic.program = "something-else"
    assert "isn't a normal Solana coin" in gate_sell(evidence(mint_info=ok("rpc", classic)), RULES, NOW).reason


def test_sell_check_round_trip_fails_closed():
    no_way = Fetched("jupiter", None, error="NO_ROUTES_FOUND", at=NOW, kind="no_route")
    r = gate_sell(evidence(sell=no_way), RULES, NOW)
    assert not r.passed and "no way to sell it back" in r.reason
    assert "no way to buy" in gate_sell(evidence(buy=no_way), RULES, NOW).reason
    busy = failed("jupiter", "Jupiter is pausing after too many requests", "rate_limited")
    assert "too many requests" in gate_sell(evidence(sell=busy), RULES, NOW).reason
    assert not gate_sell(evidence(sell=None), RULES, NOW).passed
    lossy = quote("jupiter_quote_sell", outAmount="88000000")
    r = gate_sell(evidence(sell=ok("jupiter", lossy)), RULES, NOW)
    assert not r.passed and "lose 12.0%" in r.reason
    half = quote("jupiter_quote_sell_half")
    assert "wasn't for all the coins" in gate_sell(evidence(sell=ok("jupiter", half)), RULES, NOW).reason


# ---------------------------------------------------------------------------------------------- Gate 3: Liquidity
def test_liquidity_passes_for_a_bonding_curve():
    r = gate_liquidity(evidence(), RULES, NOW)
    assert r.passed and r.data["lp_locked_pct"] == [100] and r.data["order_pct_of_pool"] < 0.1


def unlocked(pct) -> Evidence:
    r = rug()
    r["markets"][0]["lp"]["lpLockedPct"] = pct
    return evidence(rug=ok("rugcheck", r))


def test_liquidity_not_locked_fails():
    assert "Only 50%" in gate_liquidity(unlocked(50), RULES, NOW).reason
    assert not gate_liquidity(unlocked(98.9), RULES, NOW).passed
    assert gate_liquidity(unlocked(99.2), RULES, NOW).passed
    assert "couldn't tell" in gate_liquidity(unlocked(None), RULES, NOW).reason
    assert "no market" in gate_liquidity(evidence(rug=ok("rugcheck", rug(markets=[]))), RULES, NOW).reason
    no_lockers = rug()
    del no_lockers["lockers"]
    assert "time lock" in gate_liquidity(evidence(rug=ok("rugcheck", no_lockers)), RULES, NOW).reason


@pytest.mark.parametrize("unlock,passes", [
    ((NOW + timedelta(days=60)).timestamp(), True),
    ((NOW + timedelta(days=60)).timestamp() * 1000, True),  # milliseconds
    ((NOW + timedelta(days=45)).isoformat(), True),
    ((NOW + timedelta(days=10)).timestamp(), False),
    ((NOW - timedelta(days=1)).timestamp(), False),  # already open
    (None, False),  # unknown date
    (0, False),
])
def test_liquidity_lock_dates(unlock, passes):
    r = rug(lockers={"Lock1111111111111111111111111111111111111111": {"unlockDate": unlock, "tokenAccount": "x"}})
    res = gate_liquidity(evidence(rug=ok("rugcheck", r)), RULES, NOW)
    assert res.passed is passes, res.reason


def test_liquidity_pool_size_and_order_share():
    assert "$5,000" in gate_liquidity(evidence(pool_now=ok("gecko", pool_now(5000))), RULES, NOW).reason
    assert "doesn't know" in gate_liquidity(evidence(pool_now=ok("gecko", pool_now(None))), RULES, NOW).reason
    assert not gate_liquidity(evidence(pool_now=failed("gecko", "GeckoTerminal says too many requests")), RULES, NOW).passed
    small = Rules(min_liquidity_usd=100)
    r = gate_liquidity(evidence(pool_now=ok("gecko", pool_now(300))), small, NOW)  # 0.1 SOL ~ $8 of $300 = 2.7%
    assert not r.passed and "too big a splash" in r.reason
    assert gate_liquidity(evidence(pool_now=ok("gecko", pool_now(1000))), small, NOW).passed  # 0.8%
    other = pool_now()
    other.mint = "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"
    assert "different coin" in gate_liquidity(evidence(pool_now=ok("gecko", other)), RULES, NOW).reason


# ---------------------------------------------------------------------------------------------- Gate 4: Holders
def holders(*rows, **changes) -> Evidence:
    r = rug(**changes)
    r["topHolders"] = r["topHolders"][:1] + [
        {"address": f"A{i:02d}" + "a" * 38, "owner": owner, "pct": pct, "insider": insider}
        for i, (owner, pct, insider) in enumerate(rows)]
    return evidence(rug=ok("rugcheck", r))


def test_holders_leave_out_the_pool():
    r = gate_holders(evidence(), RULES, NOW)
    assert r.passed and r.data["top10_pct"] == pytest.approx(0.09, abs=0.01)
    assert POOL in r.data["left_out"]  # the pool's own account held 99.9%


def test_one_big_wallet_fails():
    r = gate_holders(holders((WHALE, 50.0, False), (wallet(1), 1.0, False)), RULES, NOW)
    assert not r.passed and "50.0%" in r.reason and "BwWK" in r.reason


def test_accounts_of_one_wallet_count_as_one():
    rows = [(wallet(1), 4.0, False)] * 3 + [(wallet(2), 3.0, False)]
    r = gate_holders(holders(*rows), RULES, NOW)
    assert not r.passed and "12.0%" in r.reason
    assert r.data["holders"][0]["accounts"] == 3


def test_insiders_merge_into_one_holder():
    r = gate_holders(holders((wallet(1), 4.0, True), (wallet(2), 4.0, True), (wallet(3), 4.0, True)), RULES, NOW)
    assert not r.passed and r.reason.startswith("Insiders together hold 12.0%")
    net = holders((wallet(1), 6.0, False), (wallet(2), 6.0, False),
                  insiderNetworks=[{"id": "n1", "size": 2, "accounts": [{"address": wallet(1)}, wallet(2)]}])
    r = gate_holders(net, RULES, NOW)
    assert not r.passed and "Insiders together" in r.reason


def test_top_ten_limit():
    rows = [(wallet(i), 3.5, False) for i in range(10)]
    r = gate_holders(holders(*rows), RULES, NOW)
    assert not r.passed and "35.0%" in r.reason
    assert gate_holders(holders(*[(wallet(i), 2.5, False) for i in range(10)]), RULES, NOW).passed  # 25%


def test_pools_lockers_and_burn_are_not_holders():
    lock = "Lock1111111111111111111111111111111111111111"
    r = holders((lock, 40.0, False), (P.SYSTEM_PROGRAM, 30.0, False), (wallet(1), 2.0, False),
                knownAccounts={lock: {"name": "Streamflow", "type": "LOCKER"}})
    assert gate_holders(r, RULES, NOW).passed
    vault = rug()["markets"][0]["liquidityA"]
    ranked, left = real_holders(rug(topHolders=[{"address": vault, "owner": wallet(9), "pct": 60.0}]))
    assert ranked == [] and left[0]["pct"] == 60.0


def test_holders_fail_closed():
    assert "insiders" in gate_holders(holders(graphInsidersDetected=3), RULES, NOW).reason
    no_graph = rug()
    del no_graph["graphInsidersDetected"]
    assert not gate_holders(evidence(rug=ok("rugcheck", no_graph)), RULES, NOW).passed
    assert not gate_holders(evidence(rug=ok("rugcheck", rug(topHolders=[]))), RULES, NOW).passed
    assert not gate_holders(evidence(rug=failed("rugcheck")), RULES, NOW).passed


def test_own_rpc_holder_cross_check():
    supply = 1_000_000_000_000_000
    pool_acct = rug()["topHolders"][0]["address"]
    rows = [{"address": pool_acct, "amount": supply * 90 // 100}, {"address": wallet(5), "amount": supply * 2 // 100}]
    assert gate_holders(evidence(holders_rpc=ok("rpc", rows)), RULES, NOW).passed
    rows.append({"address": wallet(6), "amount": supply * 15 // 100})  # RugCheck's list missed it
    r = gate_holders(evidence(holders_rpc=ok("rpc", rows)), RULES, NOW)
    assert not r.passed and "Your RPC" in r.reason
    assert not gate_holders(evidence(holders_rpc=failed("rpc", "your RPC didn't answer")), RULES, NOW).passed


# ---------------------------------------------------------------------------------------------- Gate 5: Creator
def test_creator_unknown_or_no_history_passes_at_half_size():
    r = gate_creator(evidence(), RULES, NOW)  # the sample: creatorTokens is null
    assert r.passed and r.data["size_factor"] == 0.5 and "half" in r.reason
    r = gate_creator(evidence(rug=ok("rugcheck", rug(creator=None))), RULES, NOW)
    assert r.passed and r.data["size_factor"] == 0.5 and r.reason.startswith("Nobody knows")


def made(hours_ago: float | None, n: int) -> dict:
    return {"mint": f"M{n:02d}" + "m" * 38, "marketCap": 5000,
            "createdAt": None if hours_ago is None else (NOW - timedelta(hours=hours_ago)).isoformat()}


def test_creator_history():
    risky = rug(risks=[{"name": "Creator history of rugged tokens", "level": "danger", "score": 15000, "value": ""}])
    r = gate_creator(evidence(rug=ok("rugcheck", risky)), RULES, NOW)
    assert not r.passed and "rugged before" in r.reason
    assert "already marks" in gate_creator(evidence(rug=ok("rugcheck", rug(rugged=True))), RULES, NOW).reason
    serial = rug(creatorTokens=[made(2, 1), made(5, 2), made(100, 3)])
    r = gate_creator(evidence(rug=ok("rugcheck", serial)), RULES, NOW)
    assert not r.passed and "3 coins" in r.reason  # two others today + this one
    unknown_dates = rug(creatorTokens=[made(None, 1), made(None, 2)])
    assert not gate_creator(evidence(rug=ok("rugcheck", unknown_dates)), RULES, NOW).passed
    calm = rug(creatorTokens=[made(30, 1), made(200, 2), made(500, 3)])
    r = gate_creator(evidence(rug=ok("rugcheck", calm)), RULES, NOW)
    assert r.passed and r.data["size_factor"] == 1.0 and r.data["launches_24h"] == 1


def test_creator_fails_closed():
    no_risks = rug()
    del no_risks["risks"]
    assert not gate_creator(evidence(rug=ok("rugcheck", no_risks)), RULES, NOW).passed
    no_flag = rug()
    del no_flag["rugged"]
    assert not gate_creator(evidence(rug=ok("rugcheck", no_flag)), RULES, NOW).passed
    assert not gate_creator(evidence(rug=failed("rugcheck")), RULES, NOW).passed
    assert not gate_creator(evidence(rug=ok("rugcheck", rug(creatorTokens="lots"))), RULES, NOW).passed


# ---------------------------------------------------------------------------------------------- Gate 6: Chase + impact
def test_chase_and_impact_pass_on_the_real_quote():
    r = gate_chase(evidence(), RULES, NOW)
    assert r.passed and abs(r.data["chase_pct"]) < 1 and r.data["impact_pct"] < 0.01


def test_chase_too_far_fails():
    ev = evidence()
    ev.signal_price_sol = ev.signal_price_sol / 1.2  # the price ran up ~20% since the coin was spotted
    r = gate_chase(ev, RULES, NOW)
    assert not r.passed and "don't chase" in r.reason and r.data["chase_pct"] == pytest.approx(19.9, abs=0.3)


def test_impact_is_a_fraction():
    assert not gate_chase(evidence(final=ok("jupiter", quote("jupiter_quote_buy", priceImpactPct="0.031"))), RULES, NOW).passed
    assert gate_chase(evidence(final=ok("jupiter", quote("jupiter_quote_buy", priceImpactPct="0.029"))), RULES, NOW).passed


def test_chase_fails_closed():
    no_way = Fetched("jupiter", None, error="NO_ROUTES_FOUND", at=NOW, kind="no_route")
    assert "no way to buy" in gate_chase(evidence(final=no_way), RULES, NOW).reason
    assert not gate_chase(evidence(final=failed("jupiter", "Jupiter is busy")), RULES, NOW).passed
    assert not gate_chase(evidence(final=None), RULES, NOW).passed
    assert not gate_chase(evidence(signal_price_sol=None), RULES, NOW).passed
    raw = fixture("jupiter_quote_buy")
    del raw["priceImpactPct"]
    assert "didn't say" in gate_chase(evidence(final=ok("jupiter", P.parse_quote(raw))), RULES, NOW).reason


# ---------------------------------------------------------------------------------------------- Gate 7: Freshness
def test_freshness():
    assert gate_freshness(evidence(), RULES, NOW).passed
    stale = gate_freshness(evidence(rug=ok("rugcheck", rug(), age=12)), RULES, NOW)
    assert not stale.passed and stale.data["stale"] == ["rug"] and stale.data["stale_only"] is True
    broken = gate_freshness(evidence(token=failed("jupiter"), rug=ok("rugcheck", rug(), age=12)), RULES, NOW)
    assert not broken.passed and broken.data["errors"] == ["token"] and broken.data["stale_only"] is False
    assert not gate_freshness(Evidence(MINT, POOL, 0.1, 1e-8), RULES, NOW).passed


def test_rules_come_from_settings():
    class S:
        max_chase_pct = 7.5
        min_liquidity_usd = 20000
        enabled = True

    r = Rules.from_settings(S())
    assert r.max_chase_pct == 7.5 and r.min_liquidity_usd == 20000 and r.max_top10_pct == 30
