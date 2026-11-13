"""The seven gates of the pre-flight checklist.

Each gate is a pure function over the evidence the checklist fetched, and returns a GateResult: passed or not, why in
plain English, when, and the numbers it looked at. There are no scores: a coin must pass every gate, and a pass never
outvotes a fail. Anything that couldn't be fetched, timed out, is missing or makes no sense is a FAIL ("fail closed"),
unless a gate says otherwise (Gate 5 lets an unknown creator through at half size).

They're numbered 1-7 for people, but run cheapest first, to spare Jupiter's tiny allowance: 1, 3, 4, 5, then 2, 6, 7.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timedelta
from typing import Any

from .parsers import LAMPORTS, SYSTEM_PROGRAM, MintInfo, Pool, Quote, integer, no_one, num, when

GATES = {1: "Authority", 2: "Sell check", 3: "Liquidity", 4: "Holders", 5: "Creator", 6: "Chase + impact",
         7: "Freshness"}
RUN_ORDER = (1, 3, 4, 5, 2, 6, 7)
EXPLAIN = {
    1: "Nobody can make more of the coin or freeze your coins (mint and freeze authority both given up).",
    2: "It can really be sold: no trick features in the coin itself, and a buy-then-sell-everything quote on Jupiter "
       "works and loses less than 10%.",
    3: "The pool's money can't be pulled: its liquidity is burned or locked for 30+ days, the pool is big enough, "
       "and your order is at most 2% of it.",
    4: "No whale: leaving out the pool, the top 10 real holders own under 30% and nobody over 10% (wallets owned by "
       "one person, and insiders, count as one).",
    5: "The creator isn't a known rugger or a serial launcher (3+ coins in 24 hours). An unknown creator passes at "
       "half size.",
    6: "Right before the paper buy, a fresh Jupiter quote: the price ran up at most 15% since the coin was spotted "
       "and the buy moves the price at most 3%.",
    7: "Every piece of evidence is at most 10 seconds old and error-free at the moment of the paper buy.",
}
HARMLESS_EXTENSIONS = {"metadataPointer", "tokenMetadata", "groupPointer", "groupMemberPointer", "tokenGroup",
                       "tokenGroupMember", "immutableOwner"}
EXTENSION_WHY = {
    "transferHook": "a transfer hook (the creator's own program runs on every transfer and can block sales)",
    "pausableConfig": "a pause switch (the creator can stop all transfers)",
    "permanentDelegate": "a permanent delegate (someone can move or burn anyone's coins)",
    "nonTransferable": "it can't be transferred at all",
    "defaultAccountState": "new holders start frozen",
    "confidentialTransferMint": "hidden (confidential) transfers",
    "confidentialTransferFeeConfig": "hidden (confidential) transfer fees",
    "mintCloseAuthority": "a close authority on the coin",
    "interestBearingConfig": "an 'interest' setting that changes how balances look",
    "scaledUiAmountConfig": "a multiplier that changes how balances look",
}
TOKEN_PROGRAMS = {"spl-token", "spl-token-2022"}


@dataclass
class Rules:
    """The gates' limits (Settings -> Solana)."""

    max_round_trip_loss_pct: float = 10.0
    max_transfer_fee_pct: float = 10.0
    min_lp_locked_pct: float = 99.0
    min_lock_days: float = 30.0
    min_liquidity_usd: float = 8000.0
    max_order_pct_of_pool: float = 2.0
    max_top10_pct: float = 30.0
    max_single_holder_pct: float = 10.0
    max_creator_launches_24h: int = 3
    max_chase_pct: float = 15.0
    max_price_impact_pct: float = 3.0
    max_evidence_age_seconds: float = 10.0

    @classmethod
    def from_settings(cls, s) -> Rules:
        return cls(**{f.name: getattr(s, f.name) for f in fields(cls) if hasattr(s, f.name)})


@dataclass
class Fetched:
    """One piece of evidence: what a source said (or why it didn't), and when."""

    source: str
    value: Any = None
    error: str = ""
    at: datetime | None = None
    kind: str = ""  # for an error: "no_route", "rate_limited" or "error"

    @property
    def ok(self) -> bool:
        return not self.error and self.at is not None


@dataclass
class Evidence:
    """Everything the gates look at for one coin. None = not fetched (yet)."""

    mint: str
    pool: str
    order_sol: float
    signal_price_sol: float | None
    mint_info: Fetched | None = None  # the Solana network: the coin's mint account
    rug: Fetched | None = None  # RugCheck's report
    token: Fetched | None = None  # Jupiter's token info (an extra check on Gate 1)
    pool_now: Fetched | None = None  # GeckoTerminal: the pool again, at check time
    holders_rpc: Fetched | None = None  # your own RPC's 20 biggest holder accounts (only with your own RPC)
    buy: Fetched | None = None  # Gate 2: Jupiter quote, SOL -> coin
    sell: Fetched | None = None  # Gate 2: Jupiter quote, all of it -> SOL
    final: Fetched | None = None  # Gate 6: the final Jupiter buy quote

    def pieces(self) -> dict[str, Fetched]:
        names = ("mint_info", "rug", "token", "pool_now", "holders_rpc", "buy", "sell", "final")
        return {n: getattr(self, n) for n in names if getattr(self, n) is not None}


@dataclass
class GateResult:
    gate: int
    name: str
    passed: bool
    reason: str
    checked_at: datetime
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        out = asdict(self)
        out["checked_at"] = self.checked_at.isoformat()
        return out


def _result(gate: int, passed: bool, reason: str, now: datetime, **data) -> GateResult:
    return GateResult(gate, GATES[gate], passed, reason, now, data)


def _problem(f: Fetched | None, what: str) -> str | None:
    """Why a piece of evidence can't be used (None = it can)."""
    if f is None:
        return f"{what} wasn't checked"
    if f.error:
        return f"Couldn't check {what}: {f.error}"
    if f.at is None or f.value is None:
        return f"{what} gave no answer"
    return None


def _short(address) -> str:
    text = str(address or "")
    return f"{text[:4]}...{text[-4:]}" if len(text) > 12 else text


# ---------------------------------------------------------------------------------------------- Gate 1: Authority
def gate_authority(ev: Evidence, rules: Rules, now: datetime) -> GateResult:
    """Mint and freeze authority both given up, on the Solana network and in RugCheck's report (Jupiter's audit too,
    when Jupiter lists the coin). Any disagreement, or anything unknown, fails."""
    g = 1
    why = _problem(ev.mint_info, "the coin on the Solana network")
    if why:
        return _result(g, False, why, now)
    m: MintInfo = ev.mint_info.value
    unknown = [k for k in ("mintAuthority", "freezeAuthority") if k in m.missing]
    if unknown:
        return _result(g, False, f"The Solana network didn't say who controls the coin ({', '.join(unknown)} "
                                 "unknown)", now)
    data = {"mint_authority": m.mint_authority, "freeze_authority": m.freeze_authority}
    if not no_one(m.mint_authority):
        return _result(g, False, f"Someone ({_short(m.mint_authority)}) can still make more of this coin, so your "
                                 "share can be watered down to nothing", now, **data)
    if not no_one(m.freeze_authority):
        return _result(g, False, f"Someone ({_short(m.freeze_authority)}) can freeze holders' coins so they can't "
                                 "sell", now, **data)
    why = _problem(ev.rug, "RugCheck's report")
    if why:
        return _result(g, False, why, now, **data)
    token = ev.rug.value.get("token")
    if not isinstance(token, dict) or "mintAuthority" not in token or "freezeAuthority" not in token:
        return _result(g, False, "RugCheck's report didn't say who controls the coin, so it couldn't be "
                                 "double-checked", now, **data)
    if not no_one(token.get("mintAuthority")) or not no_one(token.get("freezeAuthority")):
        return _result(g, False, "RugCheck says someone still controls the coin, but the Solana network doesn't - "
                                 "the two disagree, so it fails", now, **data,
                       rugcheck={"mint": token.get("mintAuthority"), "freeze": token.get("freezeAuthority")})
    why = _problem(ev.token, "Jupiter's token info") if ev.token is None or ev.token.error else None
    if why:
        return _result(g, False, why, now, **data)
    info = ev.token.value if ev.token is not None else None
    note = ""
    if info and info.get("has_audit"):
        flags = (info.get("mint_authority_disabled"), info.get("freeze_authority_disabled"))
        if flags != (True, True):
            return _result(g, False, "Jupiter's audit doesn't confirm that the mint and freeze authority are given "
                                     "up - the sources disagree, so it fails", now, **data, jupiter=flags)
        note = ", RugCheck and Jupiter"
    else:
        note = " and RugCheck (Jupiter has no audit of it yet)"
    data["jupiter_audit"] = bool(info and info.get("has_audit"))
    return _result(g, True, f"Nobody can make more of it or freeze anyone's coins (checked on the Solana network{note})",
                   now, **data)


# ---------------------------------------------------------------------------------------------- Gate 2: Sell check
def _transfer_fee(state: dict) -> tuple[float | None, Any]:
    """A Token-2022 transfer fee (its highest rate, %) and who can change it."""
    rates = [integer((state.get(k) or {}).get("transferFeeBasisPoints")) for k in ("olderTransferFee", "newerTransferFee")
             if isinstance(state.get(k), dict)]
    rates = [r for r in rates if r is not None]
    return (max(rates) / 100 if rates else None), state.get("transferFeeConfigAuthority")


def _mechanism(ev: Evidence, rules: Rules) -> tuple[str | None, dict]:
    """(why it fails, data) for the coin's own features (Token-2022 extensions and transfer fee)."""
    m: MintInfo = ev.mint_info.value
    data: dict = {"program": m.program, "extensions": [e.get("extension") for e in m.extensions]}
    if m.program not in TOKEN_PROGRAMS:
        return f"It isn't a normal Solana coin (program '{m.program or 'unknown'}')", data
    if "extensions" in m.missing:
        return "The Solana network's list of the coin's features made no sense", data
    for ext in m.extensions:
        name, state = ext.get("extension"), ext.get("state") or {}
        if name in HARMLESS_EXTENSIONS:
            continue
        if name == "defaultAccountState" and isinstance(state, dict) and state.get("accountState") == "initialized":
            continue
        if name == "transferFeeConfig" and isinstance(state, dict):
            fee, authority = _transfer_fee(state)
            data["transfer_fee_pct"] = fee
            if fee is None:
                return "It has a transfer fee of unknown size", data
            if not no_one(authority):
                return (f"It has a transfer fee ({fee:g}% now) that someone can still change - up to 100% when you "
                        "try to sell"), data
            if fee > rules.max_transfer_fee_pct:
                return f"Every transfer pays a {fee:g}% fee (the limit is {rules.max_transfer_fee_pct:g}%)", data
            continue
        return f"The coin has {EXTENSION_WHY.get(name, f'an unusual feature ({name})')}", data
    rug = ev.rug.value
    exts = rug.get("token_extensions")
    if m.program == "spl-token-2022" and not isinstance(exts, dict):
        return "RugCheck didn't list the coin's Token-2022 features, so they couldn't be double-checked", data
    for name, value in (exts or {}).items():
        if value in (None, False, {}, []) or name in HARMLESS_EXTENSIONS or name == "transferFeeConfig":
            continue
        if name == "defaultAccountState" and isinstance(value, dict) and value.get("accountState") == "initialized":
            continue
        return (f"RugCheck says the coin has {EXTENSION_WHY.get(name, f'an unusual feature ({name})')} - the "
                "sources disagree, so it fails"), data
    fee = rug.get("transferFee")
    if isinstance(fee, dict):
        pct = num(fee.get("pct")) or 0.0
        if not no_one(fee.get("authority")):
            return "RugCheck says someone can still change the coin's transfer fee", data
        if pct > rules.max_transfer_fee_pct:
            return f"RugCheck says every transfer pays a {pct:g}% fee", data
    return None, data


def gate_sell(ev: Evidence, rules: Rules, now: datetime) -> GateResult:
    """(a) the coin itself has no trick that blocks or taxes sales, and (b) a real round trip quote on Jupiter -
    buy the order size, then sell ALL of it back - works and loses less than the limit."""
    g = 2
    why = _problem(ev.mint_info, "the coin on the Solana network") or _problem(ev.rug, "RugCheck's report")
    if why:
        return _result(g, False, why, now)
    why, data = _mechanism(ev, rules)
    if why:
        return _result(g, False, why, now, **data)
    if ev.buy is None or ev.buy.error:
        if ev.buy is not None and ev.buy.kind == "no_route":
            return _result(g, False, "Jupiter found no way to buy it right now", now, **data)
        return _result(g, False, _problem(ev.buy, "a Jupiter buy quote"), now, **data)
    if ev.sell is None or ev.sell.error:
        if ev.sell is not None and ev.sell.kind == "no_route":
            return _result(g, False, "Jupiter found no way to sell it back - a coin you can't sell is the classic "
                                     "rug", now, **data)
        return _result(g, False, _problem(ev.sell, "a Jupiter sell quote"), now, **data)
    buy: Quote = ev.buy.value
    sell: Quote = ev.sell.value
    if sell.in_amount != buy.out_amount:
        return _result(g, False, "The sell quote wasn't for all the coins the buy would get", now, **data)
    loss = (1 - sell.out_amount / buy.in_amount) * 100
    data.update(round_trip_loss_pct=round(loss, 2), buy_route=buy.labels, sell_route=sell.labels,
                sell_impact_pct=None if sell.price_impact is None else round(sell.price_impact * 100, 2))
    if loss >= rules.max_round_trip_loss_pct:
        return _result(g, False, f"Buying and selling straight back would lose {loss:.1f}% (the limit is "
                                 f"{rules.max_round_trip_loss_pct:g}%) - too hard to sell", now, **data)
    return _result(g, True, f"No trick features, and buying then selling everything back on Jupiter would lose "
                            f"{loss:.1f}%", now, **data)


# ---------------------------------------------------------------------------------------------- Gate 3: Liquidity
def gate_liquidity(ev: Evidence, rules: Rules, now: datetime) -> GateResult:
    """Every market's liquidity burned or locked (99%+), any time lock 30+ days from opening, the pool (fetched again
    now) big enough, and the order at most 2% of it."""
    g = 3
    why = _problem(ev.rug, "RugCheck's report")
    if why:
        return _result(g, False, why, now)
    rug = ev.rug.value
    markets = rug.get("markets")
    if not isinstance(markets, list) or not markets:
        return _result(g, False, "RugCheck found no market for the coin, so the liquidity lock couldn't be checked",
                       now)
    locked = []
    for market in markets:
        pct = num(((market or {}).get("lp") or {}).get("lpLockedPct"))
        if pct is None:
            return _result(g, False, "RugCheck couldn't tell whether the pool's liquidity is locked", now)
        locked.append(pct)
        if pct < rules.min_lp_locked_pct:
            return _result(g, False, f"Only {pct:.0f}% of a pool's liquidity is burned or locked - the creator could "
                                     "pull the rest out", now, lp_locked_pct=locked)
    if "lockers" not in rug:
        return _result(g, False, "RugCheck didn't say whether the liquidity sits in a time lock", now,
                       lp_locked_pct=locked)
    unlocks = []
    for key, locker in (rug.get("lockers") or {}).items():
        opens = when((locker or {}).get("unlockDate")) if isinstance(locker, dict) else None
        if opens is None:
            return _result(g, False, f"A liquidity lock ({_short(key)}) has no known unlock date", now,
                           lp_locked_pct=locked)
        days = (opens - now).total_seconds() / 86400
        unlocks.append(round(days, 1))
        if days < rules.min_lock_days:
            return _result(g, False, f"A liquidity lock opens in {max(days, 0):.0f} days (it must be at least "
                                     f"{rules.min_lock_days:g})", now, lp_locked_pct=locked, unlock_days=unlocks)
    why = _problem(ev.pool_now, "the pool on GeckoTerminal")
    if why:
        return _result(g, False, why, now, lp_locked_pct=locked)
    pool: Pool = ev.pool_now.value
    data = {"lp_locked_pct": locked, "unlock_days": unlocks, "liquidity_usd": pool.reserve_usd}
    if pool.mint and pool.mint != ev.mint:
        return _result(g, False, "GeckoTerminal's pool is for a different coin", now, **data)
    if pool.reserve_usd is None:
        return _result(g, False, "GeckoTerminal doesn't know how much money is in the pool yet", now, **data)
    if pool.reserve_usd < rules.min_liquidity_usd:
        return _result(g, False, f"Only ${pool.reserve_usd:,.0f} in the pool (the minimum is "
                                 f"${rules.min_liquidity_usd:,.0f})", now, **data)
    if not pool.sol_usd:
        return _result(g, False, "The SOL price in the pool's data is missing", now, **data)
    share = ev.order_sol * pool.sol_usd / pool.reserve_usd * 100
    data["order_pct_of_pool"] = round(share, 3)
    if share > rules.max_order_pct_of_pool:
        return _result(g, False, f"Your order would be {share:.1f}% of the pool (the limit is "
                                 f"{rules.max_order_pct_of_pool:g}%) - too big a splash", now, **data)
    return _result(g, True, f"Liquidity burned or locked, ${pool.reserve_usd:,.0f} in the pool, your order is "
                            f"{share:.2f}% of it", now, **data)


# ---------------------------------------------------------------------------------------------- Gate 4: Holders
def _addresses(value) -> set[str]:
    """Every address mentioned in an insider network (RugCheck's format isn't fixed, so look everywhere)."""
    out: set[str] = set()
    if isinstance(value, str):
        if 32 <= len(value) <= 44:
            out.add(value)
    elif isinstance(value, list):
        for v in value:
            out |= _addresses(v)
    elif isinstance(value, dict):
        for k, v in value.items():
            if k in ("address", "owner", "wallet", "account") or isinstance(v, list | dict):
                out |= _addresses(v)
    return out


def pool_accounts(rug: dict) -> set[str]:
    """Accounts that are pools, lockers or "nobody" - not people - so they don't count as holders."""
    out = {SYSTEM_PROGRAM}
    for address, known in (rug.get("knownAccounts") or {}).items():
        if isinstance(known, dict) and str(known.get("type") or "").upper() in ("AMM", "LOCKER"):
            out.add(address)
    for market in rug.get("markets") or []:
        if not isinstance(market, dict):
            continue
        for key in ("pubkey", "liquidityA", "liquidityB"):
            if market.get(key):
                out.add(market[key])
        for key in ("liquidityAAccount", "liquidityBAccount"):
            owner = (market.get(key) or {}).get("owner") if isinstance(market.get(key), dict) else None
            if owner:
                out.add(owner)
    for address, locker in (rug.get("lockers") or {}).items():
        out.add(address)
        if isinstance(locker, dict) and locker.get("tokenAccount"):
            out.add(locker["tokenAccount"])
    return out


def real_holders(rug: dict) -> tuple[list[dict], list[dict]]:
    """RugCheck's top holders as people: pools and lockers left out, accounts of one wallet added together, and every
    insider merged into ONE holder. -> (holders biggest first, the accounts left out)."""
    skip = pool_accounts(rug)
    insiders: set[str] = set()
    for network in rug.get("insiderNetworks") or []:
        insiders |= _addresses(network)
    groups: dict[str, dict] = {}
    left_out = []
    for h in rug.get("topHolders") or []:
        if not isinstance(h, dict):
            continue
        address, owner = h.get("address"), h.get("owner") or h.get("address")
        pct = num(h.get("pct")) or 0.0
        if address in skip or owner in skip:
            left_out.append({"address": address, "owner": owner, "pct": pct})
            continue
        insider = bool(h.get("insider")) or address in insiders or owner in insiders
        key = "insiders" if insider else owner
        grp = groups.setdefault(key, {"owner": key, "pct": 0.0, "accounts": 0, "insider": insider})
        grp["pct"] += pct
        grp["accounts"] += 1
    ranked = sorted(groups.values(), key=lambda x: -x["pct"])
    for grp in ranked:
        grp["pct"] = round(grp["pct"], 4)
    return ranked, left_out


def gate_holders(ev: Evidence, rules: Rules, now: datetime) -> GateResult:
    """The real top 10 (pools and lockers left out, one wallet's accounts and all insiders counted as one) own less
    than 30%, nobody over 10%, and RugCheck found no insider network. With your own RPC its holder list is checked
    too."""
    g = 4
    why = _problem(ev.rug, "RugCheck's report")
    if why:
        return _result(g, False, why, now)
    rug = ev.rug.value
    if not isinstance(rug.get("topHolders"), list) or not rug["topHolders"]:
        return _result(g, False, "RugCheck didn't list the coin's holders", now)
    if "graphInsidersDetected" not in rug:
        return _result(g, False, "RugCheck didn't say whether insiders were found", now)
    holders, left_out = real_holders(rug)
    top10 = round(sum(h["pct"] for h in holders[:10]), 2)
    biggest = holders[0] if holders else {"owner": "", "pct": 0.0, "insider": False}
    insiders = integer(rug.get("graphInsidersDetected")) or 0
    data = {"top10_pct": top10, "biggest_pct": biggest["pct"], "biggest": biggest["owner"], "insider_wallets": insiders,
            "left_out": [x["owner"] for x in left_out][:5], "holders": holders[:10]}
    if insiders > 0:
        return _result(g, False, f"RugCheck found {insiders} wallets linked to each other (insiders who can sell "
                                 "together)", now, **data)
    if biggest["pct"] > rules.max_single_holder_pct:
        who = "Insiders together hold" if biggest.get("insider") else f"One wallet ({_short(biggest['owner'])}) holds"
        return _result(g, False, f"{who} {biggest['pct']:.1f}% of the coins (the limit is "
                                 f"{rules.max_single_holder_pct:g}%)", now, **data)
    if top10 >= rules.max_top10_pct:
        return _result(g, False, f"The top 10 real holders own {top10:.1f}% (it must be under "
                                 f"{rules.max_top10_pct:g}%)", now, **data)
    if ev.holders_rpc is not None:
        why = _problem(ev.holders_rpc, "the holders on your RPC")
        if why:
            return _result(g, False, why, now, **data)
        supply = ev.mint_info.value.supply if ev.mint_info is not None and ev.mint_info.ok else None
        if not supply:
            return _result(g, False, "The coin's supply is unknown, so your RPC's holder list couldn't be read", now,
                           **data)
        skip = pool_accounts(rug)
        owners = {h.get("address"): h.get("owner") for h in rug["topHolders"] if isinstance(h, dict)}
        sizes: dict[str, float] = {}
        for row in ev.holders_rpc.value:
            owner = owners.get(row["address"]) or row["address"]
            if row["address"] in skip or owner in skip:
                continue
            sizes[owner] = sizes.get(owner, 0.0) + row["amount"] / supply * 100
        ranked = sorted(sizes.values(), reverse=True)
        data["rpc_top10_pct"] = round(sum(ranked[:10]), 2)
        if ranked and ranked[0] > rules.max_single_holder_pct:
            return _result(g, False, f"Your RPC shows one holder with {ranked[0]:.1f}% - more than RugCheck's list "
                                     "said, so it fails", now, **data)
        if data["rpc_top10_pct"] >= rules.max_top10_pct:
            return _result(g, False, f"Your RPC shows the top 10 holders owning {data['rpc_top10_pct']:.1f}%", now,
                           **data)
    return _result(g, True, f"The top 10 real holders own {top10:.1f}%, the biggest {biggest['pct']:.1f}%", now, **data)


# ---------------------------------------------------------------------------------------------- Gate 5: Creator
def gate_creator(ev: Evidence, rules: Rules, now: datetime) -> GateResult:
    """Fails on a creator RugCheck knows for rugs, a coin already marked rugged, or 3+ launches in 24 hours (dates
    nobody knows count as recent, and this coin counts too). An unknown creator, or one with no history, PASSES
    with the order size halved."""
    g = 5
    why = _problem(ev.rug, "RugCheck's report")
    if why:
        return _result(g, False, why, now)
    rug = ev.rug.value
    if "rugged" not in rug or not isinstance(rug.get("risks"), list):
        return _result(g, False, "RugCheck's report didn't say whether the coin or its creator is known for rugs", now)
    if rug.get("rugged") is True:
        return _result(g, False, "RugCheck already marks this coin as rugged", now)
    for risk in rug["risks"]:
        name = str((risk or {}).get("name") or "") if isinstance(risk, dict) else str(risk)
        if "creator history" in name.lower() and "rug" in name.lower():
            return _result(g, False, f"RugCheck: \"{name}\" - this creator has rugged before", now,
                           risk=risk if isinstance(risk, dict) else name)
    creator = rug.get("creator")
    history = rug.get("creatorTokens")
    if no_one(creator) or not history:
        who = "Nobody knows who made it" if no_one(creator) else "The creator has no history RugCheck knows"
        return _result(g, True, f"{who} - it passes, but at half the order size", now, creator=creator,
                       size_factor=0.5, unknown_creator=True)
    if not isinstance(history, list):
        return _result(g, False, "RugCheck's list of the creator's coins made no sense", now, creator=creator)
    recent = {ev.mint}
    for t in history:
        if not isinstance(t, dict):
            continue
        made = when(t.get("createdAt"))
        if made is None or now - made <= timedelta(hours=24):
            recent.add(str(t.get("mint") or id(t)))
    data = {"creator": creator, "launches_24h": len(recent), "known_coins": len(history), "size_factor": 1.0}
    if len(recent) >= rules.max_creator_launches_24h:
        return _result(g, False, f"The creator launched {len(recent)} coins in the last 24 hours (counting this one) - "
                                 "a serial launcher", now, **data)
    return _result(g, True, f"The creator isn't known for rugs and launched {len(recent)} coin"
                            f"{'' if len(recent) == 1 else 's'} in the last 24 hours", now, **data)


# ---------------------------------------------------------------------------------------------- Gate 6: Chase + impact
def gate_chase(ev: Evidence, rules: Rules, now: datetime) -> GateResult:
    """The FINAL Jupiter buy quote, right before the paper fill: the price ran up at most 15% since the coin was
    spotted, and the buy moves the price at most 3% (Jupiter gives the impact as a fraction: 0.03 = 3%)."""
    g = 6
    if ev.final is not None and ev.final.kind == "no_route":
        return _result(g, False, "Jupiter found no way to buy it any more", now)
    why = _problem(ev.final, "the final Jupiter quote") or _problem(ev.mint_info, "the coin's decimals")
    if why:
        return _result(g, False, why, now)
    q: Quote = ev.final.value
    decimals = ev.mint_info.value.decimals
    if decimals is None or not ev.signal_price_sol:
        return _result(g, False, "The price when the coin was spotted isn't known, so the chase can't be measured",
                       now)
    if q.price_impact is None:
        return _result(g, False, "Jupiter's quote didn't say how much the buy moves the price", now)
    price = (q.in_amount / LAMPORTS) / (q.out_amount / 10 ** decimals)
    chase = (price / ev.signal_price_sol - 1) * 100
    impact = abs(q.price_impact) * 100
    data = {"price_sol": price, "signal_price_sol": ev.signal_price_sol, "chase_pct": round(chase, 2),
            "impact_pct": round(impact, 3), "route": q.labels}
    if chase > rules.max_chase_pct:
        return _result(g, False, f"The price already ran up {chase:.1f}% since it was spotted (the limit is "
                                 f"{rules.max_chase_pct:g}%) - don't chase", now, **data)
    if impact > rules.max_price_impact_pct:
        return _result(g, False, f"The buy alone would move the price {impact:.1f}% (the limit is "
                                 f"{rules.max_price_impact_pct:g}%)", now, **data)
    return _result(g, True, f"Price {chase:+.1f}% since it was spotted, the buy moves it {impact:.2f}%", now, **data)


# ---------------------------------------------------------------------------------------------- Gate 7: Freshness
def gate_freshness(ev: Evidence, rules: Rules, now: datetime) -> GateResult:
    """At the moment of the paper buy, every piece of evidence is at most 10 s old and error-free."""
    g = 7
    errors, stale, ages = [], [], {}
    for name, f in ev.pieces().items():
        if f.error or f.at is None:
            errors.append(name)
            continue
        age = (now - f.at).total_seconds()
        ages[name] = round(age, 1)
        if age > rules.max_evidence_age_seconds:
            stale.append(name)
    data = {"ages_s": ages, "stale": stale, "errors": errors, "stale_only": bool(stale) and not errors,
            "oldest_s": max(ages.values()) if ages else None}
    if not ages:
        return _result(g, False, "There was no evidence to check", now, **data)
    if errors:
        return _result(g, False, f"Some evidence had errors ({', '.join(errors)})", now, **data)
    if stale:
        return _result(g, False, f"Some evidence was more than {rules.max_evidence_age_seconds:g} s old at the moment "
                                 f"of the buy (oldest {data['oldest_s']:.0f} s)", now, **data)
    return _result(g, True, f"All the evidence was at most {data['oldest_s']:.0f} s old", now, **data)


GATE_FUNCS = {1: gate_authority, 2: gate_sell, 3: gate_liquidity, 4: gate_holders, 5: gate_creator, 6: gate_chase,
              7: gate_freshness}


def run_gates(ev: Evidence, rules: Rules, now: datetime, order=RUN_ORDER) -> list[GateResult]:
    """Run the gates in order, stopping at the first that fails."""
    out = []
    for g in order:
        res = GATE_FUNCS[g](ev, rules, now)
        out.append(res)
        if not res.passed:
            break
    return out
