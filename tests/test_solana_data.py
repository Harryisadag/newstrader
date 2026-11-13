"""The Solana paper desk's data layer: the parsers (checked against real answers saved in tests/fixtures/solana),
the shared rate limiter (429s, x-ratelimit-reset, Retry-After, backoff) and the read-only data client. Offline."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from newstrader.solana import parsers as P
from newstrader.solana.data import GECKO, JUPITER_QUOTE, PUBLIC_RPC, SolanaData
from newstrader.solana.limiter import Limiter, told_wait

FIX = Path(__file__).parent / "fixtures" / "solana"
MINT = "5N3r46YNyCuCsTsVZ4LXnzZuNqgJQ15pWm2cvemcrw5d"
POOL = "9Lvrh9mvYnKfgbWZ1fzHGyusqJZrSi1f5cfTNDdxDWno"
WALL0 = 1_794_598_900.0  # a wall-clock time just before the samples were taken


def fixture(name: str):
    return json.loads((FIX / f"{name}.json").read_text())


class FakeClock:
    """A monotonic clock, a wall clock and a sleep that only moves the clocks forward."""

    def __init__(self, wall: float = WALL0):
        self.t = 100.0
        self.wall0 = wall
        self.slept: list[float] = []

    def mono(self) -> float:
        return self.t

    def wall(self) -> float:
        return self.wall0 + self.t - 100.0

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.t += seconds


def data_with(handler, clock: FakeClock | None = None, rpc_url: str = "") -> tuple[SolanaData, FakeClock]:
    clock = clock or FakeClock()
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return SolanaData(client, rpc_url=rpc_url, clock=clock.mono, wall=clock.wall, sleep=clock.sleep), clock


# ---------------------------------------------------------------------------------------------- GeckoTerminal
def test_new_pools_fixture_parses():
    pools = P.parse_pools(fixture("gecko_new_pools"))
    assert len(pools) == 3
    p = pools[0]
    assert p.address == POOL and p.mint == MINT and p.quote_mint == P.SOL_MINT
    assert p.name == "M&F / SOL" and p.symbol == "M&F" and p.decimals == 6  # html entities undone
    assert p.dex == "pump-fun" and p.created_at == datetime(2026, 11, 13, 19, 41, 57, tzinfo=UTC)
    assert p.reserve_usd is None and p.fdv_usd == pytest.approx(2276.483098)
    assert p.window("m5") == {"buys": 1, "sells": 0, "buyers": 1, "sellers": 0}
    assert p.change["m5"] == 0 and p.volume["m5"] == pytest.approx(39.5573498324)
    assert p.price_sol == pytest.approx(p.price_usd / p.quote_price_usd)
    assert pools[2].dex == "pumpswap" and pools[2].mint == "4vPaMXGATRDRWZ17sK4JnwUV4jhKZV3TrsGiCXWWiXp9"


def test_one_pool_and_multi_fixtures_parse():
    p = P.parse_pool(fixture("gecko_pool"))
    assert p.reserve_usd == pytest.approx(1810.0009) and p.graduation_pct == 0.11 and p.graduated is False
    assert p.price_native == pytest.approx(0.00000002846757862) and p.price_sol == p.price_native
    assert p.sol_usd == pytest.approx(80.167, abs=0.01)
    multi = P.parse_pools(fixture("gecko_pools_multi"))
    assert [x.address for x in multi] == [POOL] and multi[0].reserve_usd == pytest.approx(1810.0009)


def test_gecko_errors():
    with pytest.raises(P.RateLimited):
        P.parse_pools(fixture("gecko_trending_pumpswap"), 429)
    with pytest.raises(P.RateLimited):  # the 429 body even with a 200
        P.check_gecko(fixture("gecko_trending_pumpswap"))
    with pytest.raises(P.DataError, match="title"):
        P.parse_pools({"errors": [{"status": "404", "title": "title"}]}, 404)
    with pytest.raises(P.DataError):
        P.parse_pool({"data": []})
    bad = fixture("gecko_new_pools")
    bad["data"][0]["attributes"]["address"] = "not/an address?x=1"
    bad["data"][0]["id"] = "solana_../../evil"
    assert len(P.parse_pools(bad)) == 2  # never put a strange address into a URL


# ---------------------------------------------------------------------------------------------- the Solana network
def test_mint_account_fixture_parses():
    m = P.parse_mint_account(fixture("rpc_mint_account"), MINT)
    assert m.program == "spl-token-2022" and m.owner.startswith("TokenzQd")
    assert m.mint_authority is None and m.freeze_authority is None and m.missing == ()
    assert m.decimals == 6 and m.supply == 1_000_000_000_000_000
    assert [e["extension"] for e in m.extensions] == ["metadataPointer", "tokenMetadata"]


def test_mint_account_missing_fields_and_errors():
    raw = fixture("rpc_mint_account")
    del raw["result"]["value"]["data"]["parsed"]["info"]["freezeAuthority"]
    assert P.parse_mint_account(raw, MINT).missing == ("freezeAuthority",)
    raw = fixture("rpc_mint_account")
    raw["result"]["value"]["data"]["parsed"]["info"]["mintAuthority"] = "Auth1111111111111111111111111111111111111"
    assert P.parse_mint_account(raw, MINT).mint_authority.startswith("Auth")
    with pytest.raises(P.RateLimited):
        P.parse_mint_account(fixture("rpc_largest_accounts"), MINT)  # {"error": {"code": 429}}
    with pytest.raises(P.DataError, match="no coin"):
        P.parse_mint_account({"jsonrpc": "2.0", "result": {"value": None}}, MINT)
    with pytest.raises(P.DataError, match="isn't a coin"):
        P.parse_mint_account(fixture("rpc_owner_BwWK"), MINT)  # a plain wallet, not a mint
    with pytest.raises(P.DataError):
        P.parse_mint_account(fixture("rpc_publicnode_largest_accounts"), MINT)


def test_largest_accounts_parse():
    rows = P.parse_largest_accounts({"jsonrpc": "2.0", "result": {"value": [
        {"address": "7d8mCLpKEyhCNNpxG9kjzZANEroSwKvVZb4jY1Apg5w3", "amount": "999095759795168", "decimals": 6}]}})
    assert rows == [{"address": "7d8mCLpKEyhCNNpxG9kjzZANEroSwKvVZb4jY1Apg5w3", "amount": 999095759795168}]


# ---------------------------------------------------------------------------------------------- Jupiter
def test_quotes_fixture_parse():
    buy = P.parse_quote(fixture("jupiter_quote_buy"))
    sell = P.parse_quote(fixture("jupiter_quote_sell"))
    assert buy.in_amount == 100_000_000 and buy.out_amount == 3514996623441 and buy.labels == ["Pump.fun"]
    assert sell.in_amount == buy.out_amount and sell.out_amount == 96893417
    assert sell.price_impact == pytest.approx(0.0310411, abs=1e-6)  # a fraction: 3.1%
    assert sell.to_dict()["price_impact_pct"] == pytest.approx(3.1041, abs=1e-3)
    assert 1 - sell.out_amount / buy.in_amount == pytest.approx(0.0311, abs=1e-4)  # the round trip loses ~3.1%
    bonk = P.parse_quote(fixture("jupiter_quote_buy_bonk"))
    assert bonk.labels == ["Whirlpool", "GoonFi V2", "Scorch"]


def test_quote_errors():
    with pytest.raises(P.NoRoute) as e:
        P.parse_quote(fixture("jupiter_quote_usdc"), 400)
    assert e.value.code == "TOKEN_NOT_TRADABLE"
    with pytest.raises(P.NoRoute) as e:
        P.parse_quote({"error": "No routes found", "errorCode": "NO_ROUTES_FOUND"}, 400)
    assert e.value.code == "NO_ROUTES_FOUND"
    with pytest.raises(P.DataError) as e:
        P.parse_quote({"error": "Invalid amount", "errorCode": "INVALID_AMOUNT"}, 400)
    assert not isinstance(e.value, P.NoRoute)
    zero = fixture("jupiter_quote_sell")
    zero["outAmount"] = "0"
    with pytest.raises(P.NoRoute):
        P.parse_quote(zero)
    gone = fixture("jupiter_quote_sell")
    del gone["outAmount"]
    with pytest.raises(P.DataError):
        P.parse_quote(gone)


def test_prices_and_token_info_parse():
    prices = P.parse_prices(fixture("jupiter_price"))
    assert prices[P.SOL_MINT]["usd"] == pytest.approx(80.0757, abs=1e-3)
    assert prices[MINT]["usd"] == pytest.approx(2.278e-6, rel=1e-3) and prices[MINT]["launchpad"] == "pump.fun"
    info = P.parse_token_search(fixture("jupiter_token_search"), MINT)
    assert info["mint_authority_disabled"] is True and info["freeze_authority_disabled"] is True
    assert info["dev_mints"] == 3 and info["holders"] == 2 and info["organic_label"] == "low"
    assert P.parse_token_search([], MINT) is None  # Jupiter doesn't list it (yet)
    with pytest.raises(P.DataError):
        P.parse_token_search({"error": "x"}, MINT)


# ---------------------------------------------------------------------------------------------- RugCheck
def test_rugcheck_report_parse():
    rep = P.parse_rugcheck(fixture("rugcheck_report_full"), MINT)
    assert rep["creator"] == "4tu57EX6x3WV21q9ZSGejKWkXrQPTst3iQDDg1yNqS9x"
    assert rep["markets"][0]["lp"]["lpLockedPct"] == 100 and rep["graphInsidersDetected"] == 0
    with pytest.raises(P.DataError, match="different coin"):
        P.parse_rugcheck(fixture("rugcheck_report_full"), "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263")
    with pytest.raises(P.DataError):
        P.parse_rugcheck({"error": "not found"}, MINT)
    assert P.no_one(None) and P.no_one("") and P.no_one(P.SYSTEM_PROGRAM) and not P.no_one(MINT)


def test_times_and_numbers():
    assert P.when("2026-11-13T19:41:58.010591179Z") == datetime(2026, 11, 13, 19, 41, 58, 10591, tzinfo=UTC)
    assert P.when(1794598969) == P.when(1794598969000) == P.when("1794598969")
    assert P.when("") is None and P.when("soon") is None and P.when(0) is None
    assert P.num("1.5") == 1.5 and P.num("nan") is None and P.num(True) is None and P.integer("12") == 12


def test_rpc_url_check():
    assert P.check_rpc_url("") == ""
    assert P.check_rpc_url(" https://rpc.example.com/?api-key=abc ") == "https://rpc.example.com/?api-key=abc"
    assert P.rpc_host("https://rpc.example.com/v2/abc?api-key=secret") == "rpc.example.com"
    for bad in ("http://rpc.example.com", "rpc.example.com", "https://", "https://a b.com", "ftp://x"):
        with pytest.raises(ValueError):
            P.check_rpc_url(bad)


# ---------------------------------------------------------------------------------------------- the rate limiter
async def test_limiter_spaces_calls():
    c = FakeClock()
    lim = Limiter("Jupiter", 1.5, clock=c.mono, wall=c.wall, sleep=c.sleep)
    for _ in range(3):
        await lim.acquire()
    assert c.slept == [1.5, 1.5] and lim.calls == 3
    assert lim.wait_time() == pytest.approx(1.5)


async def test_limiter_honours_x_ratelimit_reset():
    headers = fixture("jupiter_429_headers")  # x-ratelimit-reset: 1794598969 (epoch seconds)
    c = FakeClock(wall=1794598969 - 37.0)
    lim = Limiter("Jupiter", 1.5, clock=c.mono, wall=c.wall, sleep=c.sleep)
    await lim.acquire()
    assert lim.hit_limit(headers) == pytest.approx(37.0)
    assert lim.cooling_down() == pytest.approx(37.0)
    with pytest.raises(P.RateLimited, match="pausing"):
        await lim.acquire(max_wait=5)  # an exit check just skips
    await lim.acquire(max_wait=60)  # a patient caller waits until the reset
    assert c.wall() >= 1794598969 and lim.limited == 1


async def test_limiter_retry_after_and_backoff():
    c = FakeClock()
    lim = Limiter("RugCheck", 1.0, clock=c.mono, wall=c.wall, sleep=c.sleep, max_backoff=30)
    assert lim.hit_limit({"Retry-After": "7"}) == 7
    assert told_wait({"retry-after": "Fri, 13 Nov 2026 19:41:57 GMT"}, datetime(2026, 11, 13, 19, 41, 50,
                                                                                tzinfo=UTC).timestamp()) == 7
    assert told_wait({"x-ratelimit-reset": "12"}, c.wall()) == 12  # seconds from now
    assert told_wait({"x-ratelimit-reset": str(int((c.wall() + 9) * 1000))}, c.wall()) == pytest.approx(9)
    assert told_wait({}, c.wall()) is None and told_wait(None, c.wall()) is None
    assert [lim.hit_limit() for _ in range(6)] == [2, 4, 8, 16, 30, 30]  # no hint: doubles, capped
    lim.ok()
    assert lim.hit_limit({"x": "y"}) == 2  # a normal answer starts the backoff again


async def test_limiter_per_minute_budget():
    c = FakeClock()
    lim = Limiter("GeckoTerminal", 0.0, per_minute=25, clock=c.mono, wall=c.wall, sleep=c.sleep)
    for _ in range(25):
        await lim.acquire()
    assert lim.free() == 0 and c.t == 100.0
    with pytest.raises(P.RateLimited, match="busy"):
        await lim.acquire(max_wait=10)
    await lim.acquire(max_wait=61)
    assert c.t == pytest.approx(160.0)


# ---------------------------------------------------------------------------------------------- the data client
async def test_quote_call_and_jupiter_429():
    seen = []
    headers = fixture("jupiter_429_headers")

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if len(seen) == 1:
            return httpx.Response(200, json=fixture("jupiter_quote_buy"))
        return httpx.Response(429, json={"code": 429, "message": "[API Gateway] Too many requests"}, headers=headers)

    data, c = data_with(handler, FakeClock(wall=1794598969 - 20.0))
    q = await data.quote(P.SOL_MINT, MINT, 100_000_000)
    assert q.out_amount == 3514996623441
    url = seen[0].url
    assert str(url).startswith(JUPITER_QUOTE) and url.params["amount"] == "100000000"
    assert url.params["inputMint"] == P.SOL_MINT and url.params["outputMint"] == MINT
    with pytest.raises(P.RateLimited, match="too many requests"):
        await data.quote(MINT, P.SOL_MINT, q.out_amount)
    assert data.jupiter.cooling_down() == pytest.approx(18.5, abs=0.1)  # until x-ratelimit-reset
    with pytest.raises(P.RateLimited):
        await data.quote(MINT, P.SOL_MINT, q.out_amount, max_wait=3)
    assert len(seen) == 2 and data.health["jupiter"]["limited"] == 2
    await data.close()


async def test_no_route_is_an_answer_not_an_outage():
    def handler(request):
        return httpx.Response(400, json={"error": "No routes found", "errorCode": "NO_ROUTES_FOUND"})

    data, _ = data_with(handler)
    with pytest.raises(P.NoRoute):
        await data.quote(MINT, P.SOL_MINT, 5)
    assert data.health["jupiter"]["error"] is None and data.jupiter.cooling_down() == 0


async def test_gecko_calls_and_429():
    calls = []

    def handler(request):
        calls.append(str(request.url))
        if "new_pools" in str(request.url):
            return httpx.Response(200, json=fixture("gecko_new_pools"))
        if "/multi/" in str(request.url):
            return httpx.Response(200, json=fixture("gecko_pools_multi"))
        return httpx.Response(429, json=fixture("gecko_trending_pumpswap"))

    data, _ = data_with(handler)
    assert len(await data.new_pools()) == 3
    assert calls[0].startswith(f"{GECKO}/new_pools") and "include=base_token%2Cdex" in calls[0]
    multi = await data.pools([POOL, POOL])
    assert list(multi) == [POOL] and calls[1] == f"{GECKO}/pools/multi/{POOL}?include=base_token"
    with pytest.raises(P.RateLimited):
        await data.pool(POOL)
    assert data.gecko.cooling_down() == 2.0  # no hint in the answer: back off 2 s, then 4, 8...
    with pytest.raises(P.DataError):
        await data.pools([])


async def test_rpc_calls_public_and_own():
    bodies = []

    def handler(request):
        bodies.append((str(request.url), json.loads(request.content)))
        if json.loads(request.content)["method"] == "getTokenLargestAccounts":
            return httpx.Response(200, json=fixture("rpc_largest_accounts"))
        return httpx.Response(200, json=fixture("rpc_mint_account"))

    data, _ = data_with(handler)
    m = await data.mint_account(MINT)
    assert m.decimals == 6 and bodies[0][0] == PUBLIC_RPC
    assert bodies[0][1]["params"] == [MINT, {"encoding": "jsonParsed"}]
    with pytest.raises(P.DataError, match="your own RPC"):
        await data.largest_accounts(MINT)  # the public RPC always refuses: not even asked
    assert len(bodies) == 1
    data.rpc_url = "https://rpc.example.com/v2/abc?api-key=hidden"
    with pytest.raises(P.RateLimited) as e:
        await data.largest_accounts(MINT)
    assert bodies[-1][0] == data.rpc_url and data.rpc.cooling_down() > 0
    assert "hidden" not in data.rpc_label and "rpc.example.com" in data.rpc_label
    assert "hidden" not in str(e.value)


async def test_timeouts_and_bad_addresses_fail_without_guessing():
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout("timed out", request=request)

    data, _ = data_with(handler)
    with pytest.raises(P.DataError, match="didn't answer"):
        await data.rugcheck(MINT)
    assert data.health["rugcheck"]["error"]
    for call in (data.rugcheck("../../v1/x"), data.quote(P.SOL_MINT, "abc?x=1", 5), data.pool("a/b"),
                 data.mint_account("")):
        with pytest.raises(P.DataError):
            await call
    assert len(calls) == 1
    with pytest.raises(P.DataError):
        await data.quote(P.SOL_MINT, MINT, 0)


async def test_rugcheck_and_token_info_calls():
    def handler(request):
        if "rugcheck" in str(request.url):
            return httpx.Response(200, json=fixture("rugcheck_report_full"))
        if "tokens/v2/search" in str(request.url):
            assert request.url.params["query"] == MINT
            return httpx.Response(200, json=fixture("jupiter_token_search"))
        return httpx.Response(200, json=fixture("jupiter_price"))

    data, _ = data_with(handler)
    assert (await data.rugcheck(MINT))["mint"] == MINT
    assert (await data.token_info(MINT))["dev_mints"] == 3
    assert await data.sol_usd() == pytest.approx(80.0757, abs=1e-3)
