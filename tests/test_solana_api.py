"""The Solana paper desk's API, the status summary, the settings API and diagnostics. Offline (the fake data source
from test_solana_desk)."""

from __future__ import annotations

from datetime import timedelta

import pytest

from newstrader import diagnostics
from newstrader.solana import parsers as P
from newstrader.solana.service import SolanaDesk
from tests.test_solana_desk import NOW, Clock, FakeAlerts, FakeData, addr, hot_pool


class DiagData(FakeData):
    async def sol_usd(self, max_wait=20.0):
        self.calls.append("sol_usd")
        return 80.07


@pytest.fixture
def desk_env(ctx):
    clock = Clock()
    data = DiagData(clock)
    desk = SolanaDesk(ctx, data=data, now=clock.now, mono=clock.mono)
    ctx.services["solana"] = desk
    ctx.services["alerts"] = FakeAlerts()
    return desk, data, clock


def test_endpoints_need_the_desk(client):
    assert client.get("/api/solana").status_code == 503
    assert client.post("/api/solana/pause").status_code == 503
    status = client.get("/api/status").json()
    assert "solana" not in status  # the engine isn't running in this test app


def test_overview_while_off(client, ctx, desk_env):
    r = client.get("/api/solana")
    assert r.status_code == 200
    body = r.json()
    assert body["summary"]["enabled"] is False and body["summary"]["state"] == "off"
    assert body["wallet"]["sol"] == 10.0 and body["positions"] == [] and body["feed"] == []
    assert [g["gate"] for g in body["gates"]] == [1, 2, 3, 4, 5, 6, 7] and all(g["explain"] for g in body["gates"])
    assert body["reset_phrase"] == "RESET" and body["rpc_host"] == "the free public RPC"
    assert {s["source"] for s in body["sources"]} == {"gecko", "rpc", "jupiter", "rugcheck"}
    assert client.get("/api/status").json()["solana"]["state"] == "off"
    r = client.post("/api/solana/check-now")
    assert r.status_code == 409 and "off" in r.json()["detail"]
    assert desk_env[1].calls == []  # off: not a single call


def test_check_now_buy_and_candidate_details(client, ctx, desk_env):
    desk, data, clock = desk_env
    ctx.config.update({"solana": {"enabled": True}})
    data.new = [hot_pool()]
    r = client.post("/api/solana/check-now")
    assert r.status_code == 200 and r.json()["triggered"] == 1
    cand_id = ctx.db.scalar("SELECT id FROM sol_candidates")
    detail = client.get(f"/api/solana/candidates/{cand_id}").json()
    assert detail["status"] == "queued" and detail["gates"] == [] and detail["trigger"]["m5"]["buys"] == 20
    assert client.get("/api/solana/candidates/9999").status_code == 404
    body = client.get("/api/solana").json()
    assert body["feed"][0]["symbol"] and body["feed"][0]["lights"][0]["state"] == "not_run"
    assert body["last_feed"]["pools"] == 1


async def test_candidate_details_after_a_paper_buy(ctx, desk_env):
    desk, data, clock = desk_env
    ctx.config.update({"solana": {"enabled": True}})
    data.new = [hot_pool()]
    await desk.feed_tick()
    cand = await desk.work_once()
    detail = desk.candidate(cand["id"])
    assert [g["gate"] for g in detail["gates"]] == [1, 3, 4, 5, 2, 6, 7]
    assert detail["gates"][-1]["data"]["oldest_s"] is not None
    assert detail["position"]["status"] == "open" and detail["position"]["levels"]["stop_value"] > 0
    assert [f["side"] for f in detail["fills"]] == ["buy"]
    snap = desk.snapshot()
    assert snap["wallet"]["open"] == 1 and snap["positions"][0]["pnl_sol"] is not None
    assert snap["summary"]["state"] == "watching"


def test_pause_resume_and_reset(client, ctx, desk_env):
    ctx.config.update({"solana": {"enabled": True}})
    assert client.post("/api/solana/pause").json()["state"] == "paused"
    assert client.post("/api/solana/check-now").status_code == 409
    assert client.post("/api/solana/resume").json()["state"] == "watching"
    r = client.post("/api/solana/reset", json={"phrase": "reset"})
    assert r.status_code == 403 and "RESET" in r.json()["detail"]
    assert client.post("/api/solana/reset", json={"phrase": " RESET "}).json()["sol"] == 10.0
    assert desk_env[0].desk.wallet()["id"] == 2


def test_kill_switch_shows_on_the_desk(client, ctx, desk_env):
    ctx.config.update({"solana": {"enabled": True}})
    ctx.state.engage_kill_switch("test")
    s = client.get("/api/solana").json()["summary"]
    assert s["state"] == "killed" and s["killed"] is True and "no new paper buys" in s["text"]


def test_settings_api_has_the_desk_switched_off(client):
    body = client.get("/api/settings").json()
    assert body["settings"]["solana"]["enabled"] is False and body["settings"]["solana"]["order_sol"] == 0.1
    r = client.put("/api/settings", json={"solana": {"rpc_url": "http://rpc.example.com"}})
    assert r.status_code == 422 and any("https" in d for d in r.json()["details"])
    r = client.put("/api/settings", json={"solana": {"enabled": True, "rpc_url": "https://rpc.example.com/?k=1"}})
    assert r.status_code == 200 and r.json()["settings"]["solana"]["enabled"] is True


async def test_diagnostics_check_each_source_only_when_on(ctx, desk_env):
    desk, data, clock = desk_env
    assert await diagnostics._check_solana(ctx) == []
    ctx.config.update({"solana": {"enabled": True}})
    checks = await diagnostics._check_solana(ctx)
    names = [c["name"] for c in checks]
    assert names[0] == "Solana paper desk" and "PAPER ONLY" in checks[0]["detail"]
    assert all(c["status"] == "ok" for c in checks), checks
    assert any(n.startswith("Solana network (free public RPC)") for n in names) and len(checks) == 5
    data.fail["rugcheck"] = P.DataError("RugCheck didn't answer within 10 s")
    ctx.config.update({"solana": {"rpc_url": "https://rpc.example.com/v2/hidden-code"}})
    checks = await diagnostics._check_solana(ctx)
    bad = next(c for c in checks if c["name"].startswith("Solana: RugCheck"))
    assert bad["status"] == "error" and bad["fix"]
    assert any(c["name"] == "Solana network (rpc.example.com)" for c in checks)
    assert all("hidden-code" not in (c["name"] + c["detail"] + c["fix"]) for c in checks)


async def test_old_wallet_trades_stay_apart(ctx, desk_env):
    desk, data, clock = desk_env
    ctx.config.update({"solana": {"enabled": True}})
    data.new = [hot_pool()]
    await desk.feed_tick()
    await desk.work_once()
    desk.reset_wallet("RESET")
    clock.advance(25 * 3600)
    data.new = [hot_pool(address=addr(1), created_at=clock.t - timedelta(minutes=10))]
    data.pool_map[addr(1)] = hot_pool(address=addr(1), created_at=clock.t - timedelta(minutes=10))
    await desk.feed_tick()
    cand = await desk.work_once()
    assert cand["status"] == "bought" and cand["wallet"] == 2
    snap = desk.snapshot()
    assert len(snap["positions"]) == 1 and snap["closed"][0]["old_wallet"] is True
    assert snap["wallet"]["sol"] == pytest.approx(10 - 0.0505)
    assert NOW < clock.t
