from __future__ import annotations

import asyncio
import json
import logging

import httpx
import pytest

from newstrader.alerts.discord import build_payload
from newstrader.alerts.manager import AlertManager


class Recorder:
    def __init__(self, statuses=None):
        self.requests: list[dict] = []
        self.statuses = list(statuses or [204])

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        status = self.statuses.pop(0) if self.statuses else 204
        if status == 429:
            return httpx.Response(429, json={"retry_after": 0.01})
        return httpx.Response(status)


@pytest.fixture
async def setup(ctx):
    ctx.keys.update({"DISCORD_WEBHOOK_URL": "https://discord.com/api/webhooks/1/abc"})
    rec = Recorder()
    desktop_calls = []
    client = httpx.AsyncClient(transport=httpx.MockTransport(rec))
    mgr = AlertManager(ctx, http_client=client, desktop=lambda t, m, lvl: desktop_calls.append((t, m, lvl)) or True)
    ctx.loop = asyncio.get_running_loop()
    ctx.bus.bind_loop(ctx.loop)
    await mgr.start()
    yield ctx, mgr, rec, desktop_calls
    await mgr.stop()
    await client.aclose()


def test_payload_shape():
    p = build_payload("Bought 4 AAPL", "reason", "trade", "paper", {"Confidence": 88})
    e = p["embeds"][0]
    assert e["title"] == "Bought 4 AAPL" and e["color"] == 0xA855F7
    assert "paper" in e["footer"]["text"] and e["fields"][0]["value"] == "88"
    assert p["allowed_mentions"] == {"parse": []}  # never pings @everyone
    assert len(build_payload("x" * 500, "y" * 9000, "info")["embeds"][0]["title"]) == 256


async def test_alert_goes_to_discord_desktop_and_toast(setup):
    ctx, mgr, rec, desk = setup
    q = ctx.bus.subscribe()
    res = await mgr.send("trade_placed", "Bought 4 AAPL", "because", "trade")
    assert res["discord"] is True and res["desktop"] is True
    assert rec.requests[0]["embeds"][0]["title"] == "Bought 4 AAPL"
    assert desk == [("Bought 4 AAPL", "because", "trade")]
    msg = q.get_nowait()
    assert msg["type"] == "toast" and msg["data"]["title"] == "Bought 4 AAPL"


async def test_alert_type_toggle_respected(setup):
    ctx, mgr, rec, desk = setup
    ctx.config.update({"alerts": {"on_manual_review": False}})
    res = await mgr.send("manual_review", "Review AAPL", "", "warn")
    assert res["skipped"] and not rec.requests and not desk


async def test_channel_toggles_respected(setup):
    ctx, mgr, rec, desk = setup
    ctx.config.update({"alerts": {"discord_enabled": False, "desktop_enabled": False}})
    res = await mgr.send("error", "Boom", "", "error")
    assert res["discord"] is False and res["desktop"] is False
    assert not rec.requests and not desk


async def test_duplicates_suppressed(setup):
    _, mgr, rec, _ = setup
    await mgr.send("error", "Same error", "", "error")
    res = await mgr.send("error", "Same error", "", "error")
    assert res["skipped"] and len(rec.requests) == 1


async def test_discord_rate_limit_retried(setup):
    _, mgr, rec, _ = setup
    rec.statuses = [429, 204]
    res = await mgr.send("trade_filled", "Filled", "", "trade")
    assert res["discord"] is True and len(rec.requests) == 2


async def test_bad_webhook_reported(setup):
    ctx, mgr, rec, _ = setup
    rec.statuses = [404]
    res = await mgr.send("kill_switch", "Killed", "", "error")
    assert "invalid" in res["discord"]
    assert any(c["component"] == "alerts" and c["level"] == "warn" for c in ctx.state.components())


async def test_test_alert_ignores_toggles(setup):
    ctx, mgr, rec, desk = setup
    ctx.config.update({"alerts": {"discord_enabled": False, "desktop_enabled": False}})
    res = await mgr.test()
    assert res["discord"] is True and desk


async def _wait_for(cond, timeout: float = 5.0) -> bool:
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while loop.time() < end:
        if cond():
            return True
        await asyncio.sleep(0.02)
    return cond()


async def test_logged_errors_become_alerts(setup):
    ctx, mgr, rec, _ = setup

    def count():
        return sum("Alpaca exploded" in r["embeds"][0]["description"] for r in rec.requests)

    logging.getLogger("newstrader.something").error("Alpaca exploded")
    assert await _wait_for(lambda: count() == 1)
    # throttled: the same error again doesn't re-alert
    logging.getLogger("newstrader.something").error("Alpaca exploded")
    await asyncio.sleep(0.3)
    assert count() == 1


async def test_trader_uses_alert_manager(setup):
    from newstrader.trading.fake_broker import FakeBroker
    from newstrader.trading.trader import Trader

    ctx, mgr, rec, _ = setup
    ctx.services["alerts"] = mgr
    t = Trader(ctx, broker_factory=lambda: FakeBroker())
    await t.connect()
    await t.handle_signal({"id": 1, "ticker": "AAPL", "direction": "bullish", "confidence": 90, "reasoning": "r"})
    assert any(r["embeds"][0]["title"].startswith("Bought") for r in rec.requests)
    await t.kill()
    assert any("KILL SWITCH" in r["embeds"][0]["title"] for r in rec.requests)
    await t.stop()


def test_toast_text_cannot_inject_powershell():
    from newstrader.alerts.desktop import safe_toast_text

    evil = 'Buy $(Start-Process calc) `whoami` "@\n"@ ]]><x>'
    out = safe_toast_text(evil, 250)
    assert "$" not in out and "`" not in out and '"' not in out and "\n" not in out and "]]>" not in out
    assert "Start-Process calc" in out  # text is kept, just made harmless
    assert len(safe_toast_text("x" * 500, 120)) == 120
