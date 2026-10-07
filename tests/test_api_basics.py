from __future__ import annotations

from newstrader import paths


def test_api_requires_token(client):
    r = client.get("/api/status", headers={"X-NT-Token": "wrong"})
    assert r.status_code == 401
    r = client.get("/api/status")
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "paper"
    assert body["kill_switch"]["engaged"] is False


def test_index_served_without_token(client):
    r = client.get("/", headers={"X-NT-Token": ""})
    assert r.status_code == 200
    assert "NewsTrader" in r.text
    assert client.get("/static/js/api.js").status_code == 200


def test_websocket_rejects_bad_token(client):
    import pytest
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect), client.websocket_connect("/ws?token=nope") as ws:
        ws.receive_json()


def test_websocket_hello(client):
    with client.websocket_connect("/ws?token=test-token") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "hello"
        assert msg["data"]["mode"] == "paper"


def test_settings_round_trip_and_validation(client):
    r = client.get("/api/settings")
    assert r.status_code == 200
    assert "sources" not in r.json()["settings"]
    r = client.put("/api/settings", json={"risk": {"max_open_positions": 7}})
    assert r.status_code == 200
    assert r.json()["settings"]["risk"]["max_open_positions"] == 7
    r = client.put("/api/settings", json={"trading": {"review_threshold": 95}})
    assert r.status_code == 422
    assert any("review threshold" in d for d in r.json()["details"])
    r = client.post("/api/settings/reset")
    assert r.json()["settings"]["risk"]["max_open_positions"] == 5


def test_keys_written_to_env_and_masked(client):
    r = client.put("/api/keys", json={"ANTHROPIC_API_KEY": "sk-ant-abcdefgh12345678"})
    assert r.status_code == 200
    item = next(k for k in r.json()["keys"] if k["name"] == "ANTHROPIC_API_KEY")
    assert item["set"] is True
    assert item["masked"].endswith("5678")
    assert "abcdefgh" not in r.text
    assert "ANTHROPIC_API_KEY=sk-ant-abcdefgh12345678" in paths.env_file().read_text()
    # clearing
    r = client.put("/api/keys", json={"ANTHROPIC_API_KEY": ""})
    item = next(k for k in r.json()["keys"] if k["name"] == "ANTHROPIC_API_KEY")
    assert item["set"] is False
    # unknown names rejected
    assert client.put("/api/keys", json={"EVIL": "x"}).status_code == 400
    # newline injection rejected
    assert client.put("/api/keys", json={"DISCORD_WEBHOOK_URL": "https://x\nALPACA_LIVE_API_KEY=1"}).status_code == 400


def test_sources_crud(client):
    r = client.post("/api/sources", json={"type": "rss", "name": "My Feed", "url": "https://example.com/rss"})
    assert r.status_code == 200
    sid = r.json()["id"]
    assert sid == "my-feed"
    r = client.post("/api/sources", json={"type": "rss", "name": "My Feed", "url": "https://example.com/rss2"})
    assert r.json()["id"] == "my-feed-2"
    r = client.post(f"/api/sources/{sid}/toggle", json={"enabled": False})
    assert r.json()["enabled"] is False
    r = client.put(f"/api/sources/{sid}", json={"poll_seconds": 120})
    assert r.json()["poll_seconds"] == 120
    assert client.delete(f"/api/sources/{sid}").status_code == 200
    ids = [s["id"] for s in client.get("/api/sources").json()["sources"]]
    assert sid not in ids
    assert client.post("/api/sources", json={"type": "rss", "name": "No url"}).status_code == 422


def test_logs_endpoint(client, ctx):
    ctx.db.insert("logs", {"ts": "2026-01-01T00:00:00Z", "level": "ERROR", "logger": "newstrader.test", "message": "boom"})
    r = client.get("/api/logs?level=ERROR")
    assert any(row["message"] == "boom" for row in r.json()["logs"])
