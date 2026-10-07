from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def isolated_paths(tmp_path, monkeypatch):
    """Every test gets its own empty data folder and .env file - never touches real ones."""
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("NEWSTRADER_DATA_DIR", str(data))
    monkeypatch.setenv("NEWSTRADER_ENV_FILE", str(tmp_path / ".env"))
    for name in ("ALPACA_PAPER_API_KEY", "ALPACA_PAPER_SECRET_KEY", "ALPACA_LIVE_API_KEY",
                 "ALPACA_LIVE_SECRET_KEY", "ANTHROPIC_API_KEY", "DISCORD_WEBHOOK_URL", "X_BEARER_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    yield tmp_path


@pytest.fixture
def ctx():
    from newstrader.context import build_context

    c = build_context(token="test-token")
    yield c
    c.db.close()


@pytest.fixture
def client(ctx):
    from fastapi.testclient import TestClient

    from newstrader.api.server import create_app

    app = create_app(ctx, start_engine=False)
    with TestClient(app) as tc:
        tc.headers.update({"X-NT-Token": "test-token"})
        yield tc


def pytest_configure(config):
    os.environ.setdefault("NEWSTRADER_TESTING", "1")
