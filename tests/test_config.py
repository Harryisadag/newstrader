from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from newstrader import paths
from newstrader.config import AppSettings, ConfigStore


def test_defaults_match_spec(monkeypatch):
    monkeypatch.setattr("sys.platform", "win32")  # Windows defaults (Mac ones are checked in test_mac.py)
    s = AppSettings()
    assert s.trading.buy_threshold == 80
    assert s.trading.review_threshold == 60
    assert s.trading.allow_shorting is False
    assert s.risk.market_hours_only is True
    assert s.transcription.max_concurrent_streams == 4
    assert s.transcription.model == "large-v3"
    assert s.ai.model == "claude-sonnet-5-5"
    assert s.ai.engine == "local"
    ids = {src.id for src in s.sources}
    assert {"bloomberg-tv", "alpaca-news", "cnbc-top", "truth-social-trump"} <= ids


def test_config_file_created_and_round_trips():
    store = ConfigStore(paths.config_file())
    assert paths.config_file().exists()
    store.update({"risk": {"max_dollars_per_trade": 250}})
    again = ConfigStore(paths.config_file())
    assert again.settings.risk.max_dollars_per_trade == 250
    # untouched values keep defaults
    assert again.settings.risk.max_open_positions == 5


def test_invalid_update_is_rejected_and_not_saved():
    store = ConfigStore(paths.config_file())
    with pytest.raises(ValidationError):
        store.update({"trading": {"review_threshold": 90, "buy_threshold": 80}})
    assert store.settings.trading.review_threshold == 60
    assert ConfigStore(paths.config_file()).settings.trading.review_threshold == 60


@pytest.mark.parametrize("patch", [
    {"risk": {"max_dollars_per_trade": -5}},
    {"risk": {"max_pct_per_stock": 150}},
    {"trading": {"buy_threshold": 101}},
    {"ai": {"model": "gpt-4"}},
    {"risk": {"blacklist": "AAPL, not a ticker!"}},
])
def test_bad_values_rejected(patch):
    store = ConfigStore(paths.config_file())
    with pytest.raises(ValidationError):
        store.update(patch)


def test_ticker_lists_are_normalised():
    store = ConfigStore(paths.config_file())
    store.update({"risk": {"blacklist": "gme, $amc ,GME", "whitelist": []}})
    assert store.settings.risk.blacklist == ["GME", "AMC"]


def test_corrupt_config_falls_back_to_defaults():
    paths.config_file().write_text("{not json", encoding="utf-8")
    store = ConfigStore(paths.config_file())
    assert store.settings.trading.buy_threshold == 80
    backups = list(paths.data_dir().glob("config.broken-*.json"))
    assert backups, "the broken file should be kept as a backup"


def test_deleted_presets_stay_deleted():
    store = ConfigStore(paths.config_file())
    store.delete_source("cnbc-top")
    again = ConfigStore(paths.config_file())
    assert again.source("cnbc-top") is None
    raw = json.loads(paths.config_file().read_text())
    assert "cnbc-top" in raw["_removed_presets"]


def test_source_requires_url_and_unique_ids():
    store = ConfigStore(paths.config_file())
    with pytest.raises(ValidationError):
        store.upsert_source({"id": "x", "type": "rss", "name": "No URL", "url": ""})
    store.upsert_source({"id": "my-feed", "type": "rss", "name": "Mine", "url": "https://e.com/rss"})
    store.set_source_enabled("my-feed", False)
    assert store.source("my-feed").enabled is False


def test_live_trading_is_not_a_saved_setting():
    data = AppSettings().model_dump()
    flat = json.dumps(data).lower()
    assert "live_trading" not in flat and "live_armed" not in flat
