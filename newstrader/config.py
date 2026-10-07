"""App settings, saved to config.json in the data folder.

Every setting has a safe default. The file is validated on load; if it is corrupted it is backed up
and defaults are used, so a bad edit can never stop the app from starting.

Note: live trading is deliberately NOT a saved setting. See trading/live_guard.py.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .sources.presets import default_sources

log = logging.getLogger(__name__)

SourceType = Literal["stream", "rss", "social_rss", "x_account", "alpaca_news"]

# Models offered in the settings dropdown.
CLAUDE_MODELS = {
    "claude-sonnet-5-5": "Claude Sonnet 5.5 (default, best balance)",
    "claude-haiku-4-5-20251001": "Claude Haiku 4.5 (cheapest, fastest)",
    "claude-opus-5-5": "Claude Opus 5.5 (smartest, ~2x Sonnet cost)",
}

WHISPER_MODELS = ["large-v3", "large-v3-turbo", "distil-large-v3", "medium", "small", "base", "tiny"]

_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore", validate_assignment=True)


class SourceConfig(_Model):
    id: str
    type: SourceType
    name: str
    url: str = ""
    enabled: bool = True
    builtin: bool = False
    # RSS / social polling interval
    poll_seconds: int = Field(60, ge=10, le=3600)
    # For social sources: who is posting (passed to Claude as the speaker)
    speaker: str = ""

    @field_validator("id")
    @classmethod
    def _slug(cls, v: str) -> str:
        v = re.sub(r"[^a-z0-9\-]+", "-", v.strip().lower()).strip("-")
        if not v:
            raise ValueError("source id cannot be empty")
        return v[:60]

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("source name cannot be empty")
        return v[:80]

    @model_validator(mode="after")
    def _needs_url(self) -> SourceConfig:
        if self.type != "alpaca_news" and not self.url.strip():
            raise ValueError(f"source '{self.name}' needs a URL (or username for X)")
        return self


def _on_mac() -> bool:
    return sys.platform == "darwin"


class TranscriptionSettings(_Model):
    enabled: bool = True
    # Mac: large-v3-turbo is ~4x faster than large-v3 with nearly the same accuracy (Apple GPU or CPU)
    model: str = Field(default_factory=lambda: "large-v3-turbo" if _on_mac() else "large-v3")
    # cuda = NVIDIA GPU, mlx = Apple Silicon GPU, cpu, auto = best available
    device: Literal["cuda", "mlx", "cpu", "auto"] = Field(default_factory=lambda: "auto" if _on_mac() else "cuda")
    # float16 is right for RTX 50-series; INT8 is disabled on Blackwell GPUs in CTranslate2.
    # (The Apple GPU ignores this; Mac CPUs use int8.)
    compute_type: Literal["float16", "int8_float16", "int8", "float32"] = Field(
        default_factory=lambda: "int8" if _on_mac() else "float16")
    language: str = "en"  # "auto" = detect
    beam_size: int = Field(5, ge=1, le=10)
    chunk_seconds: int = Field(10, ge=3, le=30)
    vad_min_silence_ms: int = Field(500, ge=100, le=3000)
    max_concurrent_streams: int = Field(4, ge=1, le=12)
    # How much recent transcript Claude sees when a stream mentions a company/keyword
    analysis_window_seconds: int = Field(60, ge=15, le=300)
    # Wait this long after a hit so the rest of the sentence gets included
    analysis_debounce_seconds: int = Field(10, ge=0, le=60)
    # If YouTube says "sign in to confirm you're not a bot": chrome / edge / firefox / safari / brave
    cookies_from_browser: str = ""

    @field_validator("model")
    @classmethod
    def _model(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("whisper model cannot be empty")
        return v


class AISettings(_Model):
    # "local" = free on-device machine learning (FinBERT + price-trained model); "claude" = Anthropic API
    engine: Literal["local", "claude"] = "local"
    model: str = "claude-sonnet-5-5"
    effort: Literal["low", "medium", "high"] = "low"
    use_refusal_fallback: bool = True
    daily_spend_cap_usd: float = Field(5.0, ge=0, le=1000)
    max_calls_per_minute: int = Field(20, ge=1, le=300)
    max_concurrent_calls: int = Field(3, ge=1, le=10)
    # Same story from several sources within this window is analysed once
    story_dedupe_minutes: int = Field(15, ge=0, le=240)
    story_similarity: float = Field(0.6, ge=0.3, le=1.0)
    # Same ticker + direction within this window counts as one signal
    signal_dedupe_minutes: int = Field(15, ge=0, le=240)
    max_signals_per_item: int = Field(3, ge=1, le=5)
    # Also analyse text that only has market-moving keywords (e.g. "Fed cuts rates") with no company named.
    # Claude engine only - the local engine needs a named company to score.
    analyze_keyword_only: bool = True

    @field_validator("model")
    @classmethod
    def _model(cls, v: str) -> str:
        v = v.strip()
        if not v.startswith("claude-"):
            raise ValueError("model must be a Claude model id, e.g. claude-sonnet-5-5")
        return v


class MLSettings(_Model):
    """The local machine-learning engine."""

    # "finbert" = FinBERT language model (~110 MB download, best); "lexicon" = small built-in word list
    sentiment_model: Literal["finbert", "lexicon"] = "finbert"
    # Use the model trained on price history (Backtest tab -> Local ML model) when it has passed its test:
    # "auto" = only if it passed, "always" = even if it didn't, "never" = sentiment only
    use_trained_model: Literal["auto", "always", "never"] = "auto"
    # May signals scored by sentiment alone (no price model in use) auto-buy? FinBERT judges wording, not
    # whether prices will move. "auto" = yes, unless a trained price model failed its test (that's evidence
    # wording didn't predict moves in your history) - then they go to manual review instead.
    # "always" = yes; "review" = never auto-buy, only manual-review alerts.
    sentiment_only_trading: Literal["auto", "always", "review"] = "auto"
    # Training: how much history to learn from, how far ahead to measure the move, and the smallest move
    # (vs the S&P 500) that counts as a reaction
    train_days: int = Field(180, ge=30, le=730)
    train_horizon_minutes: Literal[30, 60, 120] = 60
    train_min_move_pct: float = Field(0.3, ge=0.0, le=5.0)
    train_max_articles: int = Field(20000, ge=500, le=100000)

    @field_validator("train_horizon_minutes", mode="before")
    @classmethod
    def _horizon(cls, v: Any) -> Any:
        return int(v) if isinstance(v, str) and v.strip().isdigit() else v


class TradingSettings(_Model):
    auto_trade: bool = True
    buy_threshold: int = Field(80, ge=1, le=100)
    review_threshold: int = Field(60, ge=0, le=100)
    sell_on_bearish: bool = True
    allow_shorting: bool = False
    stop_loss_pct: float = Field(2.0, gt=0, le=50)
    take_profit_pct: float = Field(4.0, gt=0, le=200)
    # "iex" is free with every Alpaca account; "sip" needs a paid market-data plan
    data_feed: Literal["iex", "sip"] = "iex"

    @model_validator(mode="after")
    def _thresholds(self) -> TradingSettings:
        if self.review_threshold > self.buy_threshold:
            raise ValueError("review threshold must be at or below the buy threshold")
        return self


class RiskSettings(_Model):
    max_dollars_per_trade: float = Field(1000.0, gt=0)
    max_pct_per_stock: float = Field(10.0, gt=0, le=100)
    max_open_positions: int = Field(5, ge=1, le=200)
    daily_loss_limit_usd: float = Field(500.0, gt=0)
    ticker_cooldown_minutes: int = Field(30, ge=0, le=10080)
    market_hours_only: bool = True
    min_share_price: float = Field(1.0, ge=0)
    blacklist: list[str] = Field(default_factory=list)
    whitelist: list[str] = Field(default_factory=list)  # empty = every stock allowed

    @field_validator("blacklist", "whitelist", mode="before")
    @classmethod
    def _tickers(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            v = re.split(r"[\s,;]+", v)
        out: list[str] = []
        for item in v or []:
            t = str(item).strip().upper().lstrip("$")
            if not t:
                continue
            if not _TICKER_RE.match(t):
                raise ValueError(f"'{item}' doesn't look like a ticker")
            if t not in out:
                out.append(t)
        return out


class AlertSettings(_Model):
    desktop_enabled: bool = True
    discord_enabled: bool = True
    on_trade_placed: bool = True
    on_trade_filled: bool = True
    on_manual_review: bool = True
    on_error: bool = True
    on_daily_loss_limit: bool = True
    on_spend_cap: bool = True
    on_kill_switch: bool = True


class AppSettings(_Model):
    schema_version: int = 1
    transcription: TranscriptionSettings = Field(default_factory=TranscriptionSettings)
    ai: AISettings = Field(default_factory=AISettings)
    ml: MLSettings = Field(default_factory=MLSettings)
    trading: TradingSettings = Field(default_factory=TradingSettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    alerts: AlertSettings = Field(default_factory=AlertSettings)
    sources: list[SourceConfig] = Field(default_factory=lambda: [SourceConfig(**s) for s in default_sources()])

    @field_validator("sources")
    @classmethod
    def _unique_ids(cls, v: list[SourceConfig]) -> list[SourceConfig]:
        seen: set[str] = set()
        for src in v:
            if src.id in seen:
                raise ValueError(f"duplicate source id '{src.id}'")
            seen.add(src.id)
        return v


def _deep_merge(base: dict, patch: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


class ConfigStore:
    """Thread-safe holder for the current settings, backed by config.json."""

    def __init__(self, path: Path):
        self.path = path
        self.engine_was_defaulted = False
        self._lock = threading.RLock()
        self._listeners: list = []
        self._settings = self._load()
        if self.engine_was_defaulted:
            self._save()  # write the new ai.engine now, so the "engine changed" notice is shown only once

    # ---- reading ----
    @property
    def settings(self) -> AppSettings:
        with self._lock:
            return self._settings

    def as_dict(self) -> dict:
        with self._lock:
            return self._settings.model_dump(mode="json")

    def source(self, source_id: str) -> SourceConfig | None:
        with self._lock:
            return next((s for s in self._settings.sources if s.id == source_id), None)

    # ---- writing ----
    def update(self, patch: dict) -> AppSettings:
        """Merge a partial settings dict, validate, save. Raises pydantic.ValidationError if invalid."""
        with self._lock:
            merged = _deep_merge(self.as_dict(), patch)
            new = AppSettings.model_validate(merged)
            self._settings = new
            self._save()
        self._notify()
        return new

    def replace(self, settings: AppSettings) -> None:
        with self._lock:
            self._settings = AppSettings.model_validate(settings.model_dump(mode="json"))
            self._save()
        self._notify()

    def reset(self, keep_sources: bool = True) -> AppSettings:
        with self._lock:
            sources = self._settings.sources if keep_sources else None
            new = AppSettings()
            if sources is not None:
                new = AppSettings.model_validate({**new.model_dump(mode="json"),
                                                  "sources": [s.model_dump(mode="json") for s in sources]})
            self._settings = new
            self._save()
        self._notify()
        return new

    def upsert_source(self, source: dict) -> SourceConfig:
        with self._lock:
            src = SourceConfig.model_validate(source)
            sources = [s.model_dump(mode="json") for s in self._settings.sources]
            idx = next((i for i, s in enumerate(sources) if s["id"] == src.id), None)
            if idx is None:
                sources.append(src.model_dump(mode="json"))
            else:
                builtin = sources[idx].get("builtin", False)
                sources[idx] = {**src.model_dump(mode="json"), "builtin": builtin}
            self.update({"sources": sources})
            return self.source(src.id)  # type: ignore[return-value]

    def set_source_enabled(self, source_id: str, enabled: bool) -> SourceConfig:
        with self._lock:
            src = self.source(source_id)
            if src is None:
                raise KeyError(source_id)
            data = src.model_dump(mode="json")
            data["enabled"] = enabled
            return self.upsert_source(data)

    def delete_source(self, source_id: str) -> None:
        with self._lock:
            sources = [s.model_dump(mode="json") for s in self._settings.sources if s.id != source_id]
            if len(sources) == len(self._settings.sources):
                raise KeyError(source_id)
            self.update({"sources": sources})

    def on_change(self, callback) -> None:
        """callback(settings) is called (from the writer's thread) after every successful save."""
        self._listeners.append(callback)

    # ---- internals ----
    def _notify(self) -> None:
        for cb in list(self._listeners):
            try:
                cb(self._settings)
            except Exception:  # a broken listener must never block saving settings
                log.exception("settings listener failed")

    def _load(self) -> AppSettings:
        if not self.path.exists():
            settings = AppSettings()
            self._settings = settings
            self._save()
            return settings
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            settings = AppSettings.model_validate(raw)
            # Add any new preset sources shipped in an update (by id), without touching user edits.
            known = {s.id for s in settings.sources}
            removed = set(raw.get("_removed_presets", []))
            added = [SourceConfig(**s) for s in default_sources() if s["id"] not in known and s["id"] not in removed]
            if added:
                settings = AppSettings.model_validate(
                    {**settings.model_dump(mode="json"),
                     "sources": [s.model_dump(mode="json") for s in settings.sources + added]})
            self._removed_presets = sorted(removed)
            # Configs from before the local ML engine existed have no ai.engine: they now use the local engine.
            self.engine_was_defaulted = isinstance(raw.get("ai"), dict) and "engine" not in raw["ai"]
            return settings
        except Exception as exc:
            backup = self.path.with_name(f"config.broken-{int(time.time())}.json")
            try:
                os.replace(self.path, backup)
            except OSError:
                pass
            log.error("config.json was invalid (%s). Backed it up to %s and started with defaults.", exc, backup.name)
            settings = AppSettings()
            self._settings = settings
            self._save()
            return settings

    def _save(self) -> None:
        data = self._settings.model_dump(mode="json")
        # Remember which built-in presets the user deleted so they don't come back on the next start.
        current_ids = {s["id"] for s in data["sources"]}
        removed = set(getattr(self, "_removed_presets", []))
        removed |= {p["id"] for p in default_sources() if p["id"] not in current_ids}
        removed -= current_ids
        self._removed_presets = sorted(removed)
        data["_removed_presets"] = self._removed_presets
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)
