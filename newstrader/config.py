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

from .llm.catalog import MODELS_BY_KEY, SERVER_BUILDS
from .sources.presets import PRESET_URL_FIXES, default_sources, old_poll_seconds

log = logging.getLogger(__name__)

SourceType = Literal["stream", "rss", "social_rss", "x_account", "alpaca_news"]

# Models offered in the settings dropdown.
CLAUDE_MODELS = {
    "claude-sonnet-5-5": "Claude Sonnet 5.5 (default, best balance)",
    "claude-haiku-4-5-20251001": "Claude Haiku 4.5 (cheapest, fastest)",
    "claude-opus-5-5": "Claude Opus 5.5 (smartest, ~2x Sonnet cost)",
}

WHISPER_MODELS = ["large-v3", "large-v3-turbo", "distil-large-v3", "medium", "small", "base", "tiny"]

# Languages Whisper can transcribe (code -> name). Kept here (not imported from faster-whisper) so loading settings
# stays light. "auto" = let Whisper detect the language of each chunk.
WHISPER_LANGUAGES: dict[str, str] = {
    "en": "English", "zh": "Chinese", "de": "German", "es": "Spanish", "ru": "Russian", "ko": "Korean",
    "fr": "French", "ja": "Japanese", "pt": "Portuguese", "tr": "Turkish", "pl": "Polish", "ca": "Catalan",
    "nl": "Dutch", "ar": "Arabic", "sv": "Swedish", "it": "Italian", "id": "Indonesian", "hi": "Hindi",
    "fi": "Finnish", "vi": "Vietnamese", "he": "Hebrew", "uk": "Ukrainian", "el": "Greek", "ms": "Malay",
    "cs": "Czech", "ro": "Romanian", "da": "Danish", "hu": "Hungarian", "ta": "Tamil", "no": "Norwegian",
    "th": "Thai", "ur": "Urdu", "hr": "Croatian", "bg": "Bulgarian", "lt": "Lithuanian", "la": "Latin",
    "mi": "Maori", "ml": "Malayalam", "cy": "Welsh", "sk": "Slovak", "te": "Telugu", "fa": "Persian",
    "lv": "Latvian", "bn": "Bengali", "sr": "Serbian", "az": "Azerbaijani", "sl": "Slovenian", "kn": "Kannada",
    "et": "Estonian", "mk": "Macedonian", "br": "Breton", "eu": "Basque", "is": "Icelandic", "hy": "Armenian",
    "ne": "Nepali", "mn": "Mongolian", "bs": "Bosnian", "kk": "Kazakh", "sq": "Albanian", "sw": "Swahili",
    "gl": "Galician", "mr": "Marathi", "pa": "Punjabi", "si": "Sinhala", "km": "Khmer", "sn": "Shona",
    "yo": "Yoruba", "so": "Somali", "af": "Afrikaans", "oc": "Occitan", "ka": "Georgian", "be": "Belarusian",
    "tg": "Tajik", "sd": "Sindhi", "gu": "Gujarati", "am": "Amharic", "yi": "Yiddish", "lo": "Lao",
    "uz": "Uzbek", "fo": "Faroese", "ht": "Haitian Creole", "ps": "Pashto", "tk": "Turkmen", "nn": "Nynorsk",
    "mt": "Maltese", "sa": "Sanskrit", "lb": "Luxembourgish", "my": "Myanmar", "bo": "Tibetan", "tl": "Tagalog",
    "mg": "Malagasy", "as": "Assamese", "tt": "Tatar", "haw": "Hawaiian", "ln": "Lingala", "ha": "Hausa",
    "ba": "Bashkir", "jw": "Javanese", "su": "Sundanese", "yue": "Cantonese",
}
_LANGUAGE_NAMES = {name.lower(): code for code, name in WHISPER_LANGUAGES.items()}


def normalize_language(value: Any, empty: str) -> str:
    """A Whisper language code, "auto", or `empty` for anything unknown.

    Never raises: a typo in config.json must not reset every other setting (a bad code would also make
    Whisper fail on every audio chunk). English names ("German") are accepted too.
    """
    v = str(value or "").strip().lower()
    if v in WHISPER_LANGUAGES or v == "auto":
        return v
    if v in _LANGUAGE_NAMES:
        return _LANGUAGE_NAMES[v]
    if v:
        log.warning("Unknown language '%s' in settings - using '%s'", value, empty or "the default")
    return empty


def parse_tickers(v: Any) -> list[str]:
    """'aapl, $NVDA msft' / ['AAPL'] -> ['AAPL', 'NVDA', 'MSFT'] (validated, de-duplicated)."""
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
    poll_seconds: int = Field(30, ge=10, le=3600)
    # For social sources: who is posting (passed to Claude as the speaker)
    speaker: str = ""
    # Where the source is from and what it is (presets fill these in; used to group sources in the app)
    region: str = ""      # e.g. "US", "UK", "Europe", "Asia", "India", "Middle East", "Global"
    category: str = ""    # e.g. "TV", "Business news", "Press releases", "Regulators", "Central banks", "Social"
    # Live streams: the language spoken ("" = use Settings -> Transcription -> Language; "auto" = detect)
    language: str = ""
    # Live streams: translate the speech to English while transcribing (Whisper's built-in translation)
    translate: bool = False
    # Live streams that only go live for events (press conferences, interviews, hearings): when one goes live
    # and every slot is busy, it borrows a slot from an always-on channel until the event ends
    live_events: bool = False

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

    @field_validator("region", "category", mode="before")
    @classmethod
    def _label(cls, v: Any) -> str:
        return str(v or "").strip()[:40]

    @field_validator("language", mode="before")
    @classmethod
    def _language(cls, v: Any) -> str:
        return normalize_language(v, "")

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
    analysis_debounce_seconds: int = Field(4, ge=0, le=60)
    # If YouTube says "sign in to confirm you're not a bot": chrome / edge / firefox / safari / brave
    cookies_from_browser: str = ""

    @field_validator("model")
    @classmethod
    def _model(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("whisper model cannot be empty")
        return v

    @field_validator("language", mode="before")
    @classmethod
    def _language(cls, v: Any) -> str:
        return normalize_language(v, "en")


def _catalog_choice(value: Any, known, what: str) -> str:
    """"auto" or a key from newstrader/llm/catalog.py. A key an update removed falls back to "auto" (with a warning)
    instead of resetting every other setting."""
    v = str(value or "").strip().lower() or "auto"
    if v == "auto" or v in known:
        return v
    log.warning("Unknown Pro AI %s '%s' in settings - using 'auto'", what, value)
    return "auto"


class ProAISettings(_Model):
    """Pro AI: an optional local language model (llama.cpp) that reads the stories the main engine finds hard.
    It never places a trade by itself."""

    enabled: bool = False
    # "watch" = only records what it would have done (never traded, never alerted - shown on the scoreboard);
    # "judge" = on hard cases it can send a main-engine signal it disagrees with to manual review, or raise a new
    # manual-review signal the main engine missed
    mode: Literal["watch", "judge"] = "watch"
    model: str = "auto"  # "auto" (the biggest that fits this PC) or a catalog key
    build: str = "auto"  # "auto" (matched to the graphics card) or a catalog key
    # "hard" = TV transcripts, wording-only or neutral local-engine calls, market-wide and non-English news;
    # "all" = every story that names a company
    scope: Literal["hard", "all"] = "hard"
    # a story that waited longer than this for Pro AI is skipped (and Pro AI gets this long to answer)
    max_wait_seconds: int = Field(20, ge=5, le=120)
    # keep graphics memory free for TV transcription when TV sources are on
    keep_tv_memory_free: bool = True

    @field_validator("model", mode="before")
    @classmethod
    def _model(cls, v: Any) -> str:
        return _catalog_choice(v, MODELS_BY_KEY, "model")

    @field_validator("build", mode="before")
    @classmethod
    def _build(cls, v: Any) -> str:
        return _catalog_choice(v, SERVER_BUILDS, "server build")


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
    pro_ai: ProAISettings = Field(default_factory=ProAISettings)

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
    # Recognise concrete news events (earnings beats/misses, guidance, buyouts, FDA decisions, analyst
    # up/downgrades, offerings...) and who they are good or bad for, and discount recaps, opinion pieces and
    # unconfirmed reports. Off = v0.2 behaviour (FinBERT wording score only).
    event_rules: bool = True
    # International macro news (e.g. "Bank of Japan raises rates") can create signals for US-listed country
    # ETFs (EWJ, FXI, EWG...). They are always sent to manual review, never auto-traded.
    country_etfs: bool = True

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
    # Don't chase: if the price already moved this much (%) in the signal's direction since the news came out,
    # send the signal to manual review instead of auto-trading it. 0 = off.
    max_chase_pct: float = Field(3.0, ge=0, le=50)
    # Where paper trades go: "auto" = your Alpaca paper account if its keys are set, otherwise the built-in
    # simulator; "simulator" = always the built-in simulator (fake money, free Yahoo prices, no account needed);
    # "alpaca" = always Alpaca. Real-money (live) trading always uses Alpaca.
    # "alpaca" until the built-in simulator has its screens; "auto" = simulator when there are no Alpaca keys
    broker: Literal["auto", "simulator", "alpaca"] = "alpaca"
    sim_starting_cash: float = Field(100_000.0, ge=1_000, le=10_000_000)
    sim_slippage_pct: float = Field(0.05, ge=0, le=2)  # each simulated fill is this much worse than the last price

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
        return parse_tickers(v)


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
    on_market_spike: bool = True   # a watched stock suddenly jumps or drops
    on_market_move: bool = True    # the whole market (S&P 500, Nasdaq...) makes a big move


class MarketSettings(_Model):
    """Market monitor: sudden price/volume spikes, big market-wide moves and top movers."""

    enabled: bool = True
    scan_seconds: int = Field(60, ge=15, le=600)
    # Also scan before 9:30am and after 4pm ET (IEX pre/after-market data is thin, so spikes are noisier)
    extended_hours: bool = False
    # Which stocks to watch: your positions, stocks with recent signals, your watchlist, and today's top movers
    watch_positions: bool = True
    watch_signals_minutes: int = Field(120, ge=0, le=1440)
    watch_movers: bool = True
    movers_top: int = Field(20, ge=5, le=50)
    max_symbols: int = Field(100, ge=10, le=400)
    watchlist: list[str] = Field(default_factory=list)
    # Market-wide gauges (US-listed ETFs: S&P 500, Nasdaq 100, Russell 2000, Dow) and world markets
    index_symbols: list[str] = Field(default_factory=lambda: ["SPY", "QQQ", "IWM", "DIA"])
    world_symbols: list[str] = Field(default_factory=lambda: ["EWJ", "FXI", "EWG", "EWU", "INDA", "EWZ", "EZU", "EWY"])
    # A spike = the price moves at least spike_pct within spike_window_minutes, on unusual volume
    spike_window_minutes: Literal[1, 5, 15] = 5
    spike_pct: float = Field(3.0, ge=0.5, le=50)
    volume_ratio: float = Field(3.0, ge=1.0, le=100)
    min_price: float = Field(2.0, ge=0)
    # Market-wide alert: an index ETF moves this much within 15 minutes, or its day change crosses each step
    market_move_pct: float = Field(1.0, ge=0.2, le=10)
    market_day_step_pct: float = Field(1.0, ge=0.5, le=10)
    # Look up the news behind a spike (Alpaca/Benzinga news for that stock)
    lookup_news: bool = True
    alert_cooldown_minutes: int = Field(30, ge=1, le=1440)
    max_alerts_per_scan: int = Field(3, ge=1, le=20)

    @field_validator("watchlist", "index_symbols", "world_symbols", mode="before")
    @classmethod
    def _tickers(cls, v: Any) -> list[str]:
        return parse_tickers(v)

    @field_validator("spike_window_minutes", mode="before")
    @classmethod
    def _window(cls, v: Any) -> Any:
        return int(v) if isinstance(v, str) and v.strip().isdigit() else v


class ChartSettings(_Model):
    """Charts: indicators and patterns (newstrader/chart). They never place a trade by themselves."""

    # Check each news signal against its chart before it is traded. "soft" = a chart that agrees nudges the
    # confidence up a little (never from manual review into an auto-trade), one that disagrees lowers it, and a
    # move that already happened ("stretched") goes to manual review; "strict" = a chart that disagrees also sends
    # it to manual review; "off" = no chart check. Selling a stock you hold is never held back by the chart.
    confirm: Literal["off", "soft", "strict"] = "soft"
    # Signals from the chart alone (breakouts, flags, double bottoms... on heavy volume) for the stocks the market
    # monitor watches. "watch" = recorded and price-checked for the scoreboard, never traded or alerted;
    # "review" = sent to manual review for you to approve; "off" = none.
    signals: Literal["off", "watch", "review"] = "watch"


class UISettings(_Model):
    # Times in the app: "local" = this computer's time zone, "market" = New York (ET), "utc"
    time_zone: Literal["local", "market", "utc"] = "local"
    # Check GitHub once a day for a newer NewsTrader and show a banner (nothing is installed by itself)
    check_updates: bool = True


class AppSettings(_Model):
    schema_version: int = 4
    transcription: TranscriptionSettings = Field(default_factory=TranscriptionSettings)
    ai: AISettings = Field(default_factory=AISettings)
    ml: MLSettings = Field(default_factory=MLSettings)
    trading: TradingSettings = Field(default_factory=TradingSettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    alerts: AlertSettings = Field(default_factory=AlertSettings)
    market: MarketSettings = Field(default_factory=MarketSettings)
    chart: ChartSettings = Field(default_factory=ChartSettings)
    ui: UISettings = Field(default_factory=UISettings)
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


def _backfill_presets(raw: dict, settings: AppSettings) -> AppSettings:
    """Configs saved before v0.3 (schema 1): give built-in sources the new preset fields (region, category,
    language, translate) they were saved without. Fields already in the user's file are never changed."""
    presets = {p["id"]: p for p in default_sources()}
    raw_sources = {s.get("id"): s for s in raw.get("sources", []) if isinstance(s, dict)}
    out = []
    for src in settings.sources:
        data = src.model_dump(mode="json")
        preset = presets.get(src.id)
        if src.builtin and preset is not None:
            saved = raw_sources.get(src.id, {})
            for key in ("region", "category", "language", "translate", "live_events"):
                if key not in saved and key in preset:
                    data[key] = preset[key]
        out.append(data)
    return AppSettings.model_validate({**settings.model_dump(mode="json"), "schema_version": 2, "sources": out})


def _speed_up(raw: dict, settings: AppSettings) -> AppSettings:
    """Configs saved before v0.4 (schema 2): feeds are checked more often and TV is analysed sooner. Only values
    still at the old default move to the new one - a check interval or pause the user changed is kept."""
    presets = {p["id"]: p for p in default_sources()}
    raw_sources = {s.get("id"): s for s in raw.get("sources", []) if isinstance(s, dict)}
    out = []
    for src in settings.sources:
        data = src.model_dump(mode="json")
        preset = presets.get(src.id)
        saved = raw_sources.get(src.id)
        if src.builtin and preset is not None and saved is not None and src.type in ("rss", "social_rss"):
            old = old_poll_seconds(preset)
            if saved.get("poll_seconds", old) == old:
                data["poll_seconds"] = preset.get("poll_seconds", SourceConfig.model_fields["poll_seconds"].default)
        out.append(data)
    dump = settings.model_dump(mode="json")
    tr = raw.get("transcription")
    if isinstance(tr, dict) and tr.get("analysis_debounce_seconds") == 10:
        dump["transcription"]["analysis_debounce_seconds"] = TranscriptionSettings.model_fields[
            "analysis_debounce_seconds"].default
    return AppSettings.model_validate({**dump, "schema_version": 3, "sources": out})


def _add_pro_ai(raw: dict, settings: AppSettings) -> AppSettings:
    """Configs saved before v0.4 Pro AI (schema 3): add the ai.pro_ai section, switched off and watch-only, whatever
    an older file says - Pro AI is only ever turned on by setting it up in Settings."""
    dump = settings.model_dump(mode="json")
    dump["ai"]["pro_ai"] = ProAISettings().model_dump(mode="json")
    return AppSettings.model_validate({**dump, "schema_version": 4})


def _fix_preset_urls(settings: AppSettings) -> AppSettings | None:
    """Move built-in sources off a feed that a NewsTrader update replaced (only if the user never edited the URL)."""
    presets = {p["id"]: p for p in default_sources()}
    changed = False
    out = []
    for src in settings.sources:
        data = src.model_dump(mode="json")
        old = PRESET_URL_FIXES.get(src.id)
        preset = presets.get(src.id)
        if src.builtin and old and preset and src.url == old and preset["url"] != old:
            data.update({k: preset[k] for k in ("url", "name", "poll_seconds") if k in preset})
            changed = True
        out.append(data)
    if not changed:
        return None
    return AppSettings.model_validate({**settings.model_dump(mode="json"), "sources": out})


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
        self.needs_save = False
        self._lock = threading.RLock()
        self._listeners: list = []
        self._settings = self._load()
        if self.engine_was_defaulted or self.needs_save:
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

    def set_sources_enabled(self, source_ids: list[str], enabled: bool) -> int:
        """Turn several sources on or off in one save. Returns how many changed."""
        with self._lock:
            wanted = set(source_ids)
            sources = [s.model_dump(mode="json") for s in self._settings.sources]
            changed = 0
            for s in sources:
                if s["id"] in wanted and s["enabled"] != enabled:
                    s["enabled"] = enabled
                    changed += 1
            if changed:
                self.update({"sources": sources})
            return changed

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
            version = int(raw.get("schema_version") or 1)
            if version < 2:
                settings = _backfill_presets(raw, settings)
                self.needs_save = True
            fixed = _fix_preset_urls(settings)
            if fixed is not None:
                settings, self.needs_save = fixed, True
            if version < 3:
                settings = _speed_up(raw, settings)
                self.needs_save = True
            if version < 4:
                settings = _add_pro_ai(raw, settings)
                self.needs_save = True
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
