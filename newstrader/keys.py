"""API keys stored in the .env file.

The Settings -> API Keys page reads and writes this file, so you never have to edit it by hand.
Keys are only ever sent back to the UI masked (last 4 characters).
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values, set_key, unset_key

KEY_FIELDS: dict[str, str] = {
    "ALPACA_PAPER_API_KEY": "Alpaca paper API key",
    "ALPACA_PAPER_SECRET_KEY": "Alpaca paper secret key",
    "ALPACA_LIVE_API_KEY": "Alpaca LIVE API key (real money - leave empty)",
    "ALPACA_LIVE_SECRET_KEY": "Alpaca LIVE secret key (real money - leave empty)",
    "ANTHROPIC_API_KEY": "Anthropic (Claude) API key",
    "DISCORD_WEBHOOK_URL": "Discord webhook URL",
    "X_BEARER_TOKEN": "X (Twitter) API bearer token (optional, paid)",
}


def mask(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "•" * len(value)
    return "•" * 8 + value[-4:]


@dataclass(frozen=True)
class Keys:
    alpaca_paper_key: str = ""
    alpaca_paper_secret: str = ""
    alpaca_live_key: str = ""
    alpaca_live_secret: str = ""
    anthropic: str = ""
    discord_webhook: str = ""
    x_bearer: str = ""

    @property
    def has_alpaca_paper(self) -> bool:
        return bool(self.alpaca_paper_key and self.alpaca_paper_secret)

    @property
    def has_alpaca_live(self) -> bool:
        return bool(self.alpaca_live_key and self.alpaca_live_secret)

    @property
    def has_anthropic(self) -> bool:
        return bool(self.anthropic)

    @property
    def has_discord(self) -> bool:
        return self.discord_webhook.startswith("https://")


class KeyStore:
    def __init__(self, env_path: Path):
        self.env_path = env_path
        self._lock = threading.Lock()
        self._keys = self._read()

    @property
    def keys(self) -> Keys:
        return self._keys

    def reload(self) -> Keys:
        with self._lock:
            self._keys = self._read()
            return self._keys

    def _raw(self) -> dict[str, str]:
        values = dotenv_values(self.env_path) if self.env_path.exists() else {}
        out = {}
        for name in KEY_FIELDS:
            # .env wins; fall back to a real environment variable (e.g. a system-wide ANTHROPIC_API_KEY)
            out[name] = (values.get(name) or os.environ.get(name) or "").strip()
        return out

    def _read(self) -> Keys:
        raw = self._raw()
        return Keys(
            alpaca_paper_key=raw["ALPACA_PAPER_API_KEY"],
            alpaca_paper_secret=raw["ALPACA_PAPER_SECRET_KEY"],
            alpaca_live_key=raw["ALPACA_LIVE_API_KEY"],
            alpaca_live_secret=raw["ALPACA_LIVE_SECRET_KEY"],
            anthropic=raw["ANTHROPIC_API_KEY"],
            discord_webhook=raw["DISCORD_WEBHOOK_URL"],
            x_bearer=raw["X_BEARER_TOKEN"],
        )

    def masked(self) -> list[dict]:
        raw = self._raw()
        return [{"name": n, "label": label, "set": bool(raw[n]), "masked": mask(raw[n])}
                for n, label in KEY_FIELDS.items()]

    def update(self, values: dict[str, str | None]) -> Keys:
        """Write keys to .env. A value of None or "" clears the key. Unknown names are rejected."""
        unknown = set(values) - set(KEY_FIELDS)
        if unknown:
            raise ValueError(f"unknown key name(s): {', '.join(sorted(unknown))}")
        with self._lock:
            self.env_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.env_path.exists():
                self.env_path.write_text("# NewsTrader API keys - keep this file private\n", encoding="utf-8")
            for name, value in values.items():
                value = (value or "").strip()
                if any(ch in value for ch in "\r\n"):
                    raise ValueError(f"{name} must be a single line")
                if value:
                    set_key(str(self.env_path), name, value, quote_mode="never")
                else:
                    unset_key(str(self.env_path), name, quote_mode="never")
                    os.environ.pop(name, None)
            self._keys = self._read()
            return self._keys
