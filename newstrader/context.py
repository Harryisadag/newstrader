"""The AppContext bundles the shared objects every part of the app needs."""

from __future__ import annotations

import asyncio
import secrets
from dataclasses import dataclass, field
from typing import Any

from . import paths
from .config import ConfigStore
from .db import Database
from .events import EventBus
from .keys import KeyStore
from .state import AppState


@dataclass
class AppContext:
    config: ConfigStore
    keys: KeyStore
    db: Database
    bus: EventBus
    state: AppState
    token: str
    services: dict[str, Any] = field(default_factory=dict)
    loop: asyncio.AbstractEventLoop | None = None

    def service(self, name: str) -> Any:
        return self.services.get(name)


def build_context(token: str | None = None) -> AppContext:
    db = Database(paths.db_file())
    bus = EventBus()
    return AppContext(
        config=ConfigStore(paths.config_file()),
        keys=KeyStore(paths.env_file()),
        db=db,
        bus=bus,
        state=AppState(db, bus),
        token=token or secrets.token_urlsafe(24),
    )
