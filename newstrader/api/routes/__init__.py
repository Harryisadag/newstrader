from __future__ import annotations

from . import keys, logs, settings, sources, status, system

ALL_ROUTERS = [
    status.router,
    settings.router,
    keys.router,
    sources.router,
    logs.router,
    system.router,
]
