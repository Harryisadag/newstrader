from __future__ import annotations

from . import (
    alerts,
    backtest,
    keys,
    logs,
    ml,
    performance,
    settings,
    signals,
    sources,
    status,
    streams,
    system,
    trading,
)

ALL_ROUTERS = [
    status.router,
    settings.router,
    keys.router,
    sources.router,
    logs.router,
    system.router,
    trading.router,
    signals.router,
    streams.router,
    alerts.router,
    performance.router,
    backtest.router,
    ml.router,
]
