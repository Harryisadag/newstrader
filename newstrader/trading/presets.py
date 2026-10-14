"""Trading styles (Settings -> Trading -> Style): one click sets how sure a signal must be, the stop-loss and
take-profit, how much each trade and stock may use, the daily limits and how strict the chart check is.

Normal is the app's defaults. The style shown is worked out from the settings themselves: change any of the settings
a style sets and it shows "Custom". Every style keeps short selling off, and none of them touches live trading (that
stays behind its own lock - trading/live_guard.py).
"""

from __future__ import annotations

from typing import Any

CUSTOM = "custom"

# (section, key, label, format) for every setting a style sets, in the order the table shows them
FIELDS: list[tuple[str, str, str, str]] = [
    ("trading", "buy_threshold", "Buy threshold", "conf"),
    ("trading", "review_threshold", "Manual review threshold", "conf"),
    ("trading", "stop_loss_pct", "Stop-loss", "pct"),
    ("trading", "take_profit_pct", "Take-profit", "pct"),
    ("risk", "max_dollars_per_trade", "Max $ per trade", "usd"),
    ("risk", "max_pct_per_stock", "Max % of account in one stock", "pct"),
    ("risk", "max_open_positions", "Max open positions", "int"),
    ("risk", "daily_loss_limit_usd", "Daily loss limit", "usd"),
    ("risk", "ticker_cooldown_minutes", "Cooldown per stock", "min"),
    ("chart", "confirm", "Chart check", "chart"),
    ("trading", "allow_shorting", "Short selling", "onoff"),
]

PRESETS: dict[str, dict[str, Any]] = {
    "cautious": {
        "label": "Cautious",
        "description": "Fewer, smaller trades: only very sure signals are bought, stops are tighter and the daily "
                       "loss limit is lower. A chart that disagrees sends the signal to you instead. A good place "
                       "to start.",
        "values": {"trading": {"buy_threshold": 88, "review_threshold": 70, "stop_loss_pct": 1.5,
                               "take_profit_pct": 3.0, "allow_shorting": False},
                   "risk": {"max_dollars_per_trade": 500.0, "max_pct_per_stock": 5.0, "max_open_positions": 3,
                            "daily_loss_limit_usd": 200.0, "ticker_cooldown_minutes": 60},
                   "chart": {"confirm": "strict"}},
    },
    "normal": {
        "label": "Normal",
        "description": "NewsTrader's standard settings (the defaults).",
        "values": {"trading": {"buy_threshold": 80, "review_threshold": 60, "stop_loss_pct": 2.0,
                               "take_profit_pct": 4.0, "allow_shorting": False},
                   "risk": {"max_dollars_per_trade": 1000.0, "max_pct_per_stock": 10.0, "max_open_positions": 5,
                            "daily_loss_limit_usd": 500.0, "ticker_cooldown_minutes": 30},
                   "chart": {"confirm": "soft"}},
    },
    "aggressive": {
        "label": "Aggressive",
        "description": "More trades, bigger positions and wider stops. That means bigger swings up AND down - not "
                       "more profit. Less sure signals are bought too, and a bad day can cost up to $1,000.",
        "values": {"trading": {"buy_threshold": 72, "review_threshold": 55, "stop_loss_pct": 3.0,
                               "take_profit_pct": 6.0, "allow_shorting": False},
                   "risk": {"max_dollars_per_trade": 2000.0, "max_pct_per_stock": 15.0, "max_open_positions": 8,
                            "daily_loss_limit_usd": 1000.0, "ticker_cooldown_minutes": 15},
                   "chart": {"confirm": "soft"}},
    },
}

_CHART_WORDS = {"off": "Off", "soft": "Soft", "strict": "Strict"}


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b or a == b
    if isinstance(a, int | float) and isinstance(b, int | float):
        return abs(float(a) - float(b)) < 1e-9
    return a == b


def values_of(settings: Any) -> dict[str, dict[str, Any]]:
    """The current values of every setting a style sets ({section: {key: value}})."""
    out: dict[str, dict[str, Any]] = {}
    for section, key, _label, _fmt in FIELDS:
        out.setdefault(section, {})[key] = getattr(getattr(settings, section), key)
    return out


def detect(settings: Any) -> str:
    """The style the settings match exactly ("cautious" / "normal" / "aggressive"), or "custom"."""
    now = values_of(settings)
    for name, preset in PRESETS.items():
        if all(_same(now[section][key], preset["values"][section][key]) for section, key, _l, _f in FIELDS):
            return name
    return CUSTOM


def patch(style: str) -> dict[str, dict[str, Any]]:
    """The settings patch that applies a style (ConfigStore.update). Raises KeyError for an unknown style."""
    preset = PRESETS[style]
    return {section: dict(values) for section, values in preset["values"].items()}


def fmt(kind: str, value: Any) -> str:
    if kind == "conf":
        return str(int(value))
    if kind == "pct":
        return f"{float(value):g}%"
    if kind == "usd":
        return f"${float(value):,.0f}"
    if kind == "min":
        return f"{int(value)} min"
    if kind == "chart":
        return _CHART_WORDS.get(str(value), str(value))
    if kind == "onoff":
        return "On" if value else "Off"
    return str(value)


def describe(settings: Any = None) -> dict[str, Any]:
    """For Settings -> Trading -> Style: each style's label, description and values, the table rows (with the
    current values when `settings` is given) and the style in use."""
    now = values_of(settings) if settings is not None else None
    rows = []
    for section, key, label, kind in FIELDS:
        row = {"key": f"{section}.{key}", "label": label, "kind": kind,
               "values": {name: fmt(kind, p["values"][section][key]) for name, p in PRESETS.items()}}
        if now is not None:
            row["now"] = fmt(kind, now[section][key])
        rows.append(row)
    return {"styles": {name: {"label": p["label"], "description": p["description"], "values": p["values"]}
                       for name, p in PRESETS.items()},
            "rows": rows, "current": detect(settings) if settings is not None else None}
