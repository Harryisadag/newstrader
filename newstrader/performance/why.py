"""The "Why did it trade?" card: one signal - or the trade it made - explained in one place.

- the news: the headline (or what was said on TV), where it came from and when
- which engine decided and why: the news event the rules found, the wording scores, the price model's probability,
  Pro AI's view and what the chart said
- the decision trail saved when it was decided (trading/trail.py): the thresholds and style in use, the confidence
  before and after each adjustment, and every check it passed or the one that sent it to manual review or blocked it.
  Signals from before the trail was saved show what is known about them.
- the speed timer, and the trade: entry, stop-loss, take-profit and exit
- a price chart from 30 minutes before the news to 2 hours after, with the news, entry, stop, target and exit marked
- the result compared with the S&P 500 (SPY) after 5 minutes, 1 hour and 1 trading day (the performance tracker's
  checkpoints), and over the trade itself

Prices come from the trading account in use (Alpaca or the simulator's free prices). Without them the card still
shows everything else and says why the chart is missing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any

from ..context import AppContext
from ..db import iso, parse_iso, utcnow
from ..trading import trail as trails
from ..trading.pnl import realized_pnl
from .prices import price_at
from .speed import signal_speed
from .stats import ENGINE_NAMES

log = logging.getLogger(__name__)

BEFORE = timedelta(minutes=30)
AFTER = timedelta(hours=2)
MARKET = "SPY"
SETTLED = timedelta(minutes=3)  # a price window that ended this long ago won't change: its bars are kept
CACHE_SIZE = 64
NO_TRAIL = ("This signal is from before NewsTrader saved each decision step, so only what was stored with it is shown.")
ENTRY_INTENTS = ("open_long", "open_short")
EXIT_INTENTS = ("take_profit", "stop_loss", "close_long", "close_short")

_bars_cache: dict[tuple, tuple[float, dict[str, list[dict]], Any]] = {}


class NotFound(LookupError):
    pass


# ---------------------------------------------------------------------------------------------- the card
async def build_card(ctx: AppContext, signal_id: int | None = None, order_id: int | None = None,
                     now: datetime | None = None) -> dict[str, Any]:
    """The card for a signal (signal_id) or a Trade Log row (order_id: the orders table's id)."""
    db = ctx.db
    now = now or utcnow()
    clicked = None
    if order_id is not None:
        clicked = db.query_one("SELECT * FROM orders WHERE id = ?", (order_id,))
        if clicked is None:
            raise NotFound("Unknown order")
        signal_id = _signal_of(db, clicked)
    sig = db.query_one("SELECT * FROM signals WHERE id = ?", (signal_id,)) if signal_id else None
    if sig is None and clicked is None:
        raise NotFound("Unknown signal")

    entry = _entry_order(db, sig, clicked)
    exit_ = _exit_order(db, entry, clicked)
    trade = _trade(db, entry, exit_)
    analysis = db.query_one("SELECT * FROM analyses WHERE id = ?", (sig["analysis_id"],)) if sig and sig.get(
        "analysis_id") else None
    item = None
    if analysis and analysis.get("item_id"):
        item = db.query_one("SELECT id, title, body, url, source_name, source_type, speaker, published_at, received_at, "
                            "language FROM news_items WHERE id = ?", (analysis["item_id"],))
    trail = trails.load(sig.get("decision_trail")) if sig else None
    card: dict[str, Any] = {
        "signal": _signal_public(sig),
        "order": _order_public(clicked),
        "news": _news(sig, item, analysis),
        "engine": _engine(db, sig, analysis, trail),
        "trail": trail,
        "trail_note": "" if trail or sig is None else NO_TRAIL,
        "speed": signal_speed(sig, entry["submitted_at"] if entry else None) if sig else None,
        "trade": trade,
    }
    if sig is None:
        card["note"] = ("This order wasn't placed for a signal (you placed or closed it by hand, or it was placed "
                        "outside NewsTrader), so there's no AI decision to show.")
    elif sig.get("merged_into"):
        card["note"] = (f"This was a repeat of the same news, counted as part of signal #{sig['merged_into']} - "
                        "open that one to see what was traded.")
    card["chart"], card["result"] = await _prices(ctx, sig, entry, exit_, trade, now)
    return card


def _signal_of(db, order: dict) -> int | None:
    if order.get("signal_id"):
        return int(order["signal_id"])
    if order.get("parent_alpaca_id"):
        parent = db.query_one("SELECT signal_id FROM orders WHERE alpaca_order_id = ?", (order["parent_alpaca_id"],))
        if parent and parent.get("signal_id"):
            return int(parent["signal_id"])
    return None


def _entry_order(db, sig: dict | None, clicked: dict | None) -> dict | None:
    """The order the signal placed (a buy / short, or a sell of a stock you held)."""
    if clicked is not None:
        if clicked.get("parent_alpaca_id"):  # a stop-loss / take-profit leg: the bracket's entry
            return db.query_one("SELECT * FROM orders WHERE alpaca_order_id = ?", (clicked["parent_alpaca_id"],))
        if clicked.get("intent") in ENTRY_INTENTS or (sig is not None and clicked.get("id") == sig.get("order_id")):
            return clicked
    if sig and sig.get("order_id"):
        return db.query_one("SELECT * FROM orders WHERE id = ?", (sig["order_id"],))
    return None


def _exit_order(db, entry: dict | None, clicked: dict | None) -> dict | None:
    """How the position the entry opened was closed: its filled stop-loss or take-profit, or a later sale."""
    if clicked is not None and clicked.get("intent") in EXIT_INTENTS and clicked.get("status") == "filled" and (
            entry is None or clicked.get("id") != entry.get("id")):
        return clicked
    if entry is None or entry.get("intent") not in ENTRY_INTENTS or not entry.get("filled_at"):
        return None
    leg = db.query_one("SELECT * FROM orders WHERE parent_alpaca_id = ? AND status = 'filled' ORDER BY filled_at "
                       "LIMIT 1", (entry["alpaca_order_id"],))
    if leg is not None:
        return leg
    closing = "close_long" if entry["intent"] == "open_long" else "close_short"  # sold by hand or on bearish news
    return db.query_one("SELECT * FROM orders WHERE symbol = ? AND COALESCE(mode, '') = COALESCE(?, '') AND "
                        "intent = ? AND status = 'filled' AND filled_at > ? ORDER BY filled_at LIMIT 1",
                        (entry["symbol"], entry.get("mode"), closing, entry["filled_at"]))


def _trade(db, entry: dict | None, exit_: dict | None) -> dict | None:
    if entry is None:
        return None
    legs = db.query("SELECT intent, stop_price, take_profit_price, status FROM orders WHERE parent_alpaca_id = ?",
                    (entry["alpaca_order_id"],))
    stop = entry.get("stop_price") or next((leg["stop_price"] for leg in legs if leg["intent"] == "stop_loss"), None)
    target = entry.get("take_profit_price") or next(
        (leg["take_profit_price"] for leg in legs if leg["intent"] == "take_profit"), None)
    out: dict[str, Any] = {
        "symbol": entry["symbol"], "intent": entry.get("intent"), "side": entry.get("side"), "mode": entry.get("mode"),
        "status": entry.get("status"), "qty": entry.get("filled_qty") or entry.get("qty"),
        "submitted_at": entry.get("submitted_at"), "entry_at": entry.get("filled_at"),
        "entry_price": entry.get("filled_avg_price"), "estimated_price": entry.get("entry_price_est"),
        "stop": stop, "target": target, "reason": entry.get("reason"), "exit": None, "realized_pl": None,
        "return_pct": None,
    }
    if exit_ is not None and exit_.get("filled_avg_price"):
        how = {"take_profit": "Take-profit hit", "stop_loss": "Stop-loss hit", "close_long": "Sold",
               "close_short": "Covered"}.get(exit_.get("intent") or "", "Closed")
        out["exit"] = {"at": exit_.get("filled_at"), "price": exit_["filled_avg_price"], "how": how,
                       "reason": exit_.get("reason")}
        fills = db.query("SELECT alpaca_order_id AS id, symbol, side, filled_qty, filled_avg_price, filled_at FROM "
                         "orders WHERE symbol = ? AND COALESCE(mode, '') = COALESCE(?, '') AND filled_qty > 0 AND "
                         "filled_avg_price IS NOT NULL", (entry["symbol"], entry.get("mode")))
        out["realized_pl"] = realized_pnl(fills).get(exit_["alpaca_order_id"])
        if out["entry_price"] and entry.get("intent") in ENTRY_INTENTS:
            sign = 1 if entry["intent"] == "open_long" else -1
            out["return_pct"] = round(sign * (exit_["filled_avg_price"] - out["entry_price"]) / out["entry_price"] * 100,
                                      2)
    return out


# ---------------------------------------------------------------------------------------------- parts
def _signal_public(sig: dict | None) -> dict | None:
    if sig is None:
        return None
    out = {k: v for k, v in sig.items() if k != "decision_trail"}
    for key in ("sources_seen", "flags"):
        try:
            out[key] = json.loads(out.get(key) or "[]")
        except (TypeError, ValueError):
            out[key] = []
    out["engine_label"] = ENGINE_NAMES.get(sig.get("engine") or "", sig.get("engine") or "AI engine")
    return out


def _order_public(order: dict | None) -> dict | None:
    return None if order is None else {k: v for k, v in order.items() if k != "raw"}


def _news(sig: dict | None, item: dict | None, analysis: dict | None) -> dict | None:
    if sig is None:
        return None
    if sig.get("engine") == "chart":
        return {"kind": "chart", "headline": sig.get("headline") or sig.get("reasoning") or "",
                "source": sig.get("source_name"), "at": sig.get("created_at"), "url": None, "quote": "", "speaker": ""}
    tv = (analysis or {}).get("item_kind") == "transcript" or sig.get("source_type") == "stream"
    body = (item or {}).get("body") or ""
    return {
        "kind": "tv" if tv else "news",
        "headline": (item or {}).get("title") or sig.get("headline") or "",
        "quote": ((item or {}).get("title") or sig.get("headline") or "") if tv else "",
        "context": body[-700:] if tv else "",
        "source": sig.get("source_name") or (item or {}).get("source_name"),
        "speaker": sig.get("speaker") or "",
        "url": sig.get("url") or (item or {}).get("url"),
        "published_at": sig.get("news_published_at") or (item or {}).get("published_at"),
        "received_at": sig.get("news_received_at") or (item or {}).get("received_at"),
        "at": iso(_news_time(sig)) if _news_time(sig) else None,
        "also": _json_list(sig.get("sources_seen")),
    }


def _engine(db, sig: dict | None, analysis: dict | None, trail: dict | None) -> dict | None:
    """Which engine decided and why, in plain lines (the trail's copy when there is one, else from the stored
    answer), with Pro AI's view and the chart's."""
    if sig is None:
        return None
    found = (trail or {}).get("engine")
    if not found:
        details = trails.local_details((analysis or {}).get("raw_response")) if (analysis or {}).get(
            "engine") == "local" else {}
        found = trails.engine_part(sig, details.get(sig["ticker"]), (analysis or {}).get("model"))
    name = sig.get("engine") or found.get("name") or ""
    out = {**found, "name": name, "label": ENGINE_NAMES.get(name, name or "AI engine"),
           "model": found.get("model") or (analysis or {}).get("model") or "",
           "latency_ms": (analysis or {}).get("latency_ms"), "cost_usd": (analysis or {}).get("cost_usd"),
           "time_sensitivity": sig.get("time_sensitivity") or "",
           "bull_case": sig.get("bull_case") or "", "bear_case": sig.get("bear_case") or "",
           "lines": _engine_lines(name, found, sig)}
    out["pro_ai"] = _pro_view(db, sig, analysis, (trail or {}).get("pro_ai"))
    out["chart"] = _chart_view(sig, (trail or {}).get("chart"))
    return out


def _engine_lines(name: str, e: dict, sig: dict) -> list[str]:
    lines = []
    if name == "chart":
        return [sig.get("reasoning") or "A pattern on the chart (no news)."]
    if e.get("event"):
        good = sig.get("direction") == "bullish"
        lines.append(f"News event: {e['event']}" + (f" ({e['event_detail']})" if e.get("event_detail") else "")
                     + f". The news rules count this kind of news as {'good' if good else 'bad'} for the stock.")
    elif name == "local":
        lines.append("No concrete news event (earnings, deal, FDA decision, analyst action...) was recognised - the "
                     "call comes from the wording.")
    w = e.get("wording")
    if isinstance(w, dict) and w:
        lines.append(f"Wording: {_pct(w.get('positive'))}% positive / {_pct(w.get('negative'))}% negative / "
                     f"{_pct(w.get('neutral'))}% neutral.")
    if e.get("price_model") is not None:
        p = float(e["price_model"])
        lines.append(f"Price model: {_pct(p)}% chance it beats the S&P 500 over the next hour (tested on news it "
                     "never saw).")
    st = e.get("student")
    if isinstance(st, dict) and st:
        best = max(st, key=lambda k: st[k] or 0)
        lines.append(f"Student model (learned from Pro AI's labels): {_pct(st[best])}% {best}.")
    if e.get("relevance") is not None and float(e["relevance"]) < 0.999:
        lines.append(f"Only part of the text is about this company (counted {_pct(e['relevance'])}%).")
    notes = {"unconfirmed": "Not confirmed yet (a report or rumour) - confidence lowered.",
             "denial": "The company denied a report.", "country": "News about a whole country - its fund is always "
             "reviewed by you.", "person_only": "Only a person linked to the company is mentioned."}
    lines += [notes[f] for f in e.get("flags") or [] if f in notes]
    if name in ("claude", "pro") and e.get("reasoning"):
        lines.append(e["reasoning"])
    return lines


def _pro_view(db, sig: dict, analysis: dict | None, saved: dict | None) -> dict | None:
    """Pro AI's reading of the same story (for a main-engine signal), or the main engine's (for Pro AI's own)."""
    if sig.get("engine") == "pro":
        main = sig.get("main_direction")
        return {"own": True, "main_direction": main,
                "text": "The main engine didn't read this story." if main is None else
                f"The main engine called it {main}." if main != sig.get("direction") else
                "The main engine made the same call."}
    view: dict[str, Any] = {"own": False, "direction": sig.get("pro_direction"), "held": (saved or {}).get("held") or ""}
    if analysis is not None:
        pro = db.query_one("SELECT s.direction, s.confidence, s.reasoning FROM analyses a JOIN signals s ON "
                           "s.analysis_id = a.id WHERE a.main_analysis_id = ? AND a.engine = 'pro' AND s.ticker = ? "
                           "ORDER BY s.id DESC LIMIT 1", (analysis["id"], sig["ticker"]))
        if pro is not None:
            view.update(confidence=pro["confidence"], reasoning=pro["reasoning"], direction=view["direction"] or
                        pro["direction"])
    if not view.get("direction") and not view["held"]:
        return None
    d = view.get("direction")
    view["text"] = (view["held"] or (f"Pro AI agreed: {d}." if d == sig.get("direction")
                                     else f"Pro AI read it as {d}."))
    return view


def _chart_view(sig: dict, saved: dict | None) -> dict | None:
    if saved:
        return saved
    if sig.get("chart_verdict"):
        return {"verdict": sig["chart_verdict"], "reason": sig.get("chart_reason") or "",
                "adjust": sig.get("chart_adjust") or 0, "score": sig.get("chart_score")}
    return None


# ---------------------------------------------------------------------------------------------- prices
async def _prices(ctx: AppContext, sig: dict | None, entry: dict | None, exit_: dict | None, trade: dict | None,
                  now: datetime) -> tuple[dict, dict]:
    """The chart (price from 30 minutes before the news to 2 hours after, with markers) and the result vs the
    S&P 500."""
    symbol = (sig or {}).get("ticker") or (entry or {}).get("symbol")
    t0 = _news_time(sig) if sig else parse_iso((entry or {}).get("submitted_at"))
    result = _checkpoints(ctx.db, sig)
    chart: dict[str, Any] = {"symbol": symbol, "t": [], "close": [], "markers": [], "stop": None, "target": None,
                             "error": ""}
    if trade:
        chart["stop"], chart["target"] = trade.get("stop"), trade.get("target")
    broker = getattr(ctx.service("trader"), "broker", None)
    if not symbol or t0 is None:
        chart["error"] = "No time for the news, so there's no chart."
        return chart, result
    if broker is None:
        chart["error"] = "Not connected to a trading account, so there are no prices for the chart."
        result["note"] = chart["error"]
        return chart, result
    start, end = t0 - BEFORE, min(t0 + AFTER, now)
    if end <= start:
        chart["error"] = "The news is too new for a chart yet."
        return chart, result
    try:
        bars = await _bars(broker, [symbol, MARKET], start, end, now)
    except Exception as exc:
        chart["error"] = f"Couldn't get prices: {exc}"
        result["note"] = chart["error"]
        return chart, result
    rows, spy = bars.get(symbol.upper(), []), bars.get(MARKET, [])
    if not rows:
        chart["error"] = (f"No 1-minute prices for {symbol} at that time from this account's data (quiet stocks can "
                          "go minutes without a trade; the simulator's free prices only go back about 30 days).")
    chart["t"] = [b["t"] for b in rows]
    chart["close"] = [b["c"] for b in rows]
    chart["markers"] = _markers(rows, t0, entry, exit_)
    await _vs_market(broker, result, spy, end, now)
    if trade:
        await _trade_vs_market(broker, result, trade, spy, start, end, now)
    return chart, result


def _markers(rows: list[dict], t0: datetime, entry: dict | None, exit_: dict | None) -> list[dict]:
    out = []
    price = price_at(rows, t0)
    if price is None and rows:
        price = rows[0]["c"]
    out.append({"kind": "news", "label": "News", "t": iso(t0), "price": price})
    if entry is not None and entry.get("filled_at") and entry.get("filled_avg_price"):
        label = {"open_long": "Bought", "open_short": "Shorted", "close_long": "Sold"}.get(entry.get("intent"), "Entry")
        out.append({"kind": "entry", "label": label, "t": entry["filled_at"], "price": entry["filled_avg_price"]})
    if exit_ is not None and exit_.get("filled_at") and exit_.get("filled_avg_price"):
        out.append({"kind": "exit", "label": "Exit", "t": exit_["filled_at"], "price": exit_["filled_avg_price"]})
    return out


def _checkpoints(db, sig: dict | None) -> dict:
    """The performance tracker's checkpoints (+5 min, +1 hour, +1 trading day) for this signal."""
    out: dict[str, Any] = {"checkpoints": [], "trade": None, "note": ""}
    if sig is None:
        return out
    p = db.query_one("SELECT * FROM signal_prices WHERE signal_id = ?", (sig["id"],))
    if p is None:
        out["note"] = ("Not price-checked yet." if sig.get("direction") in ("bullish", "bearish")
                       else "Neutral signals aren't price-checked.")
        return out
    sign = 1 if p.get("direction") == "bullish" else -1
    for key, label in (("5m", "+5 min"), ("1h", "+1 hour"), ("1d", "+1 trading day")):
        ret = p.get(f"ret_{key}")
        out["checkpoints"].append({"key": key, "label": label, "due": p.get(f"due_{key}"),
                                   "right_pct": None if ret is None else round(ret, 2),
                                   "stock_pct": None if ret is None else round(sign * ret, 2),
                                   "spy_pct": None, "vs_spy": None})
    out["t0"], out["price_t0"], out["direction"] = p.get("t0"), p.get("price_t0"), p.get("direction")
    return out


async def _vs_market(broker, result: dict, spy: list[dict], spy_end: datetime, now: datetime) -> None:
    """The S&P 500's move over each checkpoint, and how the call did compared with it (positive = better). `spy` holds
    the chart window's SPY bars (up to `spy_end`); other times are fetched around the moment needed."""
    t0 = parse_iso(result.get("t0"))
    measured = [cp for cp in result["checkpoints"] if cp["stock_pct"] is not None]
    if t0 is None or not measured:
        return
    sign = 1 if result.get("direction") == "bullish" else -1

    async def market_at(t: datetime) -> float | None:
        rows = spy if t <= spy_end else (await _bars(broker, [MARKET], t - timedelta(minutes=20),
                                                     t + timedelta(minutes=1), now)).get(MARKET, [])
        return price_at(rows, t)

    try:
        a = await market_at(t0)
        for cp in measured:
            due = parse_iso(cp["due"])
            b = await market_at(due) if a and due is not None and due <= now else None
            if b:
                cp["spy_pct"] = round((b - a) / a * 100, 2)
                cp["vs_spy"] = round(sign * (cp["stock_pct"] - cp["spy_pct"]), 2)
    except Exception as exc:
        result["note"] = f"Couldn't get the S&P 500's prices: {exc}"


async def _trade_vs_market(broker, result: dict, trade: dict, spy: list[dict], start: datetime, end: datetime,
                           now: datetime) -> None:
    """The trade's own return next to what the S&P 500 did while it was open."""
    ex = trade.get("exit")
    t_in, t_out = parse_iso(trade.get("entry_at")), parse_iso(ex["at"]) if ex else None
    if t_in is None or t_out is None or trade.get("return_pct") is None:
        return
    out = {"return_pct": trade["return_pct"], "spy_pct": None, "vs_spy": None, "realized_pl": trade.get("realized_pl")}
    try:
        rows = list(spy)
        for t in (t_in, t_out):
            if not (start <= t <= end):
                got = await _bars(broker, [MARKET], t - timedelta(minutes=20), t + timedelta(minutes=1), now)
                rows += got.get(MARKET, [])
        rows.sort(key=lambda b: b["t"])
        a, b = price_at(rows, t_in), price_at(rows, t_out)
        if a and b:
            sign = 1 if trade.get("intent") == "open_long" else -1
            out["spy_pct"] = round((b - a) / a * 100, 2)
            out["vs_spy"] = round(out["return_pct"] - sign * out["spy_pct"], 2)
    except Exception as exc:
        log.debug("S&P 500 prices for the trade unavailable: %s", exc)
    result["trade"] = out


async def _bars(broker, symbols: list[str], start: datetime, end: datetime, now: datetime) -> dict[str, list[dict]]:
    """1-minute bars, kept in memory once the window is over (so opening the same card again costs nothing)."""
    key = (id(broker), tuple(symbols), iso(start)[:16], iso(end)[:16])
    hit = _bars_cache.get(key)
    if hit is not None and hit[2] is broker:  # (the entry holds its broker, so the id can't belong to another one)
        return hit[1]
    rows = await asyncio.to_thread(broker.bars_multi, symbols, start, end, "1Min")
    out = {str(k).upper(): sorted(list(v or []), key=lambda b: b["t"]) for k, v in (rows or {}).items()}
    if now - end >= SETTLED:
        if len(_bars_cache) >= CACHE_SIZE:
            _bars_cache.pop(min(_bars_cache, key=lambda k: _bars_cache[k][0]))
        _bars_cache[key] = (time.monotonic(), out, broker)
    return out


# ---------------------------------------------------------------------------------------------- helpers
def _news_time(sig: dict | None) -> datetime | None:
    if not sig:
        return None
    for key in ("news_published_at", "news_received_at", "created_at"):
        t = parse_iso(sig.get(key))
        if t is not None:
            return t
    return None


def _json_list(value: Any) -> list:
    if isinstance(value, list):
        return value
    try:
        out = json.loads(value or "[]")
    except (TypeError, ValueError):
        return []
    return out if isinstance(out, list) else []


def _pct(x: Any) -> int:
    try:
        return round(100 * float(x))
    except (TypeError, ValueError):
        return 0
