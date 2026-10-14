from __future__ import annotations

import asyncio
import csv
import io
from datetime import datetime

from fastapi import APIRouter, Body, Depends, Query
from fastapi.responses import Response

from ... import paths
from ...context import AppContext
from ...performance.why import NotFound, build_card
from ...trading import crypto, crypto_exits, live_guard
from ...trading.trader import NEEDS_ALPACA, SIM_RESET_PHRASE
from ..deps import bad_request, get_ctx

router = APIRouter(tags=["trading"])

CSV_COLUMNS = ["submitted_at", "filled_at", "mode", "symbol", "side", "intent", "qty", "order_type", "status",
               "filled_qty", "filled_avg_price", "stop_price", "take_profit_price", "realized_pl", "confidence",
               "direction", "source_name", "reason", "signal_reasoning", "alpaca_order_id", "signal_id"]


def _trader(ctx: AppContext):
    trader = ctx.service("trader")
    if trader is None:
        raise bad_request("Trading engine isn't running", 503)
    return trader


@router.get("/portfolio")
async def portfolio(ctx: AppContext = Depends(get_ctx)):
    t = _trader(ctx)
    if t.broker is not None and t.account is None:
        await t.refresh(force=True)
    legs_by_symbol: dict[str, dict] = {}
    for o in t.open_orders:
        for leg in [o, *(o.get("legs") or [])]:
            if leg.get("status") not in ("new", "held", "accepted"):
                continue
            d = legs_by_symbol.setdefault(leg["symbol"], {})
            if leg.get("order_type") == "limit" and leg.get("limit_price"):
                d["take_profit"] = leg["limit_price"]
            if leg.get("order_type") in ("stop", "stop_limit") and leg.get("stop_price"):
                d["stop_loss"] = leg["stop_price"]
    positions = [{**p, **legs_by_symbol.get(p["symbol"], {})} for p in t.positions]
    plans = {p["symbol"]: p for p in crypto_exits.open_plans(ctx.db, getattr(t.broker, "mode", None))}
    for p in positions:
        if not crypto.is_crypto(p["symbol"]):
            continue
        plan = plans.get(p["symbol"])
        p["crypto"] = True
        p["exit_plan"] = None if plan is None else {
            "status": plan["status"], "stop_price": plan["stop_price"], "take_profit": plan["take_profit_price"],
            "stop_live": bool(p.get("stop_loss")), "note": plan["reason"] or ""}
        if plan is not None:
            p["take_profit"] = plan["take_profit_price"]  # watched by NewsTrader, not an order at Alpaca
    return {
        "mode": ctx.state.mode,
        "connected": t.account is not None,
        "error": t.last_error,
        "account": t.account,
        "market": t.clock,
        "positions": positions,
        "open_orders": t.open_orders,
        "kill_switch": ctx.state.kill_switch,
        "halted_today": ctx.state.halted_today,
    }


@router.get("/portfolio/history")
async def portfolio_history(period: str = Query("1M"), ctx: AppContext = Depends(get_ctx)):
    t = _trader(ctx)
    if period not in ("1D", "1W", "1M", "3M", "1A"):
        raise bad_request("period must be 1D, 1W, 1M, 3M or 1A")
    if t.broker is None:
        rows = ctx.db.query("SELECT ts, equity FROM equity_snapshots ORDER BY ts DESC LIMIT 500")
        rows.reverse()
        return {"timestamp": [r["ts"] for r in rows], "equity": [r["equity"] for r in rows], "source": "local"}
    try:
        hist = await asyncio.to_thread(t.broker.portfolio_history, period)
        return {**hist, "source": getattr(t, "broker_kind", "") or "alpaca"}
    except Exception as exc:
        raise bad_request(f"Couldn't load account history: {exc}", 502) from exc


@router.post("/positions/{symbol:path}/close")
async def close_position(symbol: str, ctx: AppContext = Depends(get_ctx)):
    try:
        order = await _trader(ctx).manual_close(symbol)
    except Exception as exc:
        raise bad_request(str(exc), 502) from exc
    return {"ok": True, "order": order}


@router.post("/orders/{order_id}/cancel")
async def cancel_order(order_id: str, ctx: AppContext = Depends(get_ctx)):
    try:
        await _trader(ctx).cancel_order(order_id)
    except Exception as exc:
        raise bad_request(str(exc), 502) from exc
    return {"ok": True}


@router.get("/trades")
async def trades(limit: int = Query(500, ge=1, le=5000), symbol: str = Query(""), mode: str = Query(""),
                 ctx: AppContext = Depends(get_ctx)):
    t = _trader(ctx)
    return {"trades": t.trade_log(limit=limit, symbol=symbol or None, mode=mode or None)}


def _csv_text(rows: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k) for k in CSV_COLUMNS})
    return buf.getvalue()


@router.get("/trades/export.csv")
async def export_csv_download(ctx: AppContext = Depends(get_ctx)):
    text = _csv_text(_trader(ctx).trade_log(limit=100000))
    name = f"newstrader-trades-{datetime.now():%Y%m%d-%H%M}.csv"
    return Response(text, media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{name}"'})


@router.post("/trades/export")
async def export_csv_file(ctx: AppContext = Depends(get_ctx)):
    """Save the trade log as a CSV in the exports folder (works inside the desktop window)."""
    text = _csv_text(_trader(ctx).trade_log(limit=100000))
    path = paths.exports_dir() / f"newstrader-trades-{datetime.now():%Y%m%d-%H%M%S}.csv"
    path.write_text(text, encoding="utf-8-sig", newline="")
    return {"ok": True, "path": str(path), "rows": text.count("\n") - 1}


@router.post("/kill")
async def kill(body: dict = Body(default={}), ctx: AppContext = Depends(get_ctx)):
    return await _trader(ctx).kill(close_positions=bool(body.get("close_positions")))


@router.post("/rearm")
async def rearm(ctx: AppContext = Depends(get_ctx)):
    await _trader(ctx).rearm()
    return {"ok": True, "kill_switch": ctx.state.kill_switch}


@router.post("/halt/clear")
async def clear_halt(ctx: AppContext = Depends(get_ctx)):
    """Resume trading today after a daily-loss halt (your decision)."""
    ctx.state.clear_halt()
    return {"ok": True}


@router.get("/live")
async def live_status(ctx: AppContext = Depends(get_ctx)):
    return {"mode": ctx.state.mode, "has_live_keys": ctx.keys.keys.has_alpaca_live,
            "phrase": live_guard.CONFIRM_PHRASE}


@router.post("/live/arm")
async def live_arm(body: dict = Body(...), ctx: AppContext = Depends(get_ctx)):
    try:
        live_guard.arm_live_trading(ctx.state, ctx.keys.keys, enable=body.get("enable") is True,
                                    phrase=str(body.get("phrase", "")))
    except live_guard.LiveTradingLocked as exc:
        raise bad_request(str(exc), 403) from exc
    trader = ctx.service("trader")
    if trader is not None:
        await trader.on_mode_changed()
    return {"mode": ctx.state.mode}


@router.post("/live/disarm")
async def live_disarm(ctx: AppContext = Depends(get_ctx)):
    live_guard.disarm_live_trading(ctx.state)
    trader = ctx.service("trader")
    if trader is not None:
        await trader.on_mode_changed()
    return {"mode": ctx.state.mode}


# ---------------------------------------------------------------------------------------------- the built-in simulator
def simulator_info(ctx: AppContext) -> dict:
    """Where trades go right now, and the built-in simulator's account (Settings -> Trading)."""
    t = ctx.service("trader")
    kind = getattr(t, "broker_kind", "") if t is not None else ""
    broker = getattr(t, "broker", None)
    s = ctx.config.settings.trading
    sim = None
    if kind == "simulator" and hasattr(broker, "describe"):
        sim = broker.describe()
        acct = getattr(t, "account", None) or {}
        sim.update(equity=acct.get("equity"), cash=acct.get("cash"))
    label = t._broker_label() if broker is not None and hasattr(t, "_broker_label") else ""
    return {"setting": s.broker, "in_use": kind, "label": label,
            "live": ctx.state.mode == "live", "has_alpaca_keys": ctx.keys.keys.has_alpaca_paper,
            "starting_cash": s.sim_starting_cash, "slippage_pct": s.sim_slippage_pct, "simulator": sim,
            "needs_alpaca": NEEDS_ALPACA, "reset_phrase": SIM_RESET_PHRASE}


@router.get("/simulator")
async def simulator_status(ctx: AppContext = Depends(get_ctx)):
    return simulator_info(ctx)


@router.post("/simulator/reset")
async def simulator_reset(body: dict = Body(default={}), ctx: AppContext = Depends(get_ctx)):
    """Start the built-in simulator over with fresh fake money. Needs the confirmation word typed exactly."""
    if str(body.get("phrase", "")).strip() != SIM_RESET_PHRASE:
        raise bad_request(f"Type {SIM_RESET_PHRASE} to confirm - the simulator was not reset.", 403)
    cash = body.get("starting_cash")
    if cash is not None:
        try:
            cash = float(cash)
        except (TypeError, ValueError) as exc:
            raise bad_request("Starting cash must be a number.") from exc
        try:
            ctx.config.update({"trading": {"sim_starting_cash": cash}})  # validated like the setting (1,000-10m)
        except Exception as exc:
            raise bad_request("Starting cash must be between $1,000 and $10,000,000.") from exc
    try:
        await _trader(ctx).reset_simulator()
    except Exception as exc:
        raise bad_request(f"Couldn't reset the simulator: {exc}", 500) from exc
    ctx.bus.publish("settings_changed", {})
    return simulator_info(ctx)


# ---------------------------------------------------------------------------------------------- "Why did it trade?"
@router.get("/trades/{order_id}/why")
async def why_trade(order_id: int, ctx: AppContext = Depends(get_ctx)):
    """The "Why did it trade?" card for a Trade Log row (the order's signal, or what's known without one)."""
    try:
        return await build_card(ctx, order_id=order_id)
    except NotFound as exc:
        raise bad_request(str(exc), 404) from exc
