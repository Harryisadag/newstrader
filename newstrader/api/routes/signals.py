from __future__ import annotations

import json
from datetime import UTC, datetime

from fastapi import APIRouter, Body, Depends, Query

from ...ai.costs import calls_today, spend_today
from ...ai.prefilter import prefilter
from ...context import AppContext
from ...performance.speed import signal_speed
from ...sources.base import NewsItem, stable_id
from ..deps import bad_request, get_ctx

router = APIRouter(tags=["signals"])


def _pipeline(ctx: AppContext):
    p = ctx.service("pipeline")
    if p is None:
        raise bad_request("AI engine isn't running", 503)
    return p


@router.get("/signals")
async def list_signals(
    limit: int = Query(300, ge=1, le=5000),
    ticker: str = Query(""),
    direction: str = Query(""),
    action: str = Query(""),
    min_confidence: int = Query(0, ge=0, le=100),
    include_merged: bool = Query(False),
    review_only: bool = Query(False),
    engine: str = Query("", description="'main' = leave out Pro AI's watch-only signals, 'pro' = only Pro AI's"),
    ctx: AppContext = Depends(get_ctx),
):
    sql = ("SELECT s.*, o.submitted_at AS order_submitted_at FROM signals s LEFT JOIN orders o ON o.id = s.order_id "
           "WHERE s.confidence >= ?")
    params: list = [min_confidence]
    if not include_merged:
        sql += " AND s.merged_into IS NULL"
    if ticker:
        sql += " AND s.ticker = ?"
        params.append(ticker.upper().strip())
    if direction:
        sql += " AND s.direction = ?"
        params.append(direction)
    if action == "traded":
        sql += " AND s.traded = 1"
    elif action:
        sql += " AND s.action = ?"
        params.append(action)
    if review_only:
        sql += " AND s.action = 'review' AND (s.review_status IS NULL OR s.review_status = 'pending')"
    if engine == "main":
        sql += " AND COALESCE(s.action, '') != 'watch'"
    elif engine == "pro":
        sql += " AND s.engine = 'pro'"
    sql += " ORDER BY s.id DESC LIMIT ?"
    params.append(limit)
    rows = ctx.db.query(sql, params)
    for r in rows:
        r["sources_seen"] = json.loads(r["sources_seen"] or "[]")
        r["speed"] = signal_speed(r, r.pop("order_submitted_at"))
    return {"signals": rows}


@router.get("/signals/{signal_id}")
async def signal_detail(signal_id: int, ctx: AppContext = Depends(get_ctx)):
    sig = ctx.db.query_one("SELECT * FROM signals WHERE id = ?", (signal_id,))
    if sig is None:
        raise bad_request("Unknown signal", 404)
    sig["sources_seen"] = json.loads(sig["sources_seen"] or "[]")
    analysis = ctx.db.query_one("SELECT * FROM analyses WHERE id = ?", (sig["analysis_id"],))
    item = None
    if analysis and analysis.get("item_id"):
        item = ctx.db.query_one("SELECT id, title, body, url, source_name, published_at, received_at FROM news_items "
                                "WHERE id = ?", (analysis["item_id"],))
    merged = ctx.db.query("SELECT id, created_at, source_name, confidence, reasoning FROM signals WHERE merged_into = ? "
                          "ORDER BY id", (signal_id,))
    order = None
    if sig.get("order_id"):
        order = ctx.db.query_one("SELECT * FROM orders WHERE id = ?", (sig["order_id"],))
        if order:
            order.pop("raw", None)
    sig["speed"] = signal_speed(sig, order["submitted_at"] if order else None)
    # the other engine's reading of the same story: Pro AI's for a main-engine signal, the main engine's for Pro AI's
    other = None
    if analysis and analysis.get("engine") == "pro" and analysis.get("main_analysis_id"):
        other = ctx.db.query_one("SELECT id, engine, model, status, latency_ms FROM analyses WHERE id = ?",
                                 (analysis["main_analysis_id"],))
    elif analysis:
        other = ctx.db.query_one("SELECT id, engine, model, status, latency_ms, agreement, pro_reason FROM analyses "
                                 "WHERE main_analysis_id = ? AND engine = 'pro' ORDER BY id DESC LIMIT 1",
                                 (analysis["id"],))
    return {"signal": sig, "analysis": analysis, "item": item, "merged": merged, "order": order,
            "other_analysis": other}


@router.post("/signals/{signal_id}/approve")
async def approve(signal_id: int, ctx: AppContext = Depends(get_ctx)):
    try:
        return await _pipeline(ctx).approve(signal_id)
    except KeyError as exc:
        raise bad_request("Unknown signal", 404) from exc
    except PermissionError as exc:
        raise bad_request(str(exc), 409) from exc
    except RuntimeError as exc:
        raise bad_request(str(exc), 503) from exc


@router.post("/signals/{signal_id}/dismiss")
async def dismiss(signal_id: int, ctx: AppContext = Depends(get_ctx)):
    _pipeline(ctx).dismiss(signal_id)
    return {"ok": True}


@router.get("/news")
async def recent_news(limit: int = Query(200, ge=1, le=2000), source_id: str = Query(""),
                      kind: str = Query("text"), ctx: AppContext = Depends(get_ctx)):
    sql = ("SELECT id, source_id, source_name, source_type, title, url, speaker, published_at, received_at, status, "
           "candidates, language FROM news_items WHERE 1=1")
    params: list = []
    if kind == "text":
        sql += " AND source_type != 'stream'"
    elif kind == "transcript":
        sql += " AND source_type = 'stream'"
    if source_id:
        sql += " AND source_id = ?"
        params.append(source_id)
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    rows = ctx.db.query(sql, params)
    for r in rows:
        try:
            c = json.loads(r.pop("candidates") or "{}")
        except ValueError:
            c = {}
        r["candidates"] = [x["symbol"] for x in c.get("candidates", [])]
        r["keywords"] = c.get("keywords", [])
    return {"news": rows}


@router.get("/ai/stats")
async def ai_stats(ctx: AppContext = Depends(get_ctx)):
    p = ctx.service("pipeline")
    return {
        "spend_today": round(spend_today(ctx.db), 4),
        "calls_today": calls_today(ctx.db),
        "cap": ctx.config.settings.ai.daily_spend_cap_usd,
        "engine": ctx.config.settings.ai.engine,
        "model": p.local.label() if p and ctx.config.settings.ai.engine == "local" else ctx.config.settings.ai.model,
        "queue": p.queue.qsize() if p else 0,
        "tickers_loaded": len(p.tickers.by_symbol) if p else 0,
        "stats": p.stats if p else {},
    }


@router.post("/ai/test")
async def test_ai(body: dict = Body(...), ctx: AppContext = Depends(get_ctx)):
    """Analyse pasted text with the selected engine. Shows the result but never trades
    (with Claude it costs one call; the local engine is free)."""
    text = str(body.get("text", "")).strip()
    if len(text) < 10:
        raise bad_request("Paste a headline or paragraph (at least 10 characters).")
    p = _pipeline(ctx)
    if not p.tickers.loaded:
        raise bad_request("The ticker list isn't loaded yet - add your Alpaca keys and wait a minute.", 409)
    claude = ctx.config.settings.ai.engine == "claude"
    if claude and spend_today(ctx.db) >= ctx.config.settings.ai.daily_spend_cap_usd:
        raise bad_request("Today's Claude spend cap is reached (Settings -> AI engine).", 429)
    item = NewsItem(source_id="manual-test", source_type="manual", source_name="Manual test",
                    external_id=stable_id(text, datetime.now(UTC).isoformat()), title=text[:300],
                    body=text if len(text) > 300 else "", published_at=datetime.now(UTC))
    pre = prefilter(item.text, p.tickers, [], claude, country_etfs=ctx.config.settings.ml.country_etfs)
    result = await p.analyze_item(item, pre, dry_run=True)
    return {"prefilter": pre.as_dict(), **result}


@router.post("/tickers/sync")
async def sync_tickers(ctx: AppContext = Depends(get_ctx)):
    n = await _pipeline(ctx).sync_tickers()
    if not n:
        raise bad_request("Couldn't download the ticker list - check Alpaca keys (Logs -> Diagnostics).", 502)
    return {"count": n}
