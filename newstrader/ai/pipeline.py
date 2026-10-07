"""The news -> signal -> trade pipeline.

    source item -> save -> stage 1 pre-filter -> same-story check -> daily spend cap
                -> queue -> stage 2 Claude -> validate -> merge same ticker/direction -> trader / alerts
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from datetime import UTC, datetime, timedelta

from ..context import AppContext
from ..db import iso, utcnow
from ..sources.base import NewsItem, safe_url
from ..state import trading_day
from .claude_client import ClaudeAnalyzer
from .costs import spend_today
from .dedupe import StoryDeduper
from .prefilter import PrefilterResult, prefilter
from .tickers import TickerTable
from .validator import validate_response

log = logging.getLogger(__name__)

# Items waiting longer than this in the queue are skipped - the trade would be stale.
MAX_QUEUE_AGE = timedelta(minutes=10)
# Text items published longer ago than this aren't analysed (old articles re-appearing in feeds).
MAX_ITEM_AGE = timedelta(hours=3)


class Pipeline:
    name = "pipeline"

    def __init__(self, ctx: AppContext, analyzer: ClaudeAnalyzer | None = None, tickers: TickerTable | None = None):
        self.ctx = ctx
        self.tickers = tickers or TickerTable(ctx.db)
        self.analyzer = analyzer or ClaudeAnalyzer(ctx)
        self.deduper = StoryDeduper()
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=500)
        self._workers: list[asyncio.Task] = []
        self._tasks: list[asyncio.Task] = []
        self._cap_alert_day: str | None = None
        self.stats = {"received": 0, "filtered": 0, "duplicates": 0, "analysed": 0, "signals": 0}

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        n = await asyncio.to_thread(self.tickers.load_from_db)
        if n:
            log.info("Loaded %d tickers from cache", n)
        self._tasks.append(asyncio.create_task(self._ticker_refresh_loop(), name="ticker-refresh"))
        self._restart_workers()
        loop = asyncio.get_running_loop()

        def on_settings(settings) -> None:  # called from whichever thread saved the settings
            if settings.ai.max_concurrent_calls != len(self._workers) and not loop.is_closed():
                loop.call_soon_threadsafe(self._restart_workers)

        self.ctx.config.on_change(on_settings)

    async def stop(self) -> None:
        for t in self._tasks + self._workers:
            t.cancel()
        for t in self._tasks + self._workers:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t
        self._tasks.clear()
        self._workers.clear()

    async def on_keys_changed(self) -> None:
        if self.tickers.needs_refresh(max_age_hours=0.01):
            await self.sync_tickers()

    def _restart_workers(self) -> None:
        for w in self._workers:
            w.cancel()
        n = self.ctx.config.settings.ai.max_concurrent_calls
        self._workers = [asyncio.create_task(self._worker(i), name=f"claude-worker-{i}") for i in range(n)]

    async def _ticker_refresh_loop(self) -> None:
        await asyncio.sleep(2)
        while True:
            try:
                if self.tickers.needs_refresh():
                    await self.sync_tickers()
            except Exception:
                log.exception("ticker refresh failed")
            await asyncio.sleep(1800)

    async def sync_tickers(self) -> int:
        trader = self.ctx.service("trader")
        broker = getattr(trader, "broker", None)
        if broker is None:
            if not self.tickers.loaded:
                self.ctx.state.set_status("tickers", "warn", "Ticker list needs Alpaca keys")
            return 0
        try:
            assets = await asyncio.to_thread(broker.all_assets)
            n = await asyncio.to_thread(self.tickers.replace_all, assets)
            self.ctx.state.set_status("tickers", "ok", f"{n:,} US-listed tickers loaded")
            return n
        except Exception as exc:
            self.ctx.state.set_status("tickers", "error", f"Couldn't download ticker list: {exc}")
            log.warning("Ticker sync failed: %s", exc)
            return 0

    # ------------------------------------------------------------------ intake
    async def submit(self, item: NewsItem) -> dict:
        """Called by every source. Returns what happened to the item (for tests and the UI)."""
        self.stats["received"] += 1
        item.url = safe_url(item.url)
        now = utcnow()
        db = self.ctx.db
        row = {
            "source_id": item.source_id, "source_type": item.source_type, "source_name": item.source_name,
            "external_id": item.external_id, "title": item.title[:1000], "body": (item.body or "")[:20000],
            "url": item.url, "speaker": item.speaker, "published_at": iso(item.published_at) if item.published_at else None,
            "received_at": iso(now), "symbols": json.dumps(item.symbols), "status": "received",
        }
        try:
            item.db_id = db.insert("news_items", row)
        except Exception:  # unique (source_id, external_id): already seen
            return {"status": "seen"}

        ai = self.ctx.config.settings.ai
        pre = await asyncio.to_thread(prefilter, item.text, self.tickers, item.symbols, ai.analyze_keyword_only)
        status = "queued"
        dup_of = None
        reason = ""
        if not pre.hit:
            status, reason = "filtered", "no company, ticker or market keyword"
        elif item.kind == "text" and item.published_at and now - item.published_at > MAX_ITEM_AGE:
            status, reason = "stale", "published too long ago"
        elif item.kind == "text":
            dup_of = self.deduper.check_and_add(item_id=item.db_id, text=item.title, url=item.url,
                                                source_id=item.source_id, at=now,
                                                window_minutes=ai.story_dedupe_minutes, threshold=ai.story_similarity)
            if dup_of:
                status, reason = "duplicate", f"same story as item #{dup_of}"
        if status == "queued" and spend_today(db) >= ai.daily_spend_cap_usd:
            status, reason = "skipped_cap", "daily Claude spend cap reached"
            await self._spend_cap_alert()

        db.update("news_items", item.db_id, {"prefilter_hit": int(pre.hit), "candidates": json.dumps(pre.as_dict()),
                                             "status": status, "duplicate_of": dup_of,
                                             "fingerprint": None})
        if status == "filtered":
            self.stats["filtered"] += 1
        if status == "duplicate":
            self.stats["duplicates"] += 1
            self._note_corroboration(dup_of, item)

        self.ctx.bus.publish("news", self._news_event(item, pre, status, reason))
        if status == "queued":
            try:
                self.queue.put_nowait((now, item, pre))
            except asyncio.QueueFull:
                db.update("news_items", item.db_id, {"status": "dropped"})
                log.warning("Analysis queue full - dropped %s", item.title[:80])
                return {"status": "dropped"}
        return {"status": status, "reason": reason, "prefilter": pre.as_dict(), "duplicate_of": dup_of}

    def _news_event(self, item: NewsItem, pre: PrefilterResult, status: str, reason: str) -> dict:
        return {"id": item.db_id, "source_id": item.source_id, "source_name": item.source_name,
                "source_type": item.source_type, "kind": item.kind, "title": item.title[:300], "url": item.url,
                "speaker": item.speaker,
                "published_at": iso(item.published_at) if item.published_at else None, "received_at": iso(),
                "status": status, "reason": reason, "candidates": [c.symbol for c in pre.candidates],
                "keywords": pre.keywords}

    async def _spend_cap_alert(self) -> None:
        day = trading_day()
        if self._cap_alert_day == day:
            return
        self._cap_alert_day = day
        cap = self.ctx.config.settings.ai.daily_spend_cap_usd
        msg = f"Today's estimated Claude cost reached ${cap:.2f}. No more AI analysis until tomorrow (or raise the cap)."
        log.warning(msg)
        alerts = self.ctx.service("alerts")
        if alerts is not None:
            await alerts.send("spend_cap", "Claude daily spend cap reached", msg, "warn")
        else:
            self.ctx.bus.publish("toast", {"kind": "warn", "title": "Claude daily spend cap reached", "message": msg})

    def _note_corroboration(self, original_item_id: int | None, item: NewsItem) -> None:
        """A duplicate story from another source strengthens the original item's signals."""
        if not original_item_id:
            return
        rows = self.ctx.db.query(
            "SELECT s.id, s.sources_seen FROM signals s JOIN analyses a ON a.id = s.analysis_id "
            "WHERE a.item_id = ? AND a.item_kind = 'news' AND s.merged_into IS NULL", (original_item_id,))
        for r in rows:
            seen = json.loads(r["sources_seen"] or "[]")
            if item.source_name not in seen:
                seen.append(item.source_name)
            self.ctx.db.execute("UPDATE signals SET corroborations = corroborations + 1, sources_seen = ? WHERE id = ?",
                                (json.dumps(seen), r["id"]))
            self.ctx.bus.publish("signal_updated", {"id": r["id"]})

    # ------------------------------------------------------------------ analysis
    async def _worker(self, n: int) -> None:
        while True:
            queued_at, item, pre = await self.queue.get()
            try:
                if utcnow() - queued_at > MAX_QUEUE_AGE:
                    self.ctx.db.update("news_items", item.db_id, {"status": "stale"})
                    continue
                await self.analyze_item(item, pre)
            except Exception:
                log.exception("Analysis failed for %s", item.title[:80])
                with contextlib.suppress(Exception):
                    self.ctx.db.update("news_items", item.db_id, {"status": "error"})
            finally:
                self.queue.task_done()

    def _market_open(self) -> bool | None:
        trader = self.ctx.service("trader")
        clock = getattr(trader, "clock", None) if trader else None
        return None if not clock else bool(clock.get("is_open"))

    async def analyze_item(self, item: NewsItem, pre: PrefilterResult, dry_run: bool = False) -> dict:
        ai = self.ctx.config.settings.ai
        if not dry_run and spend_today(self.ctx.db) >= ai.daily_spend_cap_usd:
            self.ctx.db.update("news_items", item.db_id, {"status": "skipped_cap"})
            await self._spend_cap_alert()
            return {"status": "skipped_cap", "signals": []}
        now = datetime.now(UTC)
        result = await self.analyzer.analyze(item, pre, self._market_open(), now)
        validation = validate_response(result.text, self.tickers, ai.max_signals_per_item) if result.ok else None
        if not result.ok:
            status, error = result.status, result.error
        else:
            status, error = validation.status, validation.summary or None
        analysis_id = self.ctx.db.insert("analyses", {
            "created_at": iso(), "trading_day": trading_day(), "item_kind": "transcript" if item.kind == "transcript" else "news",
            "item_id": item.db_id, "source_id": item.source_id, "source_name": item.source_name, "model": result.model,
            "input_tokens": result.input_tokens, "output_tokens": result.output_tokens,
            "cache_read_tokens": result.cache_read_tokens, "cache_write_tokens": result.cache_write_tokens,
            "cost_usd": result.cost_usd, "latency_ms": result.latency_ms, "status": status, "error": error,
            "raw_response": result.text[:20000], "is_backtest": 0,
        })
        self.stats["analysed"] += 1
        if item.db_id:
            self.ctx.db.update("news_items", item.db_id, {"status": "analysed" if status == "ok" else status})
        if status != "ok" or error:
            log.warning("AI response for '%s' %s: %s", item.title[:60], "rejected" if status != "ok" else "partly rejected",
                        error)
        self.ctx.bus.publish("analysis", {"id": analysis_id, "item_id": item.db_id, "status": status,
                                          "cost_usd": result.cost_usd, "error": error})
        out = []
        if validation is not None:
            for sig in validation.signals:
                out.append(await self._handle_signal(sig, item, analysis_id, dry_run=dry_run))
        if item.db_id and any(sig.direction != "neutral" for sig in (validation.signals if validation else [])):
            self.ctx.db.update("news_items", item.db_id, {"status": "signal"})
        return {"status": status, "error": error, "analysis_id": analysis_id, "signals": out,
                "cost_usd": result.cost_usd, "raw": result.text}

    # ------------------------------------------------------------------ signals
    async def _handle_signal(self, sig, item: NewsItem, analysis_id: int, dry_run: bool = False) -> dict:
        s = self.ctx.config.settings
        db = self.ctx.db
        now = utcnow()
        row = {
            "analysis_id": analysis_id, "created_at": iso(now), "ticker": sig.ticker, "company": sig.company,
            "direction": sig.direction, "confidence": sig.confidence, "speaker": sig.speaker or item.speaker or item.source_name,
            "source_id": item.source_id, "source_name": item.source_name, "source_type": item.source_type,
            "reasoning": sig.reasoning, "bull_case": sig.bull_case, "bear_case": sig.bear_case,
            "time_sensitivity": sig.time_sensitivity, "headline": item.title[:500], "url": item.url,
            "sources_seen": json.dumps([item.source_name]),
        }
        if dry_run:
            return {**row, "action": "test", "action_reason": "Test only - not traded", "traded": 0}

        existing = None
        if sig.direction != "neutral" and s.ai.signal_dedupe_minutes > 0:
            existing = db.query_one(
                "SELECT * FROM signals WHERE ticker = ? AND direction = ? AND merged_into IS NULL AND created_at >= ? "
                "ORDER BY id DESC LIMIT 1",
                (sig.ticker, sig.direction, iso(now - timedelta(minutes=s.ai.signal_dedupe_minutes))))
        if existing is not None:
            return await self._merge_into(existing, row, sig)

        row.update({"action": "pending", "action_reason": "", "traded": 0})
        row["id"] = db.insert("signals", row)
        self.stats["signals"] += 1
        await self._route(row)
        return db.query_one("SELECT * FROM signals WHERE id = ?", (row["id"],)) or row

    async def _merge_into(self, existing: dict, row: dict, sig) -> dict:
        db = self.ctx.db
        row.update({"action": "merged", "action_reason": f"Same signal as #{existing['id']} (counted once)",
                    "traded": 0, "merged_into": existing["id"]})
        row["id"] = db.insert("signals", row)
        seen = json.loads(existing["sources_seen"] or "[]")
        if row["source_name"] not in seen:
            seen.append(row["source_name"])
        new_conf = max(int(existing["confidence"]), sig.confidence)
        db.execute("UPDATE signals SET corroborations = corroborations + 1, sources_seen = ?, confidence = ? WHERE id = ?",
                   (json.dumps(seen), new_conf, existing["id"]))
        merged = db.query_one("SELECT * FROM signals WHERE id = ?", (existing["id"],))
        self.ctx.bus.publish("signal_updated", {"id": existing["id"]})
        log.info("Merged %s %s signal from %s into #%s (now %d sources)", sig.ticker, sig.direction,
                 row["source_name"], existing["id"], len(seen))
        # A stronger confirmation can turn an untraded review/blocked signal into a trade.
        buy = self.ctx.config.settings.trading.buy_threshold
        if not merged["traded"] and new_conf >= buy and merged["action"] in ("review", "blocked", "ignored", "error"):
            await self._route(merged)
        return row

    async def _route(self, signal: dict) -> None:
        """Send a signal to the trader (or flag it for manual review) and record what happened."""
        db = self.ctx.db
        trader = self.ctx.service("trader")
        if signal["direction"] == "neutral":
            action, reason, traded, order_id = "ignored", "Neutral signal - no trade.", False, None
        elif trader is None:
            action, reason, traded, order_id = "ignored", "Trading engine not running.", False, None
        else:
            res = await trader.handle_signal(signal)
            action, reason, traded, order_id = res["action"], res["reason"], res["traded"], res.get("order_db_id")
        db.update("signals", signal["id"], {"action": action, "action_reason": reason, "traded": int(traded),
                                            "order_id": order_id,
                                            "review_status": "pending" if action == "review" else None})
        event = db.query_one("SELECT * FROM signals WHERE id = ?", (signal["id"],))
        self.ctx.bus.publish("signal", event)
        tag = "TRADED" if traded else action.upper()
        log.info("Signal #%s %s %s conf %s from %s -> %s: %s", signal["id"], signal["ticker"], signal["direction"],
                 signal["confidence"], signal["source_name"], tag, reason)
        if action == "review":
            await self._review_alert(event)

    async def _review_alert(self, sig: dict) -> None:
        title = f"Manual review: {sig['ticker']} {sig['direction']} ({sig['confidence']})"
        msg = f"{sig['reasoning']}\nSource: {sig['source_name']}. Approve or dismiss in the Signals tab."
        alerts = self.ctx.service("alerts")
        if alerts is not None:
            await alerts.send("manual_review", title, msg, "warn")
        else:
            self.ctx.bus.publish("toast", {"kind": "warn", "title": title, "message": msg})

    async def approve(self, signal_id: int) -> dict:
        db = self.ctx.db
        sig = db.query_one("SELECT * FROM signals WHERE id = ?", (signal_id,))
        if sig is None:
            raise KeyError(signal_id)
        if sig["traded"]:
            return {"action": sig["action"], "reason": "Already traded", "traded": True}
        trader = self.ctx.service("trader")
        if trader is None:
            raise RuntimeError("Trading engine not running")
        res = await trader.handle_signal(sig, manual=True)
        db.update("signals", signal_id, {"action": res["action"], "action_reason": res["reason"] + " (approved by you)",
                                         "traded": int(res["traded"]), "order_id": res.get("order_db_id"),
                                         "review_status": "approved"})
        self.ctx.bus.publish("signal_updated", {"id": signal_id})
        return res

    def dismiss(self, signal_id: int) -> None:
        self.ctx.db.update("signals", signal_id, {"review_status": "dismissed"})
        self.ctx.bus.publish("signal_updated", {"id": signal_id})
