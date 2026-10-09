"""The news -> signal -> trade pipeline.

    source item -> save -> stage 1 pre-filter -> same-story check -> (Claude only: daily spend cap)
                -> queue -> stage 2 AI engine (local ML or Claude) -> validate -> merge same ticker/direction
                -> trader / alerts

Pro AI (newstrader/llm, optional): after the main engine, the stories it finds hard (llm/routing.py) are queued for
Pro AI. Its signals are stored with action "watch" - price-tracked for the scoreboard, never traded, alerted or
merged into tradeable signals. In judge mode it can also send a main-engine signal it disagrees with to manual review
(waiting up to max_wait_seconds for its answer) or raise a manual-review signal the main engine missed. It never
places a trade by itself.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from datetime import UTC, datetime, timedelta

from ..context import AppContext
from ..db import iso, utcnow
from ..llm import routing
from ..llm.service import ProJob
from ..ml.engine import LocalMLEngine
from ..performance.speed import signal_speed
from ..sources.base import NewsItem, safe_url
from ..sources.lang import detect_language
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

    def __init__(self, ctx: AppContext, analyzer: ClaudeAnalyzer | None = None, tickers: TickerTable | None = None,
                 local: LocalMLEngine | None = None):
        self.ctx = ctx
        self.tickers = tickers or TickerTable(ctx.db)
        self.analyzer = analyzer or ClaudeAnalyzer(ctx)
        self.local = local or LocalMLEngine(ctx)
        self.local.tickers = self.tickers
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
        self._tasks.append(asyncio.create_task(self.warm_local_engine(), name="ml-warmup"))
        if getattr(self.ctx.config, "engine_was_defaulted", False) and self.engine_name == "local":
            self.ctx.config.engine_was_defaulted = False
            self._tasks.append(asyncio.create_task(self._engine_changed_notice(), name="engine-notice"))
        loop = asyncio.get_running_loop()
        seen = {"engine": self.ctx.config.settings.ai.engine, "sentiment": self.ctx.config.settings.ml.sentiment_model,
                "use": self.ctx.config.settings.ml.use_trained_model}

        def on_settings(settings) -> None:  # called from whichever thread saved the settings
            if loop.is_closed():
                return
            if settings.ai.max_concurrent_calls != len(self._workers):
                loop.call_soon_threadsafe(self._restart_workers)
            now = {"engine": settings.ai.engine, "sentiment": settings.ml.sentiment_model,
                   "use": settings.ml.use_trained_model}
            if now != seen:
                seen.update(now)
                loop.call_soon_threadsafe(lambda: self._tasks.append(
                    asyncio.create_task(self.warm_local_engine(), name="ml-warmup")))

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
        self._workers = [asyncio.create_task(self._worker(i), name=f"ai-worker-{i}") for i in range(n)]

    # ------------------------------------------------------------------ engines
    @property
    def engine_name(self) -> str:
        return self.ctx.config.settings.ai.engine

    def engine(self, name: str | None = None):
        return self.local if (name or self.engine_name) == "local" else self.analyzer

    async def _engine_changed_notice(self) -> None:
        msg = ("This update switched the AI engine to the free local machine-learning engine (FinBERT). Claude is "
               "no longer called. To go back, pick Claude in Settings -> AI engine.")
        log.warning(msg)
        await asyncio.sleep(3)  # let the window connect so the toast shows too
        alerts = self.ctx.service("alerts")
        if alerts is not None:
            await alerts.send("info", "AI engine changed to local machine learning", msg, "warn")

    async def warm_local_engine(self) -> None:
        """Load FinBERT (downloading it the first time) and the trained price model in the background."""
        self._tasks = [t for t in self._tasks if not t.done()]
        if self.engine_name != "local":
            self.ctx.state.set_status("ml", "off", "Not in use - Claude engine selected (Settings -> AI engine)")
            return
        self.ctx.state.set_status("ml", "starting", "Loading the sentiment model (first run downloads ~110 MB)...")
        try:
            await self.local.ensure_ready()
            await asyncio.to_thread(self.local.load_price_model)
        except Exception as exc:
            log.exception("Local ML engine failed to load")
            self.ctx.state.set_status("ml", "error", f"Local ML engine failed to load: {exc}")
            return
        self.update_ml_status()

    def update_ml_status(self) -> None:
        if self.engine_name != "local":
            self.ctx.state.set_status("ml", "off", "Not in use - Claude engine selected (Settings -> AI engine)")
            return
        st = self.local.model_status()
        parts = [f"Sentiment: {st['sentiment']}"]
        pm = st["price_model"]
        if st["price_model_active"]:
            parts.append(f"price model active (trained {str(pm['trained_at'])[:10]} on {pm['n_samples']:,} headlines)")
        elif pm:
            parts.append(f"price model not used: {pm['why_inactive']}")
        else:
            parts.append("price model not trained yet (Backtest tab -> Local ML model)")
        level = "warn" if st["sentiment_error"] else "ok"
        self.ctx.state.set_status("ml", level, " · ".join(parts))

    async def _ticker_refresh_loop(self) -> None:
        await asyncio.sleep(2)
        while True:
            try:
                if self.tickers.needs_refresh():
                    await self.sync_tickers()
            except Exception:
                log.exception("ticker refresh failed")
            # retry every minute until the list has loaded once, then check twice an hour
            await asyncio.sleep(1800 if self.tickers.loaded else 60)

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
        if not item.language:
            item.language = detect_language(item.text)
        now = utcnow()
        item.received_at = now
        db = self.ctx.db
        row = {
            "source_id": item.source_id, "source_type": item.source_type, "source_name": item.source_name,
            "external_id": item.external_id, "title": item.title[:1000], "body": (item.body or "")[:20000],
            "url": item.url, "speaker": item.speaker, "published_at": iso(item.published_at) if item.published_at else None,
            "received_at": iso(now), "symbols": json.dumps(item.symbols), "status": "received",
            "language": item.language or None,
        }
        try:
            item.db_id = db.insert("news_items", row)
        except Exception:  # unique (source_id, external_id): already seen
            return {"status": "seen"}

        ai = self.ctx.config.settings.ai
        claude = ai.engine == "claude"
        keyword_only = ai.analyze_keyword_only and claude  # the local engine needs a named company
        pre = await asyncio.to_thread(prefilter, item.text, self.tickers, item.symbols, keyword_only, 10,
                                      self.ctx.config.settings.ml.country_etfs)
        status = "queued"
        dup_of = None
        reason = ""
        if not claude and item.language not in ("", "en"):
            # FinBERT reads English only: a foreign headline would be misread, not just missed
            status, reason = "filtered", (f"not in English ({item.language}) - the local engine reads English only "
                                          "(the Claude engine reads any language; TV can be translated to English)")
        elif not pre.hit:
            status, reason = "filtered", ("no company, ticker or market keyword" if claude
                                          else "no company or ticker named")
        elif not self.tickers.loaded:
            # Without the ticker list every answer would be rejected - don't analyse (or pay for Claude) yet.
            status, reason = "no_tickers", "ticker list not loaded yet (needs Alpaca keys)"
        elif item.kind == "text" and item.published_at and now - item.published_at > MAX_ITEM_AGE:
            status, reason = "stale", "published too long ago"
        elif item.kind == "text":
            dup_of = self.deduper.check_and_add(item_id=item.db_id, text=item.title, url=item.url,
                                                source_id=item.source_id, at=now,
                                                window_minutes=ai.story_dedupe_minutes, threshold=ai.story_similarity)
            if dup_of:
                status, reason = "duplicate", f"same story as item #{dup_of}"
        if status == "queued" and claude and spend_today(db) >= ai.daily_spend_cap_usd:
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
        if status == "filtered" and self.tickers.loaded and not (
                item.kind == "text" and item.published_at and now - item.published_at > MAX_ITEM_AGE):
            self._offer_to_pro(item, pre, None, None, [])  # e.g. not in English, or no company named
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
                "keywords": pre.keywords, "language": item.language}

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
        engine_name = ai.engine
        if engine_name == "claude" and not dry_run and spend_today(self.ctx.db) >= ai.daily_spend_cap_usd:
            self.ctx.db.update("news_items", item.db_id, {"status": "skipped_cap"})
            await self._spend_cap_alert()
            return {"status": "skipped_cap", "signals": []}
        now = datetime.now(UTC)
        result = await self.engine(engine_name).analyze(item, pre, self._market_open(), now)
        decided_at = utcnow()
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
            "raw_response": result.text[:20000], "is_backtest": 0, "engine": result.engine,
        })
        self.stats["analysed"] += 1
        if item.db_id:
            self.ctx.db.update("news_items", item.db_id, {"status": "analysed" if status == "ok" else status})
        if status != "ok" or error:
            log.warning("AI response for '%s' %s: %s", item.title[:60], "rejected" if status != "ok" else "partly rejected",
                        error)
        self.ctx.bus.publish("analysis", {"id": analysis_id, "item_id": item.db_id, "status": status,
                                          "cost_usd": result.cost_usd, "error": error})
        signals = validation.signals if validation is not None else []
        job = None if dry_run else self._pro_job(item, pre, result.engine, analysis_id, signals)
        verdict = None
        if job is not None and job.judge and any(sig.direction != "neutral" for sig in signals):
            verdict = await self.ctx.service("pro_ai").ask(job)  # judge mode: hold the trade for Pro AI's view
        out = []
        for sig in signals:
            hold, pro_dir = self._judge_signal(sig, verdict)
            out.append(await self._handle_signal(sig, item, analysis_id, dry_run=dry_run, engine=result.engine,
                                                 decided_at=decided_at, hold=hold, pro_direction=pro_dir))
        if item.db_id and any(sig.direction != "neutral" for sig in signals):
            self.ctx.db.update("news_items", item.db_id, {"status": "signal"})
        if job is not None and job.future is None:
            self.ctx.service("pro_ai").offer(job)  # watch: after the main engine, never in its way
        return {"status": status, "error": error, "analysis_id": analysis_id, "signals": out,
                "cost_usd": result.cost_usd, "raw": result.text, "engine": result.engine, "model": result.model}

    # ------------------------------------------------------------------ signals
    def _signal_row(self, sig, item: NewsItem, analysis_id: int, engine: str, now: datetime,
                    decided_at: datetime | None) -> dict:
        news_time = item.news_time
        return {
            "analysis_id": analysis_id, "created_at": iso(now), "ticker": sig.ticker, "company": sig.company,
            "direction": sig.direction, "confidence": sig.confidence, "speaker": sig.speaker or item.speaker or item.source_name,
            "source_id": item.source_id, "source_name": item.source_name, "source_type": item.source_type,
            "reasoning": sig.reasoning, "bull_case": sig.bull_case, "bear_case": sig.bear_case,
            "time_sensitivity": sig.time_sensitivity, "headline": item.title[:500], "url": item.url,
            "sources_seen": json.dumps([item.source_name]), "engine": engine,
            "event": getattr(sig, "event", "") or None, "flags": json.dumps(sig.flags) if getattr(sig, "flags", None) else None,
            "news_published_at": iso(news_time) if news_time else None,
            "news_received_at": iso(item.received_at) if item.received_at else None,
            "decided_at": iso(decided_at or now),
        }

    async def _handle_signal(self, sig, item: NewsItem, analysis_id: int, dry_run: bool = False,
                             engine: str = "claude", decided_at: datetime | None = None, hold: str = "",
                             pro_direction: str | None = None) -> dict:
        """`hold`: Pro AI (judge mode) disagreed - send the signal to manual review with this reason instead of the
        trader. `pro_direction`: Pro AI's call on this stock, when it read the story."""
        s = self.ctx.config.settings
        db = self.ctx.db
        now = utcnow()
        row = self._signal_row(sig, item, analysis_id, engine, now, decided_at)
        if pro_direction is not None:
            row["pro_direction"] = pro_direction
        if dry_run:
            return {**row, "action": "test", "action_reason": "Test only - not traded", "traded": 0}

        existing = None
        if sig.direction != "neutral" and s.ai.signal_dedupe_minutes > 0:
            existing = db.query_one(  # (Pro AI's own signals are kept apart: they never get traded through this)
                "SELECT * FROM signals WHERE ticker = ? AND direction = ? AND merged_into IS NULL AND created_at >= ? "
                "AND COALESCE(engine, '') != 'pro' ORDER BY id DESC LIMIT 1",
                (sig.ticker, sig.direction, iso(now - timedelta(minutes=s.ai.signal_dedupe_minutes))))
        if existing is not None:
            return await self._merge_into(existing, row, sig, reroute=not hold)

        row.update({"action": "pending", "action_reason": "", "traded": 0})
        row["id"] = db.insert("signals", row)
        self.stats["signals"] += 1
        await self._route(row, force_review=hold)
        return db.query_one("SELECT * FROM signals WHERE id = ?", (row["id"],)) or row

    async def _merge_into(self, existing: dict, row: dict, sig, reroute: bool = True) -> dict:
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
        if (reroute and not merged["traded"] and new_conf >= buy
                and merged["action"] in ("review", "blocked", "ignored", "error")):
            await self._route(merged)
        return row

    async def _route(self, signal: dict, force_review: str = "") -> None:
        """Send a signal to the trader (or flag it for manual review) and record what happened. `force_review`:
        don't ask the trader - send it to manual review with this reason."""
        db = self.ctx.db
        trader = self.ctx.service("trader")
        if signal["direction"] == "neutral":
            action, reason, traded, order_id = "ignored", "Neutral signal - no trade.", False, None
        elif force_review:
            action, reason, traded, order_id = "review", force_review, False, None
        elif trader is None:
            action, reason, traded, order_id = "ignored", "Trading engine not running.", False, None
        else:
            res = await trader.handle_signal(signal)
            action, reason, traded, order_id = res["action"], res["reason"], res["traded"], res.get("order_db_id")
        db.update("signals", signal["id"], {"action": action, "action_reason": reason, "traded": int(traded),
                                            "order_id": order_id,
                                            "review_status": "pending" if action == "review" else None})
        event = db.query_one("SELECT * FROM signals WHERE id = ?", (signal["id"],))
        order = db.query_one("SELECT submitted_at FROM orders WHERE id = ?", (order_id,)) if order_id else None
        event["speed"] = signal_speed(event, order["submitted_at"] if order else None)
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
        if sig.get("action") in ("watch", "merged") and sig.get("engine") == "pro":
            raise PermissionError("Pro AI is only watching this one - its watch-only signals are never traded.")
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

    # ------------------------------------------------------------------ Pro AI
    def _pro_job(self, item: NewsItem, pre: PrefilterResult, main_engine: str | None, main_analysis_id: int | None,
                 main_signals) -> ProJob | None:
        """A job for Pro AI when it is running and the story is one for it (llm/routing.py), else None."""
        pro = self.ctx.service("pro_ai")
        if pro is None or not pro.ready or not item.db_id:
            return None
        s = pro.settings
        reason, hard = routing.route(item, pre, main_engine, main_signals, s.scope)
        if not reason:
            return None
        return ProJob(item=item, pre=pre, reason=reason, judge=hard and s.mode == "judge", main_engine=main_engine,
                      main_analysis_id=main_analysis_id,
                      main_calls=None if main_engine is None else routing.calls_of(main_signals))

    def _offer_to_pro(self, item: NewsItem, pre: PrefilterResult, main_engine: str | None,
                      main_analysis_id: int | None, main_signals) -> None:
        job = self._pro_job(item, pre, main_engine, main_analysis_id, main_signals)
        if job is not None:
            self.ctx.service("pro_ai").offer(job)

    @staticmethod
    def _judge_signal(sig, verdict: dict | None) -> tuple[str, str | None]:
        """(why the signal goes to manual review instead of the trader, Pro AI's call on the stock) from Pro AI's
        answer in judge mode ("", None without one)."""
        if verdict is None:
            return "", None
        pro = verdict.get(sig.ticker)
        pro_dir = pro.direction if pro is not None else "neutral"
        if sig.direction == "neutral" or pro_dir == sig.direction:
            return "", pro_dir
        conf = f" ({pro.confidence})" if pro is not None else ""
        why = f" Pro AI's reasoning: {pro.reasoning}" if pro is not None and pro.reasoning else ""
        return (f"Pro AI disagreed - it reads this as {pro_dir}{conf} for {sig.ticker}, so it wasn't traded "
                f"automatically. Approve it to trade anyway.{why}")[:600], pro_dir

    async def record_pro(self, job: ProJob, result) -> dict | None:
        """Store Pro AI's reading of a story (engine "pro", free). Its signals are watch-only: price-tracked for the
        scoreboard, never traded, alerted or merged into tradeable signals. In judge mode a stock the main engine
        missed becomes a manual-review signal. Returns {ticker: signal}, or None when Pro AI couldn't answer."""
        ai = self.ctx.config.settings.ai
        item = job.item
        decided_at = utcnow()
        validation = validate_response(result.text, self.tickers, ai.max_signals_per_item) if result.ok else None
        if not result.ok:
            status, error = result.status, result.error
        else:
            status, error = validation.status, validation.summary or None
        calls = {sig.ticker: sig for sig in validation.signals} if status == "ok" else None
        agreement = None
        if calls is not None:
            agreement = routing.compare(job.main_calls, {t: c.direction for t, c in calls.items()})
        analysis_id = self.ctx.db.insert("analyses", {
            "created_at": iso(), "trading_day": trading_day(),
            "item_kind": "transcript" if item.kind == "transcript" else "news", "item_id": item.db_id,
            "source_id": item.source_id, "source_name": item.source_name, "model": result.model,
            "input_tokens": result.input_tokens, "output_tokens": result.output_tokens, "cache_read_tokens": 0,
            "cache_write_tokens": 0, "cost_usd": 0.0, "latency_ms": result.latency_ms, "status": status,
            "error": error, "raw_response": (result.text or "")[:20000], "is_backtest": 0, "engine": "pro",
            "main_analysis_id": job.main_analysis_id, "pro_reason": job.reason, "agreement": agreement,
        })
        self.ctx.bus.publish("pro_analysis", {"id": analysis_id, "item_id": item.db_id, "status": status,
                                              "reason": job.reason, "agreement": agreement, "error": error})
        if calls is None:
            log.info("Pro AI's answer for '%s' wasn't used (%s): %s", item.title[:60], status, error)
            return None
        for sig in calls.values():
            if sig.direction != "neutral":
                await self._pro_signal(sig, job, analysis_id, decided_at)
        if job.main_analysis_id and job.main_calls:
            changed = 0
            for ticker in job.main_calls:
                pro = calls.get(ticker)
                changed += self.ctx.db.execute(
                    "UPDATE signals SET pro_direction = ? WHERE analysis_id = ? AND ticker = ? AND pro_direction IS NULL",
                    (pro.direction if pro is not None else "neutral", job.main_analysis_id, ticker))
            if changed:
                self.ctx.bus.publish("signal_updated", {"analysis_id": job.main_analysis_id})
        return calls

    async def _pro_signal(self, sig, job: ProJob, analysis_id: int, decided_at: datetime) -> dict:
        s = self.ctx.config.settings
        db = self.ctx.db
        now = utcnow()
        row = self._signal_row(sig, job.item, analysis_id, "pro", now, decided_at)
        main_dir = None if job.main_calls is None else job.main_calls.get(sig.ticker, "neutral")
        row.update(main_direction=main_dir, traded=0)
        dedupe = s.ai.signal_dedupe_minutes > 0
        since = iso(now - timedelta(minutes=s.ai.signal_dedupe_minutes))
        if (job.judge and main_dir in (None, "neutral") and sig.confidence >= s.trading.review_threshold
                and not (dedupe and self._twin(sig, since, watch=False))):
            row.update(action="review", review_status="pending", action_reason=(
                f"Pro AI spotted this and the main engine didn't ({job.reason}). Pro AI never trades by itself - "
                "approve it to trade."))
            row["id"] = db.insert("signals", row)
            event = db.query_one("SELECT * FROM signals WHERE id = ?", (row["id"],))
            event["speed"] = signal_speed(event, None)
            self.ctx.bus.publish("signal", event)
            log.info("Pro AI signal #%s %s %s conf %s from %s -> manual review", row["id"], sig.ticker, sig.direction,
                     sig.confidence, row["source_name"])
            await self._review_alert(event)
            return event
        twin = self._twin(sig, since, watch=True) if dedupe else None
        if twin is not None:
            row.update(action="merged", merged_into=twin["id"],
                       action_reason=f"Same Pro AI call as #{twin['id']} (counted once)")
            row["id"] = db.insert("signals", row)
            seen = json.loads(twin["sources_seen"] or "[]")
            if row["source_name"] not in seen:
                seen.append(row["source_name"])
            db.execute("UPDATE signals SET corroborations = corroborations + 1, sources_seen = ?, "
                       "confidence = MAX(confidence, ?) WHERE id = ?", (json.dumps(seen), sig.confidence, twin["id"]))
            return row
        row.update(action="watch", action_reason=_watch_reason(main_dir, sig.direction, job.reason))
        row["id"] = db.insert("signals", row)
        event = db.query_one("SELECT * FROM signals WHERE id = ?", (row["id"],))
        event["speed"] = signal_speed(event, None)
        self.ctx.bus.publish("signal", event)
        return event

    def _twin(self, sig, since: str, watch: bool) -> dict | None:
        """The newest signal on the same stock and direction since `since`: Pro AI's watch-only ones (watch=True) or
        every other kind (watch=False)."""
        kind = "engine = 'pro' AND action = 'watch'" if watch else "COALESCE(action, '') != 'watch'"
        return self.ctx.db.query_one(
            f"SELECT * FROM signals WHERE ticker = ? AND direction = ? AND merged_into IS NULL AND created_at >= ? "
            f"AND {kind} ORDER BY id DESC LIMIT 1", (sig.ticker, sig.direction, since))


def _watch_reason(main_dir: str | None, direction: str, reason: str) -> str:
    head = "Pro AI is only watching - this is never traded or alerted."
    if main_dir is None:
        return f"{head} The main engine didn't read this story ({reason})."
    if main_dir == direction:
        return f"{head} The main engine made the same call."
    if main_dir == "neutral":
        return f"{head} The main engine saw nothing to trade here."
    return f"{head} The main engine called it {main_dir}."
