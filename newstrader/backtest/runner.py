"""Backtest: run the AI engine over historical Alpaca/Benzinga news for a date range and show what it would
have done, using the same pre-filter, Claude prompt, validator and your current trading/risk settings.

Caveat shown in the UI: Claude may already "know" what happened after older news (it was trained on data up
to a cutoff date), which makes backtests on older dates look better than live trading would be.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta

from ..ai.claude_client import ClaudeAnalyzer
from ..ai.costs import estimate_cost
from ..ai.prefilter import prefilter
from ..ai.validator import validate_response
from ..config import SourceConfig
from ..context import AppContext
from ..db import iso, parse_iso
from ..performance.prices import directional_return, price_at
from ..performance.stats import compute_stats
from ..sources.alpaca_news import news_to_item
from ..state import trading_day
from .simulator import simulate_bracket

log = logging.getLogger(__name__)

BT_SOURCE = SourceConfig(id="backtest", type="alpaca_news", name="Benzinga (historical)")
# Rough per-article token use for the cost estimate (prompt + candidates + thinking + JSON answer)
EST_INPUT, EST_CACHED, EST_OUTPUT = 900, 1100, 900


@dataclass
class BacktestParams:
    start: date
    end: date
    symbols: list[str] = field(default_factory=list)
    max_articles: int = 100
    model: str = ""
    budget_usd: float = 5.0
    hold: str = "eod"  # "eod" = close at the end of the trading day, or a number of hours as a string


def estimate(params: BacktestParams, default_model: str) -> dict:
    model = params.model or default_model
    per = estimate_cost(model, EST_INPUT, EST_OUTPUT, 0, EST_CACHED)
    return {"model": model, "per_article": round(per, 4), "max_cost": round(per * params.max_articles, 2),
            "articles": params.max_articles}


def _session_for(t: datetime, sessions: list[dict]) -> tuple[dict | None, bool]:
    """(session on or after t, whether t is inside that session's hours)."""
    for s in sessions:
        o, c = parse_iso(s["open"]), parse_iso(s["close"])
        if o is None or c is None:
            continue
        if t < o:
            return s, False
        if o <= t < c:
            return s, True
    return None, False


class BacktestRunner:
    name = "backtest"

    def __init__(self, ctx: AppContext, analyzer: ClaudeAnalyzer | None = None):
        self.ctx = ctx
        self.analyzer = analyzer
        self._task: asyncio.Task | None = None
        self._cancel = False
        self.current_run: int | None = None

    async def start(self) -> None:
        # mark runs interrupted by a restart
        self.ctx.db.execute("UPDATE backtest_runs SET status = 'stopped', message = 'interrupted by restart' "
                            "WHERE status = 'running'")

    async def stop(self) -> None:
        self._cancel = True
        if self._task:
            self._task.cancel()

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def launch(self, params: BacktestParams) -> int:
        if self.running:
            raise RuntimeError("A backtest is already running.")
        run_id = self.ctx.db.insert("backtest_runs", {"created_at": iso(), "params": json.dumps(asdict(params), default=str),
                                                      "status": "running", "progress": 0, "message": "starting...",
                                                      "cost_usd": 0})
        self._cancel = False
        self.current_run = run_id
        self._task = asyncio.create_task(self._guarded(run_id, params), name=f"backtest-{run_id}")
        return run_id

    def cancel(self) -> None:
        self._cancel = True

    def _progress(self, run_id: int, progress: float, message: str, **extra) -> None:
        self.ctx.db.update("backtest_runs", run_id, {"progress": round(progress, 3), "message": message, **extra})
        self.ctx.bus.publish("backtest", {"id": run_id, "progress": progress, "message": message, **extra})

    async def _guarded(self, run_id: int, params: BacktestParams) -> None:
        try:
            await self.run(run_id, params)
        except asyncio.CancelledError:
            self._progress(run_id, 1, "cancelled", status="cancelled")
        except Exception as exc:
            log.exception("Backtest %s failed", run_id)
            self._progress(run_id, 1, f"failed: {exc}", status="failed")

    # ------------------------------------------------------------------ main loop
    async def run(self, run_id: int, params: BacktestParams) -> dict:
        ctx = self.ctx
        s = ctx.config.settings
        trader = ctx.service("trader")
        broker = getattr(trader, "broker", None)
        pipeline = ctx.service("pipeline")
        if broker is None:
            raise RuntimeError("Not connected to Alpaca - historical news and prices need your Alpaca keys.")
        if pipeline is None or not pipeline.tickers.loaded:
            raise RuntimeError("Ticker list not loaded yet - wait a minute after adding Alpaca keys.")
        analyzer = self.analyzer or pipeline.analyzer
        model = params.model or s.ai.model

        start_dt = datetime.combine(params.start, datetime.min.time(), UTC)
        end_dt = datetime.combine(params.end, datetime.max.time(), UTC)
        self._progress(run_id, 0.01, "downloading historical news...")
        articles = await asyncio.to_thread(broker.news, start_dt, end_dt, params.symbols or None,
                                           max(params.max_articles * 3, 50))
        sessions = await asyncio.to_thread(broker.calendar, params.start - timedelta(days=1),
                                           params.end + timedelta(days=5))
        items = []
        for a in articles:
            item = news_to_item(a, BT_SOURCE)
            item.source_type = "backtest"
            if not item.title or item.published_at is None:
                continue
            pre = prefilter(item.text, pipeline.tickers, item.symbols, s.ai.analyze_keyword_only)
            if pre.hit:
                items.append((item, pre))
            if len(items) >= params.max_articles:
                break
        if not items:
            summary = self._summary([], len(articles), 0, 0.0)
            self._progress(run_id, 1, "no relevant news found in that range", status="done", summary=json.dumps(summary))
            return summary

        cost = 0.0
        results: list[dict] = []
        last_entry: dict[str, datetime] = {}
        batch = max(1, s.ai.max_concurrent_calls)
        analysed = 0
        for i in range(0, len(items), batch):
            if self._cancel:
                self._progress(run_id, i / len(items), "cancelled", status="cancelled")
                break
            if cost >= params.budget_usd:
                self._progress(run_id, i / len(items), f"stopped: budget ${params.budget_usd:.2f} reached")
                break
            chunk = items[i:i + batch]
            outs = await asyncio.gather(*[
                analyzer.analyze(it, pre, _session_for(it.published_at, sessions)[1],
                                 it.published_at + timedelta(minutes=1), model=model)
                for it, pre in chunk])
            for (item, _pre), res in zip(chunk, outs, strict=True):
                analysed += 1
                cost += res.cost_usd
                validation = validate_response(res.text, pipeline.tickers, s.ai.max_signals_per_item) if res.ok else None
                ctx.db.insert("analyses", {
                    "created_at": iso(), "trading_day": trading_day(), "item_kind": "backtest", "item_id": run_id,
                    "source_id": "backtest", "source_name": item.source_name, "model": res.model,
                    "input_tokens": res.input_tokens, "output_tokens": res.output_tokens,
                    "cache_read_tokens": res.cache_read_tokens, "cache_write_tokens": res.cache_write_tokens,
                    "cost_usd": res.cost_usd, "latency_ms": res.latency_ms,
                    "status": res.status if not res.ok else validation.status,
                    "error": res.error if not res.ok else (validation.summary or None),
                    "raw_response": res.text[:20000], "is_backtest": 1})
                if validation is None:
                    continue
                for sig in validation.signals:
                    row = await self._simulate_signal(run_id, item, sig, sessions, last_entry, params)
                    results.append(row)
            self._progress(run_id, min(0.99, (i + len(chunk)) / len(items)),
                           f"analysed {analysed}/{len(items)} stories · {len(results)} signals · cost ${cost:.2f}",
                           cost_usd=round(cost, 4))

        summary = self._summary(results, len(articles), analysed, cost)
        status = "cancelled" if self._cancel else "done"
        self._progress(run_id, 1, f"{status}: {summary['trades']} trades, P/L ${summary['total_pnl']:,.2f}",
                       status=status, summary=json.dumps(summary), cost_usd=round(cost, 4))
        return summary

    async def _simulate_signal(self, run_id: int, item, sig, sessions: list[dict], last_entry: dict,
                               params: BacktestParams) -> dict:
        s = self.ctx.config.settings
        broker = self.ctx.service("trader").broker
        t0: datetime = item.published_at
        row = {"run_id": run_id, "news_time": iso(t0), "headline": item.title[:500], "source": item.source_name,
               "url": item.url, "ticker": sig.ticker, "direction": sig.direction, "confidence": sig.confidence,
               "reasoning": sig.reasoning, "time_sensitivity": sig.time_sensitivity, "action": "", "qty": 0,
               "pnl": None, "ret_pct": None, "ret_1h": None}

        session, in_hours = _session_for(t0, sessions)
        try:
            horizon = t0 + timedelta(hours=1, minutes=5)
            exit_cap = parse_iso(session["close"]) if session else horizon
            fetch_end = max(horizon, exit_cap or horizon) + timedelta(minutes=2)
            bars = await asyncio.to_thread(broker.bars, sig.ticker, t0 - timedelta(hours=3), fetch_end)
        except Exception as exc:
            row["action"] = f"no price data ({exc})"
            self.ctx.db.insert("backtest_results", row)
            return row
        if sig.direction in ("bullish", "bearish"):
            row["ret_1h"] = directional_return(sig.direction, price_at(bars, t0), price_at(bars, t0 + timedelta(hours=1)))

        side = None
        if sig.direction == "neutral":
            row["action"] = "neutral - no trade"
        elif sig.confidence < s.trading.buy_threshold:
            row["action"] = ("would ask for manual review" if sig.confidence >= s.trading.review_threshold
                             else "confidence too low")
        elif sig.direction == "bullish":
            side = "long"
        elif s.trading.allow_shorting:
            side = "short"
        else:
            row["action"] = "bearish - shorting off (only sells if already held)"

        if side:
            last = last_entry.get(sig.ticker)
            if sig.ticker in s.risk.blacklist or (s.risk.whitelist and sig.ticker not in s.risk.whitelist):
                row["action"], side = "blocked by black/whitelist", None
            elif last and (t0 - last).total_seconds() < s.risk.ticker_cooldown_minutes * 60:
                row["action"], side = "blocked by cooldown", None
            elif s.risk.market_hours_only and not in_hours:
                row["action"], side = "market closed (market-hours-only)", None
            elif session is None:
                row["action"], side = "no trading session found", None

        if side:
            entry_after = t0 + timedelta(minutes=1) if in_hours else parse_iso(session["open"])
            exit_by = hold_until(params.hold, entry_after, parse_iso(session["close"]))
            sim = simulate_bracket(side, bars, entry_after, s.trading.stop_loss_pct, s.trading.take_profit_pct, exit_by)
            if not sim.entered:
                row["action"] = sim.exit_reason
            else:
                qty = math.floor(s.risk.max_dollars_per_trade / sim.entry_price) if sim.entry_price else 0
                if qty < 1:
                    row["action"] = "too expensive for max $ per trade"
                else:
                    last_entry[sig.ticker] = t0
                    row.update({"action": "bought" if side == "long" else "shorted", "qty": qty,
                                "entry_time": sim.entry_time, "entry_price": sim.entry_price,
                                "exit_time": sim.exit_time, "exit_price": sim.exit_price,
                                "exit_reason": sim.exit_reason, "ret_pct": sim.ret_pct, "pnl": sim.pnl(qty)})
        row["id"] = self.ctx.db.insert("backtest_results", row)
        return row

    @staticmethod
    def _summary(results: list[dict], fetched: int, analysed: int, cost: float) -> dict:
        trades = [r for r in results if r.get("pnl") is not None]
        wins = [r for r in trades if r["pnl"] > 0]
        total = round(sum(r["pnl"] for r in trades), 2)
        curve, cum, peak, max_dd = [], 0.0, 0.0, 0.0
        for r in sorted(trades, key=lambda x: x.get("exit_time") or ""):
            cum += r["pnl"]
            peak = max(peak, cum)
            max_dd = max(max_dd, peak - cum)
            curve.append({"t": r.get("exit_time"), "pnl": round(cum, 2)})
        acc = compute_stats([{**r, "source_name": r.get("source"), "source_type": "backtest"} for r in results], "1h")
        return {
            "articles_fetched": fetched, "analysed": analysed, "signals": len(results), "trades": len(trades),
            "wins": len(wins), "losses": len(trades) - len(wins),
            "win_rate": round(100 * len(wins) / len(trades), 1) if trades else None,
            "total_pnl": total, "avg_ret_pct": round(sum(r["ret_pct"] for r in trades) / len(trades), 3) if trades else None,
            "best": max((r["pnl"] for r in trades), default=None), "worst": min((r["pnl"] for r in trades), default=None),
            "max_drawdown": round(max_dd, 2), "cost_usd": round(cost, 4), "equity_curve": curve,
            "accuracy_1h": {"overall": acc["overall"], "by_confidence": acc["by_confidence"]},
        }


def hold_until(hold: str, entry_after: datetime, session_close: datetime) -> datetime:
    """'eod' -> that day's close; '2' -> 2 hours after entry, but never past the close."""
    if hold and hold != "eod":
        try:
            return min(entry_after + timedelta(hours=float(hold)), session_close)
        except ValueError:
            pass
    return session_close
