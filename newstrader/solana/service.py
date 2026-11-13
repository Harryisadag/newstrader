"""The Solana paper desk service (PAPER ONLY), 24/7 while the app runs and Settings -> Solana (paper) is on:

- the feed: about once a minute it reads GeckoTerminal's newest pools and looks again at young pools on the watchlist;
  a pool the entry trigger picks becomes a candidate (each pool once, each coin at most once a day, at most 20
  waiting - one that waited more than 10 minutes is skipped)
- the checklist worker: one candidate at a time through the 7 gates; a coin that passes them all gets a paper buy
- the exits: every 15 s a Jupiter quote to sell each open paper position (stop-loss, trailing stop, time stop)
- the reject audit's follow-ups, 1 hour and 24 hours after each verdict

The app's kill switch stops new paper buys (exits keep simulating), and so does the daily paper loss limit. Turned
off, the desk makes no network calls at all.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections import deque
from datetime import UTC, datetime, timedelta

from ..context import AppContext
from ..db import iso, parse_iso, utcnow
from . import audit
from .checklist import run_checklist
from .data import MULTI_MAX, SOURCES, SolanaData
from .desk import RESET_PHRASE, PaperDesk, exit_decision
from .feed import Watchlist, trigger
from .gates import EXPLAIN, GATES
from .parsers import LAMPORTS, SOL_MINT, DataError, NoRoute, Pool, RateLimited, rpc_host

log = logging.getLogger(__name__)

QUEUE_MAX = 20
QUEUE_WAIT = timedelta(minutes=10)  # a candidate that waited longer is too old to act on
COIN_ONCE_PER = timedelta(hours=24)
FEED_RESERVE = 6  # GeckoTerminal calls a minute the watchlist leaves free for the checklist and the follow-ups
WATCH_CALLS = 6  # at most this many watchlist calls per feed check
EXIT_EVERY = 15.0
EXIT_WAIT = 5.0  # an exit check waits at most this long for Jupiter - otherwise it is skipped
CANT_SELL_AFTER = 3  # no-route answers in a row before a position is marked "can't sell"
ESTIMATE_FACTOR = 0.95  # a value from the price feed (no sell route) is marked down 5%
AUDIT_EVERY = 60.0
LATE_1H = timedelta(hours=3)  # a 1-hour follow-up this late is skipped (the 24-hour one still happens)


class SolanaDesk:
    name = "solana"

    def __init__(self, ctx: AppContext, data: SolanaData | None = None, now=utcnow, mono=time.monotonic):
        self.ctx = ctx
        self.data = data or SolanaData()
        self._now = now
        self._mono = mono
        self.desk = PaperDesk(ctx.db, lambda: ctx.config.settings.solana, now)
        self.watch = Watchlist()
        self.queue: deque[int] = deque()
        self.dropped = 0
        self._tasks: list[asyncio.Task] = []
        self._wake = {k: asyncio.Event() for k in ("feed", "worker", "exits", "audit")}
        self._feed_lock = asyncio.Lock()
        self._exit_lock = asyncio.Lock()
        self.checking: dict | None = None
        self.last_feed: dict = {}
        self.last_exit_at: str | None = None
        self.last_error: str | None = None
        self.sol_usd: float | None = None

    @property
    def s(self):
        return self.ctx.config.settings.solana

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        self.data.rpc_url = self.s.rpc_url

        def on_settings(settings) -> None:  # called from whichever thread saved the settings
            self.data.rpc_url = settings.solana.rpc_url
            if not loop.is_closed():
                loop.call_soon_threadsafe(self.kick)

        self.ctx.config.on_change(on_settings)
        loops = (("feed", lambda: float(self.s.poll_seconds), self._feed_step),
                 ("worker", lambda: 5.0, self.work_once),
                 ("exits", lambda: EXIT_EVERY, self.check_exits),
                 ("audit", lambda: AUDIT_EVERY, self.follow_up))
        for name, every, step in loops:
            self._tasks.append(asyncio.create_task(self._loop(name, every, step), name=f"solana-{name}"))
        self._status()

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._tasks.clear()
        with contextlib.suppress(Exception):
            await self.data.close()

    def kick(self) -> None:
        for event in self._wake.values():
            event.set()

    async def _loop(self, name: str, every, step) -> None:
        while True:
            try:
                if self.s.enabled:
                    await step()
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"[:300]
                log.warning("Solana paper desk (%s): %s", name, exc)
                log.debug("solana desk traceback", exc_info=True)
            self._status()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._wake[name].wait(), every())
            self._wake[name].clear()

    async def _feed_step(self) -> None:
        if not self.desk.paused:
            await self.feed_tick()

    # ------------------------------------------------------------------ helpers
    def _publish(self, what: str, **extra) -> None:
        self.ctx.bus.publish("solana", {"what": what, **extra})

    async def _alert(self, kind: str, title: str, message: str, level: str = "trade") -> None:
        alerts = self.ctx.service("alerts")
        if alerts is not None:
            try:
                await alerts.send(kind, title, message, level)
                return
            except Exception:
                log.debug("alert failed", exc_info=True)
        self.ctx.bus.publish("toast", {"kind": "error" if level == "error" else level, "title": title,
                                       "message": message})

    def _note_sol_usd(self, pool: Pool) -> None:
        if pool.sol_usd:
            self.sol_usd = pool.sol_usd

    def _set(self, cand_id: int, **values) -> None:
        self.ctx.db.update("sol_candidates", cand_id, values)

    # ------------------------------------------------------------------ the feed
    async def feed_tick(self) -> dict:
        """Read the newest pools, look again at watched ones that are due, and queue the ones the trigger picks."""
        async with self._feed_lock:
            s = self.s
            now = self._now()
            stats = {"at": iso(now), "pools": 0, "new": 0, "looked_at": 0, "triggered": 0, "error": None}
            try:
                pools = await self.data.new_pools()
            except DataError as exc:
                stats["error"] = str(exc)
                pools = []
            stats["pools"] = len(pools)
            for p in pools:
                self._note_sol_usd(p)
                if self.watch.known(p.address):
                    continue
                fired, why = trigger(p, s, now)
                if fired:
                    self.watch.finish(p.address, now)
                    stats["triggered"] += int(self._enqueue(p, why, now) is not None)
                elif self.watch.add(p, s, now):
                    stats["new"] += 1
            calls = 0
            while calls < WATCH_CALLS and self.data.gecko.free() > FEED_RESERVE and not stats["error"]:
                batch = self.watch.due(now, MULTI_MAX)
                if not batch:
                    break
                calls += 1
                try:
                    got = await self.data.pools(batch, max_wait=5)
                except DataError as exc:
                    stats["error"] = str(exc)
                    break
                for address in batch:
                    p = got.get(address)
                    stats["looked_at"] += 1
                    if p is not None:
                        self._note_sol_usd(p)
                        fired, why = trigger(p, s, now)
                        if fired:
                            self.watch.finish(address, now)
                            stats["triggered"] += int(self._enqueue(p, why, now) is not None)
                            continue
                    self.watch.checked(address, p, s, now)
            stats["watching"] = len(self.watch)
            self.last_feed = stats
            return stats

    def _enqueue(self, p: Pool, why: str, now) -> int | None:
        db = self.ctx.db
        if db.query_one("SELECT id FROM sol_candidates WHERE pool = ?", (p.address,)):
            return None  # each pool once
        if db.query_one("SELECT id FROM sol_candidates WHERE mint = ? AND created_at >= ?",
                        (p.mint, iso(now - COIN_ONCE_PER))):
            return None  # each coin at most once a day
        if len(self.queue) >= QUEUE_MAX:
            self.dropped += 1
            return None
        cid = db.insert("sol_candidates", {
            "created_at": iso(now), "wallet": self.desk.wallet()["id"], "pool": p.address, "mint": p.mint,
            "symbol": p.symbol or p.name.split(" / ")[0][:20], "name": p.name[:80], "dex": p.dex,
            "pool_created_at": iso(p.created_at) if p.created_at else None, "signal_price_usd": p.price_usd,
            "signal_price_sol": p.price_sol, "signal_liquidity_usd": p.reserve_usd, "signal_fdv_usd": p.fdv_usd,
            "trigger": json.dumps({"why": why, "m5": p.window("m5"), "change_m5": p.change.get("m5"),
                                   "age_min": round(p.age_minutes(now) or 0, 1)}),
            "status": "queued", "attempts": 0})
        self.queue.append(cid)
        self._wake["worker"].set()
        self._publish("candidate", id=cid)
        log.info("Solana paper desk: %s picked by the feed (%s)", p.name, why)
        return cid

    async def check_feed_now(self) -> dict:
        if not self.s.enabled:
            raise RuntimeError("The Solana paper desk is off - turn it on in Settings -> Solana (paper)")
        if self.desk.paused:
            raise RuntimeError("The desk is paused - resume it first")
        stats = await self.feed_tick()
        self._status()
        return stats

    # ------------------------------------------------------------------ the checklist
    async def work_once(self) -> dict | None:
        """Take the oldest waiting candidate and run it through the checklist."""
        while self.queue:
            cid = self.queue.popleft()
            cand = self.ctx.db.query_one("SELECT * FROM sol_candidates WHERE id = ?", (cid,))
            if cand is None or cand["status"] != "queued":
                continue
            waited = self._now() - parse_iso(cand["created_at"])
            if waited > QUEUE_WAIT:
                self._set(cid, status="expired", finished_at=iso(self._now()),
                          reason=f"Waited {waited.total_seconds() / 60:.0f} minutes in the queue - too old to act on")
                continue
            return await self.check_candidate(cand)
        return None

    async def check_candidate(self, cand: dict) -> dict:
        db = self.ctx.db
        kill = self.ctx.state.kill_engaged
        why = self.desk.blocked(kill, cand["mint"])
        if why:
            self._set(cand["id"], status="skipped", reason=why, finished_at=iso(self._now()))
            self._publish("candidate", id=cand["id"])
            return db.query_one("SELECT * FROM sol_candidates WHERE id = ?", (cand["id"],))
        self._set(cand["id"], status="checking", started_at=iso(self._now()))
        self.checking = cand
        self._status()
        self._publish("candidate", id=cand["id"])
        try:
            outcome = await run_checklist(
                self.data, cand, self.s, lambda size: self.desk.blocked(self.ctx.state.kill_engaged, cand["mint"], size),
                now=self._now, mono=self._mono)
        except Exception as exc:  # the checklist itself broke: it's a reject, never a buy
            log.warning("Solana checklist for %s failed: %s", cand["symbol"], exc)
            self._set(cand["id"], status="rejected", reason=f"The checklist failed ({type(exc).__name__}: {exc})"[:300],
                      finished_at=iso(self._now()))
            return db.query_one("SELECT * FROM sol_candidates WHERE id = ?", (cand["id"],))
        finally:
            self.checking = None
        for att in outcome.attempts:
            for step, res in enumerate(att.results, 1):
                db.insert("sol_gate_results", {
                    "candidate_id": cand["id"], "attempt": att.number, "step": step, "gate": res.gate,
                    "name": res.name, "passed": int(res.passed), "reason": res.reason[:500],
                    "checked_at": iso(res.checked_at), "data": json.dumps(res.data, default=str)[:20000]})
        last = outcome.last
        values: dict = {"attempts": len(outcome.attempts), "finished_at": iso(self._now()),
                        "size_factor": outcome.size_factor}
        if outcome.passed:
            why = self.desk.blocked(self.ctx.state.kill_engaged, cand["mint"], last.order_sol)
            if why:
                values.update(status="passed", reason=f"Passed all 7 gates, but no paper buy: {why}")
            else:
                ev = last.evidence
                pos = self.desk.buy(cand, ev.final.value, ev.mint_info.value.decimals, last.order_sol)
                half = " (half size: the creator is unknown)" if outcome.size_factor < 1 else ""
                values.update(status="bought", position_id=pos["id"],
                              reason=f"Passed all 7 gates - paper buy of {last.order_sol:g} SOL{half}")
                log.info("Solana paper desk: paper buy of %s SOL of %s", last.order_sol, cand["symbol"])
                await self._alert("trade_placed", f"Paper buy (Solana): {cand['symbol']}",
                                  f"{last.order_sol:g} SOL of {cand['symbol']} - it passed all 7 gates{half}. PAPER ONLY: "
                                  "no real trade, and real fills would be worse.")
                self._publish("position", id=pos["id"])
        elif last.blocked:
            values.update(status="skipped", reason=last.blocked)
        else:
            k = outcome.killer
            values.update(status="rejected", killer_gate=k.gate, reason=f"Gate {k.gate} ({k.name}): {k.reason}"[:500])
        self._set(cand["id"], **values)
        self._publish("candidate", id=cand["id"])
        return db.query_one("SELECT * FROM sol_candidates WHERE id = ?", (cand["id"],))

    # ------------------------------------------------------------------ exits
    async def check_exits(self) -> dict:
        """One look at every open paper position: a Jupiter quote to sell it all, then the exit rules."""
        async with self._exit_lock:
            s = self.s
            done: dict[int, str] = {}
            estimate = []
            for pos in self.desk.open_positions():
                try:
                    q = await self.data.quote(pos["mint"], SOL_MINT, int(pos["tokens"]), max_wait=EXIT_WAIT)
                except RateLimited:
                    done[pos["id"]] = "skipped (Jupiter busy)"
                    continue
                except NoRoute:
                    n = (pos["no_route_count"] or 0) + 1
                    stuck = n >= CANT_SELL_AFTER
                    fresh = self.desk.mark(pos, None, "estimate", no_route_count=n, cant_sell=stuck)
                    estimate.append(fresh)
                    done[pos["id"]] = "no route"
                    if stuck and not pos["cant_sell"]:
                        log.warning("Solana paper desk: can't sell %s (no route %d times in a row)", pos["symbol"], n)
                        await self._alert("error", f"Solana paper desk: can't sell {pos['symbol']}",
                                          f"Jupiter found no way to sell {pos['symbol']} {n} checks in a row. With real "
                                          "money you'd be stuck holding it - the classic rug. Its value is only an "
                                          "estimate until a real sell quote comes back.", "error")
                    continue
                except DataError as exc:
                    done[pos["id"]] = f"error: {exc}"
                    continue
                value = q.out_amount / LAMPORTS
                pos = self.desk.mark(pos, value, "quote", no_route_count=0, cant_sell=False)
                decision = exit_decision(pos, value, self._now(), s)
                if decision is None:
                    done[pos["id"]] = "watching"
                    continue
                kind, reason = decision
                sold = self.desk.sell(pos, q, kind, reason)
                done[pos["id"]] = "sold"
                log.info("Solana paper desk: paper sale of %s (%s), P/L %+.4f SOL", pos["symbol"], kind, sold["pnl_sol"])
                await self._alert("trade_filled", f"Paper sale (Solana): {pos['symbol']} {sold['pnl_sol']:+.4f} SOL",
                                  f"{reason}. PAPER ONLY - after the haircut and fees.")
                self._publish("position", id=pos["id"])
            if estimate:
                await self._estimate(estimate)
            halted = self.desk.check_daily_loss()
            if halted:
                log.warning("Solana paper desk: daily loss limit hit - %s", halted)
                await self._alert("daily_loss_limit", "Solana paper desk: daily loss limit hit", halted, "error")
            self.last_exit_at = iso(self._now())
            if done:
                self._publish("exits")
            return done

    async def _estimate(self, positions: list[dict]) -> None:
        """No sell route: value the coins from the price feed, marked down 5% and shown as an estimate."""
        try:
            prices = await self.data.prices([p["mint"] for p in positions] + [SOL_MINT], max_wait=EXIT_WAIT)
        except DataError:
            return
        sol = (prices.get(SOL_MINT) or {}).get("usd")
        if sol:
            self.sol_usd = sol
        for pos in positions:
            usd = (prices.get(pos["mint"]) or {}).get("usd")
            if usd and sol and pos["decimals"] is not None:
                value = usd * int(pos["tokens"]) / 10 ** pos["decimals"] / sol * ESTIMATE_FACTOR
                self.desk.mark(pos, value, "estimate")

    # ------------------------------------------------------------------ the reject audit
    async def follow_up(self) -> int:
        """The follow-ups that are due: the pools again (30 per GeckoTerminal call), and RugCheck for coins that
        passed. Returns how many were recorded."""
        db = self.ctx.db
        now = self._now()
        items = audit.due(db, now)
        if not items:
            return 0
        n = 0
        ask = []
        for cand, label in items:
            if label == "1h" and now - parse_iso(cand["created_at"]) - audit.FOLLOW_UPS["1h"] > LATE_1H:
                audit.record(db, cand, label, None, now)  # missed (the app was closed): the 24-hour one still comes
                n += 1
            else:
                ask.append((cand, label))
        pools = list(dict.fromkeys(c["pool"] for c, _ in ask))
        got: dict[str, Pool] = {}
        asked: set[str] = set()
        for i in range(0, len(pools), MULTI_MAX):
            if self.data.gecko.free() <= 2:
                break
            batch = pools[i:i + MULTI_MAX]
            try:
                got.update(await self.data.pools(batch, max_wait=10))
            except DataError as exc:
                log.info("Solana follow-ups: %s", exc)
                break
            asked.update(batch)
        for cand, label in ask:
            if cand["pool"] not in asked:
                continue
            rugged = None
            if cand["status"] in ("bought", "passed"):
                with contextlib.suppress(DataError):
                    rugged = (await self.data.rugcheck(cand["mint"], max_wait=10)).get("rugged") is True
            audit.record(db, cand, label, got.get(cand["pool"]), now, rugged)
            n += 1
        if n:
            self._publish("audit")
        return n

    # ------------------------------------------------------------------ controls
    def pause(self) -> None:
        self.desk.set_paused(True)
        self._status()

    def resume(self) -> None:
        self.desk.set_paused(False)
        self.kick()
        self._status()

    def reset_wallet(self, phrase: str) -> dict:
        if str(phrase or "").strip() != RESET_PHRASE:
            raise PermissionError(f"Type {RESET_PHRASE} to confirm - the paper wallet was not reset.")
        w = self.desk.reset()
        log.warning("Solana paper wallet reset to %s SOL (wallet %d)", w["sol"], w["id"])
        self._publish("wallet")
        self._status()
        return w

    # ------------------------------------------------------------------ for the screen
    def _state(self) -> tuple[str, str, str]:
        """(state, level, plain line) for the header light, Logs -> System status and the tab."""
        s = self.s
        if not s.enabled:
            return "off", "off", "Solana paper desk is off (Settings -> Solana (paper))"
        w = self.desk.summary(self.sol_usd)
        held = f"{w['open']} open paper position{'' if w['open'] == 1 else 's'}"
        if w["cant_sell"]:
            return "problem", "error", f"Solana paper desk: can't sell {w['cant_sell']} position(s) - {held}"
        if self.ctx.state.kill_engaged:
            return "killed", "warn", f"Solana paper desk: kill switch on - no new paper buys, {held} still watched"
        if self.desk.paused:
            return "paused", "warn", f"Solana paper desk: paused - {held} still watched"
        if w["halted_today"]:
            return "halted", "warn", f"Solana paper desk: daily loss limit hit - no new paper buys today, {held}"
        err = self.last_error or (self.last_feed or {}).get("error")
        if err:
            return "problem", "warn", f"Solana paper desk: {err}"
        checking = f" · checking {self.checking['symbol']}" if self.checking else ""
        return "watching", "ok", f"Solana paper desk: watching new pools · {held}{checking} (PAPER ONLY)"

    def _status(self) -> None:
        state, level, text = self._state()
        self.ctx.state.set_status("solana", level, text)

    def summary(self) -> dict:
        state, level, text = self._state()
        w = self.desk.summary(self.sol_usd)
        return {"enabled": self.s.enabled, "state": state, "level": level, "text": text, "paused": w["paused"],
                "killed": self.ctx.state.kill_engaged, "halted": w["halted_today"], "sol": w["sol"],
                "total_sol": w["total_sol"], "usd": w["usd"], "open": w["open"], "cant_sell": w["cant_sell"],
                "today_pnl_sol": w["today_pnl_sol"], "queue": len(self.queue),
                "checking": self.checking["symbol"] if self.checking else None,
                "last_feed_at": (self.last_feed or {}).get("at"), "watching": len(self.watch)}

    def feed_rows(self, limit: int = 60) -> list[dict]:
        """The latest candidates with their 7 gate lights (from their last attempt)."""
        db = self.ctx.db
        rows = db.query("SELECT * FROM sol_candidates ORDER BY id DESC LIMIT ?", (limit,))
        if not rows:
            return []
        ids = [r["id"] for r in rows]
        marks = ", ".join("?" for _ in ids)
        lights: dict[int, dict] = {}
        for g in db.query(f"SELECT candidate_id, attempt, gate, passed, reason FROM sol_gate_results WHERE "
                          f"candidate_id IN ({marks}) ORDER BY attempt", ids):
            per = lights.setdefault(g["candidate_id"], {})
            if per.get("_attempt") != g["attempt"]:
                per.clear()
                per["_attempt"] = g["attempt"]
            per[g["gate"]] = g
        out = []
        for r in rows:
            got = lights.get(r["id"], {})
            r["lights"] = [{"gate": n, "name": GATES[n], "state": "not_run" if n not in got else
                            ("pass" if got[n]["passed"] else "fail"), "reason": got[n]["reason"] if n in got else ""}
                           for n in GATES]
            r["killer_name"] = GATES.get(r["killer_gate"]) if r["killer_gate"] else None
            try:
                r["trigger"] = json.loads(r["trigger"] or "{}")
            except ValueError:
                r["trigger"] = {}
            out.append(r)
        return out

    def candidate(self, cand_id: int) -> dict | None:
        db = self.ctx.db
        cand = db.query_one("SELECT * FROM sol_candidates WHERE id = ?", (cand_id,))
        if cand is None:
            return None
        results = db.query("SELECT * FROM sol_gate_results WHERE candidate_id = ? ORDER BY attempt, step", (cand_id,))
        for r in results:
            try:
                r["data"] = json.loads(r["data"] or "{}")
            except ValueError:
                r["data"] = {}
        cand["trigger"] = json.loads(cand["trigger"] or "{}")
        cand["gates"] = results
        cand["killer_name"] = GATES.get(cand["killer_gate"]) if cand["killer_gate"] else None
        if cand["position_id"]:
            pos = self.desk.position(cand["position_id"])
            cand["position"] = self.desk.view(pos, self.sol_usd) if pos else None
            cand["fills"] = self.desk.fills(cand["position_id"])
        return cand

    def sources(self) -> list[dict]:
        """How each data source answered lately (for the tab)."""
        def stamp(t):
            return iso(datetime.fromtimestamp(t, UTC)) if t else None

        out = []
        for key, name in SOURCES.items():
            h = getattr(self.data, "health", {}).get(key) or {}
            out.append({"source": key, "name": name, "calls": h.get("calls", 0), "limited": h.get("limited", 0),
                        "ok_at": stamp(h.get("ok_at")), "error": h.get("error"), "error_at": stamp(h.get("error_at"))})
        return out

    def snapshot(self, limit: int = 60) -> dict:
        s = self.s
        opens = [self.desk.view(p, self.sol_usd) for p in self.desk.open_positions()]
        closed = [self.desk.view(p, self.sol_usd) for p in self.desk.closed(100)]
        return {
            "summary": self.summary(), "wallet": self.desk.summary(self.sol_usd), "feed": self.feed_rows(limit),
            "positions": opens, "closed": closed, "audit": audit.summary(self.ctx.db),
            "gates": [{"gate": g, "name": n, "explain": EXPLAIN[g]} for g, n in GATES.items()],
            "last_feed": self.last_feed, "last_exit_at": self.last_exit_at, "dropped": self.dropped,
            "sources": self.sources(),
            "rpc_host": rpc_host(s.rpc_url) or "the free public RPC", "reset_phrase": RESET_PHRASE,
            "settings": {"order_sol": s.order_sol, "starting_sol": s.starting_sol, "fee_sol": s.fee_sol,
                         "haircut_pct": s.haircut_pct, "stop_loss_pct": s.stop_loss_pct,
                         "trail_start_pct": s.trail_start_pct, "trail_pct": s.trail_pct,
                         "time_stop_hours": s.time_stop_hours, "max_open_positions": s.max_open_positions,
                         "daily_loss_limit_sol": s.daily_loss_limit_sol},
        }
