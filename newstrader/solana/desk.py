"""The paper desk: a pretend wallet of SOL, pretend buys and sales, and the exit rules. PAPER ONLY - nothing here
touches a real wallet or the real market; every "fill" is a number written into the database.

- A paper buy fills at the final Jupiter quote minus a haircut (1%), and every paper transaction pays a fee
  (0.0005 SOL). Real trades are worse: other bots get in first, transactions fail and priority fees cost more.
- Exits are checked with a Jupiter quote to sell the WHOLE position: a stop-loss (-8% against the SOL paid), a
  trailing stop (10% below the best value, once it was up 15%) and a time stop (4 hours). A sale fills at that quote
  minus the haircut and the fee. P/L counts every fee and haircut.
- Desk limits stop new paper buys: the app's kill switch, a daily paper loss limit in SOL, max open positions, one
  position per coin, and the paper SOL left.
- Reset (you type RESET) starts a new wallet: open positions are taken off the books and older trades show as
  "old wallet".
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

from ..db import Database, iso, parse_iso, utcnow
from .parsers import LAMPORTS, Quote

RESET_PHRASE = "RESET"
OPEN = "open"
CLOSED = "closed"
OLD_WALLET = "old_wallet"


def lamports(sol: float) -> int:
    return int(round(sol * LAMPORTS))


def day_of(when: datetime) -> str:
    """The desk's day (UTC): the daily loss limit starts again at midnight UTC."""
    return when.date().isoformat()


def net_value(value_sol: float | None, s) -> float | None:
    """What a sale at this quote would really put in the wallet: minus the haircut and the fee."""
    if value_sol is None:
        return None
    return value_sol * (1 - s.haircut_pct / 100) - s.fee_sol


def exit_levels(pos: dict, s) -> dict:
    """The stop-loss, the trailing stop (once it is on) and the time stop of an open position."""
    sol_in = pos["sol_in"] or 0.0
    peak = pos.get("peak_value_sol") or 0.0
    trailing = bool(sol_in) and peak >= sol_in * (1 + s.trail_start_pct / 100)
    opened = parse_iso(pos["opened_at"])
    return {"stop_value": sol_in * (1 - s.stop_loss_pct / 100),
            "trail_start_value": sol_in * (1 + s.trail_start_pct / 100),
            "trail_value": peak * (1 - s.trail_pct / 100) if trailing else None, "trailing": trailing,
            "time_stop_at": iso(opened + timedelta(hours=s.time_stop_hours)) if opened else None}


def exit_decision(pos: dict, value_sol: float, now: datetime, s) -> tuple[str, str] | None:
    """(kind, plain reason) when a real sell quote says the position should be sold now, else None.
    value_sol = what the sell quote gives (before the haircut and fee), compared with the SOL paid."""
    sol_in = pos["sol_in"]
    if not sol_in:
        return None
    change = (value_sol / sol_in - 1) * 100
    if change <= -s.stop_loss_pct:
        return "stop_loss", f"Stop-loss: a sale would give {value_sol:.4f} SOL, {change:.1f}% against the {sol_in:.4f} SOL paid"
    peak = max(pos.get("peak_value_sol") or 0.0, value_sol)
    if peak >= sol_in * (1 + s.trail_start_pct / 100) and value_sol <= peak * (1 - s.trail_pct / 100):
        return "trailing_stop", (f"Trailing stop: {value_sol:.4f} SOL is {(1 - value_sol / peak) * 100:.0f}% below its "
                                 f"best ({peak:.4f} SOL)")
    opened = parse_iso(pos["opened_at"])
    if opened is not None and now - opened >= timedelta(hours=s.time_stop_hours):
        return "time_stop", f"Time stop: held for {s.time_stop_hours:g} hours ({change:+.1f}%)"
    return None


class PaperDesk:
    """The paper wallet and its positions, kept in the database (tables sol_state, sol_positions, sol_fills)."""

    def __init__(self, db: Database, settings, now=utcnow):
        self.db = db
        self._settings = settings  # () -> SolanaSettings
        self._now = now

    @property
    def s(self):
        return self._settings()

    # ------------------------------------------------------------------ state
    def state(self, key: str, default=None):
        row = self.db.query_one("SELECT value FROM sol_state WHERE key = ?", (key,))
        if row is None or row["value"] is None:
            return default
        return json.loads(row["value"])

    def set_state(self, key: str, value) -> None:
        self.db.execute("INSERT INTO sol_state (key, value, updated_at) VALUES (?, ?, ?) ON CONFLICT(key) DO UPDATE "
                        "SET value = excluded.value, updated_at = excluded.updated_at", (key, json.dumps(value), iso()))

    def wallet(self) -> dict:
        w = self.state("wallet")
        if not w:
            w = {"id": 1, "sol": float(self.s.starting_sol), "starting_sol": float(self.s.starting_sol),
                 "started_at": iso(self._now())}
            self.set_state("wallet", w)
        return w

    def _move(self, sol: float) -> dict:
        w = self.wallet()
        w["sol"] = round(w["sol"] + sol, 9)
        self.set_state("wallet", w)
        return w

    def reset(self) -> dict:
        """A new paper wallet with the starting SOL. Open positions are taken off the books (no sale, no P/L)."""
        old = self.wallet()
        now = iso(self._now())
        self.db.execute("UPDATE sol_positions SET status = ?, closed_at = ?, exit_reason = ? WHERE status = ?",
                        (OLD_WALLET, now, "Taken off the books by a wallet reset (never sold)", OPEN))
        w = {"id": int(old["id"]) + 1, "sol": float(self.s.starting_sol), "starting_sol": float(self.s.starting_sol),
             "started_at": now}
        self.set_state("wallet", w)
        return w  # a daily-loss halt stays until midnight UTC, new wallet or not

    @property
    def paused(self) -> bool:
        return bool(self.state("paused", False))

    def set_paused(self, paused: bool) -> None:
        self.set_state("paused", bool(paused))

    # ------------------------------------------------------------------ positions
    def open_positions(self) -> list[dict]:
        return self.db.query("SELECT * FROM sol_positions WHERE status = ? AND wallet = ? ORDER BY id",
                             (OPEN, self.wallet()["id"]))

    def position(self, pos_id: int) -> dict | None:
        return self.db.query_one("SELECT * FROM sol_positions WHERE id = ?", (pos_id,))

    def closed(self, limit: int = 100) -> list[dict]:
        return self.db.query("SELECT * FROM sol_positions WHERE status != ? ORDER BY closed_at DESC, id DESC LIMIT ?",
                             (OPEN, limit))

    def fills(self, pos_id: int) -> list[dict]:
        return self.db.query("SELECT * FROM sol_fills WHERE position_id = ? ORDER BY id", (pos_id,))

    # ------------------------------------------------------------------ limits
    def today_pnl(self) -> float:
        """Today's paper P/L in SOL (UTC day): positions sold today, plus what the open ones would make if sold now."""
        s = self.s
        wid = self.wallet()["id"]
        today = day_of(self._now())
        closed = self.db.scalar("SELECT COALESCE(SUM(pnl_sol), 0) FROM sol_positions WHERE wallet = ? AND status = ? "
                                "AND substr(closed_at, 1, 10) = ?", (wid, CLOSED, today)) or 0.0
        open_pnl = sum((net_value(p["value_sol"], s) or 0.0) - p["sol_in"] for p in self.open_positions()
                       if p["value_sol"] is not None)
        return round(float(closed) + open_pnl, 9)

    def halted_today(self) -> bool:
        return self.state("halted_day") == day_of(self._now())

    def check_daily_loss(self) -> str | None:
        """Halts new paper buys for the rest of the UTC day when today's paper loss reaches the limit. Returns the
        reason the first time it happens."""
        if self.halted_today():
            return None
        pnl = self.today_pnl()
        limit = self.s.daily_loss_limit_sol
        if pnl > -limit:
            return None
        self.set_state("halted_day", day_of(self._now()))
        reason = (f"Down {-pnl:.4f} SOL on paper today (the limit is {limit:g} SOL). No new paper buys until midnight "
                  "UTC - open positions are still watched.")
        self.set_state("halted_reason", reason)
        return reason

    def blocked(self, kill_engaged: bool, mint: str | None = None, order_sol: float | None = None) -> str:
        """Why a new paper buy isn't allowed right now ("" = it is)."""
        s = self.s
        if kill_engaged:
            return "The kill switch is on - no new paper buys (open ones are still watched)"
        if self.paused:
            return "The desk is paused"
        if self.halted_today():
            return f"The daily paper loss limit ({s.daily_loss_limit_sol:g} SOL) was hit - no new buys until midnight UTC"
        opens = self.open_positions()
        if len(opens) >= s.max_open_positions:
            return f"Already {len(opens)} open paper positions (the limit is {s.max_open_positions})"
        if mint and any(p["mint"] == mint for p in opens):
            return "Already holding this coin (one position per coin)"
        need = (order_sol if order_sol is not None else s.order_sol) + s.fee_sol
        have = self.wallet()["sol"]
        if have < need:
            return f"Not enough paper SOL ({have:.4f} SOL left, a buy needs {need:.4f})"
        return ""

    # ------------------------------------------------------------------ paper fills
    def buy(self, cand: dict, quote: Quote, decimals: int, order_sol: float) -> dict:
        """A paper buy at the final quote: the coins are the quote's amount minus the haircut, and the wallet pays the
        order plus the fee."""
        s = self.s
        now = iso(self._now())
        w = self.wallet()
        tokens = int(quote.out_amount * (1 - s.haircut_pct / 100))
        sol_in = order_sol + s.fee_sol
        price = order_sol / (tokens / 10 ** decimals) if tokens else None
        pos_id = self.db.insert("sol_positions", {
            "wallet": w["id"], "candidate_id": cand.get("id"), "mint": cand["mint"], "pool": cand.get("pool"),
            "symbol": cand.get("symbol") or cand["mint"][:6], "decimals": decimals, "opened_at": now,
            "sol_in": sol_in, "tokens": str(tokens), "entry_price_sol": price, "peak_value_sol": order_sol,
            "value_sol": order_sol, "value_kind": "quote", "value_at": now, "no_route_count": 0, "cant_sell": 0,
            "status": OPEN})
        self.db.insert("sol_fills", {
            "position_id": pos_id, "wallet": w["id"], "ts": now, "side": "buy", "sol": order_sol, "tokens": str(tokens),
            "haircut_sol": order_sol * s.haircut_pct / 100, "fee_sol": s.fee_sol, "net_sol": -sol_in,
            "price_impact_pct": None if quote.price_impact is None else quote.price_impact * 100,
            "route": ", ".join(quote.labels), "reason": "Paper buy: passed all 7 gates"})
        self._move(-sol_in)
        return self.position(pos_id)

    def mark(self, pos: dict, value_sol: float | None, kind: str, no_route_count: int | None = None,
             cant_sell: bool | None = None) -> dict:
        """Record what the position is worth now: from a real sell quote ("quote") or the price feed ("estimate")."""
        values: dict = {"value_at": iso(self._now())}
        if kind == "estimate":
            values["value_kind"] = "estimate"  # no sell route: whatever value it shows is only a guess now
        if value_sol is not None:
            values.update(value_sol=value_sol, value_kind=kind)
            if kind == "quote":
                values["peak_value_sol"] = max(pos.get("peak_value_sol") or 0.0, value_sol)
        if no_route_count is not None:
            values["no_route_count"] = no_route_count
        if cant_sell is not None:
            values["cant_sell"] = int(cant_sell)
        self.db.update("sol_positions", pos["id"], values)
        return self.position(pos["id"])

    def sell(self, pos: dict, quote: Quote, kind: str, reason: str) -> dict:
        """A paper sale of the whole position at a real sell quote, minus the haircut and the fee."""
        s = self.s
        now = iso(self._now())
        gross = quote.out_amount / LAMPORTS
        haircut = gross * s.haircut_pct / 100
        net = gross - haircut - s.fee_sol
        pnl = net - pos["sol_in"]
        self.db.insert("sol_fills", {
            "position_id": pos["id"], "wallet": pos["wallet"], "ts": now, "side": "sell", "sol": gross,
            "tokens": pos["tokens"], "haircut_sol": haircut, "fee_sol": s.fee_sol, "net_sol": net,
            "price_impact_pct": None if quote.price_impact is None else quote.price_impact * 100,
            "route": ", ".join(quote.labels), "reason": reason[:300]})
        self.db.update("sol_positions", pos["id"], {
            "status": CLOSED, "closed_at": now, "exit_reason": f"{kind}: {reason}"[:300], "sol_out": net,
            "pnl_sol": pnl, "value_sol": gross, "value_kind": "quote", "value_at": now})
        if pos["wallet"] == self.wallet()["id"]:
            self._move(net)
        return self.position(pos["id"])

    # ------------------------------------------------------------------ for the screen
    def view(self, pos: dict, sol_usd: float | None = None) -> dict:
        """A position with its live P/L (after the exit haircut and fee) and its exit levels."""
        s = self.s
        out = dict(pos)
        net = net_value(pos["value_sol"], s) if pos["status"] == OPEN else pos.get("sol_out")
        pnl = (net - pos["sol_in"]) if net is not None and pos["status"] == OPEN else pos.get("pnl_sol")
        out.update(net_value_sol=net, pnl_sol=pnl, pnl_pct=(pnl / pos["sol_in"] * 100) if pnl is not None and
                   pos["sol_in"] else None, old_wallet=pos["wallet"] != self.wallet()["id"] or pos["status"] == OLD_WALLET,
                   pnl_usd=pnl * sol_usd if pnl is not None and sol_usd else None)
        if pos["status"] == OPEN:
            out["levels"] = exit_levels(pos, s)
        return out

    def summary(self, sol_usd: float | None = None) -> dict:
        w = self.wallet()
        opens = self.open_positions()
        held = sum(net_value(p["value_sol"], self.s) or 0.0 for p in opens)
        return {"wallet_id": w["id"], "sol": round(w["sol"], 6), "starting_sol": w["starting_sol"],
                "started_at": w["started_at"], "open_value_sol": round(held, 6), "total_sol": round(w["sol"] + held, 6),
                "usd": round((w["sol"] + held) * sol_usd, 2) if sol_usd else None, "sol_usd": sol_usd,
                "open": len(opens), "cant_sell": sum(1 for p in opens if p["cant_sell"]),
                "today_pnl_sol": self.today_pnl(), "halted_today": self.halted_today(),
                "halted_reason": self.state("halted_reason", "") if self.halted_today() else "",
                "paused": self.paused}
