"""SQLite storage.

One connection per thread, WAL mode so the UI can read while the engine writes, and a single write
lock so writes from different threads never collide. Schema changes are applied as numbered
migrations (PRAGMA user_version).
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MIGRATIONS: list[str] = [
    # 1 - initial schema
    """
    CREATE TABLE kv_state (
        key TEXT PRIMARY KEY,
        value TEXT,
        updated_at TEXT
    );

    CREATE TABLE logs (
        id INTEGER PRIMARY KEY,
        ts TEXT NOT NULL,
        level TEXT NOT NULL,
        logger TEXT,
        message TEXT
    );
    CREATE INDEX idx_logs_ts ON logs(ts);

    CREATE TABLE tickers (
        symbol TEXT PRIMARY KEY,
        name TEXT,
        clean_name TEXT,
        exchange TEXT,
        tradable INTEGER,
        shortable INTEGER,
        easy_to_borrow INTEGER,
        fractionable INTEGER,
        updated_at TEXT
    );

    CREATE TABLE news_items (
        id INTEGER PRIMARY KEY,
        source_id TEXT,
        source_type TEXT,
        source_name TEXT,
        external_id TEXT,
        title TEXT,
        body TEXT,
        url TEXT,
        speaker TEXT,
        published_at TEXT,
        received_at TEXT,
        symbols TEXT,
        fingerprint TEXT,
        duplicate_of INTEGER,
        prefilter_hit INTEGER,
        candidates TEXT,
        status TEXT
    );
    CREATE UNIQUE INDEX idx_news_ext ON news_items(source_id, external_id);
    CREATE INDEX idx_news_received ON news_items(received_at);

    CREATE TABLE transcripts (
        id INTEGER PRIMARY KEY,
        source_id TEXT,
        source_name TEXT,
        start_ts TEXT,
        end_ts TEXT,
        text TEXT,
        created_at TEXT
    );
    CREATE INDEX idx_transcripts_src ON transcripts(source_id, start_ts);

    CREATE TABLE analyses (
        id INTEGER PRIMARY KEY,
        created_at TEXT,
        trading_day TEXT,
        item_kind TEXT,
        item_id INTEGER,
        source_id TEXT,
        source_name TEXT,
        model TEXT,
        input_tokens INTEGER,
        output_tokens INTEGER,
        cache_read_tokens INTEGER,
        cache_write_tokens INTEGER,
        cost_usd REAL,
        latency_ms INTEGER,
        status TEXT,
        error TEXT,
        raw_response TEXT,
        is_backtest INTEGER DEFAULT 0
    );
    CREATE INDEX idx_analyses_day ON analyses(trading_day);

    CREATE TABLE signals (
        id INTEGER PRIMARY KEY,
        analysis_id INTEGER,
        created_at TEXT,
        ticker TEXT,
        company TEXT,
        direction TEXT,
        confidence INTEGER,
        speaker TEXT,
        source_id TEXT,
        source_name TEXT,
        source_type TEXT,
        reasoning TEXT,
        bull_case TEXT,
        bear_case TEXT,
        time_sensitivity TEXT,
        headline TEXT,
        url TEXT,
        action TEXT,
        action_reason TEXT,
        traded INTEGER DEFAULT 0,
        order_id INTEGER,
        merged_into INTEGER,
        corroborations INTEGER DEFAULT 1,
        sources_seen TEXT,
        review_status TEXT,
        price_at_signal REAL
    );
    CREATE INDEX idx_signals_created ON signals(created_at);
    CREATE INDEX idx_signals_ticker ON signals(ticker, created_at);

    CREATE TABLE orders (
        id INTEGER PRIMARY KEY,
        signal_id INTEGER,
        alpaca_order_id TEXT UNIQUE,
        client_order_id TEXT,
        symbol TEXT,
        side TEXT,
        intent TEXT,
        qty REAL,
        order_type TEXT,
        order_class TEXT,
        status TEXT,
        entry_price_est REAL,
        stop_price REAL,
        take_profit_price REAL,
        filled_qty REAL,
        filled_avg_price REAL,
        submitted_at TEXT,
        filled_at TEXT,
        updated_at TEXT,
        mode TEXT,
        reason TEXT,
        raw TEXT
    );
    CREATE INDEX idx_orders_submitted ON orders(submitted_at);
    CREATE INDEX idx_orders_symbol ON orders(symbol, submitted_at);

    CREATE TABLE equity_snapshots (
        ts TEXT PRIMARY KEY,
        equity REAL,
        cash REAL,
        buying_power REAL,
        mode TEXT
    );

    CREATE TABLE signal_prices (
        signal_id INTEGER PRIMARY KEY,
        ticker TEXT,
        direction TEXT,
        t0 TEXT,
        price_t0 REAL,
        due_5m TEXT,
        due_1h TEXT,
        due_1d TEXT,
        price_5m REAL,
        price_1h REAL,
        price_1d REAL,
        ret_5m REAL,
        ret_1h REAL,
        ret_1d REAL,
        status TEXT,
        updated_at TEXT
    );

    CREATE TABLE backtest_runs (
        id INTEGER PRIMARY KEY,
        created_at TEXT,
        params TEXT,
        status TEXT,
        progress REAL,
        message TEXT,
        summary TEXT,
        cost_usd REAL
    );

    CREATE TABLE backtest_results (
        id INTEGER PRIMARY KEY,
        run_id INTEGER,
        news_time TEXT,
        headline TEXT,
        source TEXT,
        ticker TEXT,
        direction TEXT,
        confidence INTEGER,
        reasoning TEXT,
        action TEXT,
        qty REAL,
        entry_time TEXT,
        entry_price REAL,
        exit_time TEXT,
        exit_price REAL,
        exit_reason TEXT,
        pnl REAL,
        ret_pct REAL
    );
    CREATE INDEX idx_bt_results_run ON backtest_results(run_id);
    """,
    # 2 - link bracket legs to their parent order
    """
    ALTER TABLE orders ADD COLUMN parent_alpaca_id TEXT;
    CREATE INDEX idx_orders_parent ON orders(parent_alpaca_id);
    """,
    # 3 - backtest accuracy columns
    """
    ALTER TABLE backtest_results ADD COLUMN ret_1h REAL;
    ALTER TABLE backtest_results ADD COLUMN url TEXT;
    ALTER TABLE backtest_results ADD COLUMN time_sensitivity TEXT;
    """,
    # 4 - local machine-learning engine
    """
    ALTER TABLE analyses ADD COLUMN engine TEXT;
    ALTER TABLE signals ADD COLUMN engine TEXT;
    UPDATE analyses SET engine = 'claude' WHERE engine IS NULL;
    UPDATE signals SET engine = 'claude' WHERE engine IS NULL;
    CREATE TABLE ml_samples (
        news_id TEXT NOT NULL,
        symbol TEXT NOT NULL,
        horizon_min INTEGER NOT NULL,
        published_at TEXT,
        headline TEXT,
        summary TEXT,
        symbols TEXT,
        label_status TEXT,
        entry_price REAL,
        exit_price REAL,
        ret_pct REAL,
        spy_ret_pct REAL,
        adj_ret_pct REAL,
        created_at TEXT,
        PRIMARY KEY (news_id, symbol, horizon_min)
    );
    CREATE INDEX idx_ml_samples_time ON ml_samples(horizon_min, published_at);
    CREATE TABLE ml_sentiment_cache (
        model_id TEXT NOT NULL,
        text_hash TEXT NOT NULL,
        positive REAL,
        negative REAL,
        neutral REAL,
        PRIMARY KEY (model_id, text_hash)
    );
    CREATE TABLE ml_runs (
        id INTEGER PRIMARY KEY,
        created_at TEXT,
        params TEXT,
        status TEXT,
        progress REAL,
        message TEXT,
        report TEXT
    );
    """,
    # 5 - training text built like the live feed (headline + summary + article body)
    """
    ALTER TABLE ml_samples ADD COLUMN text_version INTEGER;
    """,
    # 6 - market monitor: sudden price/volume spikes and market-wide moves
    """
    CREATE TABLE market_events (
        id INTEGER PRIMARY KEY,
        ts TEXT NOT NULL,
        trading_day TEXT,
        kind TEXT NOT NULL,
        scope TEXT,
        symbol TEXT,
        price REAL,
        ref_price REAL,
        change_pct REAL,
        window_min INTEGER,
        volume REAL,
        volume_ratio REAL,
        level TEXT,
        news_id INTEGER,
        signal_id INTEGER,
        headline TEXT,
        url TEXT,
        alerted INTEGER DEFAULT 0,
        detail TEXT
    );
    CREATE INDEX idx_market_events_ts ON market_events(ts);
    CREATE INDEX idx_market_events_symbol ON market_events(symbol, ts);
    """,
    # 7 - international: the language a transcript / news item was in
    """
    ALTER TABLE transcripts ADD COLUMN language TEXT;
    ALTER TABLE news_items ADD COLUMN language TEXT;
    """,
    # 8 - better detection: the news event behind a signal, its flags, and how far the price had already moved
    """
    ALTER TABLE signals ADD COLUMN event TEXT;
    ALTER TABLE signals ADD COLUMN flags TEXT;
    ALTER TABLE signals ADD COLUMN pre_move_pct REAL;
    """,
    # 9 - the built-in paper-trading simulator (no broker account needed)
    """
    CREATE TABLE sim_state (
        key TEXT PRIMARY KEY,
        value TEXT,
        updated_at TEXT
    );
    CREATE TABLE sim_equity (
        ts TEXT PRIMARY KEY,
        equity REAL
    );
    """,
    # 10 - speed timer: when the news came out, when the app got it and when the AI decided (orders already have
    # submitted_at). Older signals are filled in from their news item; an old TV clip's "published" stays empty
    # because only the start of the clip was saved, not when its newest words were spoken.
    """
    ALTER TABLE signals ADD COLUMN news_published_at TEXT;
    ALTER TABLE signals ADD COLUMN news_received_at TEXT;
    ALTER TABLE signals ADD COLUMN decided_at TEXT;
    UPDATE signals SET decided_at = created_at;
    UPDATE signals SET news_received_at = (SELECT n.received_at FROM analyses a JOIN news_items n ON n.id = a.item_id
                                           WHERE a.id = signals.analysis_id);
    UPDATE signals SET news_published_at = (SELECT n.published_at FROM analyses a JOIN news_items n ON n.id = a.item_id
                                            WHERE a.id = signals.analysis_id AND a.item_kind = 'news');
    """,
    # 11 - Pro AI: which main-engine reading a Pro AI reading belongs to, why the story was sent to it and whether
    # the two agreed. On a Pro AI signal main_direction is the main engine's call for that stock (NULL = the main
    # engine didn't read the story); on a main-engine signal pro_direction is Pro AI's call.
    """
    ALTER TABLE analyses ADD COLUMN main_analysis_id INTEGER;
    ALTER TABLE analyses ADD COLUMN pro_reason TEXT;
    ALTER TABLE analyses ADD COLUMN agreement TEXT;
    ALTER TABLE signals ADD COLUMN main_direction TEXT;
    ALTER TABLE signals ADD COLUMN pro_direction TEXT;
    CREATE INDEX idx_signals_analysis ON signals(analysis_id);
    CREATE INDEX idx_analyses_engine ON analyses(engine, created_at);
    """,
    # 12 - charts: what the chart said about a news signal before it was traded (agrees / neutral / against /
    # stretched / unavailable, why, the chart's score -1..+1 and the confidence points it added or took away)
    """
    ALTER TABLE signals ADD COLUMN chart_verdict TEXT;
    ALTER TABLE signals ADD COLUMN chart_reason TEXT;
    ALTER TABLE signals ADD COLUMN chart_score REAL;
    ALTER TABLE signals ADD COLUMN chart_adjust INTEGER;
    """,
]


def utcnow() -> datetime:
    return datetime.now(UTC)


def iso(dt: datetime | None = None) -> str:
    """ISO-8601 UTC timestamp string (what every *_at / ts column stores)."""
    return (dt or utcnow()).astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class Database:
    def __init__(self, path: Path | str):
        self.path = str(path)
        self._local = threading.local()
        self._write_lock = threading.RLock()
        self._all_conns: list[sqlite3.Connection] = []
        self._conns_lock = threading.Lock()
        self.migrate()

    # ---- connections ----
    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self.path, timeout=30, check_same_thread=False, isolation_level=None)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=NORMAL")
            c.execute("PRAGMA foreign_keys=ON")
            c.execute("PRAGMA busy_timeout=10000")
            self._local.conn = c
            with self._conns_lock:
                self._all_conns.append(c)
        return c

    def close(self) -> None:
        with self._conns_lock:
            for c in self._all_conns:
                try:
                    c.close()
                except sqlite3.Error:
                    pass
            self._all_conns.clear()
        self._local = threading.local()

    # ---- schema ----
    def migrate(self) -> None:
        with self._write_lock:
            c = self.conn()
            version = c.execute("PRAGMA user_version").fetchone()[0]
            for i, script in enumerate(MIGRATIONS[version:], start=version + 1):
                c.execute("BEGIN")
                try:
                    for statement in script.split(";"):
                        if statement.strip():
                            c.execute(statement)
                    c.execute(f"PRAGMA user_version={i}")
                    c.execute("COMMIT")
                except Exception:
                    c.execute("ROLLBACK")
                    raise

    # ---- helpers ----
    def execute(self, sql: str, params: Sequence[Any] | dict = ()) -> int:
        """Run a write statement. Returns lastrowid (for INSERT) or rowcount."""
        with self._write_lock:
            cur = self.conn().execute(sql, params)
            return cur.lastrowid if sql.lstrip().upper().startswith("INSERT") else cur.rowcount

    def executemany(self, sql: str, rows: Iterable[Sequence[Any]]) -> None:
        with self._write_lock:
            c = self.conn()
            c.execute("BEGIN")
            try:
                c.executemany(sql, rows)
                c.execute("COMMIT")
            except Exception:
                c.execute("ROLLBACK")
                raise

    def query(self, sql: str, params: Sequence[Any] | dict = ()) -> list[dict]:
        return [dict(r) for r in self.conn().execute(sql, params).fetchall()]

    def query_one(self, sql: str, params: Sequence[Any] | dict = ()) -> dict | None:
        row = self.conn().execute(sql, params).fetchone()
        return dict(row) if row else None

    def scalar(self, sql: str, params: Sequence[Any] | dict = ()) -> Any:
        row = self.conn().execute(sql, params).fetchone()
        return row[0] if row else None

    def insert(self, table: str, values: dict) -> int:
        cols = ", ".join(values)
        marks = ", ".join("?" for _ in values)
        return self.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", list(values.values()))

    def update(self, table: str, row_id: int, values: dict, id_col: str = "id") -> None:
        if not values:
            return
        sets = ", ".join(f"{k} = ?" for k in values)
        self.execute(f"UPDATE {table} SET {sets} WHERE {id_col} = ?", [*values.values(), row_id])

    # ---- key/value state ----
    def kv_get(self, key: str, default: Any = None) -> Any:
        row = self.query_one("SELECT value FROM kv_state WHERE key = ?", (key,))
        if row is None or row["value"] is None:
            return default
        return json.loads(row["value"])

    def kv_set(self, key: str, value: Any) -> None:
        self.execute(
            "INSERT INTO kv_state (key, value, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
            (key, json.dumps(value), iso()),
        )
