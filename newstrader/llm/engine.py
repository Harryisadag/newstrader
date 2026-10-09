"""The Pro AI engine: asks the local model (run by llama-server) about one news item.

Same contract as the other engines (see ai/engine.py) and the same JSON shape as Claude (prompts.output_schema), so
the validator, signal merging and risk checks don't change. The answer is held to a schema built for each story:
tickers can only be the story's candidate companies (plus index, sector and country funds for market-wide news),
and there can't be more signals than allowed. Free: cost is always 0.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import time
from collections.abc import Awaitable, Callable
from datetime import datetime

from ..ai.engine import AnalysisResult
from ..ai.keywords import COUNTRY_ETFS, EUROZONE_ETF, MARKET_KEYWORDS
from ..ai.prefilter import PrefilterResult
from ..ai.prompts import output_schema, system_prompt, user_message
from .client import ChatError, ChatResult, chat

log = logging.getLogger(__name__)

# market-wide news ("Fed cuts rates", "Bank of Japan raises rates") may name index, sector and country funds
INDEX_ETFS = ("SPY", "QQQ", "IWM", "DIA")
SECTOR_ETFS = ("XLF", "XLE", "XLK", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC", "SMH", "KRE", "ITB",
               "JETS", "GLD", "USO", "TLT")
COUNTRY_ETF_LIST = tuple(sorted(set(COUNTRY_ETFS.values()) | {EUROZONE_ETF, "MCHI"}))
MACRO_ETFS = INDEX_ETFS + SECTOR_ETFS + COUNTRY_ETF_LIST
# the prefilter's market keywords from "federal reserve" on are about the whole economy, not one company
_MACRO_FROM = MARKET_KEYWORDS.index("federal reserve") if "federal reserve" in MARKET_KEYWORDS else len(MARKET_KEYWORDS)
MACRO_KEYWORDS = frozenset(MARKET_KEYWORDS[_MACRO_FROM:])
# short answers are fast answers (the validator allows more, so nothing it accepts is lost)
TEXT_MAX = {"company": 80, "speaker": 120, "bull_case": 300, "bear_case": 300, "reasoning": 300}
TOKENS_PER_SIGNAL = 320  # one signal with every text at its longest
EMPTY = json.dumps({"signals": []})

ChatFn = Callable[..., Awaitable[ChatResult]]


def is_macro_only(pre: PrefilterResult) -> bool:
    """No company named: only economy-wide words (rates, inflation, tariffs...), or only index / country funds."""
    if not pre.candidates:
        return any(k in MACRO_KEYWORDS for k in pre.keywords)
    return all(c.symbol.upper() in MACRO_ETFS or c.why.startswith("country:") for c in pre.candidates)


def allowed_tickers(pre: PrefilterResult) -> list[str]:
    out = [c.symbol.upper() for c in pre.candidates]
    if is_macro_only(pre):
        out += MACRO_ETFS
    return list(dict.fromkeys(out))


def request_schema(tickers: list[str], max_signals: int) -> dict:
    """output_schema() for one story: ticker limited to `tickers`, at most `max_signals` signals. Property order is
    kept, so the model writes the bull and bear case before it decides."""
    schema = copy.deepcopy(output_schema())
    signals = schema["properties"]["signals"]
    signals["maxItems"] = max(0, int(max_signals))

    def walk(node):
        if isinstance(node, dict):
            props = node.get("properties")
            if isinstance(props, dict):
                for name, prop in props.items():
                    if name == "ticker" and isinstance(prop, dict):
                        prop["enum"] = list(tickers)
                    elif name == "confidence" and isinstance(prop, dict):
                        prop.update(minimum=0, maximum=100)
                    elif name in TEXT_MAX and isinstance(prop, dict) and prop.get("type") == "string":
                        prop["maxLength"] = TEXT_MAX[name]
                        if name == "reasoning":  # the validator rejects a signal without one
                            prop["minLength"] = 1
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(schema)
    return schema


class ProAIEngine:
    engine = "pro"

    def __init__(self, server, model_key: str, ctx=None, max_signals: int = 3, timeout_s: float = 20.0,
                 max_tokens: int | None = None, slots: int = 2, chat_fn: ChatFn | None = None):
        """`server`: a LlamaServer (anything with url, api_key and state; restart_if_crashed and wait_ready_async
        are used when present). `ctx`: the app context, for the max-signals setting (else `max_signals`).
        `slots`: the server's parallel slots - at most that many questions are sent at once. `max_tokens`: answer
        length cap (default: room for every signal at its longest)."""
        self.server = server
        self.model_key = model_key
        self.ctx = ctx
        self.default_max_signals = max_signals
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self._chat = chat_fn or chat
        self.slots = max(1, slots)
        self._sems: dict[int, asyncio.Semaphore] = {}
        self._restart_task: asyncio.Task | None = None

    def _semaphore(self) -> asyncio.Semaphore:
        """At most `slots` questions at once (the server's parallel slots), so the timeout counts answering time,
        not waiting in the server's queue. One per event loop."""
        key = id(asyncio.get_running_loop())
        if key not in self._sems:
            self._sems = {key: asyncio.Semaphore(self.slots)}
        return self._sems[key]

    def max_signals(self) -> int:
        try:
            return int(self.ctx.config.settings.ai.max_signals_per_item)
        except AttributeError:
            return self.default_max_signals

    def _not_ready(self) -> str | None:
        state = getattr(self.server, "state", "ready")
        if state == "ready":
            return None
        if state == "starting":
            return "Pro AI is still loading the model."
        message = getattr(self.server, "message", "")
        return f"Pro AI isn't running.{' ' + message if message else ''}"

    def _result(self, **kw) -> AnalysisResult:
        return AnalysisResult(engine="pro", model=self.model_key, cost_usd=0.0, **kw)

    async def analyze(self, item, pre: PrefilterResult, market_open: bool | None, now: datetime,
                      model: str | None = None, skip_rate_limit: bool = False) -> AnalysisResult:
        tickers = allowed_tickers(pre)
        if not tickers:  # nothing it could name: neutral, without asking the model
            return self._result(ok=True, text=EMPTY, stop_reason="end_turn")
        problem = self._not_ready()
        if problem:
            if getattr(self.server, "state", "") == "error":
                self._maybe_restart()
            return self._result(ok=False, status="error", error=problem)
        max_signals = self.max_signals()
        schema = request_schema(tickers, max_signals)
        started = time.monotonic()
        try:
            async with self._semaphore():
                res = await self._chat(self.server.url, self.server.api_key, system_prompt(max_signals),
                                       user_message(item, pre, now, market_open), schema,
                                       max_tokens=self.max_tokens or 150 + TOKENS_PER_SIGNAL * max_signals,
                                       timeout_s=self.timeout_s)
        except ChatError as exc:
            if exc.kind == "unreachable":
                self._maybe_restart()
            log.warning("Pro AI call failed: %s", exc)
            return self._result(ok=False, status="error", error=str(exc),
                                latency_ms=int((time.monotonic() - started) * 1000))
        except Exception as exc:  # never let one story break the pipeline
            log.exception("Pro AI call failed")
            return self._result(ok=False, status="error", error=f"Pro AI error: {type(exc).__name__}: {exc}",
                                latency_ms=int((time.monotonic() - started) * 1000))
        out = self._result(ok=True, text=res.text, stop_reason=res.finish_reason or "stop",
                           input_tokens=res.prompt_tokens, output_tokens=res.completion_tokens,
                           latency_ms=res.latency_ms)
        if res.truncated:
            out.ok, out.status, out.error = False, "truncated", "Pro AI's answer was cut off (too long)."
        elif not res.text:
            out.ok, out.status, out.error = False, "error", "Pro AI gave an empty answer."
        return out

    def _maybe_restart(self) -> None:
        """The server seems to have died: restart it once (the server's own policy) and wait for it in the
        background, so the next stories work again."""
        restart = getattr(self.server, "restart_if_crashed", None)
        if restart is None or (self._restart_task is not None and not self._restart_task.done()):
            return
        self._restart_task = asyncio.get_running_loop().create_task(self._restart(restart))

    async def _restart(self, restart) -> None:
        try:
            if await asyncio.to_thread(restart):
                await self.server.wait_ready_async()
        except Exception:
            log.exception("Pro AI restart failed")
