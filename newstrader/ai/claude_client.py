"""The Claude engine: ask Claude about one news item. Handles model options, cost accounting and errors."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from datetime import datetime

from ..context import AppContext
from .costs import estimate_cost
from .engine import AnalysisResult
from .prompts import output_schema, system_prompt, user_message

log = logging.getLogger(__name__)

REFUSAL_FALLBACK_BETA = "server-side-fallback-2026-07-01"


def supports_effort(model: str) -> bool:
    """Haiku 4.5 rejects the effort parameter; current Sonnet/Opus/Fable models accept it."""
    return not model.startswith("claude-haiku")


def supports_refusal_fallback(model: str) -> bool:
    return model in ("claude-sonnet-5-5", "claude-opus-5-5", "claude-opus-5", "claude-fable-5-1")


# Kept for older imports; every engine returns the same AnalysisResult.
ClaudeResult = AnalysisResult


class RateLimiter:
    """At most `per_minute` starts in any rolling 60 seconds."""

    def __init__(self) -> None:
        self._starts: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def wait(self, per_minute: int) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                while self._starts and now - self._starts[0] > 60:
                    self._starts.popleft()
                if len(self._starts) < per_minute:
                    self._starts.append(now)
                    return
                await asyncio.sleep(max(0.05, 60 - (now - self._starts[0])))


class ClaudeAnalyzer:
    engine = "claude"

    def __init__(self, ctx: AppContext, client_factory=None):
        self.ctx = ctx
        self._client = None
        self._client_key = None
        self._factory = client_factory
        self.limiter = RateLimiter()
        self._sem: asyncio.Semaphore | None = None
        self._sem_size = 0

    def _get_client(self):
        if self._factory is not None:
            return self._factory()
        key = self.ctx.keys.keys.anthropic
        if not key:
            return None
        if self._client is None or self._client_key != key:
            import anthropic

            self._client = anthropic.AsyncAnthropic(api_key=key, max_retries=2, timeout=90)
            self._client_key = key
        return self._client

    def _semaphore(self) -> asyncio.Semaphore:
        size = self.ctx.config.settings.ai.max_concurrent_calls
        if self._sem is None or size != self._sem_size:
            self._sem = asyncio.Semaphore(size)
            self._sem_size = size
        return self._sem

    def build_request(self, user_text: str, model: str) -> dict:
        ai = self.ctx.config.settings.ai
        output_config: dict = {"format": {"type": "json_schema", "schema": output_schema()}}
        if supports_effort(model):
            output_config["effort"] = ai.effort
        return {
            "model": model,
            "max_tokens": 8000,
            "system": [{"type": "text", "text": system_prompt(ai.max_signals_per_item),
                        "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": user_text}],
            "output_config": output_config,
        }

    async def analyze(self, item, pre, market_open: bool | None, now: datetime, model: str | None = None,
                      skip_rate_limit: bool = False) -> ClaudeResult:
        ai = self.ctx.config.settings.ai
        model = model or ai.model
        client = self._get_client()
        if client is None:
            return ClaudeResult(ok=False, status="error", model=model,
                                error="No Anthropic API key - add it in Settings -> API keys")
        params = self.build_request(user_message(item, pre, now, market_open), model)
        if not skip_rate_limit:
            await self.limiter.wait(ai.max_calls_per_minute)
        async with self._semaphore():
            started = time.monotonic()
            try:
                if ai.use_refusal_fallback and supports_refusal_fallback(model):
                    resp = await client.beta.messages.create(**params, betas=[REFUSAL_FALLBACK_BETA],
                                                             fallbacks="default")
                else:
                    resp = await client.messages.create(**params)
            except Exception as exc:
                latency = int((time.monotonic() - started) * 1000)
                msg = _describe_error(exc)
                self.ctx.state.set_status("claude", "error", msg)
                log.warning("Claude call failed: %s", msg)
                return ClaudeResult(ok=False, status="error", model=model, error=msg, latency_ms=latency)
        latency = int((time.monotonic() - started) * 1000)
        return self._parse(resp, model, latency)

    def _parse(self, resp, requested_model: str, latency: int) -> ClaudeResult:
        usage = getattr(resp, "usage", None)
        served = getattr(resp, "model", None) or requested_model
        res = ClaudeResult(ok=True, model=served, stop_reason=getattr(resp, "stop_reason", None), latency_ms=latency)
        if usage is not None:
            res.input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
            res.output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
            res.cache_read_tokens = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
            res.cache_write_tokens = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
        res.cost_usd = estimate_cost(served, res.input_tokens, res.output_tokens, res.cache_write_tokens,
                                     res.cache_read_tokens)
        texts = [b.text for b in (resp.content or []) if getattr(b, "type", None) == "text"]
        res.text = "".join(texts).strip()
        if res.stop_reason == "refusal":
            details = getattr(resp, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            res.ok, res.status = False, "refusal"
            res.error = f"Claude declined to answer (category: {category or 'unspecified'})"
        elif res.stop_reason == "max_tokens":
            res.ok, res.status = False, "truncated"
            res.error = "Response was cut off (max_tokens)"
        elif not res.text:
            res.ok, res.status = False, "error"
            res.error = "Empty response"
        else:
            self.ctx.state.set_status("claude", "ok", f"{served} - last call {latency / 1000:.1f}s")
        return res


def _describe_error(exc: Exception) -> str:
    try:
        import anthropic
    except Exception:  # pragma: no cover
        return f"{type(exc).__name__}: {exc}"
    if isinstance(exc, anthropic.AuthenticationError):
        return "Invalid Anthropic API key (check Settings -> API keys)"
    if isinstance(exc, anthropic.PermissionDeniedError):
        return "API key isn't allowed to use this model"
    if isinstance(exc, anthropic.NotFoundError):
        return "Model not found - check Settings -> AI engine -> Claude model"
    if isinstance(exc, anthropic.RateLimitError):
        return "Rate limited by Anthropic (too many calls) - lower 'Max Claude calls per minute'"
    if isinstance(exc, anthropic.BadRequestError):
        return f"Bad request: {getattr(exc, 'message', exc)}"
    if isinstance(exc, anthropic.APIStatusError):
        return f"Anthropic API error {exc.status_code}: {getattr(exc, 'message', exc)}"
    if isinstance(exc, anthropic.APIConnectionError):
        return "Can't reach the Anthropic API (internet connection?)"
    return f"{type(exc).__name__}: {exc}"
