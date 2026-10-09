"""Asks the local llama-server one question and gets JSON back (its OpenAI-style /v1/chat/completions endpoint, with
the answer held to a JSON schema)."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import httpx

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)


class ChatError(Exception):
    """The server couldn't answer; the message is meant for the user."""

    def __init__(self, message: str, status: int | None = None, kind: str = "error"):
        super().__init__(message)
        self.status = status
        self.kind = kind  # error | timeout | unreachable | loading


@dataclass
class ChatResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    finish_reason: str | None = None
    timings: dict = field(default_factory=dict)
    retried_without_extras: bool = False

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


def build_body(system: str, user: str, schema: dict, max_tokens: int = 700, name: str = "news_signals",
               extras: bool = True) -> dict:
    body = {
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}},
    }
    if extras:
        body["chat_template_kwargs"] = {"enable_thinking": False}
        body["reasoning_effort"] = "none"
    return body


def clean_text(text: str) -> str:
    """The bare JSON answer: no thinking block (some models write an empty one even when told not to think) and no
    ```json fence (llama-server's schema grammar allows one)."""
    text = _THINK.sub("", text or "").strip()
    if text.lower().startswith("<think>"):  # never closed: the answer was cut off while thinking
        return ""
    fenced = _FENCE.match(text)
    if fenced:
        text = fenced.group(1)
    if text and not text.startswith(("{", "[")) and "{" in text and "}" in text:
        text = text[text.index("{"):text.rindex("}") + 1]
    return text.strip()


def _error_text(r: httpx.Response) -> str:
    try:
        data = r.json()
        err = data.get("error") if isinstance(data, dict) else None
        if isinstance(err, dict):
            return str(err.get("message") or err)[:300]
        if err:
            return str(err)[:300]
    except ValueError:
        pass
    return (r.text or "")[:300]


async def chat(url: str, key: str, system: str, user: str, schema: dict, max_tokens: int = 700,
               timeout_s: float = 20, client: httpx.AsyncClient | None = None,
               name: str = "news_signals") -> ChatResult:
    """POST one question; returns the cleaned answer text, token counts, the server's timings and the latency.
    Raises ChatError."""
    own = client is None
    client = client or httpx.AsyncClient(trust_env=False)  # never send local traffic through a proxy
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    started = time.monotonic()
    try:
        retried = False
        for extras in (True, False):
            body = build_body(system, user, schema, max_tokens, name, extras)
            try:
                r = await client.post(f"{url.rstrip('/')}/v1/chat/completions", json=body, headers=headers,
                                      timeout=timeout_s)
            except httpx.TimeoutException as exc:
                raise ChatError(f"Pro AI took longer than {timeout_s:g} seconds to answer.", kind="timeout") from exc
            except httpx.TransportError as exc:
                raise ChatError("Can't reach Pro AI's server - it isn't running.", kind="unreachable") from exc
            if r.status_code in (400, 422) and extras and "context" not in _error_text(r).lower():
                retried = True  # e.g. a field this server doesn't know: once more without the extras
                continue
            break
        latency = int((time.monotonic() - started) * 1000)
        if r.status_code == 401:
            raise ChatError("Pro AI's server refused the app's key (restart Pro AI).", 401)
        if r.status_code == 503:
            raise ChatError("Pro AI is still loading the model.", 503, kind="loading")
        if r.status_code >= 400:
            raise ChatError(f"Pro AI's server answered with an error ({r.status_code}): {_error_text(r)}",
                            r.status_code)
        try:
            data = r.json()
            choice = data["choices"][0]
            message = choice.get("message") or {}
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ChatError("Pro AI's server sent an answer the app couldn't read.") from exc
        usage = data.get("usage") or {}
        return ChatResult(text=clean_text(message.get("content") or ""),
                          prompt_tokens=int(usage.get("prompt_tokens") or 0),
                          completion_tokens=int(usage.get("completion_tokens") or 0),
                          latency_ms=latency, finish_reason=choice.get("finish_reason"),
                          timings=data.get("timings") or {}, retried_without_extras=retried)
    finally:
        if own:
            await client.aclose()
