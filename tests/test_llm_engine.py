"""Pro AI's JSON client and engine: request body, cleaning the answer, per-story schema, errors, self-test."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from types import SimpleNamespace

import httpx
import pytest

from newstrader.ai.prefilter import Candidate, PrefilterResult
from newstrader.ai.prompts import output_schema
from newstrader.ai.validator import validate_response
from newstrader.db import Database
from newstrader.llm import client as llm_client
from newstrader.llm import selftest
from newstrader.llm.client import ChatError, ChatResult, build_body, chat, clean_text
from newstrader.llm.engine import (
    EMPTY,
    MACRO_ETFS,
    ProAIEngine,
    allowed_tickers,
    is_macro_only,
    request_schema,
)
from newstrader.sources.base import NewsItem

from .helpers import make_tickers, sig

NOW = datetime(2026, 10, 7, 15, 0, tzinfo=UTC)


def item(title="Apple beats estimates") -> NewsItem:
    return NewsItem(source_id="t", source_type="rss", source_name="Reuters", external_id="1", title=title,
                    published_at=NOW)


def pre(*symbols, keywords=(), why="name") -> PrefilterResult:
    cands = [Candidate(s, s.title(), why) for s in symbols]
    return PrefilterResult(hit=bool(cands or keywords), candidates=cands, keywords=list(keywords))


# ------------------------------------------------------------------------------------------------ client
def test_body_switches_thinking_off_and_holds_the_schema():
    schema = request_schema(["AAPL"], 3)
    body = build_body("sys", "user", schema, max_tokens=500)
    assert body["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "user"}]
    assert body["temperature"] == 0 and body["max_tokens"] == 500
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["schema"] == schema
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["reasoning_effort"] == "none"
    plain = build_body("sys", "user", schema, extras=False)
    assert "chat_template_kwargs" not in plain and "reasoning_effort" not in plain


@pytest.mark.parametrize(("raw", "clean"), [
    ('{"signals": []}', '{"signals": []}'),
    ('<think>\n\n</think>\n\n{"signals": []}', '{"signals": []}'),
    ('<think>hmm, Apple...</think>{"signals": []}', '{"signals": []}'),
    ('<THINK>x</THINK> {"signals": []} ', '{"signals": []}'),
    ('<think>never finished', ""),
    ('```json\n{"signals": []}\n```', '{"signals": []}'),
    ('<think></think>\n```json\n{"signals": []}\n```\n', '{"signals": []}'),
    ('Here you go: {"signals": []} hope it helps', '{"signals": []}'),
    ("", ""),
    (None, ""),
])
def test_clean_text(raw, clean):
    assert clean_text(raw) == clean


def _transport(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _ok(content='{"signals": []}', finish="stop") -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}, "finish_reason": finish}],
                                     "usage": {"prompt_tokens": 12, "completion_tokens": 3},
                                     "timings": {"predicted_per_second": 50.0}})


def _chat(handler, **kw):
    async def go():
        async with _transport(handler) as c:
            return await chat("http://127.0.0.1:1", "KEY", "sys", "user", {"type": "object"}, client=c, **kw)

    return asyncio.run(go())


def test_chat_sends_key_and_reads_answer():
    seen = []

    def handler(request: httpx.Request):
        seen.append(request)
        return _ok("<think>\n</think>\n" + '{"signals": []}')

    res = _chat(handler)
    assert seen[0].url.path == "/v1/chat/completions"
    assert seen[0].headers["Authorization"] == "Bearer KEY"
    assert res.text == '{"signals": []}' and res.prompt_tokens == 12 and res.completion_tokens == 3
    assert res.timings["predicted_per_second"] == 50.0 and not res.retried_without_extras and not res.truncated


def test_chat_retries_without_extra_fields():
    bodies = []

    def handler(request: httpx.Request):
        body = json.loads(request.content)
        bodies.append(body)
        if "reasoning_effort" in body:
            return httpx.Response(400, json={"error": {"message": "unknown field reasoning_effort"}})
        return _ok()

    res = _chat(handler)
    assert res.retried_without_extras and len(bodies) == 2
    assert "chat_template_kwargs" not in bodies[1] and bodies[1]["response_format"] == bodies[0]["response_format"]


def test_chat_does_not_retry_a_too_long_story():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(400, json={"error": {"message": "the request exceeds the available context size"}})

    with pytest.raises(ChatError, match="context size") as exc:
        _chat(handler)
    assert exc.value.status == 400 and len(calls) == 1


@pytest.mark.parametrize(("response", "kind", "words"), [
    (httpx.Response(401, json={"error": {"message": "Invalid API Key"}}), "error", "key"),
    (httpx.Response(503, json={"error": {"message": "Loading model"}}), "loading", "still loading"),
    (httpx.Response(500, text="boom"), "error", "boom"),
    (httpx.Response(200, text="not json"), "error", "couldn't read"),
    (httpx.Response(200, json={"choices": []}), "error", "couldn't read"),
])
def test_chat_errors(response, kind, words):
    with pytest.raises(ChatError) as exc:
        _chat(lambda request: response)
    assert exc.value.kind == kind and words in str(exc.value)


@pytest.mark.parametrize(("error", "kind"), [(httpx.ReadTimeout("slow"), "timeout"),
                                              (httpx.ConnectError("refused"), "unreachable")])
def test_chat_transport_errors(error, kind):
    def handler(request):
        raise error

    with pytest.raises(ChatError) as exc:
        _chat(handler, timeout_s=3)
    assert exc.value.kind == kind


def test_chat_reports_cut_off_answers():
    res = _chat(lambda request: _ok('{"signals": [', "length"))
    assert res.truncated


# ------------------------------------------------------------------------------------------------ schema
def test_request_schema_limits_tickers_and_count():
    base = output_schema()
    schema = request_schema(["AAPL", "BRK.B"], 2)
    signals = schema["properties"]["signals"]
    assert signals["maxItems"] == 2
    props = signals["items"]["properties"]
    assert props["ticker"]["enum"] == ["AAPL", "BRK.B"]
    assert (props["confidence"]["minimum"], props["confidence"]["maximum"]) == (0, 100)
    assert props["reasoning"]["minLength"] == 1 and props["reasoning"]["maxLength"] <= 400
    assert all(props[k]["maxLength"] for k in ("company", "speaker", "bull_case", "bear_case"))
    assert list(props) == list(base["properties"]["signals"]["items"]["properties"])  # cases before the call
    assert output_schema() == base  # the shared schema isn't changed
    assert request_schema(["AAPL"], 0)["properties"]["signals"]["maxItems"] == 0


def test_allowed_tickers():
    assert allowed_tickers(pre("AAPL", "GOOGL")) == ["AAPL", "GOOGL"]
    assert allowed_tickers(pre("aapl", "AAPL")) == ["AAPL"]
    # market-wide news: the funds
    macro = allowed_tickers(pre(keywords=["the fed", "rate cut"]))
    assert macro == list(MACRO_ETFS) and {"SPY", "QQQ", "XLF", "TLT", "EWJ", "EZU"} <= set(macro)
    assert len(macro) == len(set(macro))
    # a country's fund found for "Bank of Japan raises rates": that fund first, then the rest
    japan = allowed_tickers(pre("EWJ", why="country: Bank Of Japan"))
    assert japan[0] == "EWJ" and "SPY" in japan
    # company news without a company: nothing it could name
    assert allowed_tickers(pre(keywords=["beats estimates"])) == []
    assert allowed_tickers(pre()) == []


def test_is_macro_only():
    assert is_macro_only(pre(keywords=["inflation"]))
    assert is_macro_only(pre("SPY", why="s&p 500"))
    assert not is_macro_only(pre("SPY", "AAPL"))
    assert not is_macro_only(pre(keywords=["earnings"]))


# ------------------------------------------------------------------------------------------------ engine
class FakeServer:
    def __init__(self, state="ready"):
        self.url, self.api_key, self.state, self.message = "http://127.0.0.1:1", "KEY", state, ""
        self.restarts = 0

    def restart_if_crashed(self):
        self.restarts += 1
        self.state = "starting"
        return True

    async def wait_ready_async(self, timeout=180.0):
        self.state = "ready"
        return True


class FakeChat:
    def __init__(self, text='{"signals": []}', error: Exception | None = None, finish="stop", delay=0.0):
        self.text, self.error, self.finish, self.delay = text, error, finish, delay
        self.calls: list[dict] = []
        self.active = self.peak = 0

    async def __call__(self, url, key, system, user, schema, **kw):
        self.calls.append({"url": url, "key": key, "system": system, "user": user, "schema": schema, **kw})
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(self.delay)
            if self.error:
                raise self.error
            return ChatResult(text=self.text, prompt_tokens=900, completion_tokens=70, latency_ms=850,
                              finish_reason=self.finish)
        finally:
            self.active -= 1


@pytest.fixture
def tickers(tmp_path):
    return make_tickers(Database(tmp_path / "t.db"))


def test_engine_asks_with_the_stories_schema(tickers):
    answer = json.dumps({"signals": [sig("AAPL", "bullish", 82)]})
    fake = FakeChat(answer)
    ctx = SimpleNamespace(config=SimpleNamespace(settings=SimpleNamespace(ai=SimpleNamespace(max_signals_per_item=2))))
    engine = ProAIEngine(FakeServer(), "model-x", ctx=ctx, timeout_s=15, max_tokens=600, chat_fn=fake)
    res = asyncio.run(engine.analyze(item(), pre("AAPL", "GOOGL"), True, NOW))
    assert res.ok and res.status == "ok" and res.engine == "pro" and res.model == "model-x" and res.cost_usd == 0
    assert (res.input_tokens, res.output_tokens, res.latency_ms) == (900, 70, 850)
    call = fake.calls[0]
    assert call["key"] == "KEY" and call["timeout_s"] == 15 and call["max_tokens"] == 600
    assert call["schema"]["properties"]["signals"]["maxItems"] == 2
    assert call["schema"]["properties"]["signals"]["items"]["properties"]["ticker"]["enum"] == ["AAPL", "GOOGL"]
    assert "at most 2" in call["system"] and "Apple beats estimates" in call["user"]
    v = validate_response(res.text, tickers, 2)
    assert v.status == "ok" and [s.ticker for s in v.signals] == ["AAPL"]


def test_engine_output_passes_the_validator_for_macro_news(tickers):
    answer = json.dumps({"signals": [sig("SPY", "bearish", 70, reasoning="A surprise rate hike hits stocks.")]})
    fake = FakeChat(answer)
    engine = ProAIEngine(FakeServer(), "m", chat_fn=fake)
    res = asyncio.run(engine.analyze(item("Fed raises rates"), pre(keywords=["rate hike"]), True, NOW))
    assert fake.calls[0]["max_tokens"] >= 3 * 300  # room for three signals at their longest
    v = validate_response(res.text, tickers, 3)
    assert v.status == "ok" and v.signals[0].ticker == "SPY" and v.signals[0].direction == "bearish"


def test_no_candidates_short_circuits(tickers):
    fake = FakeChat()
    engine = ProAIEngine(FakeServer(state="off"), "m", chat_fn=fake)
    res = asyncio.run(engine.analyze(item("Startup files for IPO"), pre(keywords=["ipo"]), True, NOW))
    assert res.ok and res.text == EMPTY and fake.calls == []
    assert validate_response(res.text, tickers).status == "ok"


@pytest.mark.parametrize(("error", "words"), [
    (ChatError("Pro AI took longer than 20 seconds to answer.", kind="timeout"), "longer than"),
    (ChatError("Pro AI is still loading the model.", 503, kind="loading"), "loading"),
    (ChatError("Pro AI's server answered with an error (500): boom", 500), "500"),
    (RuntimeError("surprise"), "surprise"),
])
def test_engine_maps_errors(error, words):
    engine = ProAIEngine(FakeServer(), "m", chat_fn=FakeChat(error=error))
    res = asyncio.run(engine.analyze(item(), pre("AAPL"), True, NOW))
    assert not res.ok and res.status == "error" and words in res.error and res.engine == "pro"


def test_unreachable_server_is_restarted_once():
    server = FakeServer()
    engine = ProAIEngine(server, "m", chat_fn=FakeChat(error=ChatError("Can't reach", kind="unreachable")))

    async def go():
        res = await engine.analyze(item(), pre("AAPL"), True, NOW)
        await asyncio.sleep(0)
        await engine._restart_task
        return res

    res = asyncio.run(go())
    assert not res.ok and server.restarts == 1 and server.state == "ready"


@pytest.mark.parametrize(("state", "words"), [("starting", "still loading"), ("error", "isn't running"),
                                              ("off", "isn't running")])
def test_engine_when_server_not_ready(state, words):
    server = FakeServer(state)
    server.message = "It ran out of memory."
    fake = FakeChat()
    res = asyncio.run(ProAIEngine(server, "m", chat_fn=fake).analyze(item(), pre("AAPL"), True, NOW))
    assert not res.ok and words in res.error and fake.calls == []
    if state != "starting":
        assert "out of memory" in res.error


def test_truncated_and_empty_answers():
    res = asyncio.run(ProAIEngine(FakeServer(), "m", chat_fn=FakeChat('{"signals": [', finish="length"))
                      .analyze(item(), pre("AAPL"), True, NOW))
    assert not res.ok and res.status == "truncated"
    res = asyncio.run(ProAIEngine(FakeServer(), "m", chat_fn=FakeChat("")).analyze(item(), pre("AAPL"), True, NOW))
    assert not res.ok and res.status == "error" and "empty" in res.error


def test_no_more_questions_at_once_than_slots():
    fake = FakeChat(delay=0.05)
    engine = ProAIEngine(FakeServer(), "m", slots=2, chat_fn=fake)

    async def go():
        return await asyncio.gather(*(engine.analyze(item(), pre("AAPL"), True, NOW) for _ in range(6)))

    assert all(r.ok for r in asyncio.run(go()))
    assert fake.peak == 2


# ------------------------------------------------------------------------------------------------ self-test
def test_samples_are_labelled():
    samples = selftest.load_samples()
    assert len(samples) == 20
    for s in samples:
        assert s["expect"] and s["candidates"]
        assert set(s["expect"]) <= {c["symbol"] for c in s["candidates"]} | {"GOOGL", "MCHI"}


def test_judge():
    s = {"expect": {"AAPL": "bearish", "GS": "neutral"}, "absent": ["SPY"]}
    allowed = ["AAPL", "GS"]
    ok, right, got = selftest.judge(s, json.dumps({"signals": [sig("AAPL", "bearish")]}), allowed)
    assert ok and right and got == {"AAPL": "bearish"}
    assert selftest.judge(s, json.dumps({"signals": [sig("AAPL", "bullish")]}), allowed)[:2] == (True, False)
    assert selftest.judge(s, json.dumps({"signals": [sig("MSFT", "bearish")]}), allowed)[0] is False
    assert selftest.judge(s, "not json", allowed) == (False, False, {})
    assert selftest.judge({"expect": {"GOOGL": "bullish"}}, json.dumps({"signals": [sig("GOOG", "bullish")]}),
                          ["GOOG"])[1]


def test_run_selftest_with_a_perfect_model():
    samples = selftest.load_samples()
    by_title = {s["title"]: s for s in samples}

    async def perfect(url, key, system, user, schema, **kw):
        title = next(t for t in by_title if t in user)
        signals = [sig(sym, label) for sym, label in by_title[title]["expect"].items() if label != "neutral"]
        allowed = schema["properties"]["signals"]["items"]["properties"]["ticker"]["enum"]
        signals = [s for s in signals if s["ticker"] in allowed]
        return ChatResult(text=json.dumps({"signals": signals}), finish_reason="stop", latency_ms=5)

    seen = []
    engine = ProAIEngine(FakeServer(), "m", chat_fn=perfect)
    out = asyncio.run(selftest.run_selftest(engine, progress_cb=lambda i, n: seen.append((i, n))))
    assert out["total"] == 20 and out["json_ok"] == 20
    assert out["right"] == 20, [r for r in out["items"] if not r["right"]]
    assert seen[-1] == (20, 20) and out["avg_seconds"] >= 0


def test_selftest_counts_failures():
    engine = ProAIEngine(FakeServer(), "m", chat_fn=FakeChat(error=ChatError("down", kind="unreachable")))
    out = asyncio.run(selftest.run_selftest(engine, samples=selftest.load_samples()[:3], warm_up=False))
    assert out["total"] == 3 and out["json_ok"] == 0 and out["right"] == 0
    assert all(r["error"] for r in out["items"])


def test_client_module_has_no_model_names():
    import inspect

    from newstrader.llm import engine as engine_mod

    for mod in (llm_client, engine_mod, selftest):
        text = inspect.getsource(mod).lower()
        assert "qwen" not in text and "gemma" not in text
