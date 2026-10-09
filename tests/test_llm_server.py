"""Pro AI's llama-server manager, run against a small fake server script (no real model needed)."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
from datetime import UTC, datetime

import pytest

from newstrader.ai.prefilter import Candidate, PrefilterResult
from newstrader.llm import server
from newstrader.llm.engine import ProAIEngine
from newstrader.llm.server import LlamaServer, explain_failure, parse_help_flags
from newstrader.sources.base import NewsItem

# lines copied from llama-server --help (b11512)
REAL_HELP = """
-c,    --ctx-size N                     size of the prompt context (default: 0, 0 = loaded from model)
-ngl,  --gpu-layers, --n-gpu-layers N   max. number of layers to store in VRAM, either an exact number,
-fit,  --fit [on|off]                   whether to adjust unset arguments to fit in device memory ('on' or
-fitt, --fit-target MiB0,MiB1,MiB2,...
-lv,   --verbosity, --log-verbosity N   Set the verbosity threshold. Messages with a higher verbosity will be
--log-prefix, --no-log-prefix           Enable prefix in log messages
-cram, --cache-ram N                    set the maximum cache size in MiB (default: 8192, -1 - no limit, 0 -
--cache-idle-slots, --no-cache-idle-slots
-np,   --parallel N                     number of server slots (default: -1, -1 = auto)
--port PORT                             port to listen (default: 8080)
--ui,  --webui, --no-ui, --no-webui     whether to enable the Web UI (default: enabled)
--api-key KEY                           API key to use for authentication, multiple keys can be provided as a
--jinja, --no-jinja                     whether to use jinja template engine for chat (default: enabled)
-rea,  --reasoning [on|off|auto]        Use reasoning/thinking in the chat ('on', 'off', or 'auto', default:
--reasoning-budget N                    token budget for thinking: -1 for unrestricted, 0 for immediate end,
--offline                               Offline mode: forces use of cache, prevents network access
--sleep-idle-seconds SECONDS            number of seconds of idleness after which the server will sleep
                                        e.g. a multi-line note about non-flags, -1 = disabled
"""

FAKE = r'''
import json, os, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HELP = """
-c,    --ctx-size N
-np,   --parallel N
--jinja, --no-jinja
-ngl,  --gpu-layers, --n-gpu-layers N
--host HOST
--port PORT
--api-key KEY
--log-prefix, --no-log-prefix
-lv,   --verbosity, --log-verbosity N
"""
args = sys.argv[1:]
mode = os.environ.get("FAKE_MODE", "ok")
if "--help" in args:
    print(HELP)
    sys.exit(0)


def opt(name):
    return args[args.index(name) + 1]


port, key = int(opt("--port")), opt("--api-key")
print("0.00.001.000 I srv  llama_server: initializing", flush=True)
if mode == "oom":
    print("0.00.010.000 E ggml_backend_cuda_buffer_type_alloc_buffer: allocating 9000.00 MiB on device 0: "
          "cudaMalloc failed: out of memory", flush=True)
    sys.exit(1)
layers = "20/33" if mode == "partial" else "33/33"
print(f"0.00.020.000 I load_tensors: offloaded {layers} layers to GPU", flush=True)
print(f"0.00.021.000 W srv  echo: key was {key}", flush=True)
polls = {"n": 0}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/health":
            polls["n"] += 1
            if mode == "never" or polls["n"] <= 2:
                return self.send(503, {"error": {"message": "Loading model"}})
            return self.send(200, {"status": "ok"})
        self.send(404, {})

    def do_POST(self):
        if self.headers.get("Authorization") != f"Bearer {key}":
            return self.send(401, {"error": {"message": "Invalid API Key"}})
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        enum = body["response_format"]["json_schema"]["schema"]["properties"]["signals"]["items"]["properties"][
            "ticker"]["enum"]
        signal = {"ticker": enum[0], "company": "", "speaker": "Reuters", "bull_case": "Strong demand.",
                  "bear_case": "Maybe priced in.", "direction": "bullish", "confidence": 81,
                  "time_sensitivity": "hours", "reasoning": "Beat estimates."}
        text = "<think>\n\n</think>\n\n" + json.dumps({"signals": [signal]})
        self.send(200, {"choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 900, "completion_tokens": 80}})


ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
'''


@pytest.fixture
def fake(tmp_path):
    exe = tmp_path / "fake-llama-server.py"
    exe.write_text(FAKE, encoding="utf-8")
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    return exe, model


@pytest.fixture
def srv(tmp_path):
    s = LlamaServer(pid_path=tmp_path / "llama-server.pid", launcher=[sys.executable])
    yield s
    s.stop()


def test_parse_help_flags():
    flags = parse_help_flags(REAL_HELP)
    for f in ("-c", "--ctx-size", "-ngl", "--n-gpu-layers", "--fit", "-fitt", "--fit-target", "-lv", "--log-prefix",
              "--cache-ram", "--no-cache-idle-slots", "-np", "--port", "--no-ui", "--no-webui", "--api-key",
              "--jinja", "--reasoning", "-rea", "--reasoning-budget", "--offline", "--sleep-idle-seconds"):
        assert f in flags, f
    assert "-line" not in flags and "-1" not in flags and "--reasoning-" not in flags


def test_build_args_with_every_flag_known():
    args, dropped = LlamaServer.build_args(server.Path("m.gguf"), ["llama-server"], 5000, "KEY", 6144, 8192, 2, 0,
                                           parse_help_flags(REAL_HELP))
    assert dropped == []
    assert args[:9] == ["llama-server", "-m", "m.gguf", "--host", "127.0.0.1", "--port", "5000", "--api-key", "KEY"]
    pairs = {args[i]: args[i + 1] for i in range(9, len(args) - 1)}
    assert pairs["-c"] == "8192" and pairs["-np"] == "2" and pairs["--reasoning-budget"] == "0"
    assert pairs["--reasoning"] == "off" and pairs["--fit"] == "on" and pairs["--fit-target"] == "6144"
    assert pairs["--cache-ram"] == "0" and pairs["-lv"] == "4"
    assert "--jinja" in args and "--no-ui" in args and "--log-prefix" in args and "--offline" in args
    assert "--sleep-idle-seconds" not in args  # only when asked for
    assert "-ngl" not in args  # --fit decides how many layers fit
    args, _ = LlamaServer.build_args(server.Path("m.gguf"), ["s"], 1, "K", 0, 4096, 1, 300, parse_help_flags(REAL_HELP))
    assert args[args.index("--sleep-idle-seconds") + 1] == "300"


def test_build_args_leaves_out_unknown_flags():
    old = {"-c", "--ctx-size", "-np", "--port", "--api-key", "--jinja", "-ngl", "--n-gpu-layers", "--no-webui"}
    args, dropped = LlamaServer.build_args(server.Path("m.gguf"), ["s"], 1, "K", 1024, 4096, 2, 0, old)
    assert {"--reasoning", "--reasoning-budget", "--fit", "--fit-target", "--cache-ram", "--offline",
            "--log-prefix", "-lv"} <= set(dropped)
    assert "--fit" not in args and "--no-webui" in args
    assert args[args.index("-ngl") + 1] == "999"  # no --fit: ask for every layer on the card instead


def test_build_args_without_help_uses_everything():
    args, dropped = LlamaServer.build_args(server.Path("m.gguf"), ["s"], 1, "K", 1024, 4096, 2, 0, None)
    assert dropped == [] and "--fit" in args and "--reasoning-budget" in args


@pytest.mark.parametrize(("build", "line", "on_gpu"), [
    ("cuda13", "0.00.033.948 I load_tensors: offloaded 33/33 layers to GPU", True),
    ("cuda13", "load_tensors: offloaded 20/33 layers to GPU", False),
    ("metal", "0.00.033.948 I load_tensors: offloaded 41/41 layers to GPU", True),
    ("cpu", "0.00.033.948 I load_tensors: offloaded 3/3 layers to GPU", None),  # the CPU build says this too
    ("vulkan", "0.00.033.948 I srv  load_model: loading model", None),
])
def test_offload_line(build, line, on_gpu):
    s = LlamaServer()
    s.build_key = build
    s.handle_log_line(line)
    assert s.fully_on_gpu is on_gpu


def test_log_lines_hide_the_key_and_rate_limit_warnings(caplog):
    s = LlamaServer()
    s.api_key = "SECRET123"
    with caplog.at_level("WARNING", logger="newstrader.llm.server"):
        s.handle_log_line("0.00.001.000 I srv  token SECRET123 in an info line")
        for i in range(8):
            s.handle_log_line(f"0.00.002.000 W srv  something odd {i}")
        s.handle_log_line("0.00.003.000 W slot   operator(): need to evaluate at least 1 token for each slot")
    assert "SECRET123" not in "\n".join(s.logs)
    assert len([r for r in caplog.records if "something odd" in r.getMessage()]) == 5


@pytest.mark.parametrize(("lines", "code", "words"), [
    (["ggml_backend_cuda_buffer_type_alloc_buffer: cudaMalloc failed: out of memory"], 1, "graphics memory"),
    (["llama_model_load: error loading model: tensor data is not within file bounds"], 1, "Delete the model"),
    (["couldn't bind HTTP server socket, hostname: 127.0.0.1, port: 8080"], 1, "port in use"),
    (["error: invalid argument: --bogus"], 1, "Reinstall the server"),
    ([], 3221225781, None),
    (["something else entirely"], 7, "exit code 7"),
])
def test_explain_failure(lines, code, words, monkeypatch):
    if words is None:
        monkeypatch.setattr(server.sys, "platform", "win32")
        words = "missing a system file"
    assert words in explain_failure(lines, code)


# ------------------------------------------------------------------------------------------------ fake server
def test_start_wait_ready_stop(fake, srv, tmp_path):
    exe, model = fake
    srv.start(model, exe, reserve_mib=6144, build_key="cuda13")
    assert srv.state == "starting" and srv.running
    assert srv.wait_ready(timeout=20, poll=0.05)
    assert srv.state == "ready" and srv.fully_on_gpu is True and "whole model" in srv.message
    assert srv.health() == 200
    assert "--reasoning-budget" in srv.dropped_flags and "--fit" in srv.dropped_flags
    assert srv.args[srv.args.index("-ngl") + 1] == "999"
    assert srv.api_key and srv.api_key not in "\n".join(srv.logs)
    info = json.loads((tmp_path / "llama-server.pid").read_text())
    assert info["pid"] == srv.proc.pid and info["port"] == srv.port
    snap = srv.snapshot()
    assert snap["state"] == "ready" and snap["model_file"] == "model.gguf" and snap["ready_seconds"] is not None
    proc = srv.proc
    srv.stop()
    assert proc.poll() is not None and srv.state == "off" and not srv.running
    assert not (tmp_path / "llama-server.pid").exists()


def test_partial_offload_is_reported(fake, srv, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "partial")
    exe, model = fake
    srv.start(model, exe, build_key="cuda13")
    assert srv.wait_ready(timeout=20, poll=0.05)
    assert srv.fully_on_gpu is False and "only 20 of 33 layers" in srv.message


def test_out_of_memory_is_explained(fake, srv, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "oom")
    exe, model = fake
    srv.start(model, exe, build_key="cuda13")
    assert not srv.wait_ready(timeout=20, poll=0.05)
    assert srv.state == "error" and "graphics memory" in srv.message


def test_never_ready_times_out(fake, srv, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "never")
    exe, model = fake
    srv.start(model, exe)
    assert not srv.wait_ready(timeout=1.0, poll=0.05)
    assert srv.state == "error" and "didn't finish loading" in srv.message
    assert not srv.running  # given up on: stopped


def test_missing_files(fake, srv, tmp_path):
    exe, model = fake
    with pytest.raises(FileNotFoundError):
        srv.start(model, tmp_path / "nope.exe")
    assert srv.state == "error" and "reinstall" in srv.message
    with pytest.raises(FileNotFoundError):
        srv.start(tmp_path / "nope.gguf", exe)
    assert "download it again" in srv.message


def test_crash_restarts_once(fake, srv):
    exe, model = fake
    srv.start(model, exe)
    assert srv.wait_ready(timeout=20, poll=0.05)
    srv.proc.kill()
    deadline = time.monotonic() + 10
    while srv.state != "error" and time.monotonic() < deadline:
        time.sleep(0.05)
    assert srv.state == "error" and "stopped unexpectedly" in srv.message
    assert srv.restart_if_crashed()
    assert srv.wait_ready(timeout=20, poll=0.05) and srv.restarts == 1
    srv.proc.kill()
    srv.proc.wait()
    srv._reader.join(5)
    assert not srv.restart_if_crashed()  # only once
    srv.start(model, exe)  # a fresh start resets the count
    assert srv.restarts == 0


def test_engine_against_the_fake_server(fake, srv):
    exe, model = fake
    srv.start(model, exe)
    assert srv.wait_ready(timeout=20, poll=0.05)
    engine = ProAIEngine(srv, "test-model", max_signals=2)
    item = NewsItem(source_id="t", source_type="rss", source_name="Reuters", external_id="1",
                    title="Apple beats estimates", published_at=datetime.now(UTC))
    pre = PrefilterResult(hit=True, candidates=[Candidate("AAPL", "Apple Inc.", "apple")])
    res = asyncio.run(engine.analyze(item, pre, True, datetime.now(UTC)))
    assert res.ok and res.engine == "pro" and res.cost_usd == 0 and res.model == "test-model"
    assert json.loads(res.text)["signals"][0]["ticker"] == "AAPL"
    assert (res.input_tokens, res.output_tokens) == (900, 80)


# ------------------------------------------------------------------------------------------------ stale servers
def _sleeper() -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])


def test_kill_stale_stops_a_leftover_server(tmp_path, monkeypatch):
    proc = _sleeper()
    try:
        pid_path = tmp_path / "llama-server.pid"
        pid_path.write_text(json.dumps({"pid": proc.pid}), encoding="utf-8")
        monkeypatch.setattr(server, "_process_name", lambda pid: "llama-server.exe")
        assert server.kill_stale(pid_path)
        proc.wait(timeout=10)
        assert not pid_path.exists()
    finally:
        proc.kill()


def test_kill_stale_leaves_other_programs_alone(tmp_path, monkeypatch):
    proc = _sleeper()
    try:
        pid_path = tmp_path / "llama-server.pid"
        pid_path.write_text(json.dumps({"pid": proc.pid}), encoding="utf-8")
        monkeypatch.setattr(server, "_process_name", lambda pid: "python.exe")  # the pid was reused
        assert not server.kill_stale(pid_path)
        assert proc.poll() is None and not pid_path.exists()
    finally:
        proc.kill()


def test_kill_stale_with_a_broken_pid_file(tmp_path):
    pid_path = tmp_path / "llama-server.pid"
    pid_path.write_text("not json", encoding="utf-8")
    assert not server.kill_stale(pid_path)
    assert not pid_path.exists()
    assert not server.kill_stale(tmp_path / "missing.pid")
