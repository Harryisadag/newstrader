"""Start NewsTrader headless, check the API answers, then stop it. Used by CI on Windows, Mac and Linux.

    python scripts/smoke_test.py                      # run from source
    python scripts/smoke_test.py dist/NewsTrader/NewsTrader.exe --tools   # test a built exe (+ ffmpeg/deno/yt-dlp)
    python scripts/smoke_test.py dist/NewsTrader.app/Contents/MacOS/NewsTrader --tools --ml   # built Mac app,
                                                      # and wait for FinBERT to download + load (needs internet)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

PORT = 8799
TOKEN = "smoke-test-token"


def wait_for_finbert(timeout: float = 300) -> int:
    """The engine downloads + loads FinBERT in the background on first start; wait until it's loaded."""
    deadline = time.time() + timeout
    detail = ""
    while time.time() < deadline:
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/api/ml/status", headers={"X-NT-Token": TOKEN})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                st = json.loads(r.read())
            detail = f"{st['sentiment']} (model id {st['sentiment_model_id']}, error {st['sentiment_error']})"
            if st["sentiment_model_id"].startswith("finbert"):
                print("  ok    Local ML engine:", detail)
                return 0
            if st["sentiment_error"]:
                break
        except Exception as exc:
            detail = str(exc)
        time.sleep(3)
    print("smoke test FAILED: FinBERT didn't load:", detail)
    return 1


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    check_tools = "--tools" in sys.argv
    tmp = tempfile.mkdtemp(prefix="nt-smoke-")
    env = {**os.environ, "NEWSTRADER_DATA_DIR": tmp, "NEWSTRADER_ENV_FILE": os.path.join(tmp, ".env")}
    cmd = [args[0]] if args else [sys.executable, "-m", "newstrader"]
    proc = subprocess.Popen(cmd + ["--headless", "--port", str(PORT), "--token", TOKEN], env=env)
    try:
        deadline = time.time() + 90
        last_err = None
        while time.time() < deadline:
            try:
                req = urllib.request.Request(f"http://127.0.0.1:{PORT}/api/status", headers={"X-NT-Token": TOKEN})
                with urllib.request.urlopen(req, timeout=5) as r:
                    body = json.loads(r.read())
                assert body["mode"] == "paper", body
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/", timeout=5) as r:
                    assert b"NewsTrader" in r.read()
                print("smoke test OK:", {k: body[k] for k in ("version", "mode")})
                if check_tools:
                    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/api/diagnostics", data=b"{}", method="POST",
                                                 headers={"X-NT-Token": TOKEN, "Content-Type": "application/json"})
                    with urllib.request.urlopen(req, timeout=60) as r:
                        checks = {c["name"]: c for c in json.loads(r.read())["checks"]}
                    for name in ("ffmpeg", "yt-dlp", "Deno (YouTube JavaScript runtime)", "Data folder"):
                        c = checks[name]
                        print(f"  {c['status']:5} {name}: {c['detail'][:100]}")
                        if c["status"] != "ok":
                            print("smoke test FAILED: bundled tool missing")
                            return 1
                    gpu = checks.get("GPU (CUDA)") or checks.get("Speech-to-text hardware") or {"detail": "?"}
                    print(f"  info  GPU: {gpu['detail'][:150]}")
                if "--ml" in sys.argv:
                    return wait_for_finbert()
                return 0
            except Exception as exc:  # server still starting
                last_err = exc
                time.sleep(1)
        print("smoke test FAILED:", last_err)
        return 1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
