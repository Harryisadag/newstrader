"""Start NewsTrader headless, check the API answers, then stop it. Used by CI on Windows + Linux.

    python scripts/smoke_test.py                      # run from source
    python scripts/smoke_test.py dist/NewsTrader/NewsTrader.exe --tools   # test a built exe (+ ffmpeg/deno/yt-dlp)
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
                    gpu = checks["GPU (CUDA)"]
                    print(f"  info  GPU: {gpu['detail'][:150]}")
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
