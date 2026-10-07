"""Start NewsTrader headless, check the API answers, then stop it. Used by CI on Windows + Linux."""

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
    tmp = tempfile.mkdtemp(prefix="nt-smoke-")
    env = {**os.environ, "NEWSTRADER_DATA_DIR": tmp, "NEWSTRADER_ENV_FILE": os.path.join(tmp, ".env")}
    proc = subprocess.Popen([sys.executable, "-m", "newstrader", "--headless", "--port", str(PORT), "--token", TOKEN],
                            env=env)
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
