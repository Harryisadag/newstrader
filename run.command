#!/bin/bash
# ============================================================
#  NewsTrader for Mac - double-click to start.
#  First run: creates a private Python environment (.venv) and
#  installs everything (5-10 minutes). Later runs start fast.
#  Extra options are passed through, e.g.:  ./run.command --browser
# ============================================================
cd "$(dirname "$0")" || exit 1
printf '\033]0;NewsTrader\007'

pause_and_exit() {
  echo
  read -r -p "  Press Enter to close this window..." _
  exit "${1:-1}"
}

VENV_PY=".venv/bin/python"

if [ ! -x "$VENV_PY" ]; then
  echo
  echo "  Setting up NewsTrader for the first time..."
  echo
  # ---- find Python 3.12 (or 3.13 / 3.11): Homebrew, python.org, then PATH ----
  PY=""
  for v in 3.12 3.13 3.11; do
    for c in "/opt/homebrew/bin/python$v" "/usr/local/bin/python$v" \
             "/Library/Frameworks/Python.framework/Versions/$v/bin/python$v" "python$v"; do
      if command -v "$c" >/dev/null 2>&1; then PY="$c"; break 2; fi
    done
  done
  if [ -z "$PY" ]; then
    echo "  [X] Python 3.12 was not found."
    echo "      Install it from https://www.python.org/downloads/macos/ (the 'macOS 64-bit universal2 installer'),"
    echo "      then double-click run.command again."
    pause_and_exit 1
  fi
  echo "  Using: $PY ($("$PY" --version 2>&1))"
  if [ "$(sysctl -n hw.optional.arm64 2>/dev/null)" = "1" ] && \
     [ "$("$PY" -c 'import platform; print(platform.machine())')" != "arm64" ]; then
    echo "  [!] This Python runs in Intel (Rosetta) mode on an Apple Silicon Mac - speech-to-text can't use"
    echo "      the Apple GPU. Install the python.org 'universal2' Python to fix that. Continuing anyway."
  fi
  "$PY" -m venv .venv || { echo "  [X] Could not create the Python environment in .venv"; pause_and_exit 1; }
fi

# ---- Apple Silicon + macOS 14 or newer can transcribe on the Apple GPU (MLX) ----
APPLE_GPU=0
if [ "$("$VENV_PY" -c 'import platform; print(platform.machine())')" = "arm64" ]; then
  MACOS_MAJOR="$(sw_vers -productVersion | cut -d. -f1)"
  if [ "${MACOS_MAJOR:-0}" -ge 14 ]; then APPLE_GPU=1; fi
fi

# ---- install / update packages whenever the requirement files change ----
WANT_HASH="$(cat requirements.txt requirements-mac-gpu.txt 2>/dev/null | shasum -a 256 | cut -d' ' -f1)-$APPLE_GPU"
HAVE_HASH="$(cat .venv/requirements.sha256 2>/dev/null)"
if [ "$WANT_HASH" != "$HAVE_HASH" ]; then
  echo
  echo "  Installing / updating packages. This can take several minutes the first time..."
  echo
  "$VENV_PY" -m pip install --upgrade pip
  if ! "$VENV_PY" -m pip install -r requirements.txt; then
    echo
    echo "  [X] Package install failed. Check your internet connection and try again."
    echo "      If it keeps failing, copy the error text above and ask for help."
    pause_and_exit 1
  fi
  if [ "$APPLE_GPU" = "1" ]; then
    echo
    echo "  Installing the Apple-GPU speech engine (mlx-whisper)..."
    # mlx-whisper lists PyTorch as a dependency but never uses it at runtime - skip that 130 MB download
    if "$VENV_PY" -m pip install -r requirements-mac-gpu.txt && \
       "$VENV_PY" -m pip install --no-deps "mlx-whisper==0.4.3"; then
      echo "  Apple-GPU speech engine installed."
    else
      echo "  [!] Couldn't install it - speech-to-text will use the CPU instead (slower). Everything else works."
    fi
  fi
  echo "$WANT_HASH" > .venv/requirements.sha256
fi

# ---- create .env from the template if it's missing ----
[ -f .env ] || cp .env.example .env

echo
echo "  Starting NewsTrader... (this window shows the engine log - closing it quits the app)"
echo
if ! "$VENV_PY" -m newstrader "$@"; then
  echo
  echo "  NewsTrader stopped with an error. Details are in data/logs/newstrader.log"
  pause_and_exit 1
fi
