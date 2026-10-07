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
ARM_MAC=0
[ "$(sysctl -n hw.optional.arm64 2>/dev/null)" = "1" ] && ARM_MAC=1

py_arch() { "$1" -c 'import platform; print(platform.machine())' 2>/dev/null; }

# Find Python 3.12 (or 3.13 / 3.11; 3.14 only on Apple Silicon). On an M-series Mac prefer one that runs
# natively: an Intel (Rosetta) Python can't use the Apple GPU. Homebrew in /usr/local is usually Intel.
find_python() {
  PY=""
  local fallback="" versions="3.12 3.13 3.11" v c
  [ "$ARM_MAC" = "1" ] && versions="3.12 3.13 3.14 3.11"
  for v in $versions; do
    for c in "/opt/homebrew/bin/python$v" "/Library/Frameworks/Python.framework/Versions/$v/bin/python$v" \
             "/usr/local/bin/python$v" "python$v"; do
      command -v "$c" >/dev/null 2>&1 || continue
      if [ "$ARM_MAC" = "1" ] && [ "$(py_arch "$c")" != "arm64" ]; then
        [ -z "$fallback" ] && fallback="$c"
        continue
      fi
      PY="$c"
      return 0
    done
  done
  PY="$fallback"
  [ -n "$PY" ]
}

# An environment made earlier with an Intel-only Python on an M-series Mac: rebuild it with a native one.
if [ -x "$VENV_PY" ] && [ "$ARM_MAC" = "1" ] && [ "$(py_arch "$VENV_PY")" != "arm64" ]; then
  if find_python && [ "$(py_arch "$PY")" = "arm64" ]; then
    echo "  Rebuilding the Python environment to run natively on Apple Silicon..."
    rm -rf .venv
  else
    echo "  [!] Python runs in Intel (Rosetta) mode on this Apple Silicon Mac, so speech-to-text can't use the"
    echo "      Apple GPU. Install Python 3.12 from python.org (the 'universal2' installer) to fix that."
  fi
fi

if [ ! -x "$VENV_PY" ]; then
  echo
  echo "  Setting up NewsTrader for the first time..."
  echo
  if ! find_python; then
    echo "  [X] Python 3.12 was not found."
    echo "      Install it from https://www.python.org/downloads/macos/ - look for Python 3.12 and its"
    echo "      'macOS 64-bit universal2 installer' - then double-click run.command again."
    pause_and_exit 1
  fi
  echo "  Using: $PY ($("$PY" --version 2>&1))"
  if [ "$ARM_MAC" = "1" ] && [ "$(py_arch "$PY")" != "arm64" ]; then
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
  INSTALLED_OK=1
  if [ "$APPLE_GPU" = "1" ]; then
    echo
    echo "  Installing the Apple-GPU speech engine (mlx-whisper)..."
    # mlx-whisper lists PyTorch as a dependency but never uses it at runtime - skip that 130 MB download
    if "$VENV_PY" -m pip install -r requirements-mac-gpu.txt && \
       "$VENV_PY" -m pip install --no-deps "mlx-whisper==0.4.3"; then
      echo "  Apple-GPU speech engine installed."
    else
      INSTALLED_OK=0
      echo "  [!] Couldn't install it - speech-to-text will use the CPU for now (slower). Everything else works."
      echo "      It will try again the next time you start run.command."
    fi
  fi
  # only remember the install as complete when every part worked, so a failed part is retried next time
  [ "$INSTALLED_OK" = "1" ] && echo "$WANT_HASH" > .venv/requirements.sha256
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
