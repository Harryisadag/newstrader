#!/bin/bash
# ============================================================
#  Update NewsTrader (Mac): latest code (if this is a git checkout)
#  and the latest packages - especially yt-dlp, which needs
#  updating whenever YouTube changes something.
# ============================================================
cd "$(dirname "$0")" || exit 1
printf '\033]0;Update NewsTrader\007'

pause_and_exit() {
  echo
  read -r -p "  Press Enter to close this window..." _
  exit "${1:-0}"
}

if command -v git >/dev/null 2>&1 && [ -d .git ]; then
  echo "  Getting the latest version from GitHub..."
  git pull || echo "  [!] git pull failed - you may have changed files locally. Continuing with package updates."
else
  echo "  This folder isn't a git checkout, so the code can't update itself."
  echo "  For a newer version: download the ZIP from GitHub again and unzip it over this folder."
  echo "  Your .env and data folder are kept - they aren't in the ZIP."
fi

if [ ! -x .venv/bin/python ]; then
  echo "  [X] Run run.command first."
  pause_and_exit 1
fi

echo
echo "  Updating packages..."
.venv/bin/python -m pip install --upgrade pip
if ! .venv/bin/python -m pip install --upgrade -r requirements.txt; then
  echo "  [X] Update failed - check your internet connection and try again."
  pause_and_exit 1
fi
.venv/bin/python -m pip install --upgrade "yt-dlp[default]"
# make run.command re-check the Apple-GPU engine and record the new package state
rm -f .venv/requirements.sha256
echo
echo "  Done. Start NewsTrader with run.command - or run build_app.command again if you use the app."
pause_and_exit 0
