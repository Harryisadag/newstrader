#!/bin/bash
# ============================================================
#  Build NewsTrader.app (a normal Mac app).
#  Run run.command at least once first - this uses the same .venv.
# ============================================================
cd "$(dirname "$0")" || exit 1
printf '\033]0;Build NewsTrader.app\007'

pause_and_exit() {
  echo
  read -r -p "  Press Enter to close this window..." _
  exit "${1:-0}"
}

if [ ! -x .venv/bin/python ]; then
  echo "  [X] Run run.command once first - it sets up the Python environment this build uses."
  pause_and_exit 1
fi

echo
echo "  Installing the build tool (PyInstaller)..."
.venv/bin/python -m pip install --upgrade pyinstaller pillow || pause_and_exit 1

echo
echo "  Building... this takes 3-10 minutes."
if ! .venv/bin/python -m PyInstaller --noconfirm --clean newstrader.spec; then
  echo
  echo "  [X] Build failed - see the messages above."
  pause_and_exit 1
fi

echo
echo "  Done!  Your app is:  dist/NewsTrader.app"
echo "  Drag it into your Applications folder. The first time you open it, macOS may block it because it"
echo "  isn't from the App Store: open System Settings -> Privacy & Security, scroll down, click 'Open Anyway'."
echo "  The app keeps its settings and keys in ~/Library/Application Support/NewsTrader"
open dist
pause_and_exit 0
