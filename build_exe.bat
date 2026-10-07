@echo off
rem ============================================================
rem  Build NewsTrader.exe (a normal Windows app folder).
rem  Run run.bat at least once first - this uses the same .venv.
rem ============================================================
setlocal EnableExtensions
title Build NewsTrader.exe
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" goto :novenv

echo.
echo  Installing the build tool (PyInstaller)...
".venv\Scripts\python.exe" -m pip install --upgrade pyinstaller
if errorlevel 1 goto :fail

echo.
echo  Building... this takes 3-10 minutes.
".venv\Scripts\python.exe" -m PyInstaller --noconfirm --clean newstrader.spec
if errorlevel 1 goto :fail

echo.
echo  Done!  Your app is:  dist\NewsTrader\NewsTrader.exe
echo  Keep the whole dist\NewsTrader folder together - the exe needs the _internal folder beside it.
echo  Tip: right-click NewsTrader.exe, choose Send to, then Desktop - create shortcut.
echo  The exe keeps its own settings and keys in %LOCALAPPDATA%\NewsTrader
echo  - or copy your .env file next to NewsTrader.exe to reuse the same keys.
explorer "dist\NewsTrader"
pause
exit /b 0

:novenv
echo  [X] Run run.bat once first - it sets up the Python environment this build uses.
pause
exit /b 1

:fail
echo.
echo  [X] Build failed - see the messages above.
pause
exit /b 1
