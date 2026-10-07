@echo off
rem ============================================================
rem  Update NewsTrader: latest code (if this is a git checkout)
rem  and the latest packages - especially yt-dlp, which needs
rem  updating whenever YouTube changes something.
rem ============================================================
setlocal EnableExtensions
title Update NewsTrader
cd /d "%~dp0"

where git >nul 2>nul
if errorlevel 1 goto :nogit
if not exist ".git" goto :nogit
echo  Getting the latest version from GitHub...
git pull
if errorlevel 1 echo  [!] git pull failed - you may have changed files locally. Continuing with package updates.
goto :packages

:nogit
echo  This folder isn't a git checkout, so the code can't update itself.
echo  For a newer version: download the ZIP from GitHub again and unzip it over this folder.
echo  Your .env and data folder are kept - they aren't in the ZIP.

:packages
if not exist ".venv\Scripts\python.exe" goto :novenv
echo.
echo  Updating packages...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install --upgrade -r requirements.txt
if errorlevel 1 goto :fail
".venv\Scripts\python.exe" -m pip install --upgrade "yt-dlp[default]"
".venv\Scripts\python.exe" -c "import hashlib,pathlib; pathlib.Path('.venv/requirements.sha256').write_text(hashlib.sha256(pathlib.Path('requirements.txt').read_bytes()).hexdigest())"
echo.
echo  Done. Start NewsTrader with run.bat - or run build_exe.bat again if you use the .exe.
pause
exit /b 0

:novenv
echo  [X] Run run.bat first.
pause
exit /b 1

:fail
echo  [X] Update failed - check your internet connection and try again.
pause
exit /b 1
