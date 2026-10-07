@echo off
rem ============================================================
rem  NewsTrader - start in dev mode
rem  First run: creates a private Python environment (.venv) and
rem  installs everything (5-10 minutes). Later runs start fast.
rem  Extra options are passed through, e.g.:  run.bat --browser
rem ============================================================
setlocal EnableExtensions
title NewsTrader
cd /d "%~dp0"

set "VENV_PY=%~dp0.venv\Scripts\python.exe"

if exist "%VENV_PY%" goto :have_venv

echo.
echo  Setting up NewsTrader for the first time...
echo.

rem ---- find Python 3.12 (or 3.13 / 3.11) ----
set "PY="
where py >nul 2>nul
if errorlevel 1 goto :try_python
for %%V in (3.12 3.13 3.11) do (
    if not defined PY (
        py -%%V -c "import sys" >nul 2>nul
        if not errorlevel 1 set "PY=py -%%V"
    )
)

:try_python
if defined PY goto :make_venv
where python >nul 2>nul
if errorlevel 1 goto :no_python
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>nul
if errorlevel 1 goto :no_python
set "PY=python"
goto :make_venv

:no_python
echo  [X] Python 3.12 was not found.
echo      Install it from https://www.python.org/downloads/
echo      and tick "Add python.exe to PATH" on the first installer screen.
echo      Then double-click run.bat again.
echo.
pause
exit /b 1

:make_venv
echo  Using: %PY%
%PY% -m venv .venv
if errorlevel 1 goto :venv_failed
goto :have_venv

:venv_failed
echo  [X] Could not create the Python environment in .venv
pause
exit /b 1

:have_venv
rem ---- install / update packages whenever requirements.txt changes ----
"%VENV_PY%" -c "import hashlib,pathlib,sys; h=hashlib.sha256(pathlib.Path('requirements.txt').read_bytes()).hexdigest(); p=pathlib.Path('.venv/requirements.sha256'); sys.exit(0 if p.exists() and p.read_text().strip()==h else 1)"
if not errorlevel 1 goto :deps_ok

echo.
echo  Installing / updating packages. This can take several minutes the first time...
echo.
"%VENV_PY%" -m pip install --upgrade pip
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 goto :pip_failed
"%VENV_PY%" -c "import hashlib,pathlib; pathlib.Path('.venv/requirements.sha256').write_text(hashlib.sha256(pathlib.Path('requirements.txt').read_bytes()).hexdigest())"
goto :deps_ok

:pip_failed
echo.
echo  [X] Package install failed. Check your internet connection and try again.
echo      If it keeps failing, copy the error text above and ask for help.
pause
exit /b 1

:deps_ok
rem ---- create .env from the template if it's missing ----
if not exist ".env" copy ".env.example" ".env" >nul

echo.
echo  Starting NewsTrader... (this window shows the engine log - closing it quits the app)
echo.
"%VENV_PY%" -m newstrader %*
if errorlevel 1 goto :crashed
goto :eof

:crashed
echo.
echo  NewsTrader stopped with an error. Details are in data\logs\newstrader.log
pause
exit /b 1
