@echo off
setlocal
REM ============================================================
REM  Fork switchover - point a Hermes install at bobaba76/hermes-agent
REM  Usage:   fork-switchover.cmd  [path-to-hermes-agent-checkout]
REM  Default: %LOCALAPPDATA%\hermes\hermes-agent
REM  Run this with Hermes CLOSED. Safe to re-run. Keeps your
REM  data (memories, config) untouched - it only changes code.
REM ============================================================

set "REPO=%~1"
if "%REPO%"=="" set "REPO=%LOCALAPPDATA%\hermes\hermes-agent"

if not exist "%REPO%\.git" (
  echo ERROR: no git repo found at "%REPO%"
  pause
  exit /b 1
)

cd /d "%REPO%"

echo.
echo [1/3] Fetching the fork ...
git fetch origin
if errorlevel 1 ( echo FETCH FAILED - check internet & pause & exit /b 1 )

echo.
echo [2/3] Switching checkout to fork main ...
git reset --hard origin/main
if errorlevel 1 ( echo RESET FAILED & pause & exit /b 1 )

echo.
echo Now on:
git log -1 --oneline

echo.
echo [3/3] Installing dependencies (a few minutes) ...
set "PYTHONPATH="
venv\Scripts\python.exe -m pip install -e ".[all]"
if errorlevel 1 (
  echo (base install fallback)
  venv\Scripts\python.exe -m pip install -e .
  if errorlevel 1 ( echo DEP INSTALL FAILED - paste this output to Hermes & pause & exit /b 1 )
)

echo.
echo ============================================================
echo  DONE. You can start Hermes again.
echo  If anything looks wrong, paste the output above to Hermes.
echo ============================================================
pause
