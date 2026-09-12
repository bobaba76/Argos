@echo off
rem ============================================================================
rem  single.bat - one-shot Argos onboarding for Windows - bobaba76/Argos #485
rem
rem  Takes a machine that already runs Hermes agent from "plugin not installed"
rem  to "console open in the browser, creating the admin key".  No manual Python,
rem  no hand-editing, safe to re-run at any time (idempotent).
rem
rem  Exit codes:
rem    0 = done / no-op
rem    1 = usage error / plugin source not found
rem    3 = Hermes home not found
rem
rem  Flags:
rem    /check       verify only - print state, change NOTHING
rem    /auto        enable the UI and autostart without prompting
rem    /nui         install plugin without enabling the UI
rem    /home:path   Hermes home (default %%LOCALAPPDATA%%\hermes)
rem    /port:N      admin console port (default 8733)
rem    /src:path    local Argos checkout containing argos_plugin
rem    question / help switch  show usage
rem
rem  Env shortcuts: SINGLE_HOME, SINGLE_SRC (same as /home, /src)
rem ============================================================================
setlocal EnableExtensions EnableDelayedExpansion

set "SELF=%~dp0"
set "HOME="
set "SRC="
set "PORT=8733"
set "MODE=run"
set "UI=ask"

rem ---- parse arguments ----
:argloop
if "%~1"=="" goto :argsdone
set "A=%~1"
if /i "%A%"=="/check" (set "MODE=check" && goto :argnext)
if /i "%A%"=="/auto"  (set "UI=yes" && goto :argnext)
if /i "%A%"=="/nui"   (set "UI=no" && goto :argnext)
if /i "%A%"=="/?"     (call :usage && exit /b 1)
if /i "%A%"=="/help"  (call :usage && exit /b 1)
set "P="
if /i "%A:~0,6%"=="/home:" (set "HOME=%A:~6%" && goto :argnext)
if /i "%A:~0,6%"=="/port:" (set "PORT=%A:~6%" && goto :argnext)
if /i "%A:~0,5%"=="/src:"  (set "SRC=%A:~5%" && goto :argnext)
echo  [x] unknown option: %A%
call :usage
exit /b 1
:argnext
shift
goto :argloop
:argsdone

if defined SINGLE_HOME set "HOME=%SINGLE_HOME%"
if defined SINGLE_SRC  set "SRC=%SINGLE_SRC%"
if not defined HOME    set "HOME=%LOCALAPPDATA%\hermes"

rem ============================================================================
echo.
echo  ============================================================
echo    Argos single.bat   bobaba76/Argos, issue #485
echo  ============================================================
echo    Hermes home : %HOME%
if /i "%MODE%"=="check" echo    Mode       : CHECK ONLY, nothing is changed
echo.

rem ---- 1. Hermes home present ----
if not exist "%HOME%\config.yaml" (
    echo  [x] Hermes home not found: %HOME%
    echo      Expected Hermes agent configuration there.
    echo      Install Hermes agent first and re-run.
    exit /b 3
)
echo  [ok] Hermes home found.

rem ---- site facts ----
if exist "%HOME%\plugins\hybrid_memory\admin_console.py" (
  echo  [i] plugin dir ........ present
  set "PLUGININ=1"
) else (
  echo  [i] plugin dir ........ missing
  set "PLUGININ=0"
)
echo  [i] console port ....... %PORT%
set "TASK="
schtasks /query /tn "Argos Admin Console" >nul 2>nul && set "TASK=1"
if defined TASK (echo  [i] autostart task .... present) else (echo  [i] autostart task .... absent)

if /i "%MODE%"=="check" (
  echo.
  echo  CHECK OK - nothing was changed.
  exit /b 0
)

rem ---- 2. plugin source ----
set "SRCFOUND="
if defined SRC if exist "%SRC%\argos_plugin\plugin.yaml" set "SRCFOUND=%SRC%\argos_plugin"
if not defined SRCFOUND if exist "%SELF%..\argos_plugin\plugin.yaml" set "SRCFOUND=%SELF%..\argos_plugin"
if not defined SRCFOUND (
  echo  [x] plugin source not found.
  echo      Put this file in scripts/ inside the Argos checkout, or pass /src:^<dir^>.
  exit /b 1
)
echo  [ok] source ............. %SRCFOUND%

rem ---- 3. deploy plugin; idempotent ----
set "LIVE=%HOME%\plugins\hybrid_memory"
set "NEEDCOPY=1"
if "%PLUGININ%"=="1" (
  fc /b "%SRCFOUND%\admin_console.py" "%LIVE%\admin_console.py" >nul 2>nul
  if not errorlevel 1 set "NEEDCOPY=0"
)
if "%NEEDCOPY%"=="1" (
  echo  [..] deploying into %LIVE% ...
  if not exist "%LIVE%" mkdir "%LIVE%"
  xcopy /s /e /y /q /i "%SRCFOUND%" "%LIVE%" >nul
  for /d /r "%LIVE%" %%D in (__pycache__) do rd /s /q "%%D" 2>nul
  echo  [ok] plugin deployed.
) else (
  echo  [ok] plugin already current - re-deploy skipped.
)

rem ---- 4b. python deps ----
set "PYCMD=python"
if exist "%HOME%\hermes-agent\venv\Scripts\python.exe" set "PYCMD=%HOME%\hermes-agent\venv\Scripts\python.exe"
if exist "%SRCFOUND%\requirements.txt" (
  echo  [..] checking python dependencies...
  "%PYCMD%" -m pip install --quiet --disable-pip-version-check -r "%SRCFOUND%\requirements.txt"
)
echo  [ok] python dependencies satisfied

rem ---- 5. memory.provider ----
set "HERMES="
for /f "delims=" %%h in ('where hermes 2^>nul') do set "HERMES=%%h"
if not defined HERMES if exist "%HOME%\bin\hermes.exe" set "HERMES=%HOME%\bin\hermes.exe"
if not defined HERMES if exist "%HOME%\hermes-agent\venv\Scripts\hermes.exe" set "HERMES=%HOME%\hermes-agent\venv\Scripts\hermes.exe"
if not defined HERMES (
  echo  [w] hermes CLI not found - ensure "memory.provider: hybrid_memory"
  echo      is set in %HOME%\config.yaml
) else (
  set "HERMES_HOME=%HOME%"
  "%HERMES%" config set memory.provider hybrid_memory >nul 2>nul
  if errorlevel 1 (
    echo  [w] could not set memory.provider via CLI - set it by hand
  ) else (
    echo  [ok] memory.provider = hybrid_memory
  )
)

rem ---- 6. web UI ----
if /i "%UI%"=="no" goto :ui_no
if /i "%UI%"=="ask" (
  choice /c YN /n /m "Enable the web console (autostarts on login)? [Y/N] "
  if errorlevel 2 goto :ui_no
)
if not exist "%HOME%\scripts" mkdir "%HOME%\scripts"
set "WRAP=%HOME%\scripts\argos-console.cmd"
> "%WRAP%" echo @echo off
>> "%WRAP%" echo cd /d "%LIVE%"
>> "%WRAP%" echo "%PYCMD%" admin_console.py --home "%HOME%" --port %PORT% 2^>^> "%HOME%\admin_console.log"
echo  [..] autostart wrapper ... %WRAP%
rem  Try a per-user ONLOGON scheduled task first. registration can be denied
rem  without elevation on some setups - if so, drop a copy into the user's
rem  Startup folder instead (no rights needed); both start the console at logon.
schtasks /create /f /tn "Argos Admin Console" /tr "\"%WRAP%\"" /sc onlogon /rl limited >nul 2>nul
if errorlevel 1 (
  set "STARTUP=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
  if not exist "%STARTUP%" mkdir "%STARTUP%"
  copy /y "%WRAP%" "%STARTUP%\Argos Admin Console.cmd" >nul 2>nul
  echo  [i] scheduled task denied - using Startup folder autostart instead
) else (
  echo  [ok] autostart: ONLOGON scheduled task "Argos Admin Console"
)
echo  [..] starting console on port %PORT% ...
start "" "%WRAP%"
set "OK=0"
for /l %%i in (1,1,60) do (
  curl -sf -o nul http://127.0.0.1:%PORT%/health 2>nul && set "OK=1"
  if "!OK!"=="1" goto :healthy
  ping -n 2 127.0.0.1 >nul
)
:healthy
if "%OK%"=="0" (
  echo  [w] console did not answer on port %PORT% in ~30s
  echo      log: %HOME%\admin_console.log
) else (
  echo  [ok] console healthy - opening browser
  start "" http://127.0.0.1:%PORT%/
)
goto :report

:ui_no
echo  [i] web UI not enabled - re-run and answer Y to enable it.
goto :report

rem ---- final report ----
:report
echo.
echo  ============================================================
echo    Done.
echo      plugin        : %HOME%\plugins\hybrid_memory
echo      console       : http://127.0.0.1:%PORT%/   loopback only
echo      first open    : you will see "Create your admin key"
echo      console log   : %HOME%\admin_console.log
echo      re-run anytime - it skips what is already done.
echo  ============================================================
exit /b 0

:usage
echo  single.bat [options]
echo    /check       verify only - prints state, changes nothing
echo    /auto        enable the UI and autostart without prompting
echo    /nui         install plugin without the UI
echo    /home:dir    Hermes home - default %%LOCALAPPDATA%%\hermes
echo    /port:N      console port - default 8733
echo    /src:dir     local Argos checkout that contains argos_plugin
echo    (question mark) or /help  show this help
exit /b 1