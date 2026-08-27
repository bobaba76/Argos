@echo off
REM Argos Backup - elevated setup (registers scheduled task)
REM Right-click -> "Run as administrator" to execute.
REM The backup script itself does NOT need elevation - only task registration does.

set "SCRIPT_DIR=%~dp0"
set "TASK_XML=%SCRIPT_DIR%argos_vss_backup_task.xml"
set "BACKUP_SCRIPT=%SCRIPT_DIR%backup_memory.py"
set "TASK_NAME=Argos VSS Backup"

echo === Argos Backup Setup ===
echo.

REM Check elevation
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: This script must be run as Administrator.
    echo Right-click and select "Run as administrator".
    pause
    exit /b 1
)

echo [1/2] Registering scheduled task "%TASK_NAME%"...
schtasks /Create /TN "%TASK_NAME%" /XML "%TASK_XML%" /F
if %errorlevel% neq 0 (
    echo FAILED: Could not register scheduled task.
    pause
    exit /b 1
)
echo Task registered successfully.
echo.
echo [2/2] Running first backup to verify it works...
echo This will take ~10-30 seconds for a ~500MB store.
echo.
"%LOCALAPPDATA%\hermes\hermes-agent\venv\Scripts\python.exe" "%BACKUP_SCRIPT%"
set "BACKUP_EXIT=%errorlevel%"
echo.
if %BACKUP_EXIT% equ 0 (
    echo === SETUP COMPLETE ===
    echo The scheduled task will run twice daily (06:00 + 18:00).
    echo Backups are kept at 3 snapshots (~1.5 GB).
    echo No elevation needed for future backups - the task runs as-is.
) else (
    echo === BACKUP FAILED (exit code %BACKUP_EXIT%) ===
    echo Check the output above for the specific error.
)
echo.
pause
