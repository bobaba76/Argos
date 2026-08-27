@echo off
REM Argos VSS Backup - elevated setup + first run
REM Right-click -> "Run as administrator" to execute.
REM This registers the scheduled task, sets the restore point frequency
REM to allow twice-daily backups, and runs the backup once to verify VSS works.

set "SCRIPT_DIR=%~dp0"
set "TASK_XML=%SCRIPT_DIR%argos_vss_backup_task.xml"
set "BACKUP_SCRIPT=%SCRIPT_DIR%vss_backup_memory.ps1"
set "TASK_NAME=Argos VSS Backup"

echo === Argos VSS Backup Setup ===
echo.

REM Check elevation
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: This script must be run as Administrator.
    echo Right-click and select "Run as administrator".
    pause
    exit /b 1
)

echo [1/3] Setting restore point frequency to allow twice-daily backups...
reg add "HKLM\Software\Microsoft\Windows NT\CurrentVersion\SystemRestore" /v SystemRestorePointCreationFrequency /t REG_DWORD /d 0 /f
echo Done.
echo.

echo [2/3] Registering scheduled task "%TASK_NAME%"...
schtasks /Create /TN "%TASK_NAME%" /XML "%TASK_XML%" /F
if %errorlevel% neq 0 (
    echo FAILED: Could not register scheduled task.
    pause
    exit /b 1
)
echo Task registered successfully.
echo.

echo [3/3] Running first backup (elevated) to verify VSS works...
echo This will take ~30-60 seconds for a ~500MB store.
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%BACKUP_SCRIPT%"
set "BACKUP_EXIT=%errorlevel%"
echo.
if %BACKUP_EXIT% equ 0 (
    echo === SETUP COMPLETE ===
    echo The scheduled task will run twice daily (06:00 + 18:00).
    echo Backups are kept at 3 snapshots (~1.6 GB).
) else (
    echo === BACKUP FAILED (exit code %BACKUP_EXIT%) ===
    echo Check the output above for the specific error.
)
echo.
pause
