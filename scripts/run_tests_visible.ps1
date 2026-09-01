# run_tests_visible.ps1 - run pytest in a visible terminal window (issue #98).
#
# Why: long suite runs were backgrounded with output redirected to a log,
# which (a) hides live progress from the user and (b) block-buffers pytest's
# output, so "how far?" checks lag minutes behind reality.  This script runs
# pytest in a real console where output is line-buffered and live (thanks to
# PYTHONUNBUFFERED + Tee-Object), tees everything to a timestamped log, and
# the window stays open so the user can watch dots/tests/failures themselves.
#
# Usage:
#   powershell -File scripts/run_tests_visible.ps1            # full gate
#   powershell -File scripts/run_tests_visible.ps1 -Pop       # pop a new window
#   powershell -File scripts/run_tests_visible.ps1 -TestPath argos_plugin/tests/test_distillation.py -q
#   powershell -File scripts/run_tests_visible.ps1 -Python python -PytestArgs -q,-n,4
#
# Flags:
#   -Python      python executable to use (default: hermes venv-cuda - GPU)
#   -TestPath    test path (default: argos_plugin/tests/)
#   -Pop         relaunch in a fresh, always-open window and return immediately
#   remaining    arbitrary pytest args (replaces the default gate)
#
# The full output is captured to
# %LOCALAPPDATA%\Temp\argos_tests_<timestamp>.log (printed at the end), and
# the exit code matches pytest's.
param(
    [string]$Python = "C:\Users\michael\AppData\Local\hermes\hermes-agent\venv-cuda\Scripts\python.exe",
    [string]$TestPath = "argos_plugin/tests/",
    [switch]$Pop,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PytestArgs
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot

# -- Pop mode: launch a fresh window running the same script inline -----------
if ($Pop) {
    $inlineArgs = @("-ExecutionPolicy", "Bypass", "-File", $PSCommandPath, "-TestPath", $TestPath)
    if ($Python -ne "C:\Users\michael\AppData\Local\hermes\hermes-agent\venv-cuda\Scripts\python.exe") {
        $inlineArgs += "-Python"
        $inlineArgs += $Python
    }
    $inlineArgs += $PytestArgs
    $p = Start-Process -FilePath "powershell.exe" -ArgumentList $inlineArgs -WorkingDirectory $RepoRoot -PassThru
    Write-Output "Launched test window (PID $($p.Id)) - watch it on your desktop."
    exit 0
}

# -- Resolve the python executable --------------------------------------------
if (-not (Test-Path $Python)) {
    Write-Warning "Python '$Python' not found - falling back to 'python' on PATH."
    $Python = "python"
}

# -- Default gate args (xdist service grouping from #98) ----------------------
if ($PytestArgs.Count -eq 0) {
    $PytestArgs = @("-q", "-n", "4", "--dist", "loadgroup", "-p", "no:cacheprovider")
}

# -- Environment: hermetic embeddings + unbuffered live output ----------------
$env:HF_HUB_OFFLINE = "1"
$env:ARGOS_HERMETIC_TESTS = "1"
$env:PYTHONUNBUFFERED = "1"
$log = Join-Path $env:LOCALAPPDATA "Temp/argos_tests_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

Write-Output "Argos test run - $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Output "  python : $Python"
Write-Output "  target : $TestPath"
Write-Output "  args   : $($PytestArgs -join ' ')"
Write-Output "  cwd    : $RepoRoot"
Write-Output "  log    : $log"
Write-Output ""

Push-Location $RepoRoot
& $Python -m pytest $TestPath @PytestArgs 2>&1 | Tee-Object -FilePath $log
$code = $LASTEXITCODE
Pop-Location

Write-Output ""
Write-Output "pytest exit code: $code   (full log: $log)"
exit $code
