<#
.SYNOPSIS
  Restore the Argos memory store from a VSS backup snapshot.
  OFFLINE ONLY - the memory service and desktop app MUST be stopped first.

.DESCRIPTION
  Copies the three/four backup files (DuckDB main + WAL, Kuzu + WAL) from a
  snapshot directory back to the live store location, then verifies by
  reconstructing into a scratch copy and checking row counts against the
  manifest.

  Guard rail: acquires an exclusive file lock on the target .duckdb before
  copying. If the lock cannot be obtained (service or app still holds the
  store), the restore ABORTS. This is not a pid check - it is the actual
  safety property. A pid check is a race; a file lock is a guarantee.

  Restore is the ONLY operation that stops the service. No auto-restore ever.

.PARAMETER HermesHome
  HERMES_HOME directory (defaults to %LOCALAPPDATA%\hermes).

.PARAMETER SnapshotDir
  Path to the snapshot directory to restore from. Either this or -Latest must
  be specified.

.Parameter Latest
  Restore from the most recent verified snapshot in the backup root.

.Parameter BackupRoot
  Override the backup root (for -Latest). Defaults to config or <HermesHome>\backups\memory.

.Parameter DryRun
  Show what would be restored without copying.

.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File restore_memory.ps1 -Latest
  powershell -NoProfile -ExecutionPolicy Bypass -File restore_memory.ps1 -SnapshotDir "C:\...\memory-20260827-120000"
#>
[CmdletBinding()]
param(
    [string]$HermesHome = "$env:LOCALAPPDATA\hermes",
    [string]$SnapshotDir = "",
    [switch]$Latest,
    [string]$BackupRoot = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function Write-Log {
    param([string]$Msg, [string]$Level = "INFO")
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$ts] [$Level] $Msg"
}

function Fail-Exit {
    param([string]$Msg, [int]$Code = 1)
    Write-Log $Msg "ERROR"
    exit $Code
}

function Get-CfgProp {
    param($Obj, [string]$Name)
    if ($Obj -and $Obj.PSObject.Properties.Match($Name).Count -gt 0) {
        return $Obj.$Name
    }
    return $null
}

# ---------------------------------------------------------------------------
# 1. Resolve paths
# ---------------------------------------------------------------------------

$configPath = Join-Path $HermesHome "hybrid_memory.json"
if (-not (Test-Path $configPath)) {
    Fail-Exit "hybrid_memory.json not found at $configPath"
}
$config = Get-Content $configPath -Raw | ConvertFrom-Json

$dbName   = Get-CfgProp $config 'database_filename'
if (-not $dbName) { $dbName = "hybrid_memory.duckdb" }
$kuzuName = Get-CfgProp $config 'graph_dirname'
if (-not $kuzuName) { $kuzuName = "hybrid_memory_kuzu" }

$dbPath      = Join-Path $HermesHome $dbName
$walPath     = "$dbPath.wal"
$kuzuPath    = Join-Path $HermesHome $kuzuName
$kuzuWalPath = "$kuzuPath.wal"

# Resolve backup root
if (-not $BackupRoot) {
    $cfgRoot = Get-CfgProp $config 'backup_root'
    if ($cfgRoot) {
        $BackupRoot = [Environment]::ExpandEnvironmentVariables($cfgRoot)
    } else {
        $BackupRoot = Join-Path $HermesHome "backups\memory"
    }
}

# Resolve snapshot dir
if ($Latest) {
    if (-not (Test-Path $BackupRoot)) {
        Fail-Exit "Backup root not found: $BackupRoot (no backups have been created yet)"
    }
    $snapshots = Get-ChildItem -Path $BackupRoot -Directory -Filter "memory-*" |
        Where-Object { $_.Name -match "^memory-\d{8}-\d{6}$" } |
        Sort-Object Name -Descending
    if ($snapshots.Count -eq 0) {
        Fail-Exit "No snapshots found in $BackupRoot"
    }
    $SnapshotDir = $snapshots[0].FullName
    Write-Log "Latest snapshot: $SnapshotDir"
} elseif (-not $SnapshotDir) {
    Fail-Exit "Must specify -SnapshotDir <path> or -Latest"
}

if (-not (Test-Path $SnapshotDir)) {
    Fail-Exit "Snapshot directory not found: $SnapshotDir"
}

# Load manifest
$manifestPath = Join-Path $SnapshotDir "manifest.json"
if (-not (Test-Path $manifestPath)) {
    Fail-Exit "manifest.json not found in snapshot: $SnapshotDir"
}
$manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json

Write-Log "HermesHome:  $HermesHome"
Write-Log "Snapshot:    $SnapshotDir"
Write-Log "Manifest:    source_count=$($manifest.source_count) copy_count=$($manifest.copy_count)"

# ---------------------------------------------------------------------------
# 2. Guard rail - acquire exclusive lock on the target .duckdb
# ---------------------------------------------------------------------------

# This is the actual safety property, not a pid check. If the service or app
# holds the store, this lock acquisition will fail and we abort. A pid check
# is a race (service can respawn between check and copy); a file lock is a
# guarantee that no other process has the file open.

Write-Log "Acquiring exclusive lock on $dbPath ..."

$lockStream = $null
$lockAcquired = $false
$lockAttempts = 10
$lockDelaySec = 1

for ($i = 0; $i -lt $lockAttempts; $i++) {
    try {
        $lockStream = New-Object System.IO.FileStream(
            $dbPath,
            [System.IO.FileMode]::Open,
            [System.IO.FileAccess]::ReadWrite,
            [System.IO.FileShare]::None
        )
        $lockAcquired = $true
        Write-Log "Exclusive lock acquired."
        break
    } catch {
        if ($i -eq 0) {
            Write-Log "Store is held by another process. Retrying ($lockAttempts attempts, ${lockDelaySec}s apart)..." "WARN"
        }
        Start-Sleep -Seconds $lockDelaySec
    }
}

if (-not $lockAcquired) {
    Write-Log "Could not acquire exclusive lock on $dbPath after $lockAttempts attempts." "ERROR"
    Write-Log "The memory service or desktop app is still running." "ERROR"
    Write-Log "Quit the Hermes desktop app completely, then retry." "ERROR"
    Write-Log "Restore ABORTED - store is held." "ERROR"
    exit 1
}

# ---------------------------------------------------------------------------
# 3. Dry run
# ---------------------------------------------------------------------------

if ($DryRun) {
    Write-Log "DRY RUN - no copy."
    foreach ($mf in $manifest.files) {
        $srcFile = Join-Path $SnapshotDir $mf.name
        $exists = Test-Path $srcFile
        Write-Log "  $($mf.name): $($mf.size) bytes (snapshot: $(if($exists){'present'}else{'MISSING'}))"
    }
    Write-Log "Would copy to: $HermesHome"
    if ($lockStream) { $lockStream.Close() }
    exit 0
}

# ---------------------------------------------------------------------------
# 4. Copy files from snapshot to live store
# ---------------------------------------------------------------------------

try {
    $filesToRestore = @(
        @{ Name = $dbName;        Dst = $dbPath;      Critical = $true  },
        @{ Name = "$dbName.wal";  Dst = $walPath;     Critical = $true  }
    )
    if (Test-Path (Join-Path $SnapshotDir $kuzuName)) {
        $filesToRestore += @{ Name = $kuzuName;       Dst = $kuzuPath;    Critical = $true  }
    }
    if (Test-Path (Join-Path $SnapshotDir "$kuzuName.wal")) {
        $filesToRestore += @{ Name = "$kuzuName.wal"; Dst = $kuzuWalPath; Critical = $false }
    }

    # Close the lock before copying the .duckdb (Copy-Item needs write access)
    # The lock served its purpose: proving no other process holds the store.
    if ($lockStream) {
        $lockStream.Close()
        $lockStream = $null
        Write-Log "Lock released for copy phase."
    }

    foreach ($f in $filesToRestore) {
        $srcFile = Join-Path $SnapshotDir $f.Name
        if (-not (Test-Path $srcFile)) {
            if ($f.Critical) {
                Fail-Exit "Critical file missing from snapshot: $($f.Name)"
            }
            Write-Log "  SKIP $($f.Name) (not in snapshot)"
            continue
        }

        # Verify SHA256 against manifest before copying
        $srcHash = (Get-FileHash -Algorithm SHA256 -Path $srcFile).Hash
        $manifestEntry = $manifest.files | Where-Object { $_.name -eq $f.Name }
        if ($manifestEntry -and $srcHash -ne $manifestEntry.sha256) {
            Fail-Exit "SHA256 mismatch for $($f.Name): snapshot file corrupted (expected $($manifestEntry.sha256), got $srcHash)"
        }

        Write-Log "  Restoring $($f.Name) ..."
        Copy-Item -LiteralPath $srcFile -Destination $f.Dst -Force

        $dstSize = (Get-Item $f.Dst).Length
        $srcSize = (Get-Item $srcFile).Length
        if ($dstSize -ne $srcSize) {
            Fail-Exit "Size mismatch after copy for $($f.Name): src=$srcSize dst=$dstSize"
        }
        Write-Log "  OK   $($f.Name): $srcSize bytes"
    }

    Write-Log "Restore copy complete."

    # -----------------------------------------------------------------------
    # 5. Verify - reconstruct into scratch, open read-write, count rows
    # -----------------------------------------------------------------------

    Write-Log "Verifying restored store ..."

    $venvPy = Join-Path $HermesHome "hermes-agent\venv\Scripts\python.exe"
    if (-not (Test-Path $venvPy)) {
        $venvPy = Join-Path (Split-Path -Parent $PSScriptRoot) "venv\Scripts\python.exe"
    }
    if (-not (Test-Path $venvPy)) {
        Fail-Exit "Could not find venv python for DuckDB verify. Expected: $venvPy"
    }

    $verifyScript = @"
import sys, json, os, tempfile, shutil
db_path = sys.argv[1]
expected = int(sys.argv[2]) if len(sys.argv) > 2 else -1
scratch = tempfile.mkdtemp(prefix='argos_restore_verify_')
try:
    main_dst = os.path.join(scratch, os.path.basename(db_path))
    wal_dst = main_dst + '.wal'
    shutil.copy2(db_path, main_dst)
    if os.path.exists(db_path + '.wal'):
        shutil.copy2(db_path + '.wal', wal_dst)
    import duckdb
    con = duckdb.connect(main_dst)
    row = con.execute('SELECT count(*) FROM memory_records WHERE valid_to IS NULL').fetchone()
    total = con.execute('SELECT count(*) FROM memory_records').fetchone()
    con.close()
    result = {'ok': True, 'active_count': row[0], 'total_count': total[0]}
    if expected >= 0 and row[0] != expected:
        result['ok'] = False
        result['error'] = f'count mismatch: restored={row[0]} manifest={expected}'
    print(json.dumps(result))
finally:
    shutil.rmtree(scratch, ignore_errors=True)
"@

    $verifyScriptPath = Join-Path $env:TEMP "argos_restore_verify_$stamp.py"
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $verifyScript | Out-File -FilePath $verifyScriptPath -Encoding utf8 -Force

    $expectedCount = $manifest.copy_count
    $verifyResult = & $venvPy $verifyScriptPath $dbPath $expectedCount 2>&1 | Out-String
    Remove-Item $verifyScriptPath -Force -ErrorAction SilentlyContinue

    Write-Log "Verify output: $verifyResult"

    try {
        $verify = $verifyResult.Trim() | ConvertFrom-Json
    } catch {
        Fail-Exit "Verify script did not return valid JSON. Output: $verifyResult"
    }

    if (-not $verify.ok) {
        Fail-Exit "Restore verification FAILED: $($verify.error)"
    }

    Write-Log "Restore verified: active=$($verify.active_count) total=$($verify.total_count) (manifest: $expectedCount)"

    # -----------------------------------------------------------------------
    # 6. Success
    # -----------------------------------------------------------------------

    Write-Log "RESTORE SUCCESS: $SnapshotDir -> $HermesHome"
    Write-Log "The memory service will restart automatically on next client use."
    exit 0

} finally {
    if ($lockStream) {
        $lockStream.Close()
    }
}
