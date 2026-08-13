#!/usr/bin/env python3
"""Monthly maintenance: compact state.db + snapshot hybrid-memory data.

Runs as a --no-agent cron job. Prints a short report; empty stdout = silent.
"""
import subprocess
import sys
from pathlib import Path

HOME = Path(r"C:\Users\user\AppData\Local\hermes")
REPO = Path(r"C:\Users\user\Documents\Github\Hermes")
AGENT = HOME / "hermes-agent"
PY = AGENT / "venv" / "Scripts" / "python.exe"
if not PY.exists():
    PY = Path(sys.executable)


def run(args, cwd):
    return subprocess.run(
        [str(PY), *args], cwd=str(cwd), capture_output=True, text=True, timeout=1800
    )


db = HOME / "state.db"
before = db.stat().st_size / 1e6 if db.exists() else 0.0

# 1. Compact the session store (FTS5 segment merge + VACUUM).
r1 = run(["-m", "hermes_cli.main", "sessions", "optimize"], cwd=AGENT)

after = db.stat().st_size / 1e6 if db.exists() else 0.0

# 2. Snapshot memory data + state (stop service, copy, verify).
r2 = run([str(REPO / "scripts" / "backup_data.py"), "--hermes-home", str(HOME)], cwd=REPO)

print(f"state.db: {before:.1f} MB -> {after:.1f} MB")
if r2.returncode:
    print("backup FAILED; full output:")
    print(r2.stdout.strip())
    print(r2.stderr.strip())
else:
    tail = [ln for ln in r2.stdout.splitlines() if ln.startswith("backup complete")]
    print(tail[-1] if tail else "backup: no completion line")
if r1.returncode or r2.returncode:
    print(f"errors: optimize={r1.returncode} backup={r2.returncode}")
    if r1.stderr.strip():
        print(r1.stderr.strip()[-500:])
