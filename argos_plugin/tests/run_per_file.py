"""Per-file pytest gate — one fresh process per test file (issue #51).

Background: ``pytest tests/ -q`` in a single process used to deadlock
mid-suite (see issue #51). The LLM-path tests stub the Hermes runtime
modules (``agent``, ``tools``, ...) by inserting synthetic modules into
``sys.modules``; leaked stubs can wedge a lazy import in the import
machinery. The validated workaround (2026-08-29) was to run each test
file in its own fresh process so no stub can cross a file boundary:

    cd argos_plugin
    for f in tests/test_*.py; do
      PYTHONPATH=... HF_HUB_OFFLINE=1 python -m pytest "$f" -q || break
    done

This script makes that gate reproducible without a shell loop: it runs
every ``tests/test_*.py`` file in a fresh subprocess and reports one
PASS/FAIL line per file. The single-process deadlock is fixed at the
root (conftest snapshots/restores the stub keys around every test), so
``pytest tests/ -q`` should be preferred for speed; this runner remains
the fallback gate when process isolation is needed (e.g. for bisecting
an import-state regression).

Run with:

    python tests/run_per_file.py              # all files, continue on failure
    python tests/run_per_file.py --stop       # stop at first failing file
    python tests/run_per_file.py --select distillation   # only matching files

Exit code: 0 if every file passed, 1 otherwise (matching pytest).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent


def _discover(select: str | None) -> list[Path]:
    files = sorted(_TESTS_DIR.glob("test_*.py"))
    if select:
        files = [f for f in files if select in f.name]
        if not files:
            print(f"No test files match --select {select!r}")
            sys.exit(2)
    return files


def _run_file(path: Path, stop_on_failure: bool) -> bool:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(path),
        "-q",
        "--tb=short",
        "-p",
        "no:cacheprovider",
    ]
    start = time.monotonic()
    result = subprocess.run(cmd, env=None)  # inherit PYTHONPATH/HF_HUB_OFFLINE
    elapsed = time.monotonic() - start
    status = "PASS" if result.returncode == 0 else "FAIL"
    print(f"  [{status}] {path.name} ({elapsed:.1f}s)")
    return result.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stop", action="store_true", help="stop at first failing file")
    parser.add_argument("--select", default=None, help="only files whose name contains this")
    args = parser.parse_args()

    files = _discover(args.select)
    print(f"Running {len(files)} test files, one process each...")
    failures: list[Path] = []
    attempted = 0
    for path in files:
        attempted += 1
        ok = _run_file(path, args.stop)
        if not ok:
            failures.append(path)
            if args.stop:
                break

    print()
    passed = attempted - len(failures)
    print(f"Results: {passed}/{attempted} files passed" + (f", failed: {failures}" if failures else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
