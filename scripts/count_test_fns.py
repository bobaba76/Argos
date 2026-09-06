#!/usr/bin/env python3
"""Count test modules and ``def test_`` functions in ``argos_plugin/tests/``.

Source of the numbers quoted in the CLAIMS-AUDIT.md §2 "Test suite" row and
the README Verification section. Counts are AST-based (so commented-out or
string-embedded ``def test_`` never inflate them) and tolerate a UTF-8 BOM.
``pytest --collect-only -q`` remains authoritative for parametrized totals.

Run from the repo root:

    python scripts/count_test_fns.py          # "136 modules / 2490 tests"
    python scripts/count_test_fns.py --json   # {"modules": ..., "tests": ...}
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent.parent / "argos_plugin" / "tests"


def count_test_functions(path: Path) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    )


def count_suite(tests_dir: Path = TESTS_DIR) -> dict[str, int]:
    modules = sorted(tests_dir.glob("test_*.py"))
    return {
        "modules": len(modules),
        "tests": sum(count_test_functions(m) for m in modules),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args(argv)
    counts = count_suite()
    if args.json:
        print(json.dumps(counts))
    else:
        print(f"{counts['modules']} modules / {counts['tests']} tests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
