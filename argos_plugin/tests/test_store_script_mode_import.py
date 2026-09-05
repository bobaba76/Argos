"""Entrypoint test for #304: store.py must import in script mode.

Production launches memory_service.py as a script (subprocess with
cwd=plugin_dir, __package__ None). store.py MUST work in that mode —
not just under pytest where conftest.py registers package aliases.

This test runs a real subprocess that imports store.py as a top-level
module (no package context) and asserts it succeeds. It also runs
runpy.run_path on memory_service.py to verify the production entrypoint.

Run with (Hermes venv python, hermetic):
    ARGOS_HERMETIC_TESTS=1 python -m pytest tests/test_store_script_mode_import.py -v
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
_repo_root = _plugin_dir.parent

# The Hermes venv python — same one used for the rest of the suite.
_PYTHON = sys.executable


class TestScriptModeImport:
    """store.py must import when run as a top-level script (no package)."""

    def test_store_imports_as_top_level_module(self):
        """A subprocess that does 'from store import DuckDBMemoryStore'
        with cwd=plugin_dir must succeed — this is the production path."""
        result = subprocess.run(
            [_PYTHON, "-c",
             "import sys; sys.path.insert(0, '.'); "
             "from store import DuckDBMemoryStore, MemoryRecord, "
             "VALID_CATEGORIES, sanitize_content, _INJECTION_PATTERNS; "
             "print('OK')"],
            cwd=str(_plugin_dir),
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "ARGOS_HERMETIC_TESTS": "1"},
        )
        assert result.returncode == 0, \
            f"store.py script-mode import failed:\n{result.stderr}"
        assert "OK" in result.stdout

    def test_memory_service_imports_as_script(self):
        """runpy.run_path on memory_service.py must succeed — this is
        how the shared service subprocess launches."""
        result = subprocess.run(
            [_PYTHON, "-c",
             "import runpy, sys; sys.path.insert(0, '.'); "
             "runpy.run_path('memory_service.py'); "
             "print('OK')"],
            cwd=str(_plugin_dir),
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "ARGOS_HERMETIC_TESTS": "1"},
        )
        assert result.returncode == 0, \
            f"memory_service.py script-mode import failed:\n{result.stderr}"
        assert "OK" in result.stdout

    def test_reconcile_graph_imports_as_script(self):
        """reconcile_graph.py must import as a top-level script — it
        uses 'from service_client import ...' which transitively
        imports store.py."""
        result = subprocess.run(
            [_PYTHON, "-c",
             "import sys; sys.path.insert(0, '.'); "
             "import reconcile_graph; "
             "print('OK')"],
            cwd=str(_plugin_dir),
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "ARGOS_HERMETIC_TESTS": "1"},
        )
        assert result.returncode == 0, \
            f"reconcile_graph.py script-mode import failed:\n{result.stderr}"
        assert "OK" in result.stdout

    def test_store_has_no_duplicated_name_list(self):
        """#304: the store_common re-export list must appear only ONCE
        in store.py (no duplication between package/script branches)."""
        src = (_plugin_dir / "store.py").read_text(encoding="utf-8")
        # The name list is in _STORE_COMMON_NAMES. The old pattern had
        # two copies of the full import list (try + except). Now there
        # should be exactly one _STORE_COMMON_NAMES tuple.
        assert src.count("_STORE_COMMON_NAMES") == 2  # definition + for loop
        # Must NOT have the old try/except ImportError pattern.
        assert "except ImportError" not in src
        # Must branch on __package__ (the production-safe pattern).
        assert "__package__" in src
