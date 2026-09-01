"""Pytest tests for the argos plugin.

Run with:
    python -m pytest tests/test_argos.py -v

Or use the standalone script (no pytest needed):
    python tests/run_tests.py
"""
from __future__ import annotations

import sys
import tempfile
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Ensure the plugin package is importable.
_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir.parent) not in sys.path:
    sys.path.insert(0, str(_plugin_dir.parent))


class TestProviderInit:
    """Provider initialize() must survive direct-mode construction.

    Regression: Wave-2 shipped `config.get(...)` (NameError — bare name)
    at initialize() line ~700; the service never exercised the provider
    path so 147 tests + live service missed it. The provider-level eval
    harness caught it on first run.
    """

    def test_initialize_direct_mode(self, tmp_path):
        import json
        from argos_plugin import ArgosProvider

        home = tmp_path / "home"
        home.mkdir()
        (home / "hybrid_memory.json").write_text(json.dumps({
            "storage_mode": "direct",
            "database_filename": "test.duckdb",
            "graph_dirname": "test_kuzu",
            "auto_extract": "false",
        }), encoding="utf-8")
        # Minimal valid DuckDB file (no records needed for init).
        from store import DuckDBMemoryStore
        DuckDBMemoryStore(home / "test.duckdb", user_id="test_user").close()

        p = ArgosProvider()
        p.initialize(session_id="t", hermes_home=str(home),
                     platform="cli", user_id="test_user")
        assert p._evidence_retention == "full"  # would NameError before fix
        assert p._store is not None
        assert p._graph is not None
        p.shutdown()


