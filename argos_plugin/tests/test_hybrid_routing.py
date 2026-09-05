"""Pytest tests for the argos plugin.

Run with:
    python -m pytest tests/test_argos.py -v

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


@pytest.fixture
def tmp_path_factory_dir(tmp_path):
    """Provide a clean temp directory for each test."""
    return tmp_path


class TestStorageRouting:
    def test_local_surfaces_use_primary_store(self):
        from routing import resolve_storage_names

        for platform in ("cli", "desktop", "tui", "local"):
            assert resolve_storage_names(
                platform, "hybrid_memory.duckdb", "hybrid_memory_kuzu"
            ) == ("hybrid_memory.duckdb", "hybrid_memory_kuzu")

    def test_remote_surfaces_use_gateway_store(self):
        from routing import resolve_storage_names

        assert resolve_storage_names(
            "telegram", "hybrid_memory.duckdb", "hybrid_memory_kuzu"
        ) == ("hybrid_memory_gateway.duckdb", "hybrid_memory_kuzu_gateway")


