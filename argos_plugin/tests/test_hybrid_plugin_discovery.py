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


class TestPluginDiscovery:
    def test_init_file_has_memory_provider(self):
        """The __init__.py must contain 'MemoryProvider' for discovery to find it."""
        init_path = _plugin_dir / "__init__.py"
        assert init_path.exists()
        source = init_path.read_text(encoding="utf-8")[:8192]
        assert "MemoryProvider" in source or "register_memory_provider" in source

    def test_plugin_yaml_exists(self):
        yaml_path = _plugin_dir / "plugin.yaml"
        assert yaml_path.exists()

    def test_config_schema_exists(self):
        schema_path = _plugin_dir / "config_schema.py"
        assert schema_path.exists()


