"""Pytest conftest — make the plugin importable as 'hybrid_memory'.

The fork directory is named ``hybrid_memory_plugin`` but the deployed
plugin (and the integration tests) import it as ``hybrid_memory``.
Without this alias, ``from hybrid_memory.service_client import ...``
fails with ``ModuleNotFoundError: No module named 'hybrid_memory'``
on a fresh clone — a bad first impression for anyone running
``python -m pytest tests/ -v``.

This conftest registers the alias at collection time so the integration
tests work from both the fork and the deployed location.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

_plugin_dir = Path(__file__).resolve().parent.parent

# Ensure the plugin dir is on sys.path so its modules are importable.
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))


def _register_hybrid_memory_alias() -> None:
    """Register the plugin directory as the 'hybrid_memory' package.

    This lets ``from hybrid_memory.service_client import ...`` resolve
    to ``hybrid_memory_plugin/service_client.py`` without renaming the
    directory or installing the package.
    """
    if "hybrid_memory" in sys.modules:
        return  # already registered (e.g. deployed plugin)
    # Check if a real 'hybrid_memory' package is importable first.
    try:
        importlib.import_module("hybrid_memory")
        return  # real package exists, don't shadow it
    except ImportError:
        pass
    # Create a synthetic package alias pointing at the plugin dir.
    import types
    spec = importlib.util.spec_from_file_location(
        "hybrid_memory",
        str(_plugin_dir / "__init__.py"),
        submodule_search_locations=[str(_plugin_dir)],
    )
    if spec is None:
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules["hybrid_memory"] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        # If the full __init__ fails (e.g. optional deps missing),
        # register a bare namespace package so submodule imports
        # (service_client, store, etc.) still work.
        sys.modules["hybrid_memory"] = types.ModuleType("hybrid_memory")
        sys.modules["hybrid_memory"].__path__ = [str(_plugin_dir)]


_register_hybrid_memory_alias()
