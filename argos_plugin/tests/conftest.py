"""Pytest conftest — make the plugin importable as 'argos'.

The fork directory is named ``argos_plugin`` but the deployed
plugin (and the integration tests) import it as ``argos``.
Without this alias, ``from argos.service_client import ...``
fails with ``ModuleNotFoundError: No module named 'argos'``
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


def _register_argos_alias() -> None:
    """Register the plugin directory as the 'argos' package.

    This lets ``from argos.service_client import ...`` resolve
    to ``argos_plugin/service_client.py`` without renaming the
    directory or installing the package.
    """
    if "argos" in sys.modules:
        return  # already registered (e.g. deployed plugin)
    # Check if a real 'argos' package is importable first.
    try:
        importlib.import_module("argos")
        return  # real package exists, don't shadow it
    except ImportError:
        pass
    # Create a synthetic package alias pointing at the plugin dir.
    import types
    spec = importlib.util.spec_from_file_location(
        "argos",
        str(_plugin_dir / "__init__.py"),
        submodule_search_locations=[str(_plugin_dir)],
    )
    if spec is None:
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules["argos"] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        # If the full __init__ fails (e.g. optional deps missing),
        # register a bare namespace package so submodule imports
        # (service_client, store, etc.) still work.
        sys.modules["argos"] = types.ModuleType("argos")
        sys.modules["argos"].__path__ = [str(_plugin_dir)]


_register_argos_alias()
