"""Audit tests for provider_core.py PC5-PC6 (issue #207).

PC5: _active_user_id global is not multi-user safe — thread-safe
setter/getter added, _get_insight_store accepts explicit user_id.
PC6: provider _acl_config loaded from store instead of global config.

Run with (Hermes venv python, offline):
    python -m pytest tests/test_provider_core_audit.py -v
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
for _path in (_plugin_dir.parent, _plugin_dir):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


# ---------------------------------------------------------------------------
# PC5 — _active_user_id thread safety
# ---------------------------------------------------------------------------

class TestPC5ActiveUserIdThreadSafety:
    def test_set_active_user_id_exists(self):
        """PC5: _set_active_user_id function exists."""
        from provider_core import _set_active_user_id
        assert callable(_set_active_user_id)

    def test_get_active_user_id_exists(self):
        """PC5: _get_active_user_id function exists."""
        from provider_core import _get_active_user_id
        assert callable(_get_active_user_id)

    def test_set_and_get_roundtrip(self):
        """PC5: setting and getting the user_id is consistent."""
        from provider_core import _set_active_user_id, _get_active_user_id
        _set_active_user_id("test_user_pc5")
        assert _get_active_user_id() == "test_user_pc5"
        # Restore default.
        _set_active_user_id("default_user")

    def test_setter_uses_lock(self):
        """PC5: the setter uses a lock for thread safety."""
        from provider_core import _set_active_user_id
        src = inspect.getsource(_set_active_user_id)
        assert "lock" in src.lower()

    def test_getter_uses_lock(self):
        """PC5: the getter uses a lock for thread safety."""
        from provider_core import _get_active_user_id
        src = inspect.getsource(_get_active_user_id)
        assert "lock" in src.lower()

    def test_initialize_uses_setter_not_global(self):
        """PC5: initialize() uses _set_active_user_id, not 'global _active_user_id'."""
        from provider_core import ProviderCoreMixin
        src = inspect.getsource(ProviderCoreMixin.initialize)
        assert "_set_active_user_id" in src
        assert "global _active_user_id" not in src

    def test_get_insight_store_accepts_user_id(self):
        """PC5: _get_insight_store accepts an optional user_id parameter."""
        from provider_ambient import _get_insight_store
        sig = inspect.signature(_get_insight_store)
        assert "user_id" in sig.parameters
        # Default should be None (falls back to global).
        assert sig.parameters["user_id"].default is None

    def test_get_insight_store_prefers_explicit_user_id(self):
        """PC5: _get_insight_store uses explicit user_id over the global."""
        from provider_ambient import _get_insight_store
        src = inspect.getsource(_get_insight_store)
        assert "user_id is not None" in src or "user_id if user_id" in src


# ---------------------------------------------------------------------------
# PC6 — _acl_config from store
# ---------------------------------------------------------------------------

class TestPC6AclConfigFromStore:
    def test_initialize_prefers_store_acl(self):
        """PC6: initialize() gets _acl_config from the store when available."""
        from provider_core import ProviderCoreMixin
        src = inspect.getsource(ProviderCoreMixin.initialize)
        assert "store_acl" in src or "getattr(self._store" in src
        assert "_acl_config" in src

    def test_initialize_falls_back_to_config(self):
        """PC6: when store has no _acl_config, falls back to loading from config."""
        from provider_core import ProviderCoreMixin
        src = inspect.getsource(ProviderCoreMixin.initialize)
        # The fallback path should still load from self._config["acl"].
        assert "self._config.get(\"acl\")" in src or 'self._config.get("acl")' in src
