"""Audit tests for egress EG1 (issue #266).

Covers mutable config cache returned to callers.

Run with (Hermes venv python, offline):
    python -m pytest tests/test_egress_audit2.py -v
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))


# ---------------------------------------------------------------------------
# EG1 -- immutable config cache
# ---------------------------------------------------------------------------

class TestEG1ImmutableConfig:
    def test_mapping_proxy_in_source(self):
        """EG1: load_config uses MappingProxyType for immutability."""
        import egress
        src = inspect.getsource(egress.load_config)
        assert "MappingProxyType" in src

    def test_returned_config_is_immutable(self):
        """EG1: mutating the returned config raises TypeError."""
        import egress
        egress._reset_config_cache()
        cfg = egress.load_config()
        # MappingProxyType raises TypeError on __setitem__.
        with pytest.raises(TypeError):
            cfg["local_only"] = True

    def test_mutation_does_not_poison_cache(self):
        """EG1: even if a caller copies and mutates, the cache is unaffected."""
        import egress
        egress._reset_config_cache()
        cfg1 = egress.load_config()
        # Make a mutable copy and mutate it.
        mutable = dict(cfg1)
        mutable["local_only"] = True
        # Reload — should not see the mutation.
        egress._reset_config_cache()
        cfg2 = egress.load_config()
        assert cfg2.get("local_only") == cfg1.get("local_only")

    def test_gate_still_works_with_immutable(self):
        """EG1: gate() still works with the immutable config view."""
        import egress
        egress._reset_config_cache()
        # gate should not raise — it only reads from cfg.
        result = egress.gate("watcher_extraction", "test text")
        assert isinstance(result, bool)

    def test_flag_still_works_with_immutable(self):
        """EG1: _flag() still works with the immutable config view."""
        import egress
        egress._reset_config_cache()
        cfg = egress.load_config()
        # _flag should not raise — it only reads.
        result = egress._flag(cfg, "local_only", False)
        assert isinstance(result, bool)
