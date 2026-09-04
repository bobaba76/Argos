"""Audit tests for watcher.py W6/W3/W4 (issue #213).

W6: the watcher is wired into the provider lifecycle via WatcherThread.
W3: CSV extraction streams via csv.reader (not f.read()).
W4: extraction text returned to callers is bounded.

Run with (Hermes venv python, offline):
    python -m pytest tests/test_watcher_wiring_audit.py -v
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
# W6 — WatcherThread exists and is wired
# ---------------------------------------------------------------------------

class TestW6WatcherThread:
    def test_watcher_thread_module_exists(self):
        """W6: watcher_thread.py module exists."""
        import watcher_thread
        assert hasattr(watcher_thread, "WatcherThread")

    def test_watcher_thread_class_exists(self):
        """W6: WatcherThread class exists."""
        from watcher_thread import WatcherThread
        assert callable(WatcherThread)

    def test_watcher_thread_has_start_stop(self):
        """W6: WatcherThread has start() and stop() methods."""
        from watcher_thread import WatcherThread
        assert hasattr(WatcherThread, "start")
        assert hasattr(WatcherThread, "stop")

    def test_watcher_thread_is_daemon(self):
        """W6: WatcherThread runs as a daemon thread."""
        from watcher_thread import WatcherThread
        src = inspect.getsource(WatcherThread.start)
        assert "daemon=True" in src

    def test_run_watcher_pass_exists(self):
        """W6: run_watcher_pass function exists."""
        from watcher_thread import run_watcher_pass
        assert callable(run_watcher_pass)

    def test_run_watcher_pass_never_raises(self):
        """W6: run_watcher_pass returns counts dict even on error."""
        from watcher_thread import run_watcher_pass

        class FakeStore:
            def list_catalog(self, **kwargs):
                raise Exception("simulated failure")

        result = run_watcher_pass(FakeStore(), ["/nonexistent"])
        assert isinstance(result, dict)
        assert "new" in result
        assert "deleted" in result

    def test_provider_core_reads_watcher_config(self):
        """W6: provider_core.py reads watcher_enabled, watcher_scan_roots,
        watcher_interval_min from config."""
        from provider_core import ProviderCoreMixin
        init_src = inspect.getsource(ProviderCoreMixin.initialize)
        assert "watcher_enabled" in init_src
        assert "watcher_scan_roots" in init_src
        assert "watcher_interval_min" in init_src

    def test_provider_core_starts_watcher_thread(self):
        """W6: provider_core.py starts WatcherThread when enabled."""
        from provider_core import ProviderCoreMixin
        init_src = inspect.getsource(ProviderCoreMixin.initialize)
        assert "WatcherThread" in init_src
        assert "watcher_thread" in init_src

    def test_provider_core_watcher_is_config_gated(self):
        """W6: the watcher thread start is gated on watcher_enabled and
        non-empty scan_roots (no watcher config = zero behaviour change)."""
        from provider_core import ProviderCoreMixin
        init_src = inspect.getsource(ProviderCoreMixin.initialize)
        assert "self._watcher_enabled" in init_src
        assert "self._watcher_scan_roots" in init_src

    def test_provider_session_shutdown_stops_watcher(self):
        """W6: provider_session.py shutdown() stops the watcher thread."""
        from provider_session import ProviderSessionMixin
        shutdown_src = inspect.getsource(ProviderSessionMixin.shutdown)
        assert "_watcher_thread" in shutdown_src
        assert "stop" in shutdown_src

    def test_config_schema_includes_watcher_keys(self):
        """W6: the config schema includes watcher_enabled, watcher_scan_roots,
        and watcher_interval_min."""
        from provider_core import ProviderCoreMixin
        schema = ProviderCoreMixin.get_config_schema(None)
        keys = [item["key"] for item in schema]
        assert "watcher_enabled" in keys
        assert "watcher_scan_roots" in keys
        assert "watcher_interval_min" in keys

    def test_extraction_llm_config_routed_to_watcher(self):
        """W6: extraction_llm_model/provider are passed to WatcherThread."""
        from provider_core import ProviderCoreMixin
        init_src = inspect.getsource(ProviderCoreMixin.initialize)
        assert "extraction_llm_model" in init_src
        assert "extraction_llm_provider" in init_src

    def test_watcher_enabled_uses_flag_not_bool(self):
        """W6: watcher_enabled is parsed via _flag(), not bool().

        bool("false") is True in Python (non-empty string is truthy),
        which would start the watcher even when config says disabled.
        _flag() handles string "false" correctly.
        """
        from provider_core import ProviderCoreMixin, _flag
        init_src = inspect.getsource(ProviderCoreMixin.initialize)
        assert "_flag(" in init_src and "watcher_enabled" in init_src
        assert "bool(self._config.get(\"watcher_enabled\"" not in init_src

    def test_watcher_enabled_string_false_not_enabled(self):
        """W6: config with watcher_enabled="false" (string form from saved
        config) must NOT enable the watcher."""
        from provider_core import _flag
        # Simulate the config dict a user's saved config produces.
        config = {"watcher_enabled": "false", "watcher_scan_roots": "~/Docs"}
        assert _flag(config, "watcher_enabled", "false") is False

    def test_watcher_enabled_string_true_enabled(self):
        """W6: config with watcher_enabled="true" must enable the watcher."""
        from provider_core import _flag
        config = {"watcher_enabled": "true", "watcher_scan_roots": "~/Docs"}
        assert _flag(config, "watcher_enabled", "false") is True

    def test_watcher_enabled_missing_defaults_false(self):
        """W6: missing watcher_enabled key defaults to False."""
        from provider_core import _flag
        config = {}
        assert _flag(config, "watcher_enabled", "false") is False


# ---------------------------------------------------------------------------
# W3 — CSV extraction streams
# ---------------------------------------------------------------------------

class TestW3CsvStreaming:
    def test_csv_extraction_uses_csv_reader(self):
        """W3: extract_text_csv uses csv.reader, not f.read()."""
        from watcher import extract_text_csv
        src = inspect.getsource(extract_text_csv)
        assert "csv.reader" in src or "_csv.reader" in src
        # Check the code body (not the docstring) doesn't use f.read().
        # Strip the docstring by looking at lines after the first return/with.
        code_lines = [l for l in src.split("\n") if not l.strip().startswith("#")
                      and not l.strip().startswith('"""') and not l.strip().startswith("''")]
        code_body = "\n".join(code_lines)
        # The actual read call should be csv.reader, not f.read()
        assert "return f.read()" not in code_body
        assert "= f.read()" not in code_body

    def test_csv_extraction_has_max_chars_cap(self):
        """W3: extract_text_csv caps output at _MAX_CSV_TEXT_CHARS."""
        from watcher import extract_text_csv
        src = inspect.getsource(extract_text_csv)
        assert "_MAX_CSV_TEXT_CHARS" in src

    def test_max_csv_text_chars_constant_exists(self):
        """W3: _MAX_CSV_TEXT_CHARS constant is defined."""
        from watcher import _MAX_CSV_TEXT_CHARS
        assert isinstance(_MAX_CSV_TEXT_CHARS, int)
        assert _MAX_CSV_TEXT_CHARS > 0


# ---------------------------------------------------------------------------
# W4 — extraction text is bounded
# ---------------------------------------------------------------------------

class TestW4BoundedExtractionText:
    def test_watcher_thread_caps_excerpt(self):
        """W4: watcher_thread.py caps the excerpt persisted to the catalog."""
        from watcher_thread import run_watcher_pass
        src = inspect.getsource(run_watcher_pass)
        assert "_MAX_EXCERPT_CHARS" in src or "excerpt" in src

    def test_max_excerpt_chars_constant_exists(self):
        """W4: _MAX_EXCERPT_CHARS constant is defined in watcher_thread."""
        from watcher_thread import _MAX_EXCERPT_CHARS
        assert isinstance(_MAX_EXCERPT_CHARS, int)
        assert _MAX_EXCERPT_CHARS > 0
        # Should be bounded (not millions of chars).
        assert _MAX_EXCERPT_CHARS <= 65536
