"""Structural test for #249-slice: shared state via self._state.

Asserts that the four store mixins reference shared state through
``self._state.<attr>`` and that no bare ``self._lock`` / ``self._scale_*``
/ ``self._alias_cache`` / ``self._read_only`` / ``self._retriever``
survives outside the state object. This prevents a future commit from
silently reintroducing cross-mixin bare attributes.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent


def _read(name: str) -> str:
    return (_plugin_dir / name).read_text(encoding="utf-8")


class TestStoreMixinStateStructural:
    """The four store mixins must use self._state for shared state."""

    MIXIN_FILES = ["store_core.py", "store_write.py", "store_retrieval.py", "store_maintenance.py"]

    BARE_ATTRS = [
        r"self\._lock\b",
        r"self\._alias_cache\b",
        r"self\._read_only\b",
        r"self\._retriever\b",
        r"self\._scale_warn_latency_ms\b",
        r"self\._scale_warn_records\b",
        r"self\._scale_window\b",
        r"self\._scale_latencies\b",
        r"self\._scale_queries\b",
        r"self\._scale_warnings_fired\b",
        r"self\._scale_last_count_check\b",
        r"self\._scale_record_count\b",
    ]

    def test_state_dataclass_exists(self):
        """store_state.py exists and defines StoreMixinState."""
        from store_state import StoreMixinState
        s = StoreMixinState()
        assert hasattr(s, "lock")
        assert hasattr(s, "scale_warn_latency_ms")
        assert hasattr(s, "scale_latencies")
        assert hasattr(s, "alias_cache")
        assert hasattr(s, "read_only")
        assert hasattr(s, "retriever")

    def test_core_init_creates_state(self):
        """StoreCoreMixin.__init__ creates self._state."""
        src = _read("store_core.py")
        assert "self._state = StoreMixinState()" in src

    def test_no_bare_shared_attrs_in_mixins(self):
        """No bare self._lock / self._scale_* / self._alias_cache / etc.
        in any of the four mixin files (outside comments/docstrings).

        We strip comments and docstrings before checking to avoid false
        positives from documentation that mentions the old names.
        """
        for fname in self.MIXIN_FILES:
            src = _read(fname)
            # Strip triple-quoted strings (docstrings) and # comments.
            cleaned = re.sub(r'""".*?"""', '', src, flags=re.DOTALL)
            cleaned = re.sub(r"'''.*?'''", '', cleaned, flags=re.DOTALL)
            cleaned = re.sub(r'#.*$', '', cleaned, flags=re.MULTILINE)
            for pattern in self.BARE_ATTRS:
                matches = re.findall(pattern, cleaned)
                assert not matches, (
                    f"{fname}: bare {pattern} found ({len(matches)} matches). "
                    f"Use self._state.<attr> instead."
                )

    def test_state_access_in_all_mixins(self):
        """All four mixins reference self._state at least once."""
        for fname in self.MIXIN_FILES:
            src = _read(fname)
            assert "self._state" in src, (
                f"{fname}: no self._state references found. "
                f"All shared state must go through self._state."
            )

    def test_state_fields_documented(self):
        """Every field in StoreMixinState has a docstring (non-empty line
        after the field declaration)."""
        src = _read("store_state.py")
        # Find all field declarations and check the following line starts
        # with a triple-quote or a # comment (docstring or inline doc).
        field_pattern = re.compile(r'^\s+(\w+):\s.*=\s.*$', re.MULTILINE)
        for m in field_pattern.finditer(src):
            field_name = m.group(1)
            if field_name.startswith("_"):
                continue
            # Find the next non-empty line after this field.
            end = m.end()
            rest = src[end:]
            lines = rest.split("\n")
            # Skip blank lines, find the first content line.
            next_line = ""
            for line in lines[1:]:
                if line.strip():
                    next_line = line.strip()
                    break
            assert next_line, (
                f"store_state.py: field '{field_name}' has no docstring."
            )
