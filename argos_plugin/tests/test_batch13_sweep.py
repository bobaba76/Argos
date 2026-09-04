"""Tests for batch-13 low-severity sweep (issue #91).

Seven targeted corrections from the deep-dive audit — no behavior change
beyond the fixes themselves. One test class per fix so a failure pinpoints
the exact regression.

Covers:
1. Deque undefined name in store_core.py (pyflakes + import smoke)
2. values_conflict ignores unit → false supersession proposals
3. purge_tombstone returns True on no-op (rowcount fix)
4. _detect_semantic_duplicates: break→continue (pair-budget) + parsed_ts sort
5. _flag rejects "on" (chronological_injection="on" silently false)
6. _load_config silently swallows malformed JSON
7. Dead locals (pyflakes-level — covered by the suite's pyflakes gate)
"""
from __future__ import annotations

import json
import logging
import sys
import types
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))


# ---------------------------------------------------------------------------
# Shared stubs (mirror conftest.py for standalone runnability)
# ---------------------------------------------------------------------------

def _stub_agent_modules():
    if "agent" not in sys.modules:
        sys.modules["agent"] = types.ModuleType("agent")
    if "agent.memory_provider" not in sys.modules:
        _mp = types.ModuleType("agent.memory_provider")

        class MemoryProvider:  # minimal stand-in
            pass

        _mp.MemoryProvider = MemoryProvider
        sys.modules["agent"].memory_provider = _mp
        sys.modules["agent.memory_provider"] = _mp
    if "tools" not in sys.modules:
        sys.modules["tools"] = types.ModuleType("tools")
    if "tools.registry" not in sys.modules:
        _tr = types.ModuleType("tools.registry")
        _tr.tool_error = lambda msg: json.dumps({"error": str(msg)})
        sys.modules["tools"].registry = _tr
        sys.modules["tools.registry"] = _tr


_stub_agent_modules()


# ---------------------------------------------------------------------------
# Fix 1: Deque undefined name — store_core.py:49
# ---------------------------------------------------------------------------

class TestDequeImport:
    """store_state.py annotates ``scale_latencies: Deque[float]`` — the
    ``Deque`` import must exist in store_state.py (was store_core.py before
    #249-slice moved the scale state into the dataclass)."""

    def test_deque_in_typing_import(self):
        import store_state
        src = Path(store_state.__file__).read_text(encoding="utf-8")
        assert "Deque" in src, "Deque must be imported in store_state.py"

    def test_store_core_imports_cleanly(self):
        """The module must import without NameError even under
        ``from __future__ import annotations`` introspection."""
        import importlib
        import store_core
        importlib.reload(store_core)
        # Verify the annotation resolves (typing.Deque exists at import time).
        assert hasattr(store_core, "StoreCoreMixin")

    def test_scale_latencies_annotation_resolves(self):
        """``typing.get_type_hints`` must not raise on the __init__ annotations
        — it would if ``Deque`` were undefined."""
        import typing
        import store_core
        try:
            typing.get_type_hints(store_core.StoreCoreMixin.__init__)
        except NameError as exc:
            pytest.fail(f"Deque annotation unresolved: {exc}")


# ---------------------------------------------------------------------------
# Fix 2: values_conflict ignores unit — value_extractor.py:177-198
# ---------------------------------------------------------------------------

class TestValuesConflictUnit:
    """The idempotency check was ``new_v.value == old_v.value`` only, ignoring
    unit: same-value/different-unit pairs were collapsed as idempotent, and
    different-value/different-unit pairs proposed nonsense supersessions.
    Units now define the quantity: cross-unit pairs are incomparable and
    never become a supersession candidate."""

    def test_same_value_different_unit_is_not_conflict(self):
        from value_extractor import ExtractedValue, values_conflict
        new = [ExtractedValue("interest rate", "3.5", "percent", "3.5%")]
        old = [ExtractedValue("interest rate", "3.5", "years", "3.5 years")]
        assert values_conflict(new, old) is None, \
            "same value + different unit is an incomparable quantity, not a supersession"

    def test_different_value_different_unit_is_not_conflict(self):
        """Cross-dimension pairs must never supersede — the false
        supersession-proposal class from issue #91 item 2."""
        from value_extractor import ExtractedValue, values_conflict
        new = [ExtractedValue("interest rate", "3.5", "percent", "3.5%")]
        old = [ExtractedValue("interest rate", "4.0", "years", "4.0 years")]
        assert values_conflict(new, old) is None, \
            "different unit = incomparable quantities, never a supersession"

    def test_same_value_same_unit_is_idempotent(self):
        from value_extractor import ExtractedValue, values_conflict
        new = [ExtractedValue("interest rate", "3.5", "percent", "3.5%")]
        old = [ExtractedValue("interest rate", "3.5", "percent", "3.5%")]
        assert values_conflict(new, old) is None

    def test_same_value_unit_case_insensitive_idempotent(self):
        """Unit normalisation (strip + lower) so 'Percent' vs 'percent'
        doesn't masquerade as a conflict."""
        from value_extractor import ExtractedValue, values_conflict
        new = [ExtractedValue("interest rate", "3.5", "Percent", "3.5%")]
        old = [ExtractedValue("interest rate", "3.5", "percent", "3.5%")]
        assert values_conflict(new, old) is None

    def test_same_value_none_unit_idempotent(self):
        from value_extractor import ExtractedValue, values_conflict
        new = [ExtractedValue("count", "42", None, "42")]
        old = [ExtractedValue("count", "42", None, "42")]
        assert values_conflict(new, old) is None

    def test_different_value_same_unit_conflict(self):
        from value_extractor import ExtractedValue, values_conflict
        new = [ExtractedValue("interest rate", "3.5", "percent", "3.5%")]
        old = [ExtractedValue("interest rate", "4.0", "percent", "4.0%")]
        conflict = values_conflict(new, old)
        assert conflict is not None


# ---------------------------------------------------------------------------
# Fix 3: purge_tombstone returns True on no-op — store_write.py:2093-2112
# ---------------------------------------------------------------------------

class TestPurgeTombstoneRowcount:
    """The old code did DELETE then COUNT(*)==0 → True even when no tombstone
    existed (nothing was purged). The fix uses the DELETE rowcount."""

    @pytest.fixture()
    def store(self, tmp_path):
        from store import DuckDBMemoryStore
        s = DuckDBMemoryStore(
            tmp_path / "purge.duckdb", user_id="test_user", embedder=None,
        )
        yield s
        s.close()

    def test_purge_existing_tombstone_returns_true(self, store):
        rec = store.remember(category="personal_fact", content="Alex lives in Berlin.")
        assert rec is not None
        store.delete_memory(rec.memory_id)
        # Tombstone now exists; purging it must return True.
        assert store.purge_tombstone("Alex lives in Berlin.", "personal_fact") is True

    def test_purge_nonexistent_returns_false(self, store):
        """No tombstone exists for this content — purge must return False,
        not True (the old COUNT-based no-op bug)."""
        assert store.purge_tombstone("content that was never deleted", "personal_fact") is False

    def test_double_purge_second_returns_false(self, store):
        rec = store.remember(category="personal_fact", content="Sam lives in Tokyo.")
        store.delete_memory(rec.memory_id)
        assert store.purge_tombstone("Sam lives in Tokyo.", "personal_fact") is True
        # Second purge — tombstone already removed, nothing to delete.
        assert store.purge_tombstone("Sam lives in Tokyo.", "personal_fact") is False


# ---------------------------------------------------------------------------
# Fix 4: _detect_semantic_duplicates — store_maintenance.py:317-467
# ---------------------------------------------------------------------------

class TestSemanticDedupPairBudget:
    """Bug A: ``break`` on pair-budget hit exited the whole groups loop —
    remaining categories were never scanned. Fix: ``continue`` (skip this
    group, keep going)."""

    def test_oversized_group_does_not_block_later_groups(self, tmp_path):
        """Three groups: A (small, fits), B (huge, exceeds budget), C (small,
        fits remaining budget). With ``break``, C is never checked. With
        ``continue``, C produces candidates."""
        from store_common import MemoryRecord
        from store import DuckDBMemoryStore

        # We need a store instance to access the mixin method, but we'll
        # call _detect_semantic_duplicates directly with synthetic records.
        store = DuckDBMemoryStore(
            tmp_path / "budget.duckdb", user_id="test_user", embedder=None,
        )
        try:
            # Group A: 2 records, 1 pair — semantic duplicates.
            # Group B: 50 records, 1225 pairs — exceeds budget.
            # Group C: 2 records, 1 pair — semantic duplicates.
            base_emb = [1.0, 0.0, 0.0]

            records = []
            # Group A (category=cat_a)
            for i in range(2):
                records.append(MemoryRecord(
                    memory_id=f"a{i}",
                    category="cat_a",
                    content=f"Group A duplicate number {i} with enough text here",
                    created_at="2026-01-01T00:00:00+00:00",
                    embedding=base_emb,
                    confidence=0.5,
                    user_scope="test_user",
                ))
            # Group B (category=cat_b) — oversized
            for i in range(50):
                records.append(MemoryRecord(
                    memory_id=f"b{i}",
                    category="cat_b",
                    content=f"Group B record number {i} with enough text here",
                    created_at="2026-01-01T00:00:00+00:00",
                    embedding=base_emb,
                    confidence=0.5,
                    user_scope="test_user",
                ))
            # Group C (category=cat_c) — small, should still be scanned
            for i in range(2):
                records.append(MemoryRecord(
                    memory_id=f"c{i}",
                    category="cat_c",
                    content=f"Group C duplicate number {i} with enough text here",
                    created_at="2026-01-01T00:00:00+00:00",
                    embedding=base_emb,
                    confidence=0.5,
                    user_scope="test_user",
                ))

            candidates = []

            def add_candidate(record, reason, keeper_id, **kw):
                candidates.append((record.memory_id, reason, keeper_id))

            # max_pairs=5: group A (1 pair) fits, group B (1225 pairs) doesn't,
            # group C (1 pair) fits the remaining budget (pairs_checked=1,
            # 1+1=2 ≤ 5).
            store._detect_semantic_duplicates(
                records, min_similarity=0.99, max_pairs=5,
                add_candidate_fn=add_candidate,
            )

            # Group A should produce 1 candidate (2 records, 1 keeper).
            # Group C should ALSO produce 1 candidate — the bug's ``break``
            # would have skipped it entirely.
            group_a_candidates = [c for c in candidates if c[0].startswith("a")]
            group_c_candidates = [c for c in candidates if c[0].startswith("c")]
            assert len(group_a_candidates) >= 1, \
                "group A must produce candidates"
            assert len(group_c_candidates) >= 1, \
                "group C must still produce candidates after group B is skipped"
        finally:
            store.close()


class TestSemanticDedupKeeperSort:
    """Bug B: keeper sort used ``r.created_at or ""`` (raw-string
    lexicographic recency), which mis-orders records carrying different
    UTC offsets: "2026-08-30T00:00:00+00:00" sorts BEFORE
    "2026-08-30T00:00:00+05:00" lexicographically ('0' < '5' in the
    offset), even though the +05:00 instant (Aug 29 19:00 UTC) is the
    OLDER of the two. Fix: use ``_parse_timestamp`` (UTC-normalised)."""

    @pytest.fixture()
    def store(self, tmp_path):
        from store import DuckDBMemoryStore
        s = DuckDBMemoryStore(
            tmp_path / "keeper.duckdb", user_id="test_user", embedder=None,
        )
        yield s
        s.close()

    def _detect(self, store, records):
        candidates = []

        def add_candidate(record, reason, keeper_id, **kw):
            candidates.append((record.memory_id, keeper_id))

        store._detect_semantic_duplicates(
            records, min_similarity=0.99, max_pairs=10,
            add_candidate_fn=add_candidate,
        )
        return candidates

    def test_keeper_is_chronologically_oldest_not_lexicographically(self, store):
        """Two records, equal quality/length, both with VALID ISO created_at
        but different offsets. Lexicographic order picks the wrong keeper;
        parsed (UTC-normalised) order picks the chronologically oldest."""
        from store_common import MemoryRecord

        emb = [1.0, 0.0, 0.0]
        # old_record: +05:00 wall-clock Aug 30 = Aug 29 19:00 UTC (OLDER).
        # Lexicographically LARGER ("...+05:00" > "...+00:00").
        old_record = MemoryRecord(
            memory_id="old",
            category="cat",
            content="Duplicate content for keeper sort test here",
            created_at="2026-08-30T00:00:00+05:00",
            embedding=emb,
            confidence=0.5,
            user_scope="test_user",
        )
        # new_record: +00:00 Aug 30 = Aug 30 00:00 UTC (NEWER).
        # Lexicographically SMALLER ("...+00:00" < "...+05:00").
        new_record = MemoryRecord(
            memory_id="new",
            category="cat",
            content="Duplicate content for keeper sort test here",
            created_at="2026-08-30T00:00:00+00:00",
            embedding=emb,
            confidence=0.5,
            user_scope="test_user",
        )

        candidates = self._detect(store, [old_record, new_record])

        assert len(candidates) == 1, \
            "exactly one record should be quarantined"
        quarantined_id, keeper_id = candidates[0]

        # Keeper = chronologically-oldest ("old", Aug 29 19:00 UTC).
        # Lexicographic order would have made "new" the keeper.
        assert keeper_id == "old", \
            f"keeper must be chronologically oldest ('old'), got '{keeper_id}'"
        assert quarantined_id == "new", \
            f"quarantined must be chronologically newest ('new'), got '{quarantined_id}'"

    def test_unparseable_timestamp_does_not_crash(self, store):
        """An unparseable created_at must not crash the keeper sort — it
        falls back to epoch 0 (oldest) exactly like the old ``or ""``."""
        from store_common import MemoryRecord

        emb = [1.0, 0.0, 0.0]
        unparseable = MemoryRecord(
            memory_id="unparseable",
            category="cat",
            content="Duplicate content for keeper sort fallback here",
            created_at="not-a-date",
            embedding=emb,
            confidence=0.5,
            user_scope="test_user",
        )
        dated = MemoryRecord(
            memory_id="dated",
            category="cat",
            content="Duplicate content for keeper sort fallback here",
            created_at="2026-08-30T00:00:00+00:00",
            embedding=emb,
            confidence=0.5,
            user_scope="test_user",
        )

        candidates = self._detect(store, [unparseable, dated])

        assert len(candidates) == 1, \
            "exactly one record should be quarantined"
        # Deterministic selection: the unparseable record sorts as epoch 0
        # (oldest) and is preferred as keeper, mirroring the old ``or ""``.
        assert candidates[0][1] == "unparseable"


# ---------------------------------------------------------------------------
# Fix 5: _flag rejects "on" — provider_core.py:272-280
# ---------------------------------------------------------------------------

class TestFlagAcceptsOn:
    """``_flag`` accepted only ("true","1","yes") while sibling
    ``egress._flag`` also accepts "on". ``chronological_injection="on"``
    silently behaved as false."""

    @staticmethod
    def _mod():
        try:
            import argos_plugin
            return argos_plugin
        except ModuleNotFoundError:
            import argos
            return argos

    def test_on_string_parses_true(self):
        m = self._mod()
        assert m._flag({"chronological_injection": "on"},
                       "chronological_injection") is True

    def test_on_string_case_insensitive(self):
        m = self._mod()
        assert m._flag({"key": "ON"}, "key") is True
        assert m._flag({"key": "On"}, "key") is True

    def test_surrounding_whitespace_tolerated(self):
        """Matches egress._flag, which strips before comparing."""
        m = self._mod()
        assert m._flag({"key": " on "}, "key") is True
        assert m._flag({"key": " true "}, "key") is True

    def test_off_still_false(self):
        m = self._mod()
        assert m._flag({"key": "off"}, "key") is False

    def test_existing_true_values_still_work(self):
        m = self._mod()
        for val in ("true", "1", "yes"):
            assert m._flag({"key": val}, "key") is True

    def test_initialize_wires_chrono_via_flag(self):
        """Guard: initialize() must read chronological_injection from the
        config object. #244: _flag() is replaced by MemoryConfig bool
        coercion (which handles 'on' the same way _flag did)."""
        import inspect
        m = self._mod()
        src = inspect.getsource(m.ArgosProvider.initialize)
        assert 'cfg.chronological_injection' in src


# ---------------------------------------------------------------------------
# Fix 6: _load_config silently swallows malformed JSON — provider_core.py
# ---------------------------------------------------------------------------

class TestLoadConfigMalformedJson:
    """``except Exception: pass`` meant a trailing comma in
    hybrid_memory.json silently dropped the user's entire config. Fix: log a
    warning naming the file path, then fall through to defaults."""

    def test_malformed_json_returns_defaults_with_warning(self, tmp_path, caplog):
        from provider_core import _load_config
        config_path = tmp_path / "hybrid_memory.json"
        config_path.write_text('{"chronological_injection": "true",}', encoding="utf-8")

        with caplog.at_level(logging.WARNING, logger="argos_plugin.provider_core"):
            cfg = _load_config(str(tmp_path))

        # #244: _load_config now returns a MemoryConfig object.
        # Defaults must still be returned.
        assert hasattr(cfg, "database_filename")
        assert cfg.database_filename == "hybrid_memory.duckdb"

        # A warning must be logged naming the file path.
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any(str(config_path) in r.getMessage() for r in warnings), \
            "warning must name the malformed config file path"

    def test_valid_json_still_loaded(self, tmp_path):
        from provider_core import _load_config
        config_path = tmp_path / "hybrid_memory.json"
        config_path.write_text(
            json.dumps({"chronological_injection": "on"}), encoding="utf-8",
        )
        cfg = _load_config(str(tmp_path))
        # #244: "on" is coerced to True by the bool validator.
        assert cfg.chronological_injection is True

    def test_no_config_file_returns_defaults(self, tmp_path):
        from provider_core import _load_config
        cfg = _load_config(str(tmp_path))
        # #244: returns MemoryConfig with defaults.
        assert hasattr(cfg, "database_filename")
        assert cfg.database_filename == "hybrid_memory.duckdb"

    def test_memory_service_load_config_malformed_json(self, tmp_path, caplog):
        """Same-class fix: memory_service._load_config must also log + return
        defaults on malformed JSON."""
        from memory_service import _load_config as ms_load_config
        config_path = tmp_path / "hybrid_memory.json"
        config_path.write_text('{invalid json}', encoding="utf-8")

        with caplog.at_level(logging.WARNING, logger="argos.service"):
            cfg = ms_load_config(tmp_path)

        assert cfg == {}
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any(str(config_path) in r.getMessage() for r in warnings), \
            "memory_service warning must name the malformed config file path"
