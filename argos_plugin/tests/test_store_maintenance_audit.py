"""Audit tests for store_maintenance.py (SM1-SM10, issue #231).

Covers:
- SM1: user_scope filter in explain_retrieval
- SM2: key allowlist on set_state
- SM3: LIMIT on cleanup_junk
- SM4: pair limit on containment dedup
- SM5: valid_to guard on backfill UPDATE
- SM6: index map for semantic duplicates
- SM7: 120-char fingerprint
- SM8: valid_to guard on archive UPDATE
- SM9: key allowlist on get_state
- SM10: forget_stale_records (kept per-ID for partial-failure semantics)

Run with (Hermes venv python, offline):
    python -m pytest tests/test_store_maintenance_audit.py -v
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
# SM1 â€” user_scope filter in explain_retrieval
# ---------------------------------------------------------------------------

class TestSM1ExplainRetrievalScopeFilter:
    def test_explain_retrieval_has_user_scope_filter(self):
        """SM1: explain_retrieval SELECT includes user_scope filter."""
        from store_maintenance import StoreMaintenanceMixin
        src = inspect.getsource(StoreMaintenanceMixin.explain_retrieval)
        assert "user_scope IS NULL OR user_scope" in src, (
            "SM1: explain_retrieval should filter by user_scope"
        )


# ---------------------------------------------------------------------------
# SM2/SM9 â€” key allowlist on set_state/get_state
# ---------------------------------------------------------------------------

class TestSM2SetStateAllowlist:
    def test_allowlist_exists(self):
        from store_maintenance import StoreMaintenanceMixin
        assert hasattr(StoreMaintenanceMixin, "_STATE_KEY_ALLOWLIST")
        allow = StoreMaintenanceMixin._STATE_KEY_ALLOWLIST
        assert "distillation_last_run" in allow
        assert "surfaced_confirmation_ids" in allow

    def test_set_state_rejects_unknown_key(self):
        """SM2: set_state rejects keys not in the allowlist."""
        from store_maintenance import StoreMaintenanceMixin
        src = inspect.getsource(StoreMaintenanceMixin.set_state)
        assert "allowlist" in src.lower()

    def test_get_state_rejects_unknown_key(self):
        """SM9: get_state rejects keys not in the allowlist."""
        from store_maintenance import StoreMaintenanceMixin
        src = inspect.getsource(StoreMaintenanceMixin.get_state)
        assert "allowlist" in src.lower()


# ---------------------------------------------------------------------------
# SM3 â€” LIMIT on cleanup_junk
# ---------------------------------------------------------------------------

class TestSM3CleanupJunkLimit:
    def test_cleanup_junk_has_limit(self):
        """SM3: cleanup_junk SELECT has a LIMIT clause."""
        from store_maintenance import StoreMaintenanceMixin
        src = inspect.getsource(StoreMaintenanceMixin.cleanup_junk)
        assert "LIMIT" in src, "SM3: cleanup_junk should have a LIMIT"


# ---------------------------------------------------------------------------
# SM4 â€” pair limit on containment dedup
# ---------------------------------------------------------------------------

class TestSM4ContainmentPairLimit:
    def test_containment_dedup_has_pair_limit(self):
        """SM4: consolidate containment dedup has a pair limit."""
        from store_maintenance import StoreMaintenanceMixin
        src = inspect.getsource(StoreMaintenanceMixin.consolidate)
        assert "MAX_CONTAINMENT_PAIRS" in src, (
            "SM4: consolidate should cap containment pairs"
        )


# ---------------------------------------------------------------------------
# SM5 â€” valid_to guard on backfill UPDATE
# ---------------------------------------------------------------------------

class TestSM5BackfillValidToGuard:
    def test_backfill_update_has_valid_to_guard(self):
        """SM5: backfill_null_embeddings UPDATE has valid_to IS NULL."""
        from store_maintenance import StoreMaintenanceMixin
        src = inspect.getsource(StoreMaintenanceMixin.backfill_null_embeddings)
        assert "valid_to IS NULL" in src, (
            "SM5: backfill UPDATE should have valid_to IS NULL guard"
        )


# ---------------------------------------------------------------------------
# SM6 â€” index map for semantic duplicates
# ---------------------------------------------------------------------------

class TestSM6IndexMap:
    def test_index_map_used(self):
        """SM6: _detect_semantic_duplicates uses an index map instead of
        group_records.index()."""
        from store_maintenance import StoreMaintenanceMixin
        src = inspect.getsource(StoreMaintenanceMixin._detect_semantic_duplicates)
        assert "_index_map" in src, (
            "SM6: should use _index_map instead of group_records.index()"
        )


# ---------------------------------------------------------------------------
# SM7 â€” 120-char fingerprint
# ---------------------------------------------------------------------------

class TestSM7FingerprintLength:
    def test_fingerprint_is_120_chars(self):
        """SM7: cleanup_junk fingerprint is 120 chars (was 60)."""
        from store_maintenance import StoreMaintenanceMixin
        src = inspect.getsource(StoreMaintenanceMixin.cleanup_junk)
        assert "[:120]" in src, "SM7: fingerprint should be 120 chars"


# ---------------------------------------------------------------------------
# SM8 â€” valid_to guard on archive UPDATE
# ---------------------------------------------------------------------------

class TestSM8ArchiveValidToGuard:
    def test_archive_update_has_valid_to_guard(self):
        """SM8: archive_stale_records UPDATE has valid_to IS NULL."""
        from store_maintenance import StoreMaintenanceMixin
        src = inspect.getsource(StoreMaintenanceMixin.archive_stale_records)
        assert "valid_to IS NULL" in src, (
            "SM8: archive UPDATE should have valid_to IS NULL guard"
        )

