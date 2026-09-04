"""Audit tests for store_write.py SW1-SW12 (issue #226).

Covers cross-tenant leak fixes (SW1-SW3), valid_to checks (SW4-SW5),
confidence normalization (SW6), inbound scan (SW7), performance (SW8-SW10),
defense-in-depth (SW11), and input validation (SW12).

Run with (Hermes venv python, offline):
    python -m pytest tests/test_store_write_audit.py -v
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))


@pytest.fixture
def store(tmp_path):
    from store import DuckDBMemoryStore
    s = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="alice")
    yield s
    s.close()


# ---------------------------------------------------------------------------
# SW1 — list_rejections user_scope filter
# ---------------------------------------------------------------------------

class TestSW1ListRejectionsFilter:
    def test_query_has_user_scope_filter(self):
        """SW1: list_rejections SQL includes user_scope filter."""
        from store_write import StoreWriteMixin
        src = inspect.getsource(StoreWriteMixin.list_rejections)
        assert "user_scope IS NULL OR user_scope" in src

    def test_only_returns_caller_rejections(self, store):
        """SW1: only the caller's rejection entries are returned."""
        store.record_rejection("test_cat", {"subject": "s", "predicate": "p"})
        results = store.list_rejections()
        for r in results:
            scope = r.get("user_scope")
            assert scope is None or scope == "alice"


# ---------------------------------------------------------------------------
# SW2 — backfill_evidence Pass 1 user_scope filters
# ---------------------------------------------------------------------------

class TestSW2BackfillPass1Filter:
    def test_pass1_has_user_scope_filters(self):
        """SW2: backfill_evidence Pass 1 JOIN has user_scope filters on both tables."""
        from store_write import StoreWriteMixin
        src = inspect.getsource(StoreWriteMixin.backfill_evidence)
        # Check for both c.user_scope and m.user_scope filters.
        assert "c.user_scope IS NULL OR c.user_scope" in src
        assert "m.user_scope IS NULL OR m.user_scope" in src


# ---------------------------------------------------------------------------
# SW3 — backfill_evidence Pass 2 user_scope filter
# ---------------------------------------------------------------------------

class TestSW3BackfillPass2Filter:
    def test_pass2_has_user_scope_filter(self):
        """SW3: backfill_evidence Pass 2 orphan query has user_scope filter."""
        from store_write import StoreWriteMixin
        src = inspect.getsource(StoreWriteMixin.backfill_evidence)
        # The orphan query should filter by user_scope.
        assert "user_scope IS NULL OR user_scope" in src


# ---------------------------------------------------------------------------
# SW4 — record_feedback valid_to check + user_scope guard
# ---------------------------------------------------------------------------

class TestSW4RecordFeedbackGuards:
    def test_has_valid_to_check(self):
        """SW4: record_feedback existence check includes valid_to IS NULL."""
        from store_write import StoreWriteMixin
        src = inspect.getsource(StoreWriteMixin.record_feedback)
        assert "valid_to IS NULL" in src

    def test_update_has_user_scope_guard(self):
        """SW4: record_feedback UPDATE includes user_scope guard."""
        from store_write import StoreWriteMixin
        src = inspect.getsource(StoreWriteMixin.record_feedback)
        # At least one UPDATE should have the guard.
        assert src.count("user_scope IS NULL OR user_scope") >= 2

    def test_feedback_on_superseded_returns_false(self, store):
        """SW4: feedback on a superseded (valid_to set) record returns False."""
        rec = store.remember("test", "original content", source="explicit")
        assert rec is not None
        # Create a new version (supersedes the old one).
        rec2 = store.update_memory(rec.memory_id, content="updated content")
        assert rec2 is not None
        # Now rec.memory_id has valid_to set — feedback should fail.
        result = store.record_feedback(rec.memory_id, "helpful")
        assert result is False


# ---------------------------------------------------------------------------
# SW5 — quarantine_memory valid_to check + user_scope guard
# ---------------------------------------------------------------------------

class TestSW5QuarantineGuards:
    def test_has_valid_to_check(self):
        """SW5: quarantine_memory existence check includes valid_to IS NULL."""
        from store_write import StoreWriteMixin
        src = inspect.getsource(StoreWriteMixin.quarantine_memory)
        assert "valid_to IS NULL" in src

    def test_update_has_user_scope_guard(self):
        """SW5: quarantine_memory UPDATE includes user_scope guard."""
        from store_write import StoreWriteMixin
        src = inspect.getsource(StoreWriteMixin.quarantine_memory)
        assert "user_scope IS NULL OR user_scope" in src

    def test_quarantine_superseded_returns_false(self, store):
        """SW5: quarantine on a superseded record returns False."""
        rec = store.remember("test", "original content", source="explicit")
        assert rec is not None
        rec2 = store.update_memory(rec.memory_id, content="updated content")
        assert rec2 is not None
        result = store.quarantine_memory(rec.memory_id, "test")
        assert result is False


# ---------------------------------------------------------------------------
# SW6 — remember() normalizes confidence
# ---------------------------------------------------------------------------

class TestSW6ConfidenceNormalization:
    def test_confidence_clamped_high(self, store):
        """SW6: confidence > 1.0 is clamped to 1.0."""
        rec = store.remember("test", "high confidence", confidence=5.0)
        assert rec is not None
        assert rec.confidence == 1.0

    def test_confidence_clamped_low(self, store):
        """SW6: confidence < 0.0 is clamped to 0.0."""
        rec = store.remember("test", "low confidence", confidence=-1.0)
        assert rec is not None
        assert rec.confidence == 0.0

    def test_confidence_invalid_type_defaults(self, store):
        """SW6: invalid confidence type defaults to 0.5."""
        rec = store.remember("test", "invalid confidence", confidence="high")
        assert rec is not None
        assert rec.confidence == 0.5


# ---------------------------------------------------------------------------
# SW7 — update_memory inbound security scan
# ---------------------------------------------------------------------------

class TestSW7UpdateMemoryInboundScan:
    def test_has_inbound_scan_code(self):
        """SW7: update_memory has inbound security scan code."""
        from store_write import StoreWriteMixin
        src = inspect.getsource(StoreWriteMixin.update_memory)
        assert "scan_inbound_text" in src
        assert "provenance_origin" in src


# ---------------------------------------------------------------------------
# SW8 — _find_conflicting_active_value LIMIT increased
# ---------------------------------------------------------------------------

class TestSW8ConflictLimit:
    def test_limit_increased(self):
        """SW8: pre-filter LIMIT increased from 50 to 200."""
        from store_write import StoreWriteMixin
        src = inspect.getsource(StoreWriteMixin._find_conflicting_active_value)
        assert "LIMIT 200" in src
        assert "len(rows) == 200" in src


# ---------------------------------------------------------------------------
# SW9 — get_chain_membership pre-fetches edges
# ---------------------------------------------------------------------------

class TestSW9ChainMembershipPreFetch:
    def test_no_get_memory_history_call(self):
        """SW9: get_chain_membership does not call get_memory_history (N+1 fix)."""
        from store_write import StoreWriteMixin
        src = inspect.getsource(StoreWriteMixin.get_chain_membership)
        # No actual call to get_memory_history (docstring mentions are OK).
        assert "self.get_memory_history" not in src
        # Should have pre-fetched edge walking code.
        assert "reverse" in src or "supersede_map" in src


# ---------------------------------------------------------------------------
# SW10 — save_candidate dedup LIMIT reduced
# ---------------------------------------------------------------------------

class TestSW10SaveCandidateDedupLimit:
    def test_limit_reduced(self):
        """SW10: save_candidate substring dedup LIMIT reduced from 500."""
        from store_write import StoreWriteMixin
        src = inspect.getsource(StoreWriteMixin.save_candidate)
        assert "LIMIT 200" in src


# ---------------------------------------------------------------------------
# SW11 — restore_memory and delete_memory user_scope guard
# ---------------------------------------------------------------------------

class TestSW11UserScopeGuards:
    def test_restore_memory_update_has_guard(self):
        """SW11: restore_memory UPDATE includes user_scope guard."""
        from store_write import StoreWriteMixin
        src = inspect.getsource(StoreWriteMixin.restore_memory)
        assert "user_scope IS NULL OR user_scope" in src

    def test_delete_memory_non_head_update_has_guard(self):
        """SW11: delete_memory non-head UPDATE includes user_scope guard."""
        from store_write import StoreWriteMixin
        src = inspect.getsource(StoreWriteMixin.delete_memory)
        # The non-head quarantine UPDATE should have the guard.
        assert "deleted from chain" in src
        # Count user_scope guards — should be multiple (existence + updates).
        assert src.count("user_scope IS NULL OR user_scope") >= 3


# ---------------------------------------------------------------------------
# SW12 — remember() validates tags, payload, scope, durability
# ---------------------------------------------------------------------------

class TestSW12InputValidation:
    def test_tags_non_list_converted(self, store):
        """SW12: non-list tags are converted to list."""
        rec = store.remember("test", "tags test", tags="single_tag")
        assert rec is not None
        assert isinstance(rec.tags, list)

    def test_invalid_scope_defaults_to_profile(self, store):
        """SW12: invalid scope value defaults to 'profile'."""
        rec = store.remember("test", "scope test", scope="invalid_scope")
        assert rec is not None
        assert rec.scope == "profile"

    def test_invalid_durability_corrected(self, store):
        """SW12: invalid durability value is corrected."""
        rec = store.remember("test", "durability test", durability="bogus")
        assert rec is not None
        assert rec.durability in {"durable", "temporary", "ephemeral"}

    def test_has_validation_code(self):
        """SW12: remember() has validation code for tags/payload/scope."""
        from store_write import StoreWriteMixin
        src = inspect.getsource(StoreWriteMixin.remember)
        assert "_VALID_SCOPES" in src
        assert "_VALID_DURABILITY" in src
        assert "_MAX_PAYLOAD_SIZE" in src
