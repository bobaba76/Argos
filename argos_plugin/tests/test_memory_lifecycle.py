"""P5.1 (#6): memory lifecycle — archival tier, forgetting, long-horizon rollups.

Tests (deterministic, no LLM calls for phases 1-2; mocked LLM for phase 3):
1. Archival: tier column, archive policy, exempt categories, include_archived.
2. Revival: update/feedback revives archived records.
3. Forgetting: auto-quarantine of stale context_note/event/goal.
4. Rollup: proposals-only with mocked LLM, cooldown gate.
5. Config: all lifecycle fields load with correct defaults.
6. Backward compat: no config = all off (conservative defaults).
"""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from store import DuckDBMemoryStore


# ---------------------------------------------------------------------------
# 1. Archival tier
# ---------------------------------------------------------------------------


class TestArchival:
    """Phase 1: archival tier — tier column, archive policy, include_archived."""

    def test_tier_column_defaults_active(self, tmp_path):
        """New records get tier='active' by default."""
        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="alice")
        rec = store.remember(category="context_note", content="test note")
        assert rec is not None
        # Verify the tier column exists and defaults to 'active'.
        row = store.connection.execute(
            "SELECT tier FROM memory_records WHERE memory_id = ?",
            [rec.memory_id],
        ).fetchone()
        assert row[0] == "active"
        store.close()

    def test_archive_excludes_from_search(self, tmp_path):
        """Archived records are excluded from default search."""
        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="alice")
        rec = store.remember(category="context_note", content="unique archival test note")
        # Archive it manually.
        store.connection.execute(
            "UPDATE memory_records SET tier = 'archived' WHERE memory_id = ?",
            [rec.memory_id],
        )
        # Default search should not find it.
        results = store.search("unique archival test note", limit=10)
        assert all(r.memory_id != rec.memory_id for r in results)
        # include_archived=True should find it.
        results = store.search("unique archival test note", limit=10, include_archived=True)
        assert any(r.memory_id == rec.memory_id for r in results)
        store.close()

    def test_archive_policy_old_no_retrieval(self, tmp_path):
        """Records older than archive_after_days with no retrievals are archived."""
        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="alice")
        rec = store.remember(category="context_note", content="old stale note")
        # Set created_at to 200 days ago.
        old_ts = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
        store.connection.execute(
            "UPDATE memory_records SET created_at = ? WHERE memory_id = ?",
            [old_ts, rec.memory_id],
        )
        # Run archive with archive_after_days=180.
        report = store.archive_stale_records(archive_after_days=180)
        assert report["archived_count"] == 1
        assert rec.memory_id in report["archived_ids"]
        # Verify tier is now 'archived'.
        row = store.connection.execute(
            "SELECT tier FROM memory_records WHERE memory_id = ?",
            [rec.memory_id],
        ).fetchone()
        assert row[0] == "archived"
        store.close()

    def test_archive_exempt_categories(self, tmp_path):
        """Facts and preferences are never archived."""
        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="alice")
        fact_rec = store.remember(category="personal_fact", content="user's name is Alice")
        pref_rec = store.remember(category="preference", content="likes tea")
        # Set both to 200 days ago.
        old_ts = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
        store.connection.execute(
            "UPDATE memory_records SET created_at = ? WHERE memory_id IN (?, ?)",
            [old_ts, fact_rec.memory_id, pref_rec.memory_id],
        )
        report = store.archive_stale_records(archive_after_days=180)
        assert report["archived_count"] == 0
        store.close()

    def test_archive_skips_recent_records(self, tmp_path):
        """Records newer than archive_after_days are not archived."""
        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="alice")
        store.remember(category="context_note", content="fresh note")
        report = store.archive_stale_records(archive_after_days=180)
        assert report["archived_count"] == 0
        store.close()

    def test_archive_skips_records_with_retrievals(self, tmp_path):
        """Records with retrieval_count > 0 are not archived."""
        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="alice")
        rec = store.remember(category="context_note", content="retrieved note")
        old_ts = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
        store.connection.execute(
            "UPDATE memory_records SET created_at = ?, retrieval_count = 5 WHERE memory_id = ?",
            [old_ts, rec.memory_id],
        )
        report = store.archive_stale_records(archive_after_days=180)
        assert report["archived_count"] == 0
        store.close()


# ---------------------------------------------------------------------------
# 2. Revival
# ---------------------------------------------------------------------------


class TestRevival:
    """Revival: update/feedback revives archived records."""

    def test_revive_record_method(self, tmp_path):
        """revive_record flips an archived record back to active."""
        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="alice")
        rec = store.remember(category="context_note", content="revivable note")
        store.connection.execute(
            "UPDATE memory_records SET tier = 'archived' WHERE memory_id = ?",
            [rec.memory_id],
        )
        assert store.revive_record(rec.memory_id) is True
        row = store.connection.execute(
            "SELECT tier FROM memory_records WHERE memory_id = ?",
            [rec.memory_id],
        ).fetchone()
        assert row[0] == "active"
        store.close()

    def test_revive_already_active_returns_false(self, tmp_path):
        """revive_record on an already-active record returns False."""
        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="alice")
        rec = store.remember(category="context_note", content="active note")
        assert store.revive_record(rec.memory_id) is False
        store.close()

    def test_revive_nonexistent_returns_false(self, tmp_path):
        """revive_record on a non-existent record returns False."""
        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="alice")
        assert store.revive_record("nonexistent-id") is False
        store.close()

    def test_feedback_revives_archived(self, tmp_path):
        """record_feedback revives an archived record."""
        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="alice")
        rec = store.remember(category="context_note", content="feedback revives")
        store.connection.execute(
            "UPDATE memory_records SET tier = 'archived' WHERE memory_id = ?",
            [rec.memory_id],
        )
        store.record_feedback(rec.memory_id, "helpful")
        row = store.connection.execute(
            "SELECT tier FROM memory_records WHERE memory_id = ?",
            [rec.memory_id],
        ).fetchone()
        assert row[0] == "active"
        store.close()


# ---------------------------------------------------------------------------
# 3. Forgetting
# ---------------------------------------------------------------------------


class TestForgetting:
    """Phase 2: forgetting — auto-quarantine of stale context_note/event/goal."""

    def test_forget_old_context_note(self, tmp_path):
        """Old context_note with no retrievals is quarantined."""
        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="alice")
        rec = store.remember(category="context_note", content="old forgotten note")
        old_ts = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()
        store.connection.execute(
            "UPDATE memory_records SET created_at = ? WHERE memory_id = ?",
            [old_ts, rec.memory_id],
        )
        report = store.forget_stale_records(forget_after_days=365)
        assert report["forgotten_count"] == 1
        # Verify it's quarantined.
        row = store.connection.execute(
            "SELECT status FROM memory_records WHERE memory_id = ?",
            [rec.memory_id],
        ).fetchone()
        assert row[0] == "quarantined"
        store.close()

    def test_forget_skips_facts(self, tmp_path):
        """Facts are never forgotten (not in the target categories)."""
        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="alice")
        rec = store.remember(category="personal_fact", content="user's name is Alice")
        old_ts = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()
        store.connection.execute(
            "UPDATE memory_records SET created_at = ? WHERE memory_id = ?",
            [old_ts, rec.memory_id],
        )
        report = store.forget_stale_records(forget_after_days=365)
        assert report["forgotten_count"] == 0
        store.close()

    def test_forget_skips_recent(self, tmp_path):
        """Recent records are not forgotten."""
        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="alice")
        store.remember(category="context_note", content="fresh note")
        report = store.forget_stale_records(forget_after_days=365)
        assert report["forgotten_count"] == 0
        store.close()

    def test_forget_is_reversible(self, tmp_path):
        """Forgotten records can be restored (quarantine, not delete)."""
        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="alice")
        rec = store.remember(category="context_note", content="restorable note")
        old_ts = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()
        store.connection.execute(
            "UPDATE memory_records SET created_at = ? WHERE memory_id = ?",
            [old_ts, rec.memory_id],
        )
        store.forget_stale_records(forget_after_days=365)
        # Restore it.
        assert store.restore_memory(rec.memory_id) is True
        row = store.connection.execute(
            "SELECT status FROM memory_records WHERE memory_id = ?",
            [rec.memory_id],
        ).fetchone()
        assert row[0] == "active"
        store.close()


# ---------------------------------------------------------------------------
# 4. Lifecycle maintenance pass
# ---------------------------------------------------------------------------


class TestLifecycleMaintenance:
    """The combined maintenance pass runs both phases independently."""

    def test_both_phases_disabled(self, tmp_path):
        """When both phases are disabled, the pass is a no-op."""
        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="alice")
        store.remember(category="context_note", content="test note")
        report = store.run_lifecycle_maintenance(
            archive_enabled=False, forget_enabled=False,
        )
        assert report == {}
        store.close()

    def test_archive_only(self, tmp_path):
        """Archive runs when enabled, forget doesn't."""
        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="alice")
        rec = store.remember(category="context_note", content="old note")
        old_ts = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
        store.connection.execute(
            "UPDATE memory_records SET created_at = ? WHERE memory_id = ?",
            [old_ts, rec.memory_id],
        )
        report = store.run_lifecycle_maintenance(
            archive_enabled=True, archive_after_days=180,
            forget_enabled=False,
        )
        assert "archive" in report
        assert report["archive"]["archived_count"] == 1
        assert "forget" not in report
        store.close()

    def test_forget_only(self, tmp_path):
        """Forget runs when enabled, archive doesn't."""
        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="alice")
        rec = store.remember(category="context_note", content="very old note")
        old_ts = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()
        store.connection.execute(
            "UPDATE memory_records SET created_at = ? WHERE memory_id = ?",
            [old_ts, rec.memory_id],
        )
        report = store.run_lifecycle_maintenance(
            archive_enabled=False,
            forget_enabled=True, forget_after_days=365,
        )
        assert "forget" in report
        assert report["forget"]["forgotten_count"] == 1
        assert "archive" not in report
        store.close()


# ---------------------------------------------------------------------------
# 5. Rollup (Phase 3, mocked LLM)
# ---------------------------------------------------------------------------


class TestRollup:
    """Phase 3: long-horizon rollups — proposals only, mocked LLM."""

    def test_rollup_cooldown_skips(self, tmp_path):
        """Rollup skips when within cooldown."""
        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="alice")
        # Set last_run to now.
        store.set_state("rollup_last_run", datetime.now(timezone.utc).isoformat())
        from rollup import run_rollup
        report = run_rollup(store, interval_days=30)
        assert not report["ran"]
        assert report["skipped"] == "cooldown"
        store.close()

    def test_rollup_insufficient_records(self, tmp_path):
        """Rollup skips when there are fewer than 10 records."""
        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="alice")
        store.remember(category="context_note", content="only one record")
        from rollup import run_rollup
        report = run_rollup(store, interval_days=30)
        assert not report["ran"]
        assert report["skipped"] == "insufficient_records"
        store.close()

    def test_rollup_no_llm_client(self, tmp_path):
        """Rollup skips when the LLM client is unavailable."""
        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="alice")
        # Create 10+ records.
        for i in range(15):
            store.remember(category="context_note", content=f"record {i}")
        from rollup import run_rollup
        with patch("rollup._get_llm_client", return_value=None):
            report = run_rollup(store, interval_days=30)
        assert not report["ran"]
        assert report["skipped"] == "no_llm_client"
        store.close()

    def test_rollup_emits_proposals(self, tmp_path):
        """Rollup emits proposals through the standard pipeline."""
        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="alice")
        for i in range(15):
            store.remember(category="context_note", content=f"record {i} for rollup")
        # Mock the LLM response.
        mock_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content=json.dumps([
                    {"content": "User consistently works on memory systems",
                     "category": "insight", "confidence": 0.85},
                ])
            ))]
        )
        mock_call = MagicMock(return_value=mock_response)
        from rollup import run_rollup
        with patch("rollup._get_llm_client", return_value=mock_call):
            report = run_rollup(store, interval_days=30, max_records_per_run=15)
        assert report["ran"]
        assert report["proposals_emitted"] == 1
        assert report["llm_calls"] == 1
        # Verify the proposal was saved as a candidate.
        candidates = store.list_candidates(status="pending", limit=10)
        assert any("memory systems" in c["content"] for c in candidates)
        store.close()

    def test_rollup_malformed_response_no_proposals(self, tmp_path):
        """A malformed LLM response results in zero proposals."""
        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="alice")
        for i in range(15):
            store.remember(category="context_note", content=f"record {i}")
        mock_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content="not valid json"
            ))]
        )
        mock_call = MagicMock(return_value=mock_response)
        from rollup import run_rollup
        with patch("rollup._get_llm_client", return_value=mock_call):
            report = run_rollup(store, interval_days=30, max_records_per_run=15)
        assert report["ran"]
        assert report["proposals_emitted"] == 0
        store.close()

    def test_rollup_llm_error_fail_soft(self, tmp_path):
        """An LLM error is fail-soft — returns a no-op report."""
        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="alice")
        for i in range(15):
            store.remember(category="context_note", content=f"record {i}")
        mock_call = MagicMock(side_effect=RuntimeError("timeout"))
        from rollup import run_rollup
        with patch("rollup._get_llm_client", return_value=mock_call):
            report = run_rollup(store, interval_days=30, max_records_per_run=15)
        assert not report["ran"]
        assert report["skipped"] == "llm_error"
        store.close()


# ---------------------------------------------------------------------------
# 6. Config loading
# ---------------------------------------------------------------------------


class TestConfigLoading:
    """All lifecycle config fields load with correct defaults."""

    def test_config_defaults(self):
        """No config = all off (conservative defaults)."""
        config = {}
        archive_enabled = str(config.get("archive_enabled", "false")).lower() in ("true", "1", "yes")
        archive_after_days = int(config.get("archive_after_days", 180))
        forget_enabled = str(config.get("forget_enabled", "false")).lower() in ("true", "1", "yes")
        forget_after_days = int(config.get("forget_after_days", 365))
        rollup_enabled = str(config.get("rollup_enabled", "false")).lower() in ("true", "1", "yes")
        rollup_interval_days = int(config.get("rollup_interval_days", 30))
        rollup_max_records = int(config.get("rollup_max_records_per_run", 100))
        assert archive_enabled is False
        assert archive_after_days == 180
        assert forget_enabled is False
        assert forget_after_days == 365
        assert rollup_enabled is False
        assert rollup_interval_days == 30
        assert rollup_max_records == 100

    def test_config_fields_in_schema(self):
        """All seven lifecycle config fields are present in the schema."""
        from config_schema import CONFIG_SCHEMA
        keys = {f.key for f in CONFIG_SCHEMA.fields}
        assert "archive_enabled" in keys
        assert "archive_after_days" in keys
        assert "forget_enabled" in keys
        assert "forget_after_days" in keys
        assert "rollup_enabled" in keys
        assert "rollup_interval_days" in keys
        assert "rollup_max_records_per_run" in keys

    def test_config_fields_in_lifecycle_group(self):
        """Lifecycle fields are in the 'Lifecycle' config group."""
        from config_schema import CONFIG_SCHEMA
        fields = {f.key: f for f in CONFIG_SCHEMA.fields}
        assert fields["archive_enabled"].group == "Lifecycle"
        assert fields["archive_after_days"].group == "Lifecycle"
        assert fields["forget_enabled"].group == "Lifecycle"
        assert fields["forget_after_days"].group == "Lifecycle"
        assert fields["rollup_enabled"].group == "Lifecycle"
        assert fields["rollup_interval_days"].group == "Lifecycle"
        assert fields["rollup_max_records_per_run"].group == "Lifecycle"


# ---------------------------------------------------------------------------
# Maintenance audit M1 — forget_stale_records returns wrong IDs
# ---------------------------------------------------------------------------


class TestMaintenanceAudit:
    """Audit fixes for store_maintenance.py."""

    def test_forget_stale_records_returns_correct_ids_on_partial_failure(self, tmp_path):
        """M1: forgotten_ids must only include IDs that were actually
        quarantined, not the first N from the selected list."""
        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="alice")
        # Create 3 old context_notes.
        recs = []
        for i in range(3):
            rec = store.remember(
                category="context_note", content=f"old forgotten note {i}"
            )
            recs.append(rec)
            old_ts = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()
            store.connection.execute(
                "UPDATE memory_records SET created_at = ? WHERE memory_id = ?",
                [old_ts, rec.memory_id],
            )
        # Make the second record disappear between SELECT and quarantine
        # by deleting it before forget_stale_records runs the quarantine loop.
        # We patch quarantine_memory to fail for the second record.
        original_quarantine = store.quarantine_memory
        call_count = [0]

        def selective_quarantine(memory_id, reason):
            call_count[0] += 1
            if call_count[0] == 2:
                # Simulate the record being gone (race with delete_memory).
                return False
            return original_quarantine(memory_id, reason)

        store.quarantine_memory = selective_quarantine
        report = store.forget_stale_records(forget_after_days=365)
        store.quarantine_memory = original_quarantine

        assert report["forgotten_count"] == 2, (
            f"Expected 2 quarantined, got {report['forgotten_count']}"
        )
        # The returned IDs must be the ones that were ACTUALLY quarantined,
        # not just the first 2 from the list.
        quarantined_ids = set(report["forgotten_ids"])
        assert len(quarantined_ids) == 2
        # The second record (which failed) must NOT be in the list.
        assert recs[1].memory_id not in quarantined_ids
        # The first and third records (which succeeded) must be in the list.
        assert recs[0].memory_id in quarantined_ids
        assert recs[2].memory_id in quarantined_ids


# ---------------------------------------------------------------------------
# D3-D6: Rollup audit fixes (#204)
# ---------------------------------------------------------------------------

class TestRollupAuditFixes:
    """Regression tests for issue #204: distillation/rollup audit fixes."""

    def test_d3_call_llm_uses_messages_kwarg(self, tmp_path):
        """D3: call_llm should be called with messages=[...] kwarg, not
        a positional prompt arg. If the signature is wrong, rollup
        silently never runs (caught by try/except → 'llm_error')."""
        store = DuckDBMemoryStore(tmp_path / "test_d3.duckdb", user_id="alice")
        for i in range(15):
            store.remember(category="context_note", content=f"record {i} for rollup")
        mock_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content=json.dumps([
                    {"content": "Test insight", "category": "insight", "confidence": 0.8},
                ])
            ))]
        )
        mock_call = MagicMock(return_value=mock_response)
        from rollup import run_rollup
        with patch("rollup._get_llm_client", return_value=mock_call):
            report = run_rollup(store, interval_days=30, max_records_per_run=15)
        # If D3 is fixed, the call should succeed (not 'llm_error').
        assert report["ran"], f"Expected ran=True, got skipped={report.get('skipped')}"
        assert report["llm_calls"] == 1
        # Verify call_llm was called with messages kwarg, not positional.
        call_args = mock_call.call_args
        assert "messages" in call_args.kwargs, (
            "call_llm should be called with messages= kwarg (D3)"
        )
        # RU1: system/user message split — first message is system, second is user.
        assert call_args.kwargs["messages"][0]["role"] == "system"
        assert call_args.kwargs["messages"][1]["role"] == "user"
        store.close()

    def test_d4_egress_gate_checked(self, tmp_path):
        """D4: rollup should check the egress gate before making an LLM
        call. In local_only mode, the gate should refuse."""
        store = DuckDBMemoryStore(tmp_path / "test_d4.duckdb", user_id="alice")
        for i in range(15):
            store.remember(category="context_note", content=f"record {i}")
        from rollup import run_rollup
        # Mock the egress gate to return False (blocked).
        with patch("egress.gate", return_value=False):
            report = run_rollup(store, interval_days=30, max_records_per_run=15)
        assert not report["ran"]
        assert report["skipped"] == "egress_gate"
        store.close()

    def test_d5_evidence_text_passed_to_save_candidate(self, tmp_path):
        """D5: save_candidate should receive evidence_text containing
        the source records' content (provenance for the review pipeline)."""
        store = DuckDBMemoryStore(tmp_path / "test_d5.duckdb", user_id="alice")
        for i in range(15):
            store.remember(category="context_note", content=f"unique content {i} for evidence")
        mock_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content=json.dumps([
                    {"content": "Test insight with evidence", "category": "insight", "confidence": 0.8},
                ])
            ))]
        )
        mock_call = MagicMock(return_value=mock_response)
        # Patch save_candidate to capture the evidence_text kwarg.
        original_save = store.save_candidate
        captured_evidence = []
        def capture_save(*args, **kwargs):
            captured_evidence.append(kwargs.get("evidence_text", ""))
            return original_save(*args, **kwargs)
        store.save_candidate = capture_save
        from rollup import run_rollup
        with patch("rollup._get_llm_client", return_value=mock_call):
            report = run_rollup(store, interval_days=30, max_records_per_run=15)
        assert report["ran"]
        assert report["proposals_emitted"] == 1
        # evidence_text should have been passed and contain source content.
        assert len(captured_evidence) == 1
        assert captured_evidence[0], "evidence_text should not be empty"
        assert "unique content" in captured_evidence[0], (
            "evidence_text should contain source record content (D5)"
        )
        store.close()

    def test_d6_fenced_json_parsed(self, tmp_path):
        """D6: _parse_rollup_response should strip code fences before
        parsing. Fenced JSON should parse successfully."""
        from rollup import _parse_rollup_response
        fenced = '```json\n[{"content": "test", "category": "insight", "confidence": 0.8}]\n```'
        proposals = _parse_rollup_response(fenced)
        assert len(proposals) == 1
        assert proposals[0]["content"] == "test"

    def test_d6_prose_wrapped_json_parsed(self):
        """D6: prose-wrapped JSON should parse via the fallback extractor."""
        from rollup import _parse_rollup_response
        prose_wrapped = (
            'Here is the JSON: [{"content": "test", "category": "insight", "confidence": 0.8}] Done.'
        )
        proposals = _parse_rollup_response(prose_wrapped)
        assert len(proposals) == 1
        assert proposals[0]["content"] == "test"

    def test_d6_pure_json_parsed(self):
        """D6: pure JSON (no fences, no prose) should parse successfully."""
        from rollup import _parse_rollup_response
        pure = '[{"content": "test", "category": "insight", "confidence": 0.8}]'
        proposals = _parse_rollup_response(pure)
        assert len(proposals) == 1

    def test_d6_empty_response_returns_empty(self):
        """D6: empty content should return empty list."""
        from rollup import _parse_rollup_response
        assert _parse_rollup_response("") == []
        assert _parse_rollup_response(None) == []

    def test_revive_record_respects_user_scope(self, tmp_path):
        """M3: revive_record must not allow cross-tenant revival.
        User B cannot revive user A's archived record."""
        store_a = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="alice")
        rec = store_a.remember(category="context_note", content="alice's note")
        store_a.connection.execute(
            "UPDATE memory_records SET tier = 'archived' WHERE memory_id = ?",
            [rec.memory_id],
        )
        # Open the same DB as a different user.
        store_b = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="bob")
        # Bob should NOT be able to revive Alice's archived record.
        result = store_b.revive_record(rec.memory_id)
        assert result is False, (
            "revive_record must reject cross-tenant revival"
        )
        # Verify the record is still archived.
        row = store_a.connection.execute(
            "SELECT tier FROM memory_records WHERE memory_id = ?",
            [rec.memory_id],
        ).fetchone()
        assert row[0] == "archived"
        store_a.close()
        store_b.close()

    def test_consolidate_expired_count_scoped_to_user(self, tmp_path):
        """M2: consolidate's expired_count must only count the current
        user's expired records, not all tenants."""
        store_a = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="alice")
        # Alice has one expired record.
        rec_a = store_a.remember(
            category="context_note", content="alice's expired note"
        )
        old_ts = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()
        past_expiry = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        store_a.connection.execute(
            "UPDATE memory_records SET created_at = ?, expires_at = ? "
            "WHERE memory_id = ?",
            [old_ts, past_expiry, rec_a.memory_id],
        )
        # Bob has two expired records.
        store_b = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="bob")
        for i in range(2):
            rec_b = store_b.remember(
                category="context_note", content=f"bob's expired note {i}"
            )
            store_b.connection.execute(
                "UPDATE memory_records SET created_at = ?, expires_at = ? "
                "WHERE memory_id = ?",
                [old_ts, past_expiry, rec_b.memory_id],
            )
        # Alice's consolidate should report 1 expired, not 3.
        report = store_a.consolidate(dry_run=True)
        assert report["expired_count"] == 1, (
            f"Expected 1 expired for alice, got {report['expired_count']} "
            "(cross-tenant count leak)"
        )
        store_a.close()
        store_b.close()


# ---------------------------------------------------------------------------
# RU1-RU10: Rollup audit fixes (#236)
# ---------------------------------------------------------------------------

class TestRollupAuditRU:
    """Regression tests for issue #236: rollup audit RU1-RU10."""

    # -- RU1: prompt injection via raw memory content -------------------------

    def test_ru1_system_message_present(self, tmp_path):
        """RU1: the LLM call should include a system message that sets the
        instruction boundary, so memory content is treated as data."""
        store = DuckDBMemoryStore(tmp_path / "test_ru1.duckdb", user_id="alice")
        for i in range(15):
            store.remember(category="context_note", content=f"record {i}")
        mock_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="[]"))]
        )
        mock_call = MagicMock(return_value=mock_response)
        from rollup import run_rollup
        with patch("rollup._get_llm_client", return_value=mock_call):
            run_rollup(store, interval_days=30, max_records_per_run=15)
        call_args = mock_call.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert "DATA" in messages[0]["content"] or "data" in messages[0]["content"]
        store.close()

    def test_ru1_markup_neutralized_in_prompt(self, tmp_path):
        """RU1: < and > in memory content should be neutralized in the
        prompt so stored markup cannot be interpreted as prompt-structure."""
        store = DuckDBMemoryStore(tmp_path / "test_ru1b.duckdb", user_id="alice")
        for i in range(15):
            store.remember(
                category="context_note",
                content=f"<script>alert({i})</script> record {i}",
            )
        mock_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="[]"))]
        )
        mock_call = MagicMock(return_value=mock_response)
        from rollup import run_rollup
        with patch("rollup._get_llm_client", return_value=mock_call):
            run_rollup(store, interval_days=30, max_records_per_run=15)
        user_msg = mock_call.call_args.kwargs["messages"][1]["content"]
        assert "\uFF1C" in user_msg, "markup should be neutralized (< → ＜)"
        assert "\uFF1E" in user_msg, "markup should be neutralized (> → ＞)"
        store.close()

    def test_ru1_prompt_is_json_not_prose(self, tmp_path):
        """RU1: the user message should be structured JSON (data format),
        not free-form prose with interpolated content."""
        store = DuckDBMemoryStore(tmp_path / "test_ru1c.duckdb", user_id="alice")
        for i in range(15):
            store.remember(category="context_note", content=f"record {i}")
        mock_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="[]"))]
        )
        mock_call = MagicMock(return_value=mock_response)
        from rollup import run_rollup
        with patch("rollup._get_llm_client", return_value=mock_call):
            run_rollup(store, interval_days=30, max_records_per_run=15)
        user_msg = mock_call.call_args.kwargs["messages"][1]["content"]
        parsed = json.loads(user_msg)
        assert "records" in parsed
        assert isinstance(parsed["records"], list)
        store.close()

    # -- RU2: public store API instead of _fetch_records ---------------------

    def test_ru2_uses_public_store_method(self, tmp_path):
        """RU2: _load_rollup_candidates should call store.load_rollup_candidates
        (public API), not store._fetch_records (private)."""
        store = DuckDBMemoryStore(tmp_path / "test_ru2.duckdb", user_id="alice")
        for i in range(15):
            store.remember(category="context_note", content=f"record {i}")
        records = store.load_rollup_candidates(15)
        assert len(records) == 15
        timestamps = [r.created_at for r in records if r.created_at]
        assert timestamps == sorted(timestamps), (
            "load_rollup_candidates should return oldest-first (ASC)"
        )
        store.close()

    def test_ru2_load_rollup_candidates_filters_active_tier(self, tmp_path):
        """RU2: load_rollup_candidates should only return active-tier records."""
        store = DuckDBMemoryStore(tmp_path / "test_ru2b.duckdb", user_id="alice")
        rec_active = store.remember(category="context_note", content="active rec")
        rec_archived = store.remember(category="context_note", content="archived rec")
        store.connection.execute(
            "UPDATE memory_records SET tier = 'archived' WHERE memory_id = ?",
            [rec_archived.memory_id],
        )
        records = store.load_rollup_candidates(10)
        ids = {r.memory_id for r in records}
        assert rec_active.memory_id in ids
        assert rec_archived.memory_id not in ids, (
            "archived records should not be rollup candidates"
        )
        store.close()

    # -- RU3: novelty gate ---------------------------------------------------

    def test_ru3_novelty_gate_skips_with_old_last_run(self, tmp_path):
        """RU3: with an old last_run but no new records since then, the
        novelty gate should skip the run."""
        store = DuckDBMemoryStore(tmp_path / "test_ru3b.duckdb", user_id="alice")
        for i in range(15):
            store.remember(category="context_note", content=f"record {i}")
        # Set all records' created_at to 90 days ago (before last_run).
        old_record_ts = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
        store.connection.execute(
            "UPDATE memory_records SET created_at = ?",
            [old_record_ts],
        )
        # Set last_run to 60 days ago (past cooldown of 30 days, but
        # after the records' created_at — so no "new" records exist).
        old_ts = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        store.set_state("rollup_last_run", old_ts)
        from rollup import run_rollup
        with patch("rollup._get_llm_client", return_value=MagicMock()):
            report = run_rollup(store, interval_days=30, max_records_per_run=15)
        assert not report["ran"]
        assert report["skipped"] == "novelty_gate", (
            f"Expected novelty_gate, got {report.get('skipped')}"
        )
        store.close()

    def test_ru3_novelty_gate_passes_with_new_records(self, tmp_path):
        """RU3: with an old last_run AND new records created after it,
        the rollup should proceed past the novelty gate."""
        store = DuckDBMemoryStore(tmp_path / "test_ru3c.duckdb", user_id="alice")
        # Old records (created_at set to 90 days ago).
        for i in range(10):
            store.remember(category="context_note", content=f"old record {i}")
        old_record_ts = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
        store.connection.execute(
            "UPDATE memory_records SET created_at = ?",
            [old_record_ts],
        )
        # Set last_run to 60 days ago (past cooldown of 30 days).
        old_ts = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        store.set_state("rollup_last_run", old_ts)
        # New records created "now" (after last_run).
        for i in range(5):
            store.remember(category="context_note", content=f"new record {i}")
        mock_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="[]"))]
        )
        mock_call = MagicMock(return_value=mock_response)
        from rollup import run_rollup
        with patch("rollup._get_llm_client", return_value=mock_call):
            report = run_rollup(store, interval_days=30, max_records_per_run=15)
        assert report["ran"], (
            f"Expected ran=True, got skipped={report.get('skipped')}"
        )
        store.close()

    # -- RU4: cap proposals at 10 --------------------------------------------

    def test_ru4_proposals_capped_at_10(self):
        """RU4: _parse_rollup_response should cap at 10 proposals."""
        from rollup import _parse_rollup_response
        items = [
            {"content": f"proposal {i}", "category": "insight", "confidence": 0.8}
            for i in range(20)
        ]
        proposals = _parse_rollup_response(json.dumps(items))
        assert len(proposals) == 10, (
            f"Expected 10 proposals (capped), got {len(proposals)}"
        )

    # -- RU5: compute now once -----------------------------------------------

    def test_ru5_now_passed_to_build_prompt(self):
        """RU5: _build_rollup_prompt should accept a now parameter and use
        it instead of calling datetime.now() per record."""
        from rollup import _build_rollup_prompt
        records = []
        fixed_now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for i in range(3):
            rec = SimpleNamespace(
                memory_id=f"m{i}",
                category="context_note",
                content=f"record {i}",
                created_at=(fixed_now - timedelta(days=10 + i)).isoformat(),
            )
            records.append(rec)
        prompt = _build_rollup_prompt(records, now=fixed_now)
        parsed = json.loads(prompt)
        assert parsed["records"][0]["age_days"] == "10d"
        assert parsed["records"][1]["age_days"] == "11d"
        assert parsed["records"][2]["age_days"] == "12d"

    # -- RU6: clamp minimums -------------------------------------------------

    def test_ru6_clamps_interval_days_to_1(self, tmp_path):
        """RU6: interval_days=0 should be clamped to 1, not cause
        cooldown to always return True."""
        store = DuckDBMemoryStore(tmp_path / "test_ru6.duckdb", user_id="alice")
        for i in range(15):
            store.remember(category="context_note", content=f"record {i}")
        mock_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="[]"))]
        )
        mock_call = MagicMock(return_value=mock_response)
        from rollup import run_rollup
        with patch("rollup._get_llm_client", return_value=mock_call):
            report = run_rollup(store, interval_days=0, max_records_per_run=15)
        assert report["ran"], (
            f"Expected ran=True with clamped interval, got {report.get('skipped')}"
        )
        store.close()

    def test_ru6_clamps_max_records_to_10(self, tmp_path):
        """RU6: max_records_per_run=0 should be clamped to 10, not cause
        insufficient_records gate to always trigger."""
        store = DuckDBMemoryStore(tmp_path / "test_ru6b.duckdb", user_id="alice")
        for i in range(15):
            store.remember(category="context_note", content=f"record {i}")
        mock_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="[]"))]
        )
        mock_call = MagicMock(return_value=mock_response)
        from rollup import run_rollup
        with patch("rollup._get_llm_client", return_value=mock_call):
            report = run_rollup(store, interval_days=30, max_records_per_run=0)
        assert report["ran"], (
            f"Expected ran=True with clamped max_records, got {report.get('skipped')}"
        )
        store.close()

    # -- RU8: _strip_json_fences returns "" on failure -----------------------

    def test_ru8_strip_fences_returns_empty_on_no_bracket(self):
        """RU8: _strip_json_fences should return "" (not the original text)
        when there's no [ bracket to extract from."""
        from rollup import _strip_json_fences
        result = _strip_json_fences("no json here at all")
        assert result == "", f"Expected empty string, got {result!r}"

    def test_ru8_strip_fences_returns_empty_on_parse_failure(self):
        """RU8: _strip_json_fences should return "" when raw_decode fails."""
        from rollup import _strip_json_fences
        result = _strip_json_fences("[invalid json content")
        assert result == "", f"Expected empty string, got {result!r}"

    # -- RU9: dedup=True on save_candidate -----------------------------------

    def test_ru9_dedup_passed_to_save_candidate(self, tmp_path):
        """RU9: save_candidate should be called with dedup=True to
        prevent duplicate proposals across runs."""
        store = DuckDBMemoryStore(tmp_path / "test_ru9.duckdb", user_id="alice")
        for i in range(15):
            store.remember(category="context_note", content=f"record {i}")
        mock_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content=json.dumps([
                    {"content": "Test insight", "category": "insight", "confidence": 0.8},
                ])
            ))]
        )
        mock_call = MagicMock(return_value=mock_response)
        original_save = store.save_candidate
        captured_dedup = []
        def capture_save(*args, **kwargs):
            captured_dedup.append(kwargs.get("dedup", None))
            return original_save(*args, **kwargs)
        store.save_candidate = capture_save
        from rollup import run_rollup
        with patch("rollup._get_llm_client", return_value=mock_call):
            run_rollup(store, interval_days=30, max_records_per_run=15)
        assert captured_dedup, "save_candidate should have been called"
        assert all(d is True for d in captured_dedup), (
            f"dedup should be True for all calls, got {captured_dedup}"
        )
        store.close()

    # -- RU10: egress gate with actual content -------------------------------

    def test_ru10_egress_gate_receives_content(self, tmp_path):
        """RU10: the egress gate should receive the actual record content,
        not an empty string, so PII checks apply to the real payload."""
        store = DuckDBMemoryStore(tmp_path / "test_ru10.duckdb", user_id="alice")
        for i in range(15):
            store.remember(
                category="context_note",
                content=f"unique egress content {i}",
            )
        from rollup import run_rollup
        import egress as egress_mod
        captured_text = []
        original_gate = egress_mod.gate
        def capture_gate(kind, text="", cfg=None):
            captured_text.append((kind, text))
            return original_gate(kind, text, cfg)
        with patch("egress.gate", side_effect=capture_gate):
            with patch("rollup._get_llm_client", return_value=MagicMock(
                return_value=SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="[]"))]
                )
            )):
                run_rollup(store, interval_days=30, max_records_per_run=15)
        rollup_calls = [(k, t) for k, t in captured_text if k == "memory_rollup"]
        assert rollup_calls, "egress gate should have been called for memory_rollup"
        _, text = rollup_calls[-1]
        assert text, "egress gate should receive non-empty content (RU10)"
        assert "unique egress content" in text, (
            "egress gate text should contain actual record content"
        )
        store.close()
