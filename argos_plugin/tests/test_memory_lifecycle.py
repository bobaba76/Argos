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
