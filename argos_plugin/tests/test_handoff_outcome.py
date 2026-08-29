"""Tests for #48: compile-to-handoff consumer + #16: procedure outcome records.

Covers:
- #48: Handoff block renders from store state (STATE, TODO, GOTCHAS)
- #48: Empty project → graceful "nothing yet" output
- #16: Outcome record format (counters, steps, evolution log)
- #16: Counter updates (update_success, update_fail)
- #16: Tripwire marker on failing steps
- #16: Tripwatch alerts on tripped records and steps
- #16: Tripwatch silent on healthy records
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))

from store import DuckDBMemoryStore
from handoff import compile_handoff
from outcome_records import (
    OutcomeRecord, StepCounter, TripwatchAlert,
    tripwatch_check, tripwatch_check_store,
    TRIPWIRE_MARKER,
)


# ---------------------------------------------------------------------------
# #48: Compile-to-handoff consumer
# ---------------------------------------------------------------------------

class TestCompileHandoff:
    """compile_handoff should render a valid handoff block."""

    def test_renders_all_sections(self, tmp_path):
        """The handoff block should have STATE, TODO, and GOTCHAS sections."""
        store = DuckDBMemoryStore(tmp_path / "test_handoff.duckdb")
        try:
            store.remember(category="personal_fact", content="User likes Python")
            store.save_candidate(
                category="personal_fact",
                content="User prefers Python 3.12",
                project_id="proj-alpha",
            )
            block = compile_handoff(store, project_id="proj-alpha")
            assert "## STATE" in block
            assert "## TODO" in block
            assert "## GOTCHAS" in block
        finally:
            store.close()

    def test_issue_url_included(self, tmp_path):
        """The ISSUE section should include the tracking issue URL."""
        store = DuckDBMemoryStore(tmp_path / "test_handoff_issue.duckdb")
        try:
            block = compile_handoff(store, issue_url="https://github.com/repo/issues/42")
            assert "## ISSUE" in block
            assert "https://github.com/repo/issues/42" in block
        finally:
            store.close()

    def test_empty_store_graceful(self, tmp_path):
        """An empty store should produce graceful 'nothing yet' output."""
        store = DuckDBMemoryStore(tmp_path / "test_handoff_empty.duckdb")
        try:
            block = compile_handoff(store)
            assert "## STATE" in block
            assert "no active facts" in block
            assert "## TODO" in block
            assert "no pending proposals" in block
            assert "## GOTCHAS" in block
            assert "no known gotchas" in block
        finally:
            store.close()

    def test_todo_includes_pending_proposals(self, tmp_path):
        """The TODO section should include pending proposals."""
        store = DuckDBMemoryStore(tmp_path / "test_handoff_todo.duckdb")
        try:
            store.save_candidate(
                category="personal_fact",
                content="User prefers Python 3.12",
                project_id="proj-alpha",
            )
            block = compile_handoff(store, project_id="proj-alpha")
            assert "Python 3.12" in block
        finally:
            store.close()


# ---------------------------------------------------------------------------
# #16: Outcome record format
# ---------------------------------------------------------------------------

class TestOutcomeRecord:
    """OutcomeRecord should maintain counters and tripwire markers."""

    def test_create_record(self):
        rec = OutcomeRecord(procedure_name="deploy", version="1.0")
        assert rec.procedure_name == "deploy"
        assert rec.success_count == 0
        assert rec.fail_count == 0
        assert rec.total() == 0

    def test_update_success(self):
        rec = OutcomeRecord(procedure_name="deploy")
        rec.update_success()
        assert rec.success_count == 1
        assert rec.fail_count == 0

    def test_update_fail(self):
        rec = OutcomeRecord(procedure_name="deploy")
        rec.update_fail(reason="connection timeout")
        assert rec.fail_count == 1
        assert rec.success_count == 0
        assert "connection timeout" in rec.evolution_log[0]

    def test_step_counter(self):
        rec = OutcomeRecord(procedure_name="deploy")
        rec.update_success(step_name="build")
        rec.update_success(step_name="build")
        rec.update_fail(step_name="build")
        step = rec.steps[0]
        assert step.name == "build"
        assert step.success_count == 2
        assert step.fail_count == 1

    def test_step_tripwire(self):
        """A step with >= 2 fails and fails >= successes should trip."""
        rec = OutcomeRecord(procedure_name="deploy")
        rec.update_fail(step_name="deploy")
        rec.update_fail(step_name="deploy")
        step = rec.steps[0]
        assert step.is_tripped()

    def test_step_not_tripped_with_more_successes(self):
        """A step with more successes than fails should not trip."""
        rec = OutcomeRecord(procedure_name="deploy")
        rec.update_success(step_name="build")
        rec.update_success(step_name="build")
        rec.update_success(step_name="build")
        rec.update_fail(step_name="build")
        step = rec.steps[0]
        assert not step.is_tripped()

    def test_record_tripwire(self):
        """A record with >= 2 fails and fails >= successes should trip."""
        rec = OutcomeRecord(procedure_name="deploy")
        rec.update_fail()
        rec.update_fail()
        assert rec.is_tripped()

    def test_record_not_tripped_with_more_successes(self):
        """A record with more successes than fails should not trip."""
        rec = OutcomeRecord(procedure_name="deploy")
        rec.update_success()
        rec.update_success()
        rec.update_success()
        rec.update_fail()
        assert not rec.is_tripped()

    def test_content_includes_tripwire_marker(self):
        """The rendered content should include the tripwire marker for tripped steps."""
        rec = OutcomeRecord(procedure_name="deploy")
        rec.update_fail(step_name="deploy")
        rec.update_fail(step_name="deploy")
        content = rec.to_content()
        assert TRIPWIRE_MARKER in content

    def test_content_includes_evolution_log(self):
        """The rendered content should include the evolution log."""
        rec = OutcomeRecord(procedure_name="deploy")
        rec.update_fail(reason="connection timeout")
        content = rec.to_content()
        assert "## Evolution" in content
        assert "connection timeout" in content

    def test_payload_round_trip(self):
        """The record should survive a payload round-trip."""
        rec = OutcomeRecord(procedure_name="deploy", version="2.0")
        rec.update_success(step_name="build")
        rec.update_fail(step_name="deploy", reason="timeout")
        payload = rec.to_payload()
        restored = OutcomeRecord.from_payload(payload)
        assert restored.procedure_name == "deploy"
        assert restored.version == "2.0"
        assert restored.success_count == 1
        assert restored.fail_count == 1
        assert len(restored.steps) == 2
        assert len(restored.evolution_log) == 1


# ---------------------------------------------------------------------------
# #16: Tripwatch
# ---------------------------------------------------------------------------

class TestTripwatch:
    """The tripwatch should alert on tripped records and be silent on healthy ones."""

    def test_silent_on_healthy_record(self):
        """A healthy record should produce no alerts."""
        records = [
            ("mem-1", {
                "kind": "outcome",
                "procedure_name": "deploy",
                "version": "1.0",
                "success_count": 5,
                "fail_count": 0,
                "steps": [],
                "evolution_log": [],
            }),
        ]
        alerts = tripwatch_check(records)
        assert len(alerts) == 0

    def test_alert_on_tripped_record(self):
        """A tripped record should produce a record-level alert."""
        records = [
            ("mem-1", {
                "kind": "outcome",
                "procedure_name": "deploy",
                "version": "1.0",
                "success_count": 0,
                "fail_count": 3,
                "steps": [],
                "evolution_log": [],
            }),
        ]
        alerts = tripwatch_check(records)
        assert len(alerts) >= 1
        assert any(a.severity == "record" for a in alerts)

    def test_alert_on_tripped_step(self):
        """A tripped step should produce a step-level alert."""
        records = [
            ("mem-1", {
                "kind": "outcome",
                "procedure_name": "deploy",
                "version": "1.0",
                "success_count": 5,
                "fail_count": 0,
                "steps": [
                    {"name": "deploy", "success_count": 0, "fail_count": 3},
                ],
                "evolution_log": [],
            }),
        ]
        alerts = tripwatch_check(records)
        assert any(a.severity == "step" and a.step_name == "deploy" for a in alerts)

    def test_non_outcome_records_ignored(self):
        """Non-outcome records should be ignored by the tripwatch."""
        records = [
            ("mem-1", {"kind": "insight", "success_count": 0, "fail_count": 100}),
        ]
        alerts = tripwatch_check(records)
        assert len(alerts) == 0

    def test_empty_records_no_alerts(self):
        """An empty record list should produce no alerts."""
        alerts = tripwatch_check([])
        assert len(alerts) == 0

    def test_tripwatch_check_store(self, tmp_path):
        """tripwatch_check_store should read outcome records from the store."""
        store = DuckDBMemoryStore(tmp_path / "test_tripwatch.duckdb")
        try:
            # Store a healthy outcome record.
            rec = OutcomeRecord(procedure_name="deploy")
            rec.update_success()
            rec.update_success()
            store.remember(
                category="context_note",
                content=rec.to_content(),
                payload=rec.to_payload(),
            )
            alerts = tripwatch_check_store(store)
            assert len(alerts) == 0  # healthy

            # Store a tripped outcome record.
            rec2 = OutcomeRecord(procedure_name="broken-proc")
            rec2.update_fail()
            rec2.update_fail()
            store.remember(
                category="context_note",
                content=rec2.to_content(),
                payload=rec2.to_payload(),
            )
            alerts = tripwatch_check_store(store)
            assert any(a.procedure_name == "broken-proc" for a in alerts)
        finally:
            store.close()
