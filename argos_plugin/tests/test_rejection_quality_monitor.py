"""Tests for the rejection-ledger quality monitor (#121).

Covers all 4 acceptance criteria from the issue:
1. Read-only tool reads rejection_ledger + candidate statuses and reports
   per-category decision rates over configurable windows (zero LLM, no
   schema change).
2. Drift monitor path flags category-rate changes beyond a configured
   threshold (opt-in).
3. Rejected/quarantined rows exportable as labeled hard-case eval items
   for regression gates.
4. Read-only guaranteed — never writes to or mutates the ledger or records.

All deterministic, zero LLM. Uses the real DuckDBMemoryStore with a temp
DB so the queries hit the actual schema.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))


def _make_store(tmp_path):
    from store import DuckDBMemoryStore

    return DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user")


def _seed_candidates(store, decisions):
    """Seed reviewed candidates directly into memory_candidates.

    *decisions* is a list of (category, status, reviewed_at, review_reason,
    provenance_origin, source) or (category, status, reviewed_at,
    review_reason, provenance_origin, source, quarantine_reason) tuples.
    We insert directly (the monitor is read-only and doesn't care how the
    rows got there).
    """
    import json
    import uuid

    now = datetime.now(timezone.utc).isoformat()
    for d in decisions:
        cat, status, reviewed_at, reason, prov, src = d[:6]
        quar_reason = d[6] if len(d) > 6 else None
        cid = f"cand-{uuid.uuid4().hex}"
        store.connection.execute(
            """INSERT INTO memory_candidates
               (candidate_id, category, content, tags, payload, source,
                confidence, durability, scope, project_id, namespace,
                session_id, user_scope, status, created_at, updated_at,
                reviewed_at, review_reason, quarantine_reason,
                provenance_origin, grounding)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                cid, cat, f"test content {cid[:12]}", [],
                json.dumps({"user_scope": "test_user"}),
                src, 0.5, "durable", "profile", None, "conversation",
                "", "test_user", status, reviewed_at or now, reviewed_at or now,
                reviewed_at, reason, quar_reason, prov, "extracted",
            ],
        )


# ---------------------------------------------------------------------------
# Acceptance criterion 1: per-category decision rates over configurable windows
# ---------------------------------------------------------------------------

class TestDecisionRateReport:
    """Read-only report of per-category decision counts and rates."""

    def test_per_category_counts_and_rates(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            _seed_candidates(store, [
                ("personal_fact", "approved", None, "ok", "internal", "llm_extraction"),
                ("personal_fact", "approved", None, "ok", "internal", "llm_extraction"),
                ("personal_fact", "rejected", None, "bad", "internal", "llm_extraction"),
                ("personal_fact", "quarantined", None, "injection", "external", "llm_extraction"),
                ("preference", "approved", None, "ok", "internal", "explicit"),
                ("preference", "rejected", None, "no", "internal", "explicit"),
            ])
            from rejection_quality_monitor import decision_rate_report

            report = decision_rate_report(store, window="all")
            assert report["total_decisions"] == 6
            pf = report["buckets"]["personal_fact"]
            assert pf["total"] == 4
            assert pf["approved"] == 2
            assert pf["rejected"] == 1
            assert pf["quarantined"] == 1
            assert pf["rejection_rate"] == 0.5  # (1+1)/4
            assert pf["approval_rate"] == 0.5   # 2/4
            pref = report["buckets"]["preference"]
            assert pref["total"] == 2
            assert pref["rejection_rate"] == 0.5
        finally:
            store.close()

    def test_window_filters_by_time(self, tmp_path):
        """The 'daily' window only includes rows reviewed in the last day."""
        store = _make_store(tmp_path)
        try:
            now = datetime.now(timezone.utc)
            recent = now.isoformat()
            old = (now - timedelta(days=10)).isoformat()
            _seed_candidates(store, [
                ("personal_fact", "approved", recent, "ok", "internal", "x"),
                ("personal_fact", "rejected", old, "bad", "internal", "x"),
            ])
            from rejection_quality_monitor import decision_rate_report

            report = decision_rate_report(store, window="daily")
            # Only the recent row should be counted.
            assert report["total_decisions"] == 1
            assert "personal_fact" in report["buckets"]
            assert report["buckets"]["personal_fact"]["approved"] == 1
        finally:
            store.close()

    def test_bucket_by_provenance(self, tmp_path):
        """Bucketing by provenance_origin separates internal vs external."""
        store = _make_store(tmp_path)
        try:
            _seed_candidates(store, [
                ("personal_fact", "approved", None, "ok", "internal", "x"),
                ("personal_fact", "rejected", None, "bad", "external", "x"),
            ])
            from rejection_quality_monitor import decision_rate_report

            report = decision_rate_report(store, window="all", by="provenance_origin")
            assert "internal" in report["buckets"]
            assert "external" in report["buckets"]
            assert report["buckets"]["internal"]["approved"] == 1
            assert report["buckets"]["external"]["rejected"] == 1
        finally:
            store.close()

    def test_empty_store_returns_zero(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            from rejection_quality_monitor import decision_rate_report

            report = decision_rate_report(store, window="all")
            assert report["total_decisions"] == 0
            assert report["buckets"] == {}
        finally:
            store.close()

    def test_deduplicated_excluded_from_rejection_rate(self, tmp_path):
        """deduplicated is not a rejection — it's a dedup drop."""
        store = _make_store(tmp_path)
        try:
            _seed_candidates(store, [
                ("personal_fact", "approved", None, "ok", "internal", "x"),
                ("personal_fact", "deduplicated", None, "dup", "internal", "x"),
                ("personal_fact", "rejected", None, "bad", "internal", "x"),
            ])
            from rejection_quality_monitor import decision_rate_report

            report = decision_rate_report(store, window="all")
            pf = report["buckets"]["personal_fact"]
            assert pf["total"] == 3
            assert pf["deduplicated"] == 1
            assert pf["rejection_rate"] == round(1 / 3, 4)  # only rejected/total
        finally:
            store.close()


# ---------------------------------------------------------------------------
# Acceptance criterion 2: drift monitor flags rate changes beyond threshold
# ---------------------------------------------------------------------------

class TestDriftCheck:
    """Opt-in drift detection between adjacent windows."""

    def test_flags_category_with_large_rate_increase(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            now = datetime.now(timezone.utc)
            # Use timestamps safely inside each window.
            prev_start = (now - timedelta(weeks=1, days=3)).isoformat()
            cur_start = (now - timedelta(days=3)).isoformat()
            # Previous week: 1/10 rejected = 10%
            for i in range(9):
                _seed_candidates(store, [
                    ("personal_fact", "approved", prev_start, "ok", "internal", "x"),
                ])
            _seed_candidates(store, [
                ("personal_fact", "rejected", prev_start, "bad", "internal", "x"),
            ])
            # Current week: 8/10 rejected = 80%
            for i in range(2):
                _seed_candidates(store, [
                    ("personal_fact", "approved", cur_start, "ok", "internal", "x"),
                ])
            for i in range(8):
                _seed_candidates(store, [
                    ("personal_fact", "rejected", cur_start, "bad", "internal", "x"),
                ])
            from rejection_quality_monitor import drift_check

            result = drift_check(store, window="weekly", threshold=0.5)
            flags = result["flags"]
            pf_flags = [f for f in flags if f["bucket"] == "personal_fact"]
            assert len(pf_flags) == 1
            assert pf_flags[0]["direction"] == "up"
            assert pf_flags[0]["delta"] >= 0.5
        finally:
            store.close()

    def test_no_flag_when_rate_stable(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            now = datetime.now(timezone.utc)
            # Use timestamps safely inside each window (middle of the
            # previous and current weekly windows, not at the boundary).
            prev = (now - timedelta(weeks=1, days=3)).isoformat()
            cur = (now - timedelta(days=3)).isoformat()
            # Both windows: 1/2 rejected = 50%
            _seed_candidates(store, [
                ("preference", "approved", prev, "ok", "internal", "x"),
                ("preference", "rejected", prev, "bad", "internal", "x"),
                ("preference", "approved", cur, "ok", "internal", "x"),
                ("preference", "rejected", cur, "bad", "internal", "x"),
            ])
            from rejection_quality_monitor import drift_check

            result = drift_check(store, window="weekly", threshold=0.15)
            pref_flags = [f for f in result["flags"] if f["bucket"] == "preference"]
            assert len(pref_flags) == 0
        finally:
            store.close()

    def test_all_window_returns_error(self, tmp_path):
        """drift_check with 'all' window is meaningless — returns error."""
        store = _make_store(tmp_path)
        try:
            from rejection_quality_monitor import drift_check

            result = drift_check(store, window="all")
            assert "error" in result
            assert result["flags"] == []
        finally:
            store.close()

    def test_threshold_is_configurable(self, tmp_path):
        """A low threshold flags small changes; a high one doesn't."""
        store = _make_store(tmp_path)
        try:
            now = datetime.now(timezone.utc)
            prev = (now - timedelta(weeks=1, days=3)).isoformat()
            cur = (now - timedelta(days=3)).isoformat()
            # prev: 0/4 rejected = 0%, cur: 1/4 rejected = 25%, delta = 0.25
            for i in range(4):
                _seed_candidates(store, [
                    ("insight", "approved", prev, "ok", "internal", "x"),
                ])
            _seed_candidates(store, [
                ("insight", "approved", cur, "ok", "internal", "x"),
                ("insight", "approved", cur, "ok", "internal", "x"),
                ("insight", "approved", cur, "ok", "internal", "x"),
                ("insight", "rejected", cur, "bad", "internal", "x"),
            ])
            from rejection_quality_monitor import drift_check

            # threshold=0.2 → should flag (delta 0.25 >= 0.2)
            r1 = drift_check(store, window="weekly", threshold=0.2)
            assert any(f["bucket"] == "insight" for f in r1["flags"])
            # threshold=0.3 → should NOT flag (delta 0.25 < 0.3)
            r2 = drift_check(store, window="weekly", threshold=0.3)
            assert not any(f["bucket"] == "insight" for f in r2["flags"])
        finally:
            store.close()


# ---------------------------------------------------------------------------
# Acceptance criterion 3: hard-case eval export
# ---------------------------------------------------------------------------

class TestExportHardCases:
    """Rejected/quarantined rows exportable as labeled eval items."""

    def test_export_rejected_and_quarantined(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            _seed_candidates(store, [
                ("personal_fact", "rejected", None, "review_rejected", "internal", "x"),
                ("preference", "quarantined", None, "quarantined", "external", "x",
                 "injection_pattern: sql_drop"),
                ("insight", "approved", None, "ok", "internal", "x"),
            ])
            from rejection_quality_monitor import export_hard_cases

            result = export_hard_cases(store)
            assert result["total_exported"] == 2
            statuses = {item["status"] for item in result["items"]}
            assert statuses == {"rejected", "quarantined"}
            # Labels come from the right field.
            rej = [i for i in result["items"] if i["status"] == "rejected"][0]
            assert rej["label"] == "review_rejected"
            assert rej["label_field"] == "review_reason"
            quar = [i for i in result["items"] if i["status"] == "quarantined"][0]
            assert "injection" in quar["label"]
            assert quar["label_field"] == "quarantine_reason"
        finally:
            store.close()

    def test_export_respects_limit(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            for i in range(10):
                _seed_candidates(store, [
                    ("personal_fact", "rejected", None, f"reason_{i}", "internal", "x"),
                ])
            from rejection_quality_monitor import export_hard_cases

            result = export_hard_cases(store, limit=3)
            assert result["total_exported"] == 3
        finally:
            store.close()

    def test_export_excludes_approved(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            _seed_candidates(store, [
                ("personal_fact", "approved", None, "ok", "internal", "x"),
                ("personal_fact", "reviewed_approved", None, "ok", "internal", "x"),
            ])
            from rejection_quality_monitor import export_hard_cases

            result = export_hard_cases(store)
            assert result["total_exported"] == 0
        finally:
            store.close()

    def test_export_items_are_json_serializable(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            _seed_candidates(store, [
                ("personal_fact", "rejected", None, "test reason", "internal", "x"),
            ])
            from rejection_quality_monitor import export_hard_cases
            import json

            result = export_hard_cases(store)
            # Must not raise.
            serialized = json.dumps(result, default=str)
            assert "test reason" in serialized
        finally:
            store.close()


# ---------------------------------------------------------------------------
# Acceptance criterion 4: read-only guaranteed
# ---------------------------------------------------------------------------

class TestReadOnlyGuaranteed:
    """The monitor never writes to or mutates the ledger or records."""

    def test_monitor_does_not_create_records(self, tmp_path):
        """Running all three monitor functions must not add any
        memory_records rows."""
        store = _make_store(tmp_path)
        try:
            _seed_candidates(store, [
                ("personal_fact", "rejected", None, "bad", "internal", "x"),
            ])
            from rejection_quality_monitor import (
                decision_rate_report, drift_check, export_hard_cases,
            )

            before = store.connection.execute(
                "SELECT COUNT(*) FROM memory_records"
            ).fetchone()[0]
            decision_rate_report(store, window="all")
            drift_check(store, window="weekly", threshold=0.1)
            export_hard_cases(store)
            after = store.connection.execute(
                "SELECT COUNT(*) FROM memory_records"
            ).fetchone()[0]
            assert before == after == 0, "monitor must not create memory_records"
        finally:
            store.close()

    def test_monitor_does_not_mutate_candidates(self, tmp_path):
        """Running the monitor must not change any candidate row."""
        store = _make_store(tmp_path)
        try:
            _seed_candidates(store, [
                ("personal_fact", "rejected", None, "bad", "internal", "x"),
                ("preference", "approved", None, "ok", "internal", "x"),
            ])
            from rejection_quality_monitor import (
                decision_rate_report, drift_check, export_hard_cases,
            )

            before = store.connection.execute(
                "SELECT candidate_id, status, review_reason FROM memory_candidates ORDER BY candidate_id"
            ).fetchall()
            decision_rate_report(store, window="all")
            drift_check(store, window="weekly", threshold=0.1)
            export_hard_cases(store)
            after = store.connection.execute(
                "SELECT candidate_id, status, review_reason FROM memory_candidates ORDER BY candidate_id"
            ).fetchall()
            assert before == after, "monitor must not mutate candidates"
        finally:
            store.close()

    def test_monitor_does_not_mutate_rejection_ledger(self, tmp_path):
        """Running the monitor must not change the rejection_ledger."""
        store = _make_store(tmp_path)
        try:
            # Seed a ledger entry via the real path.
            store.record_rejection(
                "personal_fact",
                {"name": "user", "attribute": "age"},
                reason="test_rejection",
            )
            from rejection_quality_monitor import (
                decision_rate_report, drift_check, export_hard_cases,
            )

            before = store.connection.execute(
                "SELECT subject, predicate, user_scope, reason FROM rejection_ledger"
            ).fetchall()
            decision_rate_report(store, window="all")
            drift_check(store, window="weekly", threshold=0.1)
            export_hard_cases(store)
            after = store.connection.execute(
                "SELECT subject, predicate, user_scope, reason FROM rejection_ledger"
            ).fetchall()
            assert before == after, "monitor must not mutate rejection_ledger"
        finally:
            store.close()

    def test_query_methods_are_read_only(self, tmp_path):
        """The store query methods themselves must not write."""
        store = _make_store(tmp_path)
        try:
            _seed_candidates(store, [
                ("personal_fact", "rejected", None, "bad", "internal", "x"),
            ])
            store.record_rejection(
                "personal_fact",
                {"name": "user", "attribute": "age"},
            )
            # Snapshot all tables.
            tables = ["memory_records", "memory_candidates", "rejection_ledger",
                      "deletion_tombstones"]
            before = {
                t: store.connection.execute(
                    f"SELECT * FROM {t}"
                ).fetchall()
                for t in tables
            }
            store.query_candidate_decisions()
            store.query_rejection_ledger()
            store.query_hard_cases()
            after = {
                t: store.connection.execute(
                    f"SELECT * FROM {t}"
                ).fetchall()
                for t in tables
            }
            assert before == after, "query methods must not mutate any table"
        finally:
            store.close()


# ---------------------------------------------------------------------------
# Rejection ledger query (store-level)
# ---------------------------------------------------------------------------

class TestRejectionLedgerQuery:
    """The store-level rejection_ledger query returns rows correctly."""

    def test_query_returns_ledger_rows(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            store.record_rejection(
                "personal_fact",
                {"name": "user", "attribute": "age"},
                reason="too_old",
            )
            rows = store.query_rejection_ledger()
            assert len(rows) == 1
            assert rows[0]["reason"] == "too_old"
            assert rows[0]["subject"] == "user"
        finally:
            store.close()

    def test_query_time_filter(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            store.record_rejection(
                "personal_fact",
                {"subject": "user", "predicate": "personal_fact:age"},
            )
            # Future cutoff → should return nothing.
            future = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
            rows = store.query_rejection_ledger(since=future)
            assert len(rows) == 0
        finally:
            store.close()


# ---------------------------------------------------------------------------
# Render text output (cron-able)
# ---------------------------------------------------------------------------

class TestRenderReportText:
    def test_render_empty_report(self):
        from rejection_quality_monitor import render_report_text

        report = {
            "window": "weekly",
            "since": None,
            "until": None,
            "bucketed_by": "category",
            "total_decisions": 0,
            "buckets": {},
        }
        text = render_report_text(report)
        assert "no reviewed candidates" in text

    def test_render_with_buckets(self):
        from rejection_quality_monitor import render_report_text

        report = {
            "window": "weekly",
            "since": "2026-01-01",
            "until": "2026-01-08",
            "bucketed_by": "category",
            "total_decisions": 2,
            "buckets": {
                "personal_fact": {
                    "total": 2, "approved": 1, "rejected": 1,
                    "quarantined": 0, "deduplicated": 0,
                    "rejection_rate": 0.5, "approval_rate": 0.5,
                },
            },
        }
        text = render_report_text(report)
        assert "personal_fact" in text
        assert "Total decisions: 2" in text
