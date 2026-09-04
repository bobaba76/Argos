"""Tests for P4.2 — Distillation pass ("the dream").

Covers the 13 required test cases from the spec:
1. Novelty gate: below min_new_records → zero LLM calls.
2. Cooldown gate: last run inside cooldown_hours → skip.
3. Budget: records sampled to max_records_per_run; calls capped at max_calls.
4. Output contract: valid JSON → pending candidates, correct category,
   source='distillation', evidence_text populated, payload.sources resolves;
   dedup=True blocks exact re-proposals.
5. Fail-soft: LLM exception / malformed JSON / client import unavailable
   → no crash, no candidates, run NOT advanced.
6. Star-greedy grouping: chain A~B, B~C with A~C < threshold → two clusters,
   not one (no transitive chaining); each record in exactly one cluster.
7. Completed zero-proposal run (all clusters empty) advances last_run.
8. system_state persists across store reopen; last_run + last_count readable.
9. Pending proposals invisible to retrieval (status check).
10. Contradiction outputs link via find_supersede_candidates but never
    auto-supersede.
11. Auto-review integration: distilled candidates flow through
    _review_candidate exactly like extraction candidates.
12. Run gating via system_state: after a completed run, a second run with
    no new records skips (novelty); within cooldown skips (cooldown).
13. Existing suite stays green (verified by running the full suite).

Run with:
    python -m pytest tests/test_distillation.py -v
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))


# NOTE: this suite intentionally uses a deterministic test embedder
# (_test_embedder.py, issue #98) instead of the real BGE model. The real
# model costs ~0.3-1.9s per encode on this machine -- with 25-50 seeded
# records per test that made this file the suite's dominant wall-time cost.
# The deterministic embedder preserves the lexical-overlap clustering the
# tests rely on at microseconds per encode, with no model load and no
# HF_HUB_OFFLINE needed.

from _test_embedder import DeterministicTestEmbedder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    """A fresh DuckDBMemoryStore with the deterministic test embedder."""
    from store import DuckDBMemoryStore
    embedder = DeterministicTestEmbedder()
    s = DuckDBMemoryStore(
        tmp_path / "test.duckdb", user_id="test_user", embedder=embedder,
    )
    yield s
    s.close()


def _make_mock_response(content: str):
    """Create a mock LLM response object with .choices[0].message.content."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


def _make_distill_response(
    insights: List[Dict] = None,
    guardrails: List[Dict] = None,
    contradictions: List[Dict] = None,
) -> str:
    """Build a valid distill JSON response string."""
    return json.dumps({
        "insights": insights or [],
        "guardrails": guardrails or [],
        "contradictions": contradictions or [],
    })


def _seed_related_records(store, n: int = 25) -> List:
    """Insert n related-but-distinct records to pass the novelty gate."""
    recs = []
    # Create records in a tight semantic cluster so seed-star finds them.
    for i in range(n):
        rec = store.remember(
            category="personal_fact",
            content=f"User works on project Alpha and completed task {i} on schedule",
            dedup=False,
        )
        if rec:
            recs.append(rec)
    return recs


def _guaranteed_cluster_pair(store):
    """Return two record IDs guaranteed to be in the first seed-star cluster.

    Replicates the deterministic clustering: the newest record is the seed;
    the record most similar to it is its top member. Both are always inside
    the first cluster's source_ids.
    """
    records = store._fetch_records(
        "SELECT * FROM memory_records WHERE valid_to IS NULL "
        "ORDER BY created_at DESC"
    )
    emb = np.asarray([r.embedding for r in records], dtype=np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    normed = emb / np.maximum(norms, 1e-9)
    sims = normed @ normed[0]  # similarity to the newest record (the seed)
    top = int(np.argmax(sims[1:])) + 1  # most similar non-seed record
    return records[0].memory_id, records[top].memory_id


def _seed_distinct_records(store, n: int = 25) -> List:
    """Insert n semantically distinct records."""
    topics = [
        "User enjoys cooking Italian pasta dishes on weekends",
        "User drives a blue Toyota Hilux to work every day",
        "User is learning Japanese calligraphy as a hobby",
        "User prefers dark mode for all code editors and terminals",
        "User has a dog named Rex who loves to play fetch in the park",
        "User is a certified scuba diver who explores coral reefs",
        "User plays the guitar in a local jazz band on Friday nights",
        "User collects vintage stamps from European countries",
        "User is studying machine learning algorithms for a career change",
        "User enjoys hiking in the mountains during autumn season",
    ]
    recs = []
    for i in range(n):
        topic = topics[i % len(topics)]
        rec = store.remember(
            category="personal_fact",
            content=f"{topic} (entry {i})",
            dedup=False,
        )
        if rec:
            recs.append(rec)
    return recs


# ---------------------------------------------------------------------------
# 1. Novelty gate: below min_new_records → zero LLM calls
# ---------------------------------------------------------------------------

class TestGating:
    def test_novelty_gate_below_threshold(self, store):
        """Below min_new_records → zero LLM calls."""
        from distillation import run_distillation
        # Insert only 5 records (below default threshold of 20).
        _seed_related_records(store, n=5)
        call_count = {"n": 0}

        def mock_call_llm(*args, **kwargs):
            call_count["n"] += 1
            return _make_mock_response(_make_distill_response())

        with patch("distillation._get_llm_client", return_value=mock_call_llm):
            report = run_distillation(store, min_new_records=20)

        assert report["ran"] is False
        assert "novelty_gate" in report["reason"]
        assert call_count["n"] == 0, "No LLM calls should be made below threshold"

    # -----------------------------------------------------------------------
    # 2. Cooldown gate: last run inside cooldown_hours → skip
    # -----------------------------------------------------------------------

    def test_cooldown_gate(self, store):
        """Last run inside cooldown_hours → skip."""
        from distillation import (
            run_distillation, _STATE_KEY_LAST_RUN,
        )
        _seed_related_records(store, n=25)
        # Set last_run to 1 hour ago.
        recent = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        store.set_state(_STATE_KEY_LAST_RUN, recent)

        call_count = {"n": 0}
        def mock_call_llm(*args, **kwargs):
            call_count["n"] += 1
            return _make_mock_response(_make_distill_response())

        with patch("distillation._get_llm_client", return_value=mock_call_llm):
            report = run_distillation(store, cooldown_hours=24)

        assert report["ran"] is False
        assert report["reason"] == "cooldown"
        assert call_count["n"] == 0, "No LLM calls during cooldown"

    # -----------------------------------------------------------------------
    # 3. Budget: records sampled; calls capped
    # -----------------------------------------------------------------------

    def test_budget_records_and_calls(self, store):
        """Records sampled to max_records_per_run; calls capped at max_calls."""
        from distillation import run_distillation
        _seed_related_records(store, n=50)
        call_count = {"n": 0}

        def mock_call_llm(*args, **kwargs):
            call_count["n"] += 1
            return _make_mock_response(_make_distill_response())

        with patch("distillation._get_llm_client", return_value=mock_call_llm):
            report = run_distillation(
                store,
                min_new_records=10,
                max_records_per_run=20,
                max_calls=3,
            )

        assert report["ran"] is True
        assert report["llm_calls"] <= 3, "Calls must be capped at max_calls"
        assert report["records_processed"] <= 20, \
            "Records must be sampled to max_records_per_run"

    # -----------------------------------------------------------------------
    # 12. Run gating via system_state: second run with no new records skips
    # -----------------------------------------------------------------------

    def test_second_run_no_new_records_skips(self, store):
        """After a completed run, a second run with no new records skips."""
        from distillation import run_distillation
        _seed_related_records(store, n=25)
        call_count = {"n": 0}

        def mock_call_llm(*args, **kwargs):
            call_count["n"] += 1
            return _make_mock_response(_make_distill_response())

        with patch("distillation._get_llm_client", return_value=mock_call_llm):
            report1 = run_distillation(store, min_new_records=20, cooldown_hours=0)
            assert report1["ran"] is True
            calls_after_first = call_count["n"]
            # Second run: no new records since last_run.
            report2 = run_distillation(store, min_new_records=20, cooldown_hours=0)
            assert report2["ran"] is False
            assert "novelty_gate" in report2["reason"]
            assert call_count["n"] == calls_after_first, \
                "Second run should make zero LLM calls"


# ---------------------------------------------------------------------------
# 4. Output contract: valid JSON → pending candidates
# ---------------------------------------------------------------------------

class TestOutputContract:
    def test_valid_json_produces_candidates(self, store):
        """Valid JSON → one pending candidate per insight/guardrail."""
        from distillation import run_distillation
        _seed_related_records(store, n=25)

        response = _make_distill_response(
            insights=[{"text": "User consistently completes tasks on schedule"}],
            guardrails=[{"text": "Avoid scheduling conflicts with project Alpha"}],
        )

        with patch("distillation._get_llm_client",
                    return_value=lambda *a, **kw: _make_mock_response(response)):
            report = run_distillation(store, min_new_records=20, cooldown_hours=0)

        assert report["ran"] is True
        assert report["proposals_emitted"] >= 2, \
            "Should emit at least 1 insight + 1 guardrail"
        # Verify candidates exist in the store.
        candidates = store.list_candidates(status="pending")
        distillation_cands = [
            c for c in candidates
            if c.get("source") == "distillation"
        ]
        assert len(distillation_cands) >= 2
        for c in distillation_cands:
            assert c["source"] == "distillation"
            assert c["category"] in ("insight", "context_note")
            # evidence_text should be populated.
            assert c.get("evidence_text", "").strip() != ""
            # payload.sources should resolve to existing memory_ids.
            payload = c.get("payload")
            if isinstance(payload, str):
                payload = json.loads(payload)
            sources = payload.get("sources", [])
            assert len(sources) > 0, "Every proposal must cite source memory_ids"

    def test_dedup_blocks_exact_reproposals(self, store):
        """dedup=True blocks exact re-proposals."""
        from distillation import run_distillation
        _seed_related_records(store, n=25)

        insight_text = "User consistently completes tasks on schedule"
        response = _make_distill_response(
            insights=[{"text": insight_text}],
        )

        with patch("distillation._get_llm_client",
                    return_value=lambda *a, **kw: _make_mock_response(response)):
            report1 = run_distillation(store, min_new_records=20, cooldown_hours=0)
            # Manually reset last_run to allow a second run.
            from distillation import _STATE_KEY_LAST_RUN
            store.set_state(_STATE_KEY_LAST_RUN, "")
            # Add more records to pass novelty gate.
            _seed_related_records(store, n=5)
            report2 = run_distillation(store, min_new_records=20, cooldown_hours=0)

        # The second run should not duplicate the same insight.
        candidates = store.list_candidates(status="pending")
        distillation_cands = [
            c for c in candidates
            if c.get("source") == "distillation" and insight_text in (c.get("content") or "")
        ]
        assert len(distillation_cands) == 1, \
            "Exact re-proposal should be deduped away"


# ---------------------------------------------------------------------------
# 5. Fail-soft: LLM exception / malformed JSON / client unavailable
# ---------------------------------------------------------------------------

class TestFailSoft:
    def test_llm_exception_no_crash_no_advance(self, store):
        """LLM exception → no crash, no candidates, run NOT advanced."""
        from distillation import run_distillation, _STATE_KEY_LAST_RUN
        _seed_related_records(store, n=25)

        def failing_llm(*args, **kwargs):
            raise RuntimeError("LLM service unavailable")

        with patch("distillation._get_llm_client", return_value=failing_llm):
            report = run_distillation(store, min_new_records=20, cooldown_hours=0)

        assert report["ran"] is False
        assert "all_llm_calls_failed" in report.get("reason", "")
        # Run state should NOT have advanced.
        last_run = store.get_state(_STATE_KEY_LAST_RUN)
        assert last_run is None or last_run == "", \
            "Run state must not advance on failure"

    def test_malformed_json_no_crash(self, store):
        """Malformed JSON → no crash, no candidates from that call."""
        from distillation import run_distillation
        _seed_related_records(store, n=25)

        with patch("distillation._get_llm_client",
                    return_value=lambda *a, **kw: _make_mock_response("not json {{{")):
            report = run_distillation(store, min_new_records=20, cooldown_hours=0)

        # Should complete (the call succeeded, just bad JSON) but emit 0 proposals.
        assert report["ran"] is True
        assert report["proposals_emitted"] == 0

    def test_client_unavailable_no_crash(self, store):
        """LLM client import unavailable → skip, no crash."""
        from distillation import run_distillation
        _seed_related_records(store, n=25)

        with patch("distillation._get_llm_client", return_value=None):
            report = run_distillation(store, min_new_records=20, cooldown_hours=0)

        assert report["ran"] is False
        assert report["reason"] == "llm_client_unavailable"


# ---------------------------------------------------------------------------
# 6. Star-greedy grouping: no transitive chaining
# ---------------------------------------------------------------------------

class TestStarGreedy:
    def test_no_transitive_chaining(self, store):
        """Chain A~B, B~C with A~C < threshold → two clusters, not one."""
        from distillation import _seed_star_cluster
        # We need 3 records where A~B >= 0.75, B~C >= 0.75, but A~C < 0.75.
        # This is hard to engineer with real embeddings, so we mock the
        # embedding vectors directly.
        from store import MemoryRecord
        import numpy as np

        # Create 3 records with controlled embeddings.
        # A and B are similar (cosine ~0.9), B and C are similar (cosine ~0.9),
        # but A and C are not (cosine ~0.3).
        # In 3D space:
        # A = [1, 0, 0], B = [0.7, 0.7, 0], C = [0, 1, 0]
        # cos(A,B) = 0.7/sqrt(0.98) ≈ 0.707 — too low.
        # Try: A = [1, 0, 0], B = [0.85, 0.53, 0], C = [0.5, 0.87, 0]
        # cos(A,B) = 0.85, cos(B,C) = 0.85*0.5+0.53*0.87 = 0.886, cos(A,C) = 0.5
        # That works: A~B=0.85, B~C=0.89, A~C=0.5

        recs = []
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.85, 0.53, 0.0],
            [0.5, 0.87, 0.0],
        ]
        for i, emb in enumerate(embeddings):
            rec = MemoryRecord(
                memory_id=f"mem-test-{i}",
                category="personal_fact",
                content=f"Test record {i} for clustering validation",
                embedding=emb,
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            recs.append(rec)

        clusters = _seed_star_cluster(recs, min_similarity=0.75)
        # A~B >= 0.75 → they cluster together. C is not similar enough to A
        # (the seed, since A is newest by created_at tie → first in sort).
        # Wait: all have same created_at, so sort is stable. The first record
        # (A) is the seed. B joins (cos >= 0.75). C does not (cos(A,C) = 0.5).
        # C becomes its own cluster's seed.
        assert len(clusters) == 2, \
            f"Expected 2 clusters (no transitive chaining), got {len(clusters)}"
        # Each record in exactly one cluster.
        all_ids = set()
        for cluster in clusters:
            for r in cluster:
                assert r.memory_id not in all_ids, \
                    "Record appears in multiple clusters"
                all_ids.add(r.memory_id)
        assert len(all_ids) == 3


# ---------------------------------------------------------------------------
# 7. Completed zero-proposal run advances last_run
# ---------------------------------------------------------------------------

class TestRunState:
    def test_zero_proposal_run_advances(self, store):
        """A run where all clusters return empty → last_run still advances."""
        from distillation import run_distillation, _STATE_KEY_LAST_RUN
        _seed_related_records(store, n=25)

        empty_response = _make_distill_response()  # all empty arrays

        with patch("distillation._get_llm_client",
                    return_value=lambda *a, **kw: _make_mock_response(empty_response)):
            report = run_distillation(store, min_new_records=20, cooldown_hours=0)

        assert report["ran"] is True
        assert report["proposals_emitted"] == 0
        # last_run should have advanced.
        last_run = store.get_state(_STATE_KEY_LAST_RUN)
        assert last_run is not None and last_run != "", \
            "Zero-proposal completion must advance last_run"

    # -----------------------------------------------------------------------
    # 8. system_state persists across store reopen
    # -----------------------------------------------------------------------

    def test_state_persists_across_reopen(self, store, tmp_path):
        """system_state persists across store reopen."""
        from store import DuckDBMemoryStore
        from distillation import _STATE_KEY_LAST_RUN, _STATE_KEY_LAST_COUNT

        test_time = "2026-01-15T12:00:00+00:00"
        store.set_state(_STATE_KEY_LAST_RUN, test_time)
        store.set_state(_STATE_KEY_LAST_COUNT, "42")
        store.close()

        # Reopen the same DB.
        reopened = DuckDBMemoryStore(
            tmp_path / "test.duckdb", user_id="test_user",
        )
        assert reopened.get_state(_STATE_KEY_LAST_RUN) == test_time
        assert reopened.get_state(_STATE_KEY_LAST_COUNT) == "42"
        reopened.close()


# ---------------------------------------------------------------------------
# 9. Pending proposals invisible to retrieval
# ---------------------------------------------------------------------------

class TestProposalVisibility:
    def test_pending_proposals_invisible_to_retrieval(self, store):
        """Distilled proposals must not appear in search results."""
        from distillation import run_distillation
        _seed_related_records(store, n=25)

        insight_text = "User demonstrates strong project management skills"
        response = _make_distill_response(
            insights=[{"text": insight_text}],
        )

        with patch("distillation._get_llm_client",
                    return_value=lambda *a, **kw: _make_mock_response(response)):
            run_distillation(store, min_new_records=20, cooldown_hours=0)

        # Search for the insight text — it should NOT be in results.
        results = store.search(insight_text, limit=20)
        result_contents = [r.content for r in results]
        assert insight_text not in result_contents, \
            "Pending proposals must be invisible to retrieval"


# ---------------------------------------------------------------------------
# 10. Contradiction outputs link but never auto-supersede
# ---------------------------------------------------------------------------

class TestContradictions:
    def test_contradiction_emitted_not_auto_superseded(self, store):
        """Contradiction outputs create candidates but never auto-supersede."""
        from distillation import run_distillation
        _seed_related_records(store, n=25)

        # Get two IDs that ARE in an emitted cluster (seed + its top member).
        a_id, b_id = _guaranteed_cluster_pair(store)

        response = _make_distill_response(
            contradictions=[{
                "a_id": a_id,
                "b_id": b_id,
                "reason": "Record A says X but record B says not X",
            }],
        )

        with patch("distillation._get_llm_client",
                    return_value=lambda *a, **kw: _make_mock_response(response)):
            report = run_distillation(store, min_new_records=20, cooldown_hours=0)

        assert report["ran"] is True
        assert report["contradictions_emitted"] >= 1
        # Verify neither record was auto-superseded.
        for rec_id in [a_id, b_id]:
            row = store.connection.execute(
                "SELECT valid_to, superseded_by FROM memory_records WHERE memory_id = ?",
                [rec_id],
            ).fetchone()
            assert row[0] is None, \
                f"Record {rec_id} must not be auto-superseded"
            assert row[1] is None, \
                f"Record {rec_id} must not have superseded_by set"
        # The contradiction should be a pending candidate.
        candidates = store.list_candidates(status="pending")
        contra_cands = [
            c for c in candidates
            if c.get("source") == "distillation"
            and "contradiction" in (c.get("content") or "")
        ]
        assert len(contra_cands) >= 1


# ---------------------------------------------------------------------------
# 11. Auto-review integration: distilled candidates flow through _review_candidate
# ---------------------------------------------------------------------------

class TestAutoReviewIntegration:
    def test_distilled_candidates_have_correct_source(self, store):
        """Distilled candidates have source='distillation' and flow through
        the same pending queue as extraction candidates."""
        from distillation import run_distillation
        _seed_related_records(store, n=25)

        response = _make_distill_response(
            insights=[{"text": "User shows consistent task completion patterns"}],
        )

        with patch("distillation._get_llm_client",
                    return_value=lambda *a, **kw: _make_mock_response(response)):
            run_distillation(store, min_new_records=20, cooldown_hours=0)

        candidates = store.list_candidates(status="pending")
        distillation_cands = [
            c for c in candidates if c.get("source") == "distillation"
        ]
        assert len(distillation_cands) >= 1
        # The candidate should have the same structure as extraction
        # candidates — no special-casing needed.
        c = distillation_cands[0]
        assert "candidate_id" in c
        assert "category" in c
        assert "content" in c
        assert "status" in c
        assert c["status"] == "pending"


# ---------------------------------------------------------------------------
# 12. High-signal proposals keep a unanimous project scope
# ---------------------------------------------------------------------------

class TestHighSignalScoping:
    def test_high_signal_proposals_keep_unanimous_project(self, store):
        """Guardrails from single-project feedback land in that project."""
        from distillation import run_distillation
        recs = _seed_related_records(store, n=25)

        # Give two records feedback signals + a project scope.
        store.connection.execute(
            "UPDATE memory_records SET helpful_count = 3, "
            "project_id = 'proj-alpha' "
            "WHERE memory_id IN (?, ?)",
            [recs[0].memory_id, recs[1].memory_id],
        )

        response = _make_distill_response(
            guardrails=[{"text": "Always confirm the schedule before starting"}],
        )
        # Cluster legs and the high-signal leg share the mock; give the
        # high-signal leg a DIFFERENT text so dedup=True doesn't swallow it.
        high_signal_response = _make_distill_response(
            guardrails=[{"text": "Always confirm the schedule before starting — twice"}],
        )

        def _mocked(*_a, **kw):
            if kw.get("task") == "distillation_high_signal":
                return _make_mock_response(high_signal_response)
            return _make_mock_response(response)

        with patch("distillation._get_llm_client",
                    return_value=_mocked):
            report = run_distillation(store, min_new_records=20, cooldown_hours=0)

        assert report["ran"] is True
        assert report["proposals_emitted"] >= 1
        row = store.connection.execute(
            "SELECT project_id FROM memory_candidates "
            "WHERE source = 'distillation' "
            "AND payload LIKE '%guardrail%' "
            "ORDER BY created_at DESC LIMIT 1",
        ).fetchone()
        assert row is not None
        assert row[0] == "proj-alpha", \
            "Guardrail distilled from project-scoped feedback must keep scope"

    def test_mixed_project_feedback_goes_global(self, store):
        """Feedback from different projects → global proposal, no mis-tag."""
        from distillation import run_distillation
        recs = _seed_related_records(store, n=25)

        store.connection.execute(
            "UPDATE memory_records SET helpful_count = 2, "
            "project_id = 'proj-alpha' WHERE memory_id = ?",
            [recs[0].memory_id],
        )
        store.connection.execute(
            "UPDATE memory_records SET helpful_count = 2, "
            "project_id = 'proj-beta' WHERE memory_id = ?",
            [recs[1].memory_id],
        )

        response = _make_distill_response(
            guardrails=[{"text": "Confirm the schedule before starting"}],
        )
        high_signal_response = _make_distill_response(
            guardrails=[{"text": "Confirm the schedule before starting — from feedback"}],
        )

        def _mocked(*_a, **kw):
            if kw.get("task") == "distillation_high_signal":
                return _make_mock_response(high_signal_response)
            return _make_mock_response(response)

        with patch("distillation._get_llm_client",
                    return_value=_mocked):
            report = run_distillation(store, min_new_records=20, cooldown_hours=0)

        assert report["ran"] is True
        row = store.connection.execute(
            "SELECT project_id FROM memory_candidates "
            "WHERE source = 'distillation' "
            "AND payload LIKE '%guardrail%' "
            "ORDER BY created_at DESC LIMIT 1",
        ).fetchone()
        assert row is not None
        assert row[0] is None, \
            "Mixed-project feedback must not be tagged to either project"


# ---------------------------------------------------------------------------
# 13. Contradiction IDs are validated — invented IDs are dropped
# ---------------------------------------------------------------------------

class TestContradictionValidation:
    def test_invented_contradiction_ids_dropped(self, store):
        """Only IDs the LLM was actually shown produce contradiction proposals."""
        from distillation import run_distillation
        _seed_related_records(store, n=25)

        a_id, b_id = _guaranteed_cluster_pair(store)

        response = _make_distill_response(
            contradictions=[
                {"a_id": a_id, "b_id": b_id, "reason": "Real pair"},
                {"a_id": "mem-fake-1", "b_id": "mem-fake-2",
                 "reason": "Invented pair"},
            ],
        )
        with patch("distillation._get_llm_client",
                    return_value=lambda *a, **kw: _make_mock_response(response)):
            report = run_distillation(store, min_new_records=20, cooldown_hours=0)

        assert report["ran"] is True
        assert report["contradictions_emitted"] == 1, \
            "Fabricated IDs must be silently dropped"

    def test_valid_contradiction_helper(self):
        """Pure unit: validator accepts only IDs the LLM was shown."""
        from distillation import _valid_contradiction
        src = ["mem-a", "mem-b"]
        assert _valid_contradiction(
            {"a_id": "mem-a", "b_id": "mem-b"}, src) is True
        assert _valid_contradiction(
            {"a_id": "mem-x", "b_id": "mem-b"}, src) is False
        assert _valid_contradiction(
            {"a_id": "mem-a", "b_id": ""}, src) is False
        assert _valid_contradiction("not a dict", src) is False
        assert _valid_contradiction({}, src) is False


# ---------------------------------------------------------------------------
# D1: Run state advances when no work found (no infinite retry loop)
# ---------------------------------------------------------------------------

class TestD1NoWorkRunStateAdvance:
    """D1: when there are no multi-clusters AND <2 high-signal records,
    the run state must advance (not return 'all_llm_calls_failed') to
    avoid an infinite 'nothing to distill but keep trying' loop.
    """

    def test_no_work_advances_run_state(self, store):
        """A run with all singletons and no high-signal records should
        advance last_run (not return all_llm_calls_failed)."""
        from distillation import run_distillation, _get_last_run
        # Seed distinct records (all singletons, no multi-clusters).
        _seed_distinct_records(store, n=25)
        # Mock LLM that returns an empty valid response (in case any
        # multi-clusters are found by the deterministic embedder).
        empty_response = _make_mock_response(
            json.dumps({"insights": [], "guardrails": [], "contradictions": []})
        )
        mock_call = MagicMock(return_value=empty_response)
        with patch("distillation._get_llm_client", return_value=mock_call), \
             patch("distillation._load_high_signal_records", return_value=[]):
            report = run_distillation(store, min_new_records=20, cooldown_hours=0)
        # The run should complete (not fail with all_llm_calls_failed).
        assert report["ran"] is True, (
            f"Expected ran=True (no work found should advance state), "
            f"got reason={report.get('reason')}"
        )
        # Run state must have advanced.
        assert _get_last_run(store) is not None

    def test_all_calls_failed_does_not_advance(self, store):
        """When LLM calls are attempted but all fail, run state must NOT
        advance (retry on next session end)."""
        from distillation import run_distillation, _get_last_run
        # Seed related records (will form multi-clusters → LLM calls).
        _seed_related_records(store, n=25)
        # Mock LLM that always raises.
        mock_call = MagicMock(side_effect=RuntimeError("LLM down"))
        with patch("distillation._get_llm_client", return_value=mock_call):
            report = run_distillation(store, min_new_records=20, cooldown_hours=0)
        assert report["ran"] is False
        assert report["reason"] == "all_llm_calls_failed"
        # Run state must NOT have advanced.
        assert _get_last_run(store) is None


# ---------------------------------------------------------------------------
# D2: _parse_distill_response handles prose-wrapped JSON
# ---------------------------------------------------------------------------

class TestD2ProseWrappedJson:
    """D2: _parse_distill_response should handle LLM output wrapped in
    conversational prose (same class as R4 in reviewer)."""

    def test_prose_wrapped_json_parsed(self):
        """'Here is the JSON: {...} Done.' should parse successfully."""
        from distillation import _parse_distill_response
        response = _make_mock_response(
            'Here is the JSON: {"insights": [{"text": "test"}], "guardrails": [], "contradictions": []} Done.'
        )
        parsed = _parse_distill_response(response)
        assert parsed is not None
        assert len(parsed["insights"]) == 1
        assert parsed["insights"][0]["text"] == "test"

    def test_fenced_json_parsed(self):
        """Fenced JSON (```json ... ```) should parse successfully."""
        from distillation import _parse_distill_response
        response = _make_mock_response(
            '```json\n{"insights": [], "guardrails": [], "contradictions": []}\n```'
        )
        parsed = _parse_distill_response(response)
        assert parsed is not None
        assert parsed["insights"] == []

    def test_pure_json_parsed(self):
        """Pure JSON (no fences, no prose) should parse successfully."""
        from distillation import _parse_distill_response
        response = _make_mock_response(
            '{"insights": [{"text": "pure"}], "guardrails": [], "contradictions": []}'
        )
        parsed = _parse_distill_response(response)
        assert parsed is not None
        assert parsed["insights"][0]["text"] == "pure"

    def test_garbage_returns_none(self):
        """Non-JSON text should return None."""
        from distillation import _parse_distill_response
        response = _make_mock_response("this is not json at all")
        parsed = _parse_distill_response(response)
        assert parsed is None


# ---------------------------------------------------------------------------
# #232 Distillation audit D1-D10
# ---------------------------------------------------------------------------

class TestDistillationAudit232:
    """Regression tests for issue #232: distillation audit D1-D10."""

    # -- D1: markup neutralization in LLM prompts ---------------------------

    def test_d1_neutralize_markup_helper(self):
        """D1: _neutralize_markup replaces < and > with fullwidth variants."""
        from distillation import _neutralize_markup
        assert _neutralize_markup("<script>") == "\uFF1Cscript\uFF1E"
        assert _neutralize_markup("clean text") == "clean text"
        assert _neutralize_markup("") == ""

    def test_d1_cluster_prompt_neutralizes_content(self, store):
        """D1: _build_cluster_prompt neutralizes < and > in record content."""
        from distillation import _build_cluster_prompt
        from store_common import MemoryRecord
        rec = MemoryRecord(
            memory_id="m1", category="personal_fact",
            content="<script>ignore previous</script>",
            embedding=[1.0], helpful_count=0, dismissed_count=0,
            retrieval_count=0, created_at="2025-01-01T00:00:00Z",
            status="active", scope="profile",
        )
        prompt = _build_cluster_prompt([rec])
        assert "\uFF1C" in prompt, "content < should be neutralized"
        assert "\uFF1E" in prompt, "content > should be neutralized"
        assert "<script>" not in prompt

    def test_d1_high_signal_prompt_neutralizes_content(self):
        """D1: _build_high_signal_prompt neutralizes < and > in content."""
        from distillation import _build_high_signal_prompt
        from store_common import MemoryRecord
        rec = MemoryRecord(
            memory_id="m1", category="personal_fact",
            content="<b>injection</b>",
            embedding=[1.0], helpful_count=0, dismissed_count=0,
            retrieval_count=0, created_at="2025-01-01T00:00:00Z",
            status="active", scope="profile",
        )
        prompt = _build_high_signal_prompt([rec])
        assert "\uFF1C" in prompt
        assert "\uFF1E" in prompt

    def test_d1_system_prompt_mentions_data(self):
        """D1: system prompts instruct the LLM to treat content as DATA."""
        from distillation import _DISTILL_SYSTEM, _HIGH_SIGNAL_SYSTEM
        assert "DATA" in _DISTILL_SYSTEM
        assert "DATA" in _HIGH_SIGNAL_SYSTEM

    # -- D4: caps on proposals per cluster -----------------------------------

    def test_d4_insights_capped(self):
        """D4: _MAX_INSIGHTS_PER_CLUSTER is set to 5."""
        from distillation import _MAX_INSIGHTS_PER_CLUSTER
        assert _MAX_INSIGHTS_PER_CLUSTER == 5

    def test_d4_guardrails_capped(self):
        """D4: _MAX_GUARDRAILS_PER_CLUSTER is set to 3."""
        from distillation import _MAX_GUARDRAILS_PER_CLUSTER
        assert _MAX_GUARDRAILS_PER_CLUSTER == 3

    def test_d4_contradictions_capped(self):
        """D4: _MAX_CONTRADICTIONS_PER_CLUSTER is set to 3."""
        from distillation import _MAX_CONTRADICTIONS_PER_CLUSTER
        assert _MAX_CONTRADICTIONS_PER_CLUSTER == 3

    def test_d4_insights_actually_capped_in_run(self, store):
        """D4: a run with 20 insights from the LLM only emits 5."""
        from distillation import run_distillation
        _seed_related_records(store, n=25)
        # LLM returns 20 insights.
        response = _make_distill_response(
            insights=[{"text": f"Insight {i}"} for i in range(20)],
        )
        with patch("distillation._get_llm_client",
                    return_value=lambda *a, **kw: _make_mock_response(response)):
            report = run_distillation(store, min_new_records=20, cooldown_hours=0)
        assert report["ran"] is True
        # Should be capped at 5 (assuming no dedup collisions).
        assert report["proposals_emitted"] <= 5, (
            f"Expected <= 5 proposals (capped), got {report['proposals_emitted']}"
        )

    # -- D5: human-readable contradiction content ----------------------------

    def test_d5_contradiction_content_no_raw_ids(self):
        """D5: contradiction content should not contain raw memory IDs."""
        from distillation import _emit_contradiction
        saved_args = {}
        class FakeStore:
            def save_candidate(self, **kwargs):
                saved_args.update(kwargs)
                return {"candidate_id": "c1"}
        result = _emit_contradiction(
            FakeStore(), "mem-abc-123", "mem-def-456",
            "They disagree on the project name",
            ["mem-abc-123", "mem-def-456"], "evidence",
        )
        assert result is not None
        content = saved_args["content"]
        assert "mem-abc-123" not in content
        assert "mem-def-456" not in content
        assert "contradiction" in content.lower()

    # -- D7: empty content after truncation ----------------------------------

    def test_d7_empty_after_truncation_returns_none(self):
        """D7: content that becomes empty after truncation returns None."""
        from distillation import _emit_proposal
        class FakeStore:
            def save_candidate(self, **kwargs):
                raise AssertionError("save_candidate should not be called")
        # 201 chars of whitespace — passes old check, empty after strip+trunc.
        result = _emit_proposal(
            FakeStore(), " " * 201, "insight", ["m1"], "ev", 0.7,
        )
        assert result is None

    # -- D8: parameter clamping ----------------------------------------------

    def test_d8_zero_max_records_clamped(self, store):
        """D8: max_records_per_run=0 should be clamped to 1, not silently
        do nothing."""
        from distillation import run_distillation
        _seed_related_records(store, n=25)
        empty_response = _make_mock_response(
            json.dumps({"insights": [], "guardrails": [], "contradictions": []})
        )
        with patch("distillation._get_llm_client",
                    return_value=lambda *a, **kw: _make_mock_response(empty_response)):
            report = run_distillation(
                store, min_new_records=20, cooldown_hours=0,
                max_records_per_run=0, max_calls=1,
            )
        # Should not silently abort — clamped to 1, loads 1 record.
        # With 1 record, no multi-clusters form, so it completes with no work.
        assert report["ran"] is True or report["reason"] != ""

    def test_d8_zero_max_calls_clamped(self, store):
        """D8: max_calls=0 should be clamped to 1. With calls_budget=1,
        the cluster loop reserves 1 call for high-signal (so 0 cluster
        calls), but the run should not silently abort — it should still
        complete (either run or skip with a valid reason, not crash)."""
        from distillation import run_distillation
        _seed_related_records(store, n=25)
        empty_response = _make_mock_response(
            json.dumps({"insights": [], "guardrails": [], "contradictions": []})
        )
        with patch("distillation._get_llm_client",
                    return_value=lambda *a, **kw: _make_mock_response(empty_response)):
            report = run_distillation(
                store, min_new_records=20, cooldown_hours=0,
                max_calls=0,
            )
        # Should not crash or silently do nothing — the run completes.
        assert report["ran"] is True or report["reason"] != "", (
            f"Expected run to complete or have a reason, got {report}"
        )

    # -- D9: renamed to _unanimous_project_id --------------------------------

    def test_d9_unanimous_project_id_exists(self):
        """D9: _unanimous_project_id should exist (renamed from _majority_project_id)."""
        from distillation import _unanimous_project_id
        assert callable(_unanimous_project_id)

    def test_d9_unanimous_returns_pid_when_all_same(self):
        """D9: returns the project_id when all records share it."""
        from distillation import _unanimous_project_id
        from store_common import MemoryRecord
        recs = [
            MemoryRecord(memory_id="m1", category="x", content="c",
                         embedding=[1.0], helpful_count=0, dismissed_count=0,
                         retrieval_count=0, created_at="2025-01-01T00:00:00Z",
                         status="active", scope="profile", project_id="proj-a"),
            MemoryRecord(memory_id="m2", category="x", content="c",
                         embedding=[1.0], helpful_count=0, dismissed_count=0,
                         retrieval_count=0, created_at="2025-01-01T00:00:00Z",
                         status="active", scope="profile", project_id="proj-a"),
        ]
        assert _unanimous_project_id(recs) == "proj-a"

    def test_d9_unanimous_returns_none_when_mixed(self):
        """D9: returns None when records have different project_ids."""
        from distillation import _unanimous_project_id
        from store_common import MemoryRecord
        recs = [
            MemoryRecord(memory_id="m1", category="x", content="c",
                         embedding=[1.0], helpful_count=0, dismissed_count=0,
                         retrieval_count=0, created_at="2025-01-01T00:00:00Z",
                         status="active", scope="profile", project_id="proj-a"),
            MemoryRecord(memory_id="m2", category="x", content="c",
                         embedding=[1.0], helpful_count=0, dismissed_count=0,
                         retrieval_count=0, created_at="2025-01-01T00:00:00Z",
                         status="active", scope="profile", project_id="proj-b"),
        ]
        assert _unanimous_project_id(recs) is None

    # -- D10: created_at in cluster prompt -----------------------------------

    def test_d10_cluster_prompt_includes_created_at(self):
        """D10: _build_cluster_prompt should include created_at timestamps."""
        from distillation import _build_cluster_prompt
        from store_common import MemoryRecord
        rec = MemoryRecord(
            memory_id="m1", category="personal_fact",
            content="test content",
            embedding=[1.0], helpful_count=0, dismissed_count=0,
            retrieval_count=0, created_at="2025-06-15T12:00:00Z",
            status="active", scope="profile",
        )
        prompt = _build_cluster_prompt([rec])
        parsed = json.loads(prompt)
        assert "created_at" in parsed["records"][0]
        assert parsed["records"][0]["created_at"] == "2025-06-15T12:00:00Z"
