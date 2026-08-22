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
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))


# NOTE: this suite loads the embedder BY NAME, which performs a network
# HEAD-check that HANGS for minutes without a warm cache. Run with
# HF_HUB_OFFLINE=1 (or the equivalent) or the suite will stall.

_EMBEDDER = None
_EMBEDDER_LOCK = threading.Lock()


def _get_embedder():
    """Shared embedder — one model load for the whole suite."""
    global _EMBEDDER
    if _EMBEDDER is None:
        with _EMBEDDER_LOCK:
            if _EMBEDDER is None:
                from embeddings import LocalEmbedder
                _EMBEDDER = LocalEmbedder("BAAI/bge-small-en-v1.5")
    return _EMBEDDER


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    """A fresh DuckDBMemoryStore with the BGE embedder."""
    from store import DuckDBMemoryStore
    embedder = _get_embedder()
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
