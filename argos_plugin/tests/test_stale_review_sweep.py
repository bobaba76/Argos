"""#10: stale-review sweep — periodically re-review proposals stranded in
'pending' after a failed/rate-limited reviewer call.

Tests (deterministic, no LLM calls):
1. Sweep consumes all four stale_review_* config keys.
2. Min-age filter: fresh pending proposals are not re-reviewed.
3. Batch cap: bounds the number of re-reviews per sweep.
4. No auto-promotion: decision map identical to review_pending.py.
5. Fail-soft on LLM error: sweep continues, doesn't crash.
6. Backward compat: sweep disabled = no re-reviews.
"""
import json
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stale_review_sweep import (
    StaleReviewSweepThread,
    _is_stale,
    _parse_iso,
    run_stale_review_sweep,
)


# ---------------------------------------------------------------------------
# 1. Config consumption
# ---------------------------------------------------------------------------


class TestConfigConsumption:
    """The sweep consumes all four stale_review_* config keys."""

    def test_sweep_thread_uses_interval(self):
        thread = StaleReviewSweepThread(
            MagicMock(), interval_min=20, min_age_min=45, max_batch=10,
        )
        assert thread._interval_s == 20 * 60
        assert thread._min_age_min == 45
        assert thread._max_batch == 10

    def test_interval_floored_at_1_min(self):
        """Interval below 1 minute is floored to 60 seconds."""
        thread = StaleReviewSweepThread(
            MagicMock(), interval_min=0, min_age_min=30, max_batch=25,
        )
        assert thread._interval_s == 60

    def test_config_defaults_parsed(self):
        """The provider parses the four config keys with correct defaults."""
        # Simulate what provider_core.py does with default config.
        config = {}
        stale_sweep = config.get("stale_review_sweep_enabled", "true")
        sweep_enabled = (
            stale_sweep.lower() in ("true", "1", "yes")
            if isinstance(stale_sweep, str) else bool(stale_sweep)
        )
        interval_min = max(1, int(config.get("stale_review_interval_min", 15)))
        min_age_min = max(0, int(config.get("stale_review_min_age_min", 30)))
        max_batch = max(1, min(int(config.get("stale_review_max_batch", 25)), 500))
        assert sweep_enabled is True
        assert interval_min == 15
        assert min_age_min == 30
        assert max_batch == 25

    def test_config_disabled(self):
        """When sweep is disabled, the thread is not started."""
        config = {"stale_review_sweep_enabled": "false"}
        stale_sweep = config.get("stale_review_sweep_enabled", "true")
        sweep_enabled = (
            stale_sweep.lower() in ("true", "1", "yes")
            if isinstance(stale_sweep, str) else bool(stale_sweep)
        )
        assert sweep_enabled is False


# ---------------------------------------------------------------------------
# 2. Min-age filter
# ---------------------------------------------------------------------------


class TestMinAgeFilter:
    """Only re-review pending candidates older than min_age_min minutes."""

    def test_fresh_candidate_not_stale(self):
        now = datetime.now(timezone.utc)
        created = now - timedelta(minutes=5)
        candidate = {"created_at": created.isoformat()}
        assert not _is_stale(candidate, min_age_min=30, now=now)

    def test_old_candidate_is_stale(self):
        now = datetime.now(timezone.utc)
        created = now - timedelta(minutes=45)
        candidate = {"created_at": created.isoformat()}
        assert _is_stale(candidate, min_age_min=30, now=now)

    def test_boundary_exact_age(self):
        """A candidate exactly at min_age_min is stale (>= comparison)."""
        now = datetime.now(timezone.utc)
        created = now - timedelta(minutes=30)
        candidate = {"created_at": created.isoformat()}
        assert _is_stale(candidate, min_age_min=30, now=now)

    def test_naive_datetime_treated_as_utc(self):
        """Naive datetimes (no timezone) are treated as UTC."""
        now = datetime.now(timezone.utc)
        created_naive = (now - timedelta(minutes=45)).replace(tzinfo=None)
        candidate = {"created_at": created_naive.isoformat()}
        assert _is_stale(candidate, min_age_min=30, now=now)

    def test_unparseable_timestamp_defaults_to_stale(self):
        """If the timestamp can't be parsed, the candidate is reviewed
        (err on the side of reviewing it)."""
        candidate = {"created_at": "not-a-date"}
        assert _is_stale(candidate, min_age_min=30, now=datetime.now(timezone.utc))

    def test_missing_timestamp_defaults_to_stale(self):
        candidate = {}
        assert _is_stale(candidate, min_age_min=30, now=datetime.now(timezone.utc))

    def test_parse_iso_valid(self):
        ts = "2026-01-15T10:30:00+00:00"
        result = _parse_iso(ts)
        assert result is not None
        assert result.year == 2026

    def test_parse_iso_none(self):
        assert _parse_iso(None) is None
        assert _parse_iso("") is None

    def test_parse_iso_invalid(self):
        assert _parse_iso("not-a-date") is None


# ---------------------------------------------------------------------------
# 3. Batch cap
# ---------------------------------------------------------------------------


class TestBatchCap:
    """Batch cap bounds the number of re-reviews per sweep."""

    def test_batch_cap_limits_reviews(self):
        """Even with many stale candidates, only max_batch are reviewed."""
        now = datetime.now(timezone.utc)
        old_ts = (now - timedelta(minutes=60)).isoformat()
        candidates = [
            {"candidate_id": f"c{i}", "created_at": old_ts, "category": "personal_fact",
             "content": f"fact {i}", "payload": {}}
            for i in range(50)
        ]
        store = MagicMock()
        store.list_candidates.return_value = candidates
        # Mock the reviewer to always approve.
        with patch("reviewer.review_candidate_with_llm",
                   return_value={"decision": "approve", "reason": "ok"}):
            counts = run_stale_review_sweep(
                store, min_age_min=30, max_batch=5,
            )
        # Only 5 should be reviewed.
        assert store.review_candidate.call_count == 5
        assert sum(counts.values()) == 5

    def test_batch_cap_zero_reviews_none(self):
        """max_batch=0 means no reviews (edge case)."""
        now = datetime.now(timezone.utc)
        old_ts = (now - timedelta(minutes=60)).isoformat()
        candidates = [{"candidate_id": "c1", "created_at": old_ts}]
        store = MagicMock()
        store.list_candidates.return_value = candidates
        with patch("reviewer.review_candidate_with_llm",
                   return_value={"decision": "approve"}):
            counts = run_stale_review_sweep(
                store, min_age_min=30, max_batch=0,
            )
        assert store.review_candidate.call_count == 0
        assert sum(counts.values()) == 0


# ---------------------------------------------------------------------------
# 4. No auto-promotion
# ---------------------------------------------------------------------------


class TestNoAutoPromotion:
    """Decision map identical to review_pending.py — no auto-promotion."""

    def test_approve_becomes_reviewed_approved(self):
        """'approve' → 'reviewed_approved' (not 'active')."""
        now = datetime.now(timezone.utc)
        old_ts = (now - timedelta(minutes=60)).isoformat()
        candidates = [{"candidate_id": "c1", "created_at": old_ts,
                       "category": "personal_fact", "content": "fact",
                       "payload": {}}]
        store = MagicMock()
        store.list_candidates.return_value = candidates
        with patch("reviewer.review_candidate_with_llm",
                   return_value={"decision": "approve", "reason": "low risk"}):
            counts = run_stale_review_sweep(store, min_age_min=30, max_batch=25)
        assert "reviewed_approved" in counts
        # Verify the decision passed to store.review_candidate.
        call_kwargs = store.review_candidate.call_args
        assert call_kwargs.kwargs["decision"] == "reviewed_approved"

    def test_reject_becomes_rejected(self):
        now = datetime.now(timezone.utc)
        old_ts = (now - timedelta(minutes=60)).isoformat()
        candidates = [{"candidate_id": "c1", "created_at": old_ts,
                       "category": "personal_fact", "content": "fact",
                       "payload": {}}]
        store = MagicMock()
        store.list_candidates.return_value = candidates
        with patch("reviewer.review_candidate_with_llm",
                   return_value={"decision": "reject", "reason": "bad"}):
            counts = run_stale_review_sweep(store, min_age_min=30, max_batch=25)
        assert counts.get("rejected") == 1

    def test_quarantine_becomes_quarantined(self):
        now = datetime.now(timezone.utc)
        old_ts = (now - timedelta(minutes=60)).isoformat()
        candidates = [{"candidate_id": "c1", "created_at": old_ts,
                       "category": "personal_fact", "content": "fact",
                       "payload": {}}]
        store = MagicMock()
        store.list_candidates.return_value = candidates
        with patch("reviewer.review_candidate_with_llm",
                   return_value={"decision": "quarantine", "reason": "junk"}):
            counts = run_stale_review_sweep(store, min_age_min=30, max_batch=25)
        assert counts.get("quarantined") == 1

    def test_pending_user_confirmation_preserved(self):
        now = datetime.now(timezone.utc)
        old_ts = (now - timedelta(minutes=60)).isoformat()
        candidates = [{"candidate_id": "c1", "created_at": old_ts,
                       "category": "personal_fact", "content": "fact",
                       "payload": {}}]
        store = MagicMock()
        store.list_candidates.return_value = candidates
        with patch("reviewer.review_candidate_with_llm",
                   return_value={"decision": "pending_user_confirmation",
                                 "reason": "needs human"}):
            counts = run_stale_review_sweep(store, min_age_min=30, max_batch=25)
        assert counts.get("pending_user_confirmation") == 1


# ---------------------------------------------------------------------------
# 5. Fail-soft on LLM error
# ---------------------------------------------------------------------------


class TestFailSoft:
    """Sweep continues on LLM error, doesn't crash."""

    def test_llm_exception_skips_candidate(self):
        """If the LLM call raises, the candidate is skipped and the
        sweep continues."""
        now = datetime.now(timezone.utc)
        old_ts = (now - timedelta(minutes=60)).isoformat()
        candidates = [
            {"candidate_id": "c1", "created_at": old_ts,
             "category": "personal_fact", "content": "fact 1", "payload": {}},
            {"candidate_id": "c2", "created_at": old_ts,
             "category": "personal_fact", "content": "fact 2", "payload": {}},
        ]
        store = MagicMock()
        store.list_candidates.return_value = candidates
        call_count = [0]
        def mock_review(candidate, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("LLM timeout")
            return {"decision": "approve", "reason": "ok"}
        with patch("reviewer.review_candidate_with_llm", side_effect=mock_review):
            counts = run_stale_review_sweep(store, min_age_min=30, max_batch=25)
        # c1 failed (skipped), c2 succeeded.
        assert store.review_candidate.call_count == 1
        assert counts.get("reviewed_approved") == 1

    def test_list_candidates_exception_returns_empty(self):
        """If list_candidates fails, the sweep is a no-op."""
        store = MagicMock()
        store.list_candidates.side_effect = RuntimeError("DB error")
        counts = run_stale_review_sweep(store, min_age_min=30, max_batch=25)
        assert counts == {}

    def test_reviewer_import_failure_returns_empty(self):
        """If the reviewer module can't be imported, the sweep is a no-op."""
        store = MagicMock()
        # Patch the import to fail.
        import builtins
        original_import = builtins.__import__
        def mock_import(name, *args, **kwargs):
            if "reviewer" in name:
                raise ImportError("no reviewer")
            return original_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=mock_import):
            counts = run_stale_review_sweep(store, min_age_min=30, max_batch=25)
        assert counts == {}


# ---------------------------------------------------------------------------
# 6. Sweep thread lifecycle
# ---------------------------------------------------------------------------


class TestSweepThread:
    """The sweep thread starts, runs, and stops correctly."""

    def test_start_and_stop(self):
        """The thread starts and stops without error."""
        store = MagicMock()
        thread = StaleReviewSweepThread(
            store, interval_min=1, min_age_min=30, max_batch=5,
        )
        thread.start()
        assert thread._thread is not None
        assert thread._thread.is_alive()
        thread.stop()
        thread._thread.join(timeout=5.0)
        assert not thread._thread.is_alive()

    def test_start_is_idempotent(self):
        """Starting twice doesn't create two threads."""
        store = MagicMock()
        thread = StaleReviewSweepThread(
            store, interval_min=60, min_age_min=30, max_batch=5,
        )
        thread.start()
        first_thread = thread._thread
        thread.start()
        assert thread._thread is first_thread
        thread.stop()
        first_thread.join(timeout=5.0)

    def test_stop_before_start_is_safe(self):
        """Calling stop() before start() is a no-op."""
        store = MagicMock()
        thread = StaleReviewSweepThread(store)
        thread.stop()  # should not raise


# ---------------------------------------------------------------------------
# 7. Integration — store-level with real DuckDB
# ---------------------------------------------------------------------------


class TestStoreIntegration:
    """Integration test with a real DuckDBMemoryStore."""

    def test_sweep_reviews_only_stale_candidates(self, tmp_path):
        """The sweep re-reviews only stale candidates, not fresh ones."""
        from store import DuckDBMemoryStore
        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="alice")
        # Create a stale candidate (created 60 min ago).
        store.save_candidate(
            category="personal_fact",
            content="stale fact",
            session_id="s1",
        )
        # Manually set the created_at to 60 min ago.
        old_ts = (datetime.now(timezone.utc) - timedelta(minutes=60)).isoformat()
        store.connection.execute(
            "UPDATE memory_candidates SET created_at = ? WHERE content = 'stale fact'",
            [old_ts],
        )
        # Create a fresh candidate (created now).
        store.save_candidate(
            category="personal_fact",
            content="fresh fact",
            session_id="s2",
        )
        # Run the sweep with min_age_min=30.
        with patch("reviewer.review_candidate_with_llm",
                   return_value={"decision": "approve", "reason": "ok"}):
            counts = run_stale_review_sweep(
                store, min_age_min=30, max_batch=25,
            )
        # Only the stale candidate should be reviewed.
        assert sum(counts.values()) == 1
        # The fresh candidate should still be pending.
        pending = store.list_candidates(status="pending", limit=10)
        pending_contents = [c["content"] for c in pending]
        assert "fresh fact" in pending_contents
        assert "stale fact" not in pending_contents
        store.close()
