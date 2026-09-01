"""Tests for the stale-pending review sweep (#10).

The sweep re-reviews proposals stranded in 'pending' (e.g. after a
failed/rate-limited reviewer call). It consumes four config keys:
- stale_review_sweep_enabled (gate)
- stale_review_interval_min (cadence — tested via the loop, not here)
- stale_review_min_age_min (min age filter)
- stale_review_max_batch (batch cap)

These tests call ``_run_stale_review_sweep`` directly (not the periodic
loop) to avoid waiting for the interval. The LLM reviewer is mocked.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Keep this test independently importable when pytest collects it first.
_plugin_dir = Path(__file__).resolve().parent.parent
for _path in (_plugin_dir.parent, _plugin_dir):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import pytest


def _make_store(tmp_path):
    from argos.store import DuckDBMemoryStore

    store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user")
    return store


def _add_pending(store, content="Alex prefers flat white coffee"):
    candidate = store.save_candidate(
        category="preference",
        content=content,
        source="llm_extraction",
        confidence=0.9,
        scope="profile",
        evidence_text="My name is Alex and I prefer flat white coffee.",
    )
    assert candidate is not None
    return candidate["candidate_id"]


def _backdate_candidate(store, candidate_id, minutes_ago):
    """Set created_at on a candidate to N minutes in the past."""
    old_ts = (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat()
    with store._lock:
        assert store.connection is not None
        store.connection.execute(
            "UPDATE memory_candidates SET created_at = ? WHERE candidate_id = ?",
            [old_ts, candidate_id],
        )


class _StubProvider:
    """Minimal provider with just enough state for the sweep."""

    def __init__(self, store, **sweep_config):
        self._store = store
        self._llm_model = None
        self._llm_provider = None
        self._evidence_retention = "full"
        self._expiry_auto_suggest = False
        self._stale_review_sweep_enabled = sweep_config.get(
            "stale_review_sweep_enabled", True
        )
        self._stale_review_interval_min = sweep_config.get(
            "stale_review_interval_min", 15
        )
        self._stale_review_min_age_min = sweep_config.get(
            "stale_review_min_age_min", 30
        )
        self._stale_review_max_batch = sweep_config.get(
            "stale_review_max_batch", 25
        )
        import threading
        self._stale_sweep_stop = threading.Event()
        # Import the real _review_candidate from ProviderSessionMixin.
        from argos_plugin.provider_session import ProviderSessionMixin
        self._review_candidate = (
            lambda cand: ProviderSessionMixin._review_candidate(self, cand)
        )

    def _run_stale_review_sweep(self):
        from argos_plugin.provider_session import ProviderSessionMixin
        return ProviderSessionMixin._run_stale_review_sweep(self)


def _patch_reviewer(monkeypatch, decision="approve"):
    """Patch review_candidate_with_llm in the provider_session module."""
    fake_review = {
        "decision": decision,
        "reason": "mocked review",
        "confidence": 0.9,
        "review_model": "test",
        "durability": "durable",
        "scope": "profile",
    }
    import argos_plugin.provider_session as _ps
    monkeypatch.setattr(
        _ps, "review_candidate_with_llm",
        lambda *a, **k: fake_review,
    )
    return fake_review


# -- min-age filter -------------------------------------------------------

def test_sweep_skips_fresh_pending(tmp_path, monkeypatch):
    """Fresh pending proposals (under min_age) are NOT re-reviewed."""
    _patch_reviewer(monkeypatch)
    store = _make_store(tmp_path)
    try:
        cid = _add_pending(store, "fresh fact")
        # Leave created_at as 'now' — under the 30min min_age threshold.
        provider = _StubProvider(store, stale_review_min_age_min=30)
        reviewed = provider._run_stale_review_sweep()
        assert reviewed == 0
        # Still pending — untouched.
        cands = store.list_candidates(candidate_id=cid, limit=1)
        assert cands[0]["status"] == "pending"
    finally:
        store.close()


def test_sweep_reviews_stale_pending(tmp_path, monkeypatch):
    """Pending proposals older than min_age ARE re-reviewed."""
    _patch_reviewer(monkeypatch, decision="approve")
    store = _make_store(tmp_path)
    try:
        cid = _add_pending(store, "old fact")
        _backdate_candidate(store, cid, minutes_ago=60)  # 60 > 30 min threshold
        provider = _StubProvider(store, stale_review_min_age_min=30)
        reviewed = provider._run_stale_review_sweep()
        assert reviewed == 1
        # Approve → reviewed_approved (no-auto-promotion invariant).
        cands = store.list_candidates(candidate_id=cid, limit=1)
        assert cands[0]["status"] == "reviewed_approved"
    finally:
        store.close()


# -- batch cap ------------------------------------------------------------

def test_sweep_respects_batch_cap(tmp_path, monkeypatch):
    """At most stale_review_max_batch proposals are re-reviewed per tick."""
    _patch_reviewer(monkeypatch, decision="approve")
    store = _make_store(tmp_path)
    try:
        cids = []
        for i in range(10):
            cid = _add_pending(store, f"old fact number {i}")
            _backdate_candidate(store, cid, minutes_ago=60)
            cids.append(cid)
        provider = _StubProvider(
            store, stale_review_min_age_min=30, stale_review_max_batch=3
        )
        reviewed = provider._run_stale_review_sweep()
        assert reviewed == 3
        # Exactly 3 should have moved out of pending.
        pending = store.list_candidates(status="pending", limit=100)
        assert len(pending) == 7
    finally:
        store.close()


# -- no-auto-promotion invariant ------------------------------------------

def test_sweep_never_auto_promotes(tmp_path, monkeypatch):
    """The sweep's decision map is identical to _review_candidate:
    approve → reviewed_approved, never 'approved'."""
    _patch_reviewer(monkeypatch, decision="approve")
    store = _make_store(tmp_path)
    try:
        cid = _add_pending(store, "sensitive old fact")
        _backdate_candidate(store, cid, minutes_ago=60)
        provider = _StubProvider(store, stale_review_min_age_min=30)
        provider._run_stale_review_sweep()
        cands = store.list_candidates(candidate_id=cid, limit=1)
        assert cands[0]["status"] == "reviewed_approved"
        assert cands[0]["status"] != "approved"
    finally:
        store.close()


def test_sweep_quarantine_decision(tmp_path, monkeypatch):
    """Quarantine decisions flow through the sweep correctly."""
    _patch_reviewer(monkeypatch, decision="quarantine")
    store = _make_store(tmp_path)
    try:
        cid = _add_pending(store, "junk old fact")
        _backdate_candidate(store, cid, minutes_ago=60)
        provider = _StubProvider(store, stale_review_min_age_min=30)
        provider._run_stale_review_sweep()
        cands = store.list_candidates(candidate_id=cid, limit=1)
        assert cands[0]["status"] == "quarantined"
    finally:
        store.close()


# -- config consumption ---------------------------------------------------

def test_sweep_disabled_returns_zero(tmp_path, monkeypatch):
    """When stale_review_sweep_enabled=False, the sweep does nothing."""
    _patch_reviewer(monkeypatch)
    store = _make_store(tmp_path)
    try:
        cid = _add_pending(store, "old fact")
        _backdate_candidate(store, cid, minutes_ago=60)
        provider = _StubProvider(
            store, stale_review_sweep_enabled=False, stale_review_min_age_min=30
        )
        reviewed = provider._run_stale_review_sweep()
        assert reviewed == 0
        cands = store.list_candidates(candidate_id=cid, limit=1)
        assert cands[0]["status"] == "pending"
    finally:
        store.close()


def test_sweep_no_store_returns_zero(tmp_path):
    """If the store is None, the sweep returns 0 without error."""
    provider = _StubProvider(None, stale_review_sweep_enabled=True)
    assert provider._run_stale_review_sweep() == 0


# -- oldest-first ordering ------------------------------------------------

def test_sweep_processes_oldest_first(tmp_path, monkeypatch):
    """When the batch cap bites, the oldest proposals are prioritized."""
    call_order = []

    def _track_review(candidate, **kw):
        call_order.append(candidate["candidate_id"])
        return {
            "decision": "approve",
            "reason": "mocked",
            "confidence": 0.9,
            "review_model": "test",
            "durability": "durable",
            "scope": "profile",
        }

    import argos_plugin.provider_session as _ps
    monkeypatch.setattr(_ps, "review_candidate_with_llm", _track_review)

    store = _make_store(tmp_path)
    try:
        # Create 5 stale candidates with increasing age.
        cids = []
        for i in range(5):
            cid = _add_pending(store, f"fact {i}")
            _backdate_candidate(store, cid, minutes_ago=60 + i * 10)
            cids.append(cid)
        # Cap at 2 — the two oldest (90 and 100 min) should be picked.
        provider = _StubProvider(
            store, stale_review_min_age_min=30, stale_review_max_batch=2
        )
        reviewed = provider._run_stale_review_sweep()
        assert reviewed == 2
        # cids[4] is 100 min old, cids[3] is 90 min old — oldest first.
        assert call_order == [cids[4], cids[3]]
    finally:
        store.close()
