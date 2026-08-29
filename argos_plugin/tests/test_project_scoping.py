"""Tests for #47: project-scoped proposals + per-project digests.

Covers:
- Candidate records created during a project-scoped session carry project_id
- Global/unsessioned sessions stay None
- Per-project pending-proposal digest works (count + list)
- Project id round-trips through save → list → review
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))

from store import DuckDBMemoryStore


class TestProjectScopedProposals:
    """save_candidate should carry project_id through to stored candidates."""

    def test_candidate_carries_project_id(self, tmp_path):
        """A candidate saved with a project_id should retain it."""
        store = DuckDBMemoryStore(tmp_path / "test_project.duckdb")
        try:
            cand = store.save_candidate(
                category="personal_fact",
                content="User prefers Python 3.12",
                project_id="proj-alpha",
            )
            assert cand is not None
            assert cand["project_id"] == "proj-alpha"
        finally:
            store.close()

    def test_candidate_without_project_id_is_none(self, tmp_path):
        """A candidate saved without a project_id should have None."""
        store = DuckDBMemoryStore(tmp_path / "test_no_project.duckdb")
        try:
            cand = store.save_candidate(
                category="personal_fact",
                content="User likes pizza",
            )
            assert cand is not None
            assert cand["project_id"] is None
        finally:
            store.close()

    def test_project_id_round_trips_through_list(self, tmp_path):
        """Project id should survive save → list_candidates."""
        store = DuckDBMemoryStore(tmp_path / "test_roundtrip.duckdb")
        try:
            store.save_candidate(
                category="personal_fact",
                content="User prefers Python 3.12",
                project_id="proj-alpha",
            )
            store.save_candidate(
                category="personal_fact",
                content="User likes pizza",
                project_id=None,
            )
            candidates = store.list_candidates(status="pending")
            assert len(candidates) == 2
            proj_cands = [c for c in candidates if c["project_id"] == "proj-alpha"]
            assert len(proj_cands) == 1
            assert "Python 3.12" in proj_cands[0]["content"]
            global_cands = [c for c in candidates if c["project_id"] is None]
            assert len(global_cands) == 1
            assert "pizza" in global_cands[0]["content"]
        finally:
            store.close()

    def test_project_id_survives_review(self, tmp_path):
        """Project id should survive through the review process."""
        store = DuckDBMemoryStore(tmp_path / "test_review_project.duckdb")
        try:
            cand = store.save_candidate(
                category="personal_fact",
                content="User prefers Python 3.12",
                project_id="proj-alpha",
            )
            result = store.review_candidate(
                cand["candidate_id"],
                "approved",
                review_source="manual",
            )
            assert result is not None
            assert result["memory"] is not None
            assert result["memory"]["project_id"] == "proj-alpha"
        finally:
            store.close()


class TestProjectDigest:
    """project_digest should group candidates by project."""

    def test_digest_groups_by_project(self, tmp_path):
        """The digest should group candidates by project_id."""
        store = DuckDBMemoryStore(tmp_path / "test_digest.duckdb")
        try:
            store.save_candidate(
                category="personal_fact",
                content="Alpha fact 1",
                project_id="proj-alpha",
            )
            store.save_candidate(
                category="personal_fact",
                content="Alpha fact 2",
                project_id="proj-alpha",
            )
            store.save_candidate(
                category="personal_fact",
                content="Beta fact 1",
                project_id="proj-beta",
            )
            store.save_candidate(
                category="personal_fact",
                content="Global fact 1",
                project_id=None,
            )
            digest = store.project_digest()
            assert len(digest["projects"]) == 2
            # Projects should be sorted by project_id.
            assert digest["projects"][0]["project_id"] == "proj-alpha"
            assert digest["projects"][0]["count"] == 2
            assert digest["projects"][1]["project_id"] == "proj-beta"
            assert digest["projects"][1]["count"] == 1
            # Global candidates (no project_id).
            assert digest["global_count"] == 1
            assert len(digest["global_candidates"]) == 1
        finally:
            store.close()

    def test_digest_filters_by_project(self, tmp_path):
        """The digest should filter to a specific project when asked."""
        store = DuckDBMemoryStore(tmp_path / "test_digest_filter.duckdb")
        try:
            store.save_candidate(
                category="personal_fact",
                content="Alpha fact 1",
                project_id="proj-alpha",
            )
            store.save_candidate(
                category="personal_fact",
                content="Beta fact 1",
                project_id="proj-beta",
            )
            digest = store.project_digest(project_id="proj-alpha")
            assert len(digest["projects"]) == 1
            assert digest["projects"][0]["project_id"] == "proj-alpha"
            assert digest["projects"][0]["count"] == 1
            assert digest["global_count"] == 0
        finally:
            store.close()

    def test_digest_empty_store(self, tmp_path):
        """An empty store should return an empty digest."""
        store = DuckDBMemoryStore(tmp_path / "test_empty_digest.duckdb")
        try:
            digest = store.project_digest()
            assert len(digest["projects"]) == 0
            assert digest["global_count"] == 0
        finally:
            store.close()

    def test_digest_only_pending(self, tmp_path):
        """The digest should only include pending candidates by default."""
        store = DuckDBMemoryStore(tmp_path / "test_digest_status.duckdb")
        try:
            cand1 = store.save_candidate(
                category="personal_fact",
                content="Alpha fact 1",
                project_id="proj-alpha",
            )
            store.save_candidate(
                category="personal_fact",
                content="Alpha fact 2",
                project_id="proj-alpha",
            )
            # Approve one candidate.
            store.review_candidate(cand1["candidate_id"], "approved", review_source="manual")
            digest = store.project_digest(project_id="proj-alpha")
            # Only the pending one should be in the digest.
            assert digest["projects"][0]["count"] == 1
        finally:
            store.close()
