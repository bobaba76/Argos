"""Regression test for #342: _text_search_raw pool LIMIT truncates before ranking.

The text-search arm fetches ILIKE matches in scan order and BM25-ranks the
fetched pool in Python afterwards. A hard LIMIT applied BEFORE ranking
silently drops any target whose scan position is beyond the cap, which
collapses RRF hybrid ranking (recall@5 0.9719 -> 0.8844 on the self-corpus
gate). The pool cap must exceed the corpus size (2000 > 1227 records).

Run with (Hermes venv python, offline):
    python -m pytest tests/test_text_pool_limit.py -v
"""
from __future__ import annotations

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


class TestTextSearchPoolLimit:
    def test_pool_includes_all_matches_beyond_500(self, store):
        """A target at scan position > 500 must still surface in the text arm.

        Regression for #342: LIMIT 500 truncated the ILIKE pool before BM25
        ranking, so any matching record beyond row 500 in scan order lost its
        text-arm RRF contribution entirely.
        """
        # Distinct contents so semantic dedup does not collapse them.
        for i in range(600):
            store.remember(
                category="context_note",
                content=f"zebraflight filler record number {i} with unique words",
            )
        target = store.remember(
            category="context_note",
            content="zebraflight target record with completely different words",
        )

        results = store._text_search_raw("zebraflight", limit=50, excluded=set())

        # The pool must contain every matching record — the cap is a
        # performance bound, never a ranking filter.
        assert len(results) == 601
        ids = [r.memory_id for r in results]
        assert target.memory_id in ids

    def test_target_surfaces_through_public_search(self, store):
        """End-to-end: the target beyond 500 is retrievable via store.search."""
        for i in range(600):
            store.remember(
                category="context_note",
                content=f"zebraflight filler record number {i} with unique words",
            )
        target = store.remember(
            category="context_note",
            content="zebraflight target record with completely different words",
        )

        results = store.search("zebraflight", limit=10, suppress_retrieval=True)
        ids = [r.memory_id for r in results]
        assert target.memory_id in ids