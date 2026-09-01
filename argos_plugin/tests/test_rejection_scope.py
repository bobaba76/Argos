"""Regression: a slot-less rejection must never block an entire category.

Live incident 2026-09-01: the review sweep rejected two pending candidates
whose payloads had no claim slot (attribute/fact_type/relation/...). The old
rejection_key derived predicate = bare category ('context_note'), so the
ledger gained ('user','context_note','default_user')-style rows and EVERY
subsequent slot-less remember() in those categories returned None via
rejection_check — silent total write failure for weeks of category.

Guard: rejections are only enforceable when the claim slot is known.
Slot-less candidates produce no blocking key. Specific claims
('user/personal_fact:age') must still block paraphrased re-assertions.
"""
from __future__ import annotations

import sys
from pathlib import Path

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))

from store import DuckDBMemoryStore  # noqa: E402


def _pollute_with_degenerate_ledger_rows(store: DuckDBMemoryStore) -> None:
    """Simulate pre-fix pollution: slot-less rejections keyed on bare category."""
    with store._lock:
        for cat in ("context_note", "insight"):
            store.connection.execute(
                """INSERT OR REPLACE INTO rejection_ledger
                   (subject, predicate, user_scope, reason, created_at)
                   VALUES ('user', ?, 'default_user', 'review_rejected', '2026-09-01T17:39:00+00:00')""",
                [cat],
            )


def test_slotless_rejection_does_not_block_category_writes(tmp_path):
    store = DuckDBMemoryStore(tmp_path / "t.duckdb", embedder=None)
    try:
        _pollute_with_degenerate_ledger_rows(store)

        r = store.remember(content="sample context note", category="context_note")
        assert r is not None, "slot-less context_note write must not be blocked"
        assert r.category == "context_note"
        store.delete_memory(r.memory_id)

        r2 = store.remember(content="a useful insight", category="insight")
        assert r2 is not None, "slot-less insight write must not be blocked"
        store.delete_memory(r2.memory_id)
    finally:
        store.close()


def test_specific_claim_rejection_still_blocks_same_slot(tmp_path):
    store = DuckDBMemoryStore(tmp_path / "t.duckdb", embedder=None)
    try:
        # Reject the specific claim slot 'personal_fact:age' for subject 'user'.
        with store._lock:
            store.record_rejection(
                "personal_fact", {"attribute": "age"},
                reason="review_rejected",
            )

        # Same slot, paraphrased content -> must be blocked.
        r = store.remember(
            content="I am forty years old", category="personal_fact",
            payload={"attribute": "age"},
        )
        assert r is None, "same claim slot must still be blocked"

        # Different slot -> must NOT be blocked.
        r2 = store.remember(
            content="I live in Roodepoort", category="personal_fact",
            payload={"attribute": "location"},
        )
        assert r2 is not None, "a different claim slot must not be blocked"
        store.delete_memory(r2.memory_id)
    finally:
        store.close()