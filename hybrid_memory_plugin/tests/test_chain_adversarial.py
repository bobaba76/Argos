"""Adversarial chain tests (Perseus review point 2 remainder).

Attack-surface checks for the evolution-chain machinery: high-similarity
supersession, updating a mid-chain member, restoring a quarantined
historical version, and provenance preservation across updates. All are
store-level, deterministic, and LLM-free.

Historical records (superseded, quarantined) are observed via
``get_memories_by_ids(..., include_quarantined=True)`` — the default
view only exposes live heads.
"""
from __future__ import annotations

import sys
from pathlib import Path

_plugin_dir = Path(__file__).resolve().parent.parent
for _path in (_plugin_dir.parent, _plugin_dir):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from store import DuckDBMemoryStore  # noqa: E402


def _all_records(store, memory_id):
    """Fetch a record regardless of status/superseded state."""
    return store.get_memories_by_ids([memory_id], include_quarantined=True)


def test_high_similarity_supersession_no_data_loss(tmp_path):
    """Near-identical supersession must version, never lose, never fork."""
    store = DuckDBMemoryStore(tmp_path / "adv1.duckdb", user_id="alice")
    v1 = store.remember(
        category="personal_fact", content="Alice's office is at 123 Main St"
    )
    v2 = store.update_memory(v1.memory_id, content="Alice's office is at 123 MAIN ST")
    v3 = store.update_memory(
        v2.memory_id, content="alice's office is at 123 Main St (confirmed)"
    )

    hist = store.get_memory_history(v1.memory_id)
    assert len(hist) == 3, f"expected 3 versions, got {len(hist)}"
    texts = [h.content for h in hist]
    assert "123 Main St" in texts[0] and v1.content in texts
    # exactly one live head in this line of history
    live = [h for h in hist if getattr(h, "valid_to", None) is None]
    assert len(live) == 1 and live[0].memory_id == v3.memory_id
    # every intermediate is superseded by its successor — no dangling pointers
    for prev, nxt in zip(hist, hist[1:]):
        assert prev.superseded_by == nxt.memory_id, "broken superseded_by link"
    # nothing was deleted: all three ids still resolvable individually
    for v in (v1, v2, v3):
        rec = _all_records(store, v.memory_id)
        assert rec and rec[0].content, f"version {v.memory_id} lost"
    store.close()


def test_update_mid_chain_member_no_cycle_no_loss(tmp_path):
    """Updating an OLD chain member creates a fork leaf, never a cycle."""
    store = DuckDBMemoryStore(tmp_path / "adv2.duckdb", user_id="alice")
    v1 = store.remember(
        category="personal_fact", content="Alice lives in Southtown"
    )
    v2 = store.update_memory(v1.memory_id, content="Alice lives in Centurion")
    v3 = store.update_memory(v2.memory_id, content="Alice lives in Bayport")

    # adversarial: update the MIDDLE version
    v2b = store.update_memory(v2.memory_id, content="Alice lives in Centurion West")

    # no crash, no cycle: history from every node terminates and never
    # revisits a node
    for start in (v1.memory_id, v2.memory_id, v3.memory_id, v2b.memory_id):
        hist = store.get_memory_history(start)
        ids = [h.memory_id for h in hist]
        assert len(ids) == len(set(ids)), f"cycle detected from {start}"

    # the fork leaf exists and is a proper successor of v2
    hist_v2b = store.get_memory_history(v2b.memory_id)
    assert hist_v2b[-1].memory_id == v2b.memory_id
    assert hist_v2b[-1].valid_to is None
    # original v2 still resolvable and marked superseded by v2b
    recs = _all_records(store, v2.memory_id)
    assert recs and recs[0].superseded_by == v2b.memory_id
    # nothing lost: v1, v3 intact
    for v in (v1, v3):
        assert _all_records(store, v.memory_id)
    store.close()


def test_restore_quarantined_history_version_reattaches(tmp_path):
    """Quarantining a historical version must hide it, not unlink it."""
    store = DuckDBMemoryStore(tmp_path / "adv3.duckdb", user_id="alice")
    v1 = store.remember(
        category="personal_fact", content="Alice works at Lobster Inc"
    )
    v2 = store.update_memory(v1.memory_id, content="Alice works at Salty Co")
    v3 = store.update_memory(v2.memory_id, content="Alice works at Meridian Ltd")

    # quarantine the MIDDLE version
    assert store.quarantine_memory(v2.memory_id, reason="adversarial test") is True
    recs = _all_records(store, v2.memory_id)
    assert recs and recs[0].status == "quarantined"
    # chain still walks (history join is not status-filtered)
    hist = store.get_memory_history(v1.memory_id)
    assert len(hist) == 3
    # restore: back to active, chain position untouched, no duplicate
    assert store.restore_memory(v2.memory_id) is True
    recs = _all_records(store, v2.memory_id)
    assert recs and recs[0].status == "active"
    assert recs[0].superseded_by == v3.memory_id  # link preserved
    hist2 = store.get_memory_history(v1.memory_id)
    assert [h.memory_id for h in hist2] == [v1.memory_id, v2.memory_id, v3.memory_id]
    store.close()


def test_restore_quarantined_head_restores_retrieval(tmp_path):
    """Quarantining the HEAD must hide it from search; restore brings it back."""
    store = DuckDBMemoryStore(tmp_path / "adv4.duckdb", user_id="alice")
    v1 = store.remember(category="personal_fact", content="Alice drives a Tesla")
    v2 = store.update_memory(v1.memory_id, content="Alice drives a Corolla")

    # quarantine the head -> the fact disappears from retrieval
    assert store.quarantine_memory(v2.memory_id, reason="adversarial test")
    assert not store.search("Alice drives", limit=5)
    # restore -> retrieval works again via the same (single) record
    assert store.restore_memory(v2.memory_id)
    hits = store.search("Alice drives", limit=5)
    assert hits and hits[0].memory_id == v2.memory_id
    store.close()


def _approve_with_evidence(store, content, evidence_text, memory_id=None):
    """Create + approve a candidate the way the production flow does."""
    cid = store.save_candidate(
        category="personal_fact",
        content=content,
        payload={},
        evidence_text=evidence_text,
        source="conversation:2026-08-01",
        session_id="sess-adv",
    )
    # save_candidate returns the candidate dict, not the id — extract it
    cid = cid["candidate_id"]
    return store.review_candidate(
        candidate_id=cid,
        decision="approved",
        review_model="test",
        reason="adversarial test approval",
        evidence_retention="full",
        supersedes_memory_id=memory_id,
    )


def test_update_preserves_provenance_and_chain_links(tmp_path):
    """Every update must carry provenance forward and version the old record."""
    store = DuckDBMemoryStore(tmp_path / "adv5.duckdb", user_id="alice")
    approval = _approve_with_evidence(
        store,
        "Alice's favourite book is Dune",
        "Alice said: I love Dune, it is my favourite book.",
    )
    v1 = approval["memory"]
    ev1 = store.get_evidence(v1["memory_id"])
    assert ev1 and ev1["evidence_text"].startswith("Alice said")
    assert v1["source"] == "conversation:2026-08-01"

    v2 = store.update_memory(v1["memory_id"], content="Alice's favourite book is Dune Messiah")

    # provenance carried to the new version — evidence trail stays attached
    ev2 = store.get_evidence(v2.memory_id)
    assert ev2 is not None, "evidence trail lost across update"
    assert ev2["evidence_text"] == ev1["evidence_text"]
    assert v2.source == v1["source"]
    assert v2.category == v1["category"]
    # old version preserved + properly linked (never overwritten) —
    # re-fetch: v1's dict was captured BEFORE the update ran
    old = _all_records(store, v1["memory_id"])[0]
    assert old.valid_to is not None
    assert old.superseded_by == v2.memory_id
    # original content still readable from the historical record
    assert old.content == "Alice's favourite book is Dune"
    store.close()