"""Standalone test script for the argos plugin.

Run with:
    python -m argos.tests.run_tests

Or from the plugin directory:
    python tests/run_tests.py

This does NOT require pytest â€” it's a plain script that prints
PASS/FAIL for each test. Use this if you don't have pytest installed.

For pytest, use:
    python -m pytest tests/test_argos.py -v
"""
from __future__ import annotations

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Import submodules directly (not via the package) so we don't trigger
# __init__.py which imports agent.memory_provider (Hermes runtime only).
_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))


def _print_result(name: str, passed: bool, detail: str = "") -> None:
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}" + (f" â€” {detail}" if detail else ""))


def test_embeddings_fallback():
    """Embedder should return [] gracefully if model can't load."""
    from embeddings import LocalEmbedder

    emb = LocalEmbedder("nonexistent-model-xyz")
    # Don't call _ensure_loaded â€” just check embed returns [] on failure.
    result = emb.embed("test text")
    # Will either load (if cached) or return [] â€” both are valid.
    # The key invariant: it never raises.
    assert isinstance(result, list)
    print("  [PASS] embeddings_fallback â€” embed() returns list, never raises")


def test_embeddings_is_query_flag():
    """is_query flag must be accepted without error and not change the list type."""
    from embeddings import LocalEmbedder

    emb = LocalEmbedder("nonexistent-model-xyz")
    result = emb.embed("test text", is_query=True)
    assert isinstance(result, list)
    batch = emb.embed_batch(["a", "b"], is_query=True)
    assert isinstance(batch, list) and len(batch) == 2
    print("  [PASS] embeddings_is_query_flag")


def test_embeddings_query_prefix_bge():
    """BGE models must get a query instruction; symmetric models must not."""
    from embeddings import LocalEmbedder, _query_instruction_for

    bge_instr = _query_instruction_for("BAAI/bge-small-en-v1.5")
    assert bge_instr != "", "BGE model should have a query instruction"

    mini_instr = _query_instruction_for("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
    assert mini_instr == "", "Symmetric model should have no query instruction"

    emb = LocalEmbedder("BAAI/bge-small-en-v1.5")
    doc_text = emb._prepare_text("hello", is_query=False)
    query_text = emb._prepare_text("hello", is_query=True)
    assert doc_text == "hello"
    assert query_text != "hello"
    assert "hello" in query_text
    print("  [PASS] embeddings_query_prefix_bge")


def test_embeddings_default_model():
    """The default model must be bge-small-en-v1.5."""
    from embeddings import _DEFAULT_MODEL

    assert "bge-small-en-v1.5" in _DEFAULT_MODEL, f"Expected bge-small-en-v1.5, got {_DEFAULT_MODEL}"
    print("  [PASS] embeddings_default_model")


def test_store_init_and_save(tmp_path: Path):
    """DuckDB store should initialize and save a memory."""
    from store import DuckDBMemoryStore

    store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user")
    rec = store.remember(
        category="personal_fact",
        content="User is 38 years old and lives in Springfield",
        tags=["age", "location"],
    )
    assert rec is not None, "remember() returned None"
    assert rec.memory_id.startswith("mem-")
    assert rec.category == "personal_fact"
    assert store.count() == 1
    store.close()
    print("  [PASS] store_init_and_save â€” memory saved with correct fields")


def test_store_search_text(tmp_path: Path):
    """Text search should find saved memories."""
    from store import DuckDBMemoryStore

    store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user")
    store.remember(category="personal_fact", content="User takes FocusTool for example condition")
    store.remember(category="relationship", content="Sam is the user's wife")
    store.remember(category="insight", content="User tends to redirect credit away from himself")

    # Text search for "FocusTool"
    results = store.search("FocusTool", limit=5)
    assert len(results) >= 1, f"Expected at least 1 result, got {len(results)}"
    assert any("FocusTool" in r.content for r in results), "FocusTool not found in results"

    # Text search for "Sam"
    results = store.search("Sam", limit=5)
    assert len(results) >= 1, f"Expected at least 1 result for Sam, got {len(results)}"

    store.close()
    print("  [PASS] store_search_text â€” text search finds relevant memories")


def test_store_dedup(tmp_path: Path):
    """Saving the same content twice should dedup."""
    from store import DuckDBMemoryStore

    store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user")
    store.remember(category="personal_fact", content="User has example condition diagnosis")
    # Same content again â€” should be deduped.
    rec2 = store.remember(category="personal_fact", content="User has example condition diagnosis")
    assert rec2 is None, "Duplicate should be deduped"
    assert store.count() == 1, f"Expected 1 record, got {store.count()}"
    store.close()
    print("  [PASS] store_dedup â€” duplicate content is not stored twice")


def test_store_update_and_delete(tmp_path: Path):
    """Update and delete should work by memory_id."""
    from store import DuckDBMemoryStore

    store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user")
    rec = store.remember(category="personal_fact", content="User takes 150mg ExampleMedication")
    assert rec is not None
    mid = rec.memory_id

    # Update.
    updated = store.update_memory(mid, content="User takes 300mg ExampleMedication")
    assert updated is not None
    assert "300mg" in updated.content
    # Versioning: update creates a new record that supersedes the old one.
    assert updated.memory_id != mid

    # Delete the current (head) version — chain-aware delete promotes the
    # predecessor to current instead of leaving the chain headless.
    deleted = store.delete_memory(updated.memory_id)
    assert deleted
    assert store.count() == 1
    # The predecessor (mid) is now current again.
    promoted = store.get_memories_by_ids([mid])
    assert promoted and promoted[0].valid_to is None
    assert promoted[0].superseded_by is None

    store.close()
    print("  [PASS] store_update_and_delete â€” update and chain-aware delete work correctly")


def test_store_record_retrieval_explicit(tmp_path: Path):
    """Provider-path contract: suppress_retrieval search must not inflate
    counts; record_retrieval must credit only the explicitly injected ids."""
    from store import DuckDBMemoryStore

    store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user")
    a = store.remember(category="personal_fact", content="User likes hiking trails")
    b = store.remember(category="personal_fact", content="User owns a mountain bike")
    assert a is not None and b is not None

    results = store.search("hiking", limit=1, suppress_retrieval=True)
    assert results, "suppress_retrieval search should still return results"
    assert all(r.retrieval_count == 0 for r in results), (
        "suppress_retrieval search must not inflate retrieval_count"
    )

    store.record_retrieval([results[0].memory_id])
    fetched = store.get_memories_by_ids([results[0].memory_id, b.memory_id])
    by_id = {r.memory_id: r for r in fetched}
    assert by_id[results[0].memory_id].retrieval_count == 1, (
        "explicitly injected memory should get exactly one retrieval credit"
    )
    assert by_id[b.memory_id].retrieval_count == 0, (
        "non-injected memory must not gain retrieval credit"
    )

    store.close()
    print("  [PASS] store_record_retrieval_explicit — only injected ids get retrieval credit")


def test_evidence_provenance_on_approval(tmp_path: Path):
    """Approving a candidate persists provenance; delete removes it (full
    retention); 'hash' mode stores a digest instead of raw text."""
    from store import DuckDBMemoryStore

    store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user")
    saved = store.save_candidate(
        category="personal_fact",
        content="User prefers cold brew coffee",
        payload={
            "source_session_id": "sess-ev1",
            "user_scope": "test_user",
            "extraction_method": "regex",
        },
        evidence_text="I only drink cold brew these days.",
        evidence_role="user",
        source_timestamp="2026-08-12T08:00:00+00:00",
    )
    assert saved and saved.get("candidate_id")
    result = store.review_candidate(saved["candidate_id"], decision="approved")
    assert result and result.get("memory"), "candidate should approve into a memory"
    mem_id = result["memory"]["memory_id"]

    ev = store.get_evidence(mem_id)
    assert ev is not None, "evidence row must exist after approval"
    assert ev["evidence_text"] == "I only drink cold brew these days."
    assert ev["evidence_role"] == "user"
    assert ev["source_session_id"] == "sess-ev1"
    assert ev["reviewer_decision"] == "approved"
    assert ev["extraction_method"] == "regex"

    # 'hash' retention stores a sha256 digest, not the raw statement.
    store2 = DuckDBMemoryStore(tmp_path / "test2.duckdb", user_id="test_user")
    saved2 = store2.save_candidate(
        category="personal_fact",
        content="User prefers espresso shots",
        payload={
            "source_session_id": "sess-ev2",
            "user_scope": "test_user",
            "extraction_method": "llm_extraction",
        },
        evidence_text="I'll have an espresso.",
        evidence_role="user",
        source_timestamp="2026-08-12T08:01:00+00:00",
    )
    r2 = store2.review_candidate(
        saved2["candidate_id"], decision="approved", evidence_retention="hash"
    )
    ev2 = store2.get_evidence(r2["memory"]["memory_id"])
    assert ev2 is not None and ev2["evidence_text"] != "I'll have an espresso."
    assert len(ev2["evidence_text"]) == 64, "sha256 hex digest expected"

    # Delete removes provenance with the memory.
    assert store.delete_memory(mem_id)
    assert store.get_evidence(mem_id) is None, "evidence must be deleted with memory"

    store.close()
    store2.close()
    print("  [PASS] evidence_provenance_on_approval — provenance lifecycle works")


def test_embedding_model_path_resolution(tmp_path: Path):
    """Local-first resolution: existing path > <home>/models/<name> > hub name."""
    from embeddings import _resolve_embedding_model_path

    model_dir = tmp_path / "models" / "bge-small-en-v1.5"
    model_dir.mkdir(parents=True)

    assert _resolve_embedding_model_path(str(model_dir)) == str(model_dir)

    resolved = _resolve_embedding_model_path(
        "BAAI/bge-small-en-v1.5", hermes_home=tmp_path
    )
    assert resolved == str(model_dir), f"expected local path, got {resolved!r}"

    fallback = _resolve_embedding_model_path(
        "BAAI/bge-small-en-v1.5", hermes_home=tmp_path / "empty"
    )
    assert fallback == "BAAI/bge-small-en-v1.5"

    # Empty/None with a local copy present stays local-first.
    assert _resolve_embedding_model_path("", hermes_home=tmp_path) == str(model_dir)
    # Empty/None with no local copy resolves to the module default hub name.
    assert _resolve_embedding_model_path("", hermes_home=tmp_path / "empty") == (
        "BAAI/bge-small-en-v1.5"
    )
    print("  [PASS] embedding_model_path_resolution — local-first resolution")


def test_retriever_seam(tmp_path: Path):
    """store.search delegates to the configured retriever; the default
    DuckDBRetriever preserves the pre-seam behavior."""
    from store import DuckDBMemoryStore

    calls = {"n": 0}

    class StubRetriever:
        def search(self, query, limit, category_filter=None, project_id=None,
                   suppress_retrieval=False, **kwargs):
            calls["n"] += 1
            calls["query"] = query
            calls["limit"] = limit
            return []

    store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user")
    store.set_retriever(StubRetriever())
    assert store.search("hello", limit=7, suppress_retrieval=True) == []
    assert calls["n"] == 1 and calls["query"] == "hello" and calls["limit"] == 7

    store2 = DuckDBMemoryStore(tmp_path / "test2.duckdb", user_id="test_user")
    store2.remember(category="personal_fact", content="User likes cold brew")
    results = store2.search("cold brew", limit=5)
    assert results and results[0].content == "User likes cold brew"

    store.close()
    store2.close()
    print("  [PASS] retriever_seam — pluggable engine + default path intact")


def test_store_category_filter(tmp_path: Path):
    """Category filter should narrow results."""
    from store import DuckDBMemoryStore

    store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user")
    store.remember(category="personal_fact", content="User lives in Springfield")
    store.remember(category="goal", content="User wants to taper off ExampleMedication")

    results = store.search("Springfield", limit=10, category_filter="goal")
    assert all(r.category == "goal" for r in results), "Category filter not working"

    store.close()
    print("  [PASS] store_category_filter â€” category filter narrows results")


def test_store_maintenance_preview(tmp_path: Path):
    """Consolidation preview must be reversible and non-mutating."""
    from datetime import datetime, timedelta, timezone
    from store import DuckDBMemoryStore

    store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user", embedder=None)
    rec = store.remember(category="context_note", content="Temporary project note about an old migration")
    assert rec is not None
    old = (datetime.now(timezone.utc) - timedelta(days=45)).isoformat()
    store.connection.execute(
        "UPDATE memory_records SET created_at = ?, expires_at = NULL, durability = 'temporary', confidence = 0.4 WHERE memory_id = ?",
        [old, rec.memory_id],
    )
    report = store.consolidate(dry_run=True, max_actions=10, min_age_days=30)
    assert report["dry_run"] is True
    assert report["quarantined_count"] == 0
    assert report["candidate_count"] >= 1
    assert store.search("old migration", limit=5)
    store.close()
    print("  [PASS] store_maintenance_preview â€” dry-run is non-mutating")


def test_graph_init_and_query(tmp_path: Path):
    """Kuzu graph should initialize and support node/edge operations."""
    from graph import KuzuGraphStore

    graph = KuzuGraphStore(tmp_path / "test_kuzu", user_id="test_user")
    graph.add_relationship(
        source="user", source_type="person",
        relation="married_to",
        target="Sam", target_type="person",
    )
    graph.add_relationship(
        source="user", source_type="person",
        relation="takes_medication",
        target="FocusTool", target_type="medication",
    )

    # Query outgoing edges from user.
    edges = graph.query_graph("user")
    assert len(edges) == 2, f"Expected 2 edges, got {len(edges)}"
    relations = {e["relation"] for e in edges}
    assert "married_to" in relations
    assert "takes_medication" in relations

    # Search by term.
    sam_edges = graph.search_graph("Sam")
    assert len(sam_edges) >= 1, "Sam not found in graph search"

    graph.close()
    print("  [PASS] graph_init_and_query â€” nodes and edges created and queried")


def test_graph_memory_evidence_query(tmp_path: Path):
    """Graph-aware lookup should return source memory evidence IDs."""
    from graph import KuzuGraphStore

    graph = KuzuGraphStore(tmp_path / "test_kuzu", user_id="test_user")
    graph.index_memory(
        "mem-kubernetes", "personal_fact",
        "User uses Kubernetes for local deployments", ["devops"], use_llm=False,
    )
    assert "mem-kubernetes" in graph.memory_ids_for_query("Kubernetes", limit=10)
    graph.close()
    print("  [PASS] graph_memory_evidence_query â€” graph returns memory IDs")


def test_graph_extraction_gate():
    """Graph extraction must reject sentence payloads before indexing."""
    from graph import _valid_graph_entity

    assert _valid_graph_entity("Sam")
    assert _valid_graph_entity("know more about the watcher")
    assert not _valid_graph_entity("expecting me to be loading new codes into this system")
    assert not _valid_graph_entity("Location")
    print("  [PASS] graph_extraction_gate â€” sentence payloads rejected")


def test_graph_purge_junk(tmp_path: Path):
    """purge_junk_entities should quarantine stop-word nodes."""
    from graph import KuzuGraphStore

    graph = KuzuGraphStore(tmp_path / "test_kuzu", user_id="test_user")
    # Add a real node and a junk node.
    graph.upsert_node("FocusTool", "medication")
    graph.upsert_node("the", "concept")  # junk
    graph.upsert_node("ab", "concept")   # too short â€” junk

    before = graph.count_nodes()
    quarantined = graph.purge_junk_entities()
    after = graph.count_nodes()

    assert quarantined >= 2, f"Expected at least 2 junk quarantined, got {quarantined}"
    assert after == before, f"Quarantine should preserve node count: {before} -> {after}"
    # Real node should survive.
    nodes = graph.list_nodes()
    ids = {n["id"] for n in nodes}
    assert "FocusTool" in ids, "Real node was purged!"
    assert "the" not in ids, "Junk node survived purge!"

    graph.close()
    print("  [PASS] graph_purge_junk â€” stop-word nodes quarantined, real nodes kept")


def test_extractor_facts():
    """Generic extractor should find facts across different topics."""
    from extractor import extract_from_turn

    # Personal/health topic.
    user_msg = (
        "I take FocusTool and CalmTool for my example condition. "
        "Sam is my wife. "
        "I'm working on tapering off ExampleMedication. "
        "I tend to redirect credit away from myself. "
        "I prefer direct communication."
    )
    facts = extract_from_turn(user_msg, "Assistant response here", use_llm_fallback=False)
    assert len(facts) >= 3, f"Expected at least 3 facts, got {len(facts)}"

    categories = {f["category"] for f in facts}
    assert "relationship" in categories, f"relationship not in {categories}"

    # Work/tech topic — should also extract facts.
    tech_msg = (
        "I use Vim as my primary editor. "
        "I work at TechCorp as a backend engineer. "
        "I'm learning Rust. "
        "I switched from Docker Swarm to Kubernetes. "
        "I always test before deploying."
    )
    tech_facts = extract_from_turn(tech_msg, "", use_llm_fallback=False)
    assert len(tech_facts) >= 3, f"Expected at least 3 tech facts, got {len(tech_facts)}"

    tech_categories = {f["category"] for f in tech_facts}
    # Should find personal_fact (use/work), goal (learning), event (switched), preference (always).
    assert "personal_fact" in tech_categories, f"personal_fact not in {tech_categories}"

    # Should NOT extract from assistant content.
    facts_from_assistant = extract_from_turn("", "I take FocusTool for example condition", use_llm_fallback=False)
    assert len(facts_from_assistant) == 0, "Should not extract from assistant content"

    # Should NOT extract transient states.
    transient_facts = extract_from_turn("I am tired and hungry right now.", "", use_llm_fallback=False)
    assert len(transient_facts) == 0, f"Should not extract transient states, got {transient_facts}"

    print("  [PASS] extractor_facts â€” generic extraction finds durable facts across topics")


def test_provider_lifecycle(tmp_path: Path):
    """Full provider lifecycle: init, save, search, shutdown."""
    # We need to test the provider directly without Hermes imports.
    # Import the store/graph/embedder directly since __init__.py imports
    # from agent.memory_provider which requires the Hermes runtime.
    from embeddings import LocalEmbedder
    from store import DuckDBMemoryStore
    from graph import KuzuGraphStore
    from extractor import extract_from_turn

    embedder = LocalEmbedder()
    store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user", embedder=embedder)
    graph = KuzuGraphStore(tmp_path / "test_kuzu", user_id="test_user")

    # Save via store.
    rec = store.remember(
        category="personal_fact",
        content="User has example condition diagnosis, example condition, and anxiety",
        tags=["diagnosis", "mental_health"],
    )
    assert rec is not None

    # Search.
    results = store.search("example condition diagnosis", limit=5)
    assert len(results) >= 1

    # Extract + save.
    facts = extract_from_turn("I take Medication-C for mood stabilization", "")
    for f in facts:
        store.remember(category=f["category"], content=f["content"], tags=f.get("tags", []))

    # Graph.
    graph.add_relationship("user", "person", "takes_medication", "Medication-C", "medication")
    edges = graph.query_graph("user")
    assert len(edges) >= 1

    store.close()
    graph.close()
    print("  [PASS] provider_lifecycle â€” full init/save/search/graph/shutdown works")


def test_graceful_degradation_no_embeddings(tmp_path: Path):
    """Store should work with text search when embeddings are unavailable."""
    from store import DuckDBMemoryStore

    # No embedder â€” text search only.
    store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user", embedder=None)
    store.remember(category="personal_fact", content="User is tapering off ExampleMedication")
    store.remember(category="event", content="User started therapy with a somatic therapist")

    results = store.search("ExampleMedication taper", limit=5)
    assert len(results) >= 1, "Text search should work without embeddings"
    assert any("ExampleMedication" in r.content for r in results)

    store.close()
    print("  [PASS] graceful_degradation â€” text search works without embeddings")


def test_user_scoping(tmp_path: Path):
    """Memories should be scoped by user_id."""
    from store import DuckDBMemoryStore

    store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="user_a")
    store.remember(category="personal_fact", content="User A's private fact")

    # Switch to user B â€” should not see user A's memories.
    store.set_user_scope("user_b")
    results = store.search("private fact", limit=5)
    assert len(results) == 0, f"User B should not see user A's memories, got {len(results)}"

    # Switch back to user A.
    store.set_user_scope("user_a")
    results = store.search("private fact", limit=5)
    assert len(results) >= 1, "User A should see their own memories"

    store.close()


def test_rrf_fuse_combines_both_lists():
    """RRF should produce a score for items in either or both lists."""
    from store import DuckDBMemoryStore, MemoryRecord

    vec = [
        MemoryRecord(memory_id="a", category="personal_fact", content="alpha", similarity=0.9),
        MemoryRecord(memory_id="b", category="personal_fact", content="beta", similarity=0.7),
    ]
    text = [
        MemoryRecord(memory_id="b", category="personal_fact", content="beta", similarity=0.5),
        MemoryRecord(memory_id="c", category="personal_fact", content="gamma", similarity=0.5),
    ]
    fused = DuckDBMemoryStore._rrf_fuse(vec, text)
    ids = {r.memory_id for r in fused}
    assert ids == {"a", "b", "c"}, f"RRF must include items from both lists, got {ids}"
    assert fused[0].memory_id == "b", "Item in both lists must rank highest"
    print("  [PASS] rrf_fuse_combines_both_lists")


def test_rrf_score_in_zero_one_range():
    """Normalized RRF scores must be in [0, 1]."""
    from store import DuckDBMemoryStore, MemoryRecord

    vec = [MemoryRecord(memory_id=f"v{i}", category="personal_fact", content=f"v{i}", similarity=0.5) for i in range(10)]
    text = [MemoryRecord(memory_id=f"t{i}", category="personal_fact", content=f"t{i}", similarity=0.5) for i in range(10)]
    fused = DuckDBMemoryStore._rrf_fuse(vec, text)
    for r in fused:
        assert 0.0 <= r.similarity <= 1.0, f"Score {r.similarity} out of [0,1]"
    print("  [PASS] rrf_score_in_zero_one_range")


def test_feedback_boosts_helpful_memories(tmp_path: Path):
    """A memory marked helpful should rank above one that wasn't."""
    from store import DuckDBMemoryStore

    store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user", embedder=None)
    rec_normal = store.remember(category="personal_fact", content="User likes apples for snacks")
    rec_helpful = store.remember(category="personal_fact", content="User likes bananas for snacks")
    assert rec_normal and rec_helpful
    store.record_feedback(rec_helpful.memory_id, "helpful")

    results = store.search("snacks", limit=5)
    assert len(results) >= 2
    helpful_rank = next(i for i, r in enumerate(results) if r.memory_id == rec_helpful.memory_id)
    normal_rank = next(i for i, r in enumerate(results) if r.memory_id == rec_normal.memory_id)
    assert helpful_rank < normal_rank, "Helpful memory should rank higher"
    store.close()
    print("  [PASS] feedback_boosts_helpful_memories")


def test_feedback_penalizes_dismissed_memories(tmp_path: Path):
    """A memory marked dismissed should rank below one that wasn't."""
    from store import DuckDBMemoryStore

    store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user", embedder=None)
    rec_dismissed = store.remember(category="personal_fact", content="User likes apples for snacks")
    rec_normal = store.remember(category="personal_fact", content="User likes bananas for snacks")
    assert rec_dismissed and rec_normal
    store.record_feedback(rec_dismissed.memory_id, "dismissed")

    results = store.search("snacks", limit=5)
    assert len(results) >= 2
    dismissed_rank = next(i for i, r in enumerate(results) if r.memory_id == rec_dismissed.memory_id)
    normal_rank = next(i for i, r in enumerate(results) if r.memory_id == rec_normal.memory_id)
    assert normal_rank < dismissed_rank, "Dismissed memory should rank lower"
    store.close()
    print("  [PASS] feedback_penalizes_dismissed_memories")


def test_recency_boost_is_nonnegative():
    """Recency boost must be >= 0 and decay with age."""
    from store import DuckDBMemoryStore
    from datetime import datetime, timezone, timedelta

    now = datetime.now(timezone.utc).isoformat()
    old = (datetime.now(timezone.utc) - timedelta(days=180)).isoformat()

    boost_now = DuckDBMemoryStore._recency_boost(now)
    boost_old = DuckDBMemoryStore._recency_boost(old)
    boost_none = DuckDBMemoryStore._recency_boost(None)

    assert boost_none == 0.0, "Missing timestamp should give 0 boost"
    assert boost_now > boost_old > 0.0, "Recent must boost more than old, both > 0"
    assert boost_now <= 0.10, "Max boost is 0.10"
    print("  [PASS] recency_boost_is_nonnegative")


def test_keyword_match_boosts_via_rrf(tmp_path: Path):
    """A precise keyword match should surface even if vector similarity is low."""
    from store import DuckDBMemoryStore

    store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user", embedder=None)
    store.remember(category="personal_fact", content="User takes Medication-X 10mg daily")
    store.remember(category="personal_fact", content="User enjoys hiking on weekends")
    results = store.search("Medication-X", limit=5)
    assert len(results) >= 1
    assert "Medication-X" in results[0].content, "Exact keyword match should rank first"
    store.close()
    print("  [PASS] keyword_match_boosts_via_rrf")

    print("  [PASS] user_scoping â€” memories are scoped by user_id")


def test_shared_store_has_update_memory():
    """SharedMemoryStore must expose update_memory (regression: was missing)."""
    from service_client import SharedMemoryStore
    assert hasattr(SharedMemoryStore, "update_memory"), \
        "SharedMemoryStore must have update_memory method"
    print("  [PASS] shared_store_has_update_memory")


def test_shared_graph_has_purge_junk_entities():
    """SharedGraphStore must expose purge_junk_entities (regression: was missing)."""
    from service_client import SharedGraphStore
    assert hasattr(SharedGraphStore, "purge_junk_entities"), \
        "SharedGraphStore must have purge_junk_entities method"
    print("  [PASS] shared_graph_has_purge_junk_entities")


def test_store_method_parity():
    """Every method the provider calls on the store must exist on SharedMemoryStore."""
    from service_client import SharedMemoryStore
    required = {
        "search", "get_memories_by_ids", "remember", "update_memory",
        "consolidate", "save_candidate", "list_candidates", "review_candidate", "quarantine_memory",
        "restore_memory", "record_feedback", "delete_memory",
        "cleanup_junk", "count", "get_insights", "close", "set_user_scope",
    }
    for method in required:
        assert hasattr(SharedMemoryStore, method), \
            f"SharedMemoryStore missing method: {method}"
    print("  [PASS] store_method_parity")


def test_graph_method_parity():
    """Every method the provider calls on the graph must exist on SharedGraphStore."""
    from service_client import SharedGraphStore
    required = {
        "search_graph", "memory_ids_for_query", "query_graph", "traverse_graph",
        "add_relationship", "index_memory", "remove_memory", "purge_junk_entities",
        "close", "set_user_scope",
    }
    for method in required:
        assert hasattr(SharedGraphStore, method), \
            f"SharedGraphStore missing method: {method}"
    print("  [PASS] graph_method_parity")


def test_service_dispatches_update_memory():
    """The memory service must route update_memory to the store."""
    import inspect
    from memory_service import MemoryService
    source = inspect.getsource(MemoryService._call_store)
    assert "update_memory" in source, \
        "MemoryService._call_store must dispatch update_memory"
    print("  [PASS] service_dispatches_update_memory")


def test_memory_update_provider_path_keyword():
    """End-to-end: the public memory_update tool path must call update_memory
    with memory_id as a keyword, compatible with SharedMemoryStore's
    keyword-only signature (def update_memory(self, **kwargs)).

    Regression: the provider passed memory_id positionally, which raised
    TypeError on the live shared-service path. The direct DuckDBMemoryStore
    path accepted positional args, so store-level tests missed it.
    """
    import sys
    import types
    import json as _json

    # The package import needs the repo root (parent of the plugin dir) on
    # sys.path; run_tests.py only inserts the plugin dir itself.
    _repo_root = str(_plugin_dir.parent)
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

    # Stub Hermes runtime deps so __init__.py imports standalone.
    if "agent" not in sys.modules:
        sys.modules["agent"] = types.ModuleType("agent")
    if "agent.memory_provider" not in sys.modules:
        _mp = types.ModuleType("agent.memory_provider")

        class MemoryProvider:
            pass

        _mp.MemoryProvider = MemoryProvider
        sys.modules["agent.memory_provider"] = _mp
    if "tools" not in sys.modules:
        sys.modules["tools"] = types.ModuleType("tools")
    if "tools.registry" not in sys.modules:
        _tr = types.ModuleType("tools.registry")

        def _tool_error(msg):
            return _json.dumps({"error": str(msg)})

        _tr.tool_error = _tool_error
        sys.modules["tools.registry"] = _tr

    try:
        import argos_plugin
    except ModuleNotFoundError:
        # The live install directory is named ``argos`` rather than
        # ``argos_plugin``.
        import argos as argos_plugin
    provider = argos_plugin.ArgosProvider()

    class _StubRecord:
        def __init__(self, memory_id, content, tags):
            self.memory_id = memory_id
            self.category = "personal_fact"
            self.content = content
            self.tags = tags or []
            self.created_at = "2026-08-09T00:00:00"

    class _KeywordOnlyStore:
        # Mirrors SharedMemoryStore.update_memory: **kwargs only. A positional
        # memory_id raises TypeError before the body runs.
        def __init__(self):
            self.last_kwargs = None

        def update_memory(self, **kwargs):
            self.last_kwargs = dict(kwargs)
            if kwargs.get("memory_id") is None:
                return None
            return _StubRecord(
                kwargs["memory_id"], kwargs.get("content"), kwargs.get("tags")
            )

    store = _KeywordOnlyStore()
    provider._store = store
    provider._graph = None  # skip the graph re-index branch

    result = provider.handle_tool_call(
        "memory_update",
        {"memory_id": "mem-123", "content": "updated content", "tags": ["t1"]},
    )
    parsed = _json.loads(result)
    assert parsed.get("status") == "updated", f"expected updated, got: {result}"
    assert parsed.get("memory_id") == "mem-123"
    assert store.last_kwargs is not None, "store.update_memory was never called"
    assert store.last_kwargs.get("memory_id") == "mem-123", \
        "memory_id must reach the store as a keyword argument"
    assert store.last_kwargs.get("content") == "updated content"
    print("  [PASS] memory_update_provider_path_keyword")


def test_llm_fallback_triggers_on_zero_facts():
    """_should_try_llm_fallback must trigger when regex found 0 facts."""
    from extractor import _should_try_llm_fallback
    long_msg = "I just got a new job at TechCorp as a backend engineer, " * 3
    assert _should_try_llm_fallback(long_msg, 0) is True
    print("  [PASS] llm_fallback_triggers_on_zero_facts")


def test_llm_fallback_triggers_on_few_facts_long_message():
    """1 fact from a 200-word message should trigger LLM."""
    from extractor import _should_try_llm_fallback
    long_msg = " ".join(["word"] * 200)
    assert _should_try_llm_fallback(long_msg, 1) is True
    print("  [PASS] llm_fallback_triggers_on_few_facts_long_message")


def test_llm_fallback_skips_short_message_with_facts():
    """1 fact from a short message should NOT trigger LLM."""
    from extractor import _should_try_llm_fallback
    short_msg = "I take Medication-X 10mg daily"
    assert _should_try_llm_fallback(short_msg, 1) is False
    print("  [PASS] llm_fallback_skips_short_message_with_facts")


def test_llm_fallback_skips_short_message_no_facts():
    """Short message with 0 facts should NOT trigger LLM."""
    from extractor import _should_try_llm_fallback
    short_msg = "hey how are you"
    assert _should_try_llm_fallback(short_msg, 0) is False
    print("  [PASS] llm_fallback_skips_short_message_no_facts")


def test_llm_fallback_skips_many_facts():
    """Several facts from a reasonable message should NOT trigger LLM."""
    from extractor import _should_try_llm_fallback
    msg = "I take Medication-X for depression. I live in Springfield. I work at TechCorp."
    assert _should_try_llm_fallback(msg, 3) is False
    print("  [PASS] llm_fallback_skips_many_facts")


def test_text_overlap_detects_high_overlap():
    """_text_overlap must detect near-duplicate phrasings."""
    from extractor import _text_overlap
    assert _text_overlap(
        "user takes medication-x 10mg daily for depression",
        "user takes medication-x 10mg daily",
    ) is True
    print("  [PASS] text_overlap_detects_high_overlap")


def test_text_overlap_rejects_unrelated():
    """_text_overlap must NOT flag unrelated content as duplicate."""
    from extractor import _text_overlap
    assert _text_overlap(
        "user takes medication-x for depression",
        "user enjoys hiking on weekends",
    ) is False
    print("  [PASS] text_overlap_rejects_unrelated")


def test_different_facts_not_deduped(tmp_path: Path):
    """Genuinely different facts must NOT be deduped."""
    from store import DuckDBMemoryStore

    store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user", embedder=None)
    rec1 = store.remember(category="personal_fact", content="User takes Medication-X 10mg daily")
    rec2 = store.remember(category="personal_fact", content="User takes Medication-Y 50mg for ADHD")
    assert rec1 is not None, "First fact should be saved"
    assert rec2 is not None, "Second (different) fact should also be saved"
    assert store.count() == 2
    store.close()
    print("  [PASS] different_facts_not_deduped")


def test_insight_is_valid_category():
    """The store must accept 'insight' as a valid category."""
    from store import VALID_CATEGORIES
    assert "insight" in VALID_CATEGORIES, "insight must be a valid category"
    print("  [PASS] insight_is_valid_category")


def test_save_and_retrieve_insight(tmp_path: Path):
    """Saving an insight and retrieving it via get_insights must work."""
    from store import DuckDBMemoryStore

    store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user", embedder=None)
    rec = store.remember(
        category="insight",
        content="I redirect credit away from myself because I'm afraid of being seen as arrogant",
        tags=["insight", "2024-03-15", "identity", "shame"],
    )
    assert rec is not None, "Insight should be saved"
    assert rec.category == "insight"

    insights = store.get_insights()
    assert len(insights) == 1
    assert "redirect credit" in insights[0].content
    store.close()
    print("  [PASS] save_and_retrieve_insight")


def test_get_insights_filtered_by_tag(tmp_path: Path):
    """get_insights with tags must filter to matching insights."""
    from store import DuckDBMemoryStore

    store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user", embedder=None)
    store.remember(category="insight", content="Insight about work stress", tags=["insight", "work", "stress"])
    store.remember(category="insight", content="Insight about relationship patterns", tags=["insight", "relationships"])
    store.remember(category="insight", content="Insight about shame at work", tags=["insight", "shame", "work"])

    work_insights = store.get_insights(tags=["work"])
    assert len(work_insights) == 2, f"Expected 2 work-tagged insights, got {len(work_insights)}"
    for r in work_insights:
        assert "work" in (r.tags or [])
    store.close()
    print("  [PASS] get_insights_filtered_by_tag")


def test_get_insights_excludes_other_categories(tmp_path: Path):
    """get_insights must only return insight-category records."""
    from store import DuckDBMemoryStore

    store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user", embedder=None)
    store.remember(category="insight", content="I notice I avoid conflict", tags=["insight", "conflict"])
    store.remember(category="personal_fact", content="User takes Medication-X 10mg", tags=["medication"])

    insights = store.get_insights()
    assert len(insights) == 1, "get_insights must not return non-insight categories"
    assert insights[0].category == "insight"
    store.close()
    print("  [PASS] get_insights_excludes_other_categories")


def test_shared_store_has_get_insights():
    """SharedMemoryStore must expose get_insights (regression guard)."""
    from service_client import SharedMemoryStore
    assert hasattr(SharedMemoryStore, "get_insights"), \
        "SharedMemoryStore must have get_insights method"
    print("  [PASS] shared_store_has_get_insights")


def test_service_dispatches_get_insights():
    """The memory service must route get_insights to the store."""
    import inspect
    from memory_service import MemoryService
    source = inspect.getsource(MemoryService._call_store)
    assert "get_insights" in source, \
        "MemoryService._call_store must dispatch get_insights"
    print("  [PASS] service_dispatches_get_insights")


def test_insight_log_skill_exists():
    """The insight-log SKILL.md file must exist."""
    from pathlib import Path
    skill_path = Path(__file__).resolve().parent.parent / "skills" / "insight-log" / "SKILL.md"
    assert skill_path.exists(), f"Skill file must exist at {skill_path}"
    print("  [PASS] insight_log_skill_exists")


def test_insight_log_skill_description_length():
    """The skill description must be <=57 chars (what shows in the prompt)."""
    from pathlib import Path
    skill_path = Path(__file__).resolve().parent.parent / "skills" / "insight-log" / "SKILL.md"
    content = skill_path.read_text(encoding="utf-8")
    for line in content.split("\n"):
        if line.strip().startswith("description:"):
            desc = line.split("description:", 1)[1].strip().strip('"').strip("'")
            assert len(desc) <= 57, f"Description too long ({len(desc)} chars): {desc}"
            break
    print("  [PASS] insight_log_skill_description_length")





def test_graph_extractor_all_categories():
    """Graph extraction should produce relations for every memory category."""
    from graph import extract_graph_relations

    cases = [
        ("Sam is my therapist", "relationship", ["therapy"]),
        ("User works at TechCorp and uses Kubernetes", "personal_fact", ["work", "devops"]),
        ("I prefer Kubernetes over Docker Swarm", "preference", ["devops"]),
        ("I just realized shame shapes my work patterns", "insight", ["insight", "shame", "work"]),
        ("User goal: learn Rust", "goal", ["goal", "rust"]),
        ("Life event: user launched Hermes", "event", ["event", "hermes"]),
        ("User has been using Docker", "context_note", ["ongoing", "devops"]),
    ]
    for content, category, tags in cases:
        relations = extract_graph_relations(content, category, tags)
        assert relations, f"No graph relations extracted for {category}: {content}"
    personal = extract_graph_relations(
        "I take Medication-X for depression and live in Berlin",
        "personal_fact",
        ["health", "location"],
    )
    assert any(r["relation"] == "uses" and r["target"] == "Medication-X" for r in personal)
    assert any(r["relation"] == "lives_in" for r in personal)
    ongoing = extract_graph_relations("User has been using Docker", "context_note", ["devops"])
    assert any(r["relation"] == "uses" and r["target"] == "Docker" for r in ongoing)
    print("  [PASS] graph_extractor_all_categories")


def test_graph_index_and_traverse(tmp_path: Path):
    """Indexed memories should share entity nodes and expose traversal evidence."""
    from graph import KuzuGraphStore

    graph = KuzuGraphStore(tmp_path / "test_kuzu", user_id="test_user")
    graph.index_memory(
        "m1", "personal_fact", "User works at TechCorp and uses Kubernetes",
        ["work", "devops"],
    )
    graph.index_memory(
        "m2", "preference", "User prefers Kubernetes over Docker Swarm",
        ["devops"],
    )
    edges = graph.search_graph("Kubernetes")
    memory_ids = {
        memory_id
        for edge in edges
        for memory_id in edge.get("attributes", {}).get("memory_ids", [])
    }
    assert {"m1", "m2"} <= memory_ids
    neighborhood = graph.traverse_graph("Kubernetes", depth=2)
    node_ids = {node["id"] for node in neighborhood["nodes"]}
    assert {"user", "memory:m1", "memory:m2"} <= node_ids
    assert neighborhood["edges"]
    graph.close()
    print("  [PASS] graph_index_and_traverse")


def test_graph_remove_and_reindex(tmp_path: Path):
    """Removing and re-indexing a memory must refresh its graph evidence."""
    from graph import KuzuGraphStore

    graph = KuzuGraphStore(tmp_path / "test_kuzu", user_id="test_user")
    graph.index_memory("m1", "personal_fact", "User works at TechCorp", ["work"])
    assert graph.traverse_graph("TechCorp")["nodes"]
    assert graph.remove_memory("m1") is True
    assert graph.traverse_graph("TechCorp")["nodes"] == []
    graph.index_memory("m1", "personal_fact", "User works at Acme", ["work"])
    assert graph.traverse_graph("Acme")["nodes"]
    assert graph.traverse_graph("TechCorp")["nodes"] == []
    graph.close()
    print("  [PASS] graph_remove_and_reindex")


def test_graph_priority3_surface():
    """Shared graph and service must expose all Priority 3 methods."""
    import inspect
    from service_client import SharedGraphStore
    from memory_service import MemoryService
    for method in ("traverse_graph", "index_memory", "remove_memory"):
        assert hasattr(SharedGraphStore, method), f"SharedGraphStore missing {method}"
    source = inspect.getsource(MemoryService._call_graph)
    for method in ("traverse_graph", "index_memory", "remove_memory"):
        assert method in source, f"MemoryService missing {method} dispatch"
    print("  [PASS] graph_priority3_surface")


def test_graph_llm_extraction_falls_back_gracefully():
    """LLM-assisted extraction must return [] when LLM unavailable."""
    from graph import extract_graph_relations_llm, extract_graph_relations_hybrid

    llm_result = extract_graph_relations_llm(
        "User works at TechCorp and uses Kubernetes for deployment", "personal_fact"
    )
    assert llm_result == [], "LLM extraction should return [] when unavailable"
    hybrid = extract_graph_relations_hybrid(
        "User works at TechCorp and uses Kubernetes for deployment",
        "personal_fact", ["work", "devops"], use_llm=True,
    )
    assert hybrid, "Hybrid should return regex results even when LLM unavailable"
    assert any(r["target"] == "TechCorp" for r in hybrid)
    print("  [PASS] graph_llm_extraction_falls_back_gracefully")


def test_graph_search_uses_kuzu_filter(tmp_path: Path):
    """search_graph should push WHERE CONTAINS into Kuzu and respect limit."""
    from graph import KuzuGraphStore

    graph = KuzuGraphStore(tmp_path / "test_kuzu", user_id="test_user")
    graph.index_memory("m1", "personal_fact", "User works at TechCorp and uses Kubernetes", ["work", "devops"], use_llm=False)
    graph.index_memory("m2", "preference", "User prefers Kubernetes over Docker Swarm", ["devops"], use_llm=False)
    edges = graph.search_graph("Kubernetes")
    assert len(edges) >= 2, f"Expected >= 2 edges, got {len(edges)}"
    limited = graph.search_graph("Kubernetes", limit=1)
    assert len(limited) <= 1
    assert graph.search_graph("NonExistentEntity123") == []
    graph.close()
    print("  [PASS] graph_search_uses_kuzu_filter")


def test_graph_traverse_uses_targeted_queries(tmp_path: Path):
    """traverse_graph should use targeted per-hop queries, not full scan."""
    from graph import KuzuGraphStore

    graph = KuzuGraphStore(tmp_path / "test_kuzu", user_id="test_user")
    graph.index_memory("m1", "personal_fact", "User works at TechCorp and uses Kubernetes", ["work", "devops"], use_llm=False)
    d1 = graph.traverse_graph("Kubernetes", depth=1)
    assert any(n["id"] == "user" for n in d1["nodes"])
    assert d1["edges"]
    d2 = graph.traverse_graph("Kubernetes", depth=2)
    d2_ids = {n["id"] for n in d2["nodes"]}
    assert "TechCorp" in d2_ids or "memory:m1" in d2_ids
    assert graph.traverse_graph("NonExistentEntity123")["nodes"] == []
    graph.close()
    print("  [PASS] graph_traverse_uses_targeted_queries")


def test_every_approved_memory_yields_graph_backlink(tmp_path: Path):
    """Every approved memory of each category must yield at least one
    graph edge with a memory_id backlink."""
    from graph import KuzuGraphStore

    graph = KuzuGraphStore(tmp_path / "test_kuzu", user_id="test_user")
    test_cases = [
        ("personal_fact", "User works at TechCorp as a backend engineer", ["work"]),
        ("preference", "User prefers Kubernetes over Docker Swarm for deployment", ["devops"]),
        ("insight", "I realized that shame shapes my work patterns", ["insight", "shame"]),
        ("event", "User started a new job at TechCorp last Monday", ["work", "career"]),
        ("relationship", "User's partner Sam is supportive of their therapy", ["relationships"]),
        ("goal", "User wants to learn Rust for systems programming", ["programming", "rust"]),
        ("context_note", "User mentioned they live in Westford and commute by bike", ["location"]),
    ]
    for idx, (category, content, tags) in enumerate(test_cases):
        memory_id = f"backlink-test-{idx}"
        graph.index_memory(memory_id, category, content, tags, use_llm=False)
        memory_node = f"memory:{memory_id}"
        node = graph._query_node(memory_node)
        assert node is not None, f"{category}: memory node not found"
        edges = graph._query_edges_for_nodes([memory_node])
        backlink_edges = [
            e for e in edges
            if memory_id in [str(x) for x in (e.get("attributes", {}).get("memory_ids") or [])]
        ]
        assert backlink_edges, f"{category}: no edges with memory_ids backlink"
        traversal = graph.traverse_graph(memory_node, depth=2)
        non_memory = [n for n in traversal["nodes"] if n["entity_type"] != "memory" and n["id"] != memory_node]
        assert non_memory, f"{category}: traversal found no linked entities"
    graph.close()
    print("  [PASS] every_approved_memory_yields_graph_backlink")


def test_query_graph_is_bidirectional(tmp_path: Path):
    """query_graph must find edges where entity is a TARGET, not just source."""
    from graph import KuzuGraphStore

    graph = KuzuGraphStore(tmp_path / "test_kuzu", user_id="test_user")
    graph.index_memory("m1", "insight", "I realized that shame shapes my work patterns", ["insight", "shame"], use_llm=False)
    edges = graph.query_graph("shame")
    assert edges, "query_graph('shame') returned 0 — bidirectional query broken"
    assert any(e["target"] == "shame" for e in edges), "should find edges where shame is target"
    graph.close()
    print("  [PASS] query_graph_is_bidirectional")


def run_all():
    """Run all tests and report results."""
    print()
    print("=" * 60)
    print("  Argos Plugin â€” Test Suite")
    print("=" * 60)
    print()

    failures = 0
    tests = [
        ("embeddings_fallback", test_embeddings_fallback, False),
        ("embeddings_is_query_flag", test_embeddings_is_query_flag, False),
        ("embeddings_query_prefix_bge", test_embeddings_query_prefix_bge, False),
        ("embeddings_default_model", test_embeddings_default_model, False),
        ("store_init_and_save", test_store_init_and_save, True),
        ("store_search_text", test_store_search_text, True),
        ("store_dedup", test_store_dedup, True),
        ("store_update_and_delete", test_store_update_and_delete, True),
        ("store_record_retrieval_explicit", test_store_record_retrieval_explicit, True),
        ("evidence_provenance_on_approval", test_evidence_provenance_on_approval, True),
        ("embedding_model_path_resolution", test_embedding_model_path_resolution, True),
        ("retriever_seam", test_retriever_seam, True),
        ("store_category_filter", test_store_category_filter, True),
        ("store_maintenance_preview", test_store_maintenance_preview, True),
        ("graph_init_and_query", test_graph_init_and_query, True),
        ("graph_memory_evidence_query", test_graph_memory_evidence_query, True),
        ("graph_extraction_gate", test_graph_extraction_gate, False),
        ("graph_purge_junk", test_graph_purge_junk, True),
        ("extractor_facts", test_extractor_facts, False),
        ("provider_lifecycle", test_provider_lifecycle, True),
        ("graceful_degradation_no_embeddings", test_graceful_degradation_no_embeddings, True),
        ("user_scoping", test_user_scoping, True),
        ("rrf_fuse_combines_both_lists", test_rrf_fuse_combines_both_lists, False),
        ("rrf_score_in_zero_one_range", test_rrf_score_in_zero_one_range, False),
        ("feedback_boosts_helpful_memories", test_feedback_boosts_helpful_memories, True),
        ("feedback_penalizes_dismissed_memories", test_feedback_penalizes_dismissed_memories, True),
        ("recency_boost_is_nonnegative", test_recency_boost_is_nonnegative, False),
        ("keyword_match_boosts_via_rrf", test_keyword_match_boosts_via_rrf, True),
        ("shared_store_has_update_memory", test_shared_store_has_update_memory, False),
        ("shared_graph_has_purge_junk_entities", test_shared_graph_has_purge_junk_entities, False),
        ("store_method_parity", test_store_method_parity, False),
        ("graph_method_parity", test_graph_method_parity, False),
        ("service_dispatches_update_memory", test_service_dispatches_update_memory, False),
        ("memory_update_provider_path_keyword", test_memory_update_provider_path_keyword, False),
        ("llm_fallback_triggers_on_zero_facts", test_llm_fallback_triggers_on_zero_facts, False),
        ("llm_fallback_triggers_on_few_facts_long_message", test_llm_fallback_triggers_on_few_facts_long_message, False),
        ("llm_fallback_skips_short_message_with_facts", test_llm_fallback_skips_short_message_with_facts, False),
        ("llm_fallback_skips_short_message_no_facts", test_llm_fallback_skips_short_message_no_facts, False),
        ("llm_fallback_skips_many_facts", test_llm_fallback_skips_many_facts, False),
        ("text_overlap_detects_high_overlap", test_text_overlap_detects_high_overlap, False),
        ("text_overlap_rejects_unrelated", test_text_overlap_rejects_unrelated, False),
        ("different_facts_not_deduped", test_different_facts_not_deduped, True),
        ("insight_is_valid_category", test_insight_is_valid_category, False),
        ("save_and_retrieve_insight", test_save_and_retrieve_insight, True),
        ("get_insights_filtered_by_tag", test_get_insights_filtered_by_tag, True),
        ("get_insights_excludes_other_categories", test_get_insights_excludes_other_categories, True),
        ("shared_store_has_get_insights", test_shared_store_has_get_insights, False),
        ("service_dispatches_get_insights", test_service_dispatches_get_insights, False),
        ("insight_log_skill_exists", test_insight_log_skill_exists, False),
        ("insight_log_skill_description_length", test_insight_log_skill_description_length, False),
        ("graph_extractor_all_categories", test_graph_extractor_all_categories, False),
        ("graph_index_and_traverse", test_graph_index_and_traverse, True),
        ("graph_remove_and_reindex", test_graph_remove_and_reindex, True),
        ("graph_priority3_surface", test_graph_priority3_surface, False),
        ("graph_llm_extraction_falls_back_gracefully", test_graph_llm_extraction_falls_back_gracefully, False),
        ("graph_search_uses_kuzu_filter", test_graph_search_uses_kuzu_filter, True),
        ("graph_traverse_uses_targeted_queries", test_graph_traverse_uses_targeted_queries, True),
        ("every_approved_memory_yields_graph_backlink", test_every_approved_memory_yields_graph_backlink, True),
        ("query_graph_is_bidirectional", test_query_graph_is_bidirectional, True),
    ]

    for name, test_fn, needs_tmp in tests:
        print(f"\n  Running: {name}")
        tmp_dir = None
        try:
            if needs_tmp:
                tmp_dir = Path(tempfile.mkdtemp(prefix=f"hmtest_{name}_"))
                test_fn(tmp_dir)
            else:
                test_fn()
        except AssertionError as e:
            _print_result(name, False, str(e))
            failures += 1
        except Exception as e:
            _print_result(name, False, f"{type(e).__name__}: {e}")
            failures += 1
        else:
            pass  # PASS printed inside each test
        finally:
            if tmp_dir and tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)

    print()
    print("=" * 60)
    total = len(tests)
    passed = total - failures
    print(f"  Results: {passed}/{total} passed, {failures} failed")
    print("=" * 60)
    print()

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(run_all())

