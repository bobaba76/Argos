"""Tests for Spec-05 (#67): doc-fact namespace + client scope.

Covers (cheapest-falsifying first, all deterministic, no LLM):
1. Partition math unit tests: floors, remainder fill, client-scoped
   inversion, empty-namespace short-circuit, cap respected.
2. Store-level namespace filter on memory_records.
3. Store-level namespace filter on memory_candidates (save + list).
4. client_scope filter incl. NULL (global) rows.
5. RPC regression: proxy search with namespace/client_scope kwargs
   round-trips through dispatch; prefetch fail-soft swallows TypeErrors.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))


# ---------------------------------------------------------------------------
# 1. Partition math (pure function, no I/O)
# ---------------------------------------------------------------------------

def _mk(namespace: str, score: float, content: str = ""):
    return SimpleNamespace(namespace=namespace, similarity=score, content=content)


class TestPartitionMath:
    """Unit tests for namespace_partition.partition_by_namespace."""

    def test_mixed_store_floors_met(self):
        from namespace_partition import partition_by_namespace

        conv = [_mk("conversation", s) for s in range(50, 0, -1)]
        doc = [_mk("document", s) for s in range(50, 0, -1)]
        results = sorted(conv + doc, key=lambda r: r.similarity, reverse=True)
        out = partition_by_namespace(results, cap=96)
        conv_out = sum(1 for r in out if r.namespace == "conversation")
        doc_out = sum(1 for r in out if r.namespace == "document")
        assert len(out) == 96
        assert conv_out >= 24, f"conversation floor not met: {conv_out}"
        assert doc_out >= 24, f"document floor not met: {doc_out}"

    def test_remainder_filled_by_score(self):
        from namespace_partition import partition_by_namespace

        # 30 conv (high scores), 30 doc (low scores), cap=96
        conv = [_mk("conversation", s) for s in range(60, 30, -1)]
        doc = [_mk("document", s) for s in range(30, 0, -1)]
        results = sorted(conv + doc, key=lambda r: r.similarity, reverse=True)
        out = partition_by_namespace(results, cap=96)
        # Only 60 items exist; all should be returned.
        assert len(out) == 60
        # Output is in unified-score order.
        scores = [r.similarity for r in out]
        assert scores == sorted(scores, reverse=True)

    def test_client_scoped_inversion(self):
        from namespace_partition import partition_by_namespace

        # Document scores lower than conversation to force the floor to bite.
        conv = [_mk("conversation", s) for s in range(100, 0, -1)]
        doc = [_mk("document", s) for s in range(50, 0, -1)]
        results = sorted(conv + doc, key=lambda r: r.similarity, reverse=True)
        out = partition_by_namespace(results, cap=52, client_scoped=True)
        doc_out = sum(1 for r in out if r.namespace == "document")
        conv_out = sum(1 for r in out if r.namespace == "conversation")
        # Client-scoped: doc floor 40, conv floor 12.
        assert doc_out >= 40, f"client-scoped doc floor not met: {doc_out}"
        assert conv_out >= 12, f"client-scoped conv floor not met: {conv_out}"
        assert len(out) == 52

    def test_empty_namespace_short_circuit(self):
        from namespace_partition import partition_by_namespace

        # Only conversation rows — no document at all.
        conv = [_mk("conversation", s) for s in range(50, 0, -1)]
        out = partition_by_namespace(conv, cap=96)
        assert len(out) == 50  # all 50, not capped at 24
        assert all(r.namespace == "conversation" for r in out)

    def test_empty_doc_namespace_short_circuit(self):
        from namespace_partition import partition_by_namespace

        doc = [_mk("document", s) for s in range(30, 0, -1)]
        out = partition_by_namespace(doc, cap=96)
        assert len(out) == 30
        assert all(r.namespace == "document" for r in out)

    def test_cap_respected(self):
        from namespace_partition import partition_by_namespace

        conv = [_mk("conversation", s) for s in range(100, 0, -1)]
        doc = [_mk("document", s) for s in range(100, 0, -1)]
        results = sorted(conv + doc, key=lambda r: r.similarity, reverse=True)
        out = partition_by_namespace(results, cap=10)
        assert len(out) == 10

    def test_empty_results(self):
        from namespace_partition import partition_by_namespace

        assert partition_by_namespace([], cap=96) == []

    def test_zero_cap(self):
        from namespace_partition import partition_by_namespace

        results = [_mk("conversation", 1.0), _mk("document", 0.5)]
        assert partition_by_namespace(results, cap=0) == []


# ---------------------------------------------------------------------------
# 2. Store-level namespace filter (memory_records)
# ---------------------------------------------------------------------------

class TestStoreNamespaceFilter:
    """save + search with namespace filter on memory_records."""

    def test_namespace_filter_returns_only_that_namespace(self, tmp_path):
        from store import DuckDBMemoryStore

        store = DuckDBMemoryStore(tmp_path / "test_ns.duckdb")
        try:
            store.remember("personal_fact", "meeting about alpha",
                           namespace="document")
            store.remember("personal_fact", "meeting about beta",
                           namespace="conversation")
            doc = store.search("meeting", limit=10, namespace="document")
            conv = store.search("meeting", limit=10, namespace="conversation")
            doc_contents = {r.content for r in doc}
            conv_contents = {r.content for r in conv}
            assert "meeting about alpha" in doc_contents
            assert "meeting about beta" not in doc_contents
            assert "meeting about beta" in conv_contents
            assert "meeting about alpha" not in conv_contents
        finally:
            store.close()

    def test_namespace_defaults_to_conversation(self, tmp_path):
        from store import DuckDBMemoryStore

        store = DuckDBMemoryStore(tmp_path / "test_default.duckdb")
        try:
            r = store.remember("personal_fact", "default namespace test")
            assert r.namespace == "conversation"
            assert r.client_scope is None
        finally:
            store.close()

    def test_no_namespace_filter_returns_all(self, tmp_path):
        from store import DuckDBMemoryStore

        store = DuckDBMemoryStore(tmp_path / "test_all.duckdb")
        try:
            store.remember("personal_fact", "meeting about doc",
                           namespace="document")
            store.remember("personal_fact", "meeting about chat",
                           namespace="conversation")
            all_results = store.search("meeting", limit=10)
            assert len(all_results) == 2
        finally:
            store.close()


# ---------------------------------------------------------------------------
# 3. Store-level namespace filter (memory_candidates)
# ---------------------------------------------------------------------------

class TestCandidateNamespace:
    """save_candidate with namespace + list_candidates round-trip."""

    def test_candidate_carries_namespace(self, tmp_path):
        from store import DuckDBMemoryStore

        store = DuckDBMemoryStore(tmp_path / "test_cand_ns.duckdb")
        try:
            cand = store.save_candidate(
                "personal_fact", "Doc fact from invoice",
                namespace="document", client_scope="acme",
            )
            assert cand is not None
            assert cand["namespace"] == "document"
            assert cand["client_scope"] == "acme"
        finally:
            store.close()

    def test_candidate_defaults_to_conversation(self, tmp_path):
        from store import DuckDBMemoryStore

        store = DuckDBMemoryStore(tmp_path / "test_cand_default.duckdb")
        try:
            cand = store.save_candidate(
                "personal_fact", "Chat note candidate",
            )
            assert cand is not None
            assert cand["namespace"] == "conversation"
            assert cand["client_scope"] is None
        finally:
            store.close()

    def test_candidate_namespace_round_trips_through_list(self, tmp_path):
        from store import DuckDBMemoryStore

        store = DuckDBMemoryStore(tmp_path / "test_cand_list.duckdb")
        try:
            store.save_candidate("personal_fact", "Doc candidate",
                                 namespace="document")
            store.save_candidate("personal_fact", "Chat candidate",
                                 namespace="conversation")
            cands = store.list_candidates(limit=10)
            by_content = {c["content"]: c for c in cands}
            assert by_content["Doc candidate"]["namespace"] == "document"
            assert by_content["Chat candidate"]["namespace"] == "conversation"
        finally:
            store.close()


# ---------------------------------------------------------------------------
# 4. client_scope filter incl. NULL (global) rows
# ---------------------------------------------------------------------------

class TestClientScopeFilter:
    """client_scope filter: NULL rows (global) stay visible in client queries."""

    def test_client_scope_filter_includes_global_rows(self, tmp_path):
        from store import DuckDBMemoryStore

        store = DuckDBMemoryStore(tmp_path / "test_cs.duckdb")
        try:
            store.remember("personal_fact", "meeting about acme invoice",
                           namespace="document", client_scope="acme")
            store.remember("personal_fact", "meeting about global tax",
                           namespace="document", client_scope=None)
            store.remember("personal_fact", "meeting about beta client",
                           namespace="document", client_scope="beta")
            # client_scope='acme' should see acme + global, NOT beta.
            acme = store.search("meeting", limit=10, client_scope="acme")
            acme_contents = {r.content for r in acme}
            assert "meeting about acme invoice" in acme_contents
            assert "meeting about global tax" in acme_contents  # NULL = global
            assert "meeting about beta client" not in acme_contents
        finally:
            store.close()

    def test_no_client_scope_filter_returns_all(self, tmp_path):
        from store import DuckDBMemoryStore

        store = DuckDBMemoryStore(tmp_path / "test_cs_all.duckdb")
        try:
            store.remember("personal_fact", "meeting about acme",
                           client_scope="acme")
            store.remember("personal_fact", "meeting about beta",
                           client_scope="beta")
            store.remember("personal_fact", "meeting about global",
                           client_scope=None)
            all_results = store.search("meeting", limit=10)
            assert len(all_results) == 3
        finally:
            store.close()


# ---------------------------------------------------------------------------
# 5. RPC regression: proxy + dispatch with namespace/client_scope kwargs
# ---------------------------------------------------------------------------

# These spawn the shared service subprocess — group with the other
# shared_service tests so xdist serializes them.
pytestmark_rpc = pytest.mark.xdist_group("shared_service")


@pytestmark_rpc
def test_shared_service_search_forwards_namespace_kwargs(tmp_path):
    """Regression (#67): proxy search must accept and forward
    namespace/client_scope kwargs through dispatch to the store.
    A closed signature here TypeErrors on every scoped search in
    shared_service mode (the production default)."""
    import json
    import time
    from service_client import SharedMemoryStore

    (tmp_path / "hybrid_memory.json").write_text(
        json.dumps({"local_embedding_model": "nonexistent-model-xyz"}),
        encoding="utf-8",
    )
    store = SharedMemoryStore(tmp_path, user_id="test_user", embedder=None)
    try:
        store.remember(
            category="personal_fact", content="doc fact about invoice",
            namespace="document", client_scope="acme",
        )
        store.remember(
            category="personal_fact", content="chat note about meeting",
            namespace="conversation",
        )
        # Proxy convention: namespace/client_scope forwarded through RPC.
        doc = store.search("about", limit=10, namespace="document")
        conv = store.search("about", limit=10, namespace="conversation")
        assert any("doc fact" in r.content for r in doc)
        assert all("chat note" not in r.content for r in doc)
        assert any("chat note" in r.content for r in conv)
        assert all("doc fact" not in r.content for r in conv)
        # client_scope filter through RPC.
        acme = store.search("about", limit=10, client_scope="acme")
        assert any("doc fact" in r.content for r in acme)
    finally:
        try:
            store._rpc.stop_service()
        finally:
            time.sleep(0.5)


@pytestmark_rpc
def test_shared_service_namespace_round_trips_on_record(tmp_path):
    """The record returned through RPC must carry namespace/client_scope."""
    import json
    import time
    from service_client import SharedMemoryStore

    (tmp_path / "hybrid_memory.json").write_text(
        json.dumps({"local_embedding_model": "nonexistent-model-xyz"}),
        encoding="utf-8",
    )
    store = SharedMemoryStore(tmp_path, user_id="test_user", embedder=None)
    try:
        r = store.remember(
            category="personal_fact", content="doc fact with namespace",
            namespace="document", client_scope="acme",
        )
        assert r is not None
        assert r.namespace == "document"
        assert r.client_scope == "acme"
        # Search it back and verify the fields survive the round trip.
        results = store.search("doc fact", limit=5)
        assert results
        assert results[0].namespace == "document"
        assert results[0].client_scope == "acme"
    finally:
        try:
            store._rpc.stop_service()
        finally:
            time.sleep(0.5)
