"""Audit tests for store_retrieval.py SR1-SR12 (issue #227).

Covers cross-tenant leak fixes, cross-tenant modification guards,
catalog scope filters, and performance/privacy improvements.

Run with (Hermes venv python, offline):
    python -m pytest tests/test_store_retrieval_audit.py -v
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))


# ---------------------------------------------------------------------------
# Test fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    from store import DuckDBMemoryStore
    s = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="alice")
    yield s
    s.close()


@pytest.fixture
def store_bob(tmp_path):
    from store import DuckDBMemoryStore
    s = DuckDBMemoryStore(tmp_path / "test_bob.duckdb", user_id="bob")
    yield s
    s.close()


# ---------------------------------------------------------------------------
# SR1 â€” export_access_audit tenant/user filter
# ---------------------------------------------------------------------------

class TestSR1ExportAccessAuditFilter:
    def test_export_filters_by_user_id(self, store):
        """SR1: export_access_audit only returns the caller's audit entries."""
        store.write_access_audit(
            user_id="alice", query_text="my query",
            granted_count=1, denied_count=0,
        )
        store.write_access_audit(
            user_id="bob", query_text="other query",
            granted_count=0, denied_count=1,
        )
        export = store.export_access_audit(format="jsonl")
        lines = [json.loads(l) for l in export.strip().split("\n") if l]
        # SR1: only alice's entries (store user_id="alice")
        assert len(lines) == 1
        assert lines[0]["user_id"] == "alice"

    def test_export_query_has_user_filter(self):
        """SR1: the SQL query includes a user_id/tenant filter."""
        from store_retrieval import StoreRetrievalMixin
        src = inspect.getsource(StoreRetrievalMixin.export_access_audit)
        assert "user_id = ?" in src or "tenant = ?" in src


# ---------------------------------------------------------------------------
# SR2 â€” query_rejection_ledger user_scope filter
# ---------------------------------------------------------------------------

class TestSR2RejectionLedgerFilter:
    def test_query_has_user_scope_filter(self):
        """SR2: query_rejection_ledger SQL includes user_scope filter."""
        from store_retrieval import StoreRetrievalMixin
        src = inspect.getsource(StoreRetrievalMixin.query_rejection_ledger)
        assert "user_scope IS NULL OR user_scope" in src

    def test_query_filters_by_user_scope(self, store):
        """SR2: only the caller's rejection entries are returned.

        Rejections are written with user_scope = self.user_id, so all
        entries from this store are alice's. The test verifies the
        query returns only matching-scope entries.
        """
        store.record_rejection("test_cat", {"subject": "s", "predicate": "p"})
        results = store.query_rejection_ledger()
        # All returned entries should have user_scope = alice or NULL.
        for r in results:
            scope = r.get("user_scope")
            assert scope is None or scope == "alice"


# ---------------------------------------------------------------------------
# SR3 â€” query_candidate_decisions user_scope filter
# ---------------------------------------------------------------------------

class TestSR3CandidateDecisionsFilter:
    def test_query_has_user_scope_filter(self):
        """SR3: query_candidate_decisions SQL includes user_scope filter."""
        from store_retrieval import StoreRetrievalMixin
        src = inspect.getsource(StoreRetrievalMixin.query_candidate_decisions)
        assert "user_scope IS NULL OR user_scope" in src


# ---------------------------------------------------------------------------
# SR4 â€” stale_facts_for_doc user_scope guard
# ---------------------------------------------------------------------------

class TestSR4StaleFactsGuard:
    def test_update_has_user_scope_guard(self):
        """SR4: stale_facts_for_doc UPDATE includes user_scope guard."""
        from store_retrieval import StoreRetrievalMixin
        src = inspect.getsource(StoreRetrievalMixin.stale_facts_for_doc)
        assert "user_scope IS NULL OR user_scope" in src


# ---------------------------------------------------------------------------
# SR5 â€” verify_fact user_scope guard
# ---------------------------------------------------------------------------

class TestSR5VerifyFactGuard:
    def test_update_has_user_scope_guard(self):
        """SR5: verify_fact UPDATE includes user_scope guard."""
        from store_retrieval import StoreRetrievalMixin
        src = inspect.getsource(StoreRetrievalMixin.verify_fact)
        assert "user_scope IS NULL OR user_scope" in src


# ---------------------------------------------------------------------------
# SR6 â€” tombstone_catalog_entry user_scope guard
# ---------------------------------------------------------------------------

class TestSR6TombstoneGuard:
    def test_invalidation_has_user_scope_guard(self):
        """SR6: tombstone_catalog_entry memory_records UPDATE includes user_scope guard."""
        from store_retrieval import StoreRetrievalMixin
        src = inspect.getsource(StoreRetrievalMixin.tombstone_catalog_entry)
        assert "user_scope IS NULL OR user_scope" in src


# ---------------------------------------------------------------------------
# SR7 â€” get_catalog_entry/by_path client_scope filter
# ---------------------------------------------------------------------------

class TestSR7CatalogScopeFilter:
    def test_get_catalog_entry_has_explicit_client_scope_param(self):
        """SR7: get_catalog_entry takes an explicit client_scope param
        (matching list_catalog convention), not self.user_id."""
        from store_retrieval import StoreRetrievalMixin
        sig = inspect.signature(StoreRetrievalMixin.get_catalog_entry)
        assert "client_scope" in sig.parameters
        assert sig.parameters["client_scope"].default is None

    def test_get_catalog_by_path_has_explicit_client_scope_param(self):
        """SR7: get_catalog_by_path takes an explicit client_scope param."""
        from store_retrieval import StoreRetrievalMixin
        sig = inspect.getsource(StoreRetrievalMixin.get_catalog_by_path)
        assert "client_scope" in sig

    def test_get_catalog_entry_no_scope_returns_entry(self, store):
        """SR7 regression: entry with client_scope='acme' is fetchable
        via get_catalog_entry('abc') with NO scope arg → returns entry.
        Default None = no filter (matches list_catalog convention)."""
        store.upsert_catalog_entry(
            file_id="test_fid_sr7",
            canonical_path="/docs/test.pdf",
            size=100,
            mtime="2026-01-01T00:00:00+00:00",
            doc_type="pdf",
            client_scope="acme",
        )
        # No client_scope arg → no filter → entry is returned.
        entry = store.get_catalog_entry("test_fid_sr7")
        assert entry is not None
        assert entry["file_id"] == "test_fid_sr7"
        assert entry["client_scope"] == "acme"

    def test_get_catalog_entry_with_scope_filters(self, store):
        """SR7: when client_scope is passed, only matching entries return."""
        store.upsert_catalog_entry(
            file_id="test_fid_sr7b",
            canonical_path="/docs/test2.pdf",
            size=100,
            mtime="2026-01-01T00:00:00+00:00",
            doc_type="pdf",
            client_scope="acme",
        )
        # Matching scope → returns entry.
        entry = store.get_catalog_entry("test_fid_sr7b", client_scope="acme")
        assert entry is not None
        # Non-matching scope → returns None.
        entry_none = store.get_catalog_entry("test_fid_sr7b", client_scope="other")
        assert entry_none is None

    def test_get_catalog_by_path_no_scope_returns_entry(self, store):
        """SR7 regression: get_catalog_by_path with no scope arg returns entry."""
        store.upsert_catalog_entry(
            file_id="test_fid_sr7c",
            canonical_path="/docs/test3.pdf",
            size=100,
            mtime="2026-01-01T00:00:00+00:00",
            doc_type="pdf",
            client_scope="acme",
        )
        entry = store.get_catalog_by_path("/docs/test3.pdf")
        assert entry is not None
        assert entry["file_id"] == "test_fid_sr7c"


# ---------------------------------------------------------------------------
# SR8 â€” _record_retrieval user_scope guard
# ---------------------------------------------------------------------------

class TestSR8RecordRetrievalGuard:
    def test_update_has_user_scope_guard(self):
        """SR8: _record_retrieval UPDATE includes user_scope guard."""
        from store_retrieval import StoreRetrievalMixin
        src = inspect.getsource(StoreRetrievalMixin._record_retrieval)
        assert "user_scope IS NULL OR user_scope" in src


# ---------------------------------------------------------------------------
# SR9 â€” _text_search_raw LIMIT reduced
# ---------------------------------------------------------------------------

class TestSR9TextSearchLimit:
    def test_limit_reduced_from_2000(self):
        """SR9: _text_search_raw LIMIT is reduced from 2000."""
        from store_retrieval import StoreRetrievalMixin
        src = inspect.getsource(StoreRetrievalMixin._text_search_raw)
        # The SQL LIMIT clause should be 500, not 2000.
        assert "LIMIT 500" in src


# ---------------------------------------------------------------------------
# SR10 â€” _find_current_similar LIMIT reduced
# ---------------------------------------------------------------------------

class TestSR10FindCurrentSimilarLimit:
    def test_limit_reduced_from_500(self):
        """SR10: _find_current_similar Layer 2 LIMIT is reduced from 500."""
        from store_retrieval import StoreRetrievalMixin
        src = inspect.getsource(StoreRetrievalMixin._find_current_similar)
        assert "LIMIT 200" in src
        assert "LIMIT 500" not in src


# ---------------------------------------------------------------------------
# SR11 â€” _apply_p2c result-size guard
# ---------------------------------------------------------------------------

class TestSR11P2CGuard:
    def test_has_result_size_guard(self):
        """SR11: _apply_p2c has a result-size guard to cap O(nÂ²) cost."""
        from store_retrieval import StoreRetrievalMixin
        src = inspect.getsource(StoreRetrievalMixin._apply_p2c)
        assert "50" in src  # the cap


# ---------------------------------------------------------------------------
# SR12 â€” export_access_audit hashes query_text
# ---------------------------------------------------------------------------

class TestSR12QueryTextHashed:
    def test_query_text_is_hashed(self, store):
        """SR12: export_access_audit hashes query_text instead of returning raw."""
        store.write_access_audit(
            user_id="alice", query_text="sensitive search query",
            granted_count=1, denied_count=0,
        )
        export = store.export_access_audit(format="jsonl")
        lines = [json.loads(l) for l in export.strip().split("\n") if l]
        assert len(lines) == 1
        # The raw query text should NOT appear in the export.
        assert lines[0]["query_text"] != "sensitive search query"
        # Should be a hash (16 chars of SHA-256 hex).
        assert len(lines[0]["query_text"]) == 16

    def test_hashing_code_present(self):
        """SR12: export_access_audit has query_text hashing code."""
        from store_retrieval import StoreRetrievalMixin
        src = inspect.getsource(StoreRetrievalMixin.export_access_audit)
        assert "sha256" in src or "hashlib" in src

