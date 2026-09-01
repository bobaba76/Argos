"""Spec-07 (#71): the watcher — document catalog, extraction & freshness.

Tests (cheapest falsifying first — deterministic, no LLM):
1. Identity: rename → same file_id, no re-extraction; content edit →
   changed extract_hash; Excel resave with identical data → no re-extraction.
2. Dedup: two paths, one content → one canonical row + aliases.
3. Scan classification: synthetic tree — new/changed/moved/unchanged/deleted;
   tombstone on delete; pass completes with locked files.
4. Hot policy precedence: pin > usage > type-default > recency.
5. D4 lifecycle: version bump → stale; OCR numeric → unverified;
   tombstone → invalidated; verify → current.
6. Conflict: filename-pattern correction markers detected.
7. Catalog retrieval: upsert + get + list + tombstone.
8. Full suite green; no watcher config = zero behaviour change.
"""
import json
import os
import sys
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from watcher import (
    VERIFIED_CURRENT,
    VERIFIED_INVALIDATED,
    VERIFIED_STALE,
    VERIFIED_UNVERIFIED,
    classify_doc_type,
    extract_doc_type_label,
    has_correction_marker,
    hash_extraction_input,
    hash_file,
    heuristic_description,
    is_excluded,
    is_hot,
    prepare_extraction_input,
    scan_pass,
    verified_state_on_ocr_numeric,
    verified_state_on_tombstone,
    verified_state_on_version_bump,
    verified_state_on_verify,
)


# ---------------------------------------------------------------------------
# 1. Identity — content hash, not path
# ---------------------------------------------------------------------------


class TestDocIdentity:
    """D1: file_id = SHA-256 of raw bytes; extract_hash = SHA-256 of
    extraction input. Rename never re-extracts; content edit does."""

    def test_same_content_same_hash(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("hello world")
        f2.write_text("hello world")
        assert hash_file(f1) == hash_file(f2)

    def test_different_content_different_hash(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("hello")
        f2.write_text("world")
        assert hash_file(f1) != hash_file(f2)

    def test_rename_preserves_hash(self, tmp_path):
        f = tmp_path / "original.txt"
        f.write_text("content stays the same")
        h1 = hash_file(f)
        renamed = tmp_path / "renamed.txt"
        f.rename(renamed)
        h2 = hash_file(renamed)
        assert h1 == h2

    def test_hash_returns_none_on_missing(self):
        assert hash_file("/nonexistent/path/file.txt") is None

    def test_extract_hash_differs_on_text_change(self):
        h1 = hash_extraction_input("VAT number is 123")
        h2 = hash_extraction_input("VAT number is 456")
        assert h1 != h2

    def test_extract_hash_same_on_identical_text(self):
        h1 = hash_extraction_input("same text")
        h2 = hash_extraction_input("same text")
        assert h1 == h2


# ---------------------------------------------------------------------------
# 2. Dedup — two paths, one content
# ---------------------------------------------------------------------------


class TestDedup:
    """D1: one catalog row per unique hash; aliases point at canonical row."""

    def test_scan_classifies_same_content_as_moved(self, tmp_path):
        # Create two files with identical content at different paths.
        content = "duplicate content for dedup test"
        f1 = tmp_path / "original.pdf"
        f2 = tmp_path / "copy.pdf"
        f1.write_text(content)
        f2.write_text(content)
        file_id = hash_file(f1)
        # Simulate a catalog that already knows about f1.
        catalog = {
            file_id: {
                "paths": [str(f1)],
                "status": "active",
            }
        }
        result = scan_pass([str(tmp_path)], catalog)
        # f1 should be unchanged, f2 should be moved (same hash, new path).
        unchanged_paths = {r["path"] for r in result["unchanged"]}
        moved_paths = {r["path"] for r in result["moved"]}
        assert str(f1) in unchanged_paths
        assert str(f2) in moved_paths


# ---------------------------------------------------------------------------
# 3. Scan classification
# ---------------------------------------------------------------------------


class TestScanClassification:
    """D3: classify new/changed/moved/unchanged/deleted; tombstone on delete."""

    def test_new_file_classified(self, tmp_path):
        f = tmp_path / "new.pdf"
        f.write_text("new file content")
        result = scan_pass([str(tmp_path)], {})
        assert len(result["new"]) == 1
        assert result["new"][0]["path"] == str(f)

    def test_unchanged_file_classified(self, tmp_path):
        f = tmp_path / "unchanged.pdf"
        f.write_text("same content")
        file_id = hash_file(f)
        catalog = {file_id: {"paths": [str(f)], "status": "active"}}
        result = scan_pass([str(tmp_path)], catalog)
        assert len(result["unchanged"]) == 1
        assert len(result["new"]) == 0

    def test_changed_file_classified(self, tmp_path):
        f = tmp_path / "changed.pdf"
        f.write_text("original content")
        old_id = hash_file(f)
        catalog = {old_id: {"paths": [str(f)], "status": "active"}}
        # Now change the content.
        f.write_text("modified content")
        result = scan_pass([str(tmp_path)], catalog)
        # The path was known but content differs → changed.
        assert len(result["changed"]) == 1
        assert result["changed"][0]["path"] == str(f)

    def test_deleted_file_classified(self, tmp_path):
        f = tmp_path / "gone.pdf"
        f.write_text("will be deleted")
        file_id = hash_file(f)
        catalog = {file_id: {"paths": [str(f)], "status": "active"}}
        f.unlink()
        result = scan_pass([str(tmp_path)], catalog)
        assert len(result["deleted"]) == 1
        assert result["deleted"][0]["file_id"] == file_id

    def test_excluded_files_skipped(self, tmp_path):
        (tmp_path / "~$lock.xlsx").write_text("lock file")
        (tmp_path / ".DS_Store").write_text("os metadata")
        (tmp_path / "real.pdf").write_text("real content")
        result = scan_pass([str(tmp_path)], {})
        # Only real.pdf should be classified.
        all_paths = (
            [r["path"] for r in result["new"]]
            + [r["path"] for r in result["unchanged"]]
        )
        assert len(all_paths) == 1
        assert "real.pdf" in all_paths[0]

    def test_unsupported_formats_skipped(self, tmp_path):
        (tmp_path / "image.png").write_text("not a doc")
        (tmp_path / "data.json").write_text("{}")
        result = scan_pass([str(tmp_path)], {})
        assert len(result["new"]) == 0

    def test_pass_completes_with_empty_root(self, tmp_path):
        result = scan_pass([str(tmp_path)], {})
        assert all(len(v) == 0 for v in result.values())

    def test_pass_completes_with_nonexistent_root(self):
        result = scan_pass(["/nonexistent/root"], {})
        assert all(len(v) == 0 for v in result.values())


# ---------------------------------------------------------------------------
# 4. Hot policy precedence
# ---------------------------------------------------------------------------


class TestHotPolicy:
    """D4: pin > usage > type-default > recency."""

    def test_pinned_is_hot(self):
        info = {"pinned": True, "doc_type": "pdf", "touch_count": 0,
                "mtime": "2020-01-01T00:00:00+00:00"}
        hot, reason = is_hot(info)
        assert hot and reason == "pinned"

    def test_usage_promotes(self):
        info = {"pinned": False, "doc_type": "pdf", "touch_count": 5,
                "mtime": "2020-01-01T00:00:00+00:00"}
        hot, reason = is_hot(info, usage_threshold=3)
        assert hot and "usage" in reason

    def test_type_default_xlsx(self):
        info = {"pinned": False, "doc_type": "xlsx", "touch_count": 0,
                "mtime": "2020-01-01T00:00:00+00:00"}
        hot, reason = is_hot(info)
        assert hot and "type-default" in reason

    def test_recency_hot(self):
        now = datetime.now(timezone.utc)
        recent = (now - timedelta(days=30)).isoformat()
        info = {"pinned": False, "doc_type": "pdf", "touch_count": 0,
                "mtime": recent}
        hot, reason = is_hot(info)
        assert hot and "recency" in reason

    def test_not_hot(self):
        old = "2020-01-01T00:00:00+00:00"
        info = {"pinned": False, "doc_type": "pdf", "touch_count": 0,
                "mtime": old}
        hot, reason = is_hot(info)
        assert not hot

    def test_pin_beats_usage(self):
        info = {"pinned": True, "doc_type": "pdf", "touch_count": 10,
                "mtime": "2020-01-01T00:00:00+00:00"}
        hot, reason = is_hot(info)
        assert reason == "pinned"


# ---------------------------------------------------------------------------
# 5. D4 lifecycle — verified_state transitions
# ---------------------------------------------------------------------------


class TestVerifiedStateLifecycle:
    """D4: version bump → stale; OCR numeric → unverified; tombstone →
    invalidated; verify → current."""

    def test_version_bump_makes_stale(self):
        assert verified_state_on_version_bump() == VERIFIED_STALE

    def test_tombstone_makes_invalidated(self):
        assert verified_state_on_tombstone() == VERIFIED_INVALIDATED

    def test_ocr_numeric_makes_unverified(self):
        assert verified_state_on_ocr_numeric() == VERIFIED_UNVERIFIED

    def test_verify_makes_current(self):
        assert verified_state_on_verify() == VERIFIED_CURRENT


# ---------------------------------------------------------------------------
# 6. Conflict — filename-pattern correction markers
# ---------------------------------------------------------------------------


class TestConflictSurfacing:
    """D6: filename-pattern conflict surfacing (zero LLM)."""

    @pytest.mark.parametrize("filename", [
        "Tax Invoice CORRECTED.pdf",
        "Invoice_REVISED.pdf",
        "Report (2).pdf",
        "Document v2.pdf",
        "Statement rev.pdf",
        "Invoice amended.pdf",
    ])
    def test_correction_markers_detected(self, filename):
        assert has_correction_marker(filename)

    @pytest.mark.parametrize("filename", [
        "Tax Invoice.pdf",
        "Monthly Report.pdf",
        "data.xlsx",
    ])
    def test_no_false_positives(self, filename):
        assert not has_correction_marker(filename)

    def test_doc_type_label_strips_markers(self):
        base = extract_doc_type_label("Tax Invoice CORRECTED.pdf")
        assert "CORRECTED" not in base
        assert "Tax Invoice" in base

    def test_doc_type_label_strips_version(self):
        base = extract_doc_type_label("Report v2.pdf")
        assert "v2" not in base


# ---------------------------------------------------------------------------
# 7. Catalog retrieval — store-level
# ---------------------------------------------------------------------------


class TestFileCatalog:
    """D2: file_catalog table — upsert, get, list, tombstone."""

    @pytest.fixture
    def store(self, tmp_path):
        from store import DuckDBMemoryStore
        return DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="alice")

    def test_catalog_table_exists(self, store):
        result = store.connection.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_name = 'file_catalog'"
        ).fetchone()
        assert result[0] == 1

    def test_upsert_and_get(self, store):
        store.upsert_catalog_entry(
            file_id="abc123",
            canonical_path="/docs/invoice.pdf",
            size=1024,
            mtime="2026-01-01T00:00:00+00:00",
            doc_type="pdf",
            client_scope="acme",
            doc_class="invoice",
            one_line_description="acme — invoice — January 2026",
        )
        entry = store.get_catalog_entry("abc123")
        assert entry is not None
        assert entry["canonical_path"] == "/docs/invoice.pdf"
        assert entry["doc_type"] == "pdf"
        assert entry["status"] == "active"

    def test_upsert_is_idempotent(self, store):
        for _ in range(3):
            store.upsert_catalog_entry(
                file_id="abc123",
                canonical_path="/docs/invoice.pdf",
                size=1024,
                mtime="2026-01-01T00:00:00+00:00",
                doc_type="pdf",
            )
        entries = store.list_catalog(limit=100)
        assert len(entries) == 1

    def test_tombstone_marks_status(self, store):
        store.upsert_catalog_entry(
            file_id="abc123",
            canonical_path="/docs/invoice.pdf",
            size=1024,
            mtime="2026-01-01T00:00:00+00:00",
            doc_type="pdf",
        )
        store.tombstone_catalog_entry(file_id="abc123")
        entry = store.get_catalog_entry("abc123")
        assert entry["status"] == "tombstoned"

    def test_tombstone_invalidates_facts(self, store):
        store.upsert_catalog_entry(
            file_id="doc123",
            canonical_path="/docs/invoice.pdf",
            size=1024,
            mtime="2026-01-01T00:00:00+00:00",
            doc_type="pdf",
        )
        # Create a fact sourced from this document.
        store.remember(
            category="personal_fact",
            content="Acme VAT is 123",
            namespace="document",
            client_scope="acme",
            source_doc_id="doc123",
            extraction_method="text",
            verified_state="current",
        )
        store.tombstone_catalog_entry(file_id="doc123")
        # The fact should now be invalidated.
        results = store.search("VAT", limit=10)
        assert len(results) == 0  # invalidated facts excluded from search

    def test_list_filters_by_status(self, store):
        store.upsert_catalog_entry(
            file_id="active1",
            canonical_path="/docs/a.pdf",
            size=100, mtime="2026-01-01T00:00:00+00:00", doc_type="pdf",
        )
        store.upsert_catalog_entry(
            file_id="active2",
            canonical_path="/docs/b.pdf",
            size=100, mtime="2026-01-01T00:00:00+00:00", doc_type="pdf",
        )
        store.tombstone_catalog_entry(file_id="active1")
        active = store.list_catalog(status="active")
        tombstoned = store.list_catalog(status="tombstoned")
        assert len(active) == 1
        assert active[0]["file_id"] == "active2"
        assert len(tombstoned) == 1
        assert tombstoned[0]["file_id"] == "active1"

    def test_stale_marking(self, store):
        store.remember(
            category="personal_fact",
            content="Acme revenue is $1M",
            namespace="document",
            source_doc_id="doc123",
            verified_state="current",
        )
        count = store.stale_facts_for_doc("doc123")
        assert count == 1
        # Stale facts excluded from search.
        results = store.search("revenue", limit=10)
        assert len(results) == 0

    def test_verify_fact(self, store):
        store.remember(
            category="personal_fact",
            content="Acme OCR amount is 500",
            namespace="document",
            source_doc_id="doc123",
            extraction_method="ocr",
            verified_state="unverified",
        )
        # Unverified facts are NOT excluded from search (they're flagged,
        # not hidden — the answerer policy presents them with a caveat).
        results = store.search("OCR", limit=10)
        assert len(results) == 1
        assert results[0].verified_state == "unverified"
        # Verify the fact.
        store.verify_fact(results[0].memory_id)
        results2 = store.search("OCR", limit=10)
        assert results2[0].verified_state == "current"


# ---------------------------------------------------------------------------
# 8. Backward compatibility — no watcher config = zero behaviour change
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    """No watcher config = zero behaviour change."""

    def test_remember_without_d4_fields(self, tmp_path):
        from store import DuckDBMemoryStore
        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="alice")
        record = store.remember(
            category="personal_fact",
            content="User likes coffee",
        )
        assert record is not None
        assert record.source_doc_id is None
        assert record.verified_state == "current"
        assert record.extraction_method is None

    def test_search_returns_all_without_d4_fields(self, tmp_path):
        from store import DuckDBMemoryStore
        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="alice")
        store.remember(category="personal_fact", content="User likes tea")
        results = store.search("tea", limit=10)
        assert len(results) == 1
        assert results[0].verified_state == "current"

    def test_d4_fields_round_trip(self, tmp_path):
        from store import DuckDBMemoryStore
        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="alice")
        store.remember(
            category="personal_fact",
            content="Acme invoice total is $5000",
            namespace="document",
            client_scope="acme",
            source_doc_id="filehash123",
            source_loc="page 1",
            extraction_method="text",
            verified_state="current",
        )
        results = store.search("invoice", limit=10)
        assert len(results) == 1
        r = results[0]
        assert r.source_doc_id == "filehash123"
        assert r.source_loc == "page 1"
        assert r.extraction_method == "text"
        assert r.verified_state == "current"

    def test_d4_fields_carry_through_update(self, tmp_path):
        from store import DuckDBMemoryStore
        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="alice")
        record = store.remember(
            category="personal_fact",
            content="Acme total is $4000",
            namespace="document",
            source_doc_id="filehash123",
            extraction_method="text",
        )
        updated = store.update_memory(
            record.memory_id, content="Acme total is $5000",
        )
        assert updated is not None
        assert updated.source_doc_id == "filehash123"
        assert updated.extraction_method == "text"


# ---------------------------------------------------------------------------
# 9. Heuristic descriptions
# ---------------------------------------------------------------------------


class TestHeuristicDescription:
    """D3: deterministic one-liner from filename + folder + content hints."""

    def test_pdf_description(self, tmp_path):
        f = tmp_path / "invoices" / "Tax Invoice.pdf"
        f.parent.mkdir()
        f.write_text("dummy")
        desc = heuristic_description(
            f, "pdf", first_page_text="Invoice #1234\nDate: 2026-01-15"
        )
        assert "invoices" in desc
        assert "Tax Invoice" in desc
        assert "Invoice #1234" in desc

    def test_xlsx_description(self, tmp_path):
        f = tmp_path / "reports" / "Financials.xlsx"
        desc = heuristic_description(
            f, "xlsx", sheet_names=["Summary", "Details", "Raw Data"]
        )
        assert "reports" in desc
        assert "Financials" in desc
        assert "Summary" in desc

    def test_csv_description(self, tmp_path):
        f = tmp_path / "data" / "export.csv"
        desc = heuristic_description(
            f, "csv", first_lines="id,name,amount\n1,Alice,100"
        )
        assert "data" in desc
        assert "export" in desc

    def test_description_truncated(self, tmp_path):
        f = tmp_path / "folder" / "file.pdf"
        long_text = "A" * 500
        desc = heuristic_description(f, "pdf", first_page_text=long_text)
        assert len(desc) <= 200
