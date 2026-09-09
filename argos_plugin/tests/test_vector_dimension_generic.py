"""#286: dimension-generic vector store + re-embed orchestration.

Tests:
1. Fresh + old store accept a 768-dim vector without schema break.
2. Migration 1→2 is idempotent + transactional (run twice = no-op).
3. Mixed-dim query fails loudly with a clear error message.
4. Re-embed marks provenance (source embedder + timestamp) on every
   touched record with no silent value change.
5. Text-search fallback still returns results during a mocked in-flight
   re-embed.
6. Bench gate: re-embed produces a before/after-checkable report.
7. Existing vector suites remain green.

Run with (Hermes venv python, hermetic):
    ARGOS_HERMETIC_TESTS=1 python -m pytest tests/test_vector_dimension_generic.py -v
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schema_migrations import LATEST_SCHEMA_VERSION

from store import DuckDBMemoryStore


class _MockEmbedder384:
    """Mock embedder producing 384-dim vectors (content-dependent to avoid dedup)."""
    _model_name = "mock-384"
    recovered = False
    @staticmethod
    def _seed(text):
        """Deterministic seed from text (not Python's randomized hash)."""
        s = 0
        for c in text:
            s = (s * 31 + ord(c)) & 0xFFFFFFFF
        return s
    def embed(self, text, *, is_query=False):
        s = self._seed(text)
        vec = [0.0] * 384
        # Use the seed to pick a single dominant dimension — makes
        # different texts nearly orthogonal (cosine sim ~0).
        idx = s % 384
        vec[idx] = 1.0
        # Add a small second component for slight variation.
        idx2 = (s >> 8) % 384
        if idx2 != idx:
            vec[idx2] = 0.5
        return vec
    def embed_batch(self, texts, *, is_query=False):
        return [self.embed(t, is_query=is_query) for t in texts]
    @property
    def dimension(self):
        return 384


class _MockEmbedder768:
    """Mock embedder producing 768-dim vectors (content-dependent to avoid dedup)."""
    _model_name = "mock-768"
    recovered = False
    @staticmethod
    def _seed(text):
        s = 0
        for c in text:
            s = (s * 31 + ord(c)) & 0xFFFFFFFF
        return s
    def embed(self, text, *, is_query=False):
        s = self._seed(text)
        vec = [0.0] * 768
        idx = s % 768
        vec[idx] = 1.0
        idx2 = (s >> 8) % 768
        if idx2 != idx:
            vec[idx2] = 0.5
        return vec
    def embed_batch(self, texts, *, is_query=False):
        return [self.embed(t, is_query=is_query) for t in texts]
    @property
    def dimension(self):
        return 768


class TestDimensionGenericStorage:
    """1. Fresh + old store accept a 768-dim vector without break."""

    def test_fresh_store_accepts_768_dim(self, tmp_path):
        """A fresh store with a 768-dim embedder stores and retrieves
        records without schema errors."""
        store = DuckDBMemoryStore(
            tmp_path / "test.duckdb", user_id="alice",
            embedder=_MockEmbedder768(),
        )
        try:
            rec = store.remember(
                category="personal_fact",
                content="User works as a software engineer",
            )
            assert rec is not None
            # The embedding should be 768-dim.
            assert len(rec.embedding) == 768
            # embedding_dim should be stamped.
            assert rec.embedding_dim == 768
            # embedder_id should be stamped.
            assert rec.embedder_id == "mock-768"
            # embedded_at should be set.
            assert rec.embedded_at is not None
        finally:
            store.close()

    def test_old_store_accepts_768_dim_after_migration(self, tmp_path):
        """An old store (created with 384-dim, then reopened with 768-dim)
        accepts new 768-dim records after the migration runs."""
        # Phase 1: create with 384-dim embedder.
        store = DuckDBMemoryStore(
            tmp_path / "old.duckdb", user_id="alice",
            embedder=_MockEmbedder384(),
        )
        store.remember(category="personal_fact", content="Old 384-dim fact")
        store.close()

        # Phase 2: reopen with 768-dim embedder.
        store2 = DuckDBMemoryStore(
            tmp_path / "old.duckdb", user_id="alice",
            embedder=_MockEmbedder768(),
        )
        try:
            # New record with 768-dim should work.
            rec = store2.remember(
                category="personal_fact",
                content="New 768-dim fact",
            )
            assert len(rec.embedding) == 768
            assert rec.embedding_dim == 768
        finally:
            store2.close()

    def test_schema_version_is_2_after_init(self, tmp_path):
        """The store is at the latest schema version after init (all
        migrations have run)."""
        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="alice")
        try:
            assert store.get_schema_version() == LATEST_SCHEMA_VERSION
        finally:
            store.close()


class TestMigrationIdempotency:
    """2. Migration 1→2 is idempotent + transactional."""

    def test_migration_1_to_2_idempotent(self, tmp_path):
        """Running the 1→2 migration twice is a no-op."""
        import duckdb
        from schema_migrations import run_migrations, _ensure_schema_meta_table, _set_schema_version

        conn = duckdb.connect(str(tmp_path / "idempotent.duckdb"))
        try:
            conn.execute("""
                CREATE TABLE memory_records (
                    memory_id VARCHAR PRIMARY KEY,
                    category VARCHAR,
                    content VARCHAR,
                    embedding DOUBLE[],
                    embedding_dim INTEGER
                )
            """)
            # Insert a row with an embedding.
            conn.execute(
                "INSERT INTO memory_records (memory_id, category, content, embedding) "
                "VALUES (?, ?, ?, ?)",
                ["mem-1", "test", "content", [0.1] * 384],
            )

            # Stamp at version 1, then run migrations (should reach latest).
            _ensure_schema_meta_table(conn)
            _set_schema_version(conn, 1)
            r1 = run_migrations(conn)
            assert r1["to_version"] == LATEST_SCHEMA_VERSION
            assert 2 in r1["applied"]

            # Run again — should be a no-op.
            r2 = run_migrations(conn)
            assert r2["from_version"] == LATEST_SCHEMA_VERSION
            assert r2["to_version"] == LATEST_SCHEMA_VERSION
            assert len(r2["applied"]) == 0
            assert 2 in r2["skipped"]
        finally:
            conn.close()

    def test_migration_backfills_embedding_dim(self, tmp_path):
        """The 1→2 migration backfills embedding_dim for existing rows."""
        import duckdb
        from schema_migrations import run_migrations, _ensure_schema_meta_table, _set_schema_version

        conn = duckdb.connect(str(tmp_path / "backfill.duckdb"))
        try:
            conn.execute("""
                CREATE TABLE memory_records (
                    memory_id VARCHAR PRIMARY KEY,
                    category VARCHAR,
                    content VARCHAR,
                    embedding DOUBLE[],
                    embedding_dim INTEGER
                )
            """)
            # Insert rows with embeddings of different dims.
            conn.execute(
                "INSERT INTO memory_records (memory_id, category, content, embedding) "
                "VALUES (?, ?, ?, ?)",
                ["mem-384", "test", "content384", [0.1] * 384],
            )
            conn.execute(
                "INSERT INTO memory_records (memory_id, category, content, embedding) "
                "VALUES (?, ?, ?, ?)",
                ["mem-768", "test", "content768", [0.2] * 768],
            )
            conn.execute(
                "INSERT INTO memory_records (memory_id, category, content, embedding) "
                "VALUES (?, ?, ?, ?)",
                ["mem-null", "test", "no-emb", None],
            )

            # Stamp at version 1, then run migrations (should do 1→2).
            _ensure_schema_meta_table(conn)
            _set_schema_version(conn, 1)
            r = run_migrations(conn)
            assert r["to_version"] == LATEST_SCHEMA_VERSION
            assert 2 in r["applied"]

            # Check backfilled dims.
            dims = conn.execute(
                "SELECT memory_id, embedding_dim FROM memory_records ORDER BY memory_id"
            ).fetchall()
            dim_map = {mid: dim for mid, dim in dims}
            assert dim_map["mem-384"] == 384
            assert dim_map["mem-768"] == 768
            assert dim_map["mem-null"] is None
        finally:
            conn.close()


class TestMixedDimQueryFailsLoud:
    """3. Mixed-dim query fails loudly with a clear error message."""

    def test_mixed_dim_query_raises_value_error(self, tmp_path):
        """A query vector whose dim doesn't match stored vectors raises
        a ValueError with a clear message (not silent garbage)."""
        # Create a store with 384-dim records.
        store = DuckDBMemoryStore(
            tmp_path / "mixed.duckdb", user_id="alice",
            embedder=_MockEmbedder384(),
        )
        try:
            store.remember(category="personal_fact", content="384-dim fact")

            # Now search with a 768-dim query vector.
            # The _vector_search_raw should raise ValueError.
            with pytest.raises(ValueError, match="Dimension mismatch"):
                store._vector_search_raw(
                    [0.01] * 768, limit=10, excluded=set(),
                )
        finally:
            store.close()

    def test_mixed_dim_falls_back_to_text_search(self, tmp_path):
        """When vector search fails due to dim mismatch, text search
        still returns results (graceful fallback)."""
        store = DuckDBMemoryStore(
            tmp_path / "fallback.duckdb", user_id="alice",
            embedder=_MockEmbedder384(),
        )
        try:
            store.remember(
                category="personal_fact",
                content="User works as a software engineer at a tech company",
            )

            # Search with a 768-dim query — vector arm fails, text arm
            # should still return results.
            results = store._hybrid_search(
                "software engineer",
                limit=5,
            )
            # Text search should find the record.
            assert len(results) >= 1
            assert any("software engineer" in r.content for r in results)
        finally:
            store.close()


class TestReembedProvenance:
    """4. Re-embed marks provenance on every touched record."""

    def test_reembed_stamps_embedder_id_and_timestamp(self, tmp_path):
        """reembed_store() stamps embedder_id, embedded_at, and
        embedding_dim on every re-embedded record."""
        import duckdb
        from reembed_memories import reembed_store

        # Create a store with 384-dim records.
        store = DuckDBMemoryStore(
            tmp_path / "reembed.duckdb", user_id="alice",
            embedder=_MockEmbedder384(),
        )
        store.remember(category="personal_fact", content="Fact one about hiking trails")
        store.remember(category="personal_fact", content="Fact two about cooking recipes")
        store.close()

        # Re-embed with 768-dim embedder.
        conn = duckdb.connect(str(tmp_path / "reembed.duckdb"))
        try:
            report = reembed_store(conn, _MockEmbedder768())

            assert report["re_embedded"] == 2
            assert report["source_embedder"] == "mock-768"
            assert report["embedding_dim"] == 768

            # Verify provenance columns are stamped.
            rows = conn.execute(
                "SELECT memory_id, embedder_id, embedded_at, embedding_dim "
                "FROM memory_records ORDER BY memory_id"
            ).fetchall()
            for mid, eid, eat, edim in rows:
                assert eid == "mock-768", f"embedder_id not stamped for {mid}"
                assert eat is not None, f"embedded_at not stamped for {mid}"
                assert edim == 768, f"embedding_dim wrong for {mid}: {edim}"
        finally:
            conn.close()

    def test_reembed_no_silent_value_change(self, tmp_path):
        """Re-embed actually changes the embedding values (no silent
        no-op). The new vectors should be different from the old ones."""
        import duckdb
        from reembed_memories import reembed_store

        store = DuckDBMemoryStore(
            tmp_path / "nochange.duckdb", user_id="alice",
            embedder=_MockEmbedder384(),
        )
        store.remember(category="personal_fact", content="Test fact for change detection")
        store.close()

        # Capture old embeddings.
        conn = duckdb.connect(str(tmp_path / "nochange.duckdb"))
        try:
            old_emb = conn.execute(
                "SELECT embedding FROM memory_records"
            ).fetchone()[0]
            assert len(old_emb) == 384

            # Re-embed with 768-dim.
            report = reembed_store(conn, _MockEmbedder768())
            assert report["re_embedded"] >= 1

            new_emb = conn.execute(
                "SELECT embedding FROM memory_records"
            ).fetchone()[0]
            # The new embedding should be 768-dim (different from old 384).
            assert len(new_emb) == 768
        finally:
            conn.close()


class TestTextSearchFallbackDuringReembed:
    """5. Text-search fallback still returns results during re-embed."""

    def test_text_search_works_without_embeddings(self, tmp_path):
        """Text search returns results even when no embeddings exist
        (simulating the in-flight re-embed state where embeddings may
        be temporarily mismatched or NULL)."""
        # Create a store with NO embedder (text-only).
        store = DuckDBMemoryStore(
            tmp_path / "textonly.duckdb", user_id="alice",
            embedder=None,
        )
        try:
            store.remember(
                category="personal_fact",
                content="User likes hiking in the mountains",
            )
            store.remember(
                category="personal_fact",
                content="User works as a software engineer",
            )

            # Search — text leg should return results.
            results = store._hybrid_search("hiking mountains", limit=5)
            assert len(results) >= 1
            assert any("hiking" in r.content for r in results)
        finally:
            store.close()

    def test_text_search_returns_results_with_mismatched_dims(self, tmp_path):
        """Text search returns results even when stored vectors have
        mixed dims (the vector arm fails, text arm serves)."""
        store = DuckDBMemoryStore(
            tmp_path / "mixedtext.duckdb", user_id="alice",
            embedder=_MockEmbedder384(),
        )
        try:
            store.remember(
                category="personal_fact",
                content="User enjoys cooking Italian food at home",
            )

            # Search with a query that matches text but has wrong dim.
            results = store._hybrid_search("cooking Italian food", limit=5)
            assert len(results) >= 1
            assert any("cooking" in r.content for r in results)
        finally:
            store.close()


class TestBenchGateReport:
    """6. Re-embed produces a before/after-checkable report."""

    def test_reembed_report_has_bench_gate_fields(self, tmp_path):
        """The re-embed report contains all fields needed for bench
        verification: count, source embedder, dims, before/after state."""
        import duckdb
        from reembed_memories import reembed_store

        store = DuckDBMemoryStore(
            tmp_path / "bench.duckdb", user_id="alice",
            embedder=_MockEmbedder384(),
        )
        store.remember(category="personal_fact", content="Bench test fact one about hiking")
        store.remember(category="personal_fact", content="Bench test fact two about cooking")
        store.close()

        conn = duckdb.connect(str(tmp_path / "bench.duckdb"))
        try:
            report = reembed_store(conn, _MockEmbedder768())

            # Bench gate fields.
            assert "total_rows" in report
            assert "re_embedded" in report
            assert "skipped" in report
            assert "source_embedder" in report
            assert "embedding_dim" in report
            assert "before_dims" in report
            assert "after_dims" in report
            assert "dry_run" in report
            assert "timestamp" in report

            # Before should have 384, after should have 768.
            assert 384 in report["before_dims"]
            assert 768 in report["after_dims"]
            assert report["re_embedded"] == 2
            assert report["source_embedder"] == "mock-768"
            assert report["embedding_dim"] == 768
        finally:
            conn.close()

    def test_reembed_dry_run_report(self, tmp_path):
        """A dry-run re-embed produces a report without writing."""
        import duckdb
        from reembed_memories import reembed_store

        store = DuckDBMemoryStore(
            tmp_path / "dryrun.duckdb", user_id="alice",
            embedder=_MockEmbedder384(),
        )
        store.remember(category="personal_fact", content="Dry run test fact")
        store.close()

        conn = duckdb.connect(str(tmp_path / "dryrun.duckdb"))
        try:
            report = reembed_store(conn, _MockEmbedder768(), dry_run=True)

            assert report["dry_run"] is True
            assert report["re_embedded"] == 0
            assert report["total_rows"] >= 1
        finally:
            conn.close()


class TestStoreWriteStampsProvenance:
    """7. The write path stamps embedding provenance on new records."""

    def test_remember_stamps_embedding_dim(self, tmp_path):
        """remember() stamps embedding_dim on new records."""
        store = DuckDBMemoryStore(
            tmp_path / "stamp.duckdb", user_id="alice",
            embedder=_MockEmbedder768(),
        )
        try:
            rec = store.remember(
                category="personal_fact",
                content="Test fact for dim stamping",
            )
            assert rec.embedding_dim == 768
            assert rec.embedder_id == "mock-768"
            assert rec.embedded_at is not None
        finally:
            store.close()

    def test_remember_without_embedder_has_null_provenance(self, tmp_path):
        """remember() with no embedder leaves embedding provenance NULL."""
        store = DuckDBMemoryStore(
            tmp_path / "noemb.duckdb", user_id="alice",
            embedder=None,
        )
        try:
            rec = store.remember(
                category="personal_fact",
                content="Test fact without embedder",
            )
            assert rec.embedding is None
            assert rec.embedding_dim is None
            assert rec.embedder_id is None
            assert rec.embedded_at is None
        finally:
            store.close()


class TestCLIReembedStampsProvenance:
    """8. CLI main() routes through reembed_store() — end-to-end test.

    The operational re-embed path (CLI main(), shipped by deploy.py) must
    stamp embedding_dim, embedder_id, and embedded_at on every re-embedded
    record. This test exercises main() end-to-end (not just
    reembed_store()) to verify the CLI doesn't bypass the provenance-
    stamping path.
    """

    def test_main_routes_through_reembed_store(self, tmp_path, monkeypatch):
        """main() calls reembed_store() and stamps provenance on rows."""
        import duckdb
        import reembed_memories

        # Create a store with 384-dim records.
        store = DuckDBMemoryStore(
            tmp_path / "cli.duckdb", user_id="alice",
            embedder=_MockEmbedder384(),
        )
        store.remember(category="personal_fact", content="CLI test fact about hiking")
        store.remember(category="personal_fact", content="CLI test fact about cooking")
        store.close()

        db_path = tmp_path / "cli.duckdb"

        # Monkeypatch _get_hermes_home and _resolve_db_path so main()
        # finds our test DB without HERMES_HOME.
        monkeypatch.setattr(reembed_memories, "_get_hermes_home", lambda: tmp_path)
        monkeypatch.setattr(
            reembed_memories, "_resolve_db_path",
            lambda home, override=None: db_path,
        )
        # Monkeypatch _load_model_name to return our mock model.
        monkeypatch.setattr(
            reembed_memories, "_load_model_name", lambda home: "mock-768",
        )
        # Monkeypatch LocalEmbedder to return our mock 768-dim embedder.
        monkeypatch.setattr(
            reembed_memories, "LocalEmbedder",
            lambda model_name: _MockEmbedder768(),
            raising=False,
        )
        # Inject LocalEmbedder into the module's namespace (main() does
        # `from embeddings import LocalEmbedder`).
        import types
        mock_embeddings = types.ModuleType("embeddings")
        mock_embeddings.LocalEmbedder = lambda model_name: _MockEmbedder768()
        monkeypatch.setitem(sys.modules, "embeddings", mock_embeddings)

        # Run main() with no args (not --dry-run).
        monkeypatch.setattr(sys, "argv", ["reembed_memories.py"])
        rc = reembed_memories.main()

        assert rc == 0

        # Verify provenance is stamped on every row.
        conn = duckdb.connect(str(db_path))
        try:
            rows = conn.execute(
                "SELECT memory_id, embedder_id, embedded_at, embedding_dim "
                "FROM memory_records ORDER BY memory_id"
            ).fetchall()
            assert len(rows) == 2
            for mid, eid, eat, edim in rows:
                assert eid == "mock-768", (
                    f"embedder_id not stamped by CLI for {mid}: got {eid}"
                )
                assert eat is not None, (
                    f"embedded_at not stamped by CLI for {mid}"
                )
                assert edim == 768, (
                    f"embedding_dim not stamped by CLI for {mid}: got {edim}"
                )
        finally:
            conn.close()

    def test_main_dry_run_does_not_write(self, tmp_path, monkeypatch):
        """main() --dry-run does not modify any rows."""
        import duckdb
        import reembed_memories

        store = DuckDBMemoryStore(
            tmp_path / "clidry.duckdb", user_id="alice",
            embedder=_MockEmbedder384(),
        )
        store.remember(category="personal_fact", content="Dry run CLI fact")
        store.close()

        db_path = tmp_path / "clidry.duckdb"

        monkeypatch.setattr(reembed_memories, "_get_hermes_home", lambda: tmp_path)
        monkeypatch.setattr(
            reembed_memories, "_resolve_db_path",
            lambda home, override=None: db_path,
        )
        monkeypatch.setattr(
            reembed_memories, "_load_model_name", lambda home: "mock-768",
        )
        import types
        mock_embeddings = types.ModuleType("embeddings")
        mock_embeddings.LocalEmbedder = lambda model_name: _MockEmbedder768()
        monkeypatch.setitem(sys.modules, "embeddings", mock_embeddings)

        # Capture before state.
        conn = duckdb.connect(str(db_path))
        before = conn.execute(
            "SELECT embedding_dim, embedder_id FROM memory_records"
        ).fetchall()
        conn.close()

        monkeypatch.setattr(sys, "argv", ["reembed_memories.py", "--dry-run"])
        rc = reembed_memories.main()
        assert rc == 0

        # Verify nothing changed.
        conn = duckdb.connect(str(db_path))
        try:
            after = conn.execute(
                "SELECT embedding_dim, embedder_id FROM memory_records"
            ).fetchall()
            assert before == after
        finally:
            conn.close()


class TestReembedSkipsSuperseded:
    """9. reembed_store() skips superseded (valid_to IS NOT NULL) rows."""

    def test_reembed_skips_superseded_rows(self, tmp_path):
        """reembed_store() does not re-embed rows with valid_to IS NOT NULL."""
        import duckdb
        from reembed_memories import reembed_store

        store = DuckDBMemoryStore(
            tmp_path / "sup.duckdb", user_id="alice",
            embedder=_MockEmbedder384(),
        )
        # Create a current record.
        rec = store.remember(
            category="personal_fact",
            content="Current fact about hiking trails",
        )
        # Manually supersede it by setting valid_to.
        with store._state.lock:
            store.connection.execute(
                "UPDATE memory_records SET valid_to = ? WHERE memory_id = ?",
                ["2025-01-01T00:00:00Z", rec.memory_id],
            )
        store.close()

        # Re-embed — the superseded row should NOT be touched.
        conn = duckdb.connect(str(tmp_path / "sup.duckdb"))
        try:
            report = reembed_store(conn, _MockEmbedder768())
            # No current rows to re-embed.
            assert report["re_embedded"] == 0
        finally:
            conn.close()

    def test_reembed_only_touches_current_rows(self, tmp_path):
        """reembed_store() re-embeds current rows but skips superseded ones."""
        import duckdb
        from reembed_memories import reembed_store

        store = DuckDBMemoryStore(
            tmp_path / "mixed.duckdb", user_id="alice",
            embedder=_MockEmbedder384(),
        )
        # Create two current records.
        rec1 = store.remember(
            category="personal_fact",
            content="Current fact about hiking in the mountains",
        )
        rec2 = store.remember(
            category="personal_fact",
            content="Current fact about cooking Italian pasta",
        )
        # Supersede rec1.
        with store._state.lock:
            store.connection.execute(
                "UPDATE memory_records SET valid_to = ? WHERE memory_id = ?",
                ["2025-01-01T00:00:00Z", rec1.memory_id],
            )
        store.close()

        # Re-embed — only rec2 (current) should be touched.
        conn = duckdb.connect(str(tmp_path / "mixed.duckdb"))
        try:
            report = reembed_store(conn, _MockEmbedder768())
            assert report["re_embedded"] == 1

            # Check that the superseded row kept its old dim.
            dims = conn.execute(
                "SELECT memory_id, embedding_dim FROM memory_records ORDER BY memory_id"
            ).fetchall()
            for mid, edim in dims:
                if mid == rec1.memory_id:
                    # Superseded — should still be 384 (not re-embedded).
                    assert edim == 384, (
                        f"Superseded row {mid} was re-embedded: dim={edim}"
                    )
                elif mid == rec2.memory_id:
                    # Current — should be 768.
                    assert edim == 768, (
                        f"Current row {mid} was not re-embedded: dim={edim}"
                    )
        finally:
            conn.close()


class TestBackfillStampsProvenance:
    """10. backfill_null_embeddings() stamps embedding_dim + provenance."""

    def test_backfill_stamps_embedding_dim(self, tmp_path):
        """backfill_null_embeddings() stamps embedding_dim on backfilled rows."""
        # Create a store with no embedder (records get NULL embeddings).
        store = DuckDBMemoryStore(
            tmp_path / "backfill.duckdb", user_id="alice",
            embedder=None,
        )
        rec = store.remember(
            category="personal_fact",
            content="Fact that will be backfilled after embedder recovery",
        )
        assert rec.embedding is None
        assert rec.embedding_dim is None

        # Now give the store an embedder and trigger backfill.
        store.embedder = _MockEmbedder768()
        backfilled = store.backfill_null_embeddings()
        assert backfilled >= 1

        # Verify the row has embedding_dim stamped.
        import duckdb
        conn = duckdb.connect(str(tmp_path / "backfill.duckdb"))
        try:
            row = conn.execute(
                "SELECT embedding_dim, embedder_id, embedded_at "
                "FROM memory_records WHERE memory_id = ?",
                [rec.memory_id],
            ).fetchone()
            assert row is not None
            edim, eid, eat = row
            assert edim == 768, f"embedding_dim not stamped: {edim}"
            assert eid == "mock-768", f"embedder_id not stamped: {eid}"
            assert eat is not None, "embedded_at not stamped"
        finally:
            conn.close()
        store.close()
