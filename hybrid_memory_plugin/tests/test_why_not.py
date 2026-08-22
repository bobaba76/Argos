"""Tests for Spec 2 — memory_why_not (explain_retrieval).

Covers:
- explain_retrieval returns structured diagnostics
- memory_not_found case
- found_in_results=True when memory is in top-k
- expired memory reason
- superseded memory reason
- scope mismatch reason
- low vector similarity reason
- deterministic (no LLM), read-only (retrieval_count unchanged)
- CLI wrapper smoke test

Run with:
    python -m pytest tests/test_why_not.py -v
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))


@pytest.fixture
def fixture_store(tmp_path):
    from store import DuckDBMemoryStore
    store = DuckDBMemoryStore(tmp_path / "fixture.duckdb", user_id="test_user")
    # A few memories with distinct content.
    store.remember(category="personal_fact", content="User is 38 years old and lives in Springfield")
    store.remember(category="preference", content="User prefers dark mode for all code editors")
    store.remember(category="context_note", content="User is working on a migration project this week")
    yield store, tmp_path / "fixture.duckdb"
    store.close()


class TestExplainRetrieval:
    def test_memory_not_found(self, fixture_store):
        store, _ = fixture_store
        result = store.explain_retrieval("anything", "mem-nonexistent")
        assert result["expected"] is None
        assert result["found_in_results"] is False
        assert any("memory_not_found" in r for r in result["reasons"])

    def test_found_in_results(self, fixture_store):
        store, _ = fixture_store
        # Save a memory and search for its exact content.
        rec = store.remember(
            category="personal_fact",
            content="User drives a blue Toyota Hilux",
        )
        result = store.explain_retrieval("blue Toyota Hilux", rec.memory_id)
        assert result["expected"] is not None
        assert result["expected"]["memory_id"] == rec.memory_id
        # Should be found (exact text match).
        assert result["found_in_results"] is True
        assert result["rank"] is not None
        assert result["rank"] >= 1

    def test_not_found_low_similarity(self, fixture_store):
        store, _ = fixture_store
        rec = store.remember(
            category="personal_fact",
            content="User likes astronomy and stargazing on clear nights",
        )
        # Search with a completely unrelated query.
        result = store.explain_retrieval("cooking pasta recipes", rec.memory_id)
        assert result["found_in_results"] is False
        # Should have a "ranked below" or "low similarity" reason.
        reasons_text = " ".join(result["reasons"])
        assert "ranked_below" in reasons_text or "low_vector" in reasons_text

    def test_expired_memory_reason(self, fixture_store):
        store, _ = fixture_store
        past = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        rec = store.remember(
            category="context_note",
            content="User has a temporary promo code EXPIRED999",
            expires_at=past,
        )
        result = store.explain_retrieval("promo code EXPIRED999", rec.memory_id)
        reasons_text = " ".join(result["reasons"])
        assert "expired" in reasons_text.lower()

    def test_superseded_memory_reason(self, fixture_store):
        store, _ = fixture_store
        rec = store.remember(
            category="personal_fact",
            content="User lives at 123 Old Street",
        )
        updated = store.update_memory(rec.memory_id, content="User lives at 456 New Avenue")
        # The old record (rec.memory_id) is now superseded.
        result = store.explain_retrieval("Old Street address", rec.memory_id)
        reasons_text = " ".join(result["reasons"])
        assert "superseded" in reasons_text.lower()

    def test_returns_top_results(self, fixture_store):
        store, _ = fixture_store
        rec = store.remember(
            category="personal_fact",
            content="User owns a parrot named Polly",
        )
        result = store.explain_retrieval("parrot named Polly", rec.memory_id)
        assert isinstance(result["top_results"], list)
        assert len(result["top_results"]) > 0
        # Each top result should have the expected fields.
        for r in result["top_results"]:
            assert "memory_id" in r
            assert "content" in r
            assert "similarity" in r

    def test_diagnostics_include_vector_similarity(self, fixture_store):
        store, _ = fixture_store
        rec = store.remember(
            category="personal_fact",
            content="User plays the guitar on weekends",
        )
        result = store.explain_retrieval("guitar weekends", rec.memory_id)
        diag = result["diagnostics"]
        # Vector similarity should be present if the embedder is available.
        if "vector_similarity" in diag:
            assert isinstance(diag["vector_similarity"], (int, float))
            assert 0.0 <= diag["vector_similarity"] <= 1.0

    def test_diagnostics_include_text_score(self, fixture_store):
        store, _ = fixture_store
        rec = store.remember(
            category="personal_fact",
            content="User has a dog named Rex",
        )
        result = store.explain_retrieval("dog named Rex", rec.memory_id)
        diag = result["diagnostics"]
        assert "text_match_score" in diag

    def test_expected_summary_fields(self, fixture_store):
        store, _ = fixture_store
        rec = store.remember(
            category="personal_fact",
            content="User is a certified scuba diver",
        )
        result = store.explain_retrieval("scuba diver", rec.memory_id)
        expected = result["expected"]
        assert expected["memory_id"] == rec.memory_id
        assert expected["category"] == "personal_fact"
        assert "scuba diver" in expected["content"]

    def test_project_scope_mismatch_reason(self, tmp_path):
        """A project-scoped memory queried with a different project_id must
        produce a project_scope_mismatch reason (top-3 cause of why-not)."""
        from store import DuckDBMemoryStore
        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user")
        rec = store.remember(
            category="context_note",
            content="User is working on the Acme migration project",
            project_id="project-acme",
        )
        # Query with a different project scope.
        result = store.explain_retrieval(
            "Acme migration", rec.memory_id, project_id="project-other",
        )
        reasons_text = " ".join(result["reasons"])
        assert "project_scope_mismatch" in reasons_text
        assert result["diagnostics"]["project_scope_mismatch"] is True
        store.close()

    def test_project_scope_match_no_mismatch_reason(self, tmp_path):
        """When project_id matches, no project_scope_mismatch reason."""
        from store import DuckDBMemoryStore
        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user")
        rec = store.remember(
            category="context_note",
            content="User is working on the Globex deployment",
            project_id="project-globex",
        )
        result = store.explain_retrieval(
            "Globex deployment", rec.memory_id, project_id="project-globex",
        )
        reasons_text = " ".join(result["reasons"])
        assert "project_scope_mismatch" not in reasons_text
        store.close()

    def test_vector_similarity_matches_production(self, tmp_path):
        """The diagnostic vector_similarity must match what the production
        pipeline computes (reuse DuckDB's list_cosine_similarity, not a
        hand-rolled Python cosine)."""
        from store import DuckDBMemoryStore
        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user")
        rec = store.remember(
            category="personal_fact",
            content="User enjoys painting watercolor landscapes",
        )
        result = store.explain_retrieval("watercolor landscapes painting", rec.memory_id)
        diag = result["diagnostics"]
        if "vector_similarity" in diag:
            # Re-compute via the same DuckDB path to verify they match.
            if store.embedder and hasattr(store.embedder, "embed"):
                query_emb = store.embedder.embed("watercolor landscapes painting", is_query=True)
                vec_text = "[" + ",".join(repr(float(x)) for x in query_emb) + "]"
                row = store.connection.execute(
                    f"""SELECT list_cosine_similarity(
                            embedding, CAST(? AS DOUBLE[{len(query_emb)}])
                        ) AS sim
                        FROM memory_records WHERE memory_id = ?""",
                    [vec_text, rec.memory_id],
                ).fetchone()
                prod_sim = round(float(row[0]), 4) if row and row[0] else None
                assert prod_sim is not None
                assert diag["vector_similarity"] == prod_sim, (
                    f"diagnostic vector_similarity {diag['vector_similarity']} "
                    f"!= production {prod_sim}"
                )
        store.close()


class TestReadOnly:
    def test_retrieval_count_unchanged(self, fixture_store):
        """explain_retrieval must not inflate retrieval_count."""
        store, _ = fixture_store
        rec = store.remember(
            category="personal_fact",
            content="User collects vintage stamps from Europe",
        )
        before = store.connection.execute(
            "SELECT retrieval_count FROM memory_records WHERE memory_id = ?",
            [rec.memory_id],
        ).fetchone()[0]
        store.explain_retrieval("vintage stamps Europe", rec.memory_id)
        after = store.connection.execute(
            "SELECT retrieval_count FROM memory_records WHERE memory_id = ?",
            [rec.memory_id],
        ).fetchone()[0]
        assert after == before, "explain_retrieval must not change retrieval_count"


class TestNoLLM:
    def test_no_llm_dependency(self, fixture_store, monkeypatch):
        """explain_retrieval must work without any LLM client."""
        store, _ = fixture_store
        rec = store.remember(
            category="personal_fact",
            content="User enjoys painting landscapes",
        )
        # Block LLM imports.
        monkeypatch.setitem(sys.modules, "agent.auxiliary_client", None)
        monkeypatch.setitem(sys.modules, "agent", None)
        result = store.explain_retrieval("painting landscapes", rec.memory_id)
        assert result is not None
        assert "reasons" in result


class TestDeterministic:
    def test_same_input_same_output(self, fixture_store):
        """explain_retrieval must be deterministic for the same input."""
        store, _ = fixture_store
        rec = store.remember(
            category="personal_fact",
            content="User is learning Japanese calligraphy",
        )
        r1 = store.explain_retrieval("Japanese calligraphy", rec.memory_id)
        r2 = store.explain_retrieval("Japanese calligraphy", rec.memory_id)
        assert r1["found_in_results"] == r2["found_in_results"]
        assert r1["rank"] == r2["rank"]
        assert r1["reasons"] == r2["reasons"]


class TestCLISmoke:
    def test_cli_help(self):
        """The CLI wrapper must show --help without errors."""
        import subprocess
        cli_path = _plugin_dir / "why_not_cli.py"
        result = subprocess.run(
            [sys.executable, str(cli_path), "--help"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "memory_why_not" in result.stdout or "diagnose" in result.stdout.lower()

    def test_cli_missing_db(self):
        """The CLI must exit 1 when the DB doesn't exist."""
        import subprocess
        cli_path = _plugin_dir / "why_not_cli.py"
        result = subprocess.run(
            [sys.executable, str(cli_path),
             "--db", "nonexistent.duckdb",
             "--query", "test",
             "--expected-memory-id", "mem-x"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 1

    def test_cli_runs_diagnosis(self, tmp_path):
        """The CLI must run a full diagnosis on a fixture store."""
        import subprocess
        from store import DuckDBMemoryStore
        db_path = tmp_path / "cli_test.duckdb"
        store = DuckDBMemoryStore(db_path, user_id="default_user")
        rec = store.remember(
            category="personal_fact",
            content="User is a certified pilot",
        )
        store.close()

        cli_path = _plugin_dir / "why_not_cli.py"
        result = subprocess.run(
            [sys.executable, str(cli_path),
             "--db", str(db_path),
             "--query", "certified pilot",
             "--expected-memory-id", rec.memory_id,
             "--json"],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        data = json.loads(result.stdout)
        assert data["expected_memory_id"] == rec.memory_id
        assert "reasons" in data
