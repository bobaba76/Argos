"""Tests for #392 part 3: catch-up hygiene lexicon sweep.

The lexicon sweep is a deterministic, zero-LLM pass that flags existing
active records matching an implementation-machinery lexicon with
``record_class='system_internal'``. This retires existing pollution over
time in addition to the one-off quarantine sweep applied 9/9.

Acceptance criteria from the issue:
- The lexicon matches implementation machinery (config values, schema
  versions, function names, tuning parameters, data structures,
  benchmark decisions) — NOT project memory (PRs, issues, features,
  roadmap items).
- "rollup config defaults to false" = excluded (engine-room noise).
- "PR #392 merged" = eligible (project memory).
- Dry-run by default — preview first, like the POPIA erase workflow.
- Apply requires an explicit flag.
- Idempotent — re-running on already-flagged records is a no-op.
- Config-gated — only runs when distillation_exclude_system_internal is
  true (the default).

Run with:
    python -m pytest tests/test_system_internal_lexicon.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))

from _test_embedder import DeterministicTestEmbedder
from system_internal_lexicon import is_system_internal, sweep_system_internal


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    """A fresh DuckDBMemoryStore with the deterministic test embedder."""
    from store import DuckDBMemoryStore
    embedder = DeterministicTestEmbedder()
    s = DuckDBMemoryStore(
        tmp_path / "test.duckdb", user_id="test_user", embedder=embedder,
    )
    yield s
    s.close()


# ---------------------------------------------------------------------------
# Lexicon unit tests — what the filter catches and what it lets through.
# ---------------------------------------------------------------------------

class TestLexiconMatches:
    """The lexicon must match implementation-machinery content."""

    @pytest.mark.parametrize("content", [
        # Config defaults / config keys
        "context_aware_retrieval defaults to true in config_schema",
        "rollup config defaults to false in config_model",
        "hybrid_memory.json has 75 keys including router_*",
        # Schema versions / migrations
        "mutation_events schema v5 added actor column",
        "LATEST_SCHEMA_VERSION bumped to 12 after the record_class migration",
        "migration_11_to_12 adds the record_class column",
        # Internal module / function names
        "_sanitize_args strips PII before the LLM call",
        "backfill_graph --prune-orphans removes graph nodes with no DuckDB record",
        "store_maintenance.py has the load_eligible_records function",
        "memory_service.py dispatches count_eligible_since over RPC",
        "api_facade exposes the review_candidate operation",
        "mcp_server.py validates JSON-RPC against the schema",
        # Internal data structures
        "mutation_events is an append-only, actor-attributed mutation log",
        "memory_records has 30 columns after the record_class addition",
        "deletion_tombstones are kept for POPIA audit",
        "rejection_ledger tracks why proposals were rejected",
        # Tuning / benchmark / threshold
        "reranker verdict superseded by graph boost at threshold 0.75",
        "similarity floor is 0.3 for the query expansion gate",
        "cluster threshold 0.75 is too aggressive for small clusters",
        "benchmark decision: drop the ANN index below 10k records",
        "context_aware_retrieval is enabled by default",
        "query_expansion_enabled gates the LLM expansion call",
        "phrase_lift_alpha is set to 0.3 in the config",
        "injection_min_score filters out low-relevance memories",
        # Rollup / distillation internals
        "rollup config defaults to false in config_model",
        "rollup_enabled ships false — the rollup must never enable itself",
        "rollup_after_days is 30 by default",
        "distillation internals: cluster threshold is 0.75",
        "distillation_enabled ships false — the dream must never enable itself",
        "distillation_exclude_system_internal is true by default",
        "distillation_min_new_records is 20 by default",
        "distillation_cooldown_hours is 24 by default",
        "distillation_max_calls is 10 by default",
        # Store layout / audit
        "store layout audit: memory_records has 30 columns",
        "store audit found 350 system-internal records",
        "record_class is the new marker column for system-internal records",
        "embedding_dim is 384 for the BGE model",
        "embedder_id is set when the embedding is computed",
    ])
    def test_matches_implementation_machinery(self, content):
        assert is_system_internal(content), f"Expected match: {content!r}"


class TestLexiconRejects:
    """The lexicon must NOT match project memory (PRs, issues, features,
    roadmap items, user facts)."""

    @pytest.mark.parametrize("content", [
        # Project memory — the canonical example from the issue
        "PR #392 merged — distillation input filter landed",
        # Other project memory
        "PR #385 merged — stale test assertions fixed",
        "Issue #384: Spec-11 write tiers enabled by default",
        "Feature: temporal-aware graph with timestamped edges shipped",
        "Roadmap: adapters for OpenAI and Anthropic are next",
        # User facts — must never be flagged
        "User lives in Cape Town and prefers dark mode",
        "User likes Python and works on project Alpha",
        "User completed task 5 on schedule",
        "User's favorite coffee is espresso",
        # Generic project notes — no internal names
        "The distillation feature is now enabled in production",
        "Rollup ran successfully and consolidated 50 records",
        "The reranker improved retrieval relevance in the eval",
        # False-positive regressions (#392 review): project memory that
        # mentions config keys / data structures / distillation terms but
        # in a project-event context, not an internal-machinery context.
        "the distillation pass is now stable",
        "distillation cluster found 3 insights",
        "context_aware_retrieval feature shipped in v2",
        "query_expansion_enabled is a great feature",
        "mutation_events table is useful for auditing",
        "memory_records count reached 1000 today",
    ])
    def test_rejects_project_memory(self, content):
        assert not is_system_internal(content), f"Expected no match: {content!r}"


# ---------------------------------------------------------------------------
# Sweep tests — dry-run, apply, idempotent, config gate.
# ---------------------------------------------------------------------------

class TestSweepDryRun:
    """Dry-run (default) previews without writing."""

    def test_dry_run_returns_report_without_writing(self, store):
        store.remember(
            category="context_note",
            content="rollup config defaults to false in config_model",
            dedup=False,
        )
        store.remember(
            category="personal_fact",
            content="User lives in Cape Town",
            dedup=False,
        )
        report = sweep_system_internal(store, dry_run=True)
        assert report["dry_run"] is True
        assert report["scanned"] == 2
        assert report["matched"] == 1
        assert report["flagged"] == 0  # dry-run writes nothing
        # The matched record is the implementation-machinery one.
        assert len(report["matches"]) == 1
        assert "rollup config" in report["matches"][0]["content_preview"]

    def test_dry_run_is_default(self, store):
        store.remember(
            category="context_note",
            content="mutation_events schema v5 added actor column",
            dedup=False,
        )
        # No dry_run kwarg → defaults to True (preview only).
        report = sweep_system_internal(store)
        assert report["dry_run"] is True
        assert report["flagged"] == 0


class TestSweepApply:
    """Apply flags matching records with record_class='system_internal'."""

    def test_apply_flags_matching_records(self, store):
        rec_internal = store.remember(
            category="context_note",
            content="rollup config defaults to false in config_model",
            dedup=False,
        )
        rec_user = store.remember(
            category="personal_fact",
            content="User lives in Cape Town",
            dedup=False,
        )
        report = sweep_system_internal(store, dry_run=False)
        assert report["dry_run"] is False
        assert report["matched"] == 1
        assert report["flagged"] == 1

        # Verify the internal record was flagged.
        records = store.load_eligible_records(
            since=None, limit=100, exclude_system_internal=False,
        )
        by_id = {r.memory_id: r for r in records}
        assert by_id[rec_internal.memory_id].record_class == "system_internal"
        # The user record was NOT flagged.
        assert by_id[rec_user.memory_id].record_class is None

    def test_apply_alias(self, store):
        """apply=True is an alias for dry_run=False."""
        store.remember(
            category="context_note",
            content="mutation_events schema v5 added actor column",
            dedup=False,
        )
        report = sweep_system_internal(store, apply=True)
        assert report["dry_run"] is False
        assert report["flagged"] == 1


class TestSweepIdempotent:
    """Re-running on already-flagged records is a no-op."""

    def test_idempotent(self, store):
        store.remember(
            category="context_note",
            content="rollup config defaults to false in config_model",
            dedup=False,
        )
        # First apply: flags 1 record.
        report1 = sweep_system_internal(store, apply=True)
        assert report1["flagged"] == 1
        assert report1["already_flagged"] == 0
        # Second apply: 0 newly flagged, 1 already flagged.
        report2 = sweep_system_internal(store, apply=True)
        assert report2["flagged"] == 0
        assert report2["already_flagged"] == 1
        assert report2["matched"] == 0  # already-flagged records are not re-matched


class TestSweepProjectMemoryEligible:
    """Project memory is NOT flagged by the sweep."""

    def test_project_memory_not_flagged(self, store):
        store.remember(
            category="personal_fact",
            content="PR #392 merged — distillation input filter landed",
            dedup=False,
        )
        store.remember(
            category="context_note",
            content="rollup config defaults to false in config_model",
            dedup=False,
        )
        report = sweep_system_internal(store, apply=True)
        assert report["matched"] == 1
        assert report["flagged"] == 1
        # The project memory record is still NULL (not flagged).
        records = store.load_eligible_records(
            since=None, limit=100, exclude_system_internal=False,
        )
        project_rec = [r for r in records if "PR #392" in r.content]
        assert len(project_rec) == 1
        assert project_rec[0].record_class is None


class TestSweepExclusionAfterFlagging:
    """After the sweep flags records, distillation excludes them."""

    def test_flagged_records_excluded_from_distillation_load(self, store):
        store.remember(
            category="personal_fact",
            content="User likes Python",
            dedup=False,
        )
        store.remember(
            category="context_note",
            content="rollup config defaults to false in config_model",
            dedup=False,
        )
        # Run the sweep to flag the system-internal record.
        sweep_system_internal(store, apply=True)
        # Now distillation excludes it.
        records = store.load_eligible_records(
            since=None, limit=100, exclude_system_internal=True,
        )
        contents = [r.content for r in records]
        assert any("Python" in c for c in contents)
        assert not any("rollup config" in c for c in contents)


class TestStartupSweepRunOnce:
    """The startup sweep helper uses a run-once guard via system_state.

    #392 review warning 4: first deploy is a DRY-RUN — it logs the report
    but does NOT apply flags or set the run-once guard. The second startup
    applies the flags and sets the guard.
    """

    def test_first_call_is_dry_run(self, store):
        """First call (no prior state) runs a dry-run, does NOT apply."""
        from system_internal_lexicon import startup_sweep_if_needed
        store.remember(
            category="context_note",
            content="rollup config defaults to false in config_model",
            dedup=False,
        )
        report = startup_sweep_if_needed(store)
        assert report is not None
        assert report["dry_run"] is True
        assert report["matched"] == 1
        assert report["flagged"] == 0  # dry-run, no writes
        # The record was NOT flagged.
        records = store.load_eligible_records(
            since=None, limit=100, exclude_system_internal=False,
        )
        rec = [r for r in records if "rollup config" in r.content]
        assert rec[0].record_class is None

    def test_second_call_applies(self, store):
        """Second call (after dry-run) applies the sweep and sets the guard."""
        from system_internal_lexicon import startup_sweep_if_needed
        store.remember(
            category="context_note",
            content="rollup config defaults to false in config_model",
            dedup=False,
        )
        # First call: dry-run.
        report1 = startup_sweep_if_needed(store)
        assert report1["dry_run"] is True
        # Second call: apply.
        report2 = startup_sweep_if_needed(store)
        assert report2 is not None
        assert report2["dry_run"] is False
        assert report2["flagged"] == 1

    def test_third_call_skips(self, store):
        """Third call (after apply) is skipped (run-once guard)."""
        from system_internal_lexicon import startup_sweep_if_needed
        store.remember(
            category="context_note",
            content="rollup config defaults to false in config_model",
            dedup=False,
        )
        startup_sweep_if_needed(store)  # dry-run
        startup_sweep_if_needed(store)  # apply
        # Third call: skipped.
        report3 = startup_sweep_if_needed(store)
        assert report3 is None
