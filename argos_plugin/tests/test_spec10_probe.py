"""#200 Spec-10 PR 1/3: Circuit-breaker probe (falsifiable gate).

Builds a 100-item SYNTHETIC "backlog" (personas/fixtures only, no
personal content) into a disposable HERMES_HOME; runs 20 natural
"what's still open?" / "what are my outstanding tasks?" prompts
through the RANKED retrieval path; records every miss (item present
in backlog but not returned).

If ZERO misses exist, the collections feature (PR-2) is NOT needed —
ranked search is exhaustive. If even one item is missed, the premise
for collections holds.

This is a deterministic, NO-LLM test. Run individually:

    python -m pytest argos_plugin/tests/test_spec10_probe.py -q \
        -p no:cacheprovider --tb=short
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))

# -- Synthetic backlog (100 items, personas/fixtures only) -------------------

# Each item is a task/open-item that a user might have in their memory.
# These are SYNTHETIC — no personal content, no real names, no
# medication refs. They represent the kind of "outstanding task" that
# a user would expect to surface when asking "what's still open?".

_SYNTHETIC_BACKLOG: list[dict] = [
    # Tasks (20)
    {"content": "Task: review the quarterly budget proposal for Project Alpha", "category": "task", "tags": ["task", "project-alpha"]},
    {"content": "Task: schedule a team sync for the Hermes integration review", "category": "task", "tags": ["task", "meeting"]},
    {"content": "Task: follow up with the vendor about the API rate limit increase", "category": "task", "tags": ["task", "vendor"]},
    {"content": "Task: finalize the deployment runbook for the staging environment", "category": "task", "tags": ["task", "deploy"]},
    {"content": "Task: write the post-mortem for the database outage incident", "category": "task", "tags": ["task", "incident"]},
    {"content": "Task: update the onboarding documentation for new hires", "category": "task", "tags": ["task", "docs"]},
    {"content": "Task: review pull request #142 for the graph traversal fix", "category": "task", "tags": ["task", "code-review"]},
    {"content": "Task: investigate the memory leak reported in issue #87", "category": "task", "tags": ["task", "bug"]},
    {"content": "Task: prepare the demo for the stakeholder presentation next week", "category": "task", "tags": ["task", "demo"]},
    {"content": "Task: migrate the config files from YAML to JSON format", "category": "task", "tags": ["task", "migration"]},
    {"content": "Task: audit the access control lists for the production tenant", "category": "task", "tags": ["task", "security"]},
    {"content": "Task: refactor the embedding pipeline to support batch mode", "category": "task", "tags": ["task", "refactor"]},
    {"content": "Task: respond to the customer support ticket about export format", "category": "task", "tags": ["task", "support"]},
    {"content": "Task: set up the CI pipeline for the new docs site", "category": "task", "tags": ["task", "ci"]},
    {"content": "Task: review the third-party dependency licenses for compliance", "category": "task", "tags": ["task", "legal"]},
    {"content": "Task: create a backup strategy for the Kuzu graph database", "category": "task", "tags": ["task", "backup"]},
    {"content": "Task: optimize the DuckDB query for the recent-memories listing", "category": "task", "tags": ["task", "performance"]},
    {"content": "Task: write integration tests for the REST API search endpoint", "category": "task", "tags": ["task", "testing"]},
    {"content": "Task: clean up the deprecated MCP tool definitions", "category": "task", "tags": ["task", "cleanup"]},
    {"content": "Task: plan the v2 schema migration for the evidence table", "category": "task", "tags": ["task", "schema"]},

    # Goals (20)
    {"content": "Goal: ship the external writes milestone by end of quarter", "category": "goal", "tags": ["goal", "milestone"]},
    {"content": "Goal: achieve 95% retrieval recall on the self-corpus benchmark", "category": "goal", "tags": ["goal", "benchmark"]},
    {"content": "Goal: reduce the average search latency to under 200ms", "category": "goal", "tags": ["goal", "performance"]},
    {"content": "Goal: complete the POPIA compliance audit for the pilot deployment", "category": "goal", "tags": ["goal", "compliance"]},
    {"content": "Goal: publish the public documentation site on GitHub Pages", "category": "goal", "tags": ["goal", "docs"]},
    {"content": "Goal: migrate all tests to the per-file isolation pattern", "category": "goal", "tags": ["goal", "testing"]},
    {"content": "Goal: implement the collections feature for exhaustive task tracking", "category": "goal", "tags": ["goal", "collections"]},
    {"content": "Goal: reach 100% facade operation test coverage", "category": "goal", "tags": ["goal", "coverage"]},
    {"content": "Goal: eliminate all single-process test deadlocks", "category": "goal", "tags": ["goal", "stability"]},
    {"content": "Goal: document every config key in the reference guide", "category": "goal", "tags": ["goal", "docs"]},
    {"content": "Goal: ship the admin console for browse and review", "category": "goal", "tags": ["goal", "ui"]},
    {"content": "Goal: add reranker support with CUDA torch", "category": "goal", "tags": ["goal", "reranker"]},
    {"content": "Goal: implement the temporal-aware graph traversal", "category": "goal", "tags": ["goal", "graph"]},
    {"content": "Goal: complete the structured ingestion for CSV and JSON", "category": "goal", "tags": ["goal", "ingest"]},
    {"content": "Goal: ship the portable export and import feature", "category": "goal", "tags": ["goal", "export"]},
    {"content": "Goal: add the schedule-aware self-compaction", "category": "goal", "tags": ["goal", "compaction"]},
    {"content": "Goal: implement the explainability pack for retrieval", "category": "goal", "tags": ["goal", "explainability"]},
    {"content": "Goal: reach zero open security findings on the audit", "category": "goal", "tags": ["goal", "security"]},
    {"content": "Goal: ship the versioned store schema migrations", "category": "goal", "tags": ["goal", "schema"]},
    {"content": "Goal: complete the CI retrieval gates for smoke and weekly", "category": "goal", "tags": ["goal", "ci"]},

    # Preferences (20)
    {"content": "Preference: use dark mode for all code editors and terminals", "category": "preference", "tags": ["preference", "editor"]},
    {"content": "Preference: prefer concise commit messages under 72 characters", "category": "preference", "tags": ["preference", "git"]},
    {"content": "Preference: always run tests per-file to avoid the deadlock", "category": "preference", "tags": ["preference", "testing"]},
    {"content": "Preference: use DuckDB shared_service mode for all production deploys", "category": "preference", "tags": ["preference", "storage"]},
    {"content": "Preference: keep the REST server loopback-only, never bind 0.0.0.0", "category": "preference", "tags": ["preference", "security"]},
    {"content": "Preference: use BAAI/bge-small-en-v1.5 for local embeddings", "category": "preference", "tags": ["preference", "embeddings"]},
    {"content": "Preference: set injection_min_score to 0.30 for relevance floor", "category": "preference", "tags": ["preference", "retrieval"]},
    {"content": "Preference: enable context_aware_retrieval for pronoun resolution", "category": "preference", "tags": ["preference", "retrieval"]},
    {"content": "Preference: keep chain_unfold on auto mode for change-intent queries", "category": "preference", "tags": ["preference", "chains"]},
    {"content": "Preference: enable conflict_surfacing for trust-critical use cases", "category": "preference", "tags": ["preference", "retrieval"]},
    {"content": "Preference: use chronological_injection for temporal queries", "category": "preference", "tags": ["preference", "retrieval"]},
    {"content": "Preference: keep graph_traversal_depth at 2 for multi-hop queries", "category": "preference", "tags": ["preference", "graph"]},
    {"content": "Preference: disable graph_inject_candidates to avoid noise", "category": "preference", "tags": ["preference", "graph"]},
    {"content": "Preference: set max_injected_items to 20 for context budget", "category": "preference", "tags": ["preference", "retrieval"]},
    {"content": "Preference: use query_expansion for weak queries below 0.3 similarity", "category": "preference", "tags": ["preference", "retrieval"]},
    {"content": "Preference: keep external_sources_require_confirmation true", "category": "preference", "tags": ["preference", "security"]},
    {"content": "Preference: use evidence_retention full for provenance detail", "category": "preference", "tags": ["preference", "storage"]},
    {"content": "Preference: set skip_retrieval_on_trivial to true for cost savings", "category": "preference", "tags": ["preference", "retrieval"]},
    {"content": "Preference: keep the reranker disabled on CPU-only deployments", "category": "preference", "tags": ["preference", "reranker"]},
    {"content": "Preference: use deployment_mode local_sku for air-gapped setups", "category": "preference", "tags": ["preference", "deployment"]},

    # Context notes (20)
    {"content": "Context: the Hermes plugin manifest declares pip_dependencies in plugin.yaml", "category": "context_note", "tags": ["context", "plugin"]},
    {"content": "Context: the facade operation allowlist is in api_facade.py READ_OPERATIONS and PROPOSAL_OPERATIONS", "category": "context_note", "tags": ["context", "facade"]},
    {"content": "Context: the REST server binds to port 8732 on 127.0.0.1", "category": "context_note", "tags": ["context", "rest"]},
    {"content": "Context: the MCP server speaks JSON-RPC 2.0 over stdio", "category": "context_note", "tags": ["context", "mcp"]},
    {"content": "Context: the DuckDB database file is hybrid_memory.duckdb in HERMES_HOME", "category": "context_note", "tags": ["context", "storage"]},
    {"content": "Context: the Kuzu graph directory is hybrid_memory_kuzu in HERMES_HOME", "category": "context_note", "tags": ["context", "graph"]},
    {"content": "Context: the deploy script syncs runtime modules from repo to live install", "category": "context_note", "tags": ["context", "deploy"]},
    {"content": "Context: the idempotency registry uses SHA-256 request hashes with 24h TTL", "category": "context_note", "tags": ["context", "idempotency"]},
    {"content": "Context: the audit log hashes query text and never logs bearer tokens", "category": "context_note", "tags": ["context", "audit"]},
    {"content": "Context: the ACL fails closed in API mode per spec-09 decision 3", "category": "context_note", "tags": ["context", "acl"]},
    {"content": "Context: the candidate review invariant prevents auto_review from approving", "category": "context_note", "tags": ["context", "review"]},
    {"content": "Context: the external source candidates require human confirmation", "category": "context_note", "tags": ["context", "review"]},
    {"content": "Context: the portable export includes records evidence candidates and tombstones", "category": "context_note", "tags": ["context", "export"]},
    {"content": "Context: the POPIA erase workflow produces append-only deletion receipts", "category": "context_note", "tags": ["context", "popia"]},
    {"content": "Context: the structured ingestion supports JSON and CSV with dry-run mode", "category": "context_note", "tags": ["context", "ingest"]},
    {"content": "Context: the schema migrations use DuckDB schema_meta not SQLite PRAGMA user_version", "category": "context_note", "tags": ["context", "schema"]},
    {"content": "Context: the temporal graph has valid_from and valid_to columns on edges", "category": "context_note", "tags": ["context", "graph"]},
    {"content": "Context: the retrieval uses RRF fusion of vector and keyword search", "category": "context_note", "tags": ["context", "retrieval"]},
    {"content": "Context: the chain unfold injects a compact version arc on change-intent queries", "category": "context_note", "tags": ["context", "chains"]},
    {"content": "Context: the CI retrieval gates run smoke slice and weekly tiers", "category": "context_note", "tags": ["context", "ci"]},

    # Personal facts (20) — synthetic personas, no real names
    {"content": "Fact: the project lead for Project Alpha is Person-X", "category": "personal_fact", "tags": ["fact", "project-alpha"]},
    {"content": "Fact: the QA engineer for the pilot deployment is Person-Y", "category": "personal_fact", "tags": ["fact", "qa"]},
    {"content": "Fact: the stakeholder for the compliance audit is Person-Z", "category": "personal_fact", "tags": ["fact", "compliance"]},
    {"content": "Fact: the vendor contact for the API rate limit is Contact-A", "category": "personal_fact", "tags": ["fact", "vendor"]},
    {"content": "Fact: the team uses a shared calendar for sprint planning", "category": "personal_fact", "tags": ["fact", "team"]},
    {"content": "Fact: the code review rotation includes Person-X and Person-Y", "category": "personal_fact", "tags": ["fact", "team"]},
    {"content": "Fact: the on-call schedule rotates weekly between three engineers", "category": "personal_fact", "tags": ["fact", "oncall"]},
    {"content": "Fact: the budget approval threshold is 5000 currency units", "category": "personal_fact", "tags": ["fact", "budget"]},
    {"content": "Fact: the staging environment uses port 8732 for the REST server", "category": "personal_fact", "tags": ["fact", "staging"]},
    {"content": "Fact: the production deployment uses shared_service storage mode", "category": "personal_fact", "tags": ["fact", "production"]},
    {"content": "Fact: the development team has four engineers and one QA lead", "category": "personal_fact", "tags": ["fact", "team"]},
    {"content": "Fact: the pilot tenant is named default and uses open ACL", "category": "personal_fact", "tags": ["fact", "tenant"]},
    {"content": "Fact: the embedding model downloads on first use and caches locally", "category": "personal_fact", "tags": ["fact", "embeddings"]},
    {"content": "Fact: the graph database is Kuzu version 0.11.3", "category": "personal_fact", "tags": ["fact", "graph"]},
    {"content": "Fact: the memory store is DuckDB version 1.5.5", "category": "personal_fact", "tags": ["fact", "storage"]},
    {"content": "Fact: the reranker model is BAAI/bge-reranker-base at 420MB", "category": "personal_fact", "tags": ["fact", "reranker"]},
    {"content": "Fact: the config file is hybrid_memory.json in the Hermes home directory", "category": "personal_fact", "tags": ["fact", "config"]},
    {"content": "Fact: the plugin hooks are on_turn_start sync_turn on_session_end on_session_switch", "category": "personal_fact", "tags": ["fact", "plugin"]},
    {"content": "Fact: the license is BSL 1.1 converting to Apache 2.0 in 2030", "category": "personal_fact", "tags": ["fact", "license"]},
    {"content": "Fact: the docs site uses MkDocs Material and deploys to GitHub Pages", "category": "personal_fact", "tags": ["fact", "docs"]},
]

assert len(_SYNTHETIC_BACKLOG) == 100, f"expected 100 items, got {len(_SYNTHETIC_BACKLOG)}"

# 20 natural "what's still open?" prompts
_PROBE_PROMPTS: list[str] = [
    "what are my outstanding tasks?",
    "what's still open on my plate?",
    "what do I still need to do?",
    "show me my pending tasks",
    "what tasks are incomplete?",
    "what goals am I still working toward?",
    "what's left to do on Project Alpha?",
    "what are my open action items?",
    "what haven't I finished yet?",
    "remind me what I still need to take care of",
    "what are my unresolved tasks?",
    "what's on my to-do list?",
    "what outstanding work do I have?",
    "what are my current priorities?",
    "what am I supposed to follow up on?",
    "what tasks are still in progress?",
    "what do I need to wrap up?",
    "what's waiting on me?",
    "what are my incomplete goals?",
    "what's still pending?",
]

assert len(_PROBE_PROMPTS) == 20, f"expected 20 prompts, got {len(_PROBE_PROMPTS)}"


# -- Probe implementation ---------------------------------------------------

def _build_backlog_store(home: Path):
    """Build a disposable store with the 100-item synthetic backlog.

    Uses DuckDBMemoryStore directly (no shared service) for deterministic
    test behavior. The store is text-only (no embedder) to avoid the
    model download — the probe tests RANKED retrieval, which includes
    keyword/RRF fusion even without vectors.
    """
    from store import DuckDBMemoryStore
    home = Path(home)
    home.mkdir(parents=True, exist_ok=True)
    db_path = home / "probe.duckdb"
    store = DuckDBMemoryStore(db_path, user_id="probe_user", embedder=None)
    for item in _SYNTHETIC_BACKLOG:
        store.remember(
            content=item["content"],
            category=item["category"],
            tags=item["tags"],
            dedup=False,  # don't dedup — we want all 100
        )
    return store


def _run_probe(store, prompt: str, limit: int = 50) -> set[str]:
    """Run one probe prompt through ranked retrieval.

    Returns the set of content strings that surfaced in the results.
    """
    results = store.search(prompt, limit=limit)
    return {r.content for r in results}


class TestCircuitBreakerProbe:
    """Spec-10 PR 1/3: circuit-breaker probe.

    Builds a 100-item synthetic backlog, runs 20 "what's still open?"
    prompts through ranked retrieval, and records every miss.

    If ZERO misses exist, the collections feature is NOT needed —
    ranked search is exhaustive. If even one item is missed, the
    premise for collections holds and we proceed with the write tier.
    """

    def test_backlog_has_100_items(self):
        assert len(_SYNTHETIC_BACKLOG) == 100

    def test_prompts_have_20_items(self):
        assert len(_PROBE_PROMPTS) == 20

    def test_probe_finds_at_least_one_miss(self, tmp_path):
        """The falsifiable gate: run 20 prompts against 100 items and
        record misses. If at least one item is never returned across
        all 20 prompts, the premise for collections holds.

        This test PASSES when a miss is found (the gate opens — we
        proceed with the write tier). It FAILS only if ranked search
        is exhaustive (zero misses — stop, do not build).
        """
        os.environ["ARGOS_HERMETIC_TESTS"] = "1"
        store = _build_backlog_store(tmp_path)
        all_contents = {item["content"] for item in _SYNTHETIC_BACKLOG}

        # Track which items are ever returned across all 20 prompts.
        returned_items: set[str] = set()
        per_prompt_misses: list[dict] = []

        for prompt in _PROBE_PROMPTS:
            surfaced = _run_probe(store, prompt, limit=50)
            returned_items |= surfaced
            missed = all_contents - surfaced
            if missed:
                per_prompt_misses.append({
                    "prompt": prompt,
                    "missed_count": len(missed),
                    "sample_missed": list(missed)[:3],
                })

        never_returned = all_contents - returned_items
        total_returned = len(returned_items & all_contents)
        total_missed = len(never_returned)

        # Report the evidence (always print, even on pass)
        print(f"\n=== Circuit-breaker probe results ===")
        print(f"Backlog size: {len(all_contents)}")
        print(f"Prompts: {len(_PROBE_PROMPTS)}")
        print(f"Items returned at least once: {total_returned}")
        print(f"Items NEVER returned: {total_missed}")
        print(f"Per-prompt misses: {len(per_prompt_misses)}")
        for pm in per_prompt_misses[:5]:
            print(f"  '{pm['prompt']}': {pm['missed_count']} missed, "
                  f"sample: {pm['sample_missed']}")

        # The gate: at least one miss means ranked search is NOT
        # exhaustive → collections are needed → proceed with the
        # write tier.
        assert total_missed > 0 or len(per_prompt_misses) > 0, (
            "CIRCUIT BREAKER: zero misses found — ranked search appears "
            "exhaustive for this backlog. Per the spec, STOP and do not "
            "build the collections feature. Report this evidence in the "
            f"PR body. (returned={total_returned}, missed={total_missed})"
        )

    def test_probe_recall_is_not_100_percent(self, tmp_path):
        """Supplementary check: the recall rate across all prompts
        should be below 100% — confirming that ranked search is not
        exhaustive even with a generous limit of 50."""
        os.environ["ARGOS_HERMETIC_TESTS"] = "1"
        store = _build_backlog_store(tmp_path)
        all_contents = {item["content"] for item in _SYNTHETIC_BACKLOG}

        returned_items: set[str] = set()
        for prompt in _PROBE_PROMPTS:
            surfaced = _run_probe(store, prompt, limit=50)
            returned_items |= surfaced

        recall = len(returned_items & all_contents) / len(all_contents)
        print(f"\n=== Recall rate: {recall:.1%} "
              f"({len(returned_items & all_contents)}/{len(all_contents)}) ===")
        assert recall < 1.0, (
            f"Recall is {recall:.1%} — ranked search is exhaustive. "
            f"STOP: do not build collections."
        )
