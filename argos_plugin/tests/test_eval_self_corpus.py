"""Tests for eval_self_corpus.py (Spec 3 — self-corpus retrieval regression).

Run with:
    python -m pytest tests/test_eval_self_corpus.py -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

# Ensure the plugin package is importable.
_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))

# Ensure the eval subdir is importable (eval_self_corpus lives there).
_eval_dir = _plugin_dir / "eval"
if str(_eval_dir) not in sys.path:
    sys.path.insert(0, str(_eval_dir))

import eval_self_corpus as esc  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture: ~20 known records covering the hit-rule cases.
# ---------------------------------------------------------------------------

@pytest.fixture
def fixture_store(tmp_path):
    """Build a temp DuckDB store with ~20 known records.

    Includes: categories spread, one superseded chain of 2, one
    near-duplicate pair, one expired row.
    """
    from store import DuckDBMemoryStore

    store = DuckDBMemoryStore(tmp_path / "fixture.duckdb", user_id="default_user")
    now = datetime.now(timezone.utc)

    records = [
        ("personal_fact", "User is 38 years old and lives in Springfield", ["age", "location"]),
        ("personal_fact", "User works as a software engineer at TechCorp", ["job", "employer"]),
        ("preference", "User prefers dark mode for all code editors", ["ui", "preference"]),
        ("preference", "User likes Python over JavaScript for backend work", ["language", "preference"]),
        ("insight", "User realized that consistent sleep improves focus dramatically", ["sleep", "focus"]),
        ("insight", "User noticed that morning workouts boost productivity all day", ["fitness", "productivity"]),
        ("event", "User started a new job at TechCorp on March 2nd 2026", ["career", "event"]),
        ("event", "User moved to Springfield in January 2026", ["relocation", "event"]),
        ("relationship", "User is married to Sam who is a graphic designer", ["family", "partner"]),
        ("relationship", "User has a close friend named Alex from college", ["friend", "social"]),
        ("goal", "User wants to learn Rust by end of 2026", ["learning", "rust"]),
        ("goal", "User aims to save R50000 for a house deposit", ["finance", "savings"]),
        ("context_note", "User is currently working on a migration project this week", ["project", "work"]),
        ("context_note", "User mentioned a tenant moves out on 31 December 2026", ["tenant", "property"]),
        ("personal_fact", "User drives a Toyota Corolla sedan car", ["vehicle", "car"]),
        # Near-duplicate pair (Jaccard >= 0.75): use dedup=False below.
    ]

    created_ids = []
    for i, (cat, content, tags) in enumerate(records):
        # Stagger created_at across age buckets.
        days_ago = [5, 10, 20, 40, 60, 100, 150, 200, 250, 300][i % 10]
        ts = (now - timedelta(days=days_ago)).isoformat()
        rec = store.remember(category=cat, content=content, tags=tags)
        if rec is not None:
            # Override created_at for age-bucket coverage.
            store.connection.execute(
                "UPDATE memory_records SET created_at = ? WHERE memory_id = ?",
                [ts, rec.memory_id],
            )
            created_ids.append(rec.memory_id)

    # Near-duplicate pair (Jaccard >= 0.75) — saved with dedup=False so
    # the store doesn't collapse them into one record.
    nd1 = store.remember(
        category="personal_fact",
        content="User drives a Toyota Corolla",
        tags=["vehicle"], dedup=False,
    )
    nd2 = store.remember(
        category="personal_fact",
        content="User drives a Toyota Corolla car",
        tags=["vehicle"], dedup=False,
    )
    # Verify the Jaccard is >= 0.75 (5 of 6 tokens overlap = 0.833).
    assert esc._token_jaccard(
        "User drives a Toyota Corolla",
        "User drives a Toyota Corolla car",
    ) >= 0.75

    # Create a superseded chain: save a fact, then update it.
    chain_rec = store.remember(
        category="personal_fact",
        content="User lives at 123 Old Street Springfield",
        tags=["address"],
    )
    if chain_rec is not None:
        updated = store.update_memory(
            memory_id=chain_rec.memory_id,
            content="User lives at 456 New Avenue Springfield",
        )
        # chain_rec.memory_id is the old (superseded) version;
        # updated.memory_id is the current version.

    # Create an expired row (past expires_at).
    expired_rec = store.remember(
        category="context_note",
        content="User has a promo code VALID123 running until Friday",
        tags=["promo", "temporary"],
    )
    if expired_rec is not None:
        past = (now - timedelta(days=10)).isoformat()
        store.connection.execute(
            "UPDATE memory_records SET expires_at = ? WHERE memory_id = ?",
            [past, expired_rec.memory_id],
        )

    # Flush any pending writes so read-only connections from tests can open.
    try:
        store.connection.execute("CHECKPOINT")
    except Exception:
        pass

    yield store, tmp_path / "fixture.duckdb"
    store.close()


class TestSampling:
    def test_sample_respects_limit(self, fixture_store):
        store, db_path = fixture_store
        # Use the store's own connection (DuckDB doesn't allow a second
        # read-only conn while a write conn is open to the same file).
        memories = esc._sample_memories(store.connection, "default_user", limit=5, seed=42)
        assert len(memories) <= 5
        assert len(memories) > 0

    def test_seed_reproducibility(self, fixture_store):
        store, db_path = fixture_store
        s1 = esc._sample_memories(store.connection, "default_user", limit=10, seed=42)
        s2 = esc._sample_memories(store.connection, "default_user", limit=10, seed=42)
        ids1 = [m["memory_id"] for m in s1]
        ids2 = [m["memory_id"] for m in s2]
        assert ids1 == ids2, "Same seed must produce the same sample"

    def test_seed_difference_changes_sample(self, fixture_store):
        store, db_path = fixture_store
        s1 = esc._sample_memories(store.connection, "default_user", limit=10, seed=42)
        s2 = esc._sample_memories(store.connection, "default_user", limit=10, seed=99)
        ids1 = {m["memory_id"] for m in s1}
        ids2 = {m["memory_id"] for m in s2}
        # Different seeds should usually produce different samples.
        assert ids1 != ids2 or len(ids1) < 2

    def test_excludes_expired(self, fixture_store):
        store, db_path = fixture_store
        memories = esc._sample_memories(store.connection, "default_user", limit=100, seed=42)
        for m in memories:
            assert m.get("expires_at") is None or m["expires_at"] > datetime.now(timezone.utc).isoformat()


class TestHitRule:
    def test_hit_exact_id(self):
        target = {"memory_id": "mem-a", "content": "User likes Python"}
        result_ids = ["mem-a", "mem-b"]
        result_contents = {"mem-a": "User likes Python", "mem-b": "Other"}
        assert esc.is_hit(target, result_ids, result_contents, {"mem-a"})

    def test_hit_chain_member(self):
        target = {"memory_id": "mem-new", "content": "User lives at 456 New Ave"}
        result_ids = ["mem-old"]  # old version in the chain
        result_contents = {"mem-old": "User lives at 123 Old Street"}
        chain_ids = {"mem-new", "mem-old"}
        assert esc.is_hit(target, result_ids, result_contents, chain_ids)

    def test_hit_near_duplicate_jaccard(self):
        target = {"memory_id": "mem-x", "content": "User drives a Toyota Corolla"}
        result_ids = ["mem-y"]
        result_contents = {"mem-y": "User drives a Toyota Corolla car"}
        # token Jaccard = 5/6 = 0.833 >= 0.75 for this near-duplicate pair.
        assert esc.is_hit(target, result_ids, result_contents, {"mem-x"})

    def test_miss_unrelated(self):
        target = {"memory_id": "mem-a", "content": "User likes Python"}
        result_ids = ["mem-z"]
        result_contents = {"mem-z": "Completely different content about cooking"}
        assert not esc.is_hit(target, result_ids, result_contents, {"mem-a"})

    def test_jaccard_threshold(self):
        assert esc._token_jaccard("a b c d", "a b c d") == 1.0
        assert esc._token_jaccard("a b c d", "a b c e") == 0.6
        assert esc._token_jaccard("", "something") == 0.0


class TestQueryGeneration:
    def test_direct_template(self):
        mem = {"content": "User is 38 years old and lives in Springfield", "category": "personal_fact", "tags": []}
        probe = esc.generate_probe(mem, "direct")
        assert probe is not None
        assert probe["template"] == "direct"
        assert "what is" in probe["query"]

    def test_temporal_template_requires_date(self):
        mem = {"content": "User started a new job on March 2nd 2026", "category": "event", "tags": []}
        probe = esc.generate_probe(mem, "temporal")
        # date_anchor should detect "on March 2nd".
        if probe is not None:
            assert probe["template"] == "temporal"
            assert "when did" in probe["query"]

    def test_temporal_template_skips_non_dated(self):
        mem = {"content": "User likes Python programming", "category": "preference", "tags": []}
        probe = esc.generate_probe(mem, "temporal")
        assert probe is None

    def test_preference_template(self):
        mem = {"content": "User prefers dark mode for all editors", "category": "preference", "tags": []}
        probe = esc.generate_probe(mem, "preference")
        assert probe is not None
        assert "prefer" in probe["query"]

    def test_preference_template_skips_non_preference(self):
        mem = {"content": "User is 38 years old", "category": "personal_fact", "tags": []}
        probe = esc.generate_probe(mem, "preference")
        assert probe is None

    def test_entity_template_requires_tags(self):
        mem = {"content": "User works at TechCorp as engineer", "category": "personal_fact", "tags": ["employer"]}
        probe = esc.generate_probe(mem, "entity")
        assert probe is not None
        mem_no_tags = {"content": "User works at TechCorp", "category": "personal_fact", "tags": []}
        assert esc.generate_probe(mem_no_tags, "entity") is None

    def test_negation_template(self):
        mem = {"content": "User is married to Sam", "category": "relationship", "tags": []}
        probe = esc.generate_probe(mem, "negation")
        assert probe is not None
        assert "NOT" in probe["query"]

    def test_synonym_template(self):
        mem = {"content": "User drives a car to work", "category": "personal_fact", "tags": []}
        probe = esc.generate_probe(mem, "synonym")
        assert probe is not None
        assert "vehicle" in probe["query"]  # car → vehicle

    def test_synonym_template_no_synonym_returns_none(self):
        mem = {"content": "User likes astronomy", "category": "insight", "tags": []}
        probe = esc.generate_probe(mem, "synonym")
        assert probe is None  # no synonymable token


class TestConfigHash:
    def test_config_hash_deterministic(self, tmp_path):
        db = tmp_path / "test.duckdb"
        db.parent.mkdir(parents=True, exist_ok=True)
        db.write_text("")  # placeholder
        h1 = esc._config_hash(db, "BAAI/bge-small-en-v1.5")
        h2 = esc._config_hash(db, "BAAI/bge-small-en-v1.5")
        assert h1 == h2
        assert len(h1) == 16

    def test_config_hash_changes_with_model(self, tmp_path):
        db = tmp_path / "test.duckdb"
        db.write_text("")
        h1 = esc._config_hash(db, "model-a")
        h2 = esc._config_hash(db, "model-b")
        assert h1 != h2


class TestResumeAndBaseline:
    def test_completed_ids(self, tmp_path):
        out = tmp_path / "out.jsonl"
        out.write_text(
            json.dumps({"target_memory_id": "mem-1", "hit": True}) + "\n"
            + json.dumps({"target_memory_id": "mem-2", "hit": False}) + "\n"
            + json.dumps({"run_summary": {"overall": {"recall@20": 0.9}}}) + "\n"
        )
        done = esc._completed_ids(out)
        assert done == {"mem-1", "mem-2"}

    def test_completed_ids_empty(self, tmp_path):
        out = tmp_path / "nonexistent.jsonl"
        assert esc._completed_ids(out) == set()

    def test_verdict_pass_on_improvement(self):
        # ladder max-k = 20 → recall@20 is the judged metric.
        current = {"ladder": [5, 20], "overall": {"recall@20": 0.92, "mrr": 0.6}}
        baseline = {"ladder": [5, 20], "overall": {"recall@20": 0.90, "mrr": 0.6}}
        v = esc._verdict(current, baseline)
        assert v.startswith("PASS")

    def test_verdict_fail_on_recall_drop(self):
        current = {"ladder": [5, 20], "overall": {"recall@20": 0.85, "mrr": 0.6}}
        baseline = {"ladder": [5, 20], "overall": {"recall@20": 0.92, "mrr": 0.6}}
        v = esc._verdict(current, baseline)
        assert v.startswith("FAIL")
        assert "recall@20" in v and "7.0pp" in v

    def test_verdict_fail_on_mrr_drop(self):
        # Recall unchanged, MRR drops > 0.01 → FAIL under the shared threshold
        # (the old 3pp verdict ignored MRR entirely — issue #21).
        current = {"ladder": [5, 20], "overall": {"recall@20": 0.92, "mrr": 0.55}}
        baseline = {"ladder": [5, 20], "overall": {"recall@20": 0.92, "mrr": 0.60}}
        v = esc._verdict(current, baseline)
        assert v.startswith("FAIL")
        assert "MRR" in v

    def test_verdict_borderline_pass_within_half_pp(self):
        # -0.4pp recall, MRR unchanged → within the 0.5pp overall tolerance → PASS.
        current = {"ladder": [5, 20], "overall": {"recall@20": 0.916, "mrr": 0.60}}
        baseline = {"ladder": [5, 20], "overall": {"recall@20": 0.920, "mrr": 0.60}}
        v = esc._verdict(current, baseline)
        assert v.startswith("PASS")

    def test_verdict_borderline_fail_just_over_half_pp(self):
        # -0.6pp recall → just over the 0.5pp overall tolerance → FAIL
        # (the old 3pp verdict would have passed this — issue #21).
        current = {"ladder": [5, 20], "overall": {"recall@20": 0.914, "mrr": 0.60}}
        baseline = {"ladder": [5, 20], "overall": {"recall@20": 0.920, "mrr": 0.60}}
        v = esc._verdict(current, baseline)
        assert v.startswith("FAIL")


class TestSuppressRetrieval:
    def test_retrieval_count_unchanged(self, fixture_store):
        """suppress_retrieval=True must not inflate retrieval_count."""
        store, db_path = fixture_store
        from store import DuckDBMemoryStore
        # Snapshot retrieval counts before.
        before = store.connection.execute(
            "SELECT memory_id, retrieval_count FROM memory_records"
        ).fetchall()
        before_map = {mid: cnt for mid, cnt in before}
        # Run a probe (direct search with suppress_retrieval=True).
        mem = {"memory_id": before[0][0], "content": "test", "category": "personal_fact"}
        probe = {"query": "what is user age?", "template": "direct"}
        chain_ids = esc._build_chain_set(store, mem["memory_id"])
        esc._run_probe(store, probe, mem, [5, 20, 96], chain_ids)
        after = store.connection.execute(
            "SELECT memory_id, retrieval_count FROM memory_records"
        ).fetchall()
        after_map = {mid: cnt for mid, cnt in after}
        # Every record's retrieval_count must be unchanged.
        for mid, cnt in before_map.items():
            assert after_map.get(mid) == cnt, f"retrieval_count changed for {mid}"


class TestNoLLMByDefault:
    def test_no_llm_without_flag(self, fixture_store, monkeypatch):
        """Without --llm-paraphrase, the LLM client must never be called.

        The script only imports ``agent.auxiliary_client.call_llm`` inside
        the ``if args.llm_paraphrase`` block.  Verify that blocking the
        import does not break probe generation or retrieval.
        """
        store, db_path = fixture_store

        # Block the agent.auxiliary_client module from being imported.
        import importlib
        monkeypatch.setitem(sys.modules, "agent.auxiliary_client", None)
        monkeypatch.setitem(sys.modules, "agent", None)

        # generate_probe and _run_probe must work without any LLM.
        real_id = store.connection.execute(
            "SELECT memory_id, content, category FROM memory_records LIMIT 1").fetchone()
        mem = {"memory_id": real_id[0], "content": real_id[1], "category": real_id[2], "tags": []}
        probe = esc.generate_probe(mem, "direct")
        assert probe is not None
        chain_ids = esc._build_chain_set(store, mem["memory_id"])
        result = esc._run_probe(store, probe, mem, [5, 20], chain_ids)
        assert "hit" in result  # ran successfully without LLM
