"""#364: traversal-engagement counters for the run_eval_provider A/B.

The #139 "traversal_on vs baseline" A/B produced byte-identical metrics.
That is the signature of a no-op, not of a flat feature: traversal walks
TYPED/LLM relations only and requires a non-concept seed, while the harness
builds a regex-only graph (generic edges, concept nodes). These tests pin
the diagnostic that lets the harness tell the two apart.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))
if str(_plugin_dir.parent) not in sys.path:
    sys.path.insert(0, str(_plugin_dir.parent))
_eval_dir = _plugin_dir / "eval"
if str(_eval_dir) not in sys.path:
    sys.path.insert(0, str(_eval_dir))

import run_eval_provider as rep  # noqa: E402


class TestSummarizeEngagement:
    """Pure aggregation — no Kuzu needed."""

    def test_all_zero_is_reported_as_never_engaged(self):
        rows = [{"terms": 3, "seeds_resolved": 0, "non_concept_seeds": 0,
                 "traversal_ids": 0, "engaged": False}] * 4
        out = rep.summarize_engagement(rows)
        assert out == {"n_queries": 4, "seeds_resolved": 0,
                       "non_concept_seeds": 0, "engaged": 0,
                       "engaged_fraction": 0.0}

    def test_counts_queries_per_stage(self):
        rows = [
            # seeds but only concepts -> gate closes
            {"seeds_resolved": 2, "non_concept_seeds": 0, "traversal_ids": 0, "engaged": False},
            # specific seed, traversal fires
            {"seeds_resolved": 1, "non_concept_seeds": 1, "traversal_ids": 3, "engaged": True},
            # nothing grounded
            {"seeds_resolved": 0, "non_concept_seeds": 0, "traversal_ids": 0, "engaged": False},
            # error row from a failed diagnostic call must not crash aggregation
            {"error": "boom"},
        ]
        out = rep.summarize_engagement(rows)
        assert out["n_queries"] == 4
        assert out["seeds_resolved"] == 2
        assert out["non_concept_seeds"] == 1
        assert out["engaged"] == 1
        assert out["engaged_fraction"] == 0.25

    def test_empty(self):
        assert rep.summarize_engagement([]) == {
            "n_queries": 0, "seeds_resolved": 0, "non_concept_seeds": 0,
            "engaged": 0, "engaged_fraction": 0.0}


class TestArmConfig:
    def test_base_arm_config_pins_traversal_off(self, tmp_path):
        """MemoryConfig defaults graph_traversal_enabled=True; an unpinned
        baseline would be config-identical to traversal_on and the A/B
        would compare an arm against itself."""
        import json
        snap = tmp_path / "snap.duckdb"
        snap.write_bytes(b"")
        home = rep.build_arm_home(snap, rep.ARMS["baseline"], tmp_path / "base")
        cfg = json.loads((home / "hybrid_memory.json").read_text(encoding="utf-8"))
        assert cfg["graph_traversal_enabled"] == "false"
        home_on = rep.build_arm_home(snap, rep.ARMS["traversal_on"], tmp_path / "on")
        cfg_on = json.loads((home_on / "hybrid_memory.json").read_text(encoding="utf-8"))
        assert cfg_on["graph_traversal_enabled"] == "true"


class TestGraphTraversalEngagement:
    """KuzuGraphStore.traversal_engagement mirrors traversal_memory_ids gates."""

    def test_generic_concept_graph_never_engages(self, tmp_path):
        """The regex-only shape (generic edges onto concept nodes): seeds
        resolve, the specific-seed gate closes, no ids — the counters must
        expose that instead of the arm looking merely 'flat'."""
        from graph import KuzuGraphStore

        graph = KuzuGraphStore(tmp_path / "generic_kuzu", user_id="test_user")
        graph.add_relationship(
            "user", "person", "related_to", "weekend plans", "concept",
            {"memory_id": "mem-plans", "extractor": "graph_patterns"})
        graph.add_relationship(
            "memory:mem-plans", "memory", "mentions", "weekend plans", "concept",
            {"memory_id": "mem-plans", "extractor": "graph_patterns"})
        eng = graph.traversal_engagement("weekend plans", depth=2)
        assert eng["terms"] >= 1
        assert eng["seeds_resolved"] >= 1, eng
        assert eng["non_concept_seeds"] == 0, eng
        assert eng["engaged"] is False, eng
        assert eng["traversal_ids"] == 0
        # engagement must agree with the production path
        assert graph.traversal_memory_ids("weekend plans", depth=2) == []
        graph.close()

    def test_typed_graph_engages(self, tmp_path):
        from graph import KuzuGraphStore

        graph = KuzuGraphStore(tmp_path / "typed_kuzu", user_id="test_user")
        graph.add_relationship(
            "user", "person", "has_wife", "Alex", "person",
            {"memory_id": "mem-wife", "extractor": "llm"})
        graph.add_relationship(
            "Alex", "person", "works_at", "TechCorp", "organization",
            {"memory_id": "mem-alex-work", "extractor": "llm"})
        eng = graph.traversal_engagement("Alex", depth=2)
        assert eng["seeds_resolved"] >= 1
        assert eng["non_concept_seeds"] >= 1
        assert eng["engaged"] is True, eng
        assert eng["traversal_ids"] == len(
            graph.traversal_memory_ids("Alex", depth=2))
        graph.close()

    def test_no_terms(self, tmp_path):
        from graph import KuzuGraphStore

        graph = KuzuGraphStore(tmp_path / "empty_kuzu", user_id="test_user")
        eng = graph.traversal_engagement("the and", depth=2)
        assert eng == {"terms": 0, "seeds_resolved": 0, "non_concept_seeds": 0,
                       "traversal_ids": 0, "engaged": False}
        graph.close()
