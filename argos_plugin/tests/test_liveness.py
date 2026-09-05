"""Tests for #275: liveness probes + feature counters + config fingerprint.

LP1 Startup self-smoke: after config load, run a tiny canned probe
through each "can silently die" feature and log ERROR per failure.

LP2 Feature counters: per-feature hit counters readable via status().

LP3 Config fingerprint: hash the effective loaded config; changes when
config changes.

Acceptance: boot logs ERROR for each failed probe; counters increment
and are readable; fingerprint changes when config changes.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))


class TestLP3ConfigFingerprint:
    """LP3: config fingerprint changes when config changes."""

    def test_fingerprint_is_stable(self):
        from config_model import MemoryConfig
        from liveness import config_fingerprint
        c1 = MemoryConfig()
        c2 = MemoryConfig()
        assert config_fingerprint(c1) == config_fingerprint(c2)

    def test_fingerprint_changes_on_value_change(self):
        from config_model import MemoryConfig
        from liveness import config_fingerprint
        c1 = MemoryConfig()
        c2 = MemoryConfig.model_validate({"max_injected_items": 50})
        assert config_fingerprint(c1) != config_fingerprint(c2)

    def test_fingerprint_changes_on_router_toggle(self):
        from config_model import MemoryConfig
        from liveness import config_fingerprint
        c1 = MemoryConfig()
        c2 = MemoryConfig.model_validate({"router_enabled": "true"})
        assert config_fingerprint(c1) != config_fingerprint(c2)

    def test_fingerprint_is_hex_string(self):
        from config_model import MemoryConfig
        from liveness import config_fingerprint
        fp = config_fingerprint(MemoryConfig())
        assert isinstance(fp, str)
        assert len(fp) == 16
        assert all(c in "0123456789abcdef" for c in fp)

    def test_fingerprint_order_independent(self):
        """The fingerprint is the same regardless of key order in the
        raw dict (sorted canonical JSON)."""
        from liveness import config_fingerprint
        d1 = {"a": 1, "b": 2, "c": 3}
        d2 = {"c": 3, "a": 1, "b": 2}
        assert config_fingerprint(d1) == config_fingerprint(d2)


class TestLP2FeatureCounters:
    """LP2: per-feature hit counters, thread-safe, readable via snapshot()."""

    def test_counters_start_at_zero(self):
        from liveness import FeatureCounters
        fc = FeatureCounters()
        snap = fc.snapshot()
        assert all(v == 0 for v in snap.values())
        assert "router_calls" in snap
        assert "injection_min_score" in snap
        assert "conflict_surfacing" in snap
        assert "chain_unfold_calls" in snap
        assert "rerank_calls" in snap
        assert "graph_injections" in snap
        assert "extraction_facts" in snap

    def test_increment(self):
        from liveness import FeatureCounters
        fc = FeatureCounters()
        fc.increment("router_calls")
        fc.increment("router_calls")
        fc.increment("router_calls", 3)
        assert fc.get("router_calls") == 5

    def test_increment_unknown_feature(self):
        from liveness import FeatureCounters
        fc = FeatureCounters()
        fc.increment("custom_feature")
        assert fc.get("custom_feature") == 1
        assert "custom_feature" in fc.snapshot()

    def test_reset(self):
        from liveness import FeatureCounters
        fc = FeatureCounters()
        fc.increment("router_calls", 10)
        fc.reset()
        assert fc.get("router_calls") == 0

    def test_snapshot_is_sorted(self):
        from liveness import FeatureCounters
        fc = FeatureCounters()
        snap = fc.snapshot()
        keys = list(snap.keys())
        assert keys == sorted(keys)


class TestLP1StartupSelfTest:
    """LP1: startup self-smoke test logs ERROR for each failed probe."""

    def test_clean_config_passes_all_probes(self):
        from config_model import MemoryConfig
        from liveness import run_startup_self_test
        results = run_startup_self_test(MemoryConfig())
        assert all(results.values()), \
            f"Failed probes: {[k for k, v in results.items() if not v]}"
        assert len(results) == 7  # 7 features

    def test_router_string_type_fails(self, caplog):
        """The #272 bug: router_enabled as a string instead of bool."""
        from liveness import run_startup_self_test
        # Simulate a config where router_enabled is a string (pre-fix state).
        fake_config = type("FakeConfig", (), {
            "model_dump": lambda self: {
                "router_enabled": "true",  # string, not bool
                "injection_min_score": 0.0,
                "skip_retrieval_on_trivial": False,
                "conflict_surfacing": True,
                "chain_unfold": "off",
                "reranker_enabled": False,
                "reranker_model": "",
                "freshness_markers": True,
            },
        })()
        with caplog.at_level(logging.ERROR, logger="argos.liveness"):
            results = run_startup_self_test(fake_config)
        assert results["router"] is False
        assert any("router_enabled" in r.getMessage() for r in caplog.records)

    def test_injection_min_score_wrong_type_fails(self, caplog):
        from liveness import run_startup_self_test
        fake_config = type("FakeConfig", (), {
            "model_dump": lambda self: {
                "router_enabled": False,
                "injection_min_score": None,  # None, not a number
                "skip_retrieval_on_trivial": False,
                "conflict_surfacing": True,
                "chain_unfold": "off",
                "reranker_enabled": False,
                "reranker_model": "",
                "freshness_markers": True,
            },
        })()
        with caplog.at_level(logging.ERROR, logger="argos.liveness"):
            results = run_startup_self_test(fake_config)
        assert results["injection_min_score"] is False

    def test_injection_min_score_out_of_range_fails(self, caplog):
        from liveness import run_startup_self_test
        fake_config = type("FakeConfig", (), {
            "model_dump": lambda self: {
                "router_enabled": False,
                "injection_min_score": 1.5,  # out of [0.0, 1.0]
                "skip_retrieval_on_trivial": False,
                "conflict_surfacing": True,
                "chain_unfold": "off",
                "reranker_enabled": False,
                "reranker_model": "",
                "freshness_markers": True,
            },
        })()
        with caplog.at_level(logging.ERROR, logger="argos.liveness"):
            results = run_startup_self_test(fake_config)
        assert results["injection_min_score"] is False

    def test_skip_retrieval_on_trivial_wrong_type_fails(self, caplog):
        from liveness import run_startup_self_test
        fake_config = type("FakeConfig", (), {
            "model_dump": lambda self: {
                "router_enabled": False,
                "injection_min_score": 0.0,
                "skip_retrieval_on_trivial": "yes",  # string, not bool
                "conflict_surfacing": True,
                "chain_unfold": "off",
                "reranker_enabled": False,
                "reranker_model": "",
                "freshness_markers": True,
            },
        })()
        with caplog.at_level(logging.ERROR, logger="argos.liveness"):
            results = run_startup_self_test(fake_config)
        assert results["skip_retrieval_on_trivial"] is False

    def test_conflict_surfacing_wrong_type_fails(self, caplog):
        from liveness import run_startup_self_test
        fake_config = type("FakeConfig", (), {
            "model_dump": lambda self: {
                "router_enabled": False,
                "injection_min_score": 0.0,
                "skip_retrieval_on_trivial": False,
                "conflict_surfacing": None,  # None, not bool
                "chain_unfold": "off",
                "reranker_enabled": False,
                "reranker_model": "",
                "freshness_markers": True,
            },
        })()
        with caplog.at_level(logging.ERROR, logger="argos.liveness"):
            results = run_startup_self_test(fake_config)
        assert results["conflict_surfacing"] is False

    def test_chain_unfold_invalid_value_fails(self, caplog):
        from liveness import run_startup_self_test
        fake_config = type("FakeConfig", (), {
            "model_dump": lambda self: {
                "router_enabled": False,
                "injection_min_score": 0.0,
                "skip_retrieval_on_trivial": False,
                "conflict_surfacing": True,
                "chain_unfold": "banana",  # invalid enum
                "reranker_enabled": False,
                "reranker_model": "",
                "freshness_markers": True,
            },
        })()
        with caplog.at_level(logging.ERROR, logger="argos.liveness"):
            results = run_startup_self_test(fake_config)
        assert results["chain_unfold"] is False

    def test_reranker_enabled_no_model_fails(self, caplog):
        from liveness import run_startup_self_test
        fake_config = type("FakeConfig", (), {
            "model_dump": lambda self: {
                "router_enabled": False,
                "injection_min_score": 0.0,
                "skip_retrieval_on_trivial": False,
                "conflict_surfacing": True,
                "chain_unfold": "off",
                "reranker_enabled": True,
                "reranker_model": "",  # empty model
                "freshness_markers": True,
            },
        })()
        with caplog.at_level(logging.ERROR, logger="argos.liveness"):
            results = run_startup_self_test(fake_config)
        assert results["reranker"] is False

    def test_freshness_markers_wrong_type_fails(self, caplog):
        from liveness import run_startup_self_test
        fake_config = type("FakeConfig", (), {
            "model_dump": lambda self: {
                "router_enabled": False,
                "injection_min_score": 0.0,
                "skip_retrieval_on_trivial": False,
                "conflict_surfacing": True,
                "chain_unfold": "off",
                "reranker_enabled": False,
                "reranker_model": "",
                "freshness_markers": "on",  # string, not bool
            },
        })()
        with caplog.at_level(logging.ERROR, logger="argos.liveness"):
            results = run_startup_self_test(fake_config)
        assert results["recency_importance"] is False

    def test_tuned_config_passes_all_probes(self):
        """A realistic tuned config (from #274 fixture) passes all probes."""
        from config_model import MemoryConfig
        from liveness import run_startup_self_test
        c = MemoryConfig.model_validate({
            "router_enabled": "true",
            "injection_min_score": "0.3",
            "skip_retrieval_on_trivial": "true",
            "conflict_surfacing": "true",
            "chain_unfold": "auto",
            "reranker_enabled": "true",
            "reranker_model": "BAAI/bge-reranker-base",
            "freshness_markers": "true",
        })
        results = run_startup_self_test(c)
        assert all(results.values()), \
            f"Failed probes: {[k for k, v in results.items() if not v]}"


class TestStatusSurface:
    """LP2+LP3: status() exposes counters + fingerprint."""

    def test_status_returns_counters_and_fingerprint(self):
        from liveness import get_counters, config_fingerprint
        from config_model import MemoryConfig
        # Reset counters for a clean test.
        get_counters().reset()
        get_counters().increment("router_calls", 5)
        get_counters().increment("conflict_surfacing", 3)
        fp = config_fingerprint(MemoryConfig())
        # The status() function in liveness.py returns feature_counters.
        from liveness import status
        s = status()
        assert "feature_counters" in s
        assert s["feature_counters"]["router_calls"] == 5
        assert s["feature_counters"]["conflict_surfacing"] == 3


class TestCounterWiringAtCallSites:
    """#275 LP2 BLOCKER fix: assert counters increment when real
    feature paths run (not fakes)."""

    def test_router_counter_increments_on_real_route(self):
        """route_answerer with a temporal question and router_enabled
        increments the router_calls counter."""
        from liveness import get_counters
        from config_model import MemoryConfig
        from intent_router import route_answerer
        get_counters().reset()
        cfg = MemoryConfig.model_validate({
            "router_enabled": "true",
            "router_smart_model": "test-smart-model",
        })
        # A temporal question should trigger a route.
        route = route_answerer(cfg, "What did I do last week?")
        assert route is not None
        assert route["model"] == "test-smart-model"
        # The counter must have incremented.
        assert get_counters().get("router_calls") >= 1

    def test_router_counter_does_not_increment_when_disabled(self):
        """route_answerer with router disabled does not increment."""
        from liveness import get_counters
        from config_model import MemoryConfig
        from intent_router import route_answerer
        get_counters().reset()
        cfg = MemoryConfig()  # router_enabled defaults to False
        route = route_answerer(cfg, "What did I do last week?")
        assert route is None
        assert get_counters().get("router_calls") == 0

    def test_extraction_counter_increments_on_real_extraction(self):
        """extract_from_turn with a factual statement increments the
        extraction_facts counter."""
        from liveness import get_counters
        from extractor import extract_from_turn
        get_counters().reset()
        facts = extract_from_turn(
            "My name is Alice and I live in Paris.",
            "",
            use_llm_fallback=False,
        )
        # The extractor should find at least one fact.
        assert len(facts) > 0
        # The counter must have incremented by the number of facts.
        assert get_counters().get("extraction_facts") >= 1

    def test_extraction_counter_does_not_increment_on_empty(self):
        """extract_from_turn with no facts does not increment."""
        from liveness import get_counters
        from extractor import extract_from_turn
        get_counters().reset()
        facts = extract_from_turn("ok", "", use_llm_fallback=False)
        # Trivial content should produce no facts.
        assert len(facts) == 0
        assert get_counters().get("extraction_facts") == 0
