"""Config parity test for #244.

Asserts that MemoryConfig() with no args produces the same effective
config as today's defaults — a default drift shows up as a test failure,
not a silent prod change.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))


class TestConfigParity:
    """MemoryConfig defaults must match the pre-#244 slurp defaults."""

    def test_storage_defaults(self):
        from config_model import MemoryConfig
        c = MemoryConfig()
        assert c.database_filename == "hybrid_memory.duckdb"
        assert c.graph_dirname == "hybrid_memory_kuzu"
        assert c.storage_mode == "shared_service"
        assert c.local_embedding_model == "BAAI/bge-small-en-v1.5"

    def test_injection_defaults(self):
        from config_model import MemoryConfig
        c = MemoryConfig()
        assert c.max_injected_items == 20
        assert c.inject_content_char_cap == 800
        assert c.freshness_markers is True
        assert c.skip_retrieval_on_trivial is False
        assert c.injection_min_score == 0.0
        assert c.chronological_injection is False
        assert c.date_anchor_rerank is False
        assert c.history_at_current_time is True

    def test_extraction_review_defaults(self):
        from config_model import MemoryConfig
        c = MemoryConfig()
        assert c.auto_extract is True
        assert c.llm_fallback is True
        assert c.extraction_shadow_diff is False
        assert c.auto_review is True
        assert c.confirmation_surfacing is True
        assert c.extraction_dup_threshold == 0.88
        assert c.stale_review_sweep_enabled is True
        assert c.stale_review_interval_min == 15
        assert c.stale_review_min_age_min == 30
        assert c.stale_review_max_batch == 25

    def test_graph_defaults(self):
        from config_model import MemoryConfig
        c = MemoryConfig()
        assert c.graph_aware_retrieval is True
        assert c.graph_retrieval_boost == 0.0
        assert c.graph_inject_candidates is False
        assert c.graph_boost_min_similarity == 0.15
        assert c.graph_traversal_enabled is True
        assert c.graph_traversal_depth == 2
        assert c.graph_traversal_boost == 0.60
        assert c.graph_ppr_enabled is False
        assert c.graph_ppr_damping == 0.5
        assert c.graph_ppr_boost == 0.0
        assert c.alias_expansion_boost == 0.7
        assert c.conflict_surfacing is True

    def test_chain_unfold_defaults(self):
        from config_model import MemoryConfig
        c = MemoryConfig()
        assert c.chain_unfold == "off"
        assert c.chain_unfold_min_similarity == 0.30
        assert c.chain_unfold_arc_min_similarity == 0.15
        assert c.chain_max_versions == 3
        assert c.chain_max_inject == 150
        assert c.chain_unfold_top_k == 3
        assert c.chain_unfold_query_fallback is False

    def test_consolidation_dedup_defaults(self):
        from config_model import MemoryConfig
        c = MemoryConfig()
        assert c.consolidation_enabled is False
        assert c.consolidation_min_age_days == 30
        assert c.consolidation_max_actions == 25
        assert c.consolidation_auto_apply is True
        assert c.duplicate_min_similarity == 0.88
        assert c.duplicate_semantic_max_pairs == 20000

    def test_reranker_defaults(self):
        from config_model import MemoryConfig
        c = MemoryConfig()
        assert c.reranker_enabled is False
        assert c.reranker_model == "BAAI/bge-reranker-base"
        assert c.reranker_top_n == 10

    def test_context_query_defaults(self):
        from config_model import MemoryConfig
        c = MemoryConfig()
        assert c.context_aware_retrieval is True
        assert c.phrase_lift_alpha == 0.0
        assert c.phrase_lift_pool == 200
        assert c.context_window_size == 3
        assert c.context_max_chars == 500
        assert c.query_expansion_enabled is True
        assert c.query_expansion_similarity_floor == 0.3

    def test_llm_endpoint_defaults(self):
        from config_model import MemoryConfig
        c = MemoryConfig()
        assert c.llm_model == ""
        assert c.llm_provider == ""
        assert c.extraction_llm_model == ""
        assert c.extraction_llm_provider == ""
        assert c.answering_llm_model == ""
        assert c.answering_llm_provider == ""
        assert c.role_alias_llm_fallback is True

    def test_watcher_defaults(self):
        from config_model import MemoryConfig
        c = MemoryConfig()
        assert c.watcher_enabled is False
        assert c.watcher_scan_roots == []
        assert c.watcher_interval_min == 30

    def test_deployment_defaults(self):
        from config_model import MemoryConfig
        c = MemoryConfig()
        assert c.deployment_mode == "cloud_pilot"
        assert c.data_residency == "cloud"

    def test_expiry_defaults(self):
        from config_model import MemoryConfig
        c = MemoryConfig()
        assert c.expiry_enabled is False
        assert c.expiry_ttl_days == '{"context_note":30,"event":180,"goal":180}'
        assert c.expiry_default_days == 90
        assert c.expiry_auto_suggest is False

    def test_distillation_defaults(self):
        from config_model import MemoryConfig
        c = MemoryConfig()
        assert c.distillation_enabled is False
        assert c.distillation_min_new_records == 20
        assert c.distillation_cooldown_hours == 24
        assert c.distillation_max_records_per_run == 100
        assert c.distillation_max_calls == 10

    def test_lifecycle_defaults(self):
        from config_model import MemoryConfig
        c = MemoryConfig()
        assert c.archive_enabled is False
        assert c.archive_after_days == 180
        assert c.forget_enabled is False
        assert c.forget_after_days == 365
        assert c.rollup_enabled is False
        assert c.rollup_interval_days == 30
        assert c.rollup_max_records_per_run == 100

    def test_egress_defaults(self):
        from config_model import MemoryConfig
        c = MemoryConfig()
        assert c.local_only is False
        assert c.external_sources_require_confirmation is True

    def test_scale_defaults(self):
        from config_model import MemoryConfig
        c = MemoryConfig()
        assert c.scale_warn_latency_ms == 300.0
        assert c.scale_warn_records == 5000

    def test_misc_defaults(self):
        from config_model import MemoryConfig
        c = MemoryConfig()
        assert c.evidence_retention == "full"
        assert c.entity_aliases == ""
        assert c.role_words == ""
        assert c.acl is None


class TestConfigFailSoft:
    """Fail-soft behaviour matches the pre-#244 slurp."""

    def test_bool_string_coercion(self):
        from config_model import MemoryConfig
        c = MemoryConfig.model_validate({"auto_extract": "on"})
        assert c.auto_extract is True
        c2 = MemoryConfig.model_validate({"auto_extract": "banana"})
        assert c2.auto_extract is False

    def test_int_clamping(self):
        from config_model import MemoryConfig
        c = MemoryConfig.model_validate({"max_injected_items": "999"})
        assert c.max_injected_items == 512  # clamped to le=512
        c2 = MemoryConfig.model_validate({"max_injected_items": "not_a_number"})
        assert c2.max_injected_items == 20  # fallback to default

    def test_float_clamping(self):
        from config_model import MemoryConfig
        c = MemoryConfig.model_validate({"graph_retrieval_boost": "2.0"})
        assert c.graph_retrieval_boost == 0.5  # clamped to le=0.5

    def test_extra_forbid(self):
        from config_model import MemoryConfig
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            MemoryConfig.model_validate({"typo_key": "value"})

    def test_get_backward_compat(self):
        from config_model import MemoryConfig
        c = MemoryConfig()
        assert c.get("max_injected_items") == 20
        assert c.get("unknown_key", "default") == "default"
        assert "database_filename" in c
        assert "unknown_key" not in c

    def test_watcher_scan_roots_string_to_list(self):
        from config_model import MemoryConfig
        c = MemoryConfig.model_validate({"watcher_scan_roots": "/a, /b, /c"})
        assert c.watcher_scan_roots == ["/a", "/b", "/c"]

    def test_chain_unfold_enum(self):
        from config_model import MemoryConfig
        c = MemoryConfig.model_validate({"chain_unfold": "ALWAYS"})
        assert c.chain_unfold == "always"
        c2 = MemoryConfig.model_validate({"chain_unfold": "banana"})
        assert c2.chain_unfold == "off"
