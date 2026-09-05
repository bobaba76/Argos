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
        assert c.chain_unfold == "auto"
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


class TestLegacyKeysSurvival:
    """#244 review blockers: router/backup keys must survive config load.

    The live hybrid_memory.json carries keys declared in config_schema.py
    that the model must represent. If these keys are missing from the model,
    .get() returns None and the router is silently disabled.
    """

    def test_router_keys_in_model(self):
        """All router_* keys from config_schema.py are model fields."""
        from config_model import MemoryConfig
        fields = set(MemoryConfig.model_fields.keys())
        for key in ("router_enabled", "router_smart_model", "router_smart_provider",
                     "router_default_model", "router_default_provider",
                     "router_subcall_enabled", "router_temporal_threshold",
                     "router_multihop_threshold"):
            assert key in fields, f"{key} missing from MemoryConfig fields"

    def test_backup_keys_in_model(self):
        """All backup_* keys from config_schema.py are model fields."""
        from config_model import MemoryConfig
        fields = set(MemoryConfig.model_fields.keys())
        for key in ("backup_enabled", "backup_dst_root", "backup_retention_snapshots"):
            assert key in fields, f"{key} missing from MemoryConfig fields"

    def test_router_values_survive_load(self):
        """Router config values survive a round-trip through MemoryConfig."""
        from config_model import MemoryConfig
        c = MemoryConfig.model_validate({
            "router_enabled": "true",
            "router_smart_model": "deepseek/deepseek-v4-pro",
            "router_smart_provider": "openrouter",
            "router_default_model": "deepseek-v4-flash",
            "router_default_provider": "opencode-go",
            "router_subcall_enabled": "true",
        })
        assert c.router_enabled is True
        assert c.router_smart_model == "deepseek/deepseek-v4-pro"
        assert c.router_smart_provider == "openrouter"
        assert c.router_default_model == "deepseek-v4-flash"
        assert c.router_default_provider == "opencode-go"
        assert c.router_subcall_enabled is True

    def test_router_get_returns_configured_value(self):
        """MemoryConfig.get() returns the configured router value, not None."""
        from config_model import MemoryConfig
        c = MemoryConfig.model_validate({"router_enabled": "true", "router_smart_model": "gpt-5"})
        assert c.get("router_enabled") is True
        assert c.get("router_smart_model") == "gpt-5"

    def test_unknown_keys_do_not_wipe_config(self):
        """#244 review blocker 1: unknown keys in the raw file must NOT
        cause the whole config to revert to defaults.

        provider_core._load_config filters raw keys to known model fields
        before validation, so extra keys are logged + dropped, not fatal.
        """
        import json
        import tempfile
        from pathlib import Path
        from provider_core import _load_config

        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "hybrid_memory.json"
            # A realistic config with tuned values + unknown keys.
            config_path.write_text(json.dumps({
                "max_injected_items": 96,
                "skip_retrieval_on_trivial": "true",
                "injection_min_score": 0.3,
                "router_enabled": "true",
                "router_smart_model": "deepseek-v4-pro",
                "some_future_key": "value",
                "tenants": {"acme": {"review_mode": "auto"}},
            }), encoding="utf-8")

            cfg = _load_config(str(tmp))

            # Tuned values must survive (NOT reverted to defaults).
            assert cfg.max_injected_items == 96
            assert cfg.skip_retrieval_on_trivial is True
            assert cfg.injection_min_score == 0.3
            assert cfg.router_enabled is True
            assert cfg.router_smart_model == "deepseek-v4-pro"

    def test_backup_nested_dict_survives(self):
        """The backup key as a nested dict survives in the model."""
        from config_model import MemoryConfig
        c = MemoryConfig.model_validate({
            "backup": {"dst_root": "/mnt/backup", "retention_snapshots": 10},
        })
        assert c.backup == {"dst_root": "/mnt/backup", "retention_snapshots": 10}

    def test_t1_schema_model_parity(self):
        """T1: every key declared in config_schema.py is a MemoryConfig field.

        Catches the 9-key mismatch that caused the router to be silently
        disabled. If a new key is added to config_schema.py without adding
        it to MemoryConfig, this test fails at CI.
        """
        import re
        from pathlib import Path
        from config_model import MemoryConfig

        schema_path = Path(__file__).resolve().parent.parent / "config_schema.py"
        schema_text = schema_path.read_text(encoding="utf-8")
        schema_keys = set(re.findall(r'key="(\w+)"', schema_text))
        model_keys = set(MemoryConfig.model_fields.keys())

        missing = schema_keys - model_keys
        assert not missing, (
            f"config_schema.py declares keys not in MemoryConfig: {sorted(missing)}. "
            f"Add them to config_model.py or .get() will return None and silently "
            f"disable the feature that reads them."
        )

    def test_t1b_internal_keys_allowlist(self):
        """T1b: every model key NOT in the UI schema is a *declared* internal key.

        The model is a superset of the schema by design (advanced/tuning knobs,
        internal objects like acl/backup, operational knobs). But a model-only
        key MUST be deliberately listed in MemoryConfig._INTERNAL_KEYS — this
        test fails on any key that's neither in the schema nor on the allowlist,
        so "hidden" keys cannot accrete silently.
        """
        import re
        from pathlib import Path
        from config_model import MemoryConfig

        schema_path = Path(__file__).resolve().parent.parent / "config_schema.py"
        schema_text = schema_path.read_text(encoding="utf-8")
        schema_keys = set(re.findall(r'key="(\w+)"', schema_text))
        model_keys = set(MemoryConfig.model_fields.keys())

        model_only = model_keys - schema_keys
        declared = set(MemoryConfig.internal_keys())

        undeclared = model_only - declared
        assert not undeclared, (
            f"model-only keys not declared internal: {sorted(undeclared)}. "
            f"Add each to MemoryConfig._INTERNAL_KEYS (with a reason) or add it "
            f"to config_schema.py — no silent hidden-key accretion."
        )
        # Also: the allowlist must not list keys that are now actually in the schema
        # (stale entries mean the "internal" classification drifted).
        stale = declared & schema_keys
        assert not stale, (
            f"_INTERNAL_KEYS entries that are actually in the schema: {sorted(stale)}. "
            f"Remove them from _INTERNAL_KEYS (they're UI-exposed now)."
        )

    def test_t3_round_trip_router_keys_readable(self):
        """T3: every router_* key written to config is readable via .get().

        The dict handed to intent_router / provider_ambient must contain all
        router_* keys with their configured values, not None.
        """
        from config_model import MemoryConfig

        raw = {
            "router_enabled": "true",
            "router_smart_model": "deepseek/deepseek-v4-pro-0813",
            "router_smart_provider": "openrouter",
            "router_default_model": "deepseek-v4-flash",
            "router_default_provider": "opencode-go",
            "router_subcall_enabled": "true",
            "router_temporal_threshold": "0.6",
            "router_multihop_threshold": "0.4",
        }
        c = MemoryConfig.model_validate(raw)

        # Every router key must be readable with the configured value.
        assert c.get("router_enabled") is True
        assert c.get("router_smart_model") == "deepseek/deepseek-v4-pro-0813"
        assert c.get("router_smart_provider") == "openrouter"
        assert c.get("router_default_model") == "deepseek-v4-flash"
        assert c.get("router_default_provider") == "opencode-go"
        assert c.get("router_subcall_enabled") is True
        assert c.get("router_temporal_threshold") == 0.6
        assert c.get("router_multihop_threshold") == 0.4
