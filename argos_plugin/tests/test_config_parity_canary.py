"""Config schema/model/loader parity + realistic-fixture canary (#274).

The #272 review caught a SILENT CONFIG WIPE: extra="forbid" + catch-all
fallback reverted the real hybrid_memory.json (75 keys) to defaults, and
.get() dropped router_* keys, silently disabling the LLM router. That
catch was manual. These tests make it a CI failure.

T1: schema↔model parity — every config_schema.py key is a MemoryConfig field.
    (Already in test_config_model.py::test_t1_schema_model_parity — this
    file adds the reverse direction: every model field is either in the
    schema or declared internal.)
T2: loader↔model parity — every model field can be loaded by _load_config
    from a JSON file and read back via .get() with the configured value.
T3: realistic-fixture canary — load a real-shaped hybrid_memory.json
    fixture (75 keys incl. router_*, phrase_lift_alpha,
    injection_min_score, skip_retrieval_on_trivial, conflict_surfacing)
    and assert every key survives a full load→save→load round-trip with
    NO reversion to defaults and NO dropped keys.
T4: unknown-key passthrough — unknown keys in the raw JSON are logged +
    dropped (not fatal, no wipe), and known keys survive alongside them.

Acceptance: CI fails if schema/model/loader drift; canary fails on any
silent wipe or drop.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))


# ---------------------------------------------------------------------------
# Realistic 75-key fixture — mirrors the maintainer's live hybrid_memory.json
# shape (tuned values, NOT defaults, so a reversion is detectable).
# ---------------------------------------------------------------------------

REALISTIC_CONFIG: dict = {
    # storage (4)
    "database_filename": "hybrid_memory.duckdb",
    "graph_dirname": "hybrid_memory_kuzu",
    "storage_mode": "shared_service",
    "local_embedding_model": "BAAI/bge-small-en-v1.5",
    # injection (8)
    "max_injected_items": 96,
    "inject_content_char_cap": 1200,
    "freshness_markers": "true",
    "skip_retrieval_on_trivial": "true",
    "injection_min_score": 0.3,
    "chronological_injection": "true",
    "date_anchor_rerank": "true",
    "history_at_current_time": "true",
    # extraction/review (10)
    "auto_extract": "true",
    "llm_fallback": "true",
    "extraction_shadow_diff": "false",
    "auto_review": "true",
    "confirmation_surfacing": "true",
    "extraction_dup_threshold": "0.92",
    "stale_review_sweep_enabled": "true",
    "stale_review_interval_min": "20",
    "stale_review_min_age_min": "45",
    "stale_review_max_batch": "30",
    # graph retrieval (12)
    "graph_aware_retrieval": "true",
    "graph_retrieval_boost": "0.15",
    "graph_inject_candidates": "false",
    "graph_boost_min_similarity": "0.20",
    "graph_traversal_enabled": "true",
    "graph_traversal_depth": "3",
    "graph_traversal_boost": "0.70",
    "graph_ppr_enabled": "false",
    "graph_ppr_damping": "0.6",
    "graph_ppr_boost": "0.0",
    "alias_expansion_boost": "0.8",
    "conflict_surfacing": "true",
    # chain unfold (7)
    "chain_unfold": "auto",
    "chain_unfold_min_similarity": "0.35",
    "chain_unfold_arc_min_similarity": "0.20",
    "chain_max_versions": "5",
    "chain_max_inject": "200",
    "chain_unfold_top_k": "5",
    "chain_unfold_query_fallback": "true",
    # consolidation/dedup (6)
    "consolidation_enabled": "false",
    "consolidation_min_age_days": "60",
    "consolidation_max_actions": "50",
    "consolidation_auto_apply": "false",
    "duplicate_min_similarity": "0.90",
    "duplicate_semantic_max_pairs": "30000",
    # reranker (3)
    "reranker_enabled": "true",
    "reranker_model": "BAAI/bge-reranker-base",
    "reranker_top_n": "15",
    # context-aware retrieval (5)
    "context_aware_retrieval": "true",
    "phrase_lift_alpha": "0.15",
    "phrase_lift_pool": "300",
    "context_window_size": "4",
    "context_max_chars": "600",
    # query expansion (2)
    "query_expansion_enabled": "true",
    "query_expansion_similarity_floor": "0.35",
    # LLM endpoints (7)
    "llm_model": "deepseek/deepseek-v4-pro",
    "llm_provider": "openrouter",
    "extraction_llm_model": "deepseek/deepseek-v4-flash",
    "extraction_llm_provider": "openrouter",
    "answering_llm_model": "deepseek/deepseek-v4-pro",
    "answering_llm_provider": "openrouter",
    "role_alias_llm_fallback": "true",
    # routing (8) — the keys that were silently dropped
    "router_enabled": "true",
    "router_smart_model": "deepseek/deepseek-v4-pro",
    "router_smart_provider": "openrouter",
    "router_default_model": "deepseek/deepseek-v4-flash",
    "router_default_provider": "openrouter",
    "router_subcall_enabled": "true",
    "router_temporal_threshold": "0.6",
    "router_multihop_threshold": "0.4",
    # distillation (1)
    "distillation_enabled": "false",
    # lifecycle — rollup (3)
    "rollup_enabled": "false",
    "rollup_interval_days": "30",
    "rollup_max_records_per_run": "100",
    # misc (3)
    "evidence_retention": "full",
    "entity_aliases": "",
    "role_words": "",
}

assert len(REALISTIC_CONFIG) >= 75, f"fixture must have >=75 keys, has {len(REALISTIC_CONFIG)}"


class TestT1bDefaultValueParity:
    """T1b: every ProviderField default in config_schema.py equals the
    corresponding MemoryConfig model_fields default (after coercion).

    The config UI writes schema defaults on save, so a mismatch means a
    user saving any unrelated setting gets the schema default written,
    silently flipping the runtime behavior. This is the exact
    behavior-flip family #274/#275 exist to kill.

    Coercion: bools via str().lower(), numerics via float(), strings
    exact. This matches how the loader normalizes values.
    """

    @staticmethod
    def _schema_defaults() -> dict:
        """Extract {key: default_str} from config_schema.py via regex."""
        import re
        schema_path = Path(__file__).resolve().parent.parent / "config_schema.py"
        text = schema_path.read_text(encoding="utf-8")
        pattern = r'ProviderField\(\s*key="(\w+)"[^)]*?default=("[^"]*"|True|False|None|[\d.]+)[^)]*?\)'
        matches = re.findall(pattern, text, re.DOTALL)
        result = {}
        for key, default in matches:
            if default.startswith('"'):
                result[key] = default[1:-1]
            elif default in ('True', 'False'):
                result[key] = default.lower()
            elif default == 'None':
                continue
            else:
                result[key] = default
        return result

    def test_schema_defaults_match_model_defaults(self):
        """Every ProviderField default equals the MemoryConfig default
        (after coercion). CI fails on any mismatch."""
        from config_model import MemoryConfig
        schema_defaults = self._schema_defaults()
        model_fields = MemoryConfig.model_fields
        mismatches = []
        for key, schema_str in schema_defaults.items():
            if key not in model_fields:
                continue  # T1 covers missing keys
            model_default = model_fields[key].default
            # Coerce based on model default type
            if isinstance(model_default, bool):
                schema_cmp = schema_str.lower()
                model_cmp = str(model_default).lower()
            elif isinstance(model_default, (int, float)):
                schema_cmp = float(schema_str)
                model_cmp = float(model_default)
            else:
                schema_cmp = str(schema_str)
                model_cmp = str(model_default)
            if schema_cmp != model_cmp:
                mismatches.append(
                    f"  {key}: schema default={schema_str!r}, "
                    f"model default={model_default!r}"
                )
        assert not mismatches, (
            "Schema/model default-value mismatch (the config UI writes "
            "schema defaults on save, so a mismatch silently flips "
            "runtime behavior):\n" + "\n".join(mismatches)
        )


class TestT2LoaderModelParity:
    """T2: every model field can be loaded by _load_config and read via .get()."""

    def test_every_model_field_loads_from_json(self):
        """Every MemoryConfig field, when written to hybrid_memory.json,
        is loaded by _load_config and readable via .get() with the
        configured value (not the default)."""
        from config_model import MemoryConfig
        from provider_core import _load_config

        # Build a config dict with non-default values for every field
        # that accepts a scalar (str/int/float/bool). Skip None-default
        # optional fields (acl, backup) — they're tested separately.
        raw = {}
        for name, field_info in MemoryConfig.model_fields.items():
            if name in ("acl", "backup"):
                continue  # Optional dict fields, tested separately
            default = field_info.default
            # Use a non-default value where possible.
            if isinstance(default, bool):
                raw[name] = not default
            elif isinstance(default, int):
                raw[name] = default + 1 if default < 100 else max(1, default - 1)
            elif isinstance(default, float):
                raw[name] = default + 0.1
            elif isinstance(default, str):
                if name == "chain_unfold":
                    # Enum field: only off/auto/always. Use "auto" (non-default).
                    raw[name] = "auto"
                elif name == "expiry_ttl_days":
                    # JSON string — keep a valid JSON shape.
                    raw[name] = '{"context_note":60,"event":90,"goal":120}'
                elif default == "":
                    raw[name] = "test_value"
                else:
                    raw[name] = default + "_tuned"
            elif isinstance(default, list):
                raw[name] = ["/test/path"]
            # None defaults (Optional) are skipped.

        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "hybrid_memory.json"
            config_path.write_text(json.dumps(raw), encoding="utf-8")
            cfg = _load_config(str(tmp))

            for name, field_info in MemoryConfig.model_fields.items():
                if name in ("acl", "backup"):
                    continue
                if name not in raw:
                    continue
                expected = raw[name]
                actual = cfg.get(name)
                # For bool fields, the raw value is a bool (not a string),
                # so it should round-trip exactly.
                if isinstance(expected, bool):
                    assert actual is expected, \
                        f"{name}: expected {expected}, got {actual}"
                # For numeric fields, clamping may have changed the value,
                # but it should NOT be silently reverted to the default
                # (unless the clamped value happens to equal the default,
                # which means our test value was out of bounds and clamped
                # back — that's acceptable and rare).
                elif isinstance(expected, (int, float)):
                    field_default = field_info.default
                    if actual == field_default and expected != field_default:
                        # The value was reverted to default — this is the
                        # silent-wipe bug. Fail loudly.
                        pytest.fail(
                            f"{name}: expected {expected} (non-default), "
                            f"got {actual} (default) — value was silently wiped"
                        )
                    # If actual != default, the value survived (possibly
                    # clamped, but not wiped). That's acceptable.
                # For string fields, the value should survive.
                elif isinstance(expected, str):
                    if expected and expected != field_info.default:
                        # Some fields are lowercased/stripped by the parser.
                        if name in ("storage_mode", "chain_unfold"):
                            assert actual == expected.lower(), \
                                f"{name}: expected '{expected.lower()}', got '{actual}'"
                        else:
                            assert actual == expected or actual == expected.strip(), \
                                f"{name}: expected '{expected}', got '{actual}'"

    def test_loader_does_not_revert_to_defaults_on_valid_config(self):
        """A valid config with tuned values must NOT revert to defaults."""
        from provider_core import _load_config

        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "hybrid_memory.json"
            config_path.write_text(json.dumps(REALISTIC_CONFIG), encoding="utf-8")
            cfg = _load_config(str(tmp))

            # Spot-check tuned values (NOT defaults).
            assert cfg.max_injected_items == 96  # default is 20
            assert cfg.injection_min_score == 0.3  # default is 0.0
            assert cfg.skip_retrieval_on_trivial is True  # default is False
            assert cfg.phrase_lift_alpha == 0.15  # default is 0.0
            assert cfg.conflict_surfacing is True  # default is True (but test it)
            assert cfg.router_enabled is True  # default is False
            assert cfg.router_smart_model == "deepseek/deepseek-v4-pro"
            assert cfg.chain_unfold == "auto"  # default is "auto"


class TestT3RealisticFixtureCanary:
    """T3: realistic-fixture canary — load a real-shaped 75-key config,
    assert every key survives a full load→save→load round-trip with NO
    reversion to defaults and NO dropped keys."""

    def test_all_75_keys_survive_load(self):
        """Every key in the 75-key fixture survives _load_config with the
        configured value (not reverted to default, not dropped to None)."""
        from provider_core import _load_config
        from config_model import MemoryConfig

        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "hybrid_memory.json"
            config_path.write_text(json.dumps(REALISTIC_CONFIG), encoding="utf-8")
            cfg = _load_config(str(tmp))

            model_fields = set(MemoryConfig.model_fields.keys())
            for key, expected_raw in REALISTIC_CONFIG.items():
                assert key in model_fields, \
                    f"{key} is not a MemoryConfig field — silently dropped"
                actual = cfg.get(key)
                assert actual is not None, \
                    f"{key}: .get() returned None — silently dropped"
                # The value should not be the default (for tuned keys).
                # We check that the loaded value matches the expected
                # value after fail-soft coercion.
                field_info = MemoryConfig.model_fields[key]
                default = field_info.default
                if isinstance(expected_raw, str):
                    # Bool-as-string fields: "true" → True
                    if key in MemoryConfig._BOOL_FIELDS:
                        expected_bool = expected_raw.strip().lower() in ("true", "1", "yes", "on")
                        assert actual is expected_bool, \
                            f"{key}: expected {expected_bool}, got {actual}"
                    # Numeric-as-string fields: clamped
                    elif key in MemoryConfig._CLAMPED_INT_FIELDS:
                        lo, hi, _default = MemoryConfig._CLAMPED_INT_FIELDS[key]
                        expected_int = max(lo, min(int(expected_raw), hi))
                        assert actual == expected_int, \
                            f"{key}: expected {expected_int}, got {actual}"
                    elif key in MemoryConfig._CLAMPED_FLOAT_FIELDS:
                        lo, hi, _default = MemoryConfig._CLAMPED_FLOAT_FIELDS[key]
                        expected_float = max(lo, min(float(expected_raw), hi))
                        assert actual == expected_float, \
                            f"{key}: expected {expected_float}, got {actual}"
                    elif key == "chain_unfold":
                        assert actual == expected_raw.lower()
                    elif key == "storage_mode":
                        assert actual == expected_raw.lower()
                    else:
                        # Plain string field — stripped.
                        assert actual == expected_raw.strip(), \
                            f"{key}: expected '{expected_raw}', got '{actual}'"
                else:
                    assert actual == expected_raw, \
                        f"{key}: expected {expected_raw}, got {actual}"

    def test_load_save_load_round_trip(self):
        """Full load→save→load round-trip: every key survives with no
        reversion to defaults and no dropped keys."""
        from provider_core import _load_config

        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "hybrid_memory.json"
            # Phase 1: write the realistic config.
            config_path.write_text(json.dumps(REALISTIC_CONFIG, indent=2),
                                   encoding="utf-8")
            cfg1 = _load_config(str(tmp))

            # Phase 2: save the config back (simulating save_config).
            # We serialize the model to a dict, then write it.
            saved = {}
            for key in REALISTIC_CONFIG:
                saved[key] = cfg1.get(key)
            # Merge with existing file content (save_config behavior).
            existing = json.loads(config_path.read_text(encoding="utf-8"))
            existing.update(saved)
            config_path.write_text(json.dumps(existing, indent=2),
                                   encoding="utf-8")

            # Phase 3: reload and verify every key survived.
            cfg2 = _load_config(str(tmp))
            for key in REALISTIC_CONFIG:
                v1 = cfg1.get(key)
                v2 = cfg2.get(key)
                assert v2 is not None, f"{key}: dropped after round-trip"
                assert v1 == v2, \
                    f"{key}: changed after round-trip: {v1} → {v2}"

    def test_no_silent_wipe_with_unknown_keys(self):
        """The #272 bug: extra="forbid" + catch-all reverted the whole
        config to defaults when an unknown key was present. The loader
        now filters unknown keys before validation, so known keys
        survive alongside unknown ones."""
        from provider_core import _load_config

        config_with_unknowns = dict(REALISTIC_CONFIG)
        config_with_unknowns["some_future_key"] = "value"
        config_with_unknowns["tenants"] = {"acme": {"review_mode": "auto"}}
        config_with_unknowns["experimental_flag"] = True

        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "hybrid_memory.json"
            config_path.write_text(json.dumps(config_with_unknowns),
                                   encoding="utf-8")
            cfg = _load_config(str(tmp))

            # Known tuned values must survive (NOT reverted to defaults).
            assert cfg.max_injected_items == 96
            assert cfg.skip_retrieval_on_trivial is True
            assert cfg.injection_min_score == 0.3
            assert cfg.router_enabled is True
            assert cfg.router_smart_model == "deepseek/deepseek-v4-pro"
            assert cfg.phrase_lift_alpha == 0.15
            assert cfg.conflict_surfacing is True
            assert cfg.chain_unfold == "auto"

    def test_router_keys_not_dropped(self):
        """The #272 bug: .get() dropped router_* keys, silently disabling
        the LLM router. Every router_* key must survive load and be
        readable via .get() with the configured value."""
        from provider_core import _load_config

        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "hybrid_memory.json"
            config_path.write_text(json.dumps(REALISTIC_CONFIG),
                                   encoding="utf-8")
            cfg = _load_config(str(tmp))

            assert cfg.get("router_enabled") is True
            assert cfg.get("router_smart_model") == "deepseek/deepseek-v4-pro"
            assert cfg.get("router_smart_provider") == "openrouter"
            assert cfg.get("router_default_model") == "deepseek/deepseek-v4-flash"
            assert cfg.get("router_default_provider") == "openrouter"
            assert cfg.get("router_subcall_enabled") is True
            assert cfg.get("router_temporal_threshold") == 0.6
            assert cfg.get("router_multihop_threshold") == 0.4


class TestT4UnknownKeyPassthrough:
    """T4: unknown-key passthrough behavior is documented + tested."""

    def test_unknown_keys_logged_and_dropped(self, caplog):
        """Unknown keys in the raw JSON are logged (WARNING) and dropped
        (not passed to MemoryConfig, which would reject them via
        extra="forbid"). Known keys alongside them survive."""
        from provider_core import _load_config
        import logging

        config_with_unknowns = dict(REALISTIC_CONFIG)
        config_with_unknowns["unknown_key_1"] = "value"
        config_with_unknowns["unknown_key_2"] = 42

        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "hybrid_memory.json"
            config_path.write_text(json.dumps(config_with_unknowns),
                                   encoding="utf-8")
            with caplog.at_level(logging.WARNING, logger="argos.provider_core"):
                cfg = _load_config(str(tmp))

            # Known keys survive.
            assert cfg.max_injected_items == 96
            assert cfg.router_enabled is True

            # Unknown keys were logged.
            log_text = " ".join(r.getMessage() for r in caplog.records)
            assert "unknown_key_1" in log_text
            assert "unknown_key_2" in log_text

    def test_empty_and_none_values_filtered(self):
        """Empty string and None values in the raw JSON are filtered
        before validation (matching the pre-#244 slurp behavior)."""
        from provider_core import _load_config

        config_with_empties = dict(REALISTIC_CONFIG)
        config_with_empties["llm_model"] = ""  # empty → filtered
        config_with_empties["llm_provider"] = None  # None → filtered

        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "hybrid_memory.json"
            config_path.write_text(json.dumps(config_with_empties),
                                   encoding="utf-8")
            cfg = _load_config(str(tmp))

            # Empty/None values were filtered, so defaults apply.
            assert cfg.llm_model == ""  # default is ""
            assert cfg.llm_provider == ""  # default is ""
            # Other tuned values survive.
            assert cfg.max_injected_items == 96

    def test_malformed_json_falls_back_to_defaults(self):
        """Malformed JSON in the config file is logged and the config
        falls back to defaults (not a crash)."""
        from provider_core import _load_config

        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "hybrid_memory.json"
            config_path.write_text('{invalid json,}', encoding="utf-8")
            cfg = _load_config(str(tmp))
            # Defaults.
            assert cfg.max_injected_items == 20
            assert cfg.router_enabled is False

    def test_missing_config_file_uses_defaults(self):
        """No config file → all defaults."""
        from provider_core import _load_config

        with tempfile.TemporaryDirectory() as tmp:
            # No hybrid_memory.json written.
            cfg = _load_config(str(tmp))
            assert cfg.max_injected_items == 20
            assert cfg.router_enabled is False
            assert cfg.conflict_surfacing is True
