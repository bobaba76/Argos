"""Pydantic-backed config model for the Argos hybrid memory plugin (#244).

Replaces the per-attribute slurp in ``provider_core.initialize()`` with a
single validated object.  Defaults match today's behaviour exactly — the
model is the new source of truth; changing a default changes behaviour.

Key properties:
- ``extra="forbid"`` — a mistyped key raises instead of silently doing nothing.
- Bool fields use ``_flag()``-compatible lax coercion ("true"/"1"/"yes"/"on" → True).
- Bounded int/float fields clamp to their valid range and fall back to the
  default on unparseable input (fail-soft, matching today's try/except pattern).
- ``config_validation.py`` helpers are reused inside field validators.
- A ``.get()`` method provides backward compatibility with code that still
  uses dict-style access (``self._config.get("key", default)``).
"""
from __future__ import annotations

import logging
from typing import Any, ClassVar, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

try:
    from .config_validation import storage_name_error
except ImportError:  # config_model.py imported as a top-level module
    from config_validation import storage_name_error

logger = logging.getLogger(__name__)


class MemoryConfig(BaseModel):
    """Validated configuration for the hybrid memory plugin.

    All field defaults match the effective defaults of the pre-#244 slurp.
    The JSON on disk is unchanged — this model is the runtime type/bounds
    layer, not a new format.
    """

    model_config = {"extra": "forbid"}

    # -- storage ---------------------------------------------------------------
    database_filename: str = "hybrid_memory.duckdb"
    graph_dirname: str = "hybrid_memory_kuzu"
    storage_mode: str = "shared_service"
    local_embedding_model: str = "BAAI/bge-small-en-v1.5"

    # -- injection -------------------------------------------------------------
    max_injected_items: int = Field(20, ge=1, le=512)
    inject_content_char_cap: int = Field(800, ge=100, le=5000)
    freshness_markers: bool = True
    skip_retrieval_on_trivial: bool = False
    injection_min_score: float = Field(0.0, ge=0.0, le=1.0)
    chronological_injection: bool = False
    date_anchor_rerank: bool = False
    history_at_current_time: bool = True

    # -- extraction / review ---------------------------------------------------
    auto_extract: bool = True
    llm_fallback: bool = True
    extraction_shadow_diff: bool = False
    auto_review: bool = True
    confirmation_surfacing: bool = True
    extraction_dup_threshold: float = Field(0.88, ge=0.0, le=1.0)
    stale_review_sweep_enabled: bool = True
    stale_review_interval_min: int = Field(15, ge=1, le=10080)
    stale_review_min_age_min: int = Field(30, ge=0, le=10080)
    stale_review_max_batch: int = Field(25, ge=1, le=500)

    # -- graph retrieval -------------------------------------------------------
    graph_aware_retrieval: bool = True
    graph_retrieval_boost: float = Field(0.0, ge=0.0, le=0.5)
    graph_inject_candidates: bool = False
    graph_boost_min_similarity: float = Field(0.15, ge=0.0, le=1.0)
    graph_traversal_enabled: bool = True
    graph_traversal_depth: int = Field(2, ge=1, le=4)
    graph_traversal_boost: float = Field(0.60, ge=0.0, le=1.0)
    graph_ppr_enabled: bool = False
    graph_ppr_damping: float = Field(0.5, ge=0.0, le=1.0)
    graph_ppr_boost: float = Field(0.0, ge=0.0, le=1.0)
    alias_expansion_boost: float = Field(0.7, ge=0.0, le=1.0)
    conflict_surfacing: bool = True

    # -- chain unfold ----------------------------------------------------------
    chain_unfold: str = "off"
    chain_unfold_min_similarity: float = Field(0.30, ge=0.0, le=1.0)
    chain_unfold_arc_min_similarity: float = Field(0.15, ge=0.0, le=1.0)
    chain_max_versions: int = Field(3, ge=1, le=10)
    chain_max_inject: int = Field(150, ge=1, le=10000)
    chain_unfold_top_k: int = Field(3, ge=1, le=20)
    chain_unfold_query_fallback: bool = False

    # -- consolidation / dedup -------------------------------------------------
    consolidation_enabled: bool = False
    consolidation_min_age_days: int = Field(30, ge=1, le=3650)
    consolidation_max_actions: int = Field(25, ge=1, le=500)
    consolidation_auto_apply: bool = True
    duplicate_min_similarity: float = Field(0.88, ge=0.0, le=1.0)
    duplicate_semantic_max_pairs: int = Field(20000, ge=100, le=1000000)

    # -- reranker --------------------------------------------------------------
    reranker_enabled: bool = False
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_top_n: int = Field(10, ge=5, le=100)

    # -- context-aware retrieval -----------------------------------------------
    context_aware_retrieval: bool = True
    phrase_lift_alpha: float = Field(0.0, ge=0.0, le=1.0)
    phrase_lift_pool: int = Field(200, ge=0, le=1000)
    context_window_size: int = Field(3, ge=1, le=10)
    context_max_chars: int = Field(500, ge=100, le=2000)

    # -- query expansion -------------------------------------------------------
    query_expansion_enabled: bool = True
    query_expansion_similarity_floor: float = Field(0.3, ge=0.0, le=1.0)

    # -- LLM endpoints ---------------------------------------------------------
    llm_model: str = ""
    llm_provider: str = ""
    extraction_llm_model: str = ""
    extraction_llm_provider: str = ""
    answering_llm_model: str = ""
    answering_llm_provider: str = ""
    role_alias_llm_fallback: bool = True

    # -- watcher ---------------------------------------------------------------
    watcher_enabled: bool = False
    watcher_scan_roots: List[str] = Field(default_factory=list)
    watcher_interval_min: int = Field(30, ge=1, le=1440)

    # -- deployment ------------------------------------------------------------
    deployment_mode: str = "cloud_pilot"
    data_residency: str = "cloud"

    # -- expiry ----------------------------------------------------------------
    expiry_enabled: bool = False
    expiry_ttl_days: str = '{"context_note":30,"event":180,"goal":180}'
    expiry_default_days: int = Field(90, ge=1, le=3650)
    expiry_auto_suggest: bool = False

    # -- distillation ----------------------------------------------------------
    distillation_enabled: bool = False
    distillation_min_new_records: int = Field(20, ge=1, le=100000)
    distillation_cooldown_hours: int = Field(24, ge=0, le=720)
    distillation_max_records_per_run: int = Field(100, ge=10, le=1000)
    distillation_max_calls: int = Field(10, ge=1, le=100)

    # -- lifecycle -------------------------------------------------------------
    archive_enabled: bool = False
    archive_after_days: int = Field(180, ge=1, le=36500)
    forget_enabled: bool = False
    forget_after_days: int = Field(365, ge=1, le=36500)
    rollup_enabled: bool = False
    rollup_interval_days: int = Field(30, ge=1, le=3650)
    rollup_max_records_per_run: int = Field(100, ge=10, le=1000)

    # -- egress ----------------------------------------------------------------
    local_only: bool = False
    external_sources_require_confirmation: bool = True

    # -- scale triggers --------------------------------------------------------
    scale_warn_latency_ms: float = Field(300.0, ge=0.0, le=60000.0)
    scale_warn_records: int = Field(5000, ge=1, le=10000000)

    # -- misc ------------------------------------------------------------------
    evidence_retention: str = "full"
    entity_aliases: str = ""
    role_words: str = ""

    # -- routing (intent router + ambient sub-call) ----------------------------
    # These are declared in config_schema.py and read by intent_router.py
    # and provider_ambient.py. They MUST be model fields so .get() returns
    # the configured value instead of None (which would silently disable
    # the router).
    router_enabled: bool = False
    router_smart_model: str = ""
    router_smart_provider: str = ""
    router_default_model: str = ""
    router_default_provider: str = ""
    router_subcall_enabled: bool = False
    router_temporal_threshold: float = Field(0.5, ge=0.1, le=1.0)
    router_multihop_threshold: float = Field(0.5, ge=0.1, le=1.0)

    # -- backup (service-coordinated) ------------------------------------------
    # Declared in config_schema.py; backup config is a nested dict in the
    # live JSON (config.get("backup", {})). The scalar keys are kept as
    # model fields for schema parity.
    backup_enabled: bool = False
    backup_dst_root: str = ""
    backup_retention_snapshots: int = Field(6, ge=1, le=100)
    backup: Optional[Dict[str, Any]] = None

    # -- ACL (optional, not in defaults dict) ----------------------------------
    acl: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # INTERNAL (non-UI) keys — advanced/tuning + internal-object fields
    # that exist in code but are deliberately NOT rendered in the Hermes
    # settings UI (config_schema.py). They are either: advanced tuning
    # thresholds (phrase_lift_*, graph_ppr_*, router_*_threshold,
    # injection_min_score, chain_unfold_arc_min_similarity), operational
    # instrumentation (scale_warn_*), internal config objects (acl,
    # backup dict), or experimental wired-but-off features (watcher_*).
    #
    # Source of truth: MemoryConfig.internal_keys() ==
    # model_fields - schema_keys. A new model field that is NOT in the UI
    # schema MUST be added here (or to the schema) — enforced by
    # test_t1b_internal_keys_allowlist (no silent hidden-key accretion).
    _INTERNAL_KEYS: ClassVar[frozenset] = frozenset({
        "acl", "backup",
        "chain_unfold_arc_min_similarity",
        "consolidation_auto_apply",
        "evidence_retention",
        "graph_ppr_enabled", "graph_ppr_damping", "graph_ppr_boost",
        "injection_min_score",
        "phrase_lift_alpha", "phrase_lift_pool",
        "router_temporal_threshold", "router_multihop_threshold",
        "scale_warn_latency_ms", "scale_warn_records",
        "skip_retrieval_on_trivial",
        "watcher_enabled", "watcher_interval_min", "watcher_scan_roots",
    })

    @classmethod
    def internal_keys(cls) -> frozenset:
        """Frozenset of model fields that are deliberately NOT UI-exposed."""
        return cls._INTERNAL_KEYS

    # ------------------------------------------------------------------
    # Fail-soft parsing — matches today's try/except + _flag behaviour.
    # All clamping and type coercion happens here so the model always
    # validates successfully.
    # ------------------------------------------------------------------

    # Bool field names (for the shared before-validator).
    _BOOL_FIELDS: ClassVar[frozenset] = frozenset({
        "freshness_markers", "skip_retrieval_on_trivial", "chronological_injection",
        "date_anchor_rerank", "history_at_current_time", "auto_extract",
        "llm_fallback", "extraction_shadow_diff", "auto_review",
        "confirmation_surfacing", "stale_review_sweep_enabled",
        "graph_aware_retrieval", "graph_inject_candidates",
        "graph_traversal_enabled", "graph_ppr_enabled", "conflict_surfacing",
        "chain_unfold_query_fallback", "consolidation_enabled",
        "consolidation_auto_apply", "reranker_enabled",
        "context_aware_retrieval", "query_expansion_enabled",
        "role_alias_llm_fallback", "watcher_enabled",
        "expiry_enabled", "expiry_auto_suggest",
        "distillation_enabled", "archive_enabled", "forget_enabled",
        "rollup_enabled", "local_only",
        "external_sources_require_confirmation",
        "router_enabled", "router_subcall_enabled", "backup_enabled",
    })

    # Clamped int fields: {name: (lo, hi, default)}.
    _CLAMPED_INT_FIELDS: ClassVar[dict] = {
        "max_injected_items": (1, 512, 20),
        "inject_content_char_cap": (100, 5000, 800),
        "stale_review_interval_min": (1, 10080, 15),
        "stale_review_min_age_min": (0, 10080, 30),
        "stale_review_max_batch": (1, 500, 25),
        "graph_traversal_depth": (1, 4, 2),
        "consolidation_min_age_days": (1, 3650, 30),
        "consolidation_max_actions": (1, 500, 25),
        "duplicate_semantic_max_pairs": (100, 1000000, 20000),
        "reranker_top_n": (5, 100, 10),
        "phrase_lift_pool": (0, 1000, 200),
        "context_window_size": (1, 10, 3),
        "context_max_chars": (100, 2000, 500),
        "watcher_interval_min": (1, 1440, 30),
        "expiry_default_days": (1, 3650, 90),
        "distillation_min_new_records": (1, 100000, 20),
        "distillation_cooldown_hours": (0, 720, 24),
        "distillation_max_records_per_run": (10, 1000, 100),
        "distillation_max_calls": (1, 100, 10),
        "archive_after_days": (1, 36500, 180),
        "forget_after_days": (1, 36500, 365),
        "rollup_interval_days": (1, 3650, 30),
        "rollup_max_records_per_run": (10, 1000, 100),
        "chain_max_versions": (1, 10, 3),
        "chain_max_inject": (1, 10000, 150),
        "chain_unfold_top_k": (1, 20, 3),
        "scale_warn_records": (1, 10000000, 5000),
        "backup_retention_snapshots": (1, 100, 6),
    }

    # Clamped float fields: {name: (lo, hi, default)}.
    _CLAMPED_FLOAT_FIELDS: ClassVar[dict] = {
        "injection_min_score": (0.0, 1.0, 0.0),
        "extraction_dup_threshold": (0.0, 1.0, 0.88),
        "graph_retrieval_boost": (0.0, 0.5, 0.0),
        "graph_boost_min_similarity": (0.0, 1.0, 0.15),
        "graph_traversal_boost": (0.0, 1.0, 0.60),
        "graph_ppr_damping": (0.0, 1.0, 0.5),
        "graph_ppr_boost": (0.0, 1.0, 0.0),
        "alias_expansion_boost": (0.0, 1.0, 0.7),
        "duplicate_min_similarity": (0.0, 1.0, 0.88),
        "phrase_lift_alpha": (0.0, 1.0, 0.0),
        "query_expansion_similarity_floor": (0.0, 1.0, 0.3),
        "chain_unfold_min_similarity": (0.0, 1.0, 0.30),
        "chain_unfold_arc_min_similarity": (0.0, 1.0, 0.15),
        "scale_warn_latency_ms": (0.0, 60000.0, 300.0),
        "router_temporal_threshold": (0.1, 1.0, 0.5),
        "router_multihop_threshold": (0.1, 1.0, 0.5),
    }

    # String fields that should be stripped.
    _STRIP_FIELDS: ClassVar[frozenset] = frozenset({
        "llm_model", "llm_provider", "extraction_llm_model",
        "extraction_llm_provider", "answering_llm_model",
        "answering_llm_provider", "deployment_mode", "data_residency",
        "evidence_retention", "reranker_model", "local_embedding_model",
        "router_smart_model", "router_smart_provider",
        "router_default_model", "router_default_provider",
        "backup_dst_root",
    })

    @model_validator(mode="before")
    @classmethod
    def _fail_soft_parse(cls, data: Any) -> Any:
        """Clamp, coerce, and fall back to defaults on bad input.

        This runs before pydantic's own validation, so by the time Field
        bounds are checked the values are already in range.  Matches the
        fail-soft behaviour of the pre-#244 slurp (try/except → default).
        """
        if not isinstance(data, dict):
            return data
        result = dict(data)

        # -- bool fields (match _flag() semantics) ----------------------------
        for key in cls._BOOL_FIELDS:
            if key in result:
                v = result[key]
                if isinstance(v, bool):
                    pass
                elif v is None:
                    result[key] = False
                else:
                    result[key] = str(v).strip().lower() in ("true", "1", "yes", "on")

        # -- clamped int fields -----------------------------------------------
        for key, (lo, hi, default) in cls._CLAMPED_INT_FIELDS.items():
            if key in result:
                try:
                    result[key] = max(lo, min(int(result[key]), hi))
                except (TypeError, ValueError):
                    result[key] = default

        # -- clamped float fields ---------------------------------------------
        for key, (lo, hi, default) in cls._CLAMPED_FLOAT_FIELDS.items():
            if key in result:
                try:
                    result[key] = max(lo, min(float(result[key]), hi))
                except (TypeError, ValueError):
                    result[key] = default

        # -- strip string fields ----------------------------------------------
        for key in cls._STRIP_FIELDS:
            if key in result and result[key] is not None:
                result[key] = str(result[key]).strip()

        # -- watcher_scan_roots: string → list --------------------------------
        if "watcher_scan_roots" in result:
            v = result["watcher_scan_roots"]
            if isinstance(v, str):
                result["watcher_scan_roots"] = [
                    r.strip() for r in v.split(",") if r.strip()
                ]
            elif v is None:
                result["watcher_scan_roots"] = []

        # -- chain_unfold: enum validation ------------------------------------
        if "chain_unfold" in result:
            v = str(result["chain_unfold"]).lower()
            if v not in {"off", "auto", "always"}:
                v = "off"
            result["chain_unfold"] = v

        # -- storage_mode: lowercase ------------------------------------------
        if "storage_mode" in result and result["storage_mode"] is not None:
            result["storage_mode"] = str(result["storage_mode"]).lower()

        return result

    # ------------------------------------------------------------------
    # Field validators (reuse config_validation helpers)
    # ------------------------------------------------------------------

    @field_validator("database_filename", "graph_dirname")
    @classmethod
    def _safe_storage_name(cls, v: str) -> str:
        err = storage_name_error(v)
        if err:
            raise ValueError(err)
        return v.strip()

    # ------------------------------------------------------------------
    # Backward-compat: dict-style .get() for code that hasn't been
    # converted to attribute access yet.
    # ------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-style access for backward compatibility.

        Returns the field value, or *default* if the field doesn't exist.
        Code that still uses ``self._config.get("key", default)`` works
        unchanged — the value is already the native type (int/bool/str).
        """
        if hasattr(self, key):
            return getattr(self, key)
        return default

    def __contains__(self, key: str) -> bool:
        """Dict-style ``in`` operator for backward compatibility."""
        return key in type(self).model_fields
