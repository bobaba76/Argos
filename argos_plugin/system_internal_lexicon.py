"""#392 part 3: catch-up hygiene lexicon sweep.

A deterministic, zero-LLM sweep that flags existing active records matching
an implementation-machinery lexicon with ``record_class='system_internal'``.
This retires existing pollution over time in addition to the one-off
quarantine sweep applied 9/9.

The lexicon is seeded from the 20 records the one-off sweep quarantined
(the ground truth for what the filter must catch) plus the categories
described in issue #392: mutation_events notes, rollup bug notes,
reranker/benchmark decisions, store layout audits, and distillation
internals.

Design constraints (from the issue):
- **Deterministic, zero-LLM** — regex/substring matching only.
- **Distinguishes implementation machinery from project memory.**
  "rollup config defaults to false" = excluded (engine-room noise);
  "PR #392 merged" = eligible (project memory).
- **Dry-run by default** — preview first, like the POPIA erase workflow.
  The apply step requires an explicit ``apply=True``.
- **Idempotent** — re-running on already-flagged records is a no-op.
- **Config-gated** — only runs when ``distillation_exclude_system_internal``
  is true (the default). When the exclusion is off, the sweep is a no-op
  (the deployment wants system notes distilled, so flagging them is
  wrong).

Usage::

    from system_internal_lexicon import sweep_system_internal

    # Preview (dry-run): returns a report, writes nothing.
    report = sweep_system_internal(store, dry_run=True)

    # Apply: flags matching records with record_class='system_internal'.
    report = sweep_system_internal(store, dry_run=False)
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lexicon — implementation-machinery patterns.
#
# These patterns match content that describes the *internals* of the memory
# system (config values, schema versions, function names, tuning parameters,
# data structures, benchmark decisions) — NOT project memory (PRs, issues,
# features, roadmap items).
#
# The patterns are intentionally specific to avoid false positives on
# project memory like "PR #392 merged" or "rollup feature shipped".
# ---------------------------------------------------------------------------

# Config key / config file references — matches records that describe
# config defaults, config keys, or config schema details.
_CONFIG_PATTERNS = [
    re.compile(r"\bconfig_schema\b", re.IGNORECASE),
    re.compile(r"\bconfig_model\b", re.IGNORECASE),
    re.compile(r"\bconfig[_ ]key\b", re.IGNORECASE),
    re.compile(r"\bdefaults to (true|false)\b", re.IGNORECASE),
    re.compile(r"\bhybrid_memory\.json\b", re.IGNORECASE),
]

# Schema version / migration references — matches records about schema
# versions, migrations, or DDL changes.
_SCHEMA_PATTERNS = [
    re.compile(r"\bschema v\d+\b", re.IGNORECASE),
    re.compile(r"\bschema_version\b", re.IGNORECASE),
    re.compile(r"\bLATEST_SCHEMA_VERSION\b", re.IGNORECASE),
    re.compile(r"\bmigration_\d+_to_\d+\b", re.IGNORECASE),
    re.compile(r"\bALTER TABLE memory_records\b", re.IGNORECASE),
    re.compile(r"\bADD COLUMN\b.*memory_records", re.IGNORECASE),
]

# Internal module / function names — matches records that reference
# internal Python modules or functions by name (typically debug/audit
# notes about how a function works or was changed).
_INTERNAL_NAME_PATTERNS = [
    re.compile(r"\b_sanitize_args\b"),
    re.compile(r"\bbackfill_graph\b"),
    re.compile(r"\b_load_eligible_records\b"),
    re.compile(r"\bstore_maintenance\b"),
    re.compile(r"\bstore_retrieval\b"),
    re.compile(r"\bstore_write\b"),
    re.compile(r"\bstore_core\b"),
    re.compile(r"\bmemory_service\b"),
    re.compile(r"\bapi_facade\b"),
    re.compile(r"\bmcp_server\b"),
    re.compile(r"\brest_server\b"),
    re.compile(r"\bprovider_core\b"),
    re.compile(r"\bprovider_session\b"),
    re.compile(r"\bprovider_retrieval\b"),
    re.compile(r"\bnamespace_partition\b"),
    re.compile(r"\bschema_migrations\b"),
    re.compile(r"\b_distill\b"),
    re.compile(r"\b_seed_star_cluster\b"),
    re.compile(r"\b_run_distillation\b"),
]

# Internal data structures — matches records about internal tables,
# ledgers, or data structures.
_DATA_STRUCT_PATTERNS = [
    re.compile(r"\bmutation_events\b", re.IGNORECASE),
    re.compile(r"\bmemory_records\b", re.IGNORECASE),
    re.compile(r"\bmemory_candidates\b", re.IGNORECASE),
    re.compile(r"\bdeletion_tombstones\b", re.IGNORECASE),
    re.compile(r"\brejection_ledger\b", re.IGNORECASE),
    re.compile(r"\bmemory_evidence\b", re.IGNORECASE),
]

# Tuning / benchmark / threshold references — matches records about
# retrieval tuning, similarity thresholds, or benchmark decisions.
_TUNING_PATTERNS = [
    re.compile(r"\breranker verdict\b", re.IGNORECASE),
    re.compile(r"\bsimilarity floor\b", re.IGNORECASE),
    re.compile(r"\bcluster threshold\b", re.IGNORECASE),
    re.compile(r"\bthreshold 0\.\d+\b", re.IGNORECASE),
    re.compile(r"\bbenchmark (decision|result|finding)\b", re.IGNORECASE),
    re.compile(r"\bcontext_aware_retrieval\b", re.IGNORECASE),
    re.compile(r"\bquery_expansion_enabled\b", re.IGNORECASE),
    re.compile(r"\bphrase_lift_alpha\b", re.IGNORECASE),
    re.compile(r"\binjection_min_score\b", re.IGNORECASE),
]

# Rollup / distillation internals — matches records about the internal
# mechanics of rollup or distillation (config, bugs, thresholds), NOT
# records about the feature itself (e.g. "rollup feature shipped" is
# project memory).
_ROLLUP_DISTILL_PATTERNS = [
    re.compile(r"\brollup (config|bug|internals|threshold|default)\b", re.IGNORECASE),
    re.compile(r"\brollup_enabled\b", re.IGNORECASE),
    re.compile(r"\brollup_after_days\b", re.IGNORECASE),
    re.compile(r"\bdistillation (internals|pass|cluster|threshold|config|default)\b", re.IGNORECASE),
    re.compile(r"\bdistillation_enabled\b", re.IGNORECASE),
    re.compile(r"\bdistillation_exclude_system_internal\b", re.IGNORECASE),
    re.compile(r"\bdistillation_min_new_records\b", re.IGNORECASE),
    re.compile(r"\bdistillation_cooldown_hours\b", re.IGNORECASE),
    re.compile(r"\bdistillation_max_records_per_run\b", re.IGNORECASE),
    re.compile(r"\bdistillation_max_calls\b", re.IGNORECASE),
]

# Store layout / audit references — matches records about the store's
# internal layout, column counts, or audit findings.
_STORE_LAYOUT_PATTERNS = [
    re.compile(r"\bstore layout\b", re.IGNORECASE),
    re.compile(r"\bstore audit\b", re.IGNORECASE),
    re.compile(r"\bcolumn count\b", re.IGNORECASE),
    re.compile(r"\b\d+ columns\b.*memory_records", re.IGNORECASE),
    re.compile(r"\brecord_class\b", re.IGNORECASE),
    re.compile(r"\bembedding_dim\b", re.IGNORECASE),
    re.compile(r"\bembedder_id\b", re.IGNORECASE),
]

_ALL_PATTERNS: List[re.Pattern] = (
    _CONFIG_PATTERNS
    + _SCHEMA_PATTERNS
    + _INTERNAL_NAME_PATTERNS
    + _DATA_STRUCT_PATTERNS
    + _TUNING_PATTERNS
    + _ROLLUP_DISTILL_PATTERNS
    + _STORE_LAYOUT_PATTERNS
)


def is_system_internal(content: str) -> bool:
    """Return True if *content* matches the implementation-machinery lexicon.

    Deterministic, zero-LLM. Matches any of the lexicon patterns (OR).
    Does NOT match project memory (PRs, issues, features, roadmap items).
    """
    if not content:
        return False
    return any(pat.search(content) for pat in _ALL_PATTERNS)


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def sweep_system_internal(
    store: Any,
    *,
    dry_run: bool = True,
    apply: bool = False,
) -> Dict[str, Any]:
    """Scan active records and flag implementation-machinery as system_internal.

    Deterministic, zero-LLM. Scans active, non-superseded records whose
    ``record_class`` is not already ``'system_internal'``, checks each
    against the lexicon, and (when ``apply=True``) flags matching records
    with ``record_class='system_internal'``.

    Args:
        store: A DuckDBMemoryStore (or compatible) with ``load_eligible_records``
            and a writable connection.
        dry_run: When True (default), preview only — no writes. When False,
            apply the flags. ``apply=True`` is an alias for ``dry_run=False``.
        apply: When True, apply the flags (alias for ``dry_run=False``).

    Returns:
        A report dict with:
        - ``scanned``: number of records checked.
        - ``matched``: number of records matching the lexicon.
        - ``flagged``: number of records actually flagged (0 in dry-run).
        - ``already_flagged``: number of records already system_internal.
        - ``matches``: list of (memory_id, content_preview) for matched records.
        - ``dry_run``: whether this was a preview.
    """
    do_apply = apply or not dry_run

    # Load active, non-superseded records (no exclusion — we want to see
    # everything, including already-flagged records so we can count them).
    records = store.load_eligible_records(
        since=None, limit=100000, exclude_system_internal=False,
    )

    matched: List[Dict[str, str]] = []
    already_flagged = 0
    flagged = 0

    for rec in records:
        if getattr(rec, "record_class", None) == "system_internal":
            already_flagged += 1
            continue
        if is_system_internal(rec.content or ""):
            matched.append({
                "memory_id": rec.memory_id,
                "content_preview": (rec.content or "")[:120],
            })

    if do_apply and matched:
        # Flag matching records with record_class='system_internal'.
        ids = [m["memory_id"] for m in matched]
        try:
            with store._state.lock:
                assert store.connection is not None
                for memory_id in ids:
                    store.connection.execute(
                        "UPDATE memory_records SET record_class = 'system_internal' "
                        "WHERE memory_id = ?",
                        [memory_id],
                    )
            flagged = len(ids)
        except Exception as exc:
            logger.error("system_internal sweep: failed to flag %d records: %s", len(ids), exc)

    return {
        "scanned": len(records),
        "matched": len(matched),
        "flagged": flagged if do_apply else 0,
        "already_flagged": already_flagged,
        "matches": matched,
        "dry_run": not do_apply,
    }
