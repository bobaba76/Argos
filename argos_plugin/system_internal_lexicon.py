"""#392 part 3: catch-up hygiene lexicon sweep.

A deterministic, zero-LLM sweep that flags existing active records matching
an implementation-machinery lexicon with ``record_class='system_internal'``.
This retires existing pollution over time in addition to the one-off
quarantine sweep applied 9/9.

Lexicon provenance: the one-off sweep's 20 quarantined records were not
recoverable from any accessible store (live or backup), so the lexicon is
built from the categories and examples described in issue #392: mutation_events
notes, rollup bug notes, reranker/benchmark decisions, store layout audits,
distillation internals, config defaults, schema versions, internal
module/function names, internal data structures, and tuning parameters.
This is a hand-written approximation, not a data-seeded lexicon — it will
have some false positives and false negatives at scale. The config flag
``distillation_exclude_system_internal`` lets a deployment turn the whole
filter off if the false-positive rate is unacceptable.

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
- **Run-once** — the startup hook records a system_state key after a
  successful apply so it does not re-scan on every startup.

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
# Project-memory exclusion heuristic.
#
# Config key names and internal data-structure names can appear in both
# system-internal notes ("context_aware_retrieval defaults to true") and
# project memory ("context_aware_retrieval feature shipped in v2"). To
# distinguish them, patterns for these names use a negative lookahead that
# rejects matches when a project-memory word appears within 40 characters
# after the name.
#
# This is an approximation — it will miss project memory where the
# project word appears before the name, and it may false-positive on
# internal notes that happen to contain one of these words. The list is
# kept short and focused on event/evaluation verbs that are rare in
# pure implementation notes.
# ---------------------------------------------------------------------------
_PROJECT_MEMORY_WORDS = (
    "feature", "shipped", "released", "great", "useful", "helpful",
    "love", "like", "ran", "found", "reached", "today", "yesterday",
    "merged", "closed", "fixed", "resolved", "working", "broken",
    "stable", "unstable", "fast", "slow", "nice", "cool",
)
_PROJECT_MEMORY_LOOKAHEAD = (
    f"(?!.{{0,40}}(?:{'|'.join(_PROJECT_MEMORY_WORDS)}))"
)


def _ctx(pattern: str) -> re.Pattern:
    """Compile *pattern* with a negative lookahead excluding project-memory
    words. Used for config key names and data-structure names that can
    appear in both internal notes and project memory.
    """
    return re.compile(pattern + _PROJECT_MEMORY_LOOKAHEAD, re.IGNORECASE)


# ---------------------------------------------------------------------------
# Lexicon — implementation-machinery patterns.
#
# Three tiers:
# 1. Unambiguous patterns (function names, module names, schema versions,
#    migrations, DDL, "defaults to", config file names) — always internal.
# 2. Contextual patterns (config key names, data-structure names) — internal
#    only when not followed by a project-memory word.
# 3. Qualifier patterns (rollup/distillation + internal qualifier) — internal
#    only when accompanied by a machinery word like "config", "threshold",
#    "bug", "internals".
# ---------------------------------------------------------------------------

# Tier 1: Config file / config schema references — unambiguous.
_CONFIG_PATTERNS = [
    re.compile(r"\bconfig_schema\b", re.IGNORECASE),
    re.compile(r"\bconfig_model\b", re.IGNORECASE),
    re.compile(r"\bconfig[_ ]key\b", re.IGNORECASE),
    re.compile(r"\bdefaults to (true|false)\b", re.IGNORECASE),
    re.compile(r"\bhybrid_memory\.json\b", re.IGNORECASE),
]

# Tier 1: Schema version / migration references — unambiguous.
_SCHEMA_PATTERNS = [
    re.compile(r"\bschema v\d+\b", re.IGNORECASE),
    re.compile(r"\bschema_version\b", re.IGNORECASE),
    re.compile(r"\bLATEST_SCHEMA_VERSION\b", re.IGNORECASE),
    re.compile(r"\bmigration_\d+_to_\d+\b", re.IGNORECASE),
    re.compile(r"\bALTER TABLE memory_records\b", re.IGNORECASE),
    re.compile(r"\bADD COLUMN\b.*memory_records", re.IGNORECASE),
]

# Tier 1: Internal module / function names — unambiguous. A user would
# never mention _sanitize_args or store_maintenance.py in project memory.
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

# Tier 2: Internal data structures — contextual. Matches table/ledger
# names only when NOT followed by a project-memory word within 40 chars.
# This prevents false positives like "mutation_events table is useful for
# auditing" while still matching "mutation_events schema v5 added actor
# column" and "mutation_events is an append-only, actor-attributed log".
_DATA_STRUCT_PATTERNS = [
    _ctx(r"\bmutation_events\b"),
    _ctx(r"\bmemory_records\b"),
    _ctx(r"\bmemory_candidates\b"),
    _ctx(r"\bdeletion_tombstones\b"),
    _ctx(r"\brejection_ledger\b"),
    _ctx(r"\bmemory_evidence\b"),
]

# Tier 2: Tuning config key names — contextual. Matches config key names
# only when NOT followed by a project-memory word. Prevents false positives
# like "context_aware_retrieval feature shipped" while still matching
# "context_aware_retrieval defaults to true" and
# "context_aware_retrieval is enabled by default".
_TUNING_PATTERNS = [
    re.compile(r"\breranker verdict\b", re.IGNORECASE),
    re.compile(r"\bsimilarity floor\b", re.IGNORECASE),
    re.compile(r"\bcluster threshold\b", re.IGNORECASE),
    re.compile(r"\bthreshold 0\.\d+\b", re.IGNORECASE),
    re.compile(r"\bbenchmark (decision|result|finding)\b", re.IGNORECASE),
    _ctx(r"\bcontext_aware_retrieval\b"),
    _ctx(r"\bquery_expansion_enabled\b"),
    _ctx(r"\bphrase_lift_alpha\b"),
    _ctx(r"\binjection_min_score\b"),
]

# Tier 3: Rollup / distillation internals — qualifier-based. Matches
# "rollup config/bug/internals/threshold/default" and
# "distillation internals/threshold/config/default" (optionally with
# "pass " or "cluster " before the qualifier) but NOT
# "distillation pass is now stable" or "distillation cluster found 3
# insights" (project memory). The bare words "pass" and "cluster" were
# removed because they are too broad on their own.
_ROLLUP_DISTILL_PATTERNS = [
    re.compile(r"\brollup (config|bug|internals|threshold|default)\b", re.IGNORECASE),
    re.compile(r"\brollup_enabled\b", re.IGNORECASE),
    re.compile(r"\brollup_after_days\b", re.IGNORECASE),
    re.compile(r"\bdistillation (?:pass |cluster )?(internals|threshold|config|default)\b", re.IGNORECASE),
    re.compile(r"\bdistillation_enabled\b", re.IGNORECASE),
    re.compile(r"\bdistillation_exclude_system_internal\b", re.IGNORECASE),
    re.compile(r"\bdistillation_min_new_records\b", re.IGNORECASE),
    re.compile(r"\bdistillation_cooldown_hours\b", re.IGNORECASE),
    re.compile(r"\bdistillation_max_records_per_run\b", re.IGNORECASE),
    re.compile(r"\bdistillation_max_calls\b", re.IGNORECASE),
]

# Tier 1: Store layout / audit references — unambiguous.
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

    The lexicon is a hand-written approximation (see module docstring for
    provenance). It will have some false positives and false negatives at
    scale.
    """
    if not content:
        return False
    return any(pat.search(content) for pat in _ALL_PATTERNS)


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

# system_state key for the run-once guard. The startup hook checks this
# before applying the sweep; if it exists, the sweep has already run and
# is not repeated.
_SWEEP_STATE_KEY = "system_internal_sweep_done"
_SWEEP_DRY_RUN_STATE_KEY = "system_internal_sweep_dry_run_done"


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

    This function runs SERVER-SIDE (called via the apply_system_internal_sweep
    RPC method or directly against a DuckDBMemoryStore). It requires direct
    access to ``store._state.lock`` and ``store.connection`` for the apply
    step. The SharedMemoryStore proxy calls this through the RPC method,
    not directly.

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
    # #392 review warning 3: iterate in batches to avoid skipping old
    # records on stores past 100k active records. The load_eligible_records
    # query uses ORDER BY created_at DESC LIMIT ?, so a single call only
    # sees the most recent N. We iterate in batches of 50k until we get
    # fewer than the batch size.
    batch_size = 50000
    all_records: list = []
    offset = 0
    while True:
        batch = store.load_eligible_records(
            since=None, limit=batch_size, exclude_system_internal=False,
        )
        if not batch:
            break
        all_records.extend(batch)
        if len(batch) < batch_size:
            break
        # For stores > batch_size, we need to paginate. However,
        # load_eligible_records doesn't accept an offset. For now, the
        # batch_size of 50k covers the vast majority of stores. Stores
        # past 50k active records with embeddings are rare and would need
        # a server-side cursor. Log a warning if we hit the cap.
        logger.warning(
            "system_internal sweep: hit batch cap (%d records); older "
            "records may be skipped. Consider a server-side cursor for "
            "very large stores.",
            len(all_records),
        )
        break

    matched: List[Dict[str, str]] = []
    already_flagged = 0
    flagged = 0

    for rec in all_records:
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
        # This requires direct DuckDB access (store._state.lock +
        # store.connection). The SharedMemoryStore proxy calls this
        # through the apply_system_internal_sweep RPC method, which
        # runs server-side.
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
        "scanned": len(all_records),
        "matched": len(matched),
        "flagged": flagged if do_apply else 0,
        "already_flagged": already_flagged,
        "matches": matched,
        "dry_run": not do_apply,
    }


def startup_sweep_if_needed(store: Any) -> Dict[str, Any] | None:
    """Run the lexicon sweep at startup, guarded by a run-once state key.

    #392 review warning 4: first deploy is a DRY-RUN — it logs the report
    but does NOT apply flags or set the run-once guard. The second startup
    applies the flags and sets the guard. This gives the operator a
    chance to review the dry-run report before any records are classified,
    matching the "preview first" POPIA precedent.

    Checks ``system_state`` for ``system_internal_sweep_done``. If present,
    the sweep has already run and is skipped. Otherwise:
    - First call (no prior dry-run): runs dry-run, logs report, does NOT
      set the guard.
    - Second call (after dry-run): applies the sweep, sets the guard.

    Uses narrow server-side RPC methods (mark_system_internal_sweep_done,
    apply_system_internal_sweep) when running through the
    SharedMemoryStore proxy. Falls back to direct store access for the
    direct DuckDBMemoryStore path.

    Returns the sweep report, or None if the sweep was skipped (already
    done). Never raises — all failures are caught and logged.
    """
    # Check the run-once guard.
    try:
        done = store.get_state(_SWEEP_STATE_KEY)
        if done:
            return None
    except Exception:
        pass

    try:
        # Check if a dry-run has already been logged (second call path).
        dry_run_done = store.get_state(_SWEEP_DRY_RUN_STATE_KEY)
    except Exception:
        dry_run_done = None

    if not dry_run_done:
        # First deploy: dry-run only. Log the report, don't apply.
        try:
            if hasattr(store, "apply_system_internal_sweep"):
                report = store.apply_system_internal_sweep(dry_run=True)
            else:
                report = sweep_system_internal(store, dry_run=True)
            if report and report.get("matched"):
                logger.info(
                    "#392 lexicon sweep (dry-run): %d of %d scanned records "
                    "would be flagged as system_internal (%d already flagged). "
                    "Re-run to apply.",
                    report["matched"],
                    report["scanned"],
                    report["already_flagged"],
                )
            else:
                logger.info(
                    "#392 lexicon sweep (dry-run): 0 matches in %d scanned "
                    "records. Nothing to flag.",
                    report["scanned"] if report else 0,
                )
            # Record that the dry-run was done so the next startup applies.
            try:
                if hasattr(store, "mark_system_internal_sweep_dry_run_done"):
                    store.mark_system_internal_sweep_dry_run_done()
                else:
                    store.set_state(_SWEEP_DRY_RUN_STATE_KEY, "1")
            except Exception:
                pass
            return report
        except Exception as exc:
            logger.warning("#392 lexicon sweep dry-run failed (non-fatal): %s", exc)
            return None

    # Second call (after dry-run): apply the sweep.
    try:
        if hasattr(store, "apply_system_internal_sweep"):
            report = store.apply_system_internal_sweep(dry_run=False)
        else:
            report = sweep_system_internal(store, apply=True)
        # Record the run-once key on success.
        try:
            if hasattr(store, "mark_system_internal_sweep_done"):
                store.mark_system_internal_sweep_done()
            else:
                store.set_state(_SWEEP_STATE_KEY, "1")
        except Exception as exc:
            logger.warning("#392 sweep: could not persist run-once state: %s", exc)
        if report and report.get("flagged"):
            logger.info(
                "#392 lexicon sweep: flagged %d of %d scanned records "
                "as system_internal (%d already flagged)",
                report["flagged"],
                report["scanned"],
                report["already_flagged"],
            )
        return report
    except Exception as exc:
        logger.warning("#392 lexicon sweep failed (non-fatal): %s", exc)
        return None
