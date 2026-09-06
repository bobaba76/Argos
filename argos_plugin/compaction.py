"""#281: Schedule-aware self-compaction (token-budget control).

Automatic consolidation of stale/duplicate/low-value memories to control
injection token bloat, WITHOUT losing provenance or history.

This module is a SCHEDULE-AWARE wrapper around the existing consolidate()
method in store_maintenance.py. It adds:

1. Cooldown-gated scheduling (reuses the same system_state KV pattern
   as rollup.py / distillation.py — no new scheduler).
2. Token-budget measurement (estimates injected-token reduction by
   counting quarantined records × avg tokens/record).
3. Provenance/evidence chain integrity check (verifies no evidence rows
   or version chains are orphaned by the quarantine).
4. A compaction aggressiveness knob (conservative default).

DESIGN CONSTRAINTS (from the issue):
- Compaction ONLY — no changes to ranking/relevance or the 96-item cap.
- No LLM required (deterministic + embeddings similarity only).
- Reversible quarantine (NEVER hard-delete) — mirrors existing quarantine
  semantics (quarantine_memory / restore_memory).
- Schedule-aware: integrates with existing session-end hook (NOT inline
  in the hot path).
- Keep provenance/evidence chains intact for anything consolidated.

OPEN QUESTIONS (decided with cheapest reasonable default):
- Where does the schedule live? → session-end hook (same as rollup/
  distillation), cooldown-gated by compaction_interval_days.
- Which threshold is "stale"? → reuse existing TTL + recency signals
  (consolidate() already uses expires_at, retrieval_count, age).
- Config knob? → yes: compaction_enabled + compaction_aggressiveness
  (0.0=off, 1.0=conservative, 2.0=aggressive). Default: off (opt-in).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_STATE_KEY_LAST_RUN = "compaction_last_run"
_STATE_KEY_LAST_COUNT = "compaction_last_count"

# Token estimation: ~4 chars per token (conservative for English text).
# Used for budget-reduction reporting only — not for any retrieval gate.
_CHARS_PER_TOKEN = 4.0

# Aggressiveness presets: map the scalar knob to consolidate() params.
# 1.0 = conservative (fewer candidates, higher similarity threshold)
# 2.0 = aggressive (more candidates, lower similarity threshold)
_AGGRESSIVENESS_PRESETS = {
    1.0: {
        "max_actions": 25,
        "min_age_days": 30,
        "duplicate_min_similarity": 0.92,
        "duplicate_semantic_max_pairs": 20000,
    },
    2.0: {
        "max_actions": 100,
        "min_age_days": 14,
        "duplicate_min_similarity": 0.85,
        "duplicate_semantic_max_pairs": 50000,
    },
}


def _get_default_params(aggressiveness: float) -> Dict[str, Any]:
    """Map aggressiveness scalar to consolidate() params.

    Values between 1.0 and 2.0 are interpolated linearly.
    Below 1.0 clamps to conservative; above 2.0 clamps to aggressive.
    """
    a = max(1.0, min(2.0, float(aggressiveness)))
    conservative = _AGGRESSIVENESS_PRESETS[1.0]
    aggressive = _AGGRESSIVENESS_PRESETS[2.0]
    t = (a - 1.0) / 1.0  # 0.0 at conservative, 1.0 at aggressive
    return {
        key: int(conservative[key] + t * (aggressive[key] - conservative[key]))
        if isinstance(conservative[key], int)
        else round(conservative[key] + t * (aggressive[key] - conservative[key]), 4)
        for key in conservative
    }


def _get_last_run(store: Any) -> Optional[str]:
    """ISO timestamp of the last compaction run, or None."""
    try:
        return store.get_state(_STATE_KEY_LAST_RUN)
    except Exception:
        return None


def _advance_run_state(store: Any, records_processed: int) -> None:
    """Mark a run as completed (advances last_run + last_count)."""
    now = datetime.now(timezone.utc).isoformat()
    try:
        store.set_state(_STATE_KEY_LAST_RUN, now)
        store.set_state(_STATE_KEY_LAST_COUNT, str(records_processed))
    except Exception as exc:
        logger.debug("compaction: advance_run_state failed: %s", exc)


def _is_within_cooldown(store: Any, interval_days: int) -> bool:
    """Check if the last run was within the cooldown interval."""
    last = _get_last_run(store)
    if not last:
        return False
    try:
        last_dt = datetime.fromisoformat(last.replace("Z", "+00:00"))
        elapsed = datetime.now(timezone.utc) - last_dt
        return elapsed < timedelta(days=interval_days)
    except Exception:
        return False


def _estimate_token_reduction(
    store: Any, quarantined_ids: List[str],
) -> Dict[str, Any]:
    """Estimate the token-budget reduction from quarantined records.

    Measures the total content length of quarantined records and
    converts to an approximate token count. This is a reporting
    metric only — it does NOT gate retrieval or change injection.
    """
    if not quarantined_ids:
        return {
            "records_quarantined": 0,
            "chars_reclaimed": 0,
            "estimated_tokens_reclaimed": 0,
        }

    total_chars = 0
    for memory_id in quarantined_ids:
        try:
            records = store._fetch_records(
                "SELECT content FROM memory_records WHERE memory_id = ?",
                [memory_id],
            )
            if records:
                total_chars += len(records[0].content or "")
        except Exception:
            pass

    estimated_tokens = int(total_chars / _CHARS_PER_TOKEN)
    return {
        "records_quarantined": len(quarantined_ids),
        "chars_reclaimed": total_chars,
        "estimated_tokens_reclaimed": estimated_tokens,
    }


def _verify_provenance_intact(
    store: Any, quarantined_ids: List[str],
) -> Dict[str, Any]:
    """Verify that quarantining records did NOT orphan evidence rows
    or break version chains.

    Quarantine sets status='quarantined' but does NOT delete the row,
    so evidence rows (memory_evidence table) still point to valid
    memory_ids, and version chains (superseded_by) are intact.

    Returns a dict with:
    - evidence_rows_intact: True if all evidence rows for quarantined
      records still exist and point to the right memory_id.
    - version_chains_intact: True if no quarantined record's
      superseded_by points to a missing record.
    - orphaned_evidence_count: number of evidence rows that lost their
      memory record (should always be 0 — quarantine ≠ delete).
    """
    if not quarantined_ids:
        return {
            "evidence_rows_intact": True,
            "version_chains_intact": True,
            "orphaned_evidence_count": 0,
        }

    orphaned_evidence = 0
    broken_chains = 0

    for memory_id in quarantined_ids:
        # Check evidence rows still exist for this memory.
        try:
            with store._state.lock:
                ev_count = store.connection.execute(
                    "SELECT COUNT(*) FROM memory_evidence WHERE memory_id = ?",
                    [memory_id],
                ).fetchone()[0]
                # Evidence rows exist — they're NOT orphaned (the memory
                # record still exists with status='quarantined').
                # Orphaned would mean the memory_records row is GONE.
                mem_count = store.connection.execute(
                    "SELECT COUNT(*) FROM memory_records WHERE memory_id = ?",
                    [memory_id],
                ).fetchone()[0]
                if ev_count > 0 and mem_count == 0:
                    orphaned_evidence += 1
        except Exception as exc:
            logger.debug("provenance check: evidence query failed for %s: %s", memory_id, exc)

        # Check version chain: if this record has superseded_by, the
        # target should still exist.
        try:
            records = store._fetch_records(
                "SELECT superseded_by FROM memory_records WHERE memory_id = ?",
                [memory_id],
            )
            if records and records[0].superseded_by:
                target_count = store.connection.execute(
                    "SELECT COUNT(*) FROM memory_records WHERE memory_id = ?",
                    [records[0].superseded_by],
                ).fetchone()[0]
                if target_count == 0:
                    broken_chains += 1
        except Exception as exc:
            logger.debug("provenance check: chain query failed for %s: %s", memory_id, exc)

    return {
        "evidence_rows_intact": orphaned_evidence == 0,
        "version_chains_intact": broken_chains == 0,
        "orphaned_evidence_count": orphaned_evidence,
        "broken_chain_count": broken_chains,
    }


def run_compaction(
    store: Any,
    *,
    interval_days: int = 7,
    aggressiveness: float = 1.0,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run one schedule-aware compaction pass.

    Gated by cooldown (interval_days). Delegates candidate selection +
    quarantine to the existing store.consolidate() method (which reuses
    TTL, recency, dedup, and quarantine semantics). Adds token-budget
    measurement and provenance integrity verification.

    Zero-LLM: consolidate() uses deterministic candidate selection
    (TTL, recency, content containment, embedding cosine similarity).
    No LLM calls.

    Fail-soft: any error returns a no-op report.

    Args:
        store: the DuckDBMemoryStore (or compatible).
        interval_days: minimum days between runs (cooldown gate).
        aggressiveness: 1.0=conservative, 2.0=aggressive. Interpolated.
        dry_run: if True, only report candidates without quarantining.

    Returns a report dict with:
    - ran: whether the compaction pass actually ran (vs. skipped).
    - skipped: reason if skipped (cooldown, disabled, etc.).
    - candidate_count: number of compaction candidates found.
    - quarantined_count: number of records quarantined (0 if dry_run).
    - quarantined_ids: list of quarantined memory IDs.
    - token_reduction: estimated token-budget reduction metrics.
    - provenance_intact: provenance/evidence chain integrity check.
    - reason_counts: candidates by reason (expired, stale, duplicate).
    """
    interval_days = max(1, int(interval_days))

    report: Dict[str, Any] = {
        "ran": False,
        "skipped": None,
        "candidate_count": 0,
        "quarantined_count": 0,
        "quarantined_ids": [],
        "token_reduction": {},
        "provenance_intact": {},
        "reason_counts": {},
        "aggressiveness": float(aggressiveness),
        "dry_run": bool(dry_run),
    }

    # Cooldown gate.
    if not dry_run and _is_within_cooldown(store, interval_days):
        report["skipped"] = "cooldown"
        return report

    # Map aggressiveness to consolidate() params.
    params = _get_default_params(aggressiveness)

    # Delegate to the existing consolidate() method.
    try:
        result = store.consolidate(
            dry_run=dry_run,
            max_actions=params["max_actions"],
            min_age_days=params["min_age_days"],
            duplicate_min_similarity=params["duplicate_min_similarity"],
            duplicate_semantic_max_pairs=params["duplicate_semantic_max_pairs"],
        )
    except Exception as exc:
        logger.warning("compaction: consolidate() failed: %s", exc)
        report["skipped"] = "consolidate_error"
        report["error"] = str(exc)
        return report

    report["candidate_count"] = result.get("candidate_count", 0)
    report["quarantined_count"] = result.get("quarantined_count", 0)
    report["quarantined_ids"] = result.get("quarantined_ids", [])
    report["reason_counts"] = result.get("reason_counts", {})

    # Token-budget measurement.
    quarantined_ids = report["quarantined_ids"]
    if quarantined_ids:
        report["token_reduction"] = _estimate_token_reduction(store, quarantined_ids)
        report["provenance_intact"] = _verify_provenance_intact(store, quarantined_ids)
    else:
        report["token_reduction"] = {
            "records_quarantined": 0,
            "chars_reclaimed": 0,
            "estimated_tokens_reclaimed": 0,
        }
        report["provenance_intact"] = {
            "evidence_rows_intact": True,
            "version_chains_intact": True,
            "orphaned_evidence_count": 0,
        }

    # Advance run state (only for real runs, not dry runs).
    if not dry_run:
        _advance_run_state(store, report["candidate_count"])

    report["ran"] = True
    logger.info(
        "compaction: %d candidates, %d quarantined, ~%d tokens reclaimed",
        report["candidate_count"],
        report["quarantined_count"],
        report["token_reduction"].get("estimated_tokens_reclaimed", 0),
    )
    return report
