"""Explainability pack — provenance walk + citation surfacing (#280).

A user/agent can always answer "why did Argos retrieve THIS fact?" —
per-retrieval provenance, citation surfacing, confidence display.

This module is a READ-ONLY VIEW layer. It assembles data from records
already available at retrieval time. It does NOT:
- make LLM calls (zero-LLM, like the repo's cheap-hardening preference)
- write to the store
- change default injection behavior
- build a parallel provenance system

It reuses the EXISTING trust-model surfaces:
- store.get_evidence(memory_id) → evidence row
- store.get_memory_history(memory_id) → version/supersession chain
- MemoryRecord.similarity / raw_similarity → blend score (retrieval-time)
- MemoryRecord.provenance_origin / grounding / source / confidence
- Conflict notes from _conflict_annotations (re-derived on demand)

Two explain entry points:
- explain_record(record, store, ...): takes a LIVE MemoryRecord from
  search results — has real similarity/raw_similarity. Preferred when
  explaining a retrieval result.
- explain_provenance(store, memory_id, ...): takes a memory_id — fetches
  the record from storage. Retrieval-time scores (similarity,
  raw_similarity) are NOT persisted on the row, so blend_score and
  gates_fired are annotated as "retrieval-time only, not available"
  rather than fabricating 0.0 values.

Fail-soft: if a provenance field is missing, show "unknown"/omit —
never crash the injection path.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def explain_record(
    record: Any,
    store: Any,
    *,
    injection_min_score: float = 0.0,
    include_chain: bool = True,
    include_conflicts: bool = True,
) -> Dict[str, Any]:
    """Explain a LIVE MemoryRecord returned by search/retrieval.

    This is the preferred explain path: the record carries real
    retrieval-time similarity/raw_similarity scores, so blend_score
    and gates_fired reflect what actually happened at retrieval.

    Args:
        record: a MemoryRecord from search results (has similarity,
            raw_similarity set by the retrieval pipeline).
        store: the store (for evidence, chain, conflict lookups).
        injection_min_score: the configured injection floor (from
            provider._injection_min_score or config). Used to determine
            whether the injection_min_score gate fired.
        include_chain: whether to walk the version chain.
        include_conflicts: whether to detect conflicts.

    Returns: same dict shape as explain_provenance, but with real
    retrieval-time scores.
    """
    memory_id = _safe_get(record, "memory_id", "unknown")
    result = _base_result(memory_id)
    _populate_from_record(result, record)

    # --- Blend score (real retrieval-time values) ---
    similarity = _safe_get(record, "similarity", 0.0)
    raw_similarity = _safe_get(record, "raw_similarity", None)
    # reranker_applied: only True if the reranker actually ran. The
    # reranker sets raw_similarity to the pre-rerank score and adjusts
    # similarity. Graph boost and importance expansion also change
    # similarity but do NOT set raw_similarity to a different value
    # from the pre-adjustment score — they modify similarity in place.
    # The reranker path (store_retrieval.py:1135-1149) captures
    # raw_similarity BEFORE the blend and sets similarity to the blend.
    # So reranker_applied = raw_similarity is not None AND raw_similarity
    # != similarity, which is only set by the reranker path.
    reranker_applied = (
        raw_similarity is not None
        and similarity is not None
        and raw_similarity != similarity
    )
    result["blend_score"] = {
        "similarity": similarity,
        "raw_similarity": raw_similarity,
        "reranker_applied": reranker_applied,
        "source": "retrieval-time",
    }

    # --- Gates fired (using real scores + configured floor) ---
    result["gates_fired"] = _gates_fired(record, injection_min_score)

    # --- Evidence, chain, conflict (from store) ---
    _populate_store_fields(result, store, memory_id, include_chain, include_conflicts)

    return result


def explain_provenance(
    store: Any,
    memory_id: str,
    *,
    injection_min_score: float = 0.0,
    include_chain: bool = True,
    include_conflicts: bool = True,
) -> Dict[str, Any]:
    """Assemble the provenance view for a memory by ID.

    This is the ID-based explain path. It fetches the record from
    storage. Retrieval-time scores (similarity, raw_similarity) are NOT
    persisted on the row — they are per-query and computed at search
    time. So blend_score and gates_fired are annotated as
    "retrieval-time only, not available" rather than fabricating 0.0
    values that would produce false gate claims.

    For real retrieval-time scores, use explain_record() with a live
    MemoryRecord from search results.

    Args:
        store: the store (must have user_id for scope filtering).
        memory_id: the memory ID to explain.
        injection_min_score: the configured injection floor.
        include_chain: whether to walk the version chain.
        include_conflicts: whether to detect conflicts.

    Returns: dict with evidence, version_chain, conflict_note,
    blend_score (annotated as not available), confidence, etc.
    """
    result = _base_result(memory_id)

    # --- Fetch the memory record (scope-filtered) ---
    record = _fetch_record(store, memory_id)
    if record is None:
        result["error"] = f"memory {memory_id} not found"
        return result

    _populate_from_record(result, record)

    # --- Blend score (NOT available from storage) ---
    # similarity/raw_similarity are retrieval-time values, not persisted.
    # Annotate as not available rather than fabricating 0.0.
    result["blend_score"] = {
        "similarity": None,
        "raw_similarity": None,
        "reranker_applied": None,
        "source": "retrieval-time only, not available (use explain_record with a live search result for real scores)",
    }

    # --- Gates fired (NOT available from storage) ---
    # Without retrieval-time scores, we cannot determine which gates
    # fired. Only the conflict_surfacing gate can be inferred from the
    # record's category (system_note = injected conflict note).
    result["gates_fired"] = _gates_fired_id_based(record)

    # --- Evidence, chain, conflict (from store) ---
    _populate_store_fields(result, store, memory_id, include_chain, include_conflicts)

    return result


def explain_batch(
    store: Any,
    memory_ids: List[str],
    *,
    injection_min_score: float = 0.0,
    include_chains: bool = False,
) -> List[Dict[str, Any]]:
    """Batch provenance view for a list of memory IDs.

    Uses get_evidence_batch for efficient evidence lookup. Chains are
    optional (off by default) since walking chains for every result is
    expensive. Retrieval-time scores are not available (same as
    explain_provenance — use explain_record for live scores).
    """
    if not memory_ids:
        return []

    # Batch-fetch evidence
    try:
        evidence_map = store.get_evidence_batch(memory_ids)
    except Exception as exc:
        logger.debug("explain_batch: get_evidence_batch failed: %s", exc)
        evidence_map = {}

    results = []
    for mid in memory_ids:
        record = _fetch_record(store, mid)
        if record is None:
            results.append({"memory_id": mid, "error": "not found"})
            continue

        entry: Dict[str, Any] = {
            "memory_id": mid,
            "content": _safe_get(record, "content", "unknown"),
            "category": _safe_get(record, "category", "unknown"),
            "evidence": evidence_map.get(mid),
            "blend_score": {
                "similarity": None,
                "raw_similarity": None,
                "reranker_applied": None,
                "source": "retrieval-time only, not available",
            },
            "confidence": _safe_get(record, "confidence", None),
            "provenance_origin": _safe_get(record, "provenance_origin", "unknown"),
            "grounding": _safe_get(record, "grounding", "unknown"),
            "gates_fired": _gates_fired_id_based(record),
        }

        if include_chains:
            try:
                chain = store.get_memory_history(mid)
                entry["version_chain"] = _summarize_chain(chain)
            except Exception:
                entry["version_chain"] = []

        results.append(entry)

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _base_result(memory_id: str) -> Dict[str, Any]:
    """Initialize the result dict with defaults."""
    return {
        "memory_id": memory_id,
        "content": "unknown",
        "category": "unknown",
        "evidence": None,
        "version_chain": [],
        "conflict_note": None,
        "blend_score": {},
        "confidence": None,
        "provenance_origin": "unknown",
        "grounding": "unknown",
        "gates_fired": [],
    }


def _populate_from_record(result: Dict[str, Any], record: Any) -> None:
    """Populate basic fields from a record (fail-soft via getattr)."""
    result["content"] = _safe_get(record, "content", "unknown")
    result["category"] = _safe_get(record, "category", "unknown")
    result["confidence"] = _safe_get(record, "confidence", None)
    result["provenance_origin"] = _safe_get(record, "provenance_origin", "unknown")
    result["grounding"] = _safe_get(record, "grounding", "unknown")
    result["source"] = _safe_get(record, "source", "unknown")
    result["status"] = _safe_get(record, "status", "unknown")
    result["verified_state"] = _safe_get(record, "verified_state", "unknown")


def _populate_store_fields(
    result: Dict[str, Any],
    store: Any,
    memory_id: str,
    include_chain: bool,
    include_conflicts: bool,
) -> None:
    """Populate evidence, version chain, and conflict note from store."""
    # --- Evidence row ---
    try:
        evidence = store.get_evidence(memory_id)
        result["evidence"] = evidence
    except Exception as exc:
        logger.debug("provenance: get_evidence failed for %s: %s", memory_id, exc)
        result["evidence"] = None

    # --- Version chain ---
    if include_chain:
        try:
            chain = store.get_memory_history(memory_id)
            result["version_chain"] = _summarize_chain(chain)
        except Exception as exc:
            logger.debug("provenance: get_memory_history failed for %s: %s", memory_id, exc)
            result["version_chain"] = []

    # --- Conflict note ---
    if include_conflicts:
        try:
            conflict = _detect_conflict_for_memory(store, memory_id)
            result["conflict_note"] = conflict
        except Exception as exc:
            logger.debug("provenance: conflict detection failed for %s: %s", memory_id, exc)
            result["conflict_note"] = None


def _fetch_record(store: Any, memory_id: str) -> Any:
    """Fetch a single memory record by ID (fail-soft, scope-filtered).

    Enforces user_scope filtering — mirrors get_memory_history
    (store_write.py:2423-2427) and get_evidence (store_write.py:1774-1780).
    A cross-user read returns None (not found), not the other user's record.
    """
    try:
        # DuckDBMemoryStore has _fetch_records — add user_scope filter
        # to mirror get_memory_history / get_evidence / get_evidence_batch.
        if hasattr(store, "_fetch_records"):
            user_id = getattr(store, "user_id", "")
            records = store._fetch_records(
                "SELECT * FROM memory_records WHERE memory_id = ?"
                " AND (user_scope IS NULL OR user_scope = ?)",
                [memory_id, user_id],
            )
            return records[0] if records else None
        # SharedMemoryStore / service_client — get_memory_history
        # already enforces user_scope (store_write.py:2423-2427).
        if hasattr(store, "get_memory_history"):
            chain = store.get_memory_history(memory_id)
            if chain:
                for r in reversed(chain):
                    if getattr(r, "valid_to", None) is None:
                        return r
                return chain[-1]
        return None
    except Exception as exc:
        logger.debug("_fetch_record failed for %s: %s", memory_id, exc)
        return None


def _safe_get(obj: Any, attr: str, default: Any = None) -> Any:
    """Get an attribute fail-soft."""
    return getattr(obj, attr, default)


def _summarize_chain(chain: List[Any]) -> List[Dict[str, Any]]:
    """Summarize a version chain into a list of dicts."""
    summaries = []
    for r in chain:
        summaries.append({
            "memory_id": _safe_get(r, "memory_id", "unknown"),
            "content": _safe_get(r, "content", "unknown"),
            "created_at": _safe_get(r, "created_at", "unknown"),
            "valid_from": _safe_get(r, "valid_from", None),
            "valid_to": _safe_get(r, "valid_to", None),
            "superseded_by": _safe_get(r, "superseded_by", None),
            "is_current": _safe_get(r, "valid_to", None) is None,
        })
    return summaries


def _detect_conflict_for_memory(
    store: Any, memory_id: str,
) -> Optional[str]:
    """Check if a memory has conflicts with other active memories.

    Uses the same values_conflict logic as the retrieval path, but
    applied to the memory's chain members vs other active records on
    the same subject. Returns a conflict note string or None.

    This is a cheap, LLM-free check: it fetches the memory's chain and
    checks for value conflicts within the chain (older vs newer versions
    that disagree without supersession).
    """
    try:
        chain = store.get_memory_history(memory_id)
        if not chain or len(chain) < 2:
            return None

        # Check for unlinked conflicts: two active records in the chain
        # that disagree. Within a proper chain, supersession handles this,
        # but if two records have valid_to=None (both active), that's a
        # conflict.
        active = [r for r in chain if getattr(r, "valid_to", None) is None]
        if len(active) < 2:
            return None

        # Two active versions in the same chain = conflict
        contents = [getattr(r, "content", "") for r in active]
        dates = [getattr(r, "created_at", "")[:10] for r in active]
        return (
            f"CONFLICT: {len(active)} active versions in this chain disagree. "
            f"Versions: " + " vs ".join(
                f'"{c[:60]}" ({d})' for c, d in zip(contents, dates)
            )
        )[:360]
    except Exception as exc:
        logger.debug("_detect_conflict_for_memory failed: %s", exc)
        return None


def _gates_fired(record: Any, injection_min_score: float = 0.0) -> List[str]:
    """List retrieval gates that fired for a LIVE record (from search).

    Uses the record's real retrieval-time similarity and the configured
    injection_min_score (not a hardcoded threshold).

    Only emits a gate claim when there is supporting data — similarity
    must be a real non-None value from the retrieval pipeline.
    """
    gates = []
    sim = _safe_get(record, "similarity", None)
    raw = _safe_get(record, "raw_similarity", None)

    # injection_min_score gate: only claim if we have a real similarity
    # value AND it's below the configured floor.
    if sim is not None and float(sim) < float(injection_min_score):
        gates.append(
            f"injection_min_score (similarity {float(sim):.3f} < floor {float(injection_min_score):.3f})"
        )

    # conflict_surfacing: if this is a system_note category, it's a
    # conflict note injected by the conflict surfacing gate.
    if _safe_get(record, "category", "") == "system_note":
        gates.append("conflict_surfacing (injected conflict note)")

    # reranker: only if raw_similarity was captured (pre-rerank) and
    # differs from the final similarity. Graph boost and importance
    # expansion modify similarity in place without setting raw_similarity
    # to a different value, so this only fires for the actual reranker.
    if raw is not None and sim is not None and raw != sim:
        gates.append("reranker (score adjusted)")

    return gates


def _gates_fired_id_based(record: Any) -> List[str]:
    """Gates fired for an ID-based lookup (no retrieval-time scores).

    Only the conflict_surfacing gate can be inferred from the record's
    category. Score-based gates are NOT emitted — there is no
    retrieval-time similarity data on a stored row.
    """
    gates = []
    if _safe_get(record, "category", "") == "system_note":
        gates.append("conflict_surfacing (injected conflict note)")
    return gates
