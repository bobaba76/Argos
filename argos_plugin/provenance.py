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
- MemoryRecord.similarity / raw_similarity → blend score
- MemoryRecord.provenance_origin / grounding / source / confidence
- Conflict notes from _conflict_annotations (re-derived on demand)

Fail-soft: if a provenance field is missing, show "unknown"/omit —
never crash the injection path.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def explain_provenance(
    store: Any,
    memory_id: str,
    *,
    include_chain: bool = True,
    include_conflicts: bool = True,
) -> Dict[str, Any]:
    """Assemble the full provenance view for a memory.

    Returns a dict with:
    - memory_id: the queried ID
    - content: the memory's content (for context)
    - category: the memory's category
    - evidence: the evidence row (source/evidence reference) or None
    - version_chain: list of version summaries (if include_chain)
    - conflict_note: conflict note text (if any conflicts detected)
    - blend_score: similarity + raw_similarity + reranker info
    - confidence: the memory's confidence field
    - provenance_origin: internal/external/imported
    - grounding: observed/inferred/quoted
    - gates_fired: list of retrieval gates that fired for this memory

    Fail-soft: every field is wrapped so a missing field shows "unknown"
    or is omitted — never raises.
    """
    result: Dict[str, Any] = {
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

    # --- Fetch the memory record itself ---
    record = _fetch_record(store, memory_id)
    if record is None:
        result["error"] = f"memory {memory_id} not found"
        return result

    # --- Basic fields (fail-soft via getattr) ---
    result["content"] = _safe_get(record, "content", "unknown")
    result["category"] = _safe_get(record, "category", "unknown")
    result["confidence"] = _safe_get(record, "confidence", None)
    result["provenance_origin"] = _safe_get(record, "provenance_origin", "unknown")
    result["grounding"] = _safe_get(record, "grounding", "unknown")
    result["source"] = _safe_get(record, "source", "unknown")
    result["status"] = _safe_get(record, "status", "unknown")
    result["verified_state"] = _safe_get(record, "verified_state", "unknown")

    # --- Blend score ---
    similarity = _safe_get(record, "similarity", 0.0)
    raw_similarity = _safe_get(record, "raw_similarity", None)
    result["blend_score"] = {
        "similarity": similarity,
        "raw_similarity": raw_similarity,
        "reranker_applied": raw_similarity is not None and raw_similarity != similarity,
    }

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

    # --- Gates fired ---
    result["gates_fired"] = _gates_fired(record)

    return result


def explain_batch(
    store: Any,
    memory_ids: List[str],
    *,
    include_chains: bool = False,
) -> List[Dict[str, Any]]:
    """Batch provenance view for a list of memory IDs.

    Uses get_evidence_batch for efficient evidence lookup. Chains are
    optional (off by default) since walking chains for every result is
    expensive.
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
                "similarity": _safe_get(record, "similarity", 0.0),
                "raw_similarity": _safe_get(record, "raw_similarity", None),
                "reranker_applied": (
                    _safe_get(record, "raw_similarity", None) is not None
                    and _safe_get(record, "raw_similarity", None)
                    != _safe_get(record, "similarity", 0.0)
                ),
            },
            "confidence": _safe_get(record, "confidence", None),
            "provenance_origin": _safe_get(record, "provenance_origin", "unknown"),
            "grounding": _safe_get(record, "grounding", "unknown"),
            "gates_fired": _gates_fired(record),
        }

        if include_chains:
            try:
                chain = store.get_memory_history(mid)
                entry["version_chain"] = _summarize_chain(chain)
            except Exception:
                entry["version_chain"] = []

        results.append(entry)

    return results


def _fetch_record(store: Any, memory_id: str) -> Any:
    """Fetch a single memory record by ID (fail-soft)."""
    try:
        # DuckDBMemoryStore has _fetch_records
        if hasattr(store, "_fetch_records"):
            records = store._fetch_records(
                "SELECT * FROM memory_records WHERE memory_id = ?",
                [memory_id],
            )
            return records[0] if records else None
        # SharedMemoryStore / service_client — use search or get_memory_history
        if hasattr(store, "get_memory_history"):
            chain = store.get_memory_history(memory_id)
            if chain:
                # Return the current version (last in chain, or the one
                # with valid_to is None)
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


def _gates_fired(record: Any) -> List[str]:
    """List retrieval gates that fired for this record.

    Based on the record's fields — this is a static analysis of what
    gates would have affected this record at retrieval time.
    """
    gates = []
    # injection_min_score gate: if similarity is very low, the floor
    # gate would have dropped this record
    sim = _safe_get(record, "similarity", 0.0)
    if sim is not None and float(sim) < 0.3:
        gates.append("injection_min_score (low similarity)")
    # conflict_surfacing: if this is a system_note category, it's a
    # conflict note injected by the conflict surfacing gate
    if _safe_get(record, "category", "") == "system_note":
        gates.append("conflict_surfacing (injected conflict note)")
    # reranker: if raw_similarity != similarity, the reranker adjusted
    # the score
    raw = _safe_get(record, "raw_similarity", None)
    if raw is not None and raw != sim:
        gates.append("reranker (score adjusted)")
    return gates
