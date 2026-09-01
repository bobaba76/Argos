"""Spec-05 (#67): presence-aware namespace partition for injection.

The injection cap (96 by default) is split between conversation-sourced and
document-sourced memories so a flood of document-facts cannot crowd
conversational memories out of the injected context window (and vice versa).

The floors are **presence-aware**: they only bite when BOTH namespaces are
populated. A single-namespace store (e.g. a practice store before any chat
notes exist) gets all ``cap`` slots — zero waste.

v1 floors are explicitly **untuned** (static 24/24, inverted 12/40 for
client-scoped queries). Dynamic v2 floors are gated on real document-fact
recall measurement (the measure-before-ship rule, 13/8).

Deterministic, no LLM calls, no new deps. Pure function — the provider calls
this on the merged retrieval results just before formatting the injection
block.
"""
from __future__ import annotations

from typing import Any, List

# v1 static floors (untuned). Mixed-store floors reserve slots for each
# namespace so neither side can be completely crowded out. Client-scoped
# queries invert: document-facts are the primary signal (40), conversation
# notes are secondary (12).
_FLOOR_CONVERSATION = 24
_FLOOR_DOCUMENT = 24
_FLOOR_CONVERSATION_CLIENT_SCOPED = 12
_FLOOR_DOCUMENT_CLIENT_SCOPED = 40


def _namespace_of(record: Any) -> str:
    """Return the record's namespace, defaulting to 'conversation'."""
    return getattr(record, "namespace", None) or "conversation"


def _score_of(record: Any) -> float:
    """Return the unified score used for remainder fill (similarity)."""
    return float(getattr(record, "similarity", 0.0) or 0.0)


def partition_by_namespace(
    results: List[Any],
    *,
    cap: int,
    client_scoped: bool = False,
) -> List[Any]:
    """Split *results* into a presence-aware, floor-reserved injection list.

    Args:
        results: Merged retrieval results (MemoryRecord-like), already ranked
            by unified score. The caller is expected to pass the full
            candidate pool (not pre-truncated to *cap*).
        cap: Maximum items to return (the injection budget, e.g. 96).
        client_scoped: When True, the query is client-scoped (document floor
            40, conversation floor 12). When False, floors are 24/24.

    Returns:
        Up to ``cap`` records, in unified-score order across both namespaces.
        The floors guarantee each populated namespace gets at least its floor
        (clamped to availability); the remainder is filled by score.

    Presence-aware short-circuit: if only one namespace is populated, the
    floor for the empty namespace reserves nothing — all ``cap`` slots go to
    the populated side.
    """
    if not results or cap <= 0:
        return []

    conv = [r for r in results if _namespace_of(r) == "conversation"]
    doc = [r for r in results if _namespace_of(r) == "document"]

    # Presence-aware short-circuit: a single-namespace store gets all cap
    # slots. Zero waste — no reservation for an empty side.
    if not conv:
        return doc[:cap]
    if not doc:
        return conv[:cap]

    # Both namespaces populated — apply floors.
    if client_scoped:
        floor_conv = _FLOOR_CONVERSATION_CLIENT_SCOPED
        floor_doc = _FLOOR_DOCUMENT_CLIENT_SCOPED
    else:
        floor_conv = _FLOOR_CONVERSATION
        floor_doc = _FLOOR_DOCUMENT

    # Clamp floors to availability and to cap.
    floor_conv = min(floor_conv, len(conv), cap)
    floor_doc = min(floor_doc, len(doc), cap)

    # Take the top-N from each side by score (results are pre-ranked, but
    # we sort each side independently to be safe).
    conv_sorted = sorted(conv, key=_score_of, reverse=True)
    doc_sorted = sorted(doc, key=_score_of, reverse=True)

    taken_conv = conv_sorted[:floor_conv]
    taken_doc = doc_sorted[:floor_doc]
    taken_ids = {id(r) for r in taken_conv} | {id(r) for r in taken_doc}

    allocated = len(taken_conv) + len(taken_doc)
    remainder = cap - allocated

    # Fill the remainder from the leftover pool, by unified score.
    leftover = [r for r in results if id(r) not in taken_ids]
    leftover_sorted = sorted(leftover, key=_score_of, reverse=True)
    taken_remainder = leftover_sorted[:max(0, remainder)]

    # Re-rank the final set by unified score so the injected block stays
    # relevance-ordered (not namespace-blocked).
    final = taken_conv + taken_doc + taken_remainder
    final.sort(key=_score_of, reverse=True)
    return final[:cap]
