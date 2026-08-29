"""Structural-loss guard for content rewrites (#42).

Before any LLM rewrite (distill, consolidation, merge) overwrites stored
content, deterministically count what would be **deleted** and merge it
back. Additions are ignored, so enrichment passes cleanly; deletion
triggers repair rather than refusal — the rewrite succeeds but cannot
destroy.

The guard compares parsed existing content against the proposed replacement
— snippets, relations, fields, scalars and array items — counts loss per
category, and resolves it by merging the lost material back into the
proposal before write.

Asymmetry is the key: fires only on deletion, never on addition. A
proposal that adds content without removing any passes unchanged.

Outcome/decision-shaped records are an immutable append-only class,
exempt from merging and from dedup quarantine.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple


# --- Append-only categories (#42 companion rule) ---------------------------

# Outcome/decision-shaped records are immutable — they record what happened
# at a point in time and must never be merged, deduped, or quarantined.
# Examples: "surgery outcome: successful", "project decision: use Python 3.12".
APPEND_ONLY_CATEGORIES: Set[str] = {
    "outcome",
    "procedure_outcome",
    "decision",
}

# Payload kinds that mark a record as append-only.
APPEND_ONLY_PAYLOAD_KINDS: Set[str] = {
    "outcome",
    "procedure_outcome",
    "decision",
    "tripwatch",
}


def is_append_only(category: str, payload: dict | None = None) -> bool:
    """Check if a record is append-only (outcome/decision-shaped).

    Append-only records are exempt from merging and dedup quarantine.
    They are identified by category or by payload.kind.
    """
    if category and category.lower() in APPEND_ONLY_CATEGORIES:
        return True
    if payload and isinstance(payload, dict):
        kind = str(payload.get("kind", "") or "").strip().lower()
        if kind in APPEND_ONLY_PAYLOAD_KINDS:
            return True
    return False


# --- Structural parsing ----------------------------------------------------

# Split content into structural items: sentences, list items, and
# key-value pairs. This is deliberately simple — no NLP, no LLM.
# The goal is to detect what would be lost, not to understand semantics.

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_LIST_ITEM_RE = re.compile(r"^\s*[-\u2022\u25CF\u25CB]\s+(.+)$", re.MULTILINE)
_KV_RE = re.compile(r"^([^:]{1,60}):\s*(.+)$", re.MULTILINE)


@dataclass
class StructuralItems:
    """Parsed structural items from a piece of content."""
    sentences: Set[str] = field(default_factory=set)
    list_items: Set[str] = field(default_factory=set)
    kv_pairs: Dict[str, str] = field(default_factory=dict)

    def all_items(self) -> Set[str]:
        """All items as a flat set for set-difference operations."""
        items = set(self.sentences)
        items.update(self.list_items)
        items.update(f"{k}: {v}" for k, v in self.kv_pairs.items())
        return items

    def total_count(self) -> int:
        return len(self.sentences) + len(self.list_items) + len(self.kv_pairs)


def parse_structural(content: str) -> StructuralItems:
    """Parse content into structural items (sentences, list items, KV pairs).

    Deliberately simple — no NLP, no LLM. The goal is to detect what
    would be lost in a rewrite, not to understand semantics.

    Items are stored in their **original case** for merge-back fidelity.
    Comparison is done case-insensitively via a casefolded lookup set.
    """
    if not content or not content.strip():
        return StructuralItems()

    items = StructuralItems()

    # Extract list items (bullet points) — preserve original case.
    for m in _LIST_ITEM_RE.finditer(content):
        item = m.group(1).strip()
        if item:
            items.list_items.add(item)

    # Extract key-value pairs ("key: value") — preserve original case.
    for m in _KV_RE.finditer(content):
        key = m.group(1).strip()
        value = m.group(2).strip()
        if key and value and len(key) < 60:
            items.kv_pairs[key] = value

    # Extract sentences (split on sentence boundaries).
    # Remove list items and KV pairs first to avoid double-counting.
    text_without_lists = _LIST_ITEM_RE.sub("", content)
    text_without_kv = _KV_RE.sub("", text_without_lists)
    sentences = _SENTENCE_SPLIT_RE.split(text_without_kv)
    for s in sentences:
        s = s.strip()
        # Skip very short fragments (< 10 chars) — likely noise.
        if len(s) >= 10:
            items.sentences.add(s)

    return items


# --- Loss guard ------------------------------------------------------------

@dataclass
class LossReport:
    """Report of what would be lost in a rewrite."""
    lost_sentences: List[str] = field(default_factory=list)
    lost_list_items: List[str] = field(default_factory=list)
    lost_kv_pairs: Dict[str, str] = field(default_factory=dict)
    _total_lost: int = -1

    @property
    def total_lost(self) -> int:
        if self._total_lost >= 0:
            return self._total_lost
        return (
            len(self.lost_sentences)
            + len(self.lost_list_items)
            + len(self.lost_kv_pairs)
        )

    @total_lost.setter
    def total_lost(self, value: int) -> None:
        self._total_lost = value

    def is_clean(self) -> bool:
        """True if no content would be lost (pure enrichment)."""
        return self.total_lost == 0

    def category_counts(self) -> Dict[str, int]:
        """Per-category loss counts for the write report."""
        return {
            "sentences": len(self.lost_sentences),
            "list_items": len(self.lost_list_items),
            "kv_pairs": len(self.lost_kv_pairs),
        }


def compute_loss(existing: str, proposed: str) -> LossReport:
    """Compute what would be lost if *existing* is replaced by *proposed*.

    Returns a LossReport with the lost items per category. Only deletion
    is counted — additions in the proposal are ignored.

    Comparison is case-insensitive (casefolded) but the lost items are
    preserved in their original case for merge-back fidelity.
    """
    existing_items = parse_structural(existing)
    proposed_items = parse_structural(proposed)

    report = LossReport()

    # Build casefolded lookup sets for the proposed content.
    proposed_sentences_cf = {s.casefold() for s in proposed_items.sentences}
    proposed_list_items_cf = {s.casefold() for s in proposed_items.list_items}
    proposed_kv_cf = {k.casefold(): v.casefold() for k, v in proposed_items.kv_pairs.items()}

    # Lost sentences: in existing but not in proposed (case-insensitive).
    report.lost_sentences = sorted(
        s for s in existing_items.sentences
        if s.casefold() not in proposed_sentences_cf
    )

    # Lost list items: in existing but not in proposed (case-insensitive).
    report.lost_list_items = sorted(
        s for s in existing_items.list_items
        if s.casefold() not in proposed_list_items_cf
    )

    # Lost KV pairs: keys in existing but not in proposed, or values differ.
    for key, old_val in existing_items.kv_pairs.items():
        new_val_cf = proposed_kv_cf.get(key.casefold())
        if new_val_cf is None or new_val_cf != old_val.casefold():
            report.lost_kv_pairs[key] = old_val

    report.total_lost = (
        len(report.lost_sentences)
        + len(report.lost_list_items)
        + len(report.lost_kv_pairs)
    )
    return report




def merge_lost_content(proposed: str, loss: LossReport) -> str:
    """Merge lost content back into the proposal.

    The rewrite succeeds but cannot destroy — lost sentences, list items,
    and KV pairs are appended to the proposal so the final content
    preserves everything from the existing version plus the new material.
    """
    if loss.is_clean():
        return proposed

    parts = [proposed]

    # Append lost KV pairs.
    if loss.lost_kv_pairs:
        parts.append("\n\n" + "\n".join(
            f"{k}: {v}" for k, v in sorted(loss.lost_kv_pairs.items())
        ))

    # Append lost list items.
    if loss.lost_list_items:
        parts.append("\n\n" + "\n".join(
            f"- {item}" for item in loss.lost_list_items
        ))

    # Append lost sentences.
    if loss.lost_sentences:
        parts.append("\n\n" + "\n".join(loss.lost_sentences))

    return "".join(parts)


def structural_loss_guard(existing: str, proposed: str) -> Tuple[str, LossReport]:
    """Apply the structural-loss guard to a rewrite.

    Compares *existing* content against *proposed* replacement, counts
    what would be deleted, and merges the lost material back into the
    proposal.

    Returns (repaired_content, loss_report). If no content would be lost
    (pure enrichment), the proposed content is returned unchanged.
    """
    loss = compute_loss(existing, proposed)
    if loss.is_clean():
        return proposed, loss
    repaired = merge_lost_content(proposed, loss)
    return repaired, loss
