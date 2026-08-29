"""Tests for #42: structural-loss guard on distill/consolidation rewrites.

Covers:
- Structural-loss guard: deletion triggers repair, enrichment passes unchanged
- Category-level loss counts surfaced in the write report
- Append-only class (outcome records) never merged by dedup
- Edge cases: empty proposal, reordered items, case-variant duplicates
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))

from structural_loss import (
    parse_structural,
    compute_loss,
    merge_lost_content,
    structural_loss_guard,
    is_append_only,
    LossReport,
)
from store import DuckDBMemoryStore


# ---------------------------------------------------------------------------
# Structural parsing
# ---------------------------------------------------------------------------

class TestParseStructural:
    """parse_structural should extract sentences, list items, and KV pairs."""

    def test_simple_sentence(self):
        items = parse_structural("User lives in Springfield. Works at Acme Corp.")
        assert len(items.sentences) >= 2
        assert any("springfield" in s.casefold() for s in items.sentences)

    def test_list_items(self):
        items = parse_structural("- Item one\n- Item two\n- Item three")
        assert len(items.list_items) == 3

    def test_kv_pairs(self):
        items = parse_structural("Name: John\nAge: 35\nCity: Springfield")
        assert len(items.kv_pairs) == 3
        assert items.kv_pairs["Name"] == "John"
        assert items.kv_pairs["Age"] == "35"

    def test_empty_content(self):
        items = parse_structural("")
        assert items.total_count() == 0

    def test_none_content(self):
        items = parse_structural(None)
        assert items.total_count() == 0

    def test_mixed_content(self):
        content = "User profile:\n- Likes hiking\n- Enjoys cooking\nName: John\nNotes: Active user."
        items = parse_structural(content)
        assert len(items.list_items) >= 2
        # KV parsing may pick up some pairs from the mixed content.
        assert items.total_count() >= 2


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

class TestComputeLoss:
    """compute_loss should count deletions, not additions."""

    def test_pure_enrichment_no_loss(self):
        """Adding content without removing any → no loss."""
        existing = "User lives in Springfield."
        proposed = "User lives in Springfield. Works at Acme Corp."
        loss = compute_loss(existing, proposed)
        assert loss.is_clean()
        assert loss.total_lost == 0

    def test_deletion_triggers_loss(self):
        """Removing content → loss detected."""
        existing = "User lives in Springfield. Works at Acme Corp."
        proposed = "User lives in Springfield."
        loss = compute_loss(existing, proposed)
        assert not loss.is_clean()
        assert loss.total_lost > 0
        assert any("acme" in s.casefold() for s in loss.lost_sentences)

    def test_reorder_no_loss(self):
        """Reordering items → no loss (same items, different order)."""
        existing = "- Item one\n- Item two\n- Item three"
        proposed = "- Item three\n- Item one\n- Item two"
        loss = compute_loss(existing, proposed)
        assert loss.is_clean()

    def test_case_variant_no_loss(self):
        """Case-variant duplicates → no loss (casefolded comparison)."""
        existing = "User lives in Springfield."
        proposed = "user lives in springfield."
        loss = compute_loss(existing, proposed)
        assert loss.is_clean()

    def test_empty_proposal(self):
        """Empty proposal → all existing content is lost."""
        existing = "User lives in Springfield. Works at Acme Corp."
        proposed = ""
        loss = compute_loss(existing, proposed)
        assert not loss.is_clean()
        assert loss.total_lost > 0

    def test_kv_pair_loss(self):
        """Removing a KV pair → loss detected."""
        existing = "Name: John\nAge: 35\nCity: Springfield"
        proposed = "Name: John\nAge: 35"
        loss = compute_loss(existing, proposed)
        assert not loss.is_clean()
        assert "City" in loss.lost_kv_pairs

    def test_kv_pair_value_change(self):
        """Changing a KV pair value → old value is lost."""
        existing = "Name: John\nAge: 35"
        proposed = "Name: John\nAge: 36"
        loss = compute_loss(existing, proposed)
        assert not loss.is_clean()
        assert loss.lost_kv_pairs.get("Age") == "35"

    def test_list_item_loss(self):
        """Removing a list item → loss detected."""
        existing = "- Item one\n- Item two\n- Item three"
        proposed = "- Item one\n- Item two"
        loss = compute_loss(existing, proposed)
        assert not loss.is_clean()
        assert any("item three" in item.casefold() for item in loss.lost_list_items)

    def test_category_counts(self):
        """Loss report should have per-category counts."""
        existing = "User lives in Springfield.\n- Likes hiking\nName: John"
        proposed = "User lives in Springfield."
        loss = compute_loss(existing, proposed)
        counts = loss.category_counts()
        assert "sentences" in counts
        assert "list_items" in counts
        assert "kv_pairs" in counts
        assert counts["list_items"] >= 1
        assert counts["kv_pairs"] >= 1


# ---------------------------------------------------------------------------
# Merge lost content
# ---------------------------------------------------------------------------

class TestMergeLostContent:
    """merge_lost_content should append lost items back to the proposal."""

    def test_clean_proposal_unchanged(self):
        """If no loss, proposed content is returned unchanged."""
        proposed = "User lives in Springfield. Works at Acme."
        loss = LossReport()
        result = merge_lost_content(proposed, loss)
        assert result == proposed

    def test_lost_sentences_appended(self):
        """Lost sentences are appended to the proposal."""
        proposed = "User lives in Springfield."
        loss = LossReport(lost_sentences=["Works at Acme Corp."])
        result = merge_lost_content(proposed, loss)
        assert "Works at Acme Corp." in result
        assert "User lives in Springfield." in result

    def test_lost_list_items_appended(self):
        """Lost list items are appended as bullet points."""
        proposed = "- Item one"
        loss = LossReport(lost_list_items=["item two"])
        result = merge_lost_content(proposed, loss)
        assert "- item two" in result
        assert "- Item one" in result

    def test_lost_kv_pairs_appended(self):
        """Lost KV pairs are appended."""
        proposed = "Name: John"
        loss = LossReport(lost_kv_pairs={"age": "35"})
        result = merge_lost_content(proposed, loss)
        assert "age: 35" in result
        assert "Name: John" in result


# ---------------------------------------------------------------------------
# Full guard: structural_loss_guard
# ---------------------------------------------------------------------------

class TestStructuralLossGuard:
    """structural_loss_guard should repair deletions and pass enrichments."""

    def test_enrichment_passes_unchanged(self):
        """A proposal that only adds content passes unchanged."""
        existing = "User lives in Springfield."
        proposed = "User lives in Springfield. Works at Acme Corp."
        repaired, loss = structural_loss_guard(existing, proposed)
        assert repaired == proposed
        assert loss.is_clean()

    def test_deletion_repaired(self):
        """A proposal that deletes content has it merged back."""
        existing = "User lives in Springfield. Works at Acme Corp."
        proposed = "User lives in Springfield."
        repaired, loss = structural_loss_guard(existing, proposed)
        assert not loss.is_clean()
        # The lost content should be in the repaired version.
        assert "acme" in repaired.casefold()

    def test_reorder_passes_unchanged(self):
        """Reordering items passes unchanged (no loss)."""
        existing = "- Item one\n- Item two\n- Item three"
        proposed = "- Item three\n- Item one\n- Item two"
        repaired, loss = structural_loss_guard(existing, proposed)
        assert repaired == proposed
        assert loss.is_clean()

    def test_case_variant_passes_unchanged(self):
        """Case-variant duplicates pass unchanged."""
        existing = "User lives in Springfield."
        proposed = "USER LIVES IN SPRINGFIELD."
        repaired, loss = structural_loss_guard(existing, proposed)
        assert loss.is_clean()

    def test_empty_proposal_repaired(self):
        """An empty proposal gets all existing content merged back."""
        existing = "User lives in Springfield. Works at Acme."
        proposed = ""
        repaired, loss = structural_loss_guard(existing, proposed)
        assert not loss.is_clean()
        assert "springfield" in repaired.casefold()


# ---------------------------------------------------------------------------
# Append-only exemption
# ---------------------------------------------------------------------------

class TestAppendOnly:
    """Outcome/decision-shaped records should be exempt from merging and dedup."""

    def test_outcome_category_is_append_only(self):
        assert is_append_only("outcome")

    def test_procedure_outcome_category_is_append_only(self):
        assert is_append_only("procedure_outcome")

    def test_decision_category_is_append_only(self):
        assert is_append_only("decision")

    def test_personal_fact_not_append_only(self):
        assert not is_append_only("personal_fact")

    def test_payload_kind_outcome_is_append_only(self):
        assert is_append_only("context_note", {"kind": "outcome"})

    def test_payload_kind_decision_is_append_only(self):
        assert is_append_only("context_note", {"kind": "decision"})

    def test_payload_kind_tripwatch_is_append_only(self):
        assert is_append_only("context_note", {"kind": "tripwatch"})

    def test_payload_kind_insight_not_append_only(self):
        assert not is_append_only("insight", {"kind": "insight"})

    def test_no_payload_not_append_only(self):
        assert not is_append_only("personal_fact", None)


# ---------------------------------------------------------------------------
# Integration: update_memory with structural-loss guard
# ---------------------------------------------------------------------------

class TestUpdateMemoryWithGuard:
    """update_memory should apply the structural-loss guard."""

    def test_update_with_enrichment_no_repair(self, tmp_path):
        """An update that only adds content should not trigger repair."""
        store = DuckDBMemoryStore(tmp_path / "test_guard.duckdb")
        try:
            mem = store.remember(
                category="personal_fact",
                content="User lives in Springfield.",
            )
            updated = store.update_memory(
                mem.memory_id,
                content="User lives in Springfield. Works at Acme Corp.",
            )
            assert updated is not None
            assert "Works at Acme Corp." in updated.content
            # No repair needed — enrichment only.
            payload = updated.payload or {}
            assert "structural_loss_repair" not in payload
        finally:
            store.close()

    def test_update_with_deletion_repaired(self, tmp_path):
        """An update that deletes content should have it merged back."""
        store = DuckDBMemoryStore(tmp_path / "test_guard_repair.duckdb")
        try:
            mem = store.remember(
                category="personal_fact",
                content="User lives in Springfield. Works at Acme Corp.",
            )
            updated = store.update_memory(
                mem.memory_id,
                content="User lives in Springfield.",
                structural_guard=True,
            )
            assert updated is not None
            # The lost content should be merged back.
            assert "acme" in updated.content.casefold()
            # The repair should be recorded in the payload.
            payload = updated.payload or {}
            assert "structural_loss_repair" in payload
            repair = payload["structural_loss_repair"]
            assert repair["lost_total"] > 0
        finally:
            store.close()

    def test_append_only_exempt_from_guard(self, tmp_path):
        """Outcome records should not be repaired by the guard."""
        store = DuckDBMemoryStore(tmp_path / "test_guard_outcome.duckdb")
        try:
            mem = store.remember(
                category="context_note",
                content="Surgery outcome: successful recovery.",
                payload={"kind": "outcome"},
            )
            # An update that "deletes" content from an outcome record
            # should NOT be repaired — outcomes are append-only.
            updated = store.update_memory(
                mem.memory_id,
                content="Surgery outcome: patient discharged.",
                structural_guard=True,
            )
            assert updated is not None
            # The old content should NOT be merged back (append-only).
            assert "successful recovery" not in updated.content
            # No repair recorded.
            payload = updated.payload or {}
            assert "structural_loss_repair" not in payload
        finally:
            store.close()


# ---------------------------------------------------------------------------
# Integration: consolidate with append-only exemption
# ---------------------------------------------------------------------------

class TestConsolidateAppendOnlyExemption:
    """consolidate should never quarantine append-only records."""

    def test_outcome_not_quarantined_by_dedup(self, tmp_path):
        """An outcome record should not be quarantined even if it's a
        near-duplicate of another outcome."""
        store = DuckDBMemoryStore(tmp_path / "test_consolidate_outcome.duckdb")
        try:
            # Two outcome records with very similar content.
            store.remember(
                category="context_note",
                content="Surgery outcome: successful recovery with no complications.",
                payload={"kind": "outcome"},
            )
            store.remember(
                category="context_note",
                content="Surgery outcome: successful recovery with minor complications.",
                payload={"kind": "outcome"},
            )
            # Run consolidate (dry_run to see what it would do).
            report = store.consolidate(dry_run=True, min_age_days=0)
            # No outcome records should be in the candidates.
            for cand in report.get("candidates", []):
                payload = cand.get("payload") or {}
                assert payload.get("kind") != "outcome", (
                    "Outcome records should be exempt from dedup quarantine"
                )
        finally:
            store.close()
