"""Trust-model cluster tests (batch-2): #43, #40, #39, #35.

Record-level metadata fields that gate what a memory may do or become:
- #43 provenance taint (per-record, fail-closed)
- #40 grounding ceiling (4-level, monotonic)
- #39 one-way trust ladder (rejection ledger, claim-slot keyed)
- #35 quote verification against source transcript (feeds #40)
"""
from __future__ import annotations

import sys
from pathlib import Path

# Keep this test independently importable when pytest collects it first.
_plugin_dir = Path(__file__).resolve().parent.parent
for _path in (_plugin_dir.parent, _plugin_dir):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import pytest


def _make_store(tmp_path):
    from argos.store import DuckDBMemoryStore

    store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user")
    return store


# ===========================================================================
# #43 — provenance taint
# ===========================================================================

class TestProvenanceTaint:
    def test_normalize_fail_closed(self):
        from argos.store import normalize_provenance, PROVENANCE_EXTERNAL, PROVENANCE_INTERNAL

        assert normalize_provenance("internal") == PROVENANCE_INTERNAL
        assert normalize_provenance("external") == PROVENANCE_EXTERNAL
        # Unknown / corrupt / empty -> external (stricter).
        assert normalize_provenance("") == PROVENANCE_EXTERNAL
        assert normalize_provenance(None) == PROVENANCE_EXTERNAL
        assert normalize_provenance("garbage") == PROVENANCE_EXTERNAL
        assert normalize_provenance("INTERNAL") == PROVENANCE_INTERNAL  # case-insensitive

    def test_external_origin_set_from_payload_flag(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            cand = store.save_candidate(
                category="context_note",
                content="Acme corp fiscal year ends in March",
                source="llm_extraction",
                external=True,
            )
            assert cand["provenance_origin"] == "external"
        finally:
            store.close()

    def test_internal_default_for_user_statement(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            mem = store.remember(
                category="preference",
                content="User prefers dark mode",
                source="explicit",
            )
            assert mem is not None
            assert mem.provenance_origin == "internal"
        finally:
            store.close()

    def test_external_origin_cannot_auto_activate(self, tmp_path):
        """An external-origin candidate can never be auto-activated (#43)."""
        store = _make_store(tmp_path)
        try:
            # Turn the policy ON (default) and propose an external candidate.
            store.external_sources_require_confirmation = True
            cand = store.save_candidate(
                category="context_note",
                content="Vendor X invoice total is 42000",
                source="llm_extraction",
                external=True,
            )
            result = store.review_candidate(
                candidate_id=cand["candidate_id"],
                decision="reviewed_approved",
                reason="auto said yes",
                review_source="auto_review",
            )
            assert result["candidate"]["status"] == "pending_user_confirmation"
            assert result["memory"] is None
        finally:
            store.close()

    def test_external_origin_still_surfaces_in_retrieval(self, tmp_path):
        """Taint gates effect-capable paths, NOT retrieval (#43)."""
        store = _make_store(tmp_path)
        try:
            mem = store.remember(
                category="context_note",
                content="Noted: vendor invoice total is 42000 dollars",
                source="explicit",
                provenance_origin="external",
            )
            assert mem is not None and mem.provenance_origin == "external"
            hits = store.search("vendor invoice", limit=10)
            assert any("invoice" in (h.content or "") for h in hits)
        finally:
            store.close()

    def test_sanitization_does_not_launder_taint(self, tmp_path):
        """Redaction/sanitization leaves the provenance label untouched (#43)."""
        store = _make_store(tmp_path)
        try:
            mem = store.remember(
                category="context_note",
                content="External note: quarterly report draft",
                source="explicit",
                provenance_origin="external",
            )
            assert mem.provenance_origin == "external"
            # sanitize_content strips hidden/format chars but must not touch
            # the provenance_origin column.
            from argos.store import sanitize_content, normalize_provenance

            cleaned, _ = sanitize_content(mem.content)
            assert cleaned == mem.content  # nothing to strip here
            # The stored label is a column, independent of content scrubbing.
            fetched = store.get_memories_by_ids([mem.memory_id])
            assert fetched and fetched[0].provenance_origin == "external"
            assert normalize_provenance("external") == "external"
        finally:
            store.close()


# ===========================================================================
# #40 — grounding ceiling
# ===========================================================================

class TestGroundingCeiling:
    def test_defaults_per_write_path(self):
        from argos.store import default_grounding_for_write, GROUNDING_OBSERVED, \
            GROUNDING_EXTRACTED, GROUNDING_INFERRED, GROUNDING_SPECULATIVE

        assert default_grounding_for_write(source="explicit") == GROUNDING_OBSERVED
        assert default_grounding_for_write(source="user") == GROUNDING_OBSERVED
        assert default_grounding_for_write(source="llm_extraction") == GROUNDING_EXTRACTED
        assert default_grounding_for_write(source="distillation") == GROUNDING_INFERRED
        assert default_grounding_for_write(external=True) == GROUNDING_INFERRED
        assert default_grounding_for_write(source="unknown_path") == GROUNDING_SPECULATIVE
        # Explicit wins.
        assert default_grounding_for_write(
            source="explicit", explicit_grounding="speculative"
        ) == GROUNDING_SPECULATIVE

    def test_distill_candidate_grounds_as_inferred(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            cand = store.save_candidate(
                category="insight",
                content="User tends to underestimate shipping work",
                source="distillation",
            )
            assert cand["grounding"] == "inferred"
        finally:
            store.close()

    def test_speculative_cannot_reach_approved_via_auto_review(self, tmp_path):
        """The core invariant: speculative/inferred can't reach the top class."""
        store = _make_store(tmp_path)
        try:
            cand = store.save_candidate(
                category="context_note",
                content="Maybe the user is considering a move to Berlin",
                source="unknown_path",  # -> speculative
                grounding="speculative",
            )
            # Auto-review tries to approve -> capped at the ceiling.
            result = store.review_candidate(
                candidate_id=cand["candidate_id"],
                decision="reviewed_approved",
                reason="auto",
                review_source="auto_review",
            )
            assert result["candidate"]["status"] == "pending_user_confirmation"
            assert result["memory"] is None
        finally:
            store.close()

    def test_inferred_cannot_be_approved_via_auto_review(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            cand = store.save_candidate(
                category="insight",
                content="User likely values direct feedback",
                source="distillation",  # -> inferred
            )
            with pytest.raises(ValueError, match="approval invariant"):
                store.review_candidate(
                    candidate_id=cand["candidate_id"],
                    decision="approved",
                    reason="auto",
                    review_source="auto_review",
                )
            # reviewed_approved is within the inferred ceiling -> allowed.
            result = store.review_candidate(
                candidate_id=cand["candidate_id"],
                decision="reviewed_approved",
                reason="auto",
                review_source="auto_review",
            )
            assert result["candidate"]["status"] == "reviewed_approved"
            assert result["memory"] is not None
            assert result["memory"]["grounding"] == "inferred"
        finally:
            store.close()

    def test_user_confirmation_lifts_grounding(self, tmp_path):
        """User confirmation raises the grounding (and the ceiling) (#40)."""
        store = _make_store(tmp_path)
        try:
            cand = store.save_candidate(
                category="context_note",
                content="User is weighing a Berlin move",
                source="llm_extraction",
                grounding="speculative",
            )
            # The tool path (user confirmation) may approve even a speculative
            # record: the grounding is lifted to extracted, the ceiling moves.
            result = store.review_candidate(
                candidate_id=cand["candidate_id"],
                decision="approved",
                reason="user confirmed",
                review_source="tool",
            )
            assert result["candidate"]["status"] == "approved"
            assert result["memory"] is not None
            assert result["memory"]["grounding"] == "extracted"
        finally:
            store.close()

    def test_recall_counts_are_not_verification(self, tmp_path):
        """Retrieval/helpful counters never change status or grounding (#40)."""
        store = _make_store(tmp_path)
        try:
            mem = store.remember(
                category="preference",
                content="User prefers plain-English explanations first",
                source="explicit",
            )
            assert mem.grounding == "observed"
            before_status = mem.status
            before_grounding = mem.grounding
            # Helpful feedback increments the counter but mutates nothing else.
            assert store.record_feedback(mem.memory_id, "helpful") is True
            fetched = store.get_memories_by_ids([mem.memory_id])[0]
            assert fetched.helpful_count == 1
            assert fetched.status == before_status
            assert fetched.grounding == before_grounding
        finally:
            store.close()

    def test_supersession_keeps_class(self, tmp_path):
        """Correction lives on the status axis; no demotion-as-punishment."""
        store = _make_store(tmp_path)
        try:
            old = store.remember(
                category="personal_fact",
                content="User's favourite colour is blue",
                source="explicit",
            )
            new = store.remember(
                category="personal_fact",
                content="User's favourite colour is green",
                source="explicit",
            )
            assert new is not None
            ok = store._mark_superseded(old.memory_id, "value change", new.memory_id)
            assert ok
            # The superseded record keeps its grounding/class. Fetch directly
            # (get_memories_by_ids filters to current records only).
            superseded = store._fetch_records(
                "SELECT * FROM memory_records WHERE memory_id = ?",
                [old.memory_id],
            )[0]
            assert superseded.grounding == "observed"
            assert superseded.valid_to is not None
        finally:
            store.close()


# ===========================================================================
# #39 — one-way trust ladder (rejection ledger)
# ===========================================================================

class TestRejectionLedger:
    def test_rejection_key_claim_slot(self):
        from argos.store import rejection_key

        # Same claim slot (subject=user, predicate=personal_fact:age) regardless
        # of the value or phrasing.
        k1 = rejection_key({"category": "personal_fact",
                            "payload": {"attribute": "age"}, "user_scope": "u"})
        k2 = rejection_key({"category": "personal_fact",
                            "payload": {"attribute": "age"}, "user_scope": "u"})
        assert k1 == k2 == ("user", "personal_fact:age", "u")
        # Different attribute -> different slot.
        k3 = rejection_key({"category": "personal_fact",
                            "payload": {"attribute": "location"}, "user_scope": "u"})
        assert k3 != k1
        # Named subject (relationship).
        k4 = rejection_key({"category": "relationship",
                            "payload": {"name": "Alex", "relation": "sister"},
                            "user_scope": "u"})
        assert k4 == ("alex", "relationship:sister", "u")

    def test_reject_records_ledger_and_blocks_reproposal(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            cand = store.save_candidate(
                category="personal_fact",
                content="User's age is 34",
                source="llm_extraction",
                payload={"attribute": "age"},
            )
            store.review_candidate(
                candidate_id=cand["candidate_id"],
                decision="rejected",
                reason="user said this is wrong",
            )
            # Ledger entry written.
            rejections = store.list_rejections()
            assert len(rejections) == 1
            assert rejections[0]["subject"] == "user"
            assert rejections[0]["predicate"] == "personal_fact:age"
            # Re-proposal of the same slot (paraphrased) is blocked.
            again = store.save_candidate(
                category="personal_fact",
                content="User is thirty-four years old",
                source="llm_extraction",
                payload={"attribute": "age"},
            )
            assert again is None
        finally:
            store.close()

    def test_recreation_via_remember_blocked(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            cand = store.save_candidate(
                category="personal_fact",
                content="User's age is 34",
                source="llm_extraction",
                payload={"attribute": "age"},
            )
            store.review_candidate(
                candidate_id=cand["candidate_id"], decision="rejected",
                reason="no",
            )
            # Direct re-creation of the rejected slot is blocked.
            mem = store.remember(
                category="personal_fact",
                content="User's age is 34",
                source="explicit",
                payload={"attribute": "age"},
                dedup=False,
            )
            assert mem is None
        finally:
            store.close()

    def test_rejected_value_absent_from_retrieval(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            cand = store.save_candidate(
                category="personal_fact",
                content="User's age is 34",
                source="llm_extraction",
                payload={"attribute": "age"},
            )
            store.review_candidate(
                candidate_id=cand["candidate_id"], decision="rejected", reason="no"
            )
            # Every re-assertion path is blocked, so the value never enters the
            # store and cannot surface in retrieval.
            for content, payload in [
                ("User's age is 34", {"attribute": "age"}),
                ("User is thirty-four years old", {"attribute": "age"}),
            ]:
                store.save_candidate(
                    category="personal_fact", content=content,
                    source="llm_extraction", payload=payload,
                )
                store.remember(
                    category="personal_fact", content=content, source="explicit",
                    payload=payload, dedup=False,
                )
            hits = store.search("user age", limit=10)
            assert not any("34" in (h.content or "") or "thirty-four" in (h.content or "")
                           for h in hits)
        finally:
            store.close()

    def test_purge_rejection_allows_back(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            cand = store.save_candidate(
                category="personal_fact",
                content="User's age is 34",
                source="llm_extraction",
                payload={"attribute": "age"},
            )
            store.review_candidate(candidate_id=cand["candidate_id"],
                                    decision="rejected", reason="no")
            assert store.purge_rejection("personal_fact", {"attribute": "age"}) is True
            # After the escape hatch, the slot may be proposed again.
            again = store.save_candidate(
                category="personal_fact",
                content="User's age is 34",
                source="llm_extraction",
                payload={"attribute": "age"},
            )
            assert again is not None
        finally:
            store.close()

    def test_agent_cannot_self_promote(self, tmp_path):
        """No tool surface lets the agent self-promote to approved (#39)."""
        store = _make_store(tmp_path)
        try:
            cand = store.save_candidate(
                category="preference",
                content="User prefers concise answers",
                source="llm_extraction",
            )
            with pytest.raises(ValueError, match="approval invariant"):
                store.review_candidate(
                    candidate_id=cand["candidate_id"],
                    decision="approved",
                    reason="agent self-approve",
                    review_source="auto_review",
                )
        finally:
            store.close()

    def test_different_slot_not_blocked(self, tmp_path):
        """Rejecting one slot does not block a different claim slot."""
        store = _make_store(tmp_path)
        try:
            cand = store.save_candidate(
                category="personal_fact",
                content="User's age is 34",
                source="llm_extraction",
                payload={"attribute": "age"},
            )
            store.review_candidate(candidate_id=cand["candidate_id"],
                                    decision="rejected", reason="no")
            # A different attribute is a different slot -> allowed.
            other = store.save_candidate(
                category="personal_fact",
                content="User's location is Berlin",
                source="llm_extraction",
                payload={"attribute": "location"},
            )
            assert other is not None
        finally:
            store.close()


# ===========================================================================
# #35 — quote verification against source transcript
# ===========================================================================

class TestQuoteVerification:
    def test_verify_found_missed_near_miss(self):
        from argos.extractor import verify_quote_against_source

        source = "I just got a new job at Stripe, I'll be starting next Monday."
        # Found (case/whitespace-insensitive).
        assert verify_quote_against_source("I just got a new job at Stripe", source) is True
        # Missed (not in the source).
        assert verify_quote_against_source("I was hired at Google last year", source) is False
        # Near-miss: extra punctuation/whitespace still matches after normalization.
        assert verify_quote_against_source("I'll be starting   next Monday!", source) is True
        # Empty quote -> not found.
        assert verify_quote_against_source("", source) is False

    def test_short_fragment_treated_as_found(self):
        from argos.extractor import verify_quote_against_source

        # Below the min length threshold -> not a meaningful claim to falsify.
        assert verify_quote_against_source("hi", "a long source transcript") is True

    def test_apply_downgrades_on_miss(self):
        from argos.extractor import apply_quote_verification, _reset_quote_verification_stats, \
            get_quote_verification_stats

        _reset_quote_verification_stats()
        fact = {
            "category": "personal_fact",
            "content": "User's salary is 120k",
            "payload": {"source": "llm_extraction"},
            "verbatim_quote": "I earn one hundred and twenty thousand",
        }
        source = "My new role starts next month, very excited about the team."
        apply_quote_verification(fact, source)
        assert fact["payload"]["quote_verified"] is False
        assert fact["payload"]["grounding"] == "inferred"
        assert get_quote_verification_stats()["quote_verification_misses"] == 1

    def test_apply_passes_on_found_quote(self):
        from argos.extractor import apply_quote_verification, _reset_quote_verification_stats, \
            get_quote_verification_stats

        _reset_quote_verification_stats()
        fact = {
            "category": "event",
            "content": "User got a new job at Stripe",
            "payload": {"source": "llm_extraction"},
            "verbatim_quote": "I just got a new job at Stripe",
        }
        source = "I just got a new job at Stripe, starting Monday."
        apply_quote_verification(fact, source)
        assert fact["payload"]["quote_verified"] is True
        assert "grounding" not in fact["payload"]  # no downgrade
        assert get_quote_verification_stats()["quote_verification_misses"] == 0

    def test_apply_untouched_without_verbatim_quote(self):
        from argos.extractor import apply_quote_verification, _reset_quote_verification_stats

        _reset_quote_verification_stats()
        fact = {
            "category": "preference",
            "content": "User prefers short answers",
            "payload": {"source": "llm_extraction"},
        }
        apply_quote_verification(fact, "any source text")
        assert "quote_verified" not in fact["payload"]
        assert "grounding" not in fact["payload"]

    def test_downgrade_flows_to_candidate_grounding(self, tmp_path):
        """The #35 downgrade lands on #40's grounding via the payload override."""
        store = _make_store(tmp_path)
        try:
            cand = store.save_candidate(
                category="personal_fact",
                content="User's salary is 120k",
                source="llm_extraction",
                payload={"source": "llm_extraction", "grounding": "inferred",
                         "quote_verified": False},
            )
            assert cand["grounding"] == "inferred"
        finally:
            store.close()
