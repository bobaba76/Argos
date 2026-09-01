"""Regression tests for issue #115: doc-tier activation dedup ignores
namespace/client_scope/doc_class/source_doc_id — distinct facts collapse.

The activation path is: ``review_candidate(approved) → remember() →
_content_exists() → _find_current_similar()``.  Before this fix, all three
dedup layers (exact / substring / semantic) filtered only by
``(category, user_scope)``.  Two facts from different documents (different
``namespace``/``client_scope``/``source_doc_id``) but with identical or
near-identical content collapsed — the very separation Spec-05/06 (#67/#69)
exists to enforce was invisible to dedup.  Worse, a same-subject value
change on the same logical line was silently ``deduplicated`` by
``remember()`` returning ``None``, so the ``elif supersedes_memory_id``
branch in ``review_candidate`` never fired — supersession was defeated by
the dedup gate ordering.

These trials mirror the probe evidence in the issue (deterministic, local,
0 LLM calls).  They use the ``DeterministicTestEmbedder`` so the semantic
layer (layer 3) fires for near-identical content, exactly as in production
with a real embedder.

Trial mapping:
- D: identical content, different namespace + client_scope → both survive.
- B: same doc, same subject, different value → routes to supersession,
     old value is superseded (not silently deduped).
- C: different subjects, same doc → both survive (control).
- Conv regression: conversation-tier global dedup still collapses
  paraphrased duplicates (the feature must not regress).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))

from _test_embedder import DeterministicTestEmbedder


def _make_store(tmp_path):
    from store import DuckDBMemoryStore

    return DuckDBMemoryStore(
        tmp_path / "test.duckdb",
        user_id="test_user",
        embedder=DeterministicTestEmbedder(),
    )


# ---------------------------------------------------------------------------
# Trial D: identical content, different namespace + client_scope → both survive
# ---------------------------------------------------------------------------

class TestTrialDCrossDocIdentityNoCollapse:
    """Spec-05/06 invariant: identical content at different doc identity
    must NOT collapse via dedup.  Before #115, layer 1 (exact) collapsed
    these because the WHERE clause ignored namespace/client_scope."""

    def test_exact_match_different_namespace_survives(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            r1 = store.remember(
                category="personal_fact",
                content="Brand A: support plan — R1,200/quarter",
                namespace="document",
                client_scope="b2b",
                source_doc_id="doc-b2b-1",
                dedup=True,
            )
            r2 = store.remember(
                category="personal_fact",
                content="Brand A: support plan — R1,200/quarter",
                namespace="document",
                client_scope="d2c",
                source_doc_id="doc-d2c-1",
                dedup=True,
            )
            assert r1 is not None, "first doc-tier fact must store"
            assert r2 is not None, (
                "second doc-tier fact with different client_scope must NOT "
                "be deduped (Spec-05/06 separation invisible to dedup #115)"
            )
            assert r1.memory_id != r2.memory_id
        finally:
            store.close()

    def test_exact_match_different_source_doc_id_survives(self, tmp_path):
        """Same namespace/client_scope but different source_doc_id: two
        documents can carry the same line (e.g. a repeated boilerplate
        clause) and both must persist."""
        store = _make_store(tmp_path)
        try:
            r1 = store.remember(
                category="personal_fact",
                content="Standard SLA: 99.9% uptime guaranteed",
                namespace="document",
                client_scope="acme",
                source_doc_id="contract-v1",
                dedup=True,
            )
            r2 = store.remember(
                category="personal_fact",
                content="Standard SLA: 99.9% uptime guaranteed",
                namespace="document",
                client_scope="acme",
                source_doc_id="contract-v2",
                dedup=True,
            )
            assert r1 is not None and r2 is not None, (
                "identical content from different source_doc_id must both "
                "survive (distinct documents, #115)"
            )
            assert r1.memory_id != r2.memory_id
        finally:
            store.close()

    def test_semantic_near_dup_different_client_scope_survives(self, tmp_path):
        """Layer 3 (semantic) must also respect doc-identity scope.
        Near-identical content at different client_scope must not collapse
        on embedding proximity alone."""
        store = _make_store(tmp_path)
        try:
            r1 = store.remember(
                category="personal_fact",
                content="Brand B: add-on service — R950 once off",
                namespace="document",
                client_scope="b2b",
                source_doc_id="doc-b2b-2",
                dedup=True,
            )
            r2 = store.remember(
                category="personal_fact",
                content="Brand B: add-on service — R950 once-off",
                namespace="document",
                client_scope="d2c",
                source_doc_id="doc-d2c-2",
                dedup=True,
            )
            assert r1 is not None and r2 is not None, (
                "near-identical content at different client_scope must not "
                "collapse via semantic dedup (#115 trial A analogue, "
                "cross-doc-identity)"
            )
        finally:
            store.close()


# ---------------------------------------------------------------------------
# Trial B: same doc, same subject, different value → supersession, not dedup
# ---------------------------------------------------------------------------

class TestTrialBSameDocValueChangeRoutesToSupersession:
    """A price change on the same logical line within the same document
    must route to the supersession path, not be silently ``deduplicated``.

    Before #115, ``remember()`` returned ``None`` (deduped), so
    ``review_candidate``'s ``elif supersedes_memory_id`` branch never ran —
    the old value stayed current forever.  The fix: when
    ``supersedes_memory_id`` is set, ``review_candidate`` passes
    ``dedup=False`` to ``remember()`` so the chain actually grows.
    """

    def test_supersession_branch_fires_when_supersedes_set(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            old = store.remember(
                category="personal_fact",
                content="Brand A annual plan price is $499 per year",
                namespace="document",
                client_scope="acme",
                source_doc_id="price-list-2025",
                dedup=False,
            )
            assert old is not None
            # Same source_doc_id: a price correction on the same document.
            # Doc-identity scoping alone would still dedup this (same doc
            # identity, semantic match) — the dedup=False fix in
            # review_candidate is what lets the supersession branch fire.
            cand = store.save_candidate(
                category="personal_fact",
                content="Brand A annual plan switched to $999 per year",
                source="llm_extraction",
                namespace="document",
                client_scope="acme",
                source_doc_id="price-list-2025",
            )
            assert cand is not None
            payload = cand.get("payload", {})
            if isinstance(payload, str):
                payload = json.loads(payload)
            vs = payload.get("value_supersession")
            assert vs is not None, (
                "save_candidate must detect the value conflict and stamp "
                "value_supersession into the payload"
            )
            supersedes_id = vs["supersedes_memory_id"]
            result = store.review_candidate(
                candidate_id=cand["candidate_id"],
                decision="approved",
                reason="user confirmed new price",
                supersedes_memory_id=supersedes_id,
                review_source="tool",
            )
            assert result is not None
            assert result["candidate"]["status"] == "approved", (
                "candidate must be approved, not deduplicated"
            )
            assert result["memory"] is not None, (
                "remember() must NOT return None when supersedes_memory_id "
                "is set — the supersession branch depends on it (#115)"
            )
            old_row = store._fetch_records(
                "SELECT * FROM memory_records WHERE memory_id = ?",
                [old.memory_id],
            )
            assert old_row and old_row[0].valid_to is not None, (
                "old value must be superseded (valid_to set), not left "
                "current forever (#115 trial B)"
            )
            assert old_row[0].superseded_by == result["memory"]["memory_id"]
        finally:
            store.close()


# ---------------------------------------------------------------------------
# Trial C: different subjects, same doc → both survive (control)
# ---------------------------------------------------------------------------

class TestTrialCDifferentSubjectsSameDocSurvive:
    """Control: genuinely different facts from the same document must
    coexist.  This already worked before #115 and must not regress."""

    def test_different_subjects_same_doc_both_stored(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            r1 = store.remember(
                category="personal_fact",
                content="Brand A: annual plan price — R4,999",
                namespace="document",
                client_scope="acme",
                source_doc_id="price-list-2025",
                dedup=True,
            )
            r2 = store.remember(
                category="personal_fact",
                content="Brand A: setup fee — R750",
                namespace="document",
                client_scope="acme",
                source_doc_id="price-list-2025",
                dedup=True,
            )
            assert r1 is not None and r2 is not None, (
                "different subjects from the same doc must both survive"
            )
            assert r1.memory_id != r2.memory_id
        finally:
            store.close()


# ---------------------------------------------------------------------------
# Conversation-tier regression: global dedup must still collapse paraphrases
# ---------------------------------------------------------------------------

class TestConversationTierGlobalDedupPreserved:
    """The fix scopes dedup by doc identity ONLY for the doc tier
    (namespace != 'conversation').  Conversational paraphrase dedup — a
    feature, not a bug — must still collapse near-identical content when
    no doc identity is present."""

    def test_conversation_near_dup_still_deduped(self, tmp_path):
        """Near-duplicate content in the conversation tier (no doc identity)
        must still collapse.  Uses content with high char-trigram overlap so
        the deterministic test embedder's semantic layer fires, plus a
        substring containment that triggers layer 2 — both layers must
        remain global for namespace=conversation."""
        store = _make_store(tmp_path)
        try:
            r1 = store.remember(
                category="personal_fact",
                content="User works at Acme Corp headquarters office today",
                dedup=True,
            )
            # Substring of r1, overlap 44/51 ≈ 0.86 ≥ 0.8 → layer 2 fires
            # globally (no doc-identity scope narrowing).
            r2 = store.remember(
                category="personal_fact",
                content="User works at Acme Corp headquarters office",
                dedup=True,
            )
            assert r1 is not None
            assert r2 is None, (
                "conversation-tier near-duplicate must still be deduped "
                "(global dedup preserved for namespace=conversation, #115)"
            )
        finally:
            store.close()

    def test_conversation_exact_still_deduped(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            r1 = store.remember(
                category="personal_fact",
                content="User works at Acme Corp",
                dedup=True,
            )
            r2 = store.remember(
                category="personal_fact",
                content="User works at Acme Corp",
                dedup=True,
            )
            assert r1 is not None
            assert r2 is None, "exact duplicate in conversation tier must dedup"
        finally:
            store.close()


# ---------------------------------------------------------------------------
# review_candidate must forward source_doc_id (companion fix)
# ---------------------------------------------------------------------------

class TestReviewCandidateForwardsSourceDocId:
    """Before #115, ``review_candidate`` built ``remember_kwargs`` with
    namespace/client_scope/doc_class but NOT source_doc_id, so approved
    doc-tier memories landed with ``source_doc_id = NULL`` — making the
    source_doc_id dedup scope clause useless for activation-path records.
    """

    def test_approved_doc_memory_keeps_source_doc_id(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            cand = store.save_candidate(
                category="personal_fact",
                content="Acme VAT registration number is ZA123456789",
                source="llm_extraction",
                namespace="document",
                client_scope="acme",
                source_doc_id="vat-certificate",
            )
            assert cand is not None
            result = store.review_candidate(
                candidate_id=cand["candidate_id"],
                decision="approved",
                reason="user confirmed",
                review_source="tool",
            )
            assert result and result["memory"] is not None
            mem_id = result["memory"]["memory_id"]
            row = store._fetch_records(
                "SELECT source_doc_id, namespace, client_scope "
                "FROM memory_records WHERE memory_id = ?",
                [mem_id],
            )
            assert row, "approved memory must be persisted"
            assert row[0].source_doc_id == "vat-certificate", (
                "source_doc_id must be forwarded from candidate to memory "
                "(#115 companion fix)"
            )
            assert row[0].namespace == "document"
            assert row[0].client_scope == "acme"
        finally:
            store.close()
