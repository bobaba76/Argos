"""Spec-13 (#393 slice S1): write-policy core tests.

approval_mode is the SINGLE server-derived knob (one-tenancy: per-tenant
overridable):

  - "auto" (default): saves materialize immediately - zero queue. Medium/
    high-risk signals are stamped trust_class="unreviewed" (server-set,
    bounded rank penalty of <= 3 positions at retrieval, never removed
    from the injection window, never auto-promoted).
  - "human": v1 behavior - candidates wait in the review queue for an
    explicit human decision.
  - invalid values fail closed to "human".

Covers: deterministic classification, store-level materialization,
outcome statuses (auto_saved / deduplicated / quarantined), the RPC
sanitize boundary, tenant-policy resolution, retrieval-penalty bounds,
version carry-forward (never auto-promote via edit), and the review
path remaining untouched.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

_plugin_dir = Path(__file__).resolve().parent.parent
for _path in (_plugin_dir.parent, _plugin_dir):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import pytest


def _make_store(tmp_path):
    from argos.store import DuckDBMemoryStore

    return DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user")


CLEAN = "Michael prefers the window seat on long flights to Cape Town"


# ===========================================================================
# Classification (pure, deterministic - zero LLM)
# ===========================================================================

class TestClassifyTrustTier:
    def _classify(self, **kw):
        from trust_tier import classify_trust_tier

        return classify_trust_tier(**kw)

    def test_clean_internal_statement_is_clean(self):
        tier, reasons, sens = self._classify(
            content=CLEAN, category="preference", source="llm_extraction",
            evidence_text="user said it",
        )
        assert tier == "clean"
        assert reasons == []
        assert sens is False

    def test_external_origin_is_unreviewed(self):
        tier, reasons, _ = self._classify(content=CLEAN, external=True)
        assert tier == "unreviewed"
        assert "external_origin" in reasons

    def test_missing_evidence_on_extraction_is_unreviewed(self):
        tier, reasons, _ = self._classify(
            content=CLEAN, source="llm_extraction", evidence_text="",
        )
        assert tier == "unreviewed"
        assert "no_evidence" in reasons

    def test_speculative_grounding_is_unreviewed(self):
        tier, reasons, _ = self._classify(
            content=CLEAN, source="regex_extraction", grounding="speculative",
        )
        assert tier == "unreviewed"
        assert "speculative_grounding" in reasons

    def test_low_confidence_is_unreviewed(self):
        tier, reasons, _ = self._classify(
            content=CLEAN, source="regex_extraction", confidence=0.2,
        )
        assert tier == "unreviewed"
        assert "low_confidence" in reasons

    def test_sensitive_identifiers_flagged(self):
        tier, reasons, sens = self._classify(
            content="Reach Michael at michael@example.com tomorrow",
            source="regex_extraction", evidence_text="said in chat",
        )
        assert sens is True
        assert tier == "unreviewed"
        assert any(r.startswith("sensitive_content:") for r in reasons)

    def test_hard_quality_flags_block_extraction_source(self):
        tier, reasons, _ = self._classify(
            content="ok", category="context_note", source="llm_extraction",
        )
        assert tier == "blocked"
        assert reasons and reasons[0].startswith("quality:")

    def test_quality_flags_do_not_block_other_sources(self):
        # The hard-flag set is tuned for extracted facts; a short API
        # string is classified (unreviewed via external) not blocked.
        tier, _, _ = self._classify(content="ok", source="api")
        assert tier == "unreviewed"


# ===========================================================================
# Store materialization (approval_mode="auto")
# ===========================================================================

class TestAutoModeMaterialization:
    def test_clean_save_materializes_immediately(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            cand = store.save_candidate(
                category="preference",
                content=CLEAN,
                source="llm_extraction",
                evidence_text="User said it on the call",
                approval_mode="auto",
            )
            # No queue: the candidate is stamped auto_saved and the memory
            # is active right now.
            assert cand is not None
            assert cand["status"] == "auto_saved"
            assert cand["review_model"] == "approval_mode_auto"
            mem_id = cand["payload"]["materialized_memory_id"]
            mems = store.get_memories_by_ids([mem_id])
            assert len(mems) == 1
            mem = mems[0]
            assert mem.trust_class is None  # clean tier has no marker
            assert mem.payload.get("trust", {}).get("class") == "clean"
            # Nothing pending anywhere.
            assert store.list_candidates() == []
        finally:
            store.close()

    def test_unreviewed_save_stamps_marker(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            cand = store.save_candidate(
                category="context_note",
                content=CLEAN,
                source="llm_extraction",  # no evidence -> unreviewed
                approval_mode="auto",
            )
            assert cand["status"] == "auto_saved"
            mem = store.get_memories_by_ids(
                [cand["payload"]["materialized_memory_id"]]
            )[0]
            assert mem.trust_class == "unreviewed"
            assert "no_evidence" in mem.payload["trust"]["reasons"]
        finally:
            store.close()

    def test_blocked_candidate_is_quarantined_not_materialized(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            cand = store.save_candidate(
                category="context_note",
                content="ok",
                source="llm_extraction",
                approval_mode="auto",
            )
            assert cand["status"] == "quarantined"
            assert "deterministic quality gate" in (
                cand.get("quarantine_reason") or ""
            )
            assert "materialized_memory_id" not in cand["payload"]
            assert store.list_candidates() == []
        finally:
            store.close()

    def test_human_mode_keeps_v1_queue(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            cand = store.save_candidate(
                category="preference",
                content=CLEAN,
                source="regex_extraction",
                approval_mode="human",
            )
            assert cand["status"] == "pending"
            assert "materialized_memory_id" not in cand["payload"]
            assert len(store.list_candidates()) == 1
        finally:
            store.close()

    def test_none_mode_fails_closed_to_human(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            cand = store.save_candidate(
                category="preference",
                content=CLEAN,
                source="regex_extraction",
                approval_mode=None,
            )
            assert cand["status"] == "pending"
        finally:
            store.close()

    def test_invalid_mode_fails_closed_to_human(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            cand = store.save_candidate(
                category="preference",
                content=CLEAN,
                source="regex_extraction",
                approval_mode="banana",
            )
            assert cand["status"] == "pending"
        finally:
            store.close()

    def test_auto_dedup_against_existing_memory(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            first = store.save_candidate(
                category="preference",
                content=CLEAN,
                source="regex_extraction",
                evidence_text="said it",
                approval_mode="auto",
            )
            assert first["status"] == "auto_saved"
            second = store.save_candidate(
                category="preference",
                content=CLEAN,
                source="regex_extraction",
                evidence_text="said it again",
                approval_mode="auto",
            )
            # remember() deduped the twin; the candidate records that
            # outcome honestly instead of pretending it auto-saved.
            assert second["status"] == "deduplicated"
            assert "materialized_memory_id" not in second["payload"]
        finally:
            store.close()

    def test_flag_off_leaves_review_path_untouched(self, tmp_path):
        """The old reviewer flow still works on a pending candidate."""
        store = _make_store(tmp_path)
        try:
            cand = store.save_candidate(
                category="preference",
                content=CLEAN,
                source="regex_extraction",
                approval_mode="human",
            )
            upid = cand["candidate_id"]
            result = store.review_candidate(
                upid,
                decision="approved",
                review_source="tool",
            )
            assert result is not None
            assert result["candidate"]["status"] == "approved"
            # The reviewed memory carries no trust marker (clean class):
            # the human path is unchanged by Spec-13.
            mems = store.search(CLEAN, limit=5)
            assert any(m.trust_class is None for m in mems)
        finally:
            store.close()


# ===========================================================================
# Retrieval penalty bounds (Spec-13: <= 3 positions, never out of window)
# ===========================================================================

class TestRankPenalty:
    def _rec(self, trust, name):
        return SimpleNamespace(trust_class=trust, name=name)

    def test_sink_is_bounded_to_three(self):
        from argos.store import DuckDBMemoryStore

        records = [
            self._rec("unreviewed", "U"),
            self._rec(None, "c1"),
            self._rec(None, "c2"),
            self._rec(None, "c3"),
            self._rec(None, "c4"),
        ]
        DuckDBMemoryStore._apply_trust_penalty(records, limit=5)
        names = [r.name for r in records]
        assert names.index("U") == 3  # moved 3, not more
        # Clean relative order preserved.
        assert [n for n in names if n.startswith("c")] == ["c1", "c2", "c3", "c4"]

    def test_never_pushed_out_of_window(self):
        from argos.store import DuckDBMemoryStore

        records = [
            self._rec("unreviewed", "U"),
            self._rec(None, "c1"),
            self._rec(None, "c2"),
        ]
        DuckDBMemoryStore._apply_trust_penalty(records, limit=2)
        names = [r.name for r in records]
        # Window is 2; the unreviewed record must remain inside it.
        assert names.index("U") < 2

    def test_insufficient_clean_records_sinks_less(self):
        from argos.store import DuckDBMemoryStore

        records = [self._rec("unreviewed", "U"), self._rec(None, "c1")]
        DuckDBMemoryStore._apply_trust_penalty(records, limit=5)
        assert [r.name for r in records].index("U") == 1

    def test_no_marker_no_reorder(self):
        from argos.store import DuckDBMemoryStore

        records = [self._rec(None, "a"), self._rec(None, "b"),
                   self._rec("clean", "c")]
        before = [r.name for r in records]
        DuckDBMemoryStore._apply_trust_penalty(records, limit=3)
        assert [r.name for r in records] == before

    def test_end_to_end_search_penalty(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            # The unreviewed record is the strongest lexical match; the
            # penalty must still place clean memories above it (bounded).
            store.remember(
                category="context_note",
                content="falcon falcon falcon falcon falcon sighting",
                source="explicit", trust_class="unreviewed",
            )
            for i, word in enumerate(["alpha", "bravo", "charlie", "delta"]):
                store.remember(
                    category="context_note",
                    content=f"falcon report {word}",
                    source="explicit",
                )
            hits = store.search("falcon", limit=5)
            assert len(hits) >= 4
            unrev_idx = next(
                i for i, h in enumerate(hits)
                if (h.trust_class or "") == "unreviewed"
            )
            assert 0 < unrev_idx <= 3
            # Window guarantee: still returned at limit=2 (clamped sink).
            hits2 = store.search("falcon", limit=2)
            assert any((h.trust_class or "") == "unreviewed" for h in hits2)
        finally:
            store.close()


# ===========================================================================
# Version carry-forward: edits can never launder a marker
# ===========================================================================

class TestCarryForward:
    def test_update_memory_carries_trust_class(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            mem = store.remember(
                category="context_note",
                content="draft note about the harbour project",
                source="explicit",
                trust_class="unreviewed",
            )
            assert mem.trust_class == "unreviewed"
            new = store.update_memory(mem.memory_id, content="final note about the harbour project")
            assert new is not None
            assert new.trust_class == "unreviewed"
        finally:
            store.close()

    def test_remember_unknown_trust_class_fails_closed(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            mem = store.remember(
                category="context_note",
                content="note with a bogus marker",
                source="explicit",
                trust_class="banana",
            )
            assert mem.trust_class == "unreviewed"
        finally:
            store.close()

    def test_remember_clean_default(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            mem = store.remember(
                category="context_note",
                content="ordinary note",
                source="explicit",
            )
            assert mem.trust_class is None
        finally:
            store.close()


# ===========================================================================
# Service boundary: server-derived policy, client values stripped
# ===========================================================================

class TestRpcRoundTrip:
    """Spec-13 (#393 S1): trust_class survives the RPC serialization seam.

    The live acceptance probe caught this gap: the store row and to_dict()
    carried the marker, but the client-side _record_from_dict whitelist
    dropped it, so consumers saw trust_class=None for every record.
    """

    def test_record_from_dict_carries_trust_class(self):
        from argos.service_client import _record_from_dict

        rec = _record_from_dict({
            "memory_id": "mem-roundtrip",
            "category": "context_note",
            "content": "round-trip carrier",
            "source": "llm_extraction",
            "trust_class": "unreviewed",
        })
        assert rec.trust_class == "unreviewed"

    def test_record_from_dict_missing_trust_class_defaults_none(self):
        from argos.service_client import _record_from_dict

        rec = _record_from_dict({
            "memory_id": "mem-plain",
            "category": "context_note",
            "content": "no marker",
            "source": "llm_extraction",
        })
        assert rec.trust_class is None


class TestServiceBoundary:
    def test_approval_mode_is_forbidden_client_arg(self):
        from memory_service import _FORBIDDEN_CLIENT_ARGS

        assert "approval_mode" in _FORBIDDEN_CLIENT_ARGS

    def test_sanitize_strips_client_approval_mode(self):
        from memory_service import _sanitize_args

        cleaned = _sanitize_args({"approval_mode": "auto", "category": "x"})
        assert "approval_mode" not in cleaned
        assert cleaned["category"] == "x"

    def test_tenant_policy_default_auto(self):
        from memory_service import TenantPolicy

        assert TenantPolicy({}).approval_mode == "auto"

    def test_tenant_policy_override(self):
        from memory_service import TenantPolicy

        assert TenantPolicy({"approval_mode": "human"}).approval_mode == "human"

    def test_tenant_policy_invalid_fails_closed(self):
        from memory_service import TenantPolicy

        assert TenantPolicy({"approval_mode": "banana"}).approval_mode == "human"

    def test_tenant_policy_to_dict_includes_key(self):
        from memory_service import TenantPolicy

        assert TenantPolicy({}).to_dict()["approval_mode"] == "auto"

    def test_call_store_applies_policy_value(self):
        """_call_store resolves approval_mode from the tenant policy and
        overwrites whatever the wire carried."""
        from memory_service import MemoryService, TenantPolicy

        captured = {}

        class FakeStore:
            def set_user_scope(self, user_id):
                pass

            def save_candidate(self, **kwargs):
                captured.update(kwargs)
                return {"candidate_id": "cand-x", "status": "pending"}

        svc = SimpleNamespace()
        args = {
            "category": "preference",
            "content": CLEAN,
            "approval_mode": "human",  # client-supplied: must be ignored
        }
        MemoryService._call_store(
            svc, "save_candidate", args, "test_user", FakeStore(),
            policy=TenantPolicy({"approval_mode": "auto"}),
        )
        assert captured["approval_mode"] == "auto"

    def test_call_store_without_policy_defaults_auto(self):
        from memory_service import MemoryService

        captured = {}

        class FakeStore:
            def set_user_scope(self, user_id):
                pass

            def save_candidate(self, **kwargs):
                captured.update(kwargs)
                return None

        svc = SimpleNamespace()
        MemoryService._call_store(
            svc, "save_candidate",
            {"category": "preference", "content": CLEAN},
            "test_user", FakeStore(), policy=None,
        )
        assert captured["approval_mode"] == "auto"


# ===========================================================================
# Config plumbing
# ===========================================================================

class TestConfigPlumbing:
    def test_memory_config_default_auto(self):
        from config_model import MemoryConfig

        cfg = MemoryConfig()
        assert cfg.approval_mode == "auto"

    def test_memory_config_invalid_fails_closed(self):
        from config_model import MemoryConfig

        cfg = MemoryConfig(approval_mode="banana")
        assert cfg.approval_mode == "human"

    def test_schema_carries_approval_mode(self):
        from config_schema import CONFIG_SCHEMA

        keys = [f.key for f in CONFIG_SCHEMA.fields]
        assert "approval_mode" in keys
