"""Write mixin: remember, supersession, candidates, evidence and deletion.

Extracted verbatim from store.py during the god-file split (behavior-
neutral: no renames, no fixes).
"""
from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

try:
    from .store_common import (
        GROUNDING_EXTRACTED,
        GROUNDING_INFERRED,
        GROUNDING_SPECULATIVE,
        MemoryRecord,
        PROVENANCE_EXTERNAL,
        PROVENANCE_INTERNAL,
        VALID_CATEGORIES,
        _DEFAULT_TTL_DAYS,
        _GROUNDING_CEILING,
        _NOT_PROVIDED,
        default_grounding_for_write,
        grounding_allows_status,
        normalize_grounding,
        normalize_provenance,
        rejection_key,
        sanitize_content,
    )
except ImportError:  # store_write.py imported as a top-level module
    from store_common import (
        GROUNDING_EXTRACTED,
        GROUNDING_INFERRED,
        GROUNDING_SPECULATIVE,
        MemoryRecord,
        PROVENANCE_EXTERNAL,
        PROVENANCE_INTERNAL,
        VALID_CATEGORIES,
        _DEFAULT_TTL_DAYS,
        _GROUNDING_CEILING,
        _NOT_PROVIDED,
        default_grounding_for_write,
        grounding_allows_status,
        normalize_grounding,
        normalize_provenance,
        rejection_key,
        sanitize_content,
    )
try:
    from .value_extractor import extract_values, is_transition_statement, values_conflict
except ImportError:  # store_write.py imported as a top-level module
    from value_extractor import extract_values, is_transition_statement, values_conflict
try:
    from .structural_loss import LossReport, is_append_only, structural_loss_guard
except ImportError:  # store_write.py imported as a top-level module
    from structural_loss import LossReport, is_append_only, structural_loss_guard

logger = logging.getLogger(__name__)


class StoreWriteMixin:
    """Write-path methods for DuckDBMemoryStore."""

    def remember(
        self,
        category: str,
        content: str,
        tags: List[str] | None = None,
        payload: Dict[str, Any] | None = None,
        dedup: bool = True,
        *,
        source: str = "explicit",
        confidence: float | None = 1.0,
        durability: str | None = None,
        scope: str = "profile",
        project_id: str | None = None,
        namespace: str = "conversation",
        client_scope: str | None = None,
        doc_class: str | None = None,
        source_doc_id: str | None = None,
        source_loc: str | None = None,
        extraction_method: str | None = None,
        extracted_at: str | None = None,
        verified_state: str = "current",
        verified_at: str | None = None,
        status: str = "active",
        expires_at: Any = _NOT_PROVIDED,
        provenance_origin: Any = None,
        grounding: Any = None,
        created_at: Any = None,
    ) -> MemoryRecord | None:
        """Insert a memory record. Returns None if deduped away.

        *expires_at* semantics (Spec 1):
        - ``_NOT_PROVIDED`` (default): auto-TTL logic applies (current behavior).
        - ``None``: explicitly no expiry (skip auto-TTL, store NULL).
        - ISO-8601 string: set to that value (explicit wins over TTL map).

        *created_at* (issue #8): override the creation timestamp. By default
        the wall clock is used. Pass an ISO-8601 string to backdate a memory
        to its in-world date (e.g. a conversation from 2022-01-03 ingested
        today). This also sets ``valid_from`` so version-chain/supersession
        logic operates on in-world order, not ingest order. The ``updated_at``
        column always gets the wall clock (the record was physically written
        now).

        Trust-model (batch-2):
        - *provenance_origin* (#43): internal/external taint. When None, derived
          from the payload's external_source flag. Fail-closed to external.
        - *grounding* (#40): observed/extracted/inferred/speculative. When None,
          derived from the write path (source / external). Fail-closed to
          speculative. Sanitization never alters either label.
        """
        if not content or not content.strip():
            return None
        content, _inj = sanitize_content(content)
        if _inj:
            raise ValueError(
                f"Content blocked: stored text matches an instruction-injection "
                f"pattern ({_inj}). Refusing to write."
            )
        if category not in VALID_CATEGORIES:
            logger.warning("Unknown category '%s', defaulting to context_note", category)
            category = "context_note"

        record_payload = dict(payload or {})
        record_payload.setdefault("user_scope", self.user_id)
        source = str(source or record_payload.get("source") or "explicit")
        # Provenance taint (#43): per-record, fail-closed. An explicit label
        # wins; otherwise derive from the payload's external_source flag.
        is_external = bool(record_payload.get("external_source")) or (
            str(record_payload.get("provenance_origin") or "").strip().lower()
            == PROVENANCE_EXTERNAL
        )
        # Ingestion-time inbound security scan (#19): when content arrives
        # from an external/untrusted channel, scan it at the boundary before
        # it enters the store. The scanner catches injection, suppression,
        # and mutation patterns that sanitize_content's instruction-injection
        # check doesn't cover. Blocked content is refused (raises ValueError)
        # so it never becomes retrievable memory.
        if is_external:
            try:
                if __package__:
                    from .inbound_security import scan_inbound_text
                else:
                    from inbound_security import scan_inbound_text
                _scan = scan_inbound_text(content)
                if _scan.blocked:
                    raise ValueError(
                        f"Content blocked by inbound security scan: "
                        f"{_scan.summary()}. External-origin content "
                        f"matching poisoning/injection patterns is refused."
                    )
            except ImportError:
                logger.warning(
                    "Inbound security scanner unavailable for external memory "
                    "— refusing write as fail-closed"
                )
                raise ValueError(
                    "Inbound security scanner unavailable; external-origin "
                    "memory cannot be written without the security gate."
                )
        if provenance_origin is not None:
            prov = normalize_provenance(provenance_origin)
        else:
            prov = PROVENANCE_EXTERNAL if is_external else PROVENANCE_INTERNAL
        # Grounding (#40): default per write path; explicit wins.
        ground = default_grounding_for_write(
            source=source, external=is_external, explicit_grounding=grounding
        )
        durability = str(
            durability or (
                "temporary" if category in {"context_note", "event", "goal"} else "durable"
            )
        )
        scope = str(scope or "profile")
        status = str(status or "active")
        if status not in {"active", "quarantined"}:
            status = "active"
        record_payload.setdefault("source", source)

        # Explicit expires_at wins over the TTL map.  None = no expiry.
        skip_auto_ttl = expires_at is not _NOT_PROVIDED
        if expires_at is not _NOT_PROVIDED and expires_at is not None:
            # #80: normalize to the aware ISO format (_now() shape) so
            # lexicographic VARCHAR comparisons in SQL are consistent
            # across mixed ISO forms (Z vs +00:00, date-only vs full).
            normalized_expiry = self._normalize_timestamp(expires_at)
            if normalized_expiry is not None:
                record_payload["expires_at"] = normalized_expiry
            else:
                logger.warning(
                    "Unparseable expires_at %r — storing as-is (may compare "
                    "wrong lexicographically)", expires_at,
                )
                record_payload["expires_at"] = str(expires_at)
        elif expires_at is None:
            # Explicit None: clear any pre-existing payload expiry so the
            # column (populated from record_payload below) is NULL, not
            # a stale value from the caller's payload dict.
            record_payload.pop("expires_at", None)

        if dedup:
            _dedup_reason = self._content_exists(
                content, category,
                namespace=namespace,
                client_scope=client_scope,
                doc_class=doc_class,
                source_doc_id=source_doc_id,
            )
            if _dedup_reason:
                # #82: surface the dedup-drop reason at warning level so
                # silent no-ops are visible to the operator/caller.
                logger.warning(
                    "Deduped memory (%s): %s", _dedup_reason, content[:60],
                )
                return None

        # Deletion tombstone check: this exact fact was hard-deleted by the
        # user; re-feeding the same content would silently resurrect it.
        _ts = self.tombstone_check(content, category)
        if _ts:
            logger.info(
                "Blocked re-creation of deleted memory (tombstone %s): %s",
                _ts.get("created_at", ""), content[:60],
            )
            return None

        # Rejection-ledger check (#39): a previously-rejected claim slot may not
        # be re-created directly. Re-assertion must come back through the
        # proposal queue (save_candidate) so the gates re-apply. Keyed by
        # (subject, predicate, scope) so paraphrased re-assertions are blocked.
        _rj = self.rejection_check(category, record_payload)
        if _rj:
            logger.info(
                "Blocked re-creation of rejected claim (ledger %s): %s",
                _rj.get("created_at", ""), content[:60],
            )
            return None

        memory_id = f"mem-{uuid.uuid4().hex}"
        now = self._now()
        # created_at override (issue #8): backdate to an in-world date so
        # version-chain/supersession logic sees in-world order. valid_from
        # follows created_at (a memory is valid from its in-world creation,
        # not from when it was ingested). updated_at stays at the wall clock
        # (the row was physically written now).
        if created_at is not None:
            # #80: normalize created_at to the aware ISO format so
            # valid_from and version-chain comparisons are consistent.
            created_ts = self._normalize_timestamp(created_at) or str(created_at)
        else:
            created_ts = now
        if not skip_auto_ttl and not record_payload.get("expires_at") and durability == "temporary":
            if getattr(self, "expiry_enabled", False):
                ttl_map = getattr(self, "ttl_days", _DEFAULT_TTL_DAYS)
                default_days = getattr(self, "expiry_default_days", 90)
                ttl_days = ttl_map.get(category, default_days)
            else:
                ttl_days = _DEFAULT_TTL_DAYS.get(category)
            if ttl_days:
                record_payload["expires_at"] = (
                    datetime.now(timezone.utc) + timedelta(days=ttl_days)
                ).isoformat()
        emb: List[float] = []
        if self.embedder and hasattr(self.embedder, "embed"):
            emb = self.embedder.embed(content)
        # Issue #83: a record stored without an embedding can never be
        # similarity-scored (the vector leg filters `embedding IS NOT NULL`
        # and graph-boost retrieval scores NULL-embedding records 0.0).
        # Surface this loudly instead of silently degrading retrieval
        # quality for the affected records. On embedder recovery the
        # opportunistic backfill below re-embeds them.
        if not emb:
            logger.warning(
                "Storing memory without embedding (embedder unavailable) — "
                "record will be text-search-only until backfilled: %s",
                content[:80],
            )

        sql = """
            INSERT INTO memory_records
                (memory_id, category, content, tags, payload, created_at, updated_at,
                 expires_at, embedding, status, source, confidence, durability, scope,
                 project_id, user_scope, namespace, client_scope, doc_class,
                 source_doc_id, source_loc, extraction_method, extracted_at,
                 verified_state, verified_at,
                 retrieval_count, helpful_count, dismissed_count,
                 valid_from, provenance_origin, grounding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, 0, ?, ?, ?)
        """
        with self._lock:
            assert self.connection is not None
            self.connection.execute(sql, [
                memory_id, category, content, tags or [],
                json.dumps(record_payload), created_ts, now,
                record_payload.get("expires_at"),
                emb if emb else None,
                status, source, confidence, durability, scope, project_id,
                record_payload.get("user_scope"),
                namespace or "conversation", client_scope, doc_class,
                source_doc_id, source_loc, extraction_method, extracted_at,
                verified_state or "current", verified_at,
                created_ts,  # valid_from = in-world creation time (issue #8)
                prov, ground,
            ])
        fetched = self._fetch_records(
            "SELECT * FROM memory_records WHERE memory_id = ?", [memory_id]
        )
        # Issue #83: if the embedder just recovered from a prior failure,
        # opportunistically backfill records that were written with NULL
        # embeddings during the outage. Runs once per recovery (the flag
        # is cleared after) so it doesn't fire on every remember().
        if emb and getattr(self.embedder, "recovered", False):
            try:
                self.embedder.recovered = False
                backfilled = self.backfill_null_embeddings()
                if backfilled:
                    logger.info(
                        "Re-embedded %d NULL-embedding records after embedder "
                        "recovery (issue #83)", backfilled,
                    )
            except Exception as e:
                logger.debug("Opportunistic NULL-embedding backfill skipped: %s", e)
        return fetched[0] if fetched else None

    # -- versioned ingest (benchmark update-arithmetic, #74) ------------------

    def ingest_versioned(
        self,
        category: str,
        content: str,
        tags: List[str] | None = None,
        payload: Dict[str, Any] | None = None,
        **remember_kwargs: Any,
    ) -> tuple[MemoryRecord | None, str]:
        """Ingest with store-level versioning — fire update-arithmetic on restatements.

        Issue #74: the benchmark ingest path called ``remember(dedup=False)``
        for every turn, so supersession/version links and tombstones never
        engaged — ``valid_to``/``superseded_by`` stayed NULL on every row and
        chain-unfold had nothing to walk. This method is the store-side
        option-2 fix: a drop-in ingest API that detects restatements and
        routes them through :meth:`update_memory` so version chains form.

        Production behavior is unchanged — ``remember()``'s default
        ``dedup=True`` is untouched. Benchmark adapters call this instead of
        ``remember(dedup=False)``.

        Decision table (matched against CURRENT records only):

        - No similar record → ``remember(dedup=False)`` inserts a standalone
          row. Outcome ``"inserted"``.
        - Similar record, IDENTICAL content → true duplicate; return the
          existing record unchanged (no new row, no chain). Outcome
          ``"duplicate"``.
        - Similar record, DIFFERENT content → restatement / update; route
          through ``update_memory`` so the old record gets
          ``valid_to``/``superseded_by`` and a new version becomes the head.
          Outcome ``"superseded"``.

        *remember_kwargs* are forwarded to :meth:`remember` (insert path)
        or :meth:`update_memory` (supersede path). On the insert path all
        kwargs pass through (``created_at``, ``source``, ``confidence``,
        ``scope``, ``project_id``, ``provenance_origin``, ``grounding``,
        ``expires_at``). On the supersede path only ``created_at``,
        ``expires_at``, and ``structural_guard`` are forwarded — the other
        metadata (source, confidence, scope, etc.) is carried forward from
        the prior record by ``update_memory``, which is the correct
        behavior for a version update.

        Returns ``(record, outcome)`` where *outcome* is one of
        ``"inserted"`` / ``"superseded"`` / ``"duplicate"`` (or
        ``"blocked"`` when the insert was refused by a tombstone/rejection
        gate, mirroring ``remember`` returning ``None``).
        """
        if not content or not content.strip():
            return None, "blocked"
        existing_id, _reason = self._find_current_similar(content, category)
        if existing_id is None:
            rec = self.remember(
                category=category, content=content, tags=tags,
                payload=payload, dedup=False, **remember_kwargs,
            )
            return rec, "inserted" if rec is not None else "blocked"
        # A current record restates this content. If the text is identical
        # it is a true duplicate (no chain needed); otherwise treat it as an
        # update and supersede via update_memory so a version chain forms.
        existing = self._fetch_records(
            """SELECT * FROM memory_records
               WHERE memory_id = ?
                 AND (user_scope IS NULL OR user_scope = ?)""",
            [existing_id, self.user_id],
        )
        if not existing:
            # Raced away between the scan and the fetch — fall back to insert.
            rec = self.remember(
                category=category, content=content, tags=tags,
                payload=payload, dedup=False, **remember_kwargs,
            )
            return rec, "inserted" if rec is not None else "blocked"
        if existing[0].content == content:
            return existing[0], "duplicate"
        update_kwargs = {
            k: v for k, v in remember_kwargs.items()
            if k in {"expires_at", "structural_guard", "created_at"}
        }
        new_head = self.update_memory(
            memory_id=existing_id, content=content, tags=tags,
            payload_updates=payload, **update_kwargs,
        )
        return new_head, "superseded" if new_head is not None else "blocked"

    # -- value-supersession (stale-number detection) -------------------------

    def _find_conflicting_active_value(
        self,
        content: str,
        _category: str,
    ) -> Optional[tuple[str, str, str, str]]:
        """Find an active memory whose value conflicts with *content*.

        Extracts numeric values from *content* and checks ACTIVE records in
        ANY category for the same subject with a different value.  The scan
        is deliberately cross-category: stale-number pairs like an ``insight``
        headline and a ``context_note`` carry the same subject under different
        categories (the original 82.2%/89.8% incident).  False positives are
        cheap — a conflict only downgrades the candidate to
        ``pending_user_confirmation``, never auto-activates anything.
        Zero LLM — pure regex + token-overlap matching.

        Transition-verb gate (#36): only transition statements ("switched
        to", "changed to", "stopped", "now uses") trigger a value conflict.
        A plain restatement ("I use 449 rows") is treated as corroboration,
        not a supersession candidate — the user may hold several things at
        once. Without this gate, an over-eager conflict scan could silently
        supersede a coexisting true value.

        ``_category`` is kept for call-site compatibility but intentionally
        unused (dedup/embedding layers still respect category scoping).

        Returns ``(old_memory_id, old_content, new_value, old_value)`` for the
        first conflict found, or None if no conflict.
        """
        # Transition-verb gate (#36): plain restatements are corroboration,
        # not conflicts. Only transition statements can close a standing fact.
        if not is_transition_statement(content):
            return None
        new_values = extract_values(content)
        if not new_values:
            return None
        # D6 fix: pre-filter on subject tokens + LIMIT so the scan doesn't
        # full-table-scan every active record. Extract significant tokens
        # from the new content's value subjects (non-numeric, >=3 chars)
        # and filter SQL rows whose content contains at least one. This
        # prevents false negatives (a real conflict shares subject tokens)
        # while avoiding scanning unrelated records (e.g. weather records
        # when looking for salary conflicts).
        import re as _re
        _subject_tokens: set[str] = set()
        for ev in new_values:
            for tok in _re.findall(r"[a-z]+", ev.subject.lower()):
                if len(tok) >= 3:
                    _subject_tokens.add(tok)
        # Also include significant tokens from the full content so the
        # pre-filter isn't overly narrow.
        for tok in _re.findall(r"[a-z]+", content.lower()):
            if len(tok) >= 4:
                _subject_tokens.add(tok)
        with self._lock:
            assert self.connection is not None
            if _subject_tokens:
                # Build a LIKE clause for each token. DuckDB supports
                # ILIKE for case-insensitive matching.
                _like_clauses = " OR ".join(
                    f"content ILIKE '%' || ? || '%'" for _ in _subject_tokens
                )
                rows = self.connection.execute(
                    f"""SELECT memory_id, content FROM memory_records
                       WHERE valid_to IS NULL
                         AND (user_scope IS NULL OR user_scope = ?)
                         AND COALESCE(status, 'active') = 'active'
                         AND ({_like_clauses})
                       LIMIT 50""",
                    [self.user_id, *_subject_tokens],
                ).fetchall()
            else:
                rows = self.connection.execute(
                    """SELECT memory_id, content FROM memory_records
                       WHERE valid_to IS NULL
                         AND (user_scope IS NULL OR user_scope = ?)
                         AND COALESCE(status, 'active') = 'active'
                       LIMIT 50""",
                    [self.user_id],
                ).fetchall()
        for old_id, old_content in rows:
            if old_content == content:
                continue  # same text — dedup handles this
            old_values = extract_values(old_content)
            if not old_values:
                continue
            # D6: use a lower subject_threshold (0.2) than the default 0.5
            # because the token window around the value can be wide (6
            # tokens each side), diluting Jaccard similarity for legitimate
            # same-subject pairs. False positives are cheap — the candidate
            # is only downgraded to pending_user_confirmation.
            conflict = values_conflict(new_values, old_values, subject_threshold=0.2)
            if conflict:
                new_v, old_v = conflict
                return (old_id, old_content, new_v.value, old_v.value)
        return None

    def _mark_superseded(
        self,
        memory_id: str,
        reason: str,
        superseded_by: str | None = None,
    ) -> bool:
        """Mark a memory as superseded (set valid_to).

        This is the storage-level supersession — it sets ``valid_to = now``
        and optionally ``superseded_by``.  The old record is preserved for
        history queries.  Returns True if the record was found and updated.
        """
        now = self._now()
        with self._lock:
            assert self.connection is not None
            check = self.connection.execute(
                """SELECT 1 FROM memory_records
                   WHERE memory_id = ?
                     AND (user_scope IS NULL OR user_scope = ?)
                     AND valid_to IS NULL""",
                [memory_id, self.user_id],
            ).fetchone()
            if not check:
                return False
            self.connection.execute(
                """UPDATE memory_records
                   SET valid_to = ?, superseded_by = ?, updated_at = ?
                   WHERE memory_id = ?
                     AND valid_to IS NULL
                     AND (user_scope IS NULL OR user_scope = ?)""",
                [now, superseded_by, now, memory_id, self.user_id],
            )
        logger.info(
            "Value-supersession: %s superseded (%s) by %s",
            memory_id, reason, superseded_by or "N/A",
        )
        return True

    # -- candidate/proposal queue ---------------------------------------------

    @staticmethod
    def _candidate_payload(raw: Any) -> dict:
        if isinstance(raw, str):
            try:
                value = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                value = {}
        else:
            value = raw or {}
        return value if isinstance(value, dict) else {}

    def _candidate_row_to_dict(self, row: tuple) -> dict:
        (
            candidate_id, category, content, tags, payload, source, confidence,
            durability, scope, project_id, namespace, client_scope, doc_class, session_id,
            status, created_at,
            updated_at, reviewed_at, review_reason, evidence_text, evidence_role,
            source_timestamp, review_confidence, review_model,
            quarantine_reason, quarantined_at, provenance_origin, grounding,
        ) = row
        return {
            "candidate_id": candidate_id,
            "category": category,
            "content": content,
            "tags": list(tags or []),
            "payload": self._candidate_payload(payload),
            "source": source,
            "confidence": confidence,
            "durability": durability,
            "scope": scope,
            "project_id": project_id,
            "namespace": namespace or "conversation",
            "client_scope": client_scope,
            "doc_class": doc_class,
            "session_id": session_id,
            "status": status,
            "created_at": created_at,
            "updated_at": updated_at,
            "reviewed_at": reviewed_at,
            "review_reason": review_reason,
            "evidence_text": evidence_text,
            "evidence_role": evidence_role,
            "source_timestamp": source_timestamp,
            "review_confidence": review_confidence,
            "review_model": review_model,
            "quarantine_reason": quarantine_reason,
            "quarantined_at": quarantined_at,
            "provenance_origin": normalize_provenance(provenance_origin),
            "grounding": normalize_grounding(grounding),
        }

    def save_candidate(
        self,
        category: str,
        content: str,
        tags: List[str] | None = None,
        payload: Dict[str, Any] | None = None,
        *,
        source: str = "llm_extraction",
        confidence: float | None = 0.5,
        durability: str = "durable",
        scope: str = "profile",
        project_id: str | None = None,
        namespace: str = "conversation",
        client_scope: str | None = None,
        doc_class: str | None = None,
        source_doc_id: str | None = None,
        session_id: str = "",
        evidence_text: str = "",
        evidence_role: str = "user_turn",
        source_timestamp: str | None = None,
        dedup: bool = True,
        external: bool = False,
        provenance_origin: Any = None,
        grounding: Any = None,
    ) -> dict | None:
        """Store a pending proposal without making it retrievable memory.

        Trust-model (batch-2): *provenance_origin* (#43) and *grounding* (#40)
        are derived from the write path when not passed explicitly, and carried
        onto the approved memory when the candidate is activated.
        """
        if not content or not content.strip():
            return None
        content, _inj = sanitize_content(content)
        if not content or not content.strip():
            return None
        candidate_status = "quarantined" if _inj else "pending"
        quarantine_reason = f"injection_pattern: {_inj}" if _inj else None
        # Ingestion-time inbound security scan (#19): when content arrives
        # from an external/untrusted channel, scan it at the boundary before
        # it enters the candidate queue. The scanner catches injection,
        # suppression, and mutation patterns that sanitize_content's
        # instruction-injection check doesn't cover. Blocked content is
        # quarantined (not silently dropped) so a human can review it.
        if external or bool((payload or {}).get("external_source")):
            try:
                if __package__:
                    from .inbound_security import scan_inbound_text
                else:
                    from inbound_security import scan_inbound_text
                _scan = scan_inbound_text(content)
                if _scan.blocked:
                    candidate_status = "quarantined"
                    quarantine_reason = (
                        f"inbound_security: {_scan.summary()}"
                    )
                    logger.warning(
                        "Inbound security scan blocked external candidate: %s",
                        _scan.summary(),
                    )
            except ImportError:
                logger.warning(
                    "Inbound security scanner unavailable for external candidate "
                    "— quarantining as fail-closed"
                )
                candidate_status = "quarantined"
                quarantine_reason = "inbound_security_scanner_unavailable"
        if category not in VALID_CATEGORIES:
            category = "context_note"
        if dedup:
            with self._lock:
                assert self.connection is not None
                existing = self.connection.execute(
                    """SELECT candidate_id FROM memory_candidates
                       WHERE category = ? AND content = ? AND status = 'pending'
                         AND (user_scope IS NULL OR user_scope = ?)
                       LIMIT 1""",
                    [category, content.strip(), self.user_id],
                ).fetchone()
            if existing:
                return None
            # D7 fix: substring-overlap dedup. The exact-match check above
            # only catches identical text. A paraphrased candidate that
            # contains (or is contained by) an existing pending candidate
            # with >=0.8 overlap ratio should also be deduped — mirroring
            # remember()'s Layer 2 substring dedup. Without this, the
            # candidate queue accumulates near-duplicate proposals that
            # the reviewer has to dismiss one by one.
            content_stripped = content.strip()
            content_lower = content_stripped.lower()
            if len(content_lower) > 20:
                with self._lock:
                    assert self.connection is not None
                    near_dupes = self.connection.execute(
                        """SELECT candidate_id, content FROM memory_candidates
                           WHERE category = ? AND status = 'pending'
                             AND (user_scope IS NULL OR user_scope = ?)
                             AND length(content) > 20
                           ORDER BY created_at DESC
                           LIMIT 500""",
                        [category, self.user_id],
                    ).fetchall()
                for _cand_id, _existing_content in near_dupes:
                    _existing_lower = _existing_content.lower().strip()
                    if content_lower == _existing_lower:
                        return None
                    if content_lower in _existing_lower or _existing_lower in content_lower:
                        _shorter = min(len(content_lower), len(_existing_lower))
                        _longer = max(len(content_lower), len(_existing_lower))
                        if _longer > 0 and _shorter / _longer >= 0.85:
                            logger.info(
                                "Deduped candidate (substring overlap "
                                "%.0f%%): %s",
                                100 * _shorter / _longer, content_stripped[:60],
                            )
                            return None
            # Tombstone check (issue #46): a hard-deleted fact must not
            # re-enter the proposal queue. Same fingerprint as remember()'s
            # tombstone_check, applied at proposal time so the reviewer
            # never sees a resurrected fact.
            _ts = self.tombstone_check(content, category)
            if _ts:
                logger.info(
                    "Blocked re-proposal of deleted fact (tombstone %s): %s",
                    _ts.get("created_at", ""), content[:60],
                )
                return None
            # Rejection-ledger check (#39): a previously-rejected claim slot may
            # not re-enter the proposal queue. The one-way ladder refuses
            # resurrection at the gate so the reviewer never sees it again.
            _rj = self.rejection_check(category, payload or {})
            if _rj:
                logger.info(
                    "Blocked re-proposal of rejected claim (ledger %s): %s",
                    _rj.get("created_at", ""), content[:60],
                )
                return None

        candidate_id = f"cand-{uuid.uuid4().hex}"
        now = self._now()
        evidence_text = (evidence_text or "")[:8000]
        source_timestamp = source_timestamp or now
        candidate_payload = dict(payload or {})
        candidate_payload.setdefault("user_scope", self.user_id)
        candidate_payload.setdefault("source", source)
        # #115: carry source_doc_id through the candidate payload so
        # review_candidate can forward it to remember() as a dedicated
        # kwarg (memory_candidates has no source_doc_id column, but
        # memory_records does — the dedup scope clause needs the column).
        if source_doc_id is not None:
            candidate_payload["source_doc_id"] = source_doc_id
        if external:
            candidate_payload["external_source"] = True
        # Trust-model defaults (batch-2): provenance taint + grounding.
        # An explicit kwarg wins; otherwise a payload override (e.g. the #35
        # quote-verification downgrade writing payload.grounding=inferred) is
        # honored; otherwise the value is derived from the write path.
        is_external = external or bool(candidate_payload.get("external_source"))
        if provenance_origin is not None:
            prov = normalize_provenance(provenance_origin)
        elif candidate_payload.get("provenance_origin"):
            prov = normalize_provenance(candidate_payload.get("provenance_origin"))
        else:
            prov = PROVENANCE_EXTERNAL if is_external else PROVENANCE_INTERNAL
        payload_grounding = candidate_payload.get("grounding")
        ground = default_grounding_for_write(
            source=source, external=is_external,
            explicit_grounding=(grounding if grounding is not None else payload_grounding),
        )
        # Value-supersession detection (issue #4): check if this candidate's
        # numeric value conflicts with an existing active fact.  If so, record
        # the conflict in the payload so the reviewer can surface it as a
        # supersession proposal.  Zero LLM — pure regex + token-overlap.
        try:
            conflict = self._find_conflicting_active_value(content, category)
            if conflict:
                old_id, old_content, new_val, old_val = conflict
                candidate_payload["value_supersession"] = {
                    "supersedes_memory_id": old_id,
                    "old_content": old_content[:200],
                    "new_value": new_val,
                    "old_value": old_val,
                }
        except Exception as exc:
            logger.debug("Value-supersession check failed: %s", exc)
        try:
            normalized_confidence = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            normalized_confidence = 0.5
        with self._lock:
            assert self.connection is not None
            self.connection.execute(
                """INSERT INTO memory_candidates
                  (candidate_id, category, content, tags, payload, source,
                   confidence, durability, scope, project_id,
                   namespace, client_scope, doc_class,
                   session_id,
                   user_scope, status, created_at, updated_at, evidence_text,
                   evidence_role, source_timestamp, quarantine_reason, quarantined_at,
                   provenance_origin, grounding)
                  VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    candidate_id, category, content.strip(), tags or [],
                    json.dumps(candidate_payload), source, normalized_confidence,
                    durability or "durable", scope or "profile", project_id,
                    namespace or "conversation", client_scope, doc_class,
                    session_id or "", candidate_payload.get("user_scope"),
                    candidate_status, now, now, evidence_text,
                    evidence_role or "user_turn", source_timestamp,
                    quarantine_reason, (now if _inj else None),
                    prov, ground,
                ],
            )
        return self.list_candidates(candidate_id=candidate_id, limit=1)[0]

    def list_candidates(
        self,
        status: str = "pending",
        *,
        candidate_id: str | None = None,
        limit: int = 100,
    ) -> List[dict]:
        conditions = [
            "(user_scope IS NULL OR user_scope = ?)"
        ]
        params: list[Any] = [self.user_id]
        if candidate_id:
            conditions.append("candidate_id = ?")
            params.append(candidate_id)
        elif status:
            conditions.append("status = ?")
            params.append(status)
        sql = "SELECT candidate_id, category, content, tags, payload, source, confidence, durability, scope, project_id, namespace, client_scope, doc_class, session_id, status, created_at, updated_at, reviewed_at, review_reason, evidence_text, evidence_role, source_timestamp, review_confidence, review_model, quarantine_reason, quarantined_at, provenance_origin, grounding FROM memory_candidates"
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(max(1, min(int(limit), 500)))
        with self._lock:
            assert self.connection is not None
            rows = self.connection.execute(sql, params).fetchall()
        return [self._candidate_row_to_dict(row) for row in rows]

    def project_digest(
        self,
        project_id: str | None = None,
        *,
        status: str = "pending",
        limit: int = 100,
    ) -> Dict[str, Any]:
        """Per-project pending-proposal digest (#47).

        Returns a digest of candidates grouped by project, with count and
        list for each. When *project_id* is provided, only that project's
        candidates are returned. When None, all projects are grouped.

        Args:
            project_id: Filter to a specific project, or None for all.
            status: Candidate status to filter on (default "pending").
            limit: Max candidates per project.

        Returns:
            A dict with:
            - ``projects``: list of {project_id, count, candidates}
            - ``global_count``: count of unscoped (project_id IS NULL) candidates
        """
        conditions = [
            "(user_scope IS NULL OR user_scope = ?)",
            "status = ?",
        ]
        params: list[Any] = [self.user_id, status]
        if project_id is not None:
            conditions.append("project_id = ?")
            params.append(project_id)
        sql = (
            "SELECT candidate_id, category, content, tags, payload, source, "
            "confidence, durability, scope, project_id, "
            "namespace, client_scope, doc_class, session_id, status, "
            "created_at, updated_at, reviewed_at, review_reason, evidence_text, "
            "evidence_role, source_timestamp, review_confidence, review_model, "
            "quarantine_reason, quarantined_at, provenance_origin, grounding "
            "FROM memory_candidates"
        )
        sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(max(1, min(int(limit), 500)))
        with self._lock:
            assert self.connection is not None
            rows = self.connection.execute(sql, params).fetchall()
        candidates = [self._candidate_row_to_dict(row) for row in rows]
        # Group by project_id.
        by_project: Dict[str, List[dict]] = {}
        global_candidates: List[dict] = []
        for cand in candidates:
            pid = cand.get("project_id")
            if pid:
                by_project.setdefault(pid, []).append(cand)
            else:
                global_candidates.append(cand)
        projects = [
            {
                "project_id": pid,
                "count": len(cands),
                "candidates": cands,
            }
            for pid, cands in sorted(by_project.items())
        ]
        return {
            "projects": projects,
            "global_count": len(global_candidates),
            "global_candidates": global_candidates,
        }

    def review_candidate(
        self,
        candidate_id: str,
        decision: str,
        reason: str = "",
        *,
        review_confidence: float | None = None,
        review_model: str = "",
        durability: str | None = None,
        scope: str | None = None,
        evidence_retention: str = "full",
        supersedes_memory_id: str | None = None,
        expires_at: Any = _NOT_PROVIDED,
        review_source: str = "manual",
    ) -> dict | None:
        """Approve, reject, quarantine, or classify a pending proposal.
        When *supersedes_memory_id* is set and the decision is an approval,
        the newly created memory is chained behind the named current memory
        (the old record is superseded: valid_to set, superseded_by pointed
        at the new version). This is the confirm-first way chains grow —
        never automatic, always a reviewer decision.
        *expires_at* (Spec 1): when provided (not _NOT_PROVIDED), passed to
        remember() for the new memory. None = no expiry; string = set.
        *review_source* labels who is making the transition. The storage
        layer enforces one invariant (review point 4): the "approved"
        transition is reserved for the agent-facing confirmation tool
        (review_source="tool") and manual callers — an unsupervised
        automatic review (review_source="auto_review") can never write
        "approved"; it must use "reviewed_approved" (LLM-approved, awaiting
        user confirmation). This is enforced here, at the storage boundary,
        not by prompt instructions.
        """
        decision = decision.strip().lower()
        allowed = {
            "approved", "rejected", "quarantined", "reviewed_approved",
            "pending_user_confirmation",
        }
        if decision not in allowed:
            raise ValueError("invalid candidate review decision")
        if decision == "approved" and review_source == "auto_review":
            raise ValueError(
                "approval invariant: automatic review may not set 'approved'; "
                "use 'reviewed_approved' (user confirmation happens via the "
                "memory_candidate_review tool)"
            )
        candidates = self.list_candidates(candidate_id=candidate_id, limit=1)
        if not candidates:
            return None
        candidate = candidates[0]
        if candidate["status"] not in {"pending", "reviewed_approved", "pending_user_confirmation"}:
            return {"candidate": candidate, "memory": None}
        # Stored-prompt-injection guard: a pending candidate whose content
        # mimics instructions (written before the injection scan shipped, or
        # via a bypass) can never be approved — refuse loudly instead.
        if decision in {"approved", "reviewed_approved"}:
            _, _inj = sanitize_content(candidate["content"])
            if _inj:
                raise ValueError(
                    f"approval refused: candidate content matches an "
                    f"instruction-injection pattern ({_inj})"
                )
        # Storage-boundary external-source invariant (#43, structural): when
        # the policy is on, an unsupervised auto-review can never activate
        # external-origin memory — the decision is downgraded to
        # pending_user_confirmation. This mirrors the payload.external_source
        # check but keys on the permanent provenance_origin column, which
        # survives payload stripping/sanitization (the taint is not laundered).
        # The agent-facing confirmation tool (review_source="tool") and manual
        # callers may still approve it.
        candidate_provenance = candidate.get("provenance_origin", PROVENANCE_INTERNAL)
        is_external_origin = (
            candidate_provenance == PROVENANCE_EXTERNAL
            or (isinstance(candidate.get("payload"), dict)
                and candidate["payload"].get("external_source"))
        )
        if (
            review_source == "auto_review"
            and decision in {"approved", "reviewed_approved"}
            and getattr(self, "external_sources_require_confirmation", True)
            and is_external_origin
        ):
            decision = "pending_user_confirmation"
            reason = (
                "Storage boundary: external-origin memory cannot auto-activate "
                "(external_sources_require_confirmation). " + reason
            ).strip()
        # Grounding ceiling (#40): promotion may not raise a record past what
        # its grounding allows. User confirmation (tool/manual) LIFTS the
        # grounding to the minimum required for the requested class (the
        # ceiling moves with grounding, not with use); auto-review is capped
        # and downgrades to the ceiling instead. Recall counts are never
        # verification — they cannot reach this path.
        candidate_grounding = candidate.get("grounding", GROUNDING_SPECULATIVE)
        user_confirmed = review_source in {"tool", "manual"}
        if decision in {"approved", "reviewed_approved"}:
            if not grounding_allows_status(candidate_grounding, decision):
                if user_confirmed:
                    # User confirmation lifts the grounding (and the ceiling).
                    candidate_grounding = (
                        GROUNDING_EXTRACTED if decision == "approved"
                        else GROUNDING_INFERRED
                    )
                else:
                    ceiling = _GROUNDING_CEILING.get(
                        normalize_grounding(candidate_grounding),
                        "pending_user_confirmation",
                    )
                    decision = ceiling
                    reason = (
                        f"Grounding ceiling ({candidate.get('grounding')} "
                        f"-> {ceiling}). " + reason
                    ).strip()
        now = self._now()
        memory = None
        final_status = decision
        superseded_ok = False
        if decision in {"approved", "reviewed_approved"}:
            selected_durability = durability or candidate["durability"]
            selected_scope = scope or candidate["scope"]
            # #115: source_doc_id is carried in the candidate payload (no
            # source_doc_id column on memory_candidates) — extract it so
            # remember() stores it in the memory_records.source_doc_id column
            # and the doc-identity dedup scope can use it.
            _cand_payload = candidate.get("payload") or {}
            if isinstance(_cand_payload, str):
                try:
                    _cand_payload = json.loads(_cand_payload)
                except (json.JSONDecodeError, TypeError):
                    _cand_payload = {}
            remember_kwargs: Dict[str, Any] = dict(
                category=candidate["category"],
                content=candidate["content"],
                tags=candidate["tags"],
                payload=candidate["payload"],
                source=candidate["source"],
                confidence=candidate["confidence"],
                durability=selected_durability,
                scope=selected_scope,
                project_id=candidate["project_id"],
                namespace=candidate.get("namespace", "conversation"),
                client_scope=candidate.get("client_scope"),
                doc_class=candidate.get("doc_class"),
                source_doc_id=_cand_payload.get("source_doc_id"),
                provenance_origin=candidate_provenance,
                grounding=candidate_grounding,
            )
            # #115: when the reviewer has confirmed a supersession chain
            # (supersedes_memory_id is set), the new memory MUST be created
            # so the chain can grow.  If dedup fires first, remember()
            # returns None and the elif supersedes_memory_id branch below
            # never executes — the old value stays current forever.  The
            # reviewer has already decided this is a replacement, so dedup
            # would only re-litigate that decision and silently undo it.
            if supersedes_memory_id:
                remember_kwargs["dedup"] = False
            # Pass explicit expires_at through to remember() (Spec 1).
            if expires_at is not _NOT_PROVIDED:
                remember_kwargs["expires_at"] = expires_at
        # Wrap the entire review path (remember + supersede + candidate-status
        # + evidence) in one transaction so a crash between steps cannot leave
        # the chain forked or the candidate status inconsistent (issue #9).
        with self._lock:
            assert self.connection is not None
            self.connection.execute("BEGIN TRANSACTION")
            try:
                if decision in {"approved", "reviewed_approved"}:
                    memory = self.remember(**remember_kwargs)
                    if memory is None:
                        final_status = "deduplicated"
                    elif supersedes_memory_id:
                        # Chain the new memory behind the named current record.
                        # Same supersession semantics as update_memory, but the
                        # new version's content comes from the candidate, not
                        # the old record. Scope-checked: cannot supersede
                        # another user's memory.
                        check = self.connection.execute(
                            """SELECT 1 FROM memory_records
                               WHERE memory_id = ?
                                 AND (user_scope IS NULL OR user_scope = ?)
                                 AND valid_to IS NULL""",
                            [supersedes_memory_id, self.user_id],
                        ).fetchone()
                        if check:
                            self.connection.execute(
                                """UPDATE memory_records
                                   SET valid_to = ?, superseded_by = ?, updated_at = ?
                                   WHERE memory_id = ?
                                     AND valid_to IS NULL
                                     AND (user_scope IS NULL OR user_scope = ?)""",
                                [now, memory.memory_id, now, supersedes_memory_id,
                                 self.user_id],
                            )
                            superseded_ok = True
                            # Superseded-value re-assertion block (#36):
                            # Record the OLD memory's content+category in the
                            # tombstone table so a later session that
                            # re-mentions the old value cannot re-propose it
                            # as active. Without this, a superseded value
                            # re-mentioned in session B would re-enter the
                            # proposal queue and potentially re-activate,
                            # undoing the supersession from session A.
                            try:
                                old_row = self.connection.execute(
                                    """SELECT content, category FROM memory_records
                                       WHERE memory_id = ?""",
                                    [supersedes_memory_id],
                                ).fetchone()
                                if old_row:
                                    old_content, old_category = old_row
                                    h = self._tombstone_hash(old_content)
                                    if h and old_category:
                                        self.connection.execute(
                                            """INSERT OR REPLACE INTO deletion_tombstones
                                               (content_hash, category, user_scope, reason, created_at)
                                               VALUES (?, ?, ?, ?, ?)""",
                                            [h, old_category, self.user_id,
                                             f"superseded by {memory.memory_id}", now],
                                        )
                            except Exception as exc:
                                logger.debug(
                                    "Supersession tombstone write failed: %s", exc
                                )
                self.connection.execute(
                    """UPDATE memory_candidates
                       SET status = ?, updated_at = ?, reviewed_at = ?, review_reason = ?,
                           review_confidence = ?, review_model = ?,
                           durability = COALESCE(?, durability), scope = COALESCE(?, scope)
                       WHERE candidate_id = ?""",
                    [
                        final_status, now, now, reason or "", review_confidence,
                        review_model or "", durability, scope, candidate_id,
                    ],
                )

                # Rejection ledger (#39): record the rejected claim slot so no
                # approval path may resurrect it without a NEW record passing
                # the gates. One-way trust ladder — approve never launders.
                if final_status == "rejected":
                    self.record_rejection(
                        candidate["category"],
                        candidate.get("payload"),
                        reason=(reason or "review_rejected")[:200],
                    )

                # Wave 2: carry candidate evidence into memory_evidence so every
                # approved memory keeps provenance ("why does Hermes believe
                # this, and from exactly which user statement?").
                # Retention modes: full | hash | none.
                if memory is not None and evidence_retention != "none":
                    evidence_text = candidate.get("evidence_text") or ""
                    if evidence_retention == "hash" and evidence_text:
                        evidence_text = hashlib.sha256(
                            evidence_text.encode("utf-8")
                        ).hexdigest()
                    payload = candidate.get("payload") or {}
                    self.connection.execute(
                        """INSERT INTO memory_evidence
                           (memory_id, user_scope, source_session_id, source_timestamp,
                            evidence_role, evidence_text, extraction_method,
                            reviewer_decision, created_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                           ON CONFLICT (memory_id) DO NOTHING""",
                        [
                            memory.memory_id,
                            (
                                payload.get("user_scope")
                                if isinstance(payload, dict) else None
                            ) or self.user_id,
                            (
                                payload.get("source_session_id")
                                if isinstance(payload, dict) else None
                            ) or candidate.get("session_id") or "",
                            candidate.get("source_timestamp") or now,
                            candidate.get("evidence_role") or "",
                            evidence_text,
                            (
                                payload.get("extraction_method", "")
                                if isinstance(payload, dict) else ""
                            ),
                            final_status,
                            now,
                        ],
                    )
                self.connection.execute("COMMIT")
            except Exception:
                self.connection.execute("ROLLBACK")
                raise
        result = {
            "candidate": self.list_candidates(candidate_id=candidate_id, limit=1)[0],
            "memory": memory.to_dict() if memory else None,
        }
        if supersedes_memory_id:
            result["supersedes_memory_id"] = supersedes_memory_id
            result["superseded"] = superseded_ok
        return result

    def resolve_conflict(
        self,
        candidate_id: str,
        outcome: str,
        *,
        reason: str = "",
        reconciliation_content: str = "",
        review_source: str = "manual",
    ) -> dict | None:
        """Resolve a conflict with an explicit decision (#41).

        Conflict detection currently ends at pending confirmation — detection
        without a decision surface. This method adds five explicit resolution
        outcomes so a contradiction ends in a decision, including "both are
        true":

        - ``keep_old``: reject the new candidate; the existing active memory
          stays as-is. The candidate is marked rejected.
        - ``keep_new``: approve the new candidate and supersede the old
          memory. Equivalent to approve-with-supersede.
        - ``keep_both``: approve the new candidate WITHOUT superseding the
          old memory. Both records are retained as active and marked
          non-conflicting via payload metadata.
        - ``remove_both``: reject the candidate AND supersede/remove the old
          memory. Both values are recorded in the tombstone table so
          re-extraction stays blocked.
        - ``manual``: approve a human-written reconciliation content as the
          new memory, supersede the old, and reject the original candidate.
          Requires non-empty ``reconciliation_content`` (validator-level).

        Args:
            candidate_id: The pending candidate that conflicts with an
                          existing memory.
            outcome: One of keep_old, keep_new, keep_both, remove_both, manual.
            reason: Optional reason for the resolution.
            reconciliation_content: Required for ``manual`` outcome — the
                                     human-authored reconciliation text.
            review_source: Who is making the decision (manual/tool/auto).

        Returns:
            A dict with the candidate, memory (if any), and outcome details.
        """
        outcome = outcome.strip().lower()
        allowed_outcomes = {
            "keep_old", "keep_new", "keep_both", "remove_both", "manual",
        }
        if outcome not in allowed_outcomes:
            raise ValueError(
                f"invalid conflict resolution outcome: {outcome}. "
                f"Must be one of: {', '.join(sorted(allowed_outcomes))}"
            )
        # Validator-level: manual requires non-empty reconciliation content.
        if outcome == "manual":
            if not reconciliation_content or not reconciliation_content.strip():
                raise ValueError(
                    "manual conflict resolution requires non-empty "
                    "reconciliation_content (human-written reconciliation)"
                )
        candidates = self.list_candidates(candidate_id=candidate_id, limit=1)
        if not candidates:
            return None
        candidate = candidates[0]
        if candidate["status"] not in {"pending", "pending_user_confirmation"}:
            return {"candidate": candidate, "outcome": outcome, "memory": None}
        # Find the conflicting memory from the candidate's payload.
        payload = candidate.get("payload") or {}
        supersession_info = payload.get("value_supersession") or {}
        old_memory_id = supersession_info.get("supersedes_memory_id")
        now = self._now()
        memory = None
        with self._lock:
            assert self.connection is not None
            self.connection.execute("BEGIN TRANSACTION")
            try:
                if outcome == "keep_old":
                    # Reject the new candidate; old memory stays as-is.
                    self.connection.execute(
                        """UPDATE memory_candidates
                           SET status = 'rejected', updated_at = ?, reviewed_at = ?,
                               review_reason = ?
                           WHERE candidate_id = ?""",
                        [now, now,
                         f"conflict_resolved:keep_old ({reason})".strip(),
                         candidate_id],
                    )
                    self.record_rejection(
                        candidate["category"], payload,
                        reason=f"conflict_resolved:keep_old ({reason})"[:200],
                    )

                elif outcome == "keep_new":
                    # Approve the new candidate and supersede the old memory.
                    memory = self.remember(
                        category=candidate["category"],
                        content=candidate["content"],
                        tags=candidate["tags"],
                        payload=payload,
                        source=candidate["source"],
                        confidence=candidate["confidence"],
                        durability=candidate["durability"],
                        scope=candidate["scope"],
                        project_id=candidate["project_id"],
                        namespace=candidate.get("namespace", "conversation"),
                        client_scope=candidate.get("client_scope"),
                        doc_class=candidate.get("doc_class"),
                        provenance_origin=candidate.get("provenance_origin"),
                        grounding=candidate.get("grounding"),
                    )
                    # #79: when remember() returns None (dedup/tombstone/
                    # rejection-ledger block), the candidate must NOT be
                    # marked 'approved' with no memory behind it. Mirror
                    # review_candidate's 'deduplicated' status.
                    keep_new_status = "approved"
                    if memory is None:
                        keep_new_status = "deduplicated"
                    if memory and old_memory_id:
                        # #78: guard the supersede with valid_to IS NULL and
                        # user_scope — the old_memory_id comes from the
                        # candidate's payload and may be stale (already
                        # superseded) or point at another tenant's record.
                        # Skip the supersede (and report it) when the target
                        # is no longer current or out of scope.
                        check = self.connection.execute(
                            """SELECT 1 FROM memory_records
                               WHERE memory_id = ?
                                 AND (user_scope IS NULL OR user_scope = ?)
                                 AND valid_to IS NULL""",
                            [old_memory_id, self.user_id],
                        ).fetchone()
                        if check:
                            self.connection.execute(
                                """UPDATE memory_records
                                   SET valid_to = ?, superseded_by = ?, updated_at = ?
                                   WHERE memory_id = ?
                                     AND valid_to IS NULL
                                     AND (user_scope IS NULL OR user_scope = ?)""",
                                [now, memory.memory_id, now, old_memory_id,
                                 self.user_id],
                            )
                            # Tombstone the old value (#36 re-assertion block).
                            try:
                                old_row = self.connection.execute(
                                    """SELECT content, category FROM memory_records
                                       WHERE memory_id = ?""",
                                    [old_memory_id],
                                ).fetchone()
                                if old_row:
                                    h = self._tombstone_hash(old_row[0])
                                    if h and old_row[1]:
                                        self.connection.execute(
                                            """INSERT OR REPLACE INTO deletion_tombstones
                                               (content_hash, category, user_scope, reason, created_at)
                                               VALUES (?, ?, ?, ?, ?)""",
                                            [h, old_row[1], self.user_id,
                                             f"superseded by {memory.memory_id}", now],
                                        )
                            except Exception as exc:
                                logger.debug("Tombstone write failed: %s", exc)
                        else:
                            logger.warning(
                                "resolve_conflict keep_new: old_memory_id %s "
                                "is not current/in-scope — supersede skipped",
                                old_memory_id,
                            )
                    self.connection.execute(
                        """UPDATE memory_candidates
                           SET status = ?, updated_at = ?, reviewed_at = ?,
                               review_reason = ?
                           WHERE candidate_id = ?""",
                        [keep_new_status, now, now,
                         f"conflict_resolved:keep_new ({reason})".strip(),
                         candidate_id],
                    )

                elif outcome == "keep_both":
                    # Approve the new candidate WITHOUT superseding the old.
                    # Both records are retained as active. Mark them as
                    # non-conflicting via payload metadata.
                    both_payload = dict(payload)
                    both_payload["conflict_resolved"] = "keep_both"
                    both_payload["conflict_partner"] = old_memory_id
                    memory = self.remember(
                        category=candidate["category"],
                        content=candidate["content"],
                        tags=candidate["tags"],
                        payload=both_payload,
                        source=candidate["source"],
                        confidence=candidate["confidence"],
                        durability=candidate["durability"],
                        scope=candidate["scope"],
                        project_id=candidate["project_id"],
                        namespace=candidate.get("namespace", "conversation"),
                        client_scope=candidate.get("client_scope"),
                        doc_class=candidate.get("doc_class"),
                        provenance_origin=candidate.get("provenance_origin"),
                        grounding=candidate.get("grounding"),
                    )
                    # Mark the old memory's payload as non-conflicting too.
                    # D8: add valid_to IS NULL + user_scope guards (#78-style)
                    # — the old_memory_id comes from the candidate's payload
                    # and may be stale (already superseded) or point at another
                    # tenant's record. Without these guards this is the only
                    # resolve_conflict path that can write to a superseded or
                    # cross-tenant record.
                    if old_memory_id:
                        try:
                            old_payload_row = self.connection.execute(
                                """SELECT payload FROM memory_records
                                   WHERE memory_id = ?
                                     AND valid_to IS NULL
                                     AND (user_scope IS NULL OR user_scope = ?)""",
                                [old_memory_id, self.user_id],
                            ).fetchone()
                            if old_payload_row and old_payload_row[0]:
                                old_payload = json.loads(old_payload_row[0])
                                old_payload["conflict_resolved"] = "keep_both"
                                old_payload["conflict_partner"] = (
                                    memory.memory_id if memory else None
                                )
                                self.connection.execute(
                                    """UPDATE memory_records
                                       SET payload = ?, updated_at = ?
                                       WHERE memory_id = ?
                                         AND valid_to IS NULL
                                         AND (user_scope IS NULL OR user_scope = ?)""",
                                    [json.dumps(old_payload), now, old_memory_id,
                                     self.user_id],
                                )
                            else:
                                logger.warning(
                                    "resolve_conflict keep_both: old_memory_id "
                                    "%s is not current/in-scope — payload not "
                                    "modified",
                                    old_memory_id,
                                )
                        except Exception as exc:
                            logger.debug("keep_both old payload update failed: %s", exc)
                    # B7: mirror keep_new's #79 fix — when remember() returns
                    # None (dedup/tombstone/rejection block), the candidate
                    # must NOT be marked 'approved' with no memory behind it.
                    keep_both_status = "approved" if memory is not None else "deduplicated"
                    self.connection.execute(
                        """UPDATE memory_candidates
                           SET status = ?, updated_at = ?, reviewed_at = ?,
                               review_reason = ?
                           WHERE candidate_id = ?""",
                        [keep_both_status, now, now,
                         f"conflict_resolved:keep_both ({reason})".strip(),
                         candidate_id],
                    )

                elif outcome == "remove_both":
                    # Reject the candidate AND remove the old memory.
                    # Both values are tombstoned so re-extraction stays blocked.
                    self.connection.execute(
                        """UPDATE memory_candidates
                           SET status = 'rejected', updated_at = ?, reviewed_at = ?,
                               review_reason = ?
                           WHERE candidate_id = ?""",
                        [now, now,
                         f"conflict_resolved:remove_both ({reason})".strip(),
                         candidate_id],
                    )
                    self.record_rejection(
                        candidate["category"], payload,
                        reason=f"conflict_resolved:remove_both ({reason})"[:200],
                    )
                    if old_memory_id:
                        # Supersede the old memory (set valid_to).
                        # #78: add user_scope guard (valid_to IS NULL was
                        # already present).
                        self.connection.execute(
                            """UPDATE memory_records
                               SET valid_to = ?, updated_at = ?
                               WHERE memory_id = ? AND valid_to IS NULL
                                 AND (user_scope IS NULL OR user_scope = ?)""",
                            [now, now, old_memory_id, self.user_id],
                        )
                        # Tombstone the old value.
                        try:
                            old_row = self.connection.execute(
                                """SELECT content, category FROM memory_records
                                   WHERE memory_id = ?""",
                                [old_memory_id],
                            ).fetchone()
                            if old_row:
                                h = self._tombstone_hash(old_row[0])
                                if h and old_row[1]:
                                    self.connection.execute(
                                        """INSERT OR REPLACE INTO deletion_tombstones
                                           (content_hash, category, user_scope, reason, created_at)
                                           VALUES (?, ?, ?, ?, ?)""",
                                        [h, old_row[1], self.user_id,
                                         "conflict_resolved:remove_both", now],
                                    )
                        except Exception as exc:
                            logger.debug("remove_both tombstone failed: %s", exc)

                elif outcome == "manual":
                    # Approve the human-written reconciliation as the new
                    # memory, supersede the old, reject the original candidate.
                    reconciled_payload = dict(payload)
                    reconciled_payload["conflict_resolved"] = "manual"
                    reconciled_payload["reconciliation"] = True
                    memory = self.remember(
                        category=candidate["category"],
                        content=reconciliation_content,
                        tags=candidate["tags"],
                        payload=reconciled_payload,
                        source="manual_reconciliation",
                        confidence=1.0,
                        durability=candidate["durability"],
                        scope=candidate["scope"],
                        project_id=candidate["project_id"],
                        namespace=candidate.get("namespace", "conversation"),
                        client_scope=candidate.get("client_scope"),
                        doc_class=candidate.get("doc_class"),
                        provenance_origin=candidate.get("provenance_origin"),
                        grounding=candidate.get("grounding"),
                    )
                    if memory and old_memory_id:
                        # #78: guard the supersede (same as keep_new).
                        check = self.connection.execute(
                            """SELECT 1 FROM memory_records
                               WHERE memory_id = ?
                                 AND (user_scope IS NULL OR user_scope = ?)
                                 AND valid_to IS NULL""",
                            [old_memory_id, self.user_id],
                        ).fetchone()
                        if check:
                            self.connection.execute(
                                """UPDATE memory_records
                                   SET valid_to = ?, superseded_by = ?, updated_at = ?
                                   WHERE memory_id = ?
                                     AND valid_to IS NULL
                                     AND (user_scope IS NULL OR user_scope = ?)""",
                                [now, memory.memory_id, now, old_memory_id,
                                 self.user_id],
                            )
                            # Tombstone the old value.
                            try:
                                old_row = self.connection.execute(
                                    """SELECT content, category FROM memory_records
                                       WHERE memory_id = ?""",
                                    [old_memory_id],
                                ).fetchone()
                                if old_row:
                                    h = self._tombstone_hash(old_row[0])
                                    if h and old_row[1]:
                                        self.connection.execute(
                                            """INSERT OR REPLACE INTO deletion_tombstones
                                               (content_hash, category, user_scope, reason, created_at)
                                               VALUES (?, ?, ?, ?, ?)""",
                                            [h, old_row[1], self.user_id,
                                             f"superseded by {memory.memory_id} (manual reconciliation)", now],
                                        )
                            except Exception as exc:
                                logger.debug("manual tombstone failed: %s", exc)
                        else:
                            logger.warning(
                                "resolve_conflict manual: old_memory_id %s "
                                "is not current/in-scope — supersede skipped",
                                old_memory_id,
                            )
                    # B8: when remember() returns None (dedup/tombstone/
                    # rejection block), the reconciliation failed to create
                    # a memory. Do NOT reject the candidate or record a
                    # rejection-ledger entry — that would permanently burn
                    # the candidate and block paraphrased retries via the
                    # one-way ladder. Instead, mark it pending_user_confirmation
                    # so the user can retry with different reconciliation
                    # content.
                    if memory is None:
                        self.connection.execute(
                            """UPDATE memory_candidates
                               SET status = 'pending_user_confirmation',
                                   updated_at = ?, reviewed_at = ?,
                                   review_reason = ?
                               WHERE candidate_id = ?""",
                            [now, now,
                             (f"conflict_resolved:manual failed — "
                              f"reconciliation content was deduped/blocked; "
                              f"retry with different content ({reason})").strip(),
                             candidate_id],
                        )
                    else:
                        # Reconciliation succeeded — reject the original
                        # candidate (it's been replaced by the reconciliation).
                        self.connection.execute(
                            """UPDATE memory_candidates
                               SET status = 'rejected', updated_at = ?, reviewed_at = ?,
                                   review_reason = ?
                               WHERE candidate_id = ?""",
                            [now, now,
                             f"conflict_resolved:manual ({reason})".strip(),
                             candidate_id],
                        )
                        self.record_rejection(
                            candidate["category"], payload,
                            reason=f"conflict_resolved:manual ({reason})"[:200],
                        )

                self.connection.execute("COMMIT")
            except Exception:
                self.connection.execute("ROLLBACK")
                raise
        result = {
            "candidate": self.list_candidates(candidate_id=candidate_id, limit=1)[0],
            "outcome": outcome,
            "memory": memory.to_dict() if memory else None,
        }
        if old_memory_id:
            result["conflict_memory_id"] = old_memory_id
        return result

    def find_conflict_pairs(
        self,
        *,
        recent_days: int = 7,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Find candidate conflict pairs for review (#41).

        Bounded-scan rule: only report conflicts involving at least one
        recent memory (O(today x history); never old-vs-old). This prevents
        the queue from re-surfacing settled pairs nightly and becoming noise
        nobody reads.

        Returns a list of conflict dicts, each with:
        - candidate_id, candidate_content, candidate_created_at
        - conflict_memory_id, conflict_content
        - new_value, old_value (from value_supersession payload)
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=recent_days)).isoformat()
        with self._lock:
            assert self.connection is not None
            # Candidates with a value_supersession payload, created within
            # the recent window, still pending or pending_user_confirmation.
            rows = self.connection.execute(
                """SELECT candidate_id, content, payload, created_at
                   FROM memory_candidates
                   WHERE status IN ('pending', 'pending_user_confirmation')
                     AND (user_scope IS NULL OR user_scope = ?)
                     AND created_at >= ?
                     AND json_extract_string(payload, '$.value_supersession.supersedes_memory_id') IS NOT NULL
                   ORDER BY created_at DESC
                   LIMIT ?""",
                [self.user_id, cutoff, max(1, min(int(limit), 500))],
            ).fetchall()
        conflicts = []
        for candidate_id, content, payload_raw, created_at in rows:
            try:
                payload = json.loads(payload_raw) if payload_raw else {}
            except (json.JSONDecodeError, TypeError):
                continue
            sup = payload.get("value_supersession") or {}
            old_id = sup.get("supersedes_memory_id")
            if not old_id:
                continue
            # Skip same-id pairs (should never happen, but guard).
            conflicts.append({
                "candidate_id": candidate_id,
                "candidate_content": content,
                "candidate_created_at": created_at,
                "conflict_memory_id": old_id,
                "conflict_content": sup.get("old_content", ""),
                "new_value": sup.get("new_value", ""),
                "old_value": sup.get("old_value", ""),
            })
        return conflicts

    def find_supersede_candidates(
        self, candidate_id: str, limit: int = 3,
    ) -> List[Dict[str, Any]]:
        """Surface current memories similar to a candidate's content.

        Used by the reviewer path to offer an approve-with-supersede option
        when a candidate is a replacement/contradiction of an existing
        current fact (e.g. residence, employer, medication). Returns current
        (valid_to IS NULL) memories ranked by raw similarity, each with its
        chain length so the reviewer can see what would be chained behind.
        Confirm-first: this only SURFACES options; it never writes.
        """
        candidates = self.list_candidates(candidate_id=candidate_id, limit=1)
        if not candidates:
            return []
        content = candidates[0].get("content") or ""
        if not content:
            return []
        # suppress_retrieval: internal search must not inflate counters.
        hits = self.search(
            content, limit=limit, suppress_retrieval=True,
        )
        out: List[Dict[str, Any]] = []
        for hit in hits:
            if hit.valid_to is not None:
                continue  # only current records are supersede targets
            raw = getattr(hit, "raw_similarity", None)
            if raw is None:
                raw = hit.similarity
            out.append({
                "memory_id": hit.memory_id,
                "content": hit.content,
                "category": hit.category,
                "raw_similarity": round(float(raw), 4),
                "valid_from": hit.valid_from,
                "valid_to": hit.valid_to,
            })
        return out

    def get_evidence(self, memory_id: str) -> dict | None:
        """Return the provenance record for a memory, or None."""
        with self._lock:
            assert self.connection is not None
            row = self.connection.execute(
                """SELECT memory_id, source_session_id, source_timestamp,
                          evidence_role, evidence_text, extraction_method,
                          reviewer_decision, created_at
                   FROM memory_evidence
                   WHERE memory_id = ? AND (user_scope IS NULL OR user_scope = ?)""",
                [memory_id, self.user_id],
            ).fetchone()
            if not row:
                return None
            return {
                "memory_id": row[0],
                "source_session_id": row[1],
                "source_timestamp": row[2],
                "evidence_role": row[3],
                "evidence_text": row[4],
                "extraction_method": row[5],
                "reviewer_decision": row[6],
                "created_at": row[7],
            }

    def backfill_evidence(self, retention: str = "full") -> int:
        """Copy evidence from approved candidates into memory_evidence.

        One-time migration for memories approved before Wave 2 (which carries
        evidence only for NEW approvals).  Matches candidates to their memory
        by content equality + same-era source timestamp; skips candidates that
        already have an evidence row.  Retention: full | hash | none.
        Returns the number of evidence rows written.

        Pass 2 (fallback): candidates whose original memory was deleted (the
        fact lives on in a semantically-identical replacement) are attached to
        the best-matching active memory when raw similarity is strong.
        """
        # Pass 2 runs OUTSIDE the lock: self.search() acquires the same
        # non-reentrant lock, so calling it inside `with self._lock` deadlocks.
        orphan_rows = None
        with self._lock:
            assert self.connection is not None
            rows = self.connection.execute(
                """SELECT c.candidate_id, c.evidence_text, c.evidence_role,
                          c.source_timestamp, c.session_id, c.created_at,
                          c.payload, m.memory_id
                   FROM memory_candidates c
                   JOIN memory_records m ON m.content = c.content
                   WHERE c.status = 'approved'
                     AND c.evidence_text IS NOT NULL AND c.evidence_text != ''
                     AND m.status = 'active'
                     AND NOT EXISTS (
                         SELECT 1 FROM memory_evidence e
                         WHERE e.memory_id = m.memory_id
                     )"""
            ).fetchall()
            written = 0
            for (cand_id, evidence_text, evidence_role, source_ts,
                 session_id, created_at, payload, memory_id) in rows:
                written += self._write_evidence_row(
                    memory_id, cand_id, evidence_text, evidence_role,
                    source_ts, session_id, created_at, payload, retention,
                )

            # Pass 2: semantic fallback for candidates whose original memory
            # was deleted — attach to the strongest surviving match.
            orphan_rows = self.connection.execute(
                """SELECT candidate_id, content, evidence_text, evidence_role,
                          source_timestamp, session_id, created_at, payload
                   FROM memory_candidates
                   WHERE status = 'approved'
                     AND evidence_text IS NOT NULL AND evidence_text != ''
                     AND NOT EXISTS (
                         SELECT 1 FROM memory_records m
                         WHERE m.content = memory_candidates.content
                           AND m.status = 'active'
                     )
                     AND NOT EXISTS (
                         SELECT 1 FROM memory_evidence e
                         WHERE e.candidate_id = memory_candidates.candidate_id
                     )"""
            ).fetchall()

        for (cand_id, content, evidence_text, evidence_role, source_ts,
             session_id, created_at, payload) in (orphan_rows or []):
            try:
                best = self.search(
                    content or "", limit=3, suppress_retrieval=True,
                )
            except Exception:
                best = []
            if not best:
                continue
            top = best[0]
            raw = getattr(top, "raw_similarity", None)
            if raw is None:
                raw = top.similarity if hasattr(top, "similarity") else 0.0
            if raw < 0.5:
                continue  # not confident enough — leave orphaned
            with self._lock:
                inserted = self._write_evidence_row(
                    top.memory_id, cand_id, evidence_text, evidence_role,
                    source_ts, session_id, created_at, payload, retention,
                )
                if inserted:
                    written += 1
                else:
                    # Row exists (conflict) — link it if written pre-candidate_id.
                    try:
                        cur = self.connection.cursor()
                        cur.execute(
                            "UPDATE memory_evidence SET candidate_id = ? "
                            "WHERE memory_id = ? AND candidate_id IS NULL",
                            [cand_id, top.memory_id],
                        )
                        cur.close()
                    except Exception:
                        pass
        return written

    def _write_evidence_row(
        self, memory_id, cand_id, evidence_text, evidence_role, source_ts,
        session_id, created_at, payload, retention: str,
    ) -> int:
        """Insert one evidence row (shared by exact + fallback passes)."""
        text = evidence_text or ""
        if retention == "hash" and text:
            text = hashlib.sha256(text.encode("utf-8")).hexdigest()
        payload_dict = payload if isinstance(payload, dict) else {}
        # DuckDB reports rowcount=-1 for INSERT ... ON CONFLICT, so we
        # pre-check existence: the NOT EXISTS guards in backfill_evidence
        # already dedupe; this makes the return value truthful.
        existing = self.connection.execute(
            "SELECT 1 FROM memory_evidence WHERE memory_id = ?",
            [memory_id],
        ).fetchone()
        if existing:
            return 0
        cur = self.connection.cursor()
        cur.execute(
            """INSERT INTO memory_evidence
               (memory_id, user_scope, source_session_id, source_timestamp,
                evidence_role, evidence_text, extraction_method,
                reviewer_decision, created_at, candidate_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT (memory_id) DO NOTHING""",
            [
                memory_id,
                payload_dict.get("user_scope") or self.user_id,
                payload_dict.get("source_session_id") or session_id or "",
                source_ts or created_at,
                evidence_role or "",
                text,
                payload_dict.get("extraction_method", ""),
                "backfilled",
                self._now(),
                cand_id,
            ],
        )
        cur.close()
        return 1  # pre-checked: no existing row, so the insert succeeded

    def quarantine_memory(self, memory_id: str, reason: str) -> bool:
        """Hide a memory from retrieval without deleting its record."""
        now = self._now()
        with self._lock:
            assert self.connection is not None
            check = self.connection.execute(
                """SELECT COUNT(*) FROM memory_records
                   WHERE memory_id = ?
                     AND (user_scope IS NULL OR user_scope = ?)""",
                [memory_id, self.user_id],
            ).fetchone()
            if not check or check[0] == 0:
                return False
            self.connection.execute(
                """UPDATE memory_records
                   SET status = 'quarantined', quarantine_reason = ?, quarantined_at = ?, updated_at = ?
                   WHERE memory_id = ?""",
                [reason or "manual review", now, now, memory_id],
            )
        return True

    def restore_memory(self, memory_id: str) -> bool:
        """Restore a quarantined memory to active retrieval.

        B9: if the record is a non-head version (valid_to is set — e.g. a
        middle version quarantined by delete_memory), restoring it to
        status='active' creates an inconsistent state: the retrieval filter
        (valid_to IS NULL) still hides it, but the status field says active.
        Log a warning so the inconsistency is visible to diagnostics
        (memory_why_not, memory_tombstones) rather than silent.
        """
        now = self._now()
        with self._lock:
            assert self.connection is not None
            # B9: fetch valid_to alongside the existence check so we can
            # warn about non-head restores.
            row = self.connection.execute(
                """SELECT valid_to FROM memory_records
                   WHERE memory_id = ?
                     AND (user_scope IS NULL OR user_scope = ?)""",
                [memory_id, self.user_id],
            ).fetchone()
            if not row:
                return False
            if row[0] is not None:
                logger.warning(
                    "restore_memory: %s is a non-head version (valid_to is "
                    "set) — restoring to status='active' but the retrieval "
                    "filter (valid_to IS NULL) will still hide it; promote "
                    "to head or clear valid_to to make it retrievable",
                    memory_id,
                )
            self.connection.execute(
                """UPDATE memory_records
                   SET status = 'active', quarantine_reason = NULL,
                       quarantined_at = NULL, updated_at = ?
                   WHERE memory_id = ?""",
                [now, memory_id],
            )
        return True

    def record_feedback(self, memory_id: str, feedback: str) -> bool:
        """Record helpful/dismissed/incorrect feedback; incorrect hides the memory."""
        feedback = feedback.strip().lower()
        if feedback not in {"helpful", "dismissed", "incorrect"}:
            raise ValueError("feedback must be helpful, dismissed, or incorrect")
        now = self._now()
        with self._lock:
            assert self.connection is not None
            exists = self.connection.execute(
                """SELECT COUNT(*) FROM memory_records
                   WHERE memory_id = ?
                     AND (user_scope IS NULL OR user_scope = ?)""",
                [memory_id, self.user_id],
            ).fetchone()
            if not exists or exists[0] == 0:
                return False
            if feedback == "helpful":
                sql = "UPDATE memory_records SET helpful_count = COALESCE(helpful_count, 0) + 1, updated_at = ?, tier = 'active' WHERE memory_id = ?"
                params = [now, memory_id]
            elif feedback == "dismissed":
                sql = "UPDATE memory_records SET dismissed_count = COALESCE(dismissed_count, 0) + 1, updated_at = ?, tier = 'active' WHERE memory_id = ?"
                params = [now, memory_id]
            else:
                sql = """UPDATE memory_records
                         SET dismissed_count = COALESCE(dismissed_count, 0) + 1,
                             status = 'quarantined', quarantine_reason = 'marked incorrect',
                             quarantined_at = ?, updated_at = ?
                         WHERE memory_id = ?"""
                params = [now, now, memory_id]
            self.connection.execute(sql, params)
            return True

    def update_memory(
        self,
        memory_id: str,
        content: str | None = None,
        tags: List[str] | None = None,
        payload_updates: Dict[str, Any] | None = None,
        expires_at: Any = _NOT_PROVIDED,
        *,
        structural_guard: bool = False,
        created_at: Any = None,
    ) -> MemoryRecord | None:
        """Update an existing memory by creating a new version.

        Instead of overwriting the old record, this:
        1. Creates a new memory record with a new ID (the new version)
        2. Sets the old record's valid_to = now and superseded_by = new_id
        3. Returns the new version

        *expires_at* semantics (Spec 1):
        - ``_NOT_PROVIDED`` (default): carry the old expiry forward unchanged.
        - ``None``: clear the expiry (revive the memory).
        - ISO-8601 string: set to that value.

        *created_at* (issue #8 / #74): override the new version's creation
        timestamp. By default the wall clock is used. Pass an ISO-8601 string
        to backdate the new version to its in-world date (e.g. "married
        Helen" on 2022-06-01 superseding "dating Helen" from 2022-01-01).
        This sets both ``created_at`` and ``valid_from`` on the new version
        so version-chain/supersession logic and ``as_of`` temporal queries
        operate on in-world order, not ingest order. ``updated_at`` always
        gets the wall clock (the row was physically written now).

        The old record is preserved for history queries (as_of parameter).
        If no content/tags/payload changes are provided, returns the existing
        record unchanged.
        """
        if content is not None:
            content, _inj = sanitize_content(content)
            if _inj:
                raise ValueError(
                    f"Content blocked: updated text matches an instruction-injection "
                    f"pattern ({_inj}). Refusing to write."
                )
        existing = self._fetch_records(
            """SELECT * FROM memory_records
               WHERE memory_id = ?
                 AND (user_scope IS NULL OR user_scope = ?)""",
            [memory_id, self.user_id],
        )
        if not existing:
            return None
        rec = existing[0]
        # Chain-fork guard: if the caller passed a superseded (non-current)
        # version ID, resolve forward along superseded_by to the CURRENT head
        # before updating. Updating a stale version directly forks the chain:
        # both the live head and the new record would end up valid_to IS NULL
        # and be retrieved as two concurrent "current" versions of the same
        # fact (reproduced; get_memory_history then walks only the stale
        # branch). Cycle-guarded like get_memory_history so a corrupt
        # superseded_by loop terminates instead of spinning.
        head = rec
        visited: set[str] = {rec.memory_id}
        while head.valid_to is not None and head.superseded_by:
            nxt_id = head.superseded_by
            if nxt_id in visited:
                logger.warning(
                    "Cycle detected in memory chain at %s -> %s; updating the "
                    "stale version instead of walking (corrupt edge).",
                    head.memory_id, nxt_id,
                )
                break
            nxt = self._fetch_records(
                """SELECT * FROM memory_records WHERE memory_id = ?
                   AND (user_scope IS NULL OR user_scope = ?)""",
                [nxt_id, self.user_id],
            )
            if not nxt:
                break
            visited.add(nxt[0].memory_id)
            head = nxt[0]
        rec = head
        memory_id = rec.memory_id  # supersede the head, not the stale version
        now = self._now()
        new_content = content if content is not None else rec.content
        new_tags = tags if tags is not None else rec.tags
        new_payload = dict(rec.payload)
        if payload_updates:
            new_payload.update(payload_updates)

        # Structural-loss guard (#42): if content is being rewritten, count
        # what would be deleted and merge it back. The rewrite succeeds but
        # cannot destroy — lost sentences, list items, and KV pairs are
        # appended to the new content. Pure enrichment passes unchanged.
        # Outcome/decision-shaped records are append-only and exempt.
        # Opt-in via structural_guard=True — user-initiated updates (value
        # changes, corrections) don't need the guard; LLM rewrites do.
        loss_report: LossReport | None = None
        if (
            structural_guard
            and content is not None
            and not is_append_only(rec.category, rec.payload)
        ):
            try:
                new_content, loss_report = structural_loss_guard(rec.content, new_content)
                if not loss_report.is_clean():
                    logger.info(
                        "Structural-loss guard: merged back %d lost items "
                        "(sentences=%d, list_items=%d, kv_pairs=%d) for %s",
                        loss_report.total_lost,
                        len(loss_report.lost_sentences),
                        len(loss_report.lost_list_items),
                        len(loss_report.lost_kv_pairs),
                        memory_id,
                    )
                    # Record the loss report in the payload for audit.
                    new_payload["structural_loss_repair"] = {
                        "lost_total": loss_report.total_lost,
                        "category_counts": loss_report.category_counts(),
                    }
            except Exception as exc:
                logger.debug("Structural-loss guard failed: %s", exc)

        # Resolve effective expires_at:
        # _NOT_PROVIDED → carry forward; None → clear (revive); str → set.
        # #80: normalize string expires_at to the canonical aware-UTC form.
        if expires_at is _NOT_PROVIDED:
            effective_expires = rec.expires_at
        elif expires_at is None:
            effective_expires = None
        else:
            effective_expires = self._normalize_timestamp(expires_at)
            if effective_expires is None:
                logger.warning(
                    "Unparseable expires_at %r in update_memory — storing "
                    "as-is", expires_at,
                )
                effective_expires = str(expires_at)

        # created_at override (issue #8 / #74): backdate the new version to
        # its in-world date so valid_from and version-chain order reflect
        # when the event happened, not when it was ingested. Mirrors the
        # same logic in remember(). updated_at stays at the wall clock.
        if created_at is not None:
            created_ts = self._normalize_timestamp(created_at) or str(created_at)
        else:
            created_ts = now

        # If nothing actually changed, return the existing record
        if (content is None and tags is None and not payload_updates
                and expires_at is _NOT_PROVIDED):
            return rec

        # Re-embed if content changed
        new_emb: List[float] = []
        if content is not None and self.embedder and hasattr(self.embedder, "embed"):
            new_emb = self.embedder.embed(new_content)
        elif rec.embedding:
            new_emb = rec.embedding

        # Generate new version ID
        new_id = f"mem-{uuid.uuid4().hex}"

        with self._lock:
            assert self.connection is not None
            # B6 TOCTOU fix: the head was resolved outside the lock (above).
            # Between resolution and this transaction another thread may
            # have superseded it, leaving our resolved memory_id with
            # valid_to != NULL. Re-verify inside the lock and, if the head
            # moved, walk the chain to the *current* head before writing.
            # Without this, both the racer's new record and ours would end
            # up valid_to IS NULL — two concurrent "current" versions.
            head_check = self.connection.execute(
                """SELECT valid_to, superseded_by FROM memory_records
                   WHERE memory_id = ?""",
                [memory_id],
            ).fetchone()
            if head_check and head_check[0] is not None:
                # Head was superseded during the race window — re-resolve.
                visited_b6 = {memory_id}
                cur_id = memory_id
                while True:
                    row = self.connection.execute(
                        """SELECT memory_id, valid_to, superseded_by
                           FROM memory_records WHERE memory_id = ?""",
                        [cur_id],
                    ).fetchone()
                    if not row or row[1] is None:
                        break  # found the current head
                    nxt = row[2]
                    if not nxt or nxt in visited_b6:
                        break  # cycle / dead end
                    visited_b6.add(nxt)
                    cur_id = nxt
                if cur_id != memory_id:
                    logger.debug(
                        "B6 TOCTOU: head moved %s -> %s during update; "
                        "superseding the current head", memory_id, cur_id,
                    )
                    memory_id = cur_id
            # Wrap the version-chain write in an explicit transaction so a
            # crash between steps cannot leave two current versions or
            # orphan the evidence trail (issue #9).
            self.connection.execute("BEGIN TRANSACTION")
            try:
                # 1. Create the new version, carrying feedback counters forward
                #    from the superseded record so importance evidence survives
                #    edits. A memory with 10 helpful votes keeps them after a
                #    content fix. retrieval_count also carries forward because the
                #    new version represents the same fact the user has been
                #    retrieving.
                self.connection.execute(
                    """INSERT INTO memory_records
                       (memory_id, category, content, tags, payload, created_at, updated_at,
                        expires_at, embedding, status, source, confidence, durability, scope,
                        project_id, user_scope, namespace, client_scope, doc_class,
                        source_doc_id, source_loc, extraction_method, extracted_at,
                        verified_state, verified_at,
                        retrieval_count, helpful_count, dismissed_count,
                        valid_from, valid_to, superseded_by)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)""",
                    [new_id, rec.category, new_content, new_tags,
                     json.dumps(new_payload), created_ts, now,
                     effective_expires,
                     new_emb if new_emb else None,
                     rec.status, rec.source, rec.confidence, rec.durability, rec.scope,
                     rec.project_id,
                     rec.payload.get("user_scope"),
                     getattr(rec, "namespace", "conversation") or "conversation",
                     getattr(rec, "client_scope", None),
                     getattr(rec, "doc_class", None),
                     getattr(rec, "source_doc_id", None),
                     getattr(rec, "source_loc", None),
                     getattr(rec, "extraction_method", None),
                     getattr(rec, "extracted_at", None),
                     getattr(rec, "verified_state", "current") or "current",
                     getattr(rec, "verified_at", None),
                     rec.retrieval_count, rec.helpful_count, rec.dismissed_count,
                     created_ts],
                )
                # 2. Supersede the old version. D5 fix: guard with
                #    AND valid_to IS NULL so an already-superseded record's
                #    superseded_by is never overwritten (orphaning the racer).
                # B6 TOCTOU fix: the head may have been superseded between
                # resolution (outside the lock) and this transaction —
                # including during the INSERT above. If the original
                # memory_id is no longer current, walk the chain to the
                # actual current head and supersede that instead, so we
                # never leave two records with valid_to IS NULL.
                self.connection.execute(
                    """UPDATE memory_records
                       SET valid_to = ?, superseded_by = ?, updated_at = ?
                       WHERE memory_id = ? AND valid_to IS NULL""",
                    [now, new_id, now, memory_id],
                )
                # B6 TOCTOU fix: the head may have been superseded between
                # resolution (outside the lock) and this UPDATE — including
                # during our own INSERT above. If a racer created a new
                # current head, the UPDATE above targeted the stale record
                # (0 rows via the valid_to IS NULL guard). Walk the chain
                # from memory_id; if we find another record with
                # valid_to IS NULL that isn't our new_id, supersede it so
                # we collapse to a single current head.
                _cur = memory_id
                _seen = {memory_id, new_id}
                while True:
                    _row = self.connection.execute(
                        """SELECT superseded_by, valid_to FROM memory_records
                           WHERE memory_id = ?""",
                        [_cur],
                    ).fetchone()
                    if not _row:
                        break
                    _sup_by, _valid_to = _row
                    # If this record is current and isn't our new version,
                    # it's the racer's head — supersede it.
                    if _valid_to is None and _cur != memory_id:
                        self.connection.execute(
                            """UPDATE memory_records
                               SET valid_to = ?, superseded_by = ?, updated_at = ?
                               WHERE memory_id = ? AND valid_to IS NULL""",
                            [now, new_id, now, _cur],
                        )
                        break
                    if not _sup_by or _sup_by in _seen:
                        break
                    _seen.add(_sup_by)
                    _cur = _sup_by
                # 3. Carry the evidence trail forward. memory_evidence is keyed
                #    by memory_id, so the new version would otherwise orphan the
                #    provenance row (review point 2: provenance after updates).
                self.connection.execute(
                    """INSERT INTO memory_evidence
                       (memory_id, user_scope, source_session_id, source_timestamp,
                        evidence_role, evidence_text, extraction_method,
                        reviewer_decision, created_at, candidate_id)
                       SELECT ?, user_scope, source_session_id, source_timestamp,
                              evidence_role, evidence_text, extraction_method,
                              reviewer_decision, created_at, candidate_id
                       FROM memory_evidence WHERE memory_id = ?""",
                    [new_id, memory_id],
                )
                self.connection.execute("COMMIT")
            except Exception:
                self.connection.execute("ROLLBACK")
                raise
        fetched = self._fetch_records(
            "SELECT * FROM memory_records WHERE memory_id = ?", [new_id]
        )
        return fetched[0] if fetched else None

    def get_memory_history(
        self, memory_id: str, max_versions: int | None = None,
    ) -> List[MemoryRecord]:
        """Return the full version chain for a memory.

        Given any version's ID, walks the superseded_by chain to find
        all versions: the original, all intermediate versions, and the
        current version. Returns them in chronological order (oldest first).

        *max_versions* truncates the result to the most recent N versions
        (keeping the current/head version) when set.

        Cycle guard: a visited-set defends both directions against corrupt
        A→B→A cycles (logs loudly and stops the walk). Scope isolation:
        every hop enforces user_scope so a cross-scope hop cannot leak
        another user's chain.
        """
        # Walk forward from the given ID to find all successors
        chain: List[MemoryRecord] = []
        current = self._fetch_records(
            """SELECT * FROM memory_records WHERE memory_id = ?
              AND (user_scope IS NULL OR user_scope = ?)""",
            [memory_id, self.user_id],
        )
        if not current:
            return []
        chain.append(current[0])

        # Follow superseded_by forward (visited-set cycle guard + scope)
        visited: set[str] = {current[0].memory_id}
        node = current[0]
        while node.superseded_by:
            nxt_id = node.superseded_by
            if nxt_id in visited:
                logger.warning(
                    "Cycle detected in memory chain at %s → %s; stopping "
                    "forward walk (corrupt superseded_by edge).",
                    node.memory_id, nxt_id,
                )
                break
            nxt = self._fetch_records(
                """SELECT * FROM memory_records WHERE memory_id = ?
                  AND (user_scope IS NULL OR user_scope = ?)""",
                [nxt_id, self.user_id],
            )
            if not nxt or nxt[0].memory_id == node.memory_id:
                break
            visited.add(nxt[0].memory_id)
            chain.append(nxt[0])
            node = nxt[0]

        # Now walk backward to find predecessors (records that were
        # superseded by the oldest version in our chain). Scope-isolated
        # and cycle-guarded like the forward walk.
        oldest = chain[0]
        predecessors: List[MemoryRecord] = []
        prev_visited: set[str] = {oldest.memory_id}
        prev_search = self._fetch_records(
            """SELECT * FROM memory_records WHERE superseded_by = ?
              AND (user_scope IS NULL OR user_scope = ?)""",
            [oldest.memory_id, self.user_id],
        )
        while prev_search:
            prev = prev_search[0]
            if prev.memory_id in prev_visited:
                logger.warning(
                    "Cycle detected in memory chain (backward) at %s; "
                    "stopping backward walk (corrupt superseded_by edge).",
                    prev.memory_id,
                )
                break
            prev_visited.add(prev.memory_id)
            predecessors.append(prev)
            prev_search = self._fetch_records(
                """SELECT * FROM memory_records WHERE superseded_by = ?
                  AND (user_scope IS NULL OR user_scope = ?)""",
                [prev.memory_id, self.user_id],
            )
        # Prepend predecessors (oldest first)
        predecessors.reverse()
        full = predecessors + chain
        if max_versions is not None and max_versions > 0 and len(full) > max_versions:
            # Keep the most recent N versions (head/current always retained).
            return full[-max_versions:]
        return full

    def get_evidence_batch(
        self, memory_ids: List[str],
    ) -> Dict[str, dict]:
        """Return provenance rows for a batch of memory IDs in one query.

        Maps memory_id → evidence dict (same shape as get_evidence).
        Retention-aware: hash-mode rows already store a digest and none-mode
        rows store nothing, so callers never receive raw text for those.
        IDs with no evidence row are simply absent from the result.
        """
        if not memory_ids:
            return {}
        with self._lock:
            assert self.connection is not None
            placeholders = ", ".join(["?"] * len(memory_ids))
            result = self.connection.execute(
                f"""SELECT memory_id, source_session_id, source_timestamp,
                           evidence_role, evidence_text, extraction_method,
                           reviewer_decision, created_at
                    FROM memory_evidence
                    WHERE memory_id IN ({placeholders})
                      AND (user_scope IS NULL OR user_scope = ?)""",
                [*memory_ids, self.user_id],
            )
            out: Dict[str, dict] = {}
            for row in result.fetchall():
                out[row[0]] = {
                    "memory_id": row[0],
                    "source_session_id": row[1],
                    "source_timestamp": row[2],
                    "evidence_role": row[3],
                    "evidence_text": row[4],
                    "extraction_method": row[5],
                    "reviewer_decision": row[6],
                    "created_at": row[7],
                }
            return out

    def get_chain_membership(self, memory_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Batched chain-membership annotation for search results.

        Given a list of memory IDs (e.g. top-k search hits), returns a map
        memory_id → {"versions": N, "has_history": bool} where N is the
        total chain length for the chain that ID belongs to. A single
        batched query finds every chain that any hit participates in, then
        each chain is walked to count its versions. This is the trigger
        annotation that lets the agent know a fact has a history without
        unfolding it automatically.
        """
        if not memory_ids:
            return {}
        with self._lock:
            assert self.connection is not None
            placeholders = ", ".join(["?"] * len(memory_ids))
            # Find every record that either IS a hit or is superseded BY a
            # hit (i.e. a hit is the head of a chain) — plus every record
            # whose superseded_by points at a hit (hit is a predecessor).
            rows = self.connection.execute(
                f"""SELECT memory_id, superseded_by FROM memory_records
                    WHERE memory_id IN ({placeholders})
                       OR superseded_by IN ({placeholders})""",
                [*memory_ids, *memory_ids],
            ).fetchall()
        if not rows:
            return {mid: {"versions": 1, "has_history": False} for mid in memory_ids}
        # Build a quick lookup of superseded_by edges (scope-agnostic here;
        # the per-chain walk via get_memory_history enforces scope).
        supersede_map: Dict[str, str | None] = {
            row[0]: row[1] for row in rows
        }
        out: Dict[str, Dict[str, Any]] = {}
        for mid in memory_ids:
            if mid in out:
                continue
            if mid not in supersede_map:
                out[mid] = {"versions": 1, "has_history": False}
                continue
            # Walk the full chain for this hit to count versions. Reuse
            # get_memory_history (scope-isolated, cycle-guarded). Cache by
            # chain so two hits in the same chain share one walk.
            chain = self.get_memory_history(mid)
            n = len(chain)
            has = n > 1
            entry = {"versions": n, "has_history": has}
            # Annotate every other hit that belongs to this same chain.
            for rec in chain:
                if rec.memory_id in memory_ids:
                    out[rec.memory_id] = entry
            if mid not in out:
                out[mid] = entry
        # Any hit not yet annotated has no chain row at all.
        for mid in memory_ids:
            out.setdefault(mid, {"versions": 1, "has_history": False})
        return out

    def delete_memory(self, memory_id: str) -> bool | dict:
        """Delete a memory, chain-aware.

        - Head (current) deletion: promotes the predecessor to current
          (valid_to=NULL, superseded_by=NULL) — reversible supersession.
          Returns {"action": "promoted", "promoted_memory_id": ...}.
        - Non-head (historical) deletion: converts to quarantine instead of
          a hard delete, because deleting a middle version severs the causal
          arc. Returns {"action": "quarantined"}.
        - Single-version (no chain): hard delete.
          Returns {"action": "deleted"}.
        Returns False if the memory is not found.
        """
        with self._lock:
            assert self.connection is not None
            # Check existence first — DuckDB's execute() return value for
            # DELETE is not a reliable indicator of whether rows were deleted.
            # Also fetch content/category so hard-deleted facts can be
            # tombstoned (deletion must stay decisive against re-feeds).
            row = self.connection.execute(
                """SELECT content, category, valid_to FROM memory_records
                   WHERE memory_id = ?
                     AND (user_scope IS NULL OR user_scope = ?)""",
                [memory_id, self.user_id],
            ).fetchone()
            if not row:
                return False
            _del_content, _del_category = str(row[0] or ""), str(row[1] or "")
            # Chain position: is this the head (valid_to IS NULL)?
            is_head = row[2] is None
            now = self._now()
            if is_head:
                # Promote the predecessor (the record superseded BY this
                # head) to current. If there is no predecessor, hard-delete.
                pred = self.connection.execute(
                    """SELECT memory_id FROM memory_records
                       WHERE superseded_by = ?
                         AND (user_scope IS NULL OR user_scope = ?)""",
                    [memory_id, self.user_id],
                ).fetchone()
                if pred:
                    # Wrap the multi-statement promote path in a transaction
                    # (#77): a crash between the predecessor UPDATE and the
                    # head DELETE would leave two active versions of the same
                    # fact (valid_to IS NULL on both). Single-statement paths
                    # (quarantine, hard-delete) are already atomic.
                    self.connection.execute("BEGIN TRANSACTION")
                    try:
                        self.connection.execute(
                            """UPDATE memory_records
                               SET valid_to = NULL, superseded_by = NULL, updated_at = ?
                               WHERE memory_id = ?""",
                            [now, pred[0]],
                        )
                        # The deleted head's content is tombstoned too:
                        # deleting the current version of a fact must survive
                        # re-feeds, while the promoted predecessor keeps
                        # history intact.
                        self._record_tombstone(_del_content, _del_category)
                        self.connection.execute(
                            "DELETE FROM memory_records WHERE memory_id = ?"
                            " AND (user_scope IS NULL OR user_scope = ?)",
                            [memory_id, self.user_id],
                        )
                        self.connection.execute(
                            "DELETE FROM memory_evidence WHERE memory_id = ?"
                            " AND (user_scope IS NULL OR user_scope = ?)",
                            [memory_id, self.user_id],
                        )
                        self.connection.execute("COMMIT")
                    except Exception:
                        self.connection.execute("ROLLBACK")
                        raise
                    return {
                        "deleted": True, "action": "promoted",
                        "promoted_memory_id": pred[0],
                    }
            # Non-head: quarantine instead of severing the causal arc.
            if not is_head:
                self.connection.execute(
                    """UPDATE memory_records
                       SET status = 'quarantined',
                           quarantine_reason = 'deleted from chain (reversible)',
                           quarantined_at = ?, updated_at = ?
                       WHERE memory_id = ?""",
                    [now, now, memory_id],
                )
                return {"deleted": True, "action": "quarantined"}
            # Head with no predecessor: hard delete. Fingerprint the content
            # first so a later re-feed (source re-ingest, extractor replay)
            # cannot silently resurrect it — atlas deletion-canary step 6.
            # Wrap in a transaction (matching the promote path at #77): a
            # crash between the tombstone INSERT and the memory DELETE would
            # leave a tombstone for a record that still exists, blocking
            # re-creation via remember()'s tombstone check even though the
            # original is still active.
            self.connection.execute("BEGIN TRANSACTION")
            try:
                self._record_tombstone(_del_content, _del_category)
                self.connection.execute(
                    "DELETE FROM memory_records WHERE memory_id = ?"
                    " AND (user_scope IS NULL OR user_scope = ?)",
                    [memory_id, self.user_id],
                )
                self.connection.execute(
                    "DELETE FROM memory_evidence WHERE memory_id = ?"
                    " AND (user_scope IS NULL OR user_scope = ?)",
                    [memory_id, self.user_id],
                )
                self.connection.execute("COMMIT")
            except Exception:
                self.connection.execute("ROLLBACK")
                raise
            return {"deleted": True, "action": "deleted"}

    # -- deletion tombstones ---------------------------------------------------

    @staticmethod
    def _tombstone_hash(content: str) -> str:
        """Fingerprint of deleted content: case/whitespace-insensitive."""
        normalized = " ".join(str(content or "").lower().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _record_tombstone(self, content: str, category: str,
                          reason: str = "user_delete") -> None:
        """Fingerprint hard-deleted content so re-feeding it is blocked.

        MUST be called while holding self._lock (threading.Lock is not
        reentrant; both call sites live inside delete_memory's locked
        section). External tombstone writes go through delete_memory.
        """
        h = self._tombstone_hash(content)
        if not h or not category:
            return
        assert self.connection is not None
        self.connection.execute(
            """INSERT OR REPLACE INTO deletion_tombstones
               (content_hash, category, user_scope, reason, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            [h, category, self.user_id, reason, self._now()],
        )

    def tombstone_check(self, content: str, category: str) -> dict | None:
        """Return the tombstone row (as a dict) if this content was deleted."""
        with self._lock:
            assert self.connection is not None
            row = self.connection.execute(
                """SELECT reason, created_at FROM deletion_tombstones
                   WHERE content_hash = ? AND category = ?
                     AND user_scope = ?""",
                [self._tombstone_hash(content), category, self.user_id],
            ).fetchone()
        if not row:
            return None
        return {"reason": row[0], "created_at": row[1]}

    def purge_tombstone(self, content: str, category: str) -> bool:
        """Explicitly allow a previously-deleted fact back into memory.

        The user-facing escape hatch: 'delete' stays decisive until the
        user says otherwise. Returns True if a tombstone was removed.
        """
        h = self._tombstone_hash(content)
        with self._lock:
            assert self.connection is not None
            # RETURNING yields the deleted rows so we can distinguish a
            # real purge from a no-op. DuckDB's cursor.rowcount is always
            # -1 for DELETE, so we can't use the rowcount approach; the
            # old code's follow-up COUNT(*) == 0 returned True even when
            # nothing existed to purge (a no-op masquerading as success).
            cursor = self.connection.execute(
                """DELETE FROM deletion_tombstones
                   WHERE content_hash = ? AND category = ? AND user_scope = ?
                   RETURNING content_hash""",
                [h, category, self.user_id],
            )
            deleted = cursor.fetchone()
        return deleted is not None

    def list_tombstones(self, limit: int = 200) -> List[Dict[str, Any]]:
        """Read-only census of deletion tombstones, newest first.

        Visibility for the tombstone mechanism: shows what is blocked from
        re-creation without exposing raw content (hash + metadata only).
        Scope-filtered (user_scope = current scope) so one tenant never
        sees another tenant's tombstones (#49 isolation surface).
        """
        limit = max(1, min(int(limit), 1000))
        with self._lock:
            assert self.connection is not None
            rows = self.connection.execute(
                """SELECT content_hash, category, user_scope, reason, created_at
                   FROM deletion_tombstones
                   WHERE (user_scope IS NULL OR user_scope = ?)
                   ORDER BY created_at DESC
                   LIMIT ?""",
                [self.user_id, limit],
            ).fetchall()
        return [
            {
                "content_hash": r[0],
                "category": r[1],
                "user_scope": r[2],
                "reason": r[3],
                "created_at": r[4],
            }
            for r in rows
        ]

    # -- rejection ledger (one-way trust ladder, #39) -------------------------

    def rejection_check(self, category: str, payload: Dict[str, Any] | None) -> dict | None:
        """Return the ledger row (as a dict) if this claim slot was rejected.

        Keyed by (subject, predicate, scope) — the claim slot, not the exact
        value — so paraphrased re-assertions of a rejected fact are also
        blocked. ``payload`` is the candidate/record payload (or None).
        """
        key = rejection_key({"category": category, "payload": payload or {},
                             "user_scope": self.user_id})
        if not key[0] or not key[1]:
            return None
        with self._lock:
            assert self.connection is not None
            row = self.connection.execute(
                """SELECT reason, created_at FROM rejection_ledger
                   WHERE subject = ? AND predicate = ? AND user_scope = ?""",
                [key[0], key[1], key[2]],
            ).fetchone()
        if not row:
            return None
        return {"reason": row[0], "created_at": row[1]}

    def record_rejection(self, category: str, payload: Dict[str, Any] | None,
                         reason: str = "review_rejected") -> None:
        """Fingerprint a rejected claim slot so it cannot be resurrected (#39).

        MUST be called while holding self._lock (call sites live inside the
        locked review section). External rejection writes go through
        review_candidate.
        """
        key = rejection_key({"category": category, "payload": payload or {},
                             "user_scope": self.user_id})
        if not key[0] or not key[1]:
            return
        assert self.connection is not None
        self.connection.execute(
            """INSERT OR REPLACE INTO rejection_ledger
               (subject, predicate, user_scope, reason, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            [key[0], key[1], key[2], reason, self._now()],
        )

    def purge_rejection(self, category: str, payload: Dict[str, Any] | None) -> bool:
        """Explicitly allow a previously-rejected claim slot back in (#39).

        The escape hatch: rejection stays decisive until the user says
        otherwise. Returns True if a ledger entry was removed.
        """
        key = rejection_key({"category": category, "payload": payload or {},
                             "user_scope": self.user_id})
        if not key[0] or not key[1]:
            return False
        with self._lock:
            assert self.connection is not None
            self.connection.execute(
                """DELETE FROM rejection_ledger
                   WHERE subject = ? AND predicate = ? AND user_scope = ?""",
                [key[0], key[1], key[2]],
            )
            check = self.connection.execute(
                """SELECT COUNT(*) FROM rejection_ledger
                   WHERE subject = ? AND predicate = ? AND user_scope = ?""",
                [key[0], key[1], key[2]],
            ).fetchone()
        return bool(check and check[0] == 0)

    def list_rejections(self, limit: int = 200) -> List[Dict[str, Any]]:
        """Read-only census of the rejection ledger, newest first (#39)."""
        limit = max(1, min(int(limit), 1000))
        with self._lock:
            assert self.connection is not None
            rows = self.connection.execute(
                """SELECT subject, predicate, user_scope, reason, created_at
                   FROM rejection_ledger
                   ORDER BY created_at DESC
                   LIMIT ?""",
                [limit],
            ).fetchall()
        return [
            {
                "subject": r[0],
                "predicate": r[1],
                "user_scope": r[2],
                "reason": r[3],
                "created_at": r[4],
            }
            for r in rows
        ]
