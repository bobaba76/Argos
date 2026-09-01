"""Core store mixin: connection, schema init and scope helpers.

Extracted verbatim from store.py during the god-file split (behavior-
neutral: no renames, no fixes). Composed into DuckDBMemoryStore via MRO.
"""
from __future__ import annotations

import logging
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, Optional

try:
    from .store_common import _DEFAULT_TTL_DAYS
except ImportError:  # store_core.py imported as a top-level module
    from store_common import _DEFAULT_TTL_DAYS

import duckdb

logger = logging.getLogger(__name__)


class StoreCoreMixin:
    """Connection, schema and scope helpers for DuckDBMemoryStore."""

    def __init__(
        self,
        db_path: str | Path,
        user_id: str = "default_user",
        embedder=None,
        reranker=None,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.user_id = (user_id or "default_user").strip()
        self.embedder = embedder
        self.reranker = reranker
        self._lock = threading.RLock()
        self.connection: Optional[duckdb.DuckDBPyConnection] = None
        self._connect()
        self._init_db()
        # Scale-trigger metrics (measured gate for ANN/BM25 per the roadmap):
        # rolling p95 latency window + record count, warn when thresholds cross.
        self._scale_warn_latency_ms = 300.0
        self._scale_warn_records = 5000
        self._scale_window = 50
        self._scale_latencies: Deque[float] = deque(maxlen=self._scale_window)
        self._scale_queries = 0
        self._scale_warnings_fired = 0
        self._scale_last_count_check = 0
        self._scale_record_count: Optional[int] = None
        # Expiry (Spec 1): configurable TTL tiers. When expiry_enabled is
        # False, the hardcoded _DEFAULT_TTL_DAYS is used (current behavior).
        # When True, the provider sets ttl_days from config and the tool
        # surface can pass durability/expires_at explicitly.
        self.expiry_enabled: bool = False
        self.ttl_days: Dict[str, int] = dict(_DEFAULT_TTL_DAYS)
        self.expiry_default_days: int = 90
        # External-source write policy (set from config by the provider and
        # the shared service). When True, candidates tagged external_source
        # can never auto-activate: the storage boundary downgrades auto_review
        # approvals to pending_user_confirmation. Default ON (out-of-the-box
        # installs enforce the human-confirmation boundary).
        self.external_sources_require_confirmation: bool = True
        # Alias cache: avoids a full-table scan on every search query
        # (issue #27). Invalidated on add_alias / remove_alias.
        self._alias_cache: dict[str, list[tuple[str, str]]] | None = None

    # -- connection management ------------------------------------------------

    @staticmethod
    def _is_lock_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return "being used by another process" in msg or "cannot access the file" in msg

    def _connect(self) -> None:
        try:
            self.connection = duckdb.connect(str(self.db_path))
        except Exception as exc:
            if self._is_lock_error(exc):
                logger.warning("DuckDB locked by another process; opening read-only")
                self.connection = duckdb.connect(str(self.db_path), read_only=True)
            else:
                raise

    def _init_db(self) -> None:
        assert self.connection is not None
        with self._lock:
            self.connection.execute("""
                CREATE SEQUENCE IF NOT EXISTS seq_memory_id;
                CREATE TABLE IF NOT EXISTS memory_records (
                    memory_id          VARCHAR PRIMARY KEY,
                    category           VARCHAR,
                    content            VARCHAR,
                    tags               VARCHAR[],
                    payload            JSON,
                    created_at         VARCHAR,
                    updated_at         VARCHAR,
                    expires_at         VARCHAR,
                    embedding          DOUBLE[],
                    status             VARCHAR DEFAULT 'active',
                    source             VARCHAR DEFAULT 'explicit',
                    confidence         DOUBLE DEFAULT 1.0,
                    durability         VARCHAR DEFAULT 'durable',
                    scope              VARCHAR DEFAULT 'profile',
                    project_id         VARCHAR,
                    user_scope         VARCHAR,
                    retrieval_count    INTEGER DEFAULT 0,
                    last_retrieved_at  VARCHAR,
                    helpful_count      INTEGER DEFAULT 0,
                    dismissed_count    INTEGER DEFAULT 0,
                    quarantine_reason  VARCHAR,
                    quarantined_at     VARCHAR,
                    valid_from         VARCHAR,
                    valid_to           VARCHAR,
                    superseded_by      VARCHAR,
                    provenance_origin  VARCHAR DEFAULT 'internal',
                    grounding          VARCHAR DEFAULT 'observed'
                );
                CREATE TABLE IF NOT EXISTS memory_candidates (
                    candidate_id       VARCHAR PRIMARY KEY,
                    category           VARCHAR,
                    content            VARCHAR,
                    tags               VARCHAR[],
                    payload            JSON,
                    source             VARCHAR,
                    confidence         DOUBLE,
                    durability         VARCHAR,
                    scope              VARCHAR,
                    project_id         VARCHAR,
                    session_id         VARCHAR,
                    status             VARCHAR DEFAULT 'pending',
                    created_at         VARCHAR,
                    updated_at         VARCHAR,
                    reviewed_at        VARCHAR,
                    review_reason      VARCHAR,
                    evidence_text     VARCHAR,
                    evidence_role     VARCHAR DEFAULT 'user_turn',
                    source_timestamp   VARCHAR,
                    review_confidence  DOUBLE,
                    review_model       VARCHAR,
                    quarantine_reason  VARCHAR,
                    quarantined_at     VARCHAR,
                    provenance_origin  VARCHAR DEFAULT 'internal',
                    grounding          VARCHAR DEFAULT 'extracted'
                );
                CREATE TABLE IF NOT EXISTS entity_aliases (
                    alias              VARCHAR,
                    canonical_entity   VARCHAR,
                    user_scope         VARCHAR DEFAULT 'default_user',
                    created_at         VARCHAR,
                    PRIMARY KEY (alias, canonical_entity, user_scope)
                );
            """)

            # Evidence/provenance: links an active memory back to the exact
            # user statement that produced it (Wave 2).
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS memory_evidence (
                    memory_id          VARCHAR PRIMARY KEY,
                    user_scope         VARCHAR,
                    source_session_id  VARCHAR,
                    source_timestamp   VARCHAR,
                    evidence_role      VARCHAR,
                    evidence_text      VARCHAR,
                    extraction_method  VARCHAR,
                    reviewer_decision  VARCHAR,
                    created_at         VARCHAR,
                    candidate_id       VARCHAR
                )
            """)
            # Deletion tombstones (2026-08-24): fingerprint of hard-deleted
            # content so a later re-feed cannot silently resurrect it.
            # Closes atlas deletion-canary step 6 (observational test
            # test_deletion_step_5_6_refeed_resurrection in
            # test_contradiction_matrix.py confirmed the resurrection).
            # Reversible via purge_tombstone(); scoped per user_scope.
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS deletion_tombstones (
                    content_hash   VARCHAR,
                    category       VARCHAR,
                    user_scope     VARCHAR,
                    reason         VARCHAR DEFAULT 'user_delete',
                    created_at     VARCHAR,
                    PRIMARY KEY (content_hash, category, user_scope)
                )
            """)
            # Rejection ledger (#39, batch-2): one-way trust ladder. Records the
            # (subject, predicate, scope) identity of a REJECTED value so no
            # approval path (auto-review, user confirmation, restore) may
            # resurrect it without a NEW record passing the same gates. Distinct
            # from deletion_tombstones (which key on exact content hash): this
            # keys on the claim slot, so paraphrased re-assertions are blocked.
            # Reversible via purge_rejection().
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS rejection_ledger (
                    subject       VARCHAR,
                    predicate     VARCHAR,
                    user_scope    VARCHAR,
                    reason        VARCHAR DEFAULT 'review_rejected',
                    created_at    VARCHAR,
                    PRIMARY KEY (subject, predicate, user_scope)
                )
            """)
            # Additive migration for databases created by earlier versions.
            columns = {
                "status": "VARCHAR DEFAULT 'active'",
                "source": "VARCHAR DEFAULT 'explicit'",
                "confidence": "DOUBLE DEFAULT 1.0",
                "durability": "VARCHAR DEFAULT 'durable'",
                "scope": "VARCHAR DEFAULT 'profile'",
                "project_id": "VARCHAR",
                "user_scope": "VARCHAR",
                "retrieval_count": "INTEGER DEFAULT 0",
                "last_retrieved_at": "VARCHAR",
                "helpful_count": "INTEGER DEFAULT 0",
                "dismissed_count": "INTEGER DEFAULT 0",
                "quarantine_reason": "VARCHAR",
                "quarantined_at": "VARCHAR",
                "valid_from": "VARCHAR",
                "valid_to": "VARCHAR",
                "superseded_by": "VARCHAR",
                "provenance_origin": "VARCHAR DEFAULT 'internal'",
                "grounding": "VARCHAR DEFAULT 'observed'",
            }
            candidate_columns = {
                "user_scope": "VARCHAR",
                "evidence_text": "VARCHAR",
                "evidence_role": "VARCHAR DEFAULT 'user_turn'",
                "source_timestamp": "VARCHAR",
                "review_confidence": "DOUBLE",
                "review_model": "VARCHAR",
                "quarantine_reason": "VARCHAR",
                "quarantined_at": "VARCHAR",
                "provenance_origin": "VARCHAR DEFAULT 'internal'",
                "grounding": "VARCHAR DEFAULT 'extracted'",
            }
            for name, definition in columns.items():
                try:
                    self.connection.execute(
                        f"ALTER TABLE memory_records ADD COLUMN IF NOT EXISTS {name} {definition}"
                    )
                except Exception as exc:
                    logger.warning("Memory schema migration for %s failed: %s", name, exc)
            for name, definition in candidate_columns.items():
                try:
                    self.connection.execute(
                        f"ALTER TABLE memory_candidates ADD COLUMN IF NOT EXISTS {name} {definition}"
                    )
                except Exception as exc:
                    logger.warning("Candidate schema migration for %s failed: %s", name, exc)

            try:
                self.connection.execute(
                    "ALTER TABLE memory_evidence ADD COLUMN IF NOT EXISTS candidate_id VARCHAR"
                )
            except Exception as exc:
                logger.warning("Evidence schema migration (candidate_id) failed: %s", exc)

            try:
                # One-time: link pre-existing evidence rows (written before the
                # column existed) back to their source candidate.
                self.connection.execute("""
                    UPDATE memory_evidence e
                    SET candidate_id = c.candidate_id
                    FROM memory_candidates c
                    WHERE e.candidate_id IS NULL
                      AND e.source_session_id = c.session_id
                      AND e.source_timestamp = c.source_timestamp
                """)
            except Exception as exc:
                logger.warning("Evidence candidate_id backfill failed: %s", exc)

            try:
                self.connection.execute("""
                    UPDATE memory_records
                    SET source = COALESCE(
                            NULLIF(json_extract_string(payload, '$.source'), ''),
                            NULLIF(source, ''), 'explicit'
                        ),
                        status = COALESCE(NULLIF(status, ''), 'active'),
                        durability = CASE
                            WHEN category IN ('context_note', 'event') THEN 'temporary'
                            ELSE COALESCE(NULLIF(durability, ''), 'durable')
                        END,
                        scope = COALESCE(NULLIF(scope, ''), 'profile'),
                        confidence = CASE
                            WHEN json_extract_string(payload, '$.source') = 'llm_extraction'
                                 AND (confidence IS NULL OR confidence >= 0.99) THEN 0.45
                            ELSE COALESCE(confidence, 0.75)
                        END,
                        retrieval_count = COALESCE(retrieval_count, 0),
                        helpful_count = COALESCE(helpful_count, 0),
                        dismissed_count = COALESCE(dismissed_count, 0)
                """)
            except Exception as exc:
                logger.warning("Memory metadata backfill failed: %s", exc)

            # Retroactive temporal-validity migration: every existing memory
            # gets valid_from = created_at. valid_to stays NULL (current).
            # This runs once on databases that predate the versioning feature.
            try:
                self.connection.execute("""
                    UPDATE memory_records
                    SET valid_from = COALESCE(valid_from, created_at)
                    WHERE valid_from IS NULL
                """)
            except Exception as exc:
                logger.warning("Temporal validity backfill failed: %s", exc)

            # user_scope column backfill (NULL stays NULL = legacy global scope).
            try:
                self.connection.execute("""
                    UPDATE memory_records
                    SET user_scope = json_extract_string(payload, '$.user_scope')
                    WHERE user_scope IS NULL
                      AND json_extract_string(payload, '$.user_scope') IS NOT NULL
                """)
            except Exception as exc:
                logger.warning("user_scope backfill failed: %s", exc)
            try:
                self.connection.execute("""
                    UPDATE memory_candidates
                    SET user_scope = json_extract_string(payload, '$.user_scope')
                    WHERE user_scope IS NULL
                      AND json_extract_string(payload, '$.user_scope') IS NOT NULL
                """)
            except Exception as exc:
                logger.warning("candidate user_scope backfill failed: %s", exc)

            # Provenance-origin backfill (#43): derive the taint label from the
            # payload's external_source flag for pre-existing records. Fails
            # closed — anything not clearly internal becomes external.
            try:
                self.connection.execute("""
                    UPDATE memory_records
                    SET provenance_origin = CASE
                        WHEN provenance_origin IN ('internal', 'external')
                            THEN provenance_origin
                        WHEN json_extract_string(payload, '$.external_source') = 'true'
                             OR json_extract_string(payload, '$.external_source') = '1'
                            THEN 'external'
                        ELSE 'internal'
                    END
                """)
            except Exception as exc:
                logger.warning("provenance_origin backfill failed: %s", exc)
            try:
                self.connection.execute("""
                    UPDATE memory_candidates
                    SET provenance_origin = CASE
                        WHEN provenance_origin IN ('internal', 'external')
                            THEN provenance_origin
                        WHEN json_extract_string(payload, '$.external_source') = 'true'
                             OR json_extract_string(payload, '$.external_source') = '1'
                            THEN 'external'
                        ELSE 'internal'
                    END
                """)
            except Exception as exc:
                logger.warning("candidate provenance_origin backfill failed: %s", exc)

            # Grounding backfill (#40): derive from the write-path source for
            # pre-existing records. Distill-derived and external-origin records
            # ground as inferred; llm_extraction as extracted; explicit/user as
            # observed. Anything unresolved stays speculative (strictest).
            try:
                self.connection.execute("""
                    UPDATE memory_records
                    SET grounding = CASE
                        WHEN grounding IN ('speculative','inferred','extracted','observed')
                            THEN grounding
                        WHEN source IN ('explicit', 'user', 'manual') THEN 'observed'
                        WHEN source IN ('llm_extraction', 'extraction') THEN 'extracted'
                        WHEN source IN ('distillation', 'distill') THEN 'inferred'
                        WHEN provenance_origin = 'external' THEN 'inferred'
                        ELSE 'speculative'
                    END
                """)
            except Exception as exc:
                logger.warning("grounding backfill failed: %s", exc)
            try:
                self.connection.execute("""
                    UPDATE memory_candidates
                    SET grounding = CASE
                        WHEN grounding IN ('speculative','inferred','extracted','observed')
                            THEN grounding
                        WHEN source IN ('explicit', 'user', 'manual') THEN 'observed'
                        WHEN source IN ('llm_extraction', 'extraction') THEN 'extracted'
                        WHEN source IN ('distillation', 'distill') THEN 'inferred'
                        WHEN provenance_origin = 'external' THEN 'inferred'
                        ELSE 'extracted'
                    END
                """)
            except Exception as exc:
                logger.warning("candidate grounding backfill failed: %s", exc)
            # Composite index: scope → status → validity.
            try:
                self.connection.execute("""
                    CREATE INDEX IF NOT EXISTS idx_memory_scope_status_valid
                    ON memory_records (user_scope, status, valid_to)
                """)
            except Exception as exc:
                logger.warning("user_scope index creation failed: %s", exc)

            # System state KV table (P4.2 distillation run state, future
            # maintenance passes). Zero migration — CREATE IF NOT EXISTS.
            try:
                self.connection.execute("""
                    CREATE TABLE IF NOT EXISTS system_state (
                        key    VARCHAR PRIMARY KEY,
                        value  VARCHAR
                    )
                """)
            except Exception as exc:
                logger.warning("system_state table creation failed: %s", exc)

    # -- helpers --------------------------------------------------------------

    def set_user_scope(self, user_id: str | None) -> None:
        self.user_id = (user_id or "default_user").strip()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _normalize_timestamp(ts: str | None) -> str | None:
        """Normalize an ISO-8601 timestamp to the aware UTC form used by
        ``_now()`` (``datetime.now(timezone.utc).isoformat()``).

        Handles ``Z`` suffixes, naive timestamps (assumed UTC), and date-only
        strings (``2020-01-15`` → ``2020-01-15T00:00:00+00:00``). Returns
        ``None`` if the input is missing or unparseable. This is the write-
        boundary normalization for #80: storing all timestamps in one
        canonical form prevents lexicographic VARCHAR comparison mismatches
        across mixed ISO formats.
        """
        if not ts:
            return None
        try:
            parsed = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.isoformat()
        except (ValueError, TypeError, AttributeError):
            return None

    @staticmethod
    def _is_expired(expires_at: str | None, at: str | None = None) -> bool:
        if not expires_at:
            return False
        try:
            exp = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            if exp.tzinfo is None:
                exp = exp.replace(tzinfo=timezone.utc)
            if at:
                ref = datetime.fromisoformat(at.replace("Z", "+00:00"))
                if ref.tzinfo is None:
                    ref = ref.replace(tzinfo=timezone.utc)
            else:
                ref = datetime.now(timezone.utc)
            return exp <= ref
        except (ValueError, TypeError, AttributeError) as exc:
            # #80: fail loud — a date-only or unparseable expires_at must
            # not be silently treated as "never expires". Log the warning so
            # the broken value is visible; return False only because there is
            # no valid expiry to enforce (the record is treated as non-
            # expiring, but the operator can see the bad data in logs).
            logger.warning(
                "Unparseable expires_at %r (at=%r): %s — treating as no "
                "expiry. Normalize timestamps at the write boundary.",
                expires_at, at, exc,
            )
            return False

    def _matches_scope(self, payload: dict) -> bool:
        scope = payload.get("user_scope")
        if scope is None:
            return True  # global memory
        return scope == self.user_id

