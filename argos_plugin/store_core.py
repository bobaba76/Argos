"""Core store mixin: connection, schema init and scope helpers.

Extracted verbatim from store.py during the god-file split (behavior-
neutral: no renames, no fixes). Composed into DuckDBMemoryStore via MRO.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

try:
    from .store_common import _DEFAULT_TTL_DAYS, restrict_store_artifact
    from .store_state import StoreMixinState
except ImportError:  # store_core.py imported as a top-level module
    from store_common import _DEFAULT_TTL_DAYS, restrict_store_artifact
    from store_state import StoreMixinState

# #330: audit paths stay fail-soft but must not be silent — a failure is
# logged at ERROR and recorded on the liveness health surface.
# Import liveness at the top level (consistent with api_facade.py's
# top-level access_scoping import) so one canonical module drives the
# health singleton across every import layout — package vs top-level.
from liveness import record_subsystem_failure, record_subsystem_ok

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
        memory_limit: str = "512MB",
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.user_id = (user_id or "default_user").strip()
        self.embedder = embedder
        self.reranker = reranker
        # DuckDB resource budget (desktop-hostile defaults fixed 13/9): cap
        # the buffer pool and spill to disk instead of letting DuckDB cache
        # up to ~80% of RAM. Any caller may override before construction.
        self._memory_limit: str = memory_limit
        self._temp_directory: Path = Path(db_path).parent / ".duckdb-tmp"
        # #249-slice: shared state lives in one documented dataclass.
        self._state = StoreMixinState()
        self.connection: Optional[duckdb.DuckDBPyConnection] = None
        self._connect()
        self._init_db()
        # #414: restrict the DuckDB store artifacts to owner-only on POSIX.
        # Runs every init so recreated .wal files are covered each run, not
        # only on first init. Fail-soft (mirrors SC2): chmod is a no-op on
        # Windows and never blocks init.
        restrict_store_artifact(self.db_path, 0o600)
        restrict_store_artifact(Path(str(self.db_path) + ".wal"), 0o600)
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
        self._state.alias_cache = None
        # #347: Actor context for mutation_events. Defaults to the store's
        # user_id and "human" for local paths. The facade sets this from
        # ctx.principal + ctx.principal_type before mutations; the RPC
        # service sets it from the resolved user_id and auth mode. Never
        # a client-passed string — server-derived identity only.
        self._actor: str = self.user_id
        self._actor_type: str = "human"

    # -- connection management ------------------------------------------------

    @staticmethod
    def _is_lock_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return "being used by another process" in msg or "cannot access the file" in msg

    def _connect(self) -> None:
        self._state.read_only = False
        try:
            self.connection = duckdb.connect(
                str(self.db_path),
                config={
                    # Explicit buffer-pool cap instead of DuckDB's default
                    # (~80% of RAM as cache): pages spill to temp_directory
                    # over budget instead of ballooning the process RSS.
                    "memory_limit": self._memory_limit,
                    "temp_directory": str(self._temp_directory),
                },
            )
        except Exception as exc:
            if self._is_lock_error(exc):
                # SC1: track read-only state and log ERROR (not WARNING) —
                # writes will fail with confusing DuckDB errors without this.
                logger.error(
                    "DuckDB locked by another process; opening read-only. "
                    "All write operations (remember, save_candidate, etc.) "
                    "will fail until the lock is released."
                )
                self._state.read_only = True
                self.connection = duckdb.connect(
                    str(self.db_path),
                    read_only=True,
                    config={
                        "memory_limit": self._memory_limit,
                        "temp_directory": str(self._temp_directory),
                    },
                )
            else:
                raise

    def is_read_only(self) -> bool:
        """SC1: returns True if the store was opened in read-only fallback mode."""
        return self._state.read_only

    def _init_db(self) -> None:
        assert self.connection is not None
        with self._state.lock:
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
                    grounding          VARCHAR DEFAULT 'observed',
                    namespace          VARCHAR DEFAULT 'conversation',
                    client_scope       VARCHAR,
                    doc_class          VARCHAR,
                    -- Spec-07 (#71) D4: document-sourced fact provenance.
                    source_doc_id      VARCHAR,
                    source_loc         VARCHAR,
                    extraction_method  VARCHAR,
                    extracted_at       VARCHAR,
                    verified_state     VARCHAR DEFAULT 'current',
                    verified_at        VARCHAR,
                    -- P5.1 (#6): memory lifecycle tier. 'active' (default,
                    -- in injection pool) or 'archived' (out of injection
                    -- pool, searchable via include_archived=True). Zero-
                    -- migration: existing rows default 'active'.
                    tier               VARCHAR DEFAULT 'active',
                    -- #286: dimension-generic vector provenance. embedding_dim
                    -- records the dimension of the stored vector (so the query
                    -- path can detect mixed-dim stores). embedder_id records
                    -- which model produced the vector. embedded_at records
                    -- when the vector was (re-)embedded. NULL on legacy rows.
                    embedding_dim      INTEGER,
                    embedder_id        VARCHAR,
                    embedded_at        VARCHAR,
                    -- #392: record class marker. NULL (default, normal user
                    -- memory) or 'system_internal' (implementation/config/
                    -- tuning notes about the memory system itself). Used by
                    -- the distillation input filter to exclude engine-room
                    -- noise from clustering. NOT overloaded with `tier`
                    -- (lifecycle status) or `namespace` (partitioning). Zero-
                    -- migration: existing rows default NULL (eligible).
                    record_class       VARCHAR,
                    -- Spec-13 (#393): provider write-policy trust class.
                    -- NULL = normal/clean memory. 'unreviewed' = auto-saved
                    -- under approval_mode=auto with medium/high risk signals:
                    -- retrievable but rank-penalized (<= 3 positions) and
                    -- resolvable only by explicit human action (never
                    -- auto-promoted). Zero-migration: existing rows default
                    -- NULL (pre-policy records are untouched by the flip).
                    trust_class        VARCHAR
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
                    namespace          VARCHAR DEFAULT 'conversation',
                    client_scope       VARCHAR,
                    doc_class          VARCHAR,
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
                CREATE TABLE IF NOT EXISTS file_catalog (
                    file_id              VARCHAR PRIMARY KEY,
                    canonical_path       VARCHAR,
                    size                 BIGINT,
                    mtime                VARCHAR,
                    first_seen           VARCHAR,
                    last_seen            VARCHAR,
                    status               VARCHAR DEFAULT 'active',
                    client_scope         VARCHAR,
                    doc_class            VARCHAR,
                    doc_type             VARCHAR,
                    one_line_description VARCHAR,
                    description_method   VARCHAR DEFAULT 'heuristic',
                    hot_flags            VARCHAR,
                    hot_reason           VARCHAR,
                    extract_hash         VARCHAR,
                    extracted_at         VARCHAR,
                    last_touch           VARCHAR,
                    touch_count          INTEGER DEFAULT 0,
                    pinned               BOOLEAN DEFAULT FALSE,
                    -- Spec-09 (#112): form-level identity. Deterministic
                    -- structural fingerprint (page count, table regions,
                    -- column signatures, heading structure) — distinct from
                    -- file_id (content) and extract_hash (extraction input).
                    -- NULL until the catalog pass computes it; never LLM.
                    layout_family        VARCHAR
                );
                CREATE TABLE IF NOT EXISTS file_aliases (
                    file_id    VARCHAR,
                    path       VARCHAR,
                    first_seen VARCHAR,
                    PRIMARY KEY (file_id, path)
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
                # Spec-05 (#67): doc-fact namespace + client scope. Additive,
                # mirrors the project_id pattern. namespace defaults to
                # 'conversation' so legacy rows are conversation-sourced; NULL
                # client_scope = global (visible inside any client query).
                "namespace": "VARCHAR DEFAULT 'conversation'",
                "client_scope": "VARCHAR",
                # Spec-06 (#69): document class for access scoping.
                # NULL = no class (legacy/conversation records). Reserved
                # value 'practice-internal' = principals-only.
                "doc_class": "VARCHAR",
                # Spec-07 (#71) D4: document-sourced fact provenance.
                # source_doc_id = file_id (hash, never a path). verified_state
                # defaults to 'current' so legacy rows are unaffected.
                "source_doc_id": "VARCHAR",
                "source_loc": "VARCHAR",
                "extraction_method": "VARCHAR",
                "extracted_at": "VARCHAR",
                "verified_state": "VARCHAR DEFAULT 'current'",
                "verified_at": "VARCHAR",
                # P5.1 (#6): memory lifecycle tier. Missing from the legacy
                # migration map on the first batch-F release — fresh DBs got it
                # from CREATE TABLE but pre-existing stores never did, so every
                # search binder-errored on COALESCE(tier,...). Additive ALTER.
                "tier": "VARCHAR DEFAULT 'active'",
                # #286: dimension-generic vector provenance. Additive ALTER
                # for pre-existing stores. NULL on legacy rows (the query
                # path handles NULL embedding_dim by falling back to text
                # search or computing dim from the stored vector).
                "embedding_dim": "INTEGER",
                "embedder_id": "VARCHAR",
                "embedded_at": "VARCHAR",
                # #392: record class marker. NULL = normal user memory,
                # 'system_internal' = engine-room noise excluded from
                # distillation input. Additive ALTER for pre-existing stores.
                "record_class": "VARCHAR",
                # Spec-13 (#393): write-policy trust class for memory_records.
                "trust_class": "VARCHAR",
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
                "namespace": "VARCHAR DEFAULT 'conversation'",
                "client_scope": "VARCHAR",
                "doc_class": "VARCHAR",
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

            # SC6: wrap backfill UPDATEs in a transaction for atomicity.
            # The backfills are idempotent (WHERE ... IS NULL), but a
            # transaction makes the atomicity explicit — a crash mid-backfill
            # rolls back the partial state instead of leaving it inconsistent.
            # SC6: skip entirely in read-only mode (UPDATEs would fail).
            if not self._state.read_only:
                _started = self._begin_transaction_if_needed()
                try:
                    self.connection.execute("""
                        UPDATE memory_evidence e
                        SET candidate_id = c.candidate_id
                        FROM memory_candidates c
                        WHERE e.candidate_id IS NULL
                          AND e.source_session_id = c.session_id
                          AND e.source_timestamp = c.source_timestamp
                    """)
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
                    self.connection.execute("""
                        UPDATE memory_records
                        SET valid_from = COALESCE(valid_from, created_at)
                        WHERE valid_from IS NULL
                    """)
                    self.connection.execute("""
                        UPDATE memory_records
                        SET user_scope = json_extract_string(payload, '$.user_scope')
                        WHERE user_scope IS NULL
                          AND json_extract_string(payload, '$.user_scope') IS NOT NULL
                    """)
                    self.connection.execute("""
                        UPDATE memory_candidates
                        SET user_scope = json_extract_string(payload, '$.user_scope')
                        WHERE user_scope IS NULL
                          AND json_extract_string(payload, '$.user_scope') IS NOT NULL
                    """)
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
                    self._commit_if_started(_started)
                except Exception as exc:
                    try:
                        self._rollback_if_started(_started)
                    except Exception:
                        pass
                    logger.warning("Backfill transaction failed (rolled back): %s", exc)
            # Composite index: scope → status → validity.
            try:
                self.connection.execute("""
                    CREATE INDEX IF NOT EXISTS idx_memory_scope_status_valid
                    ON memory_records (user_scope, status, valid_to)
                """)
            except Exception as exc:
                logger.warning("user_scope index creation failed: %s", exc)

            # Spec-06 (#69): access scoping index on client_scope + doc_class
            # for the retrieval pre-filter. Additive, no behaviour change
            # when the columns are NULL (legacy rows).
            try:
                self.connection.execute("""
                    CREATE INDEX IF NOT EXISTS idx_memory_client_scope_doc_class
                    ON memory_records (client_scope, doc_class)
                """)
            except Exception as exc:
                logger.warning("client_scope/doc_class index creation failed: %s", exc)

            # Spec-06 (#69): append-only access audit log. Every query and
            # every deny writes a row. Exportable via the service API.
            # Principals-only read; rotates on a configurable window.
            # SC2: rotation is now implemented via _purge_access_audit.
            try:
                self.connection.execute("""
                    CREATE TABLE IF NOT EXISTS access_audit (
                        audit_id      VARCHAR PRIMARY KEY,
                        ts            VARCHAR,
                        tenant        VARCHAR,
                        user_id       VARCHAR,
                        query_text    VARCHAR,
                        granted_count INTEGER DEFAULT 0,
                        denied_count  INTEGER DEFAULT 0,
                        denied_scopes VARCHAR,
                        excluded      BOOLEAN DEFAULT FALSE
                    )
                """)
            except Exception as exc:
                logger.error(
                    "access_audit table creation failed — the audit trail is "
                    "not being recorded: %s", exc, exc_info=True,
                )
                record_subsystem_failure("audit_write", exc)

            # #347: Append-only, actor-attributed mutation event log.
            # Every store mutation (create, approve, reject, update, delete,
            # tombstone, rejection, purge, erase, import) writes one row in
            # the SAME transaction as the mutation itself. Unlike access_audit
            # (which rotates at 100k and logs reads/denials only),
            # mutation_events NEVER rotates — no startup purge, no cap. A
            # capped provenance log cannot backfill; the commercial
            # "showable history" claim dies at the cap. Growth is bounded by
            # mutation volume (not query volume) and export-and-archive is
            # the only offload path.
            #
            # DDL is shared with _migration_4_to_5 via
            # _create_mutation_events_ddl to avoid schema drift (round-2
            # review: schema DDL was duplicated between _init_db and the
            # migration).
            try:
                from .schema_migrations import _create_mutation_events_ddl
                _create_mutation_events_ddl(self.connection)
            except ImportError:
                # store_core.py imported as a top-level module
                from schema_migrations import _create_mutation_events_ddl
                _create_mutation_events_ddl(self.connection)
            except Exception as exc:
                logger.warning("mutation_events table creation failed: %s", exc)

            # SC2: purge old access_audit rows on startup (keep latest 100k).
            # NOTE: mutation_events is deliberately NOT purged — it is the
            # permanent provenance log. See #347 and the table comment above.
            if not self._state.read_only:
                try:
                    self._purge_access_audit(max_rows=100000)
                except Exception as exc:
                    logger.error(
                        "access_audit startup purge failed: %s", exc, exc_info=True,
                    )
                    record_subsystem_failure("audit_purge", exc)

            # Spec-09 (#112): form-level identity — layout_family column on
            # file_catalog. Additive migration for DBs that predate the
            # column. NULL = not yet computed (legacy rows stay NULL until
            # the next catalog pass fingerprints them).
            try:
                self.connection.execute(
                    "ALTER TABLE file_catalog ADD COLUMN IF NOT EXISTS "
                    "layout_family VARCHAR"
                )
            except Exception as exc:
                logger.warning("file_catalog.layout_family migration failed: %s", exc)

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

            # #288: Ordered schema migrations + versioning. Runs AFTER
            # the additive layer (above). The additive layer (ALTER
            # TABLE ADD COLUMN IF NOT EXISTS, CREATE TABLE IF NOT EXISTS,
            # backfill UPDATEs) keeps running first as always — it's the
            # "additive patch" layer that brings every DB to the v1
            # baseline. The migration runner then stamps user_version=1
            # and applies any future ordered migrations (1→2, 2→3, …).
            # Each migration is transactional + idempotent; a failure
            # rolls back and raises loudly (not warn-and-continue).
            # Skip in read-only mode (can't write user_version).
            if not self._state.read_only:
                try:
                    try:
                        from .schema_migrations import run_migrations
                    except ImportError:
                        from schema_migrations import run_migrations
                    self._migration_report = run_migrations(self.connection)
                except Exception as exc:
                    logger.error(
                        "Schema migration failed at init: %s. "
                        "The store may be in a pre-migration state. "
                        "See schema_migrations.py for recovery steps.",
                        exc,
                    )
                    # Fail loudly — do NOT silently continue with a
                    # half-migrated store. The caller (DuckDBMemoryStore
                    # __init__) will see this raise and propagate.
                    raise

    # -- helpers --------------------------------------------------------------

    def set_user_scope(self, user_id: str | None) -> None:
        self.user_id = (user_id or "default_user").strip()
        # Invalidate the alias cache on scope switch: resolve_aliases() builds
        # _alias_cache filtered by the previous user_id, and a stale cache would
        # leak one user's canonical entity names into another user's
        # resolve_aliases() result within a shared-tenant store. The cache is
        # already invalidated on add_alias/remove_alias; this closes the
        # scope-switch path (every RPC request calls set_user_scope first).
        self._state.alias_cache = None
        # #347: reset actor context to the new scope's defaults. The
        # facade/RPC layer will override with the real principal before
        # any mutation if one is available.
        self._actor = self.user_id
        self._actor_type = "human"

    def set_actor_context(self, actor: str | None, actor_type: str = "human") -> None:
        """#347: Set the actor identity for mutation_events.

        Called by the facade (from ctx.principal + ctx.principal_type) or
        the RPC service (from the resolved user_id + auth mode) before a
        mutation. Never a client-passed string — server-derived only.
        """
        self._actor = (actor or self.user_id).strip() or self.user_id
        self._actor_type = actor_type if actor_type in {"human", "model"} else "human"

    def get_schema_version(self) -> int:
        """#288: return the persisted schema version (PRAGMA user_version).

        Read at init by the migration runner. Exposed for diagnostics
        and the backup manifest.
        """
        try:
            from .schema_migrations import get_schema_version as _gsv
        except ImportError:
            from schema_migrations import get_schema_version as _gsv
        return _gsv(self.connection)

    def get_migration_report(self) -> dict | None:
        """#288: return the migration report from the last init, or None
        if migrations haven't run yet."""
        return getattr(self, "_migration_report", None)

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
        except (ValueError, TypeError, AttributeError) as exc:
            # SC7: log a warning so the operator can see the dropped
            # timestamp. The silent-drop behavior can lead to subtle data
            # quality issues (e.g. valid_from = None breaks temporal queries).
            logger.warning(
                "Unparseable timestamp %r dropped (returned None): %s",
                ts, exc,
            )
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
            # SC4: fail-safe — an unparseable expires_at is treated as
            # expired (return True) rather than non-expiring (return False).
            # The old behavior (return False) meant broken timestamps caused
            # sensitive memories to never expire. The safe side is to expire.
            logger.warning(
                "Unparseable expires_at %r (at=%r): %s — treating as "
                "expired (fail-safe). Normalize timestamps at the write "
                "boundary to prevent this.",
                expires_at, at, exc,
            )
            return True

    def _matches_scope(self, payload: dict | object) -> bool:
        """Check whether a record belongs to the current user's scope.

        SC5: accepts either a dict (legacy) or a MemoryRecord. When given
        a MemoryRecord, checks the ``user_scope`` column attribute (the
        SQL-level filter's source of truth) rather than the payload JSON
        (which could diverge from the column after a migration).
        """
        # SC5: prefer the record's user_scope attribute over payload dict.
        scope = getattr(payload, "user_scope", None)
        if scope is None and isinstance(payload, dict):
            scope = payload.get("user_scope")
        if scope is None:
            return True  # global memory
        return scope == self.user_id

    def _purge_access_audit(self, max_rows: int = 100000) -> int:
        """SC2: purge old access_audit rows, keeping only the latest *max_rows*.

        Called on startup to prevent unbounded growth. Returns the number
        of rows deleted.
        """
        try:
            with self._state.lock:
                assert self.connection is not None
                count = self.connection.execute(
                    "SELECT COUNT(*) FROM access_audit"
                ).fetchone()
                if not count or count[0] <= max_rows:
                    record_subsystem_ok("audit_purge")
                    return 0
                # Delete the oldest rows beyond max_rows.
                result = self.connection.execute(
                    f"""DELETE FROM access_audit
                        WHERE audit_id NOT IN (
                            SELECT audit_id FROM access_audit
                            ORDER BY ts DESC
                            LIMIT {int(max_rows)}
                        )"""
                )
                deleted = int(result.fetchone()[0]) if result else 0
                if deleted:
                    logger.info("access_audit purged %d old rows (kept %d)", deleted, max_rows)
                record_subsystem_ok("audit_purge")
                return deleted
        except Exception as exc:
            logger.error("access_audit purge failed: %s", exc, exc_info=True)
            record_subsystem_failure("audit_purge", exc)
            return 0

