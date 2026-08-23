"""DuckDB storage layer for hybrid memory.

Stores memory records with category, content, tags, JSON payload, timestamps,
and a DOUBLE[] embedding column for vector search via ``list_cosine_similarity``.
Falls back to ILIKE text search when embeddings are unavailable.

Categories (general-purpose — any topic):
  personal_fact  — stable things about the user (age, location, job, tools, diagnoses)
  preference     — how the user likes things (tools, communication style, habits)
  insight        — self-observations, realizations, patterns noticed
  event          — notable events with date context (job changes, milestones)
  relationship   — people in the user's life and dynamics
  goal           — things the user is working toward
  context_note   — situational context that helps future conversations
"""
from __future__ import annotations

import hashlib
import json
import re
import logging
import math
import threading

try:
    from .retriever import DuckDBRetriever
except ImportError:  # store.py imported as a top-level module (tests)
    from retriever import DuckDBRetriever
import time
import uuid
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb

# English function words excluded from text-leg scoring. Kept deliberately
# small: these are tokens that never discriminate between memories.
_TEXT_STOPWORDS = frozenset({
    "the", "and", "for", "with", "that", "this", "from", "was", "are",
    "were", "been", "have", "has", "had", "will", "would", "can", "could",
    "should", "into", "onto", "about", "after", "before", "between",
    "what", "when", "where", "which", "while", "who", "whom", "whose",
    "why", "how", "did", "does", "done", "doing", "not", "but", "nor",
    "his", "her", "hers", "its", "their", "theirs", "our", "ours",
    "your", "yours", "she", "him", "them", "they", "than", "then",
    "there", "here", "over", "under", "out", "off", "all", "any",
    "each", "few", "more", "most", "other", "some", "such", "only",
    "own", "same", "too", "very", "just", "also",
})

try:
    import numpy as np
except ImportError:
    np = None  # semantic dedup falls back to skip if numpy unavailable

logger = logging.getLogger(__name__)

VALID_CATEGORIES = frozenset({
    "personal_fact",
    "preference",
    "insight",
    "event",
    "relationship",
    "goal",
    "context_note",
})

_DEFAULT_TTL_DAYS = {
    "context_note": 30,
    "event": 180,
    "goal": 180,
}

# Sentinel for "parameter not provided" — distinguishes explicit None
# (clear/revive expiry) from "caller didn't pass this kwarg" (carry forward).
_NOT_PROVIDED = object()


class MemoryRecord:
    """In-memory representation of a stored memory row."""

    __slots__ = (
        "memory_id", "category", "content", "tags", "payload",
        "created_at", "updated_at", "expires_at", "embedding", "similarity",
        "raw_similarity",
        "status", "source", "confidence", "durability", "scope", "project_id",
        "user_scope",
        "retrieval_count", "last_retrieved_at", "helpful_count", "dismissed_count",
        "quarantine_reason", "quarantined_at",
        "valid_from", "valid_to", "superseded_by",
    )

    def __init__(
        self,
        memory_id: str,
        category: str,
        content: str,
        tags: List[str] | None = None,
        payload: Dict[str, Any] | None = None,
        created_at: str | None = None,
        updated_at: str | None = None,
        expires_at: str | None = None,
        embedding: List[float] | None = None,
        similarity: float = 0.0,
        raw_similarity: float | None = None,
        status: str = "active",
        source: str = "explicit",
        confidence: float | None = None,
        durability: str = "durable",
        scope: str = "profile",
        project_id: str | None = None,
        user_scope: str | None = None,
        retrieval_count: int = 0,
        last_retrieved_at: str | None = None,
        helpful_count: int = 0,
        dismissed_count: int = 0,
        quarantine_reason: str | None = None,
        quarantined_at: str | None = None,
        valid_from: str | None = None,
        valid_to: str | None = None,
        superseded_by: str | None = None,
    ) -> None:
        self.memory_id = memory_id
        self.category = category
        self.content = content
        self.tags = tags or []
        self.payload = payload or {}
        self.created_at = created_at
        self.updated_at = updated_at
        self.expires_at = expires_at
        self.embedding = embedding
        self.similarity = similarity
        # raw_similarity preserves the pre-importance-adjustment score for
        # gates that need pure retrieval strength (e.g. query expansion).
        # Defaults to similarity if not explicitly set.
        self.raw_similarity = raw_similarity if raw_similarity is not None else similarity
        self.status = status or "active"
        self.source = source or "explicit"
        self.confidence = confidence
        self.durability = durability or "durable"
        self.scope = scope or "profile"
        self.project_id = project_id
        self.user_scope = user_scope
        self.retrieval_count = int(retrieval_count or 0)
        self.last_retrieved_at = last_retrieved_at
        self.helpful_count = int(helpful_count or 0)
        self.dismissed_count = int(dismissed_count or 0)
        self.quarantine_reason = quarantine_reason
        self.quarantined_at = quarantined_at
        # Temporal validity: valid_from/valid_to define when this version
        # was/is current. superseded_by points to the newer version that
        # replaced it (NULL = current). Retrieval defaults to current state
        # (valid_to IS NULL); history is queryable via as_of parameter.
        self.valid_from = valid_from
        self.valid_to = valid_to
        self.superseded_by = superseded_by

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "category": self.category,
            "content": self.content,
            "tags": self.tags,
            "payload": self.payload,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "expires_at": self.expires_at,
            "similarity": round(self.similarity, 4),
            "raw_similarity": round(self.raw_similarity, 4),
            "status": self.status,
            "source": self.source,
            "confidence": self.confidence,
            "durability": self.durability,
            "scope": self.scope,
            "project_id": self.project_id,
            "retrieval_count": self.retrieval_count,
            "last_retrieved_at": self.last_retrieved_at,
            "helpful_count": self.helpful_count,
            "dismissed_count": self.dismissed_count,
            "quarantine_reason": self.quarantine_reason,
            "quarantined_at": self.quarantined_at,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "superseded_by": self.superseded_by,
        }


class DuckDBMemoryStore:
    """DuckDB-backed memory store with vector + text search."""

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
        self._lock = threading.Lock()
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
                    superseded_by      VARCHAR
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
                    review_model       VARCHAR
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
            }
            candidate_columns = {
                "user_scope": "VARCHAR",
                "evidence_text": "VARCHAR",
                "evidence_role": "VARCHAR DEFAULT 'user_turn'",
                "source_timestamp": "VARCHAR",
                "review_confidence": "DOUBLE",
                "review_model": "VARCHAR",
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
    def _is_expired(expires_at: str | None, at: str | None = None) -> bool:
        if not expires_at:
            return False
        try:
            exp = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            ref = (
                datetime.fromisoformat(at.replace("Z", "+00:00"))
                if at else datetime.now(timezone.utc)
            )
            return exp <= ref
        except Exception:
            return False

    def _matches_scope(self, payload: dict) -> bool:
        scope = payload.get("user_scope")
        if scope is None:
            return True  # global memory
        return scope == self.user_id

    def _row_to_record(self, row: dict, similarity: float = 0.0) -> MemoryRecord:
        return MemoryRecord(
            memory_id=row.get("memory_id", ""),
            category=row.get("category", "context_note"),
            content=row.get("content", ""),
            tags=row.get("tags") if row.get("tags") is not None else [],
            payload=json.loads(row.get("payload")) if row.get("payload") else {},
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
            expires_at=row.get("expires_at"),
            embedding=row.get("embedding"),
            similarity=similarity,
            status=row.get("status", "active"),
            source=row.get("source", "explicit"),
            confidence=row.get("confidence"),
            durability=row.get("durability", "durable"),
            scope=row.get("scope", "profile"),
            project_id=row.get("project_id"),
            user_scope=row.get("user_scope"),
            retrieval_count=row.get("retrieval_count", 0),
            last_retrieved_at=row.get("last_retrieved_at"),
            helpful_count=row.get("helpful_count", 0),
            dismissed_count=row.get("dismissed_count", 0),
            quarantine_reason=row.get("quarantine_reason"),
            quarantined_at=row.get("quarantined_at"),
            valid_from=row.get("valid_from"),
            valid_to=row.get("valid_to"),
            superseded_by=row.get("superseded_by"),
        )

    def _fetch_records(
        self, query: str, params: list | None = None, sim_col: str = ""
    ) -> List[MemoryRecord]:
        with self._lock:
            assert self.connection is not None
            result = self.connection.execute(query, params or [])
            columns = [desc[0] for desc in result.description]
            records: List[MemoryRecord] = []
            for row in result.fetchall():
                row_dict = dict(zip(columns, row))
                sim = row_dict.pop(sim_col, 0.0) if sim_col else 0.0
                records.append(self._row_to_record(row_dict, float(sim) if sim else 0.0))
            return records

    # -- vector search --------------------------------------------------------

    @staticmethod
    def _is_vector_search_unavailable(exc: Exception) -> bool:
        msg = str(exc).lower()
        return (
            "list_cosine_similarity" in msg
            or "catalog error" in msg
            or "binder error" in msg
            or "no function matches" in msg
            or "type with name double[]" in msg
            or "vss" in msg
        )

    def _text_search_raw(
        self, query: str, limit: int, excluded: set[str],
        category_filter: str | None = None,
        project_id: str | None = None,
        as_of: str | None = None,
        include_expired: bool = False,
    ) -> List[MemoryRecord]:
        """BM25-lite text search. Returns filtered records ranked by BM25.

        Does NOT record retrieval — the caller is responsible for that.
        Sets ``similarity`` to a max-normalized BM25 score so downstream
        fusion has a comparable signal.
        When *project_id* is provided, memories from other projects are
        excluded; global memories (project_id IS NULL) remain visible.

        When *include_expired* is True, the expiry filter is omitted (expired
        memories are returned, ranked normally).
        """
        tokens = [
            t for t in (
                w.lower() for w in re.findall(
                    r"[a-z0-9]+", query, flags=re.IGNORECASE)
            )
            if len(t) > 2 and t not in _TEXT_STOPWORDS
        ][:8]
        if not tokens:
            return []
        patterns = [f"%{t}%" for t in tokens]
        conditions = " OR ".join(["content ILIKE ?" for _ in patterns])
        project_clause = ""
        expiry_ref = as_of if as_of else self._now()
        if include_expired:
            expiry_clause = ""
            expiry_params: list = []
        else:
            expiry_clause = "AND (expires_at IS NULL OR expires_at > ?) "
            expiry_params = [expiry_ref]
        params: list = [self.user_id, *expiry_params]
        if project_id:
            project_clause = " AND (project_id IS NULL OR project_id = ?)"
            params.append(project_id)
        # Temporal filter: default to current (valid_to IS NULL),
        # or as_of (valid_from <= as_of AND (valid_to IS NULL OR valid_to > as_of))
        if as_of:
            temporal_clause = (
                "AND valid_from <= ? "
                "AND (valid_to IS NULL OR valid_to > ?) "
            )
            temporal_params = [as_of, as_of]
        else:
            temporal_clause = "AND valid_to IS NULL "
            temporal_params = []

        sql = (
            "SELECT * FROM memory_records "
            "WHERE COALESCE(status, 'active') = 'active' "
            f"{temporal_clause} "
            "AND (user_scope IS NULL OR user_scope = ?) "
            f"{expiry_clause}"
            f"{project_clause} AND ("
            f"{conditions}) LIMIT 2000"
        )
        results = self._fetch_records(sql, [*temporal_params, *params, *patterns])
        out: List[MemoryRecord] = []
        for r in results:
            if excluded and r.category.lower() in excluded:
                continue
            if category_filter and r.category != category_filter:
                continue
            if not self._matches_scope(r.payload):
                continue
            if not include_expired and self._is_expired(r.expires_at, at=expiry_ref):
                continue
            out.append(r)
        # BM25-lite ranking over the candidate pool: per-token document
        # frequency -> idf, term frequency -> saturation, length normalization.
        # Pure-Python so no extension/network dependency; O(tokens x docs).
        if out:
            contents = [(r.content or "").lower() for r in out]
            doc_lens = [len(c) or 1 for c in contents]
            avg_len = max(1.0, sum(doc_lens) / len(doc_lens))
            n_docs = len(out)
            dfs = {t: sum(1 for c in contents if t in c) for t in tokens}
            for r, content_lower, dlen in zip(out, contents, doc_lens):
                score = 0.0
                for t in tokens:
                    tf = content_lower.count(t)
                    if not tf:
                        continue
                    idf = math.log(1.0 + (n_docs - dfs[t] + 0.5) / (dfs[t] + 0.5))
                    score += idf * (2.2 * tf / (tf + 0.75 * (dlen / avg_len)))
                r.similarity = score
            top = max((r.similarity for r in out), default=0.0)
            if top > 0:
                for r in out:
                    r.similarity /= top
        # Sort by text-match score descending.
        out.sort(key=lambda r: r.similarity, reverse=True)
        return out

    def _vector_search_raw(
        self, emb: List[float], limit: int, excluded: set[str],
        category_filter: str | None = None,
        project_id: str | None = None,
        as_of: str | None = None,
        include_expired: bool = False,
    ) -> List[MemoryRecord]:
        """Vector similarity search. Returns filtered records ranked by cosine.

        Does NOT record retrieval — the caller is responsible for that.
        Raises on vector search errors so the caller can fall back.
        When *project_id* is provided, memories from other projects are
        excluded; global memories (project_id IS NULL) remain visible.

        When *include_expired* is True, the expiry filter is omitted (expired
        memories are returned, ranked normally).
        """
        project_clause = ""
        # String-cast the query vector as a fixed-size array constant.
        # A Python-list parameter binds through an interpreted per-row path
        # (~1ms/row — measured ~1.2s at 1k rows); the string-cast form is
        # materialized once by the planner and scanned natively (~7ms at 1k,
        # ~14ms at 5k).  Exact — identical ranking, no approximation.  The
        # dimension is derived from the actual vector, so a model swap with a
        # different embedding size keeps working.
        vec_text = "[" + ",".join(repr(float(x)) for x in emb) + "]"
        params: List[Any] = [vec_text]
        # Temporal filter: default to current (valid_to IS NULL),
        # or as_of (valid_from <= as_of AND (valid_to IS NULL OR valid_to > as_of))
        if as_of:
            temporal_clause = (
                "AND valid_from <= ? "
                "AND (valid_to IS NULL OR valid_to > ?) "
            )
            params.extend([as_of, as_of])
        else:
            temporal_clause = "AND valid_to IS NULL "
        expiry_ref = as_of if as_of else self._now()
        params.append(self.user_id)
        if include_expired:
            expiry_clause = ""
        else:
            expiry_clause = "  AND (expires_at IS NULL OR expires_at > ?) "
            params.append(expiry_ref)
        if project_id:
            project_clause = " AND (project_id IS NULL OR project_id = ?)"
            params.append(project_id)
        params.append(limit * 4)
        sql = (
            f"SELECT *, list_cosine_similarity(embedding, CAST(? AS DOUBLE[{len(emb)}])) AS sim "
            "FROM memory_records "
            "WHERE COALESCE(status, 'active') = 'active' "
            f"  {temporal_clause} "
            "  AND embedding IS NOT NULL "
            "  AND (user_scope IS NULL OR user_scope = ?) "
            f"{expiry_clause}"
            f"{project_clause} "
            "ORDER BY sim DESC "
            "LIMIT ?"
        )
        results = self._fetch_records(sql, params, sim_col="sim")
        out: List[MemoryRecord] = []
        for r in results:
            if excluded and r.category.lower() in excluded:
                continue
            if category_filter and r.category != category_filter:
                continue
            if not self._matches_scope(r.payload):
                continue
            if not include_expired and self._is_expired(r.expires_at, at=expiry_ref):
                continue
            out.append(r)
        return out

    # -- ranking: RRF + feedback + recency -----------------------------------

    _PHRASE_STOPWORDS = frozenset(
        ("the", "a", "an", "is", "are", "was", "were", "am", "be", "been",
         "i", "me", "my", "you", "your", "we", "our", "us", "this", "that",
         "these", "those", "it", "its", "of", "to", "in", "for", "and", "or",
         "but", "on", "with", "at", "by", "from", "as", "do", "does", "did",
         "what", "who", "how", "where", "when", "why", "which", "about", "so",
         "am", "very", "have", "has", "had", "would", "will", "can", "could")
    )

    _RRF_K = 20  # Lowered from 60 to sharpen relevance discrimination.
    # With k=60, rank 1 → 0.0164 and rank 10 → 0.0143 (spread ~0.002).
    # With k=20, rank 1 → 0.0476 and rank 10 → 0.0323 (spread ~0.015).
    # The wider spread lets relevance survive the importance adjustment.

    @classmethod
    def _rrf_fuse(
        cls,
        vector_results: List[MemoryRecord],
        text_results: List[MemoryRecord],
    ) -> List[MemoryRecord]:
        """Fuse vector and text rankings via Reciprocal Rank Fusion.

        RRF score(d) = sum over lists: 1 / (k + rank_in_list)
        A document appearing at rank 1 in both lists scores higher than
        one appearing at rank 1 in only one list.  The fused score is
        stored in ``similarity`` so callers see a single relevance number.
        """
        scores: Dict[str, float] = {}
        records_by_id: Dict[str, MemoryRecord] = {}

        for rank, record in enumerate(vector_results):
            rrf = 1.0 / (cls._RRF_K + rank + 1)
            scores[record.memory_id] = scores.get(record.memory_id, 0.0) + rrf
            records_by_id[record.memory_id] = record

        for rank, record in enumerate(text_results):
            rrf = 1.0 / (cls._RRF_K + rank + 1)
            scores[record.memory_id] = scores.get(record.memory_id, 0.0) + rrf
            # Prefer the vector record if it exists (has cosine sim);
            # otherwise use the text record.
            if record.memory_id not in records_by_id:
                records_by_id[record.memory_id] = record

        # Normalize: max possible is 2/(k+1) (rank 1 in both lists).
        max_rrf = 2.0 / (cls._RRF_K + 1)
        for mid, score in scores.items():
            records_by_id[mid].similarity = score / max_rrf  # 0.0 - 1.0

        fused = list(records_by_id.values())
        fused.sort(key=lambda r: r.similarity, reverse=True)
        return fused

    @staticmethod
    def _recency_boost(created_at: str | None) -> float:
        """Exponential decay: +0.10 today, ~0.037 at 90 days, ~0.014 at 180.

        Returns 0.0 if the timestamp is missing or unparseable.
        """
        if not created_at:
            return 0.0
        try:
            created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            days_old = max(0, (datetime.now(timezone.utc) - created).days)
            return 0.10 * math.exp(-days_old / 90.0)
        except Exception:
            return 0.0

    # Importance scoring weights — configurable via the store's attributes.
    # These are applied as additive adjustments to similarity (0-1 scale).
    _IMPORTANCE_HELPFUL_WEIGHT = 0.05       # per helpful vote
    _IMPORTANCE_DISMISSED_WEIGHT = -0.10    # per dismissed vote (before decay)
    _IMPORTANCE_CONFIDENCE_WEIGHT = 0.08    # * (confidence - 0.5)
    _IMPORTANCE_RETRIEVAL_WEIGHT = 0.02     # per retrieval (capped at 20)
    _IMPORTANCE_RETRIEVAL_CAP = 20
    _IMPORTANCE_AGE_DECAY_PER_DAY = 0.0005  # slow age penalty
    _IMPORTANCE_AGE_DECAY_CAP_DAYS = 730    # cap at 2 years
    _IMPORTANCE_DORMANCY_DECAY_PER_DAY = 0.001  # decay for not being retrieved
    _IMPORTANCE_DORMANCY_CAP_DAYS = 365     # cap at 1 year
    # Dismissal forgiveness: dismissals age out over this many days so a
    # single bad dismiss (frustration click, misattribution) doesn't
    # permanently sink a memory. The effective dismissed penalty is:
    #   -0.10 * dismissed_count * max(0, 1 - days_since / FORGIVENESS_DAYS)
    _IMPORTANCE_DISMISSAL_FORGIVENESS_DAYS = 180  # 6 months

    @classmethod
    def _importance_adjustment(cls, r: MemoryRecord) -> float:
        """Compute the importance-based similarity adjustment for a record.

        This is a transparent, additive scoring formula that combines:
          - Feedback signals (helpful/dismissed votes, with dismissal forgiveness)
          - Extraction confidence
          - Retrieval frequency (capped — popular memories get a boost)
          - Recency boost (exponential decay from creation date)
          - Age penalty (slow linear decay — old memories slowly fade)
          - Dormancy penalty (memories not retrieved recently slowly fade)

        Dismissal forgiveness: dismissals age out over
        _IMPORTANCE_DISMISSAL_FORGIVENESS_DAYS so a single bad dismiss
        (frustration click, misattribution) doesn't permanently sink a
        memory. Uses updated_at as a proxy for when the dismissal happened.
        """
        adj = 0.0
        # Feedback: helpful votes are permanent positive signal
        adj += cls._IMPORTANCE_HELPFUL_WEIGHT * r.helpful_count
        # Dismissed votes decay over time (forgiveness factor)
        if r.dismissed_count > 0:
            dismissal_factor = 1.0  # full penalty if no timestamp available
            if r.updated_at:
                try:
                    updated = datetime.fromisoformat(r.updated_at.replace("Z", "+00:00"))
                    days_since = max(0, (datetime.now(timezone.utc) - updated).days)
                    dismissal_factor = max(
                        0.0,
                        1.0 - days_since / cls._IMPORTANCE_DISMISSAL_FORGIVENESS_DAYS,
                    )
                except Exception:
                    pass
            adj += cls._IMPORTANCE_DISMISSED_WEIGHT * r.dismissed_count * dismissal_factor
        # Confidence: reward high-confidence extractions
        if r.confidence is not None:
            adj += cls._IMPORTANCE_CONFIDENCE_WEIGHT * (r.confidence - 0.5)
        # Retrieval frequency: frequently-retrieved memories are important
        # (capped to prevent rich-get-richer spiral)
        adj += cls._IMPORTANCE_RETRIEVAL_WEIGHT * min(r.retrieval_count, cls._IMPORTANCE_RETRIEVAL_CAP)
        # Recency boost (from creation date)
        adj += cls._recency_boost(r.created_at)
        # Age penalty: slow linear decay
        if r.created_at:
            try:
                created = datetime.fromisoformat(r.created_at.replace("Z", "+00:00"))
                age_days = max(0, (datetime.now(timezone.utc) - created).days)
                adj -= cls._IMPORTANCE_AGE_DECAY_PER_DAY * min(age_days, cls._IMPORTANCE_AGE_DECAY_CAP_DAYS)
            except Exception:
                pass
        # Dormancy penalty: memories not retrieved recently slowly fade
        if r.last_retrieved_at:
            try:
                last_ret = datetime.fromisoformat(r.last_retrieved_at.replace("Z", "+00:00"))
                dormant_days = max(0, (datetime.now(timezone.utc) - last_ret).days)
                adj -= cls._IMPORTANCE_DORMANCY_DECAY_PER_DAY * min(dormant_days, cls._IMPORTANCE_DORMANCY_CAP_DAYS)
            except Exception:
                pass
        return adj

    # Clamp the importance adjustment so it can't erase a relevance gap.
    # With k=20 RRF, the relevance spread between rank 1 and rank 10 is
    # ~0.015. An unclamped adjustment of +0.50 (from retrieval_count alone)
    # swamps that entirely. We split the adjustment into:
    #   - base signals (recency, age, dormancy) that are similar for nearby
    #     memories and don't discriminate — these get a tight clamp
    #   - feedback signals (helpful, dismissed, confidence, retrieval) that
    #     DO discriminate between equally-relevant memories — these get a
    #     wider clamp so a helpful vote still breaks ties
    _IMPORTANCE_BASE_CLAMP = 0.03   # recency/age/dormancy: tight clamp
    _IMPORTANCE_FEEDBACK_CLAMP = 0.08  # helpful/dismissed/confidence/retrieval

    @classmethod
    def _importance_adjustment_split(cls, r: MemoryRecord) -> tuple[float, float]:
        """Compute importance adjustment split into (base, feedback) components.

        base: recency + age + dormancy (similar for nearby memories)
        feedback: helpful + dismissed + confidence + retrieval (discriminates)
        """
        base = cls._recency_boost(r.created_at)
        # Age penalty
        if r.created_at:
            try:
                created = datetime.fromisoformat(r.created_at.replace("Z", "+00:00"))
                age_days = max(0, (datetime.now(timezone.utc) - created).days)
                base -= cls._IMPORTANCE_AGE_DECAY_PER_DAY * min(age_days, cls._IMPORTANCE_AGE_DECAY_CAP_DAYS)
            except Exception:
                pass
        # Dormancy penalty
        if r.last_retrieved_at:
            try:
                last_ret = datetime.fromisoformat(r.last_retrieved_at.replace("Z", "+00:00"))
                dormant_days = max(0, (datetime.now(timezone.utc) - last_ret).days)
                base -= cls._IMPORTANCE_DORMANCY_DECAY_PER_DAY * min(dormant_days, cls._IMPORTANCE_DORMANCY_CAP_DAYS)
            except Exception:
                pass

        feedback = cls._IMPORTANCE_HELPFUL_WEIGHT * r.helpful_count
        if r.dismissed_count > 0:
            dismissal_factor = 1.0
            if r.updated_at:
                try:
                    updated = datetime.fromisoformat(r.updated_at.replace("Z", "+00:00"))
                    days_since = max(0, (datetime.now(timezone.utc) - updated).days)
                    dismissal_factor = max(0.0, 1.0 - days_since / cls._IMPORTANCE_DISMISSAL_FORGIVENESS_DAYS)
                except Exception:
                    pass
            feedback += cls._IMPORTANCE_DISMISSED_WEIGHT * r.dismissed_count * dismissal_factor
        if r.confidence is not None:
            feedback += cls._IMPORTANCE_CONFIDENCE_WEIGHT * (r.confidence - 0.5)
        feedback += cls._IMPORTANCE_RETRIEVAL_WEIGHT * min(r.retrieval_count, cls._IMPORTANCE_RETRIEVAL_CAP)

        return base, feedback

    @classmethod
    def _apply_feedback_and_recency(cls, records: List[MemoryRecord]) -> None:
        """Adjust ``similarity`` in-place using importance scoring.

        The adjustment is split into base signals (recency/age/dormancy)
        and feedback signals (helpful/dismissed/confidence/retrieval).
        Each is clamped separately so:
        - base signals can't swamp relevance (tight ±0.03 clamp)
        - feedback signals can still break ties between equally-relevant
          memories (wider ±0.08 clamp)
        Without this split-clamp, a +0.40 retrieval boost (identical for
        all memories in the eval corpus) would swamp the ~0.015 RRF gap.
        """
        for r in records:
            base, feedback = cls._importance_adjustment_split(r)
            base = max(-cls._IMPORTANCE_BASE_CLAMP, min(cls._IMPORTANCE_BASE_CLAMP, base))
            feedback = max(-cls._IMPORTANCE_FEEDBACK_CLAMP, min(cls._IMPORTANCE_FEEDBACK_CLAMP, feedback))
            r.similarity = max(0.0, r.similarity + base + feedback)
        records.sort(key=lambda r: r.similarity, reverse=True)

    # -- P2C: recency/supersede-aware rank demotion (flag-gated, default OFF) --
    # Formally-superseded versions are already filtered out at retrieval time
    # (valid_to IS NULL), so this only covers the rarer case of two UNLINKED flat
    # memories that state the same fact (old address + new address) with separate
    # memory_ids. When a near-duplicate pair surfaces in the same result set, the
    # newer one should outrank the older. Gated by _P2C_ENABLED (default False) so
    # this never changes behaviour out of the box. Cheap: token-overlap only, no
    # LLM, no embeddings. Bounded: an older memory sinks at most a few ranks.
    _P2C_ENABLED = False
    _P2C_MIN_OVERLAP = 0.60        # token-Jaccard above this = "same fact"
    _P2C_MAX_SINK = 3              # bounded: never leapfrog more than this many ranks
    _P2C_SINK_EPSILON = 0.005      # newer must clear older by this tiny margin

    @staticmethod
    def _p2c_overlap(a: str, b: str) -> float:
        """Token Jaccard similarity between two content strings (casefolded)."""
        if not a or not b:
            return 0.0
        sa = set(a.casefold().split())
        sb = set(b.casefold().split())
        if not sa or not sb:
            return 0.0
        inter = len(sa & sb)
        union = len(sa | sb)
        return inter / union if union else 0.0

    @staticmethod
    def _p2c_ts(content: str) -> float:
        """Parse an ISO created_at to an epoch float, or -inf if unparseable."""
        if not content:
            return float("-inf")
        try:
            return datetime.fromisoformat(content.replace("Z", "+00:00")).timestamp()
        except Exception:
            return float("-inf")

    @classmethod
    def _apply_p2c(cls, records: List[MemoryRecord]) -> None:
        """If enabled, demote the newer member of each near-duplicate pair above the older."""
        if not cls._P2C_ENABLED:
            return
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                ri, rj = records[i], records[j]
                if cls._p2c_overlap(ri.content or "", rj.content or "") < cls._P2C_MIN_OVERLAP:
                    continue
                ti, tj = cls._p2c_ts(ri.created_at), cls._p2c_ts(rj.created_at)
                if ti == tj:
                    continue
                # Identical facts at two timestamps; assert older < newer order.
                if ti > tj:
                    ri, rj, i, j = rj, ri, j, i  # now i is the older, j the newer
                # Only demote when the older currently ranks higher AND the pair
                # is within the bounded sink window (no big leapfrogs).
                if i < j and (j - i) <= cls._P2C_MAX_SINK:
                    older, newer = records[i], records[j]
                    # Guarantee the newer outranks the older by a small epsilon.
                    target = min(1.0, older.similarity + cls._P2C_SINK_EPSILON)
                    if newer.similarity < target:
                        newer.similarity = target
        records.sort(key=lambda r: r.similarity, reverse=True)

    def _record_retrieval(self, records: List[MemoryRecord]) -> None:
        """Record only memories that were actually returned to the caller."""
        if not records:
            return
        now = self._now()
        ids = [record.memory_id for record in records]
        try:
            with self._lock:
                assert self.connection is not None
                # Single batched UPDATE (was a per-ID loop: 96 round-trips ->
                # 1). Measured 2026-08-22: the loop cost 0.5-1.3s on a 96-row
                # search at ~1k records; one IN-list statement is ~1 round-trip.
                placeholders = ", ".join("?" for _ in ids)
                self.connection.execute(
                    f"""UPDATE memory_records
                        SET retrieval_count = COALESCE(retrieval_count, 0) + 1,
                            last_retrieved_at = ?
                        WHERE memory_id IN ({placeholders})
                          AND COALESCE(status, 'active') = 'active'
                          AND valid_to IS NULL""",
                    [now, *ids],
                )
            for record in records:
                record.retrieval_count += 1
                record.last_retrieved_at = now
        except Exception as exc:
            # A read-only fallback connection must still be able to search.
            logger.debug("Could not record memory retrieval: %s", exc)

    def record_retrieval(self, memory_ids: List[str]) -> None:
        """Explicitly credit retrieval for the final injected memory list.

        The provider truncates a suppress_retrieval search to the memories it
        actually injects, then re-records them here — so pool filler and
        internal sub-queries never gain popularity credit.
        """
        if not memory_ids:
            return
        ids = [str(memory_id) for memory_id in memory_ids if str(memory_id).strip()]
        if not ids:
            return
        placeholders = ", ".join("?" for _ in ids)
        records = self._fetch_records(
            f"SELECT * FROM memory_records WHERE memory_id IN ({placeholders})"
            " AND (user_scope IS NULL OR user_scope = ?)"
            " AND COALESCE(status, 'active') = 'active' AND valid_to IS NULL",
            [*ids, self.user_id],
        )
        self._record_retrieval(records)

    def get_memories_by_ids(
        self,
        memory_ids: List[str],
        *,
        include_quarantined: bool = False,
    ) -> List[MemoryRecord]:
        """Fetch visible, in-scope memories by ID for graph-aware retrieval."""
        ids = [str(memory_id) for memory_id in memory_ids if str(memory_id).strip()]
        if not ids:
            return []
        placeholders = ", ".join("?" for _ in ids)
        status_clause = "" if include_quarantined else " AND COALESCE(status, 'active') = 'active' AND valid_to IS NULL"
        records = self._fetch_records(
            f"SELECT * FROM memory_records WHERE memory_id IN ({placeholders})"
            " AND (user_scope IS NULL OR user_scope = ?)"
            f"{status_clause}",
            [*ids, self.user_id],
        )
        by_id = {record.memory_id: record for record in records}
        return [by_id[memory_id] for memory_id in ids if memory_id in by_id
                and self._matches_scope(by_id[memory_id].payload)
                and not self._is_expired(by_id[memory_id].expires_at)]

    def search(
        self,
        query: str,
        *args: Any,
        **kwargs: Any,
    ) -> List[MemoryRecord]:
        """Run retrieval through the configured retriever.
        The default ``DuckDBRetriever`` wraps the existing scan-based hybrid
        search (vector + ILIKE + RRF fusion).  Alternative engines (ANN,
        BM25, graph-candidate) can be plugged in via ``set_retriever``
        without touching the provider.  Behavior is identical to the
        pre-seam implementation.
        """
        _t0 = time.perf_counter()
        try:
            retriever = getattr(self, "_retriever", None)
            if retriever is None:
                retriever = self._retriever = DuckDBRetriever(self)
            return retriever.search(query, *args, **kwargs)
        finally:
            self._record_scale_metric(time.perf_counter() - _t0)

    def _record_scale_metric(self, elapsed_s: float) -> None:
        """Track query latency and corpus size; warn when thresholds cross.

        The measured trigger for expensive-engine gating (ANN/BM25, fact
        families): a rolling p95 latency window and the active record count.
        Warnings fire once per crossing and are actionable — they name the
        threshold that was exceeded, not a vague "slow".
        """
        try:
            self._scale_latencies.append(elapsed_s * 1000.0)
            self._scale_queries += 1
            # Sample the record count every 25 queries (COUNT(*) on every
            # query would add overhead we're trying to measure).
            if self._scale_queries - self._scale_last_count_check >= 25:
                self._scale_last_count_check = self._scale_queries
                try:
                    self._scale_record_count = int(self.count())
                except Exception:
                    pass
            n = len(self._scale_latencies)
            if n < 5:
                return  # too few samples to be meaningful
            avg_ms = sum(self._scale_latencies) / n
            p95_ms = sorted(self._scale_latencies)[int(n * 0.95) - 1]
            over_latency = p95_ms > self._scale_warn_latency_ms
            over_count = (self._scale_record_count or 0) > self._scale_warn_records
            if over_latency or over_count:
                self._scale_warnings_fired += 1
                logger.warning(
                    "ARGOS_SCALE: p95=%.0fms avg=%.0fms (warn>%.0fms) "
                    "records=%s (warn>%s) — if this persists, enable ANN/BM25 "
                    "indexing or fact-family consolidation per the scaling "
                    "roadmap (trigger: %s)",
                    p95_ms, avg_ms, self._scale_warn_latency_ms,
                    self._scale_record_count, self._scale_warn_records,
                    "latency" if over_latency else "corpus-size",
                )
        except Exception:
            pass  # metrics must never break retrieval

    def set_scale_thresholds(self, warn_latency_ms: float, warn_records: int) -> None:
        """Configure the scale-trigger thresholds (from the provider config)."""
        self._scale_warn_latency_ms = float(warn_latency_ms)
        self._scale_warn_records = int(warn_records)

    def get_scale_metrics(self) -> Dict[str, Any]:
        """Return current scale-trigger state (for dashboards/handoffs)."""
        n = len(self._scale_latencies)
        p95 = sorted(self._scale_latencies)[int(n * 0.95) - 1] if n >= 5 else 0.0
        return {
            "queries_measured": self._scale_queries,
            "window": n,
            "avg_latency_ms": round(sum(self._scale_latencies) / n, 1) if n else 0.0,
            "p95_latency_ms": round(p95, 1),
            "max_latency_ms": round(max(self._scale_latencies), 1) if n else 0.0,
            "record_count": self._scale_record_count,
            "warnings_fired": self._scale_warnings_fired,
            "warn_latency_ms": self._scale_warn_latency_ms,
            "warn_records": self._scale_warn_records,
        }

    def set_retriever(self, retriever: Any) -> None:
        """Swap the retrieval engine (advanced; must match the protocol)."""
        self._retriever = retriever

    def _hybrid_search(
        self,
        query: str,
        limit: int = 5,
        exclude_categories: List[str] | None = None,
        category_filter: str | None = None,
        project_id: str | None = None,
        as_of: str | None = None,
        suppress_retrieval: bool = False,
        include_expired: bool = False,
    ) -> List[MemoryRecord]:
        """Hybrid search: RRF-fused vector + text, with optional cross-encoder
        re-ranking, feedback, and recency.

        When embeddings are available, runs vector and text search in
        parallel and fuses results via Reciprocal Rank Fusion.  When
        embeddings are unavailable, falls back to text-only search.  If
        a cross-encoder reranker is available, the top candidates are
        re-scored with full bidirectional attention before final ranking.
        In all cases, the final ranking is adjusted by feedback signals
        (helpful/dismissed), confidence, and recency.

        When *project_id* is provided, memories from other projects are
        excluded; global memories (project_id IS NULL) remain visible.

        When *suppress_retrieval* is True, _record_retrieval is skipped —
        eval/diagnostic runs won't inflate retrieval_count on the memories
        they search. This prevents eval self-pollution where repeated eval
        runs pump the retrieval_count of eval-relevant memories to the cap,
        flattening the retrieval signal as a discriminator.

        When *include_expired* is True, expired memories are included in
        results (ranked normally) — for auditing "what did I know then".
        """
        excluded = {c.lower() for c in (exclude_categories or [])}
        emb: List[float] = []
        if self.embedder and hasattr(self.embedder, "embed"):
            emb = self.embedder.embed(query, is_query=True)

        # Retrieve more candidates than requested so the reranker has a
        # larger pool to work with. If no reranker, just use limit.
        # Phrase-lift needs a wider pool so exact-phrase matches sitting just
        # outside the window can be pulled in.
        reranker_top_n = getattr(self, "_reranker_top_n", 20)
        phrase_pool = getattr(self, "_phrase_lift_pool", 0)
        if phrase_pool:
            pool_size = max(limit, reranker_top_n, phrase_pool)
        else:
            pool_size = max(limit, reranker_top_n) if self.reranker else limit

        # Gather candidate results from both paths.
        vector_results: List[MemoryRecord] = []
        text_results: List[MemoryRecord] = self._text_search_raw(
            query, pool_size, excluded, category_filter,
            project_id=project_id, as_of=as_of,
            include_expired=include_expired,
        )

        if emb:
            try:
                vector_results = self._vector_search_raw(
                    emb, pool_size, excluded, category_filter,
                    project_id=project_id, as_of=as_of,
                    include_expired=include_expired,
                )
            except Exception as exc:
                if not self._is_vector_search_unavailable(exc):
                    logger.warning("Vector search error: %s", exc)
                vector_results = []

        # Fuse or select.
        if vector_results and text_results:
            fused = self._rrf_fuse(vector_results, text_results)
        elif vector_results:
            fused = vector_results
        elif text_results:
            fused = text_results
        else:
            return []

        # Cross-encoder re-ranking: re-score the top N candidates with
        # full bidirectional attention. The cross-encoder score is blended
        # with the existing RRF similarity (not replacing it) so we keep
        # the bi-encoder's signal while adding the cross-encoder's nuance.
        # Blend: 20% cross-encoder + 80% original similarity, after
        # normalizing the cross-encoder scores to 0-1. This is intentionally
        # conservative — the cross-encoder acts as a gentle tie-breaker
        # rather than overriding the bi-encoder's ranking.
        if self.reranker and len(fused) > 1:
            rerank_pool = fused[:reranker_top_n]
            documents = [r.content for r in rerank_pool]
            scores = self.reranker.score(query, documents)
            if scores and len(scores) == len(rerank_pool):
                min_s, max_s = min(scores), max(scores)
                range_s = max_s - min_s
                for i, record in enumerate(rerank_pool):
                    if range_s > 0:
                        ce_norm = (scores[i] - min_s) / range_s
                    else:
                        ce_norm = 0.5
                    record.similarity = 0.8 * record.similarity + 0.2 * ce_norm
                rerank_pool.sort(key=lambda r: r.similarity, reverse=True)
                fused = rerank_pool + fused[reranker_top_n:]

        # Preserve raw similarity (pre-importance) for gates that need
        # pure retrieval strength (e.g. query expansion trigger).
        for r in fused:
            r.raw_similarity = r.similarity

        # Exact-phrase lift (optional, default off): reward contiguous
        # query bigrams present verbatim in the memory, which the unigram-
        # only text search never scores. Fixes the class of query where the
        # gold memory shares the exact phrase (e.g. "who is the sales
        # director" -> "...Raymond is the Sales Director...") but was ranked
        # low because token overlap tied it with merely-similar content.
        _alpha = getattr(self, "_phrase_lift_alpha", 0.0)
        if _alpha and _alpha > 0.0:
            qwords = re.findall(r"[a-z0-9']+", query.lower())
            qwords = [w for w in qwords if w not in self._PHRASE_STOPWORDS]
            qbigrams = [(t0, t1) for t0, t1 in zip(qwords, qwords[1:])]
            for r in fused:
                if not qbigrams or not r.content:
                    continue
                c = r.content.lower()
                ngram_win = sum(1 for a, b in qbigrams if f"{a} {b}" in c)
                if ngram_win:
                    r.similarity += _alpha * (ngram_win / len(qbigrams))
            fused.sort(key=lambda r: r.similarity, reverse=True)

        # Apply feedback weighting and recency boost, then truncate.
        self._apply_feedback_and_recency(fused)
        self._apply_p2c(fused)  # P2C: demote older of a near-duplicate pair (flag-gated)
        final = fused[:limit]
        if not suppress_retrieval:
            self._record_retrieval(final)
        return final

    # -- write operations -----------------------------------------------------

    # Semantic dedup threshold: cosine similarity above this means "same fact".
    _DEDUP_SIMILARITY_THRESHOLD = 0.85

    def _content_exists(self, content: str, category: str) -> bool:
        """Check if a very similar content already exists (dedup).

        Three-layer check:
        1. Exact match (case-sensitive, same category).
        2. Substring containment (case-insensitive, for >20 char strings).
        3. Semantic similarity (cosine similarity on embeddings, when an
           embedder is available).  Catches paraphrased duplicates like
           "User is married to Sam" vs "Sam is the user's partner".
        """
        with self._lock:
            assert self.connection is not None
            # Layer 1: exact match (only against current versions).
            result = self.connection.execute(
                """SELECT COUNT(*) FROM memory_records
                  WHERE content = ? AND category = ?
                    AND valid_to IS NULL
                    AND (user_scope IS NULL OR user_scope = ?)""",
                [content, category, self.user_id],
            ).fetchone()
            if result and result[0] > 0:
                return True
            # Layer 2: substring containment (case-insensitive, current only).
            result = self.connection.execute(
                """SELECT content FROM memory_records
                  WHERE category = ?
                    AND valid_to IS NULL
                    AND (user_scope IS NULL OR user_scope = ?)
                  LIMIT 500""",
                [category, self.user_id],
            ).fetchall()
            content_lower = content.lower().strip()
            for (existing,) in result:
                existing_lower = existing.lower().strip()
                if content_lower == existing_lower:
                    return True
                # Skip very short strings to avoid false positives.
                if len(content_lower) > 20 and len(existing_lower) > 20:
                    if content_lower in existing_lower or existing_lower in content_lower:
                        return True
            # Layer 3: semantic similarity via embeddings.
            if self.embedder and hasattr(self.embedder, "embed"):
                emb = self.embedder.embed(content)
                if emb:
                    try:
                        # Same string-cast constant trick as _vector_search_raw
                        # (Python-list params bind per-row; ~1s at 1k rows).
                        vec_text = "[" + ",".join(repr(float(x)) for x in emb) + "]"
                        result = self.connection.execute(
                            f"""SELECT memory_id FROM memory_records
                              WHERE category = ? AND embedding IS NOT NULL
                                AND valid_to IS NULL
                                AND (user_scope IS NULL OR user_scope = ?)
                                AND list_cosine_similarity(embedding, CAST(? AS DOUBLE[{len(emb)}])) > ?
                               LIMIT 1""",
                            [category, self.user_id, vec_text, self._DEDUP_SIMILARITY_THRESHOLD],
                        ).fetchone()
                        if result:
                            return True
                    except Exception as exc:
                        if not self._is_vector_search_unavailable(exc):
                            logger.debug("Semantic dedup check failed: %s", exc)
            return False

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
        status: str = "active",
        expires_at: Any = _NOT_PROVIDED,
    ) -> MemoryRecord | None:
        """Insert a memory record. Returns None if deduped away.

        *expires_at* semantics (Spec 1):
        - ``_NOT_PROVIDED`` (default): auto-TTL logic applies (current behavior).
        - ``None``: explicitly no expiry (skip auto-TTL, store NULL).
        - ISO-8601 string: set to that value (explicit wins over TTL map).
        """
        if not content or not content.strip():
            return None
        if category not in VALID_CATEGORIES:
            logger.warning("Unknown category '%s', defaulting to context_note", category)
            category = "context_note"

        record_payload = dict(payload or {})
        record_payload.setdefault("user_scope", self.user_id)
        source = str(source or record_payload.get("source") or "explicit")
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
            record_payload["expires_at"] = expires_at
        elif expires_at is None:
            # Explicit None: clear any pre-existing payload expiry so the
            # column (populated from record_payload below) is NULL, not
            # a stale value from the caller's payload dict.
            record_payload.pop("expires_at", None)

        if dedup and self._content_exists(content, category):
            logger.debug("Deduped memory: %s", content[:60])
            return None

        memory_id = f"mem-{uuid.uuid4().hex}"
        now = self._now()
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

        sql = """
            INSERT INTO memory_records
                (memory_id, category, content, tags, payload, created_at, updated_at,
                 expires_at, embedding, status, source, confidence, durability, scope,
                 project_id, user_scope, retrieval_count, helpful_count, dismissed_count,
                 valid_from)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, 0, ?)
        """
        with self._lock:
            assert self.connection is not None
            self.connection.execute(sql, [
                memory_id, category, content, tags or [],
                json.dumps(record_payload), now, now,
                record_payload.get("expires_at"),
                emb if emb else None,
                status, source, confidence, durability, scope, project_id,
                record_payload.get("user_scope"),
                now,  # valid_from = creation time
            ])
        fetched = self._fetch_records(
            "SELECT * FROM memory_records WHERE memory_id = ?", [memory_id]
        )
        return fetched[0] if fetched else None

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
            durability, scope, project_id, session_id, status, created_at,
            updated_at, reviewed_at, review_reason, evidence_text, evidence_role,
            source_timestamp, review_confidence, review_model,
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
        session_id: str = "",
        evidence_text: str = "",
        evidence_role: str = "user_turn",
        source_timestamp: str | None = None,
        dedup: bool = True,
    ) -> dict | None:
        """Store a pending proposal without making it retrievable memory."""
        if not content or not content.strip():
            return None
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

        candidate_id = f"cand-{uuid.uuid4().hex}"
        now = self._now()
        evidence_text = (evidence_text or "")[:8000]
        source_timestamp = source_timestamp or now
        candidate_payload = dict(payload or {})
        candidate_payload.setdefault("user_scope", self.user_id)
        candidate_payload.setdefault("source", source)
        try:
            normalized_confidence = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            normalized_confidence = 0.5
        with self._lock:
            assert self.connection is not None
            self.connection.execute(
                """INSERT INTO memory_candidates
                  (candidate_id, category, content, tags, payload, source,
                   confidence, durability, scope, project_id, session_id,
                   user_scope, status, created_at, updated_at, evidence_text,
                   evidence_role, source_timestamp)
                  VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?)""",
                [
                    candidate_id, category, content.strip(), tags or [],
                    json.dumps(candidate_payload), source, normalized_confidence,
                    durability or "durable", scope or "profile", project_id,
                    session_id or "", candidate_payload.get("user_scope"),
                    now, now, evidence_text,
                    evidence_role or "user_turn", source_timestamp,
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
        sql = "SELECT candidate_id, category, content, tags, payload, source, confidence, durability, scope, project_id, session_id, status, created_at, updated_at, reviewed_at, review_reason, evidence_text, evidence_role, source_timestamp, review_confidence, review_model FROM memory_candidates"
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(max(1, min(int(limit), 500)))
        with self._lock:
            assert self.connection is not None
            rows = self.connection.execute(sql, params).fetchall()
        return [self._candidate_row_to_dict(row) for row in rows]

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
        now = self._now()
        memory = None
        final_status = decision
        superseded_ok = False
        if decision in {"approved", "reviewed_approved"}:
            selected_durability = durability or candidate["durability"]
            selected_scope = scope or candidate["scope"]
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
            )
            # Pass explicit expires_at through to remember() (Spec 1).
            if expires_at is not _NOT_PROVIDED:
                remember_kwargs["expires_at"] = expires_at
            memory = self.remember(**remember_kwargs)
            if memory is None:
                final_status = "deduplicated"
            elif supersedes_memory_id:
                # Chain the new memory behind the named current record.
                # Same supersession semantics as update_memory, but the new
                # version's content comes from the candidate, not the old
                # record. Scope-checked: cannot supersede another user's
                # memory.
                with self._lock:
                    assert self.connection is not None
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
                               WHERE memory_id = ?""",
                            [now, memory.memory_id, now, supersedes_memory_id],
                        )
                        superseded_ok = True
        with self._lock:
            assert self.connection is not None
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
        result = {
            "candidate": self.list_candidates(candidate_id=candidate_id, limit=1)[0],
            "memory": memory.to_dict() if memory else None,
        }
        if supersedes_memory_id:
            result["supersedes_memory_id"] = supersedes_memory_id
            result["superseded"] = superseded_ok
        return result

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
        """Restore a quarantined memory to active retrieval."""
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
                sql = "UPDATE memory_records SET helpful_count = COALESCE(helpful_count, 0) + 1, updated_at = ? WHERE memory_id = ?"
                params = [now, memory_id]
            elif feedback == "dismissed":
                sql = "UPDATE memory_records SET dismissed_count = COALESCE(dismissed_count, 0) + 1, updated_at = ? WHERE memory_id = ?"
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

        The old record is preserved for history queries (as_of parameter).
        If no content/tags/payload changes are provided, returns the existing
        record unchanged.
        """
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

        # Resolve effective expires_at:
        # _NOT_PROVIDED → carry forward; None → clear (revive); str → set.
        if expires_at is _NOT_PROVIDED:
            effective_expires = rec.expires_at
        else:
            effective_expires = expires_at  # None or string

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
                    project_id, user_scope, retrieval_count, helpful_count, dismissed_count,
                    valid_from, valid_to, superseded_by)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)""",
                [new_id, rec.category, new_content, new_tags,
                 json.dumps(new_payload), now, now,
                 effective_expires,
                 new_emb if new_emb else None,
                 rec.status, rec.source, rec.confidence, rec.durability, rec.scope,
                 rec.project_id,
                 rec.payload.get("user_scope"),
                 rec.retrieval_count, rec.helpful_count, rec.dismissed_count,
                 now],
            )
            # 2. Supersede the old version
            self.connection.execute(
                """UPDATE memory_records
                   SET valid_to = ?, superseded_by = ?, updated_at = ?
                   WHERE memory_id = ?""",
                [now, new_id, now, memory_id],
            )
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
            check = self.connection.execute(
                """SELECT COUNT(*) FROM memory_records
                   WHERE memory_id = ?
                     AND (user_scope IS NULL OR user_scope = ?)""",
                [memory_id, self.user_id],
            ).fetchone()
            if not check or check[0] == 0:
                return False
            # Chain position: is this the head (valid_to IS NULL)?
            row = self.connection.execute(
                """SELECT valid_to FROM memory_records
                   WHERE memory_id = ?
                     AND (user_scope IS NULL OR user_scope = ?)""",
                [memory_id, self.user_id],
            ).fetchone()
            is_head = row is not None and row[0] is None
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
                    self.connection.execute(
                        """UPDATE memory_records
                           SET valid_to = NULL, superseded_by = NULL, updated_at = ?
                           WHERE memory_id = ?""",
                        [now, pred[0]],
                    )
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
            # Head with no predecessor: hard delete.
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
            return {"deleted": True, "action": "deleted"}

    # -- entity aliases -------------------------------------------------------

    def add_alias(self, alias: str, canonical_entity: str) -> None:
        """Map an alias to a canonical entity name.

        Example: add_alias("my wife", "Alex") means that searching for
        "my wife" will also match graph entities for "Alex".
        """
        alias = alias.strip().lower()
        canonical = canonical_entity.strip()
        if not alias or not canonical:
            return
        now = self._now()
        with self._lock:
            assert self.connection is not None
            self.connection.execute(
                """INSERT OR REPLACE INTO entity_aliases
                   (alias, canonical_entity, user_scope, created_at)
                   VALUES (?, ?, ?, ?)""",
                [alias, canonical.lower(), self.user_id, now],
            )

    def remove_alias(self, alias: str, canonical_entity: str | None = None) -> bool:
        """Remove an alias mapping. If canonical_entity is None, removes all
        mappings for that alias."""
        alias = alias.strip().lower()
        with self._lock:
            assert self.connection is not None
            if canonical_entity:
                canonical = canonical_entity.strip().lower()
                result = self.connection.execute(
                    """DELETE FROM entity_aliases
                       WHERE alias = ? AND canonical_entity = ? AND user_scope = ?""",
                    [alias, canonical, self.user_id],
                )
            else:
                result = self.connection.execute(
                    """DELETE FROM entity_aliases
                       WHERE alias = ? AND user_scope = ?""",
                    [alias, self.user_id],
                )
            return True

    def resolve_aliases(self, text: str) -> List[str]:
        """Given a text query, return canonical entity names for any aliases
        found in the text.

        Example: resolve_aliases("tell me about my wife") → ["Alex"]
        """
        if not text:
            return []
        text_lower = text.lower()
        with self._lock:
            assert self.connection is not None
            rows = self.connection.execute(
                """SELECT alias, canonical_entity FROM entity_aliases
                   WHERE user_scope = ?""",
                [self.user_id],
            ).fetchall()
        canonicals: set[str] = set()
        for alias, canonical in rows:
            if alias and alias in text_lower:
                canonicals.add(canonical)
        return sorted(canonicals)

    def list_aliases(self) -> List[Dict[str, str]]:
        """List all alias mappings for this user."""
        with self._lock:
            assert self.connection is not None
            rows = self.connection.execute(
                """SELECT alias, canonical_entity FROM entity_aliases
                   WHERE user_scope = ? ORDER BY canonical_entity, alias""",
                [self.user_id],
            ).fetchall()
        return [{"alias": r[0], "canonical_entity": r[1]} for r in rows]

    def aliases_for_canonical(self, canonical_entity: str) -> List[str]:
        """Return all aliases that map to a canonical entity name.

        This is the reverse of resolve_aliases: given "Alex", returns
        ["my wife", "the wife"] — so a search for "Alex" can also
        search for memories that mention "my wife" without naming Alex.
        """
        canonical = canonical_entity.strip().lower()
        if not canonical:
            return []
        with self._lock:
            assert self.connection is not None
            rows = self.connection.execute(
                """SELECT alias FROM entity_aliases
                   WHERE canonical_entity = ? AND user_scope = ?""",
                [canonical, self.user_id],
            ).fetchall()
        return [r[0] for r in rows if r[0]]

    # -- listing --------------------------------------------------------------

    def list_recent(self, limit: int = 10) -> List[MemoryRecord]:
        sql = (
            "SELECT * FROM memory_records WHERE COALESCE(status, 'active') = 'active' "
            "AND valid_to IS NULL "
            "AND (user_scope IS NULL OR user_scope = ?) "
            "AND (expires_at IS NULL OR expires_at > ?) "
            "ORDER BY created_at DESC LIMIT ?"
        )
        results = self._fetch_records(sql, [self.user_id, self._now(), limit])
        return [r for r in results if self._matches_scope(r.payload) and not self._is_expired(r.expires_at)]

    def list_by_category(self, category: str, limit: int = 50) -> List[MemoryRecord]:
        sql = (
            "SELECT * FROM memory_records WHERE category = ? "
            "AND COALESCE(status, 'active') = 'active' "
            "AND valid_to IS NULL "
            "AND (user_scope IS NULL OR user_scope = ?) "
            "AND (expires_at IS NULL OR expires_at > ?) "
            "ORDER BY created_at DESC LIMIT ?"
        )
        results = self._fetch_records(sql, [category, self.user_id, self._now(), limit])
        return [r for r in results if self._matches_scope(r.payload) and not self._is_expired(r.expires_at)]

    def list_memories(
        self, category: str | None = None, limit: int = 100
    ) -> List[MemoryRecord]:
        if category:
            return self.list_by_category(category, limit)
        return self.list_recent(limit)

    def get_insights(
        self,
        tags: List[str] | None = None,
        since: str | None = None,
        limit: int = 50,
    ) -> List[MemoryRecord]:
        """Retrieve insight-category memories, newest-first.

        Args:
            tags: If provided, only return insights whose tags list
                contains at least one of the given tags (OR semantics).
            since: ISO timestamp; only return insights created at or
                after this time.
            limit: Maximum number of results (default 50).

        Returns:
            List of MemoryRecords with category='insight', sorted
            newest-first by created_at.
        """
        conditions = [
            "category = 'insight'",
            "COALESCE(status, 'active') = 'active'",
            "valid_to IS NULL",
            "(user_scope IS NULL OR user_scope = ?)",
        ]
        params: list = [self.user_id]
        if since:
            conditions.append("created_at >= ?")
            params.append(since)
        if tags:
            # DuckDB list_contains checks if a tag is in the tags array.
            placeholders = ", ".join(["?" for _ in tags])
            conditions.append(f"EXISTS (SELECT 1 FROM range(0, list_length(tags)) AS i WHERE list_contains(tags, i, t) IN ({placeholders}))")
            # Simpler: use OR of list_contains per tag.
            # Actually DuckDB's list_contains is: list_contains(list, value)
            # Let's use the simpler approach.
            conditions = conditions[:-1]  # remove the complex one
            tag_conditions = " OR ".join(["list_contains(tags, ?)" for _ in tags])
            conditions.append(f"({tag_conditions})")
            params.extend(tags)
        where = " AND ".join(conditions)
        sql = (
            f"SELECT * FROM memory_records WHERE {where} "
            "ORDER BY created_at DESC LIMIT ?"
        )
        params.append(limit)
        try:
            return self._fetch_records(sql, params)
        except Exception as exc:
            logger.debug("get_insights query failed: %s", exc)
            # Fallback: filter in Python (tags column might not be queryable).
            sql_simple = (
                "SELECT * FROM memory_records WHERE category = 'insight' "
                "AND COALESCE(status, 'active') = 'active' "
                "AND valid_to IS NULL "
                "AND (user_scope IS NULL OR user_scope = ?) "
                "ORDER BY created_at DESC LIMIT ?"
            )
            results = self._fetch_records(sql_simple, [self.user_id, limit])
            if tags:
                tag_set = {t.lower() for t in tags}
                results = [r for r in results if tag_set & {t.lower() for t in (r.tags or [])}]
            if since:
                results = [r for r in results if r.created_at and r.created_at >= since]
            return results

    def count(self) -> int:
        """Count current (non-superseded) memories for this user."""
        with self._lock:
            assert self.connection is not None
            result = self.connection.execute(
                """SELECT COUNT(*) FROM memory_records
                   WHERE valid_to IS NULL
                     AND (user_scope IS NULL OR user_scope = ?)""",
                [self.user_id],
            ).fetchone()
            return result[0] if result else 0

    # -- junk cleanup ---------------------------------------------------------

    def cleanup_junk(self, return_ids: bool = False) -> int | Dict[str, Any]:
        """Quarantine low-quality memories without deleting their records.

        This method name remains for lifecycle compatibility, but cleanup is now
        reversible: questionable records are marked ``quarantined`` and hidden
        from search/injection. Reviewers can restore or delete them later.
        """
        try:
            from .extractor import hard_quality_flags, quality_flags_for_fact
        except ImportError:
            from extractor import hard_quality_flags, quality_flags_for_fact

        all_records = self._fetch_records(
            """SELECT * FROM memory_records
               WHERE COALESCE(status, 'active') = 'active'
                 AND valid_to IS NULL
                 AND (user_scope IS NULL OR user_scope = ?)""",
            [self.user_id],
        )
        to_quarantine: Dict[str, str] = {}
        seen_content: Dict[tuple[str, str], str] = {}

        for rec in all_records:
            fact = {
                "category": rec.category,
                "content": rec.content,
                "tags": rec.tags,
                "payload": rec.payload,
            }
            flags = hard_quality_flags(quality_flags_for_fact(fact))
            if flags:
                to_quarantine[rec.memory_id] = "; ".join(flags)
                continue

            fingerprint = rec.content.lower().strip()[:60]
            key = (rec.category, fingerprint)
            if key in seen_content:
                existing_id = seen_content[key]
                existing_rec = next(
                    (r for r in all_records if r.memory_id == existing_id), None
                )
                if existing_rec and len(rec.content) > len(existing_rec.content):
                    to_quarantine[existing_id] = "near_duplicate_shorter"
                    seen_content[key] = rec.memory_id
                else:
                    to_quarantine[rec.memory_id] = "near_duplicate_shorter"
            else:
                seen_content[key] = rec.memory_id

        quarantined = 0
        quarantined_ids: List[str] = []
        for memory_id, reason in to_quarantine.items():
            if self.quarantine_memory(memory_id, reason):
                quarantined += 1
                quarantined_ids.append(memory_id)

        if quarantined:
            logger.info("Quarantined %d questionable memories", quarantined)
        if return_ids:
            return {"count": quarantined, "memory_ids": quarantined_ids}
        return quarantined

    # -- consolidation / forgetting -----------------------------------------

    @staticmethod
    def _memory_quality_score(record: MemoryRecord) -> float:
        """Prefer records with stronger evidence when consolidating duplicates."""
        confidence = float(record.confidence or 0.0)
        return (
            confidence * 2.0
            + record.helpful_count * 3.0
            + record.retrieval_count * 0.1
            - record.dismissed_count * 2.0
            + min(len(record.content or "") / 1000.0, 1.0)
        )

    def _detect_semantic_duplicates(
        self,
        records: List[MemoryRecord],
        min_similarity: float,
        max_pairs: int,
        add_candidate_fn,
    ) -> int:
        """Detect semantic near-duplicates via embedding cosine similarity.

        Within each category, builds a normalized embedding matrix and
        computes pairwise cosine via numpy dot product. Pairs with cosine
        ≥ *min_similarity* are clustered greedily: the highest-quality
        record in each cluster is the keeper, the rest are candidates for
        quarantine with reason ``duplicate_semantic``.

        Safety invariants:
        - Within-category only (cross-category is OFF for v1).
        - Skips records with no embedding, expired, quarantined, or
          superseded (valid_to IS NOT NULL).
        - Skips records with content < 20 chars.
        - Never merges across user_scope or project_id.
        - No LLM calls; deterministic; no content rewriting.

        Returns the number of semantic duplicate candidates found.
        """
        if np is None or not records:
            return 0

        # Filter to records eligible for semantic dedup:
        # active, non-superseded, non-expired, has embedding, content ≥ 20 chars.
        eligible: List[MemoryRecord] = []
        for r in records:
            if r.embedding is None:
                continue
            if self._is_expired(r.expires_at):
                continue
            if r.valid_to is not None:
                continue
            if r.status != "active":
                continue
            if len((r.content or "").strip()) < 20:
                continue
            eligible.append(r)

        if len(eligible) < 2:
            return 0

        # Group by (category, user_scope, project_id) so we never merge
        # across scope boundaries. Use the record's user_scope attribute
        # (populated from the DB column) rather than r.payload — the
        # payload dict is the user's raw input and may be missing or stale
        # relative to the canonical column.
        groups: Dict[tuple, List[MemoryRecord]] = {}
        for r in eligible:
            key = (r.category, r.user_scope, r.project_id)
            groups.setdefault(key, []).append(r)

        total_candidates = 0
        pairs_checked = 0

        for key, group_records in groups.items():
            if len(group_records) < 2:
                continue
            # Pre-check: bail before the O(n²) matrix if this group alone
            # would blow the budget. Without this, a single oversized
            # category does its full dot product before the post-hoc
            # guard fires.
            group_pairs = len(group_records) * (len(group_records) - 1) // 2
            if pairs_checked + group_pairs > max_pairs:
                logger.debug(
                    "Semantic dedup max_pairs (%d) reached; skipping group "
                    "of %d records (%d pairs)",
                    max_pairs, len(group_records), group_pairs,
                )
                break
            # Build the embedding matrix for this group.
            try:
                emb_dim = len(group_records[0].embedding)
                mat = np.zeros((len(group_records), emb_dim), dtype=np.float32)
                for i, r in enumerate(group_records):
                    mat[i] = np.asarray(r.embedding, dtype=np.float32)
            except (ValueError, TypeError):
                continue

            # Normalize rows to unit length for cosine similarity.
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0  # avoid div-by-zero
            normed = mat / norms

            # Compute pairwise cosine = dot product of normalized vectors.
            # Only the upper triangle matters (symmetric matrix).
            sim_matrix = normed @ normed.T
            pairs_checked += group_pairs

            # Find pairs above threshold using the upper triangle.
            n = len(group_records)
            # Build adjacency: which records are connected to which.
            # Use a simple union-find / connected-components approach.
            parent = list(range(n))

            def find(x: int) -> int:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(a: int, b: int) -> None:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb

            for i in range(n):
                for j in range(i + 1, n):
                    if sim_matrix[i, j] >= min_similarity:
                        union(i, j)

            # Group records by connected component.
            clusters: Dict[int, List[int]] = {}
            for i in range(n):
                root = find(i)
                clusters.setdefault(root, []).append(i)

            # For each cluster with > 1 member, pick keeper and quarantine rest.
            for root, members in clusters.items():
                if len(members) < 2:
                    continue
                # Sort by quality score descending, then recency, then content length.
                member_records = [group_records[i] for i in members]
                member_records.sort(
                    key=lambda r: (
                        -self._memory_quality_score(r),
                        r.created_at or "",
                        -len(r.content or ""),
                    ),
                )
                keeper = member_records[0]
                cluster_size = len(member_records)
                for dup in member_records[1:]:
                    # Encode keeper link in the reason for audit trail.
                    reason = f"duplicate_semantic:keeper={keeper.memory_id}"
                    add_candidate_fn(
                        dup, reason, keeper.memory_id,
                        cluster_size=cluster_size,
                        cosine=float(sim_matrix[
                            group_records.index(dup),
                            group_records.index(keeper),
                        ]),
                    )
                    total_candidates += 1

        return total_candidates

    def consolidate(
        self,
        *,
        dry_run: bool = True,
        max_actions: int = 25,
        min_age_days: int = 30,
        duplicate_min_similarity: float = 0.88,
        duplicate_semantic_max_pairs: int = 20000,
    ) -> Dict[str, Any]:
        """Preview or apply conservative, reversible memory maintenance.

        The operation never permanently deletes records. It quarantines only
        expired records, stale unused temporary records, lower-quality
        exact/containment duplicates, and semantic near-duplicates
        (embedding cosine ≥ ``duplicate_min_similarity``). Durable memories
        are not forgotten merely because they are old or rarely retrieved.

        Semantic dedup (P4.1):
        - Within-category only (cross-category OFF for v1).
        - Never merges across user_scope or project_id.
        - Never touches chain members (valid_to IS NOT NULL) or expired records.
        - Keeper stays byte-identical (no content fusion — that's P4.2).
        - Quarantine reason encodes the keeper link for audit:
          ``duplicate_semantic:keeper=mem-abc123``.
        - Everything reversible via ``memory_restore``.
        """
        max_actions = max(1, min(int(max_actions), 500))
        min_age_days = max(1, int(min_age_days))
        records = self._fetch_records(
            """SELECT * FROM memory_records
               WHERE COALESCE(status, 'active') = 'active'
                 AND valid_to IS NULL
                 AND (user_scope IS NULL OR user_scope = ?)""",
            [self.user_id],
        )
        now = datetime.now(timezone.utc)
        candidates: Dict[str, Dict[str, Any]] = {}

        def age_days(record: MemoryRecord) -> int | None:
            if not record.created_at:
                return None
            try:
                created = datetime.fromisoformat(record.created_at.replace("Z", "+00:00"))
                return max(0, (now - created).days)
            except Exception:
                return None

        def add_candidate(
            record: MemoryRecord,
            reason: str,
            keeper_id: str | None = None,
            cluster_size: int = 0,
            cosine: float = 0.0,
        ) -> None:
            if record.memory_id in candidates:
                return
            candidates[record.memory_id] = {
                "memory_id": record.memory_id,
                "category": record.category,
                "reason": reason,
                "keeper_id": keeper_id,
                "age_days": age_days(record),
                "retrieval_count": record.retrieval_count,
                "confidence": record.confidence,
                "cluster_size": cluster_size,
                "cosine": round(cosine, 4) if cosine else 0.0,
            }

        for record in records:
            if self._is_expired(record.expires_at):
                add_candidate(record, "expired")
                continue
            age = age_days(record)
            if (
                record.durability == "temporary"
                and age is not None
                and age >= min_age_days
                and record.retrieval_count == 0
                and record.helpful_count == 0
                and record.dismissed_count == 0
                and float(record.confidence or 0.0) <= 0.6
            ):
                add_candidate(record, "stale_unused_temporary")

        by_category: Dict[str, List[MemoryRecord]] = {}
        for record in records:
            by_category.setdefault(record.category, []).append(record)
        for category_records in by_category.values():
            for index, record in enumerate(category_records):
                content = (record.content or "").strip().casefold()
                if len(content) < 20:
                    continue
                for other in category_records[index + 1:]:
                    other_content = (other.content or "").strip().casefold()
                    if len(other_content) < 20:
                        continue
                    if content != other_content and content not in other_content and other_content not in content:
                        continue
                    if self._memory_quality_score(record) >= self._memory_quality_score(other):
                        duplicate, keeper = other, record
                    else:
                        duplicate, keeper = record, other
                    add_candidate(duplicate, "duplicate_containment", keeper.memory_id)

        # Semantic dedup (P4.1): embedding-similarity near-duplicate detection.
        semantic_count = self._detect_semantic_duplicates(
            records,
            min_similarity=duplicate_min_similarity,
            max_pairs=duplicate_semantic_max_pairs,
            add_candidate_fn=add_candidate,
        )

        priority = {
            "expired": 0,
            "duplicate_containment": 1,
            "duplicate_semantic": 1,
            "stale_unused_temporary": 2,
        }
        selected = sorted(
            candidates.values(),
            key=lambda item: (
                priority.get(item["reason"].split(":")[0], 9),
                item["memory_id"],
            ),
        )[:max_actions]
        quarantined = 0
        quarantined_ids: List[str] = []
        if not dry_run:
            for item in selected:
                # Use the full reason string (includes keeper link for semantic).
                reason = item["reason"]
                if self.quarantine_memory(item["memory_id"], reason):
                    quarantined += 1
                    quarantined_ids.append(item["memory_id"])

        # Expiry reporting (Spec 1): count expired and expiring-soon rows.
        # These are filtered from retrieval but never auto-deleted.
        expired_count = 0
        expiring_soon_count = 0
        now_iso = now.isoformat()
        soon_iso = (now + timedelta(days=7)).isoformat()
        try:
            expired_count = int(self.connection.execute(
                """SELECT COUNT(*) FROM memory_records
                   WHERE COALESCE(status, 'active') = 'active'
                     AND valid_to IS NULL
                     AND expires_at IS NOT NULL
                     AND expires_at <= ?""",
                [now_iso],
            ).fetchone()[0])
            expiring_soon_count = int(self.connection.execute(
                """SELECT COUNT(*) FROM memory_records
                   WHERE COALESCE(status, 'active') = 'active'
                     AND valid_to IS NULL
                     AND expires_at IS NOT NULL
                     AND expires_at > ?
                     AND expires_at <= ?""",
                [now_iso, soon_iso],
            ).fetchone()[0])
        except Exception as exc:
            logger.debug("Expiry count query failed: %s", exc)

        # Count candidates by reason for the report.
        reason_counts: Dict[str, int] = {}
        for item in selected:
            # Normalize reason: strip the keeper link suffix for counting.
            base_reason = item["reason"].split(":")[0]
            reason_counts[base_reason] = reason_counts.get(base_reason, 0) + 1

        return {
            "dry_run": bool(dry_run),
            "candidate_count": len(selected),
            "quarantined_count": quarantined,
            "quarantined_ids": quarantined_ids,
            "max_actions": max_actions,
            "min_age_days": min_age_days,
            "candidates": selected,
            "expired_count": expired_count,
            "expiring_soon_count": expiring_soon_count,
            "expired_revivable_count": expired_count,
            "semantic_duplicate_count": semantic_count,
            "reason_counts": reason_counts,
            "duplicate_min_similarity": duplicate_min_similarity,
        }

    # -- Spec 2: explain_retrieval (memory_why_not) --------------------------

    def explain_retrieval(
        self,
        query: str,
        expected_memory_id: str,
        *,
        top_k: int = 20,
        project_id: str | None = None,
    ) -> Dict[str, Any]:
        """Diagnose why a memory did not surface in retrieval.

        Deterministic, free (no LLM), strictly read-only. Runs a parallel
        diagnostic pass that does NOT touch the production pipeline:
        - suppress_retrieval=True on all searches
        - no writes, no quarantine, no consolidation
        - no reranker side-effects

        When *project_id* is provided, the diagnostic search is scoped to
        that project (matching the production path). If the expected memory
        belongs to a different project, a project_scope_mismatch reason is
        reported — one of the top-3 causes of "why didn't this surface".

        Returns a structured explanation with:
        - expected: the target memory (or None if not found)
        - found_in_results: whether it appeared in the top-k
        - rank: its rank if found (1-indexed), else None
        - top_results: the top-k results with scores
        - reasons: list of human-readable reason strings
        - diagnostics: per-stage scores (vector_sim, text_score, etc.)
        """
        # 1. Fetch the expected memory.
        expected_rows = self._fetch_records(
            """SELECT * FROM memory_records WHERE memory_id = ?""",
            [expected_memory_id],
        )
        if not expected_rows:
            return {
                "expected_memory_id": expected_memory_id,
                "expected": None,
                "found_in_results": False,
                "rank": None,
                "top_results": [],
                "reasons": ["memory_not_found: no record with this memory_id"],
                "diagnostics": {},
            }
        expected = expected_rows[0]

        # 2. Run a diagnostic search (suppress_retrieval=True, include_expired
        #    so we can see if expiry is the reason). Thread project_id so
        #    the diagnostic search matches the production scoping path.
        results = self._hybrid_search(
            query, limit=top_k, suppress_retrieval=True, include_expired=True,
            project_id=project_id,
        )

        # 3. Check if the expected memory is in the results.
        result_ids = [r.memory_id for r in results]
        found = expected_memory_id in result_ids
        rank = result_ids.index(expected_memory_id) + 1 if found else None

        # 4. Build the top_results summary.
        top_results = []
        for r in results[:top_k]:
            top_results.append({
                "memory_id": r.memory_id,
                "content": (r.content or "")[:120],
                "category": r.category,
                "similarity": round(r.similarity, 4) if r.similarity else 0.0,
                "raw_similarity": round(getattr(r, "raw_similarity", 0.0), 4),
            })

        # 5. Compute per-stage diagnostics for the expected memory.
        diagnostics: Dict[str, Any] = {}
        reasons: List[str] = []

        # Vector similarity vs the query — reuse DuckDB's list_cosine_similarity
        # so the diagnostic score matches the production ranking score exactly.
        # A hand-rolled Python cosine can diverge from DuckDB's if the stored
        # embedding was normalized differently at write time; for a diagnostic
        # tool, that discrepancy is worse than useless (the user sees 0.41 and
        # concludes "not low" when the pipeline scored it 0.38 and filtered it).
        vec_sim = None
        if self.embedder and hasattr(self.embedder, "embed") and expected.embedding:
            try:
                query_emb = self.embedder.embed(query, is_query=True)
                if query_emb and expected.embedding:
                    # Use the same string-cast + list_cosine_similarity path
                    # as _vector_search_raw so the score is identical to what
                    # the production pipeline computed.
                    vec_text = "[" + ",".join(
                        repr(float(x)) for x in query_emb
                    ) + "]"
                    row = self.connection.execute(
                        f"""SELECT list_cosine_similarity(
                                embedding, CAST(? AS DOUBLE[{len(query_emb)}])
                            ) AS sim
                            FROM memory_records WHERE memory_id = ?""",
                        [vec_text, expected_memory_id],
                    ).fetchone()
                    if row and row[0] is not None:
                        vec_sim = float(row[0])
                        diagnostics["vector_similarity"] = round(vec_sim, 4)
            except Exception as exc:
                diagnostics["vector_similarity_error"] = str(exc)

        # Text match score.
        words = [t for t in query.split() if len(t) > 2][:4]
        if words and expected.content:
            content_lower = expected.content.lower()
            matched = sum(1 for w in words if w.lower() in content_lower)
            text_score = matched / len(words) if words else 0.0
            diagnostics["text_match_score"] = round(text_score, 4)

        # Status check.
        status = getattr(expected, "status", "active") or "active"
        diagnostics["status"] = status
        if status != "active":
            reasons.append(f"status={status}: memory is not active (quarantined)")

        # Superseded check.
        if expected.valid_to is not None:
            reasons.append(
                f"superseded: valid_to={expected.valid_to}, "
                f"superseded_by={expected.superseded_by}"
            )
            diagnostics["superseded"] = True

        # Expiry check.
        if expected.expires_at:
            now_iso = self._now()
            diagnostics["expires_at"] = expected.expires_at
            if expected.expires_at <= now_iso:
                reasons.append(
                    f"expired: expires_at={expected.expires_at} is in the past"
                )

        # User scope check.
        if not self._matches_scope(expected.payload):
            reasons.append(
                f"scope_mismatch: memory user_scope does not match "
                f"current user_id={self.user_id}"
            )
            diagnostics["scope_mismatch"] = True

        # Project scope check (Spec 2 fix): if the expected memory is
        # project-scoped and the diagnostic query used a different project_id
        # (or None), that's a top-3 cause of "why didn't this surface".
        expected_project = getattr(expected, "project_id", None)
        diagnostics["memory_project_id"] = expected_project
        diagnostics["query_project_id"] = project_id
        if expected_project is not None and expected_project != project_id:
            reasons.append(
                f"project_scope_mismatch: memory belongs to project "
                f"'{expected_project}' but the query was scoped to "
                f"'{project_id or 'None (global)'}'"
            )
            diagnostics["project_scope_mismatch"] = True

        # Low similarity.
        if vec_sim is not None and vec_sim < 0.3:
            reasons.append(
                f"low_vector_similarity: {round(vec_sim, 4)} < 0.3 threshold"
            )
        if not found and not reasons:
            # Present and not expired, but not in top-k → ranked too low.
            if vec_sim is not None:
                reasons.append(
                    f"ranked_below_top_{top_k}: vector_sim={round(vec_sim, 4)} "
                    f"did not make the cutoff"
                )
            else:
                reasons.append(
                    f"ranked_below_top_{top_k}: not in top-{top_k} results"
                )

        if not reasons:
            reasons.append("found: memory is in the results (no issue detected)")

        return {
            "expected_memory_id": expected_memory_id,
            "expected": {
                "memory_id": expected.memory_id,
                "content": (expected.content or "")[:200],
                "category": expected.category,
                "status": status,
                "expires_at": expected.expires_at,
                "valid_to": expected.valid_to,
                "superseded_by": expected.superseded_by,
                "project_id": expected_project,
            },
            "found_in_results": found,
            "rank": rank,
            "top_results": top_results,
            "reasons": reasons,
            "diagnostics": diagnostics,
        }

    # -- system state KV (P4.2 distillation, future maintenance) -------------

    def get_state(self, key: str) -> str | None:
        """Read a value from the ``system_state`` KV table."""
        try:
            with self._lock:
                assert self.connection is not None
                row = self.connection.execute(
                    "SELECT value FROM system_state WHERE key = ?", [key],
                ).fetchone()
            return row[0] if row else None
        except Exception:
            return None

    def set_state(self, key: str, value: str) -> None:
        """Write a value to the ``system_state`` KV table (upsert)."""
        try:
            with self._lock:
                assert self.connection is not None
                self.connection.execute(
                    """INSERT INTO system_state (key, value) VALUES (?, ?)
                       ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
                    [key, value],
                )
        except Exception as exc:
            logger.debug("set_state(%s) failed: %s", key, exc)

    # -- lifecycle ------------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            conn = getattr(self, "connection", None)
            if conn is None:
                return
            try:
                conn.close()
            except Exception:
                pass
            self.connection = None
