"""DuckDB storage layer for hybrid memory.

Stores memory records with category, content, tags, JSON payload, timestamps,
and a DOUBLE[] embedding column for vector search via ``list_cosine_similarity``.
Falls back to ILIKE text search when embeddings are unavailable.

Categories (general-purpose — any topic):
  personal_fact  — stable things about the user (age, location, job, tools, traits)
  preference     — how the user likes things (tools, communication style, habits)
  insight        — self-observations, realizations, patterns noticed
  event          — notable events with date context (job changes, milestones)
  relationship   — people in the user's life and dynamics
  goal           — things the user is working toward
  context_note   — situational context that helps future conversations
"""
from __future__ import annotations

import json
import logging
import math
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb

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


class MemoryRecord:
    """In-memory representation of a stored memory row."""

    __slots__ = (
        "memory_id", "category", "content", "tags", "payload",
        "created_at", "updated_at", "expires_at", "embedding", "similarity",
        "status", "source", "confidence", "durability", "scope", "project_id",
        "retrieval_count", "last_retrieved_at", "helpful_count", "dismissed_count",
        "quarantine_reason", "quarantined_at",
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
        status: str = "active",
        source: str = "explicit",
        confidence: float | None = None,
        durability: str = "durable",
        scope: str = "profile",
        project_id: str | None = None,
        retrieval_count: int = 0,
        last_retrieved_at: str | None = None,
        helpful_count: int = 0,
        dismissed_count: int = 0,
        quarantine_reason: str | None = None,
        quarantined_at: str | None = None,
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
        self.status = status or "active"
        self.source = source or "explicit"
        self.confidence = confidence
        self.durability = durability or "durable"
        self.scope = scope or "profile"
        self.project_id = project_id
        self.retrieval_count = int(retrieval_count or 0)
        self.last_retrieved_at = last_retrieved_at
        self.helpful_count = int(helpful_count or 0)
        self.dismissed_count = int(dismissed_count or 0)
        self.quarantine_reason = quarantine_reason
        self.quarantined_at = quarantined_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "category": self.category,
            "content": self.content,
            "tags": self.tags,
            "payload": self.payload,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "similarity": round(self.similarity, 4),
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
        }


class DuckDBMemoryStore:
    """DuckDB-backed memory store with vector + text search."""

    def __init__(
        self,
        db_path: str | Path,
        user_id: str = "default_user",
        embedder=None,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.user_id = (user_id or "default_user").strip()
        self.embedder = embedder
        self._lock = threading.Lock()
        self.connection: Optional[duckdb.DuckDBPyConnection] = None
        self._connect()
        self._init_db()

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
                    retrieval_count    INTEGER DEFAULT 0,
                    last_retrieved_at  VARCHAR,
                    helpful_count      INTEGER DEFAULT 0,
                    dismissed_count    INTEGER DEFAULT 0,
                    quarantine_reason  VARCHAR,
                    quarantined_at     VARCHAR
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
            """)
            # Additive migration for databases created by earlier versions.
            columns = {
                "status": "VARCHAR DEFAULT 'active'",
                "source": "VARCHAR DEFAULT 'explicit'",
                "confidence": "DOUBLE DEFAULT 1.0",
                "durability": "VARCHAR DEFAULT 'durable'",
                "scope": "VARCHAR DEFAULT 'profile'",
                "project_id": "VARCHAR",
                "retrieval_count": "INTEGER DEFAULT 0",
                "last_retrieved_at": "VARCHAR",
                "helpful_count": "INTEGER DEFAULT 0",
                "dismissed_count": "INTEGER DEFAULT 0",
                "quarantine_reason": "VARCHAR",
                "quarantined_at": "VARCHAR",
            }
            candidate_columns = {
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

    # -- helpers --------------------------------------------------------------

    def set_user_scope(self, user_id: str | None) -> None:
        self.user_id = (user_id or "default_user").strip()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _is_expired(expires_at: str | None) -> bool:
        if not expires_at:
            return False
        try:
            exp = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            return exp <= datetime.now(timezone.utc)
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
            retrieval_count=row.get("retrieval_count", 0),
            last_retrieved_at=row.get("last_retrieved_at"),
            helpful_count=row.get("helpful_count", 0),
            dismissed_count=row.get("dismissed_count", 0),
            quarantine_reason=row.get("quarantine_reason"),
            quarantined_at=row.get("quarantined_at"),
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
    ) -> List[MemoryRecord]:
        """ILIKE text search. Returns filtered records ranked by token overlap.

        Does NOT record retrieval — the caller is responsible for that.
        Sets ``similarity`` to a text-match score (fraction of query tokens
        that matched the content) so downstream fusion has a comparable signal.
        """
        words = [t for t in query.split() if len(t) > 2][:4]
        if not words:
            return []
        tokens = [f"%{t}%" for t in words]
        conditions = " OR ".join(["content ILIKE ?" for _ in tokens])
        sql = (
            "SELECT * FROM memory_records "
            "WHERE COALESCE(status, 'active') = 'active' AND ("
            f"{conditions}) LIMIT 200"
        )
        results = self._fetch_records(sql, tokens)
        out: List[MemoryRecord] = []
        for r in results:
            if excluded and r.category.lower() in excluded:
                continue
            if category_filter and r.category != category_filter:
                continue
            if not self._matches_scope(r.payload) or self._is_expired(r.expires_at):
                continue
            # Text-match score: fraction of query tokens found in content.
            content_lower = (r.content or "").lower()
            matched = sum(1 for w in words if w.lower() in content_lower)
            r.similarity = matched / len(words) if words else 0.0
            out.append(r)
        # Sort by text-match score descending.
        out.sort(key=lambda r: r.similarity, reverse=True)
        return out

    def _vector_search_raw(
        self, emb: List[float], limit: int, excluded: set[str],
        category_filter: str | None = None,
    ) -> List[MemoryRecord]:
        """Vector similarity search. Returns filtered records ranked by cosine.

        Does NOT record retrieval — the caller is responsible for that.
        Raises on vector search errors so the caller can fall back.
        """
        sql = """
            SELECT *, list_cosine_similarity(embedding, ?::DOUBLE[]) AS sim
            FROM memory_records
            WHERE COALESCE(status, 'active') = 'active'
              AND embedding IS NOT NULL
            ORDER BY sim DESC
            LIMIT ?
        """
        results = self._fetch_records(sql, [emb, limit * 4], sim_col="sim")
        out: List[MemoryRecord] = []
        for r in results:
            if excluded and r.category.lower() in excluded:
                continue
            if category_filter and r.category != category_filter:
                continue
            if not self._matches_scope(r.payload) or self._is_expired(r.expires_at):
                continue
            out.append(r)
        return out

    # -- ranking: RRF + feedback + recency -----------------------------------

    _RRF_K = 60  # Standard Reciprocal Rank Fusion constant.

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

    @classmethod
    def _apply_feedback_and_recency(cls, records: List[MemoryRecord]) -> None:
        """Adjust ``similarity`` in-place using feedback, confidence, and recency.

        Adjustments are additive on a 0-1 scale:
          - helpful_count:  +0.03 per vote (boosts memories the user found useful)
          - dismissed_count: -0.05 per vote (penalises dismissed memories)
          - confidence:     +0.05 * (confidence - 0.5) (rewards high-confidence)
          - recency:        0 to +0.10 based on age (recent memories get a boost)
        """
        for r in records:
            adj = 0.0
            adj += 0.03 * r.helpful_count
            adj -= 0.05 * r.dismissed_count
            if r.confidence is not None:
                adj += 0.05 * (r.confidence - 0.5)
            adj += cls._recency_boost(r.created_at)
            r.similarity = max(0.0, r.similarity + adj)
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
                for memory_id in ids:
                    self.connection.execute(
                        """UPDATE memory_records
                           SET retrieval_count = COALESCE(retrieval_count, 0) + 1,
                               last_retrieved_at = ?
                           WHERE memory_id = ? AND COALESCE(status, 'active') = 'active'""",
                        [now, memory_id],
                    )
            for record in records:
                record.retrieval_count += 1
                record.last_retrieved_at = now
        except Exception as exc:
            # A read-only fallback connection must still be able to search.
            logger.debug("Could not record memory retrieval: %s", exc)

    def search(
        self,
        query: str,
        limit: int = 5,
        exclude_categories: List[str] | None = None,
        category_filter: str | None = None,
    ) -> List[MemoryRecord]:
        """Hybrid search: RRF-fused vector + text, with feedback and recency.

        When embeddings are available, runs vector and text search in
        parallel and fuses results via Reciprocal Rank Fusion.  When
        embeddings are unavailable, falls back to text-only search.  In
        both cases, the final ranking is adjusted by feedback signals
        (helpful/dismissed), confidence, and recency.
        """
        excluded = {c.lower() for c in (exclude_categories or [])}
        emb: List[float] = []
        if self.embedder and hasattr(self.embedder, "embed"):
            emb = self.embedder.embed(query, is_query=True)

        # Gather candidate results from both paths.
        vector_results: List[MemoryRecord] = []
        text_results: List[MemoryRecord] = self._text_search_raw(
            query, limit, excluded, category_filter
        )

        if emb:
            try:
                vector_results = self._vector_search_raw(
                    emb, limit, excluded, category_filter
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

        # Apply feedback weighting and recency boost, then truncate.
        self._apply_feedback_and_recency(fused)
        final = fused[:limit]
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
           "User is related to Entity-B" vs "Entity-B is the user's role".
        """
        with self._lock:
            assert self.connection is not None
            # Layer 1: exact match.
            result = self.connection.execute(
                "SELECT COUNT(*) FROM memory_records WHERE content = ? AND category = ?",
                [content, category],
            ).fetchone()
            if result and result[0] > 0:
                return True
            # Layer 2: substring containment (case-insensitive).
            result = self.connection.execute(
                "SELECT content FROM memory_records WHERE category = ? LIMIT 500",
                [category],
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
                        result = self.connection.execute(
                            """SELECT memory_id FROM memory_records
                               WHERE category = ? AND embedding IS NOT NULL
                                 AND list_cosine_similarity(embedding, ?::DOUBLE[]) > ?
                               LIMIT 1""",
                            [category, emb, self._DEDUP_SIMILARITY_THRESHOLD],
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
    ) -> MemoryRecord | None:
        """Insert a memory record. Returns None if deduped away."""
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

        if dedup and self._content_exists(content, category):
            logger.debug("Deduped memory: %s", content[:60])
            return None

        memory_id = f"mem-{uuid.uuid4().hex}"
        now = self._now()
        if not record_payload.get("expires_at") and durability == "temporary":
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
                 project_id, retrieval_count, helpful_count, dismissed_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, 0)
        """
        with self._lock:
            assert self.connection is not None
            self.connection.execute(sql, [
                memory_id, category, content, tags or [],
                json.dumps(record_payload), now, now,
                record_payload.get("expires_at"),
                emb if emb else None,
                status, source, confidence, durability, scope, project_id,
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
                       LIMIT 1""",
                    [category, content.strip()],
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
                    status, created_at, updated_at, evidence_text, evidence_role,
                    source_timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?)""",
                [
                    candidate_id, category, content.strip(), tags or [],
                    json.dumps(candidate_payload), source, normalized_confidence,
                    durability or "durable", scope or "profile", project_id,
                    session_id or "", now, now, evidence_text,
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
            "(json_extract_string(payload, '$.user_scope') IS NULL OR "
            "json_extract_string(payload, '$.user_scope') = ?)"
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
    ) -> dict | None:
        """Approve, reject, quarantine, or classify a pending proposal."""
        decision = decision.strip().lower()
        allowed = {
            "approved", "rejected", "quarantined", "reviewed_approved",
            "pending_user_confirmation",
        }
        if decision not in allowed:
            raise ValueError("invalid candidate review decision")
        candidates = self.list_candidates(candidate_id=candidate_id, limit=1)
        if not candidates:
            return None
        candidate = candidates[0]
        if candidate["status"] not in {"pending", "reviewed_approved", "pending_user_confirmation"}:
            return {"candidate": candidate, "memory": None}
        now = self._now()
        memory = None
        final_status = decision
        if decision == "approved":
            memory = self.remember(
                category=candidate["category"],
                content=candidate["content"],
                tags=candidate["tags"],
                payload=candidate["payload"],
                source=candidate["source"],
                confidence=candidate["confidence"],
                durability=candidate["durability"],
                scope=candidate["scope"],
                project_id=candidate["project_id"],
            )
            if memory is None:
                final_status = "deduplicated"
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
        return {
            "candidate": self.list_candidates(candidate_id=candidate_id, limit=1)[0],
            "memory": memory.to_dict() if memory else None,
        }

    def quarantine_memory(self, memory_id: str, reason: str) -> bool:
        """Hide a memory from retrieval without deleting its record."""
        now = self._now()
        with self._lock:
            assert self.connection is not None
            check = self.connection.execute(
                "SELECT COUNT(*) FROM memory_records WHERE memory_id = ?", [memory_id]
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
                "SELECT COUNT(*) FROM memory_records WHERE memory_id = ?", [memory_id]
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
                "SELECT COUNT(*) FROM memory_records WHERE memory_id = ?", [memory_id]
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
    ) -> MemoryRecord | None:
        """Update an existing memory by ID. Returns updated record or None."""
        existing = self._fetch_records(
            "SELECT * FROM memory_records WHERE memory_id = ?", [memory_id]
        )
        if not existing:
            return None
        rec = existing[0]
        now = self._now()
        new_content = content if content is not None else rec.content
        new_tags = tags if tags is not None else rec.tags
        new_payload = dict(rec.payload)
        if payload_updates:
            new_payload.update(payload_updates)
        new_emb: List[float] = []
        if content is not None and self.embedder and hasattr(self.embedder, "embed"):
            new_emb = self.embedder.embed(new_content)

        with self._lock:
            assert self.connection is not None
            if content is not None and new_emb:
                self.connection.execute(
                    """UPDATE memory_records
                       SET content = ?, tags = ?, payload = ?, updated_at = ?, embedding = ?
                       WHERE memory_id = ?""",
                    [new_content, new_tags, json.dumps(new_payload), now, new_emb, memory_id],
                )
            elif content is not None:
                self.connection.execute(
                    """UPDATE memory_records
                       SET content = ?, tags = ?, payload = ?, updated_at = ?
                       WHERE memory_id = ?""",
                    [new_content, new_tags, json.dumps(new_payload), now, memory_id],
                )
            else:
                self.connection.execute(
                    """UPDATE memory_records
                       SET tags = ?, payload = ?, updated_at = ?
                       WHERE memory_id = ?""",
                    [new_tags, json.dumps(new_payload), now, memory_id],
                )
        fetched = self._fetch_records(
            "SELECT * FROM memory_records WHERE memory_id = ?", [memory_id]
        )
        return fetched[0] if fetched else None

    def delete_memory(self, memory_id: str) -> bool:
        with self._lock:
            assert self.connection is not None
            # Check existence first — DuckDB's execute() return value for
            # DELETE is not a reliable indicator of whether rows were deleted.
            check = self.connection.execute(
                "SELECT COUNT(*) FROM memory_records WHERE memory_id = ?", [memory_id]
            ).fetchone()
            if not check or check[0] == 0:
                return False
            self.connection.execute(
                "DELETE FROM memory_records WHERE memory_id = ?", [memory_id]
            )
            return True

    # -- listing --------------------------------------------------------------

    def list_recent(self, limit: int = 10) -> List[MemoryRecord]:
        sql = "SELECT * FROM memory_records WHERE COALESCE(status, 'active') = 'active' ORDER BY created_at DESC LIMIT ?"
        results = self._fetch_records(sql, [limit])
        return [r for r in results if self._matches_scope(r.payload) and not self._is_expired(r.expires_at)]

    def list_by_category(self, category: str, limit: int = 50) -> List[MemoryRecord]:
        sql = "SELECT * FROM memory_records WHERE category = ? AND COALESCE(status, 'active') = 'active' ORDER BY created_at DESC LIMIT ?"
        results = self._fetch_records(sql, [category, limit])
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
        conditions = ["category = 'insight'", "COALESCE(status, 'active') = 'active'"]
        params: list = []
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
                "ORDER BY created_at DESC LIMIT ?"
            )
            results = self._fetch_records(sql_simple, [limit])
            if tags:
                tag_set = {t.lower() for t in tags}
                results = [r for r in results if tag_set & {t.lower() for t in (r.tags or [])}]
            if since:
                results = [r for r in results if r.created_at and r.created_at >= since]
            return results

    def count(self) -> int:
        with self._lock:
            assert self.connection is not None
            result = self.connection.execute("SELECT COUNT(*) FROM memory_records").fetchone()
            return result[0] if result else 0

    # -- junk cleanup ---------------------------------------------------------

    def cleanup_junk(self) -> int:
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
            "SELECT * FROM memory_records WHERE COALESCE(status, 'active') = 'active'"
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
        for memory_id, reason in to_quarantine.items():
            if self.quarantine_memory(memory_id, reason):
                quarantined += 1

        if quarantined:
            logger.info("Quarantined %d questionable memories", quarantined)
        return quarantined

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
