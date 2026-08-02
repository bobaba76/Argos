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
import threading
import uuid
from datetime import datetime, timezone
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


class MemoryRecord:
    """In-memory representation of a stored memory row."""

    __slots__ = (
        "memory_id", "category", "content", "tags", "payload",
        "created_at", "updated_at", "expires_at", "embedding", "similarity",
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
                    memory_id   VARCHAR PRIMARY KEY,
                    category    VARCHAR,
                    content     VARCHAR,
                    tags        VARCHAR[],
                    payload     JSON,
                    created_at  VARCHAR,
                    updated_at  VARCHAR,
                    expires_at  VARCHAR,
                    embedding   DOUBLE[]
                );
            """)

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
            memory_id=row["memory_id"],
            category=row["category"],
            content=row["content"],
            tags=row["tags"] if row["tags"] is not None else [],
            payload=json.loads(row["payload"]) if row["payload"] else {},
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
            expires_at=row.get("expires_at"),
            embedding=row.get("embedding"),
            similarity=similarity,
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

    def _text_search(
        self, query: str, limit: int, excluded: set[str],
        category_filter: str | None = None,
    ) -> List[MemoryRecord]:
        tokens = [f"%{t}%" for t in query.split() if len(t) > 2][:4]
        if not tokens:
            return []
        conditions = " OR ".join(["content ILIKE ?" for _ in tokens])
        sql = f"SELECT * FROM memory_records WHERE ({conditions}) LIMIT 200"
        results = self._fetch_records(sql, tokens)
        out: List[MemoryRecord] = []
        for r in results:
            if excluded and r.category.lower() in excluded:
                continue
            if category_filter and r.category != category_filter:
                continue
            if self._matches_scope(r.payload) and not self._is_expired(r.expires_at):
                out.append(r)
        return out[:limit]

    def search(
        self,
        query: str,
        limit: int = 5,
        exclude_categories: List[str] | None = None,
        category_filter: str | None = None,
    ) -> List[MemoryRecord]:
        """Hybrid search: vector similarity if embeddings available, else text."""
        excluded = {c.lower() for c in (exclude_categories or [])}
        emb: List[float] = []
        if self.embedder and hasattr(self.embedder, "embed"):
            emb = self.embedder.embed(query)

        def _apply_filters(records: List[MemoryRecord]) -> List[MemoryRecord]:
            out: List[MemoryRecord] = []
            for r in records:
                if excluded and r.category.lower() in excluded:
                    continue
                if category_filter and r.category != category_filter:
                    continue
                if not self._matches_scope(r.payload):
                    continue
                if self._is_expired(r.expires_at):
                    continue
                out.append(r)
            return out[:limit]

        if emb:
            sql = """
                SELECT *, list_cosine_similarity(embedding, ?::DOUBLE[]) AS sim
                FROM memory_records
                WHERE embedding IS NOT NULL
                ORDER BY sim DESC
                LIMIT ?
            """
            try:
                results = self._fetch_records(sql, [emb, limit * 4], sim_col="sim")
                filtered = _apply_filters(results)
                if filtered:
                    return filtered
                # Vector search returned nothing after filtering — try text.
                return self._text_search(query, limit, excluded, category_filter)
            except Exception as exc:
                if not self._is_vector_search_unavailable(exc):
                    logger.warning("Vector search error: %s", exc)
                return self._text_search(query, limit, excluded, category_filter)
        else:
            return self._text_search(query, limit, excluded, category_filter)

    # -- write operations -----------------------------------------------------

    def _content_exists(self, content: str, category: str) -> bool:
        """Check if a very similar content already exists (dedup)."""
        with self._lock:
            assert self.connection is not None
            # Exact match check.
            result = self.connection.execute(
                "SELECT COUNT(*) FROM memory_records WHERE content = ? AND category = ?",
                [content, category],
            ).fetchone()
            if result and result[0] > 0:
                return True
            # Fuzzy: check if existing content contains or is contained by this content.
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
            return False

    def remember(
        self,
        category: str,
        content: str,
        tags: List[str] | None = None,
        payload: Dict[str, Any] | None = None,
        dedup: bool = True,
    ) -> MemoryRecord | None:
        """Insert a memory record. Returns None if deduped away."""
        if not content or not content.strip():
            return None
        if category not in VALID_CATEGORIES:
            logger.warning("Unknown category '%s', defaulting to context_note", category)
            category = "context_note"

        record_payload = dict(payload or {})
        record_payload.setdefault("user_scope", self.user_id)

        if dedup and self._content_exists(content, category):
            logger.debug("Deduped memory: %s", content[:60])
            return None

        memory_id = f"mem-{uuid.uuid4().hex}"
        now = self._now()
        emb: List[float] = []
        if self.embedder and hasattr(self.embedder, "embed"):
            emb = self.embedder.embed(content)

        sql = """
            INSERT INTO memory_records
                (memory_id, category, content, tags, payload, created_at, updated_at, expires_at, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._lock:
            assert self.connection is not None
            self.connection.execute(sql, [
                memory_id, category, content, tags or [],
                json.dumps(record_payload), now, now,
                record_payload.get("expires_at"),
                emb if emb else None,
            ])
        fetched = self._fetch_records(
            "SELECT * FROM memory_records WHERE memory_id = ?", [memory_id]
        )
        return fetched[0] if fetched else None

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
        sql = "SELECT * FROM memory_records ORDER BY created_at DESC LIMIT ?"
        results = self._fetch_records(sql, [limit])
        return [r for r in results if self._matches_scope(r.payload) and not self._is_expired(r.expires_at)]

    def list_by_category(self, category: str, limit: int = 50) -> List[MemoryRecord]:
        sql = "SELECT * FROM memory_records WHERE category = ? ORDER BY created_at DESC LIMIT ?"
        results = self._fetch_records(sql, [category, limit])
        return [r for r in results if self._matches_scope(r.payload) and not self._is_expired(r.expires_at)]

    def list_memories(
        self, category: str | None = None, limit: int = 100
    ) -> List[MemoryRecord]:
        if category:
            return self.list_by_category(category, limit)
        return self.list_recent(limit)

    def count(self) -> int:
        with self._lock:
            assert self.connection is not None
            result = self.connection.execute("SELECT COUNT(*) FROM memory_records").fetchone()
            return result[0] if result else 0

    # -- junk cleanup ---------------------------------------------------------

    def cleanup_junk(self) -> int:
        """Identify and delete low-quality memories.

        Removes:
        - Memories with content shorter than 10 chars
        - Memories with agent-speak fragments
        - Memories with unmatched parentheses (truncated fragments)
        - Near-duplicate memories (same category, >80% content overlap)

        Returns the number of memories deleted.
        """
        from .extractor import _is_junk

        # Fetch all memories.
        all_records = self._fetch_records(
            "SELECT memory_id, category, content, tags, payload, created_at "
            "FROM memory_records"
        )

        to_delete: List[str] = []
        seen_content: Dict[str, str] = {}  # normalized content -> memory_id

        for rec in all_records:
            fact = {
                "category": rec.category,
                "content": rec.content,
                "tags": rec.tags,
                "payload": rec.payload,
            }

            # Check junk filter.
            if _is_junk(fact):
                to_delete.append(rec.memory_id)
                continue

            # Check for near-duplicates (same category, very similar content).
            normalized = rec.content.lower().strip()
            # Use first 60 chars as a fingerprint for dedup.
            fingerprint = normalized[:60]
            if fingerprint in seen_content and rec.category == all_records[0].category:
                # Keep the longer one, delete the shorter.
                existing_id = seen_content[fingerprint]
                # Find existing content length.
                existing_rec = next(
                    (r for r in all_records if r.memory_id == existing_id), None
                )
                if existing_rec and len(rec.content) > len(existing_rec.content):
                    # New one is longer — delete the old one, keep this one.
                    if existing_id not in to_delete:
                        to_delete.append(existing_id)
                    seen_content[fingerprint] = rec.memory_id
                else:
                    # Old one is longer (or equal) — delete this one.
                    to_delete.append(rec.memory_id)
            else:
                seen_content[fingerprint] = rec.memory_id

        # Delete junk memories.
        deleted = 0
        for mid in to_delete:
            if self.delete_memory(mid):
                deleted += 1

        if deleted:
            logger.info("Cleaned up %d junk memories", deleted)
        return deleted

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
