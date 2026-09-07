"""Maintenance mixin: aliases, listings, cleanup, dedup, consolidation and KV.

Extracted verbatim from store.py during the god-file split (behavior-
neutral: no renames, no fixes).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List

try:
    from .store_common import MemoryRecord, np
except ImportError:  # store_maintenance.py imported as a top-level module
    from store_common import MemoryRecord, np
try:
    from .structural_loss import is_append_only
except ImportError:  # store_maintenance.py imported as a top-level module
    from structural_loss import is_append_only

logger = logging.getLogger(__name__)


class StoreMaintenanceMixin:
    """Maintenance and listing methods for DuckDBMemoryStore."""

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
        with self._state.lock:
            assert self.connection is not None
            self.connection.execute(
                """INSERT OR REPLACE INTO entity_aliases
                   (alias, canonical_entity, user_scope, created_at)
                   VALUES (?, ?, ?, ?)""",
                [alias, canonical.lower(), self.user_id, now],
            )
            self._state.alias_cache = None  # Invalidate cache on write

    def remove_alias(self, alias: str, canonical_entity: str | None = None) -> bool:
        """Remove an alias mapping. If canonical_entity is None, removes all
        mappings for that alias."""
        alias = alias.strip().lower()
        with self._state.lock:
            assert self.connection is not None
            if canonical_entity:
                canonical = canonical_entity.strip().lower()
                self.connection.execute(
                    """DELETE FROM entity_aliases
                       WHERE alias = ? AND canonical_entity = ? AND user_scope = ?""",
                    [alias, canonical, self.user_id],
                )
            else:
                self.connection.execute(
                    """DELETE FROM entity_aliases
                       WHERE alias = ? AND user_scope = ?""",
                    [alias, self.user_id],
                )
            self._state.alias_cache = None  # Invalidate cache on write
            return True

    def resolve_aliases(self, text: str) -> List[str]:
        """Given a text query, return canonical entity names for any aliases
        found in the text.

        Example: resolve_aliases("tell me about my wife") → ["Alex"]

        Uses a per-scope cache to avoid a full-table scan on every search
        query (issue #27). The cache is invalidated on add_alias /
        remove_alias.
        """
        if not text:
            return []
        text_lower = text.lower()
        with self._state.lock:
            assert self.connection is not None
            if self._state.alias_cache is None:
                rows = self.connection.execute(
                    """SELECT alias, canonical_entity FROM entity_aliases
                       WHERE user_scope = ?""",
                    [self.user_id],
                ).fetchall()
                self._state.alias_cache = [(r[0], r[1]) for r in rows]
            aliases = self._state.alias_cache
        canonicals: set[str] = set()
        for alias, canonical in aliases:
            if alias and alias in text_lower:
                canonicals.add(canonical)
        return sorted(canonicals)

    def list_aliases(self) -> List[Dict[str, str]]:
        """List all alias mappings for this user."""
        with self._state.lock:
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
        with self._state.lock:
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
            # DuckDB's list_contains is: list_contains(list, value).
            # Use OR of list_contains per tag.
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
        with self._state.lock:
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

        # SM3: add a LIMIT so a store with 10k+ active records doesn't
        # load everything into Python at every session end. 1000 is well
        # above the typical personal store (~1k records); larger stores
        # are processed across multiple session ends.
        _SM3_CLEANUP_LIMIT = 1000
        all_records = self._fetch_records(
            """SELECT * FROM memory_records
               WHERE COALESCE(status, 'active') = 'active'
                 AND valid_to IS NULL
                 AND (user_scope IS NULL OR user_scope = ?)
               LIMIT ?""",
            [self.user_id, _SM3_CLEANUP_LIMIT],
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

            # SM7: use a 120-char fingerprint (was 60) so records with
            # identical content after char 60 but different first 60 chars
            # are detected as near-duplicates.
            fingerprint = rec.content.lower().strip()[:120]
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
                # Skip THIS group but keep scanning the rest — `break` would
                # exit the whole groups loop so later categories are never
                # checked even when they're small enough to fit the budget.
                continue
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
                # Sort by quality score descending, then recency (parsed
                # timestamp — raw-string lexicographic order mis-orders
                # mixed ISO forms, e.g. "2026-08-30T00:00:00+00:00" sorts
                # before "2026-08-30T00:00:00+05:00" even though the +05:00
                # instant is earlier in UTC), then content length.
                # Unparseable/missing timestamps fall back to epoch 0 — the
                # oldest possible instant — so, in this ascending sort, they
                # sit before any real timestamp and win the recency tiebreak
                # exactly like the old `r.created_at or ""` (empty string)
                # did. Quality remains the primary key, so this only matters
                # between equal-quality duplicates.
                member_records = [group_records[i] for i in members]
                member_records.sort(
                    key=lambda r: (
                        -self._memory_quality_score(r),
                        self._parse_timestamp(r.created_at)
                        or datetime.fromtimestamp(0, tz=timezone.utc),
                        -len(r.content or ""),
                    ),
                )
                keeper = member_records[0]
                cluster_size = len(member_records)
                # SM6: build an index map once so we don't call
                # group_records.index() (O(n)) inside the loop.
                _index_map = {r.memory_id: i for i, r in enumerate(group_records)}
                for dup in member_records[1:]:
                    # Encode keeper link in the reason for audit trail.
                    reason = f"duplicate_semantic:keeper={keeper.memory_id}"
                    add_candidate_fn(
                        dup, reason, keeper.memory_id,
                        cluster_size=cluster_size,
                        cosine=float(sim_matrix[
                            _index_map[dup.memory_id],
                            _index_map[keeper.memory_id],
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
            # Append-only exemption (#42): outcome/decision-shaped records
            # are immutable and never quarantined by dedup. They record what
            # happened at a point in time and must survive consolidation.
            if is_append_only(record.category, record.payload):
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
        # SM4: cap the number of containment-dedup pairs per category so a
        # category with 500 records doesn't do 124,750 comparisons. The
        # semantic dedup already has a max_pairs guard; this matches it.
        _SM4_MAX_CONTAINMENT_PAIRS = 5000
        for category_records in by_category.values():
            _pair_count = 0
            for index, record in enumerate(category_records):
                content = (record.content or "").strip().casefold()
                if len(content) < 20:
                    continue
                for other in category_records[index + 1:]:
                    _pair_count += 1
                    if _pair_count > _SM4_MAX_CONTAINMENT_PAIRS:
                        break
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
                if _pair_count > _SM4_MAX_CONTAINMENT_PAIRS:
                    break

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
        # Scoped to this user — a multi-tenant store must not count other
        # tenants' expired records.
        expired_count = 0
        expiring_soon_count = 0
        now_iso = now.isoformat()
        soon_iso = (now + timedelta(days=7)).isoformat()
        try:
            with self._state.lock:
                assert self.connection is not None
                expired_count = int(self.connection.execute(
                    """SELECT COUNT(*) FROM memory_records
                       WHERE COALESCE(status, 'active') = 'active'
                         AND valid_to IS NULL
                         AND expires_at IS NOT NULL
                         AND expires_at <= ?
                         AND (user_scope IS NULL OR user_scope = ?)""",
                    [now_iso, self.user_id],
                ).fetchone()[0])
                expiring_soon_count = int(self.connection.execute(
                    """SELECT COUNT(*) FROM memory_records
                       WHERE COALESCE(status, 'active') = 'active'
                         AND valid_to IS NULL
                         AND expires_at IS NOT NULL
                         AND expires_at > ?
                         AND expires_at <= ?
                         AND (user_scope IS NULL OR user_scope = ?)""",
                    [now_iso, soon_iso, self.user_id],
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
        # SM1: filter by user_scope so a user can't access another tenant's
        # memory by passing its memory_id. Records with NULL user_scope are
        # global and visible to all users.
        expected_rows = self._fetch_records(
            """SELECT * FROM memory_records
               WHERE memory_id = ?
                 AND (user_scope IS NULL OR user_scope = ?)""",
            [expected_memory_id, self.user_id],
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

    # -- P5.1 (#6): memory lifecycle — archival, forgetting, rollups ----------

    def archive_stale_records(
        self,
        *,
        archive_after_days: int = 180,
        exempt_categories: list[str] | None = None,
    ) -> dict[str, Any]:
        """Archive active records older than *archive_after_days* with no
        retrievals/feedback. Archived = out of injection pool + default
        search, still retrievable via ``include_archived=True``.

        Exempt categories (default: facts, preferences, insights,
        relationships, goal) are never archived. Any update/feedback
        revives an archived record (see ``revive_record``).

        Deterministic, zero LLM. Returns a summary dict.
        """
        if exempt_categories is None:
            exempt_categories = ["personal_fact", "preference",
                                 "insight", "relationship",
                                 "goal"]
        exempt_set = {c.lower() for c in exempt_categories}
        now = datetime.now(timezone.utc)
        cutoff_iso = (now - timedelta(days=max(1, archive_after_days))).isoformat()
        with self._state.lock:
            assert self.connection is not None
            # Build the exempt clause dynamically.
            if exempt_set:
                placeholders = ", ".join(["?" for _ in exempt_set])
                exempt_clause = f" AND LOWER(category) NOT IN ({placeholders})"
                params: list = [cutoff_iso, self.user_id, *exempt_set]
            else:
                exempt_clause = ""
                params = [cutoff_iso, self.user_id]
            rows = self.connection.execute(
                f"""SELECT memory_id FROM memory_records
                   WHERE COALESCE(status, 'active') = 'active'
                     AND valid_to IS NULL
                     AND COALESCE(tier, 'active') = 'active'
                     AND created_at < ?
                     AND (user_scope IS NULL OR user_scope = ?)
                     AND retrieval_count = 0
                     AND helpful_count = 0
                     AND dismissed_count = 0
                     {exempt_clause}""",
                params,
            ).fetchall()
            archived_ids = [r[0] for r in rows]
            for memory_id in archived_ids:
                self.connection.execute(
                    "UPDATE memory_records SET tier = 'archived', updated_at = ? "
                    "WHERE memory_id = ? AND COALESCE(tier, 'active') = 'active'"
                    # SM8: add valid_to IS NULL guard so a record
                    # superseded between SELECT and UPDATE (TOCTOU)
                    # doesn't get archived despite being superseded.
                    " AND valid_to IS NULL",
                    [now.isoformat(), memory_id],
                )
        if archived_ids:
            logger.info("Archived %d stale records (P5.1)", len(archived_ids))
        return {
            "archived_count": len(archived_ids),
            "archived_ids": archived_ids,
            "archive_after_days": archive_after_days,
        }

    def revive_record(self, memory_id: str) -> bool:
        """Revive an archived record back to active tier.

        Called automatically on update_memory / feedback / explicit re-save.
        Also callable directly. Returns True if the record was archived
        and is now revived, False if it was already active or not found.
        """
        with self._state.lock:
            assert self.connection is not None
            row = self.connection.execute(
                "SELECT tier FROM memory_records WHERE memory_id = ?"
                " AND (user_scope IS NULL OR user_scope = ?)",
                [memory_id, self.user_id],
            ).fetchone()
            if not row:
                return False
            if (row[0] or "active") != "archived":
                return False
            self.connection.execute(
                "UPDATE memory_records SET tier = 'active', updated_at = ? "
                "WHERE memory_id = ? AND (user_scope IS NULL OR user_scope = ?)",
                [self._now(), memory_id, self.user_id],
            )
        logger.debug("Revived archived record %s", memory_id)
        return True

    def forget_stale_records(
        self,
        *,
        forget_after_days: int = 365,
        categories: list[str] | None = None,
    ) -> dict[str, Any]:
        """Auto-quarantine zero-value drift: context_note/event/goal older
        than *forget_after_days* with no retrievals/feedback.

        Reversible — quarantine, never delete. The user can restore via
        ``memory_restore``. Deterministic, zero LLM.
        """
        if categories is None:
            categories = ["context_note", "event", "goal"]
        cat_set = {c.lower() for c in categories}
        now = datetime.now(timezone.utc)
        cutoff_iso = (now - timedelta(days=max(1, forget_after_days))).isoformat()
        with self._state.lock:
            assert self.connection is not None
            placeholders = ", ".join(["?" for _ in cat_set])
            rows = self.connection.execute(
                f"""SELECT memory_id FROM memory_records
                   WHERE COALESCE(status, 'active') = 'active'
                     AND valid_to IS NULL
                     AND COALESCE(tier, 'active') = 'active'
                     AND created_at < ?
                     AND (user_scope IS NULL OR user_scope = ?)
                     AND retrieval_count = 0
                     AND helpful_count = 0
                     AND dismissed_count = 0
                     AND LOWER(category) IN ({placeholders})""",
                [cutoff_iso, self.user_id, *cat_set],
            ).fetchall()
            forgotten_ids = [r[0] for r in rows]
        quarantined = 0
        quarantined_ids: List[str] = []
        # SM10: keep per-ID quarantine_memory calls (preserves partial-
        # failure semantics — quarantine_memory may fail for a record that
        # was deleted between SELECT and quarantine). The per-call lock
        # acquisition is acceptable because forget_stale_records runs at
        # session end, not on the hot path.
        for memory_id in forgotten_ids:
            if self.quarantine_memory(memory_id, "forgotten: stale zero-value drift (P5.1)"):
                quarantined += 1
                quarantined_ids.append(memory_id)
        if quarantined:
            logger.info("Forgot %d stale records (P5.1 quarantine)", quarantined)
        return {
            "forgotten_count": quarantined,
            "forgotten_ids": quarantined_ids,
            "forget_after_days": forget_after_days,
        }

    def run_lifecycle_maintenance(
        self,
        *,
        archive_enabled: bool = False,
        archive_after_days: int = 180,
        archive_exempt_categories: list[str] | None = None,
        forget_enabled: bool = False,
        forget_after_days: int = 365,
    ) -> dict[str, Any]:
        """Run the lifecycle maintenance pass (archive + forget).

        Called from the session-end hook. Both phases are independently
        gated. Deterministic, zero LLM. Fail-soft: any error in one
        phase doesn't block the other.
        """
        report: dict[str, Any] = {}
        if archive_enabled:
            try:
                report["archive"] = self.archive_stale_records(
                    archive_after_days=archive_after_days,
                    exempt_categories=archive_exempt_categories,
                )
            except Exception as exc:
                logger.debug("Archive pass failed: %s", exc)
                report["archive"] = {"error": str(exc)}
        if forget_enabled:
            try:
                report["forget"] = self.forget_stale_records(
                    forget_after_days=forget_after_days,
                )
            except Exception as exc:
                logger.debug("Forget pass failed: %s", exc)
                report["forget"] = {"error": str(exc)}
        return report

    # -- POPIA retention (#293) ------------------------------------------------

    def enforce_retention_policies(
        self,
        policies: Dict[str, int],
        *,
        now: datetime | None = None,
        dry_run: bool = False,
        legal_hold_check: Callable[[Dict[str, Any]], bool] | None = None,
    ) -> Dict[str, Any]:
        """Expire records whose record-class retention period has passed.

        #293: per-class retention (e.g. {"context_note": 730} = chat
        notes kept 2 years). EXTENDS the existing TTL machinery — no new
        scheduler: enforcement stamps ``expires_at`` to the retention
        deadline (created_at + days) on current active records whose
        retention has passed; the standard expiry filter
        (``expires_at <= now``) then drops them from retrieval exactly
        like any other expired record. Deletion itself happens via the
        erase-request workflow (with a deletion receipt) — expiry is a
        visibility change, not a deletion.

        Deterministic + testable: pass an explicit *now* clock and the
        outcome is a pure function of (records, policies, now).

        Idempotent: a record whose expires_at is already <= the
        retention deadline is left untouched, so re-running produces the
        same state (no early expiry, no double-stamping).

        Legal hold: when *legal_hold_check* is provided and returns True
        for a record's row dict, the record is NOT expired (reported as
        ``held``). Absent by default — the blocker is opt-in.

        Scoped per user (``user_scope``) like every maintenance pass.

        Returns a report dict:
        {
          "dry_run": bool,
          "now": iso,
          "policies": {category: days},
          "expired_count": int,
          "held_count": int,
          "expired_ids": [...],
          "held_ids": [...],
          "per_class": {category: {"expired": n, "held": n}},
        }
        """
        now = now or datetime.now(timezone.utc)
        now_iso = now.isoformat()
        report: Dict[str, Any] = {
            "dry_run": bool(dry_run),
            "now": now_iso,
            "policies": {str(k): int(v) for k, v in (policies or {}).items()},
            "expired_count": 0,
            "held_count": 0,
            "expired_ids": [],
            "held_ids": [],
            "per_class": {},
        }
        if not policies:
            return report

        for category, days in policies.items():
            days = int(days)
            if days < 1:
                continue
            per_class = {"expired": 0, "held": 0}
            with self._state.lock:
                assert self.connection is not None
                rows = self.connection.execute(
                    """SELECT memory_id, content, category, created_at,
                              expires_at, client_scope, doc_class
                       FROM memory_records
                       WHERE COALESCE(status, 'active') = 'active'
                         AND valid_to IS NULL
                         AND LOWER(category) = LOWER(?)
                         AND created_at IS NOT NULL
                         AND (user_scope IS NULL OR user_scope = ?)""",
                    [str(category), self.user_id],
                ).fetchall()
            for (memory_id, content, cat, created_at, expires_at,
                 client_scope, doc_class) in rows:
                try:
                    created = datetime.fromisoformat(str(created_at))
                    # Defensive tz guard (#293 review): a naive created_at
                    # (no tzinfo) would make `deadline > now` raise
                    # TypeError against the aware clock — coerce to UTC
                    # rather than crash the pass or silently skip.
                    if created.tzinfo is None:
                        created = created.replace(tzinfo=timezone.utc)
                except (TypeError, ValueError):
                    continue  # unparseable clock — never guess
                deadline = created + timedelta(days=days)
                if deadline > now:
                    continue  # retention not yet past — leave it
                row_dict = {
                    "memory_id": memory_id,
                    "content": content,
                    "category": cat,
                    "created_at": created_at,
                    "client_scope": client_scope,
                    "doc_class": doc_class,
                }
                if legal_hold_check is not None:
                    try:
                        if legal_hold_check(row_dict):
                            per_class["held"] += 1
                            report["held_count"] += 1
                            report["held_ids"].append(memory_id)
                            continue
                    except Exception as exc:
                        # Fail-closed: a broken hold check must not
                        # expire held data — treat the record as held.
                        logger.warning(
                            "legal_hold_check errored for %s: %s",
                            memory_id, exc,
                        )
                        per_class["held"] += 1
                        report["held_count"] += 1
                        report["held_ids"].append(memory_id)
                        continue
                deadline_iso = deadline.isoformat()
                if expires_at is not None:
                    try:
                        if datetime.fromisoformat(str(expires_at)) <= deadline:
                            continue  # already expired earlier — idempotent
                    except (TypeError, ValueError):
                        pass
                if not dry_run:
                    with self._state.lock:
                        assert self.connection is not None
                        self.connection.execute(
                            """UPDATE memory_records
                               SET expires_at = ?, updated_at = ?
                               WHERE memory_id = ?
                                 AND (user_scope IS NULL OR user_scope = ?)
                                 AND valid_to IS NULL""",
                            [deadline_iso, now_iso, memory_id, self.user_id],
                        )
                per_class["expired"] += 1
                report["expired_count"] += 1
                report["expired_ids"].append(memory_id)
            report["per_class"][str(category)] = per_class

        if report["expired_count"] and not dry_run:
            logger.info(
                "Retention: expired %d record(s) past their class retention "
                "(%d held)",
                report["expired_count"], report["held_count"],
            )
        return report

    # -- POPIA erase-request workflow (#293) -----------------------------------

    def erase_subject(
        self,
        subject: str,
        *,
        mode: str = "preview",
        confirm: bool = False,
        categories: List[str] | None = None,
        client_scope: str | None = None,
        doc_class: str | None = None,
        namespace: str | None = None,
        requested_by: str | None = None,
        legal_hold_check: Callable[[Dict[str, Any]], bool] | None = None,
        graph=None,
        limit: int = 500,
    ) -> Dict[str, Any]:
        """Erase all records about *subject* — provable deletion (POPIA).

        #293: subject-scoped erase with a persisted, verifiable receipt.
        Built on the #289 pattern: preview (dry-run) BEFORE the
        destructive action, per-record report, explicit confirm gate
        (STRICT — only the literal boolean True confirms; the string
        "false" must NOT pass).

        Deletion semantics per record (reuses the existing tombstone
        machinery — no new deletion semantics):
        - deletion_tombstones row (content fingerprint) so a re-feed
          cannot resurrect the erased content.
        - memory_records row DELETEd, memory_evidence row DELETEd —
          inside ONE transaction with the receipt INSERT, so the proof
          and the deletion are atomic (a crash cannot produce a receipt
          for a record that still exists, or vice versa).
        - deletion_receipts row APPENDED (before/during the deletion,
          same transaction). Receipts are append-only: the erase flow
          matches memory_records only — it cannot touch the receipt log,
          so receipts survive the deletion they prove.
        - Graph mirror: when *graph* is provided, ``remove_memory`` is
          called for each erased id AFTER the commit (fail-soft — the
          graph is a derived index; a mirror failure is logged and
          reported, never fatal).

        Legal hold: when *legal_hold_check* is provided and returns True
        for a record, the record is NOT deleted — reported as
        ``blocked_legal_hold`` with a clear reason. Absent by default
        (the blocker is opt-in; see the facade docs).

        Scoping: matches only the store's current ``user_scope`` (multi-
        tenant safe — an erase under one tenant cannot touch another);
        optional category/client_scope/doc_class/namespace narrow the
        subject match further.

        The receipt stores the content HASH (verifiable proof of WHAT
        was deleted) — never the content itself (POPIA minimality).

        Returns a report dict:
        {
          "mode", "subject", "request_id", "confirm",
          "matched_count", "total_matched", "has_more", "truncated_count",
          "erased_count", "blocked_count",
          "records": [ {memory_id, category, outcome, reason?,
                        receipt_id?} ... ],
          "wrote": bool,   # True only when apply erased >= 1 record
        }

        Truncation (#293): the batch processes at most *limit* records.
        ``total_matched``/``has_more``/``truncated_count`` surface any
        truncation; apply mode REFUSES a truncated batch (never a
        partial erasure that looks complete) — narrow the scope or pass
        an explicit higher limit.
        """
        import uuid as _uuid

        subject = str(subject or "").strip()
        if not subject:
            raise ValueError("erase subject is required")
        if mode not in {"preview", "apply"}:
            raise ValueError("mode must be 'preview' or 'apply'")
        # STRICT confirm gate (#289 alignment): bool("false") is True in
        # Python — only the literal boolean True may confirm a
        # destructive erase.
        if mode == "apply" and confirm is not True:
            raise ValueError(
                "erase apply requires confirm=True (preview first, "
                "then confirm)"
            )
        now = self._now()
        request_id = f"erase-{_uuid.uuid4().hex[:12]}"
        report: Dict[str, Any] = {
            "mode": mode,
            "subject": subject,
            "request_id": request_id,
            "confirm": confirm is True,
            "matched_count": 0,
            "erased_count": 0,
            "blocked_count": 0,
            "records": [],
            "wrote": False,
        }

        # Match: subject substring (case-insensitive), ALL versions and
        # states — erasure covers current, historical, archived and
        # quarantined rows alike.
        like = f"%{subject.lower()}%"
        clauses = [
            "LOWER(content) LIKE ?",
            "(user_scope IS NULL OR user_scope = ?)",
        ]
        params: List[Any] = [like, self.user_id]
        if categories:
            placeholders = ", ".join(["?" for _ in categories])
            clauses.append(
                f"LOWER(category) IN ({placeholders})"
            )
            params.extend(str(c).lower() for c in categories)
        if client_scope:
            clauses.append("client_scope = ?")
            params.append(client_scope)
        if doc_class:
            clauses.append("doc_class = ?")
            params.append(doc_class)
        if namespace:
            clauses.append("namespace = ?")
            params.append(namespace)
        with self._state.lock:
            assert self.connection is not None
            # Total match count (no LIMIT) — drives the truncation signal
            # so a POPIA request matching more than the batch limit can
            # never silently under-erase while its receipt log looks
            # complete.
            total_matched = int(self.connection.execute(
                f"""SELECT COUNT(*) FROM memory_records
                    WHERE {' AND '.join(clauses)}""",
                params,
            ).fetchone()[0])
            rows = self.connection.execute(
                f"""SELECT memory_id, content, category, client_scope,
                           doc_class
                    FROM memory_records
                    WHERE {' AND '.join(clauses)}
                    ORDER BY created_at ASC
                    LIMIT ?""",
                [*params, max(1, int(limit))],
            ).fetchall()
        report["matched_count"] = len(rows)
        report["total_matched"] = total_matched
        report["has_more"] = total_matched > len(rows)
        report["truncated_count"] = max(0, total_matched - len(rows))
        # Never silently under-erase: an apply batch that would leave
        # matched records unprocessed is refused with a clear error —
        # narrow the scope (categories/client_scope/doc_class/namespace)
        # or pass an explicit higher limit. Preview reports the
        # truncation so the operator sees it BEFORE confirming.
        if mode == "apply" and report["has_more"]:
            raise ValueError(
                f"erase request matches {total_matched} record(s) but the "
                f"batch limit is {limit} — refusing PARTIAL erasure. "
                f"Narrow the scope or pass an explicit higher limit."
            )

        for (memory_id, content, category, rec_scope, rec_doc_class) in rows:
            entry: Dict[str, Any] = {
                "memory_id": memory_id,
                "category": category,
            }
            # Legal hold: fail-closed — a broken check blocks the erase.
            if legal_hold_check is not None:
                try:
                    held = legal_hold_check({
                        "memory_id": memory_id,
                        "content": content,
                        "category": category,
                        "client_scope": rec_scope,
                        "doc_class": rec_doc_class,
                    })
                except Exception as exc:
                    held = True
                    entry["reason"] = f"legal_hold_check errored: {exc}"
                if held:
                    entry.setdefault("reason", "record is under legal hold")
                    entry["outcome"] = "blocked_legal_hold"
                    report["blocked_count"] += 1
                    report["records"].append(entry)
                    continue
            if mode == "preview":
                entry["outcome"] = "would_erase"
                report["records"].append(entry)
                continue

            # APPLY: receipt + tombstone + delete, one atomic transaction.
            receipt_id = f"rcpt-{_uuid.uuid4().hex}"
            content_hash = self._tombstone_hash(content)
            details = json.dumps({
                "subject": subject,
                "category": category,
                "client_scope": rec_scope,
                "doc_class": rec_doc_class,
            })
            with self._state.lock:
                assert self.connection is not None
                self._tx_begin()
                try:
                    # Receipt FIRST (append-only log) — inside the same
                    # transaction as the deletion so the proof and the
                    # erasure commit atomically.
                    self.connection.execute(
                        """INSERT INTO deletion_receipts
                           (receipt_id, request_id, subject, memory_id,
                            content_hash, category, user_scope, requested_by,
                            reason, outcome, details, created_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        [receipt_id, request_id, subject, memory_id,
                         content_hash, category, self.user_id,
                         requested_by or self.user_id,
                         "erase_request", "erased", details, now],
                    )
                    # Tombstone the content (re-feed blocked) — same
                    # machinery as delete_memory.
                    self._record_tombstone(
                        content, category,
                        reason=f"erase_request:{request_id}",
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
                    # #347: record the erase event in the same transaction.
                    self._record_event(
                        event_type="memory_erased",
                        entity_type="memory",
                        entity_key=memory_id,
                        content_hash=content_hash,
                        reason=f"erase_request:{request_id}",
                        refs={
                            "receipt_id": receipt_id,
                            "request_id": request_id,
                            "subject": subject,
                            "category": category,
                            "requested_by": requested_by or self.user_id,
                        },
                    )
                    self._tx_commit()
                except Exception:
                    self._tx_rollback()
                    raise
            entry["outcome"] = "erased"
            entry["receipt_id"] = receipt_id
            report["erased_count"] += 1
            report["records"].append(entry)
            # Graph mirror (after commit, fail-soft): the temporal graph
            # must not keep dangling rows for erased memories (#287).
            if graph is not None:
                try:
                    graph.remove_memory(memory_id)
                except Exception as exc:
                    entry["graph_mirror_error"] = str(exc)
                    logger.warning(
                        "Graph mirror failed for erased %s: %s",
                        memory_id, exc,
                    )

        report["wrote"] = mode == "apply" and report["erased_count"] > 0
        if report["erased_count"]:
            logger.info(
                "Erase request %s: erased %d record(s) for subject %r "
                "(%d blocked)",
                request_id, report["erased_count"], subject,
                report["blocked_count"],
            )
        return report

    def list_deletion_receipts(
        self,
        *,
        request_id: str | None = None,
        subject: str | None = None,
        memory_id: str | None = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Query the append-only deletion-receipt log (#293).

        Receipts survive the deletion they prove and are queryable by
        request_id (one erase batch), subject, or memory_id. Scoped per
        user_scope.
        """
        clauses = ["(user_scope IS NULL OR user_scope = ?)"]
        params: List[Any] = [self.user_id]
        if request_id:
            clauses.append("request_id = ?")
            params.append(request_id)
        if subject:
            clauses.append("subject = ?")
            params.append(subject)
        if memory_id:
            clauses.append("memory_id = ?")
            params.append(memory_id)
        with self._state.lock:
            assert self.connection is not None
            rows = self.connection.execute(
                f"""SELECT receipt_id, request_id, subject, memory_id,
                           content_hash, category, user_scope, requested_by,
                           reason, outcome, details, created_at
                    FROM deletion_receipts
                    WHERE {' AND '.join(clauses)}
                    ORDER BY created_at DESC
                    LIMIT ?""",
                [*params, max(1, int(limit))],
            ).fetchall()
        receipts: List[Dict[str, Any]] = []
        for (rid, req_id, subj, mid, chash, cat, scope, req_by, reason,
             outcome, details, created_at) in rows:
            try:
                parsed_details = json.loads(details) if details else {}
            except (TypeError, ValueError):
                parsed_details = {}
            receipts.append({
                "receipt_id": rid,
                "request_id": req_id,
                "subject": subj,
                "memory_id": mid,
                "content_hash": chash,
                "category": cat,
                "user_scope": scope,
                "requested_by": req_by,
                "reason": reason,
                "outcome": outcome,
                "details": parsed_details,
                "created_at": created_at,
            })
        return receipts

    def verify_erase_receipt(
        self,
        receipt_id: str,
    ) -> Dict[str, Any]:
        """Verify a deletion receipt against the live store (#293).

        Provable end-to-end: the receipt is valid iff (a) it exists in
        the append-only log, (b) the record it names is really gone from
        memory_records, and (c) the content is tombstoned (re-feed
        blocked). Returns a dict with ``valid`` + per-check details.
        """
        receipts = self.list_deletion_receipts(limit=10000)
        receipt = next(
            (r for r in receipts if r["receipt_id"] == receipt_id), None
        )
        if receipt is None:
            return {"valid": False, "reason": "receipt not found"}
        memory_id = receipt["memory_id"]
        with self._state.lock:
            assert self.connection is not None
            row = self.connection.execute(
                "SELECT memory_id FROM memory_records "
                "WHERE memory_id = ?",
                [memory_id],
            ).fetchone()
            # The receipt carries the deleted content's fingerprint —
            # verify the tombstone directly against it (the record is
            # gone, so its content cannot be re-hashed).
            tomb = self.connection.execute(
                "SELECT reason FROM deletion_tombstones "
                "WHERE content_hash = ? AND category = ?"
                " AND (user_scope IS NULL OR user_scope = ?)",
                [receipt["content_hash"], receipt["category"] or "",
                 self.user_id],
            ).fetchone()
        record_gone = row is None
        return {
            "valid": record_gone,
            "receipt": receipt,
            "record_gone": record_gone,
            "tombstoned": tomb is not None,
        }

    # -- portable export/import (#294) ----------------------------------------
    # Anti-lock-in / data sovereignty: a COMPLETE, versioned, deterministic
    # JSONL export of everything a tenant owns (records + provenance +
    # evidence + version chains + tombstones + deletion receipts +
    # candidates + rejection ledger + aliases) plus a human-readable
    # Markdown digest; and an idempotent re-import path that restores
    # provenance and respects tombstones (deleted stays deleted).

    # Tables included in a portable export, with their deterministic
    # ORDER BY and row type. The graph is deliberately NOT exported —
    # it is a derived index rebuilt by backfill_graph.py / rebuild_graph.py
    # (documented in export_portable's docstring).
    _PORTABLE_TABLES: Dict[str, tuple] = {
        "record": ("memory_records", "memory_id"),
        "evidence": ("memory_evidence", "memory_id"),
        "tombstone": ("deletion_tombstones", "content_hash, category"),
        "receipt": ("deletion_receipts", "created_at, receipt_id"),
        "candidate": ("memory_candidates", "candidate_id"),
        "rejection": ("rejection_ledger", "subject, predicate"),
        "alias": ("entity_aliases", "alias, canonical_entity"),
    }

    def _rows_as_dicts(self, sql: str, params: List[Any]) -> List[Dict[str, Any]]:
        """Run a query and return rows as plain dicts (column-name keyed)."""
        assert self.connection is not None
        cur = self.connection.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def export_portable(
        self,
        *,
        categories: List[str] | None = None,
        namespace: str | None = None,
        client_scope: str | None = None,
        doc_class: str | None = None,
    ) -> Dict[str, Any]:
        """Full portable export — complete, versioned, deterministic (#294).

        Anti-lock-in: the export carries EVERYTHING a user needs to take
        their memory elsewhere — all record fields (provenance origin/
        grounding, version chains valid_from/valid_to/superseded_by,
        embedding metadata, ACL metadata), evidence rows, pending
        candidates, the rejection ledger, entity aliases, deletion
        tombstones, and the append-only deletion receipts (#293).
        Nothing user-owned is left behind.

        NOT exported (deliberately, documented):
        - the Kuzu graph: a derived index rebuilt from the records via
          backfill_graph.py / rebuild_graph.py (its #287 valid windows
          are re-derived on re-index, so no temporal state is lost);
        - access_audit / system_state / file_catalog: operational
          telemetry, not user memory.

        Scope: rows are filtered to the store's current ``user_id``
        (multi-tenant safe — one tenant's export never includes
        another's) plus optional categories/namespace/client_scope/
        doc_class narrowing. Records are the only filtered table; the
        governance tables (tombstones/receipts/rejections/aliases/
        evidence/candidates) are user-scoped as stored.

        Deterministic: rows are ordered by primary key and serialized
        with sorted keys — the same store state produces byte-identical
        JSONL (the export timestamp lives only in the Markdown digest).

        Returns:
        {
          "header": {export_format, export_version, schema_version,
                     scope, counts},
          "rows": [ {type, data} ... ],   # deterministic order
          "jsonl": str,                   # the portable document
          "markdown": str,                # human-readable digest
        }
        """
        try:
            try:
                from .portable_export import (
                    make_header, order_rows, render_markdown, serialize_export,
                )
            except ImportError:
                from portable_export import (
                    make_header, order_rows, render_markdown, serialize_export,
                )
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("portable_export module unavailable") from exc

        scope: Dict[str, Any] = {"user_scope": self.user_id}
        if categories:
            scope["categories"] = [str(c) for c in categories]
        if namespace:
            scope["namespace"] = namespace
        if client_scope:
            scope["client_scope"] = client_scope
        if doc_class:
            scope["doc_class"] = doc_class

        # Record filter — always user-scoped; optional narrowing.
        rec_clauses = ["(user_scope IS NULL OR user_scope = ?)"]
        rec_params: List[Any] = [self.user_id]
        if categories:
            placeholders = ", ".join(["?" for _ in categories])
            rec_clauses.append(f"LOWER(category) IN ({placeholders})")
            rec_params.extend(str(c).lower() for c in categories)
        if namespace:
            rec_clauses.append("namespace = ?")
            rec_params.append(namespace)
        if client_scope:
            rec_clauses.append("client_scope = ?")
            rec_params.append(client_scope)
        if doc_class:
            rec_clauses.append("doc_class = ?")
            rec_params.append(doc_class)
        rec_where = " AND ".join(rec_clauses)

        rows: List[Dict[str, Any]] = []

        # Records: user-scoped + filtered, ordered by memory_id.
        with self._state.lock:
            assert self.connection is not None
            cur = self.connection.execute(
                f"SELECT * FROM memory_records WHERE {rec_where} "
                "ORDER BY memory_id",
                rec_params,
            )
            cols = [d[0] for d in cur.description]
            for row in cur.fetchall():
                rows.append({"type": "record", "data": dict(zip(cols, row))})

        # Governance/provenance tables: user-scoped as stored, no extra
        # narrowing (their rows belong to the exporting user by
        # construction; tombstones/receipts/rejections key on user_scope).
        for rtype, (table, order_by) in self._PORTABLE_TABLES.items():
            if rtype == "record":
                continue
            with self._state.lock:
                assert self.connection is not None
                cur = self.connection.execute(
                    f"SELECT * FROM {table} "
                    "WHERE (user_scope IS NULL OR user_scope = ?) "
                    f"ORDER BY {order_by}",
                    [self.user_id],
                )
                cols = [d[0] for d in cur.description]
                for row in cur.fetchall():
                    rows.append({"type": rtype, "data": dict(zip(cols, row))})

        rows = order_rows(rows)
        counts: Dict[str, int] = {}
        for r in rows:
            counts[r["type"]] = counts.get(r["type"], 0) + 1

        try:
            try:
                from .schema_migrations import get_schema_version
            except ImportError:
                from schema_migrations import get_schema_version
            with self._state.lock:
                schema_version = get_schema_version(self.connection)
        except Exception:
            schema_version = 0

        header = make_header(
            schema_version=schema_version, scope=scope, counts=counts,
        )
        return {
            "header": header,
            "rows": rows,
            "jsonl": serialize_export(header, rows),
            "markdown": render_markdown(header, rows, source=self.user_id),
        }

    def import_portable(
        self,
        data: str,
        *,
        mode: str = "preview",
        confirm: bool = False,
    ) -> Dict[str, Any]:
        """Re-import a portable export — idempotent, tombstone-aware (#294).

        Accepts the JSONL produced by :meth:`export_portable`, validates
        it (versioned format — unknown/newer versions are refused), and
        restores records WITH their provenance, evidence, version chains
        and governance state.

        #289 pattern: preview (dry-run) reports what WOULD be written —
        per-row outcomes and validation errors — and writes NOTHING;
        apply requires a STRICT confirm (only the literal boolean True;
        the string "false" must NOT pass). Validation errors abort the
        whole batch before any write (no partial silent write).

        Idempotent: rows are written by primary key (INSERT OR REPLACE),
        so replaying the same export is a no-op — no duplicates.

        Tombstone-aware (POPIA): a record whose content hash is
        tombstoned in the TARGET store is NOT imported (deleted stays
        deleted); the export's tombstone/receipt rows are restored so
        the provenance of deletions survives the round-trip.

        Rejection-aware (#39 ghost-defense): a record whose claim slot
        (subject, predicate, scope — see ``rejection_check``) is in the
        TARGET store's rejection ledger is NOT imported either, so
        replaying an export taken before a rejection decision cannot
        resurrect the rejected claim (or a paraphrase of it).

        Scope: every imported row is stamped with the TARGET store's
        current ``user_id`` — an import never writes into another
        tenant's scope, and imported data is owned by the importing
        cell.

        Returns a report dict:
        {
          "mode", "export_version", "schema_version",
          "total_rows", "valid_rows", "error_rows",
          "restored", "unchanged", "tombstone_blocked", "rejection_blocked",
          "rows": [ {line, type, identity, outcome} ... ],
          "errors": [ {line, errors: [...]} ... ],
          "wrote": bool,
        }
        """
        try:
            try:
                from .portable_export import parse_import
            except ImportError:
                from portable_export import parse_import
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("portable_export module unavailable") from exc

        if mode not in {"preview", "apply"}:
            raise ValueError("mode must be 'preview' or 'apply'")
        if mode == "apply" and confirm is not True:
            raise ValueError(
                "import apply requires confirm=True (preview first, "
                "then confirm)"
            )

        header, rows, parse_errors = parse_import(data)
        report: Dict[str, Any] = {
            "mode": mode,
            "export_version": header.get("export_version"),
            "schema_version": header.get("schema_version"),
            "total_rows": len(rows) + len(parse_errors),
            "valid_rows": len(rows),
            "error_rows": len(parse_errors),
            "restored": 0,
            "unchanged": 0,
            "tombstone_blocked": 0,
            "rejection_blocked": 0,
            "rows": [],
            "errors": parse_errors,
            "wrote": False,
        }
        if not rows:
            return report

        # Fail loud: ANY validation error aborts an apply batch before
        # any write (no partial silent skip).
        if mode == "apply" and parse_errors:
            raise ValueError(
                f"import validation failed for {len(parse_errors)} "
                f"row(s) — nothing written."
            )

        # Tombstone fingerprints for the target scope (deleted stays
        # deleted — a record erased after the export must not be
        # resurrected by replaying an older export).
        with self._state.lock:
            assert self.connection is not None
            tomb_rows = self.connection.execute(
                "SELECT content_hash, category FROM deletion_tombstones "
                "WHERE (user_scope IS NULL OR user_scope = ?)",
                [self.user_id],
            ).fetchall()
        tombstoned = {(h, c) for h, c in tomb_rows}

        def _record_payload(data: Dict[str, Any]) -> Dict[str, Any]:
            raw = data.get("payload")
            if isinstance(raw, dict):
                return raw
            if isinstance(raw, str) and raw:
                try:
                    parsed = json.loads(raw)
                except (TypeError, ValueError):
                    return {}
                return parsed if isinstance(parsed, dict) else {}
            return {}

        def _record_blocked(data: Dict[str, Any]) -> str | None:
            """Ghost-defense gate for an inbound record row (mirrors
            ``remember``): tombstone first, then the rejection ledger."""
            content = str(data.get("content") or "")
            category = str(data.get("category") or "")
            if (self._tombstone_hash(content), category) in tombstoned:
                return "tombstone_blocked"
            if self.rejection_check(category, _record_payload(data)):
                return "rejection_blocked"
            return None

        def _identity(rtype: str, data: Dict[str, Any]) -> str:
            if rtype == "record":
                return str(data.get("memory_id", ""))
            if rtype == "evidence":
                return str(data.get("memory_id", ""))
            if rtype == "tombstone":
                return f"{data.get('content_hash', '')}:{data.get('category', '')}"
            if rtype == "receipt":
                return str(data.get("receipt_id", ""))
            if rtype == "candidate":
                return str(data.get("candidate_id", ""))
            if rtype == "rejection":
                return f"{data.get('subject', '')}:{data.get('predicate', '')}"
            if rtype == "alias":
                return f"{data.get('alias', '')}:{data.get('canonical_entity', '')}"
            return ""

        # Per-row outcomes FIRST (read-only presence checks) so an apply
        # run reports "restored" for genuinely new rows and "unchanged"
        # for idempotent replays — then write.
        for row in rows:
            rtype, data = row["type"], row["data"]
            identity = _identity(rtype, data)
            entry = {
                "line": row.get("line"),
                "type": rtype,
                "identity": identity,
            }
            if rtype == "record":
                content = str(data.get("content") or "")
                blocked = _record_blocked(data)
                if blocked:
                    entry["outcome"] = blocked
                    report[blocked] += 1
                    report["rows"].append(entry)
                    continue
                exists = self._import_row_exists(rtype, data)
                same = False
                if exists:
                    with self._state.lock:
                        assert self.connection is not None
                        existing = self.connection.execute(
                            "SELECT content FROM memory_records "
                            "WHERE memory_id = ?",
                            [identity],
                        ).fetchone()
                    same = bool(existing and existing[0] == content)
                if exists and same:
                    entry["outcome"] = "unchanged"
                else:
                    entry["outcome"] = (
                        "would_restore" if mode == "preview" else "restored"
                    )
                report["rows"].append(entry)
                continue
            # Non-record types: presence check by identity.
            exists = self._import_row_exists(rtype, data)
            entry["outcome"] = (
                "unchanged" if exists else
                ("would_restore" if mode == "preview" else "restored")
            )
            report["rows"].append(entry)

        # Apply: one transaction for the whole batch (all-or-nothing).
        # Tombstone-/rejection-blocked records are skipped (deleted stays
        # deleted, rejected stays rejected).
        if mode == "apply":
            with self._state.lock:
                assert self.connection is not None
                self._tx_begin()
                try:
                    for row in rows:
                        if row["type"] == "record" and _record_blocked(row["data"]):
                            continue
                        self._import_row(row["type"], row["data"])
                        # #347: emit a mutation event for each imported row
                        # so the import path is auditable like every other
                        # write (#353: import boundary must not skip the
                        # event seam).
                        _rtype = row["type"]
                        _data = row["data"]
                        _event_type = {
                            "record": "memory_created",
                            "evidence": "memory_created",
                            "tombstone": "memory_deleted",
                            "receipt": "memory_erased",
                            # #347 round-2: imported candidates were never
                            # reviewed — map to candidate_created, not
                            # candidate_reviewed. Imported rejections are
                            # rejection ledger state, not review decisions —
                            # map to rejection_imported (not candidate_rejected
                            # which implies a review happened).
                            "candidate": "candidate_created",
                            "rejection": "rejection_imported",
                            "alias": "memory_created",
                        }.get(_rtype, "memory_created")
                        _entity_key = _identity(_rtype, _data)
                        self._record_event(
                            event_type=_event_type,
                            entity_type=_rtype,
                            entity_key=_entity_key,
                            reason=f"import_portable:{_rtype}",
                            refs={"type": _rtype, "imported": True},
                        )
                    self._tx_commit()
                except Exception:
                    self._tx_rollback()
                    raise

        # Final counts from the per-row outcomes.
        for entry in report["rows"]:
            if entry["outcome"] == "restored":
                report["restored"] += 1
            elif entry["outcome"] == "unchanged":
                report["unchanged"] += 1

        report["wrote"] = mode == "apply"
        return report

    def _import_row(self, rtype: str, data: Dict[str, Any]) -> None:
        """Write one import row by primary key (INSERT OR REPLACE).

        The row's user_scope is FORCED to the target store's current
        user — an import never writes into another tenant's scope.
        Columns unknown to the live schema are dropped (forward
        compatibility); missing columns take the schema defaults.
        """
        table = self._PORTABLE_TABLES[rtype][0]
        with self._state.lock:
            assert self.connection is not None
            cols_cur = self.connection.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = ?",
                [table],
            ).fetchall()
        live_cols = {r[0] for r in cols_cur}
        row = {k: v for k, v in data.items() if k in live_cols}
        row["user_scope"] = self.user_id
        if not row:
            return
        col_list = ", ".join(row.keys())
        ph = ", ".join(["?" for _ in row])
        self.connection.execute(
            f"INSERT OR REPLACE INTO {table} ({col_list}) "
            f"VALUES ({ph})",
            list(row.values()),
        )

    def _import_row_exists(self, rtype: str, data: Dict[str, Any]) -> bool:
        """Read-only presence check for preview/unchanged reporting."""
        table = self._PORTABLE_TABLES[rtype][0]
        if rtype in {"record", "evidence"}:
            key_col, key_val = "memory_id", data.get("memory_id")
        elif rtype == "receipt":
            key_col, key_val = "receipt_id", data.get("receipt_id")
        elif rtype == "candidate":
            key_col, key_val = "candidate_id", data.get("candidate_id")
        elif rtype == "tombstone":
            with self._state.lock:
                assert self.connection is not None
                row = self.connection.execute(
                    "SELECT 1 FROM deletion_tombstones "
                    "WHERE content_hash = ? AND category = ?"
                    " AND (user_scope IS NULL OR user_scope = ?)",
                    [data.get("content_hash"), data.get("category"),
                     self.user_id],
                ).fetchone()
            return row is not None
        elif rtype == "rejection":
            with self._state.lock:
                assert self.connection is not None
                row = self.connection.execute(
                    "SELECT 1 FROM rejection_ledger "
                    "WHERE subject = ? AND predicate = ?"
                    " AND (user_scope IS NULL OR user_scope = ?)",
                    [data.get("subject"), data.get("predicate"),
                     self.user_id],
                ).fetchone()
            return row is not None
        elif rtype == "alias":
            with self._state.lock:
                assert self.connection is not None
                row = self.connection.execute(
                    "SELECT 1 FROM entity_aliases "
                    "WHERE alias = ? AND canonical_entity = ?"
                    " AND (user_scope IS NULL OR user_scope = ?)",
                    [data.get("alias"), data.get("canonical_entity"),
                     self.user_id],
                ).fetchone()
            return row is not None
        else:
            return False
        with self._state.lock:
            assert self.connection is not None
            row = self.connection.execute(
                f"SELECT 1 FROM {table} WHERE {key_col} = ?",
                [key_val],
            ).fetchone()
        return row is not None

    # -- system state KV (P4.2 distillation, future maintenance) -------------

    # SM2/SM9: key allowlist for system_state. Only these keys may be
    # read or written via get_state/set_state. This prevents a caller from
    # corrupting internal state (e.g. overwriting distillation_last_run to
    # force re-runs) or reading internal state to infer system activity.
    _STATE_KEY_ALLOWLIST = frozenset({
        "distillation_last_run",
        "surfaced_confirmation_ids",
        "rollup_last_run",
        "distillation_cursor",
        "rollup_cursor",
        "last_session_end",
        "last_cleanup_junk",
        "compaction_last_run",
        "compaction_last_count",
        "retention_last_run",  # #293
    })

    def get_state(self, key: str) -> str | None:
        """Read a value from the ``system_state`` KV table.

        SM9: only keys in the allowlist may be read.
        """
        if key not in self._STATE_KEY_ALLOWLIST:
            logger.warning("get_state: key %r not in allowlist (SM9)", key)
            return None
        try:
            with self._state.lock:
                assert self.connection is not None
                row = self.connection.execute(
                    "SELECT value FROM system_state WHERE key = ?", [key],
                ).fetchone()
            return row[0] if row else None
        except Exception:
            return None

    def set_state(self, key: str, value: str) -> None:
        """Write a value to the ``system_state`` KV table (upsert).

        SM2: only keys in the allowlist may be written.
        """
        if key not in self._STATE_KEY_ALLOWLIST:
            logger.warning("set_state: key %r not in allowlist (SM2)", key)
            return
        try:
            with self._state.lock:
                assert self.connection is not None
                self.connection.execute(
                    """INSERT INTO system_state (key, value) VALUES (?, ?)
                       ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
                    [key, value],
                )
        except Exception as exc:
            logger.debug("set_state(%s) failed: %s", key, exc)

    # -- distillation data access (P4.2) ---------------------------------------
    # These encapsulate the SQL the distillation pass needs so it can run
    # against either a direct DuckDBMemoryStore or a SharedMemoryStore proxy
    # without reaching into _lock / connection / _fetch_records.

    def count_eligible_since(self, since: str | None) -> int:
        """Count active, non-superseded records created/updated since *since*.

        If *since* is None (never run), counts all eligible records.
        """
        conditions = [
            "COALESCE(status, 'active') = 'active'",
            "valid_to IS NULL",
            "(user_scope IS NULL OR user_scope = ?)",
            "embedding IS NOT NULL",
        ]
        params: list[Any] = [self.user_id]
        if since:
            conditions.append("(created_at > ? OR updated_at > ?)")
            params.extend([since, since])
        sql = (
            f"SELECT COUNT(*) FROM memory_records WHERE "
            + " AND ".join(conditions)
        )
        try:
            with self._state.lock:
                assert self.connection is not None
                row = self.connection.execute(sql, params).fetchone()
            return int(row[0]) if row else 0
        except Exception:
            return 0

    def load_eligible_records(
        self, since: str | None, limit: int,
    ) -> List[MemoryRecord]:
        """Load active, non-superseded records for distillation.

        If *since* is provided, only records created/updated after it.
        Falls back to most recent N if never run (since=None).
        """
        conditions = [
            "COALESCE(status, 'active') = 'active'",
            "valid_to IS NULL",
            "(user_scope IS NULL OR user_scope = ?)",
            "embedding IS NOT NULL",
        ]
        params: list[Any] = [self.user_id]
        if since:
            conditions.append("(created_at > ? OR updated_at > ?)")
            params.extend([since, since])
        sql = (
            "SELECT * FROM memory_records WHERE "
            + " AND ".join(conditions)
            + " ORDER BY created_at DESC LIMIT ?"
        )
        params.append(limit)
        return self._fetch_records(sql, params)

    def load_high_signal_records(self, limit: int = 20) -> List[MemoryRecord]:
        """Load records with feedback signals for the high-signal scan."""
        sql = (
            "SELECT * FROM memory_records WHERE "
            "COALESCE(status, 'active') = 'active' "
            "AND valid_to IS NULL "
            "AND (user_scope IS NULL OR user_scope = ?) "
            "AND embedding IS NOT NULL "
            "AND (helpful_count > 0 OR dismissed_count > 0) "
            "ORDER BY (helpful_count + dismissed_count) DESC, retrieval_count DESC "
            "LIMIT ?"
        )
        return self._fetch_records(sql, [self.user_id, limit])

    def load_rollup_candidates(self, limit: int) -> List[MemoryRecord]:
        """Load the oldest active low-retrieval records for rollup (RU2).

        Targets records beyond the rollup horizon: oldest first, active
        tier, no feedback. These are the records whose long-horizon
        patterns are most likely to benefit from compaction into profile
        summaries. Unlike ``load_eligible_records``, this does NOT require
        embeddings (rollup does not cluster) and orders ASC (oldest first).
        """
        sql = (
            "SELECT * FROM memory_records WHERE "
            "COALESCE(status, 'active') = 'active' "
            "AND valid_to IS NULL "
            "AND COALESCE(tier, 'active') = 'active' "
            "AND (user_scope IS NULL OR user_scope = ?) "
            "ORDER BY created_at ASC "
            "LIMIT ?"
        )
        return self._fetch_records(sql, [self.user_id, limit])

    def count_rollup_candidates_since(self, since: str | None) -> int:
        """Count active records created since *since* for the rollup novelty
        gate (RU3). Unlike ``count_eligible_since``, this does NOT require
        embeddings — rollup does not cluster.
        """
        conditions = [
            "COALESCE(status, 'active') = 'active'",
            "valid_to IS NULL",
            "COALESCE(tier, 'active') = 'active'",
            "(user_scope IS NULL OR user_scope = ?)",
        ]
        params: list[Any] = [self.user_id]
        if since:
            conditions.append("created_at > ?")
            params.append(since)
        sql = (
            "SELECT COUNT(*) FROM memory_records WHERE "
            + " AND ".join(conditions)
        )
        try:
            with self._state.lock:
                assert self.connection is not None
                row = self.connection.execute(sql, params).fetchone()
            return int(row[0]) if row else 0
        except Exception:
            return 0

    # -- lifecycle ------------------------------------------------------------

    def backfill_null_embeddings(self, batch_size: int = 64) -> int:
        """Re-embed memory records whose ``embedding`` column is NULL.

        Called after the embedder recovers from a transient load failure
        (issue #83): records written during the outage stored NULL
        embeddings and were therefore invisible to vector/graph-boost
        retrieval (``_search_memories`` scores them 0.0 and the SQL leg
        filters ``embedding IS NOT NULL``). This walks those rows and
        re-embeds their stored ``content`` with the current embedder.

        Returns the number of rows re-embedded. No-op (returns 0) when
        the embedder is unavailable or there are no NULL-embedding rows.
        Only current (``valid_to IS NULL``) records in the store's own
        ``user_scope`` are touched — historical versions and other
        tenants' rows are left alone.
        """
        embedder = getattr(self, "embedder", None)
        if embedder is None or not hasattr(embedder, "embed"):
            return 0
        # Confirm the embedder can actually produce vectors right now.
        if hasattr(embedder, "is_available") and not embedder.is_available:
            # Trigger a (re)load attempt — it may have recovered.
            if hasattr(embedder, "_ensure_loaded"):
                embedder._ensure_loaded()
            if hasattr(embedder, "is_available") and not embedder.is_available:
                return 0
        with self._state.lock:
            assert self.connection is not None
            rows = self.connection.execute(
                """SELECT memory_id, content FROM memory_records
                   WHERE embedding IS NULL
                     AND content IS NOT NULL
                     AND content <> ''
                     AND valid_to IS NULL
                     AND (user_scope IS NULL OR user_scope = ?)
                   ORDER BY created_at""",
                [self.user_id],
            ).fetchall()
        if not rows:
            return 0
        updated = 0
        # #286: stamp embedding provenance on backfilled rows, same as
        # reembed_store() and remember(). Without this, backfilled rows
        # have embeddings but NULL embedding_dim — invisible to the
        # _vector_search_raw pre-check (WHERE embedding_dim IS NOT NULL),
        # causing a stale-metadata trap on embedder-change during an
        # outage.
        embedder_id = getattr(embedder, "_model_name", None) or getattr(
            embedder, "model_name", None
        )
        from datetime import datetime, timezone
        backfill_ts = datetime.now(timezone.utc).isoformat()
        for i in range(0, len(rows), batch_size):
            chunk = rows[i:i + batch_size]
            texts = [r[1] for r in chunk]
            if hasattr(embedder, "embed_batch"):
                vecs = embedder.embed_batch(texts)
            else:
                vecs = [embedder.embed(t) for t in texts]
            with self._state.lock:
                assert self.connection is not None
                for (memory_id, _content), vec in zip(chunk, vecs):
                    if not vec:
                        continue
                    self.connection.execute(
                        "UPDATE memory_records SET embedding = ?, "
                        "embedding_dim = ?, embedder_id = ?, embedded_at = ? "
                        "WHERE memory_id = ?"
                        " AND (user_scope IS NULL OR user_scope = ?)"
                        # SM5: add valid_to IS NULL guard so a record
                        # superseded between SELECT and UPDATE (TOCTOU)
                        # doesn't get its embedding written.
                        " AND valid_to IS NULL",
                        [vec, len(vec), embedder_id, backfill_ts,
                         memory_id, self.user_id],
                    )
                    updated += 1
            logger.info(
                "Backfilled %d/%d NULL-embedding records (issue #83 recovery)",
                updated, len(rows),
            )
        return updated

    def close(self) -> None:
        with self._state.lock:
            conn = getattr(self, "connection", None)
            if conn is None:
                return
            try:
                conn.close()
            except Exception:
                pass
            self.connection = None
