"""Maintenance mixin: aliases, listings, cleanup, dedup, consolidation and KV.

Extracted verbatim from store.py during the god-file split (behavior-
neutral: no renames, no fixes).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

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
                        "UPDATE memory_records SET embedding = ? "
                        "WHERE memory_id = ?"
                        " AND (user_scope IS NULL OR user_scope = ?)"
                        # SM5: add valid_to IS NULL guard so a record
                        # superseded between SELECT and UPDATE (TOCTOU)
                        # doesn't get its embedding written.
                        " AND valid_to IS NULL",
                        [vec, memory_id, self.user_id],
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
