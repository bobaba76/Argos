"""Kuzu graph layer for hybrid memory.

Stores entity nodes (people, concepts, items, concepts, events) and
RelatesTo edges with relation_type (e.g. "uses", "married_to",
"triggered_by", "resolved_by").  Graph queries find connections between
concepts that the vector store alone cannot surface.

Includes ``purge_junk_entities()`` to clean out stop-word / nonsense nodes
that heuristic extraction may create.
"""
from __future__ import annotations

import json
import logging
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _is_already_exists_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "already exists" in msg or "catalog exception" in msg


class KuzuGraphStore:
    """Kuzu-backed relationship graph with thread-safe access.

    Uses a process-level shared database connection so that multiple
    HybridMemoryProvider instances (Hermes creates one per agent/session)
    can all access the same Kuzu database without file lock conflicts.
    """

    # Class-level shared state: {db_path_str: (database, connection, lock, ref_count)}
    _shared: Dict[str, Any] = {}
    _shared_lock = threading.Lock()

    def __init__(self, db_dir: str | Path, user_id: str = "default_user") -> None:
        self.db_dir = Path(db_dir)
        self.db_dir.parent.mkdir(parents=True, exist_ok=True)
        self.user_id = (user_id or "default_user").strip()
        self._lock = threading.Lock()
        self.database = None
        self.conn = None
        self._db_key = str(self.db_dir.resolve())
        self._init_db()

    def _init_db(self) -> None:
        import kuzu

        with KuzuGraphStore._shared_lock:
            shared = KuzuGraphStore._shared.get(self._db_key)
            if shared is None:
                # First instance in this process — open the database.
                database = kuzu.Database(str(self.db_dir))
                conn = kuzu.Connection(database)
                lock = threading.Lock()
                ref_count = 0
                KuzuGraphStore._shared[self._db_key] = (database, conn, lock, ref_count)
                shared = KuzuGraphStore._shared[self._db_key]
                # Initialize schema on first open.
                try:
                    conn.execute(
                        "CREATE NODE TABLE Entity("
                        "id STRING, entity_type STRING, attributes STRING, "
                        "user_scope STRING, PRIMARY KEY (id))"
                    )
                except RuntimeError as e:
                    if not _is_already_exists_error(e):
                        logger.warning("Kuzu node init issue: %s", e)
                try:
                    conn.execute(
                        "CREATE REL TABLE RelatesTo("
                        "FROM Entity TO Entity, "
                        "relation_type STRING, attributes STRING, user_scope STRING)"
                    )
                except RuntimeError as e:
                    if not _is_already_exists_error(e):
                        logger.warning("Kuzu edge init issue: %s", e)

            # Reuse the shared connection.
            self.database, self.conn, self._shared_conn_lock, ref_count = shared
            KuzuGraphStore._shared[self._db_key] = (
                self.database, self.conn, self._shared_conn_lock, ref_count + 1
            )
        logger.debug("Kuzu graph connected (shared, ref_count=%d)", ref_count + 1)

    def set_user_scope(self, user_id: str | None) -> None:
        self.user_id = (user_id or "default_user").strip()

    def _flush(self) -> None:
        """Re-open the connection to force Kuzu to flush the WAL."""
        import kuzu

        with self._shared_conn_lock:
            if self.database is None:
                return
            self.conn = kuzu.Connection(self.database)
            # Update the shared connection so all instances see the fresh one.
            with KuzuGraphStore._shared_lock:
                shared = KuzuGraphStore._shared.get(self._db_key)
                if shared:
                    KuzuGraphStore._shared[self._db_key] = (
                        self.database, self.conn, self._shared_conn_lock, shared[3]
                    )

    # -- write operations -----------------------------------------------------

    def upsert_node(
        self,
        node_id: str,
        node_type: str,
        attributes: Dict[str, Any] | None = None,
        user_scope: str | None = None,
    ) -> None:
        attrs_json = json.dumps(attributes or {})
        scope = user_scope or self.user_id
        query = """
        MERGE (n:Entity {id: $id})
        ON MATCH SET n.entity_type = $type, n.attributes = $attrs, n.user_scope = $scope
        ON CREATE SET n.entity_type = $type, n.attributes = $attrs, n.user_scope = $scope
        """
        with self._shared_conn_lock:
            self.conn.execute(query, parameters={
                "id": node_id, "type": node_type, "attrs": attrs_json, "scope": scope,
            })

    def upsert_edge(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        attributes: Dict[str, Any] | None = None,
        user_scope: str | None = None,
    ) -> None:
        attrs_json = json.dumps(attributes or {})
        scope = user_scope or self.user_id
        query = """
        MATCH (a:Entity {id: $source}), (b:Entity {id: $target})
        MERGE (a)-[r:RelatesTo {relation_type: $rel_type}]->(b)
        ON MATCH SET r.attributes = $attrs, r.user_scope = $scope
        ON CREATE SET r.attributes = $attrs, r.user_scope = $scope
        """
        with self._shared_conn_lock:
            self.conn.execute(query, parameters={
                "source": source_id, "target": target_id,
                "rel_type": relation_type, "attrs": attrs_json, "scope": scope,
            })

    def add_relationship(
        self,
        source: str,
        source_type: str,
        relation: str,
        target: str,
        target_type: str,
        attributes: Dict[str, Any] | None = None,
    ) -> None:
        """Convenience: upsert both nodes then the edge."""
        self.upsert_node(source, source_type)
        self.upsert_node(target, target_type)
        self.upsert_edge(source, target, relation, attributes)

    # -- read operations ------------------------------------------------------

    @staticmethod
    def _visible_attributes(raw: Any) -> bool:
        try:
            attrs = json.loads(raw) if isinstance(raw, str) and raw else (raw or {})
        except Exception:
            attrs = {}
        return not isinstance(attrs, dict) or attrs.get("status") != "quarantined"

    def query_graph(self, entity_id: str) -> List[Dict[str, Any]]:
        """Find visible outgoing edges from entity_id."""
        query = """
        MATCH (a:Entity {id: $id})-[r:RelatesTo]->(b:Entity)
        RETURN a.id AS source, r.relation_type AS relation,
               b.id AS target, b.entity_type AS target_type,
               a.attributes AS source_attrs, b.attributes AS target_attrs,
               r.attributes AS relation_attrs
        """
        with self._shared_conn_lock:
            results = self.conn.execute(query, parameters={"id": entity_id})
        edges: List[Dict[str, Any]] = []
        while results.has_next():
            row = results.get_next()
            if not all(self._visible_attributes(value) for value in row[4:7]):
                continue
            edges.append({
                "source": row[0], "relation": row[1],
                "target": row[2], "target_type": row[3],
            })
        return edges

    def search_graph(self, term: str) -> List[Dict[str, Any]]:
        """Bidirectional fuzzy search over visible edges."""
        term_lower = term.lower()
        query = """
        MATCH (a:Entity)-[r:RelatesTo]->(b:Entity)
        RETURN a.id AS source, r.relation_type AS relation, b.id AS target,
               a.attributes AS source_attrs, b.attributes AS target_attrs,
               r.attributes AS relation_attrs
        """
        with self._shared_conn_lock:
            results = self.conn.execute(query)
        edges: List[Dict[str, Any]] = []
        while results.has_next():
            row = results.get_next()
            if not all(self._visible_attributes(value) for value in row[3:6]):
                continue
            src, rel, tgt = row[0], row[1], row[2]
            if term_lower in str(src).lower() or term_lower in str(tgt).lower():
                edges.append({"source": src, "relation": rel, "target": tgt})
        return edges

    def list_nodes(self, node_type: str | None = None, limit: int = 100) -> List[Dict[str, Any]]:
        if node_type:
            query = "MATCH (n:Entity) WHERE n.entity_type = $type RETURN n.id, n.entity_type, n.attributes LIMIT $limit"
            with self._shared_conn_lock:
                results = self.conn.execute(query, parameters={"type": node_type, "limit": limit})
        else:
            query = "MATCH (n:Entity) RETURN n.id, n.entity_type, n.attributes LIMIT $limit"
            with self._shared_conn_lock:
                results = self.conn.execute(query, parameters={"limit": limit})
        nodes: List[Dict[str, Any]] = []
        while results.has_next():
            row = results.get_next()
            attrs = {}
            try:
                attrs = json.loads(row[2]) if row[2] else {}
            except Exception:
                pass
            if not self._visible_attributes(attrs):
                continue
            nodes.append({"id": row[0], "entity_type": row[1], "attributes": attrs})
        return nodes

    def count_nodes(self) -> int:
        with self._shared_conn_lock:
            results = self.conn.execute("MATCH (n:Entity) RETURN COUNT(*)")
            row = results.get_next()
            return int(row[0]) if row else 0

    def count_edges(self) -> int:
        with self._shared_conn_lock:
            results = self.conn.execute("MATCH ()-[r:RelatesTo]->() RETURN COUNT(*)")
            row = results.get_next()
            return int(row[0]) if row else 0

    # -- maintenance ----------------------------------------------------------

    # Interrogative/stop words that are never valid entity ids.
    _JUNK_ENTITY_PREFIXES = frozenset({
        "who", "what", "where", "when", "why", "how", "which",
        "the", "this", "that", "a", "an", "is", "are",
        "top", "best", "show", "give", "list", "i", "my", "me",
        "it", "they", "he", "she", "we", "you", "and", "or", "but",
        "so", "if", "then", "just", "like", "was", "were", "been",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "can", "could", "should", "about", "into", "for", "with",
    })

    def _quarantine_node(self, node_id: str, reason: str) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        with self._shared_conn_lock:
            result = self.conn.execute(
                "MATCH (n:Entity {id: $id}) RETURN n.attributes",
                parameters={"id": node_id},
            )
            if not result.has_next():
                return False
            raw = result.get_next()[0]
            try:
                attrs = json.loads(raw) if raw else {}
            except Exception:
                attrs = {}
            attrs.update({
                "status": "quarantined",
                "quarantine_reason": reason,
                "quarantined_at": now,
            })
            self.conn.execute(
                "MATCH (n:Entity {id: $id}) SET n.attributes = $attrs",
                parameters={"id": node_id, "attrs": json.dumps(attrs)},
            )
        return True

    def _quarantine_edge(self, source: str, target: str, relation: str, reason: str) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        with self._shared_conn_lock:
            result = self.conn.execute(
                """MATCH (a:Entity {id: $source})-[r:RelatesTo]->(b:Entity {id: $target})
                   WHERE r.relation_type = $relation
                   RETURN r.attributes""",
                parameters={"source": source, "target": target, "relation": relation},
            )
            if not result.has_next():
                return False
            raw = result.get_next()[0]
            try:
                attrs = json.loads(raw) if raw else {}
            except Exception:
                attrs = {}
            attrs.update({
                "status": "quarantined",
                "quarantine_reason": reason,
                "quarantined_at": now,
            })
            self.conn.execute(
                """MATCH (a:Entity {id: $source})-[r:RelatesTo]->(b:Entity {id: $target})
                   WHERE r.relation_type = $relation
                   SET r.attributes = $attrs""",
                parameters={
                    "source": source, "target": target,
                    "relation": relation, "attrs": json.dumps(attrs),
                },
            )
        return True

    def quarantine_junk_entities(self) -> int:
        """Hide obviously malformed nodes/edges without deleting graph data."""
        with self._shared_conn_lock:
            node_results = self.conn.execute(
                "MATCH (n:Entity) RETURN n.id AS id, n.attributes AS attributes"
            )
            nodes = []
            while node_results.has_next():
                nodes.append(node_results.get_next())
            edge_results = self.conn.execute(
                """MATCH (a:Entity)-[r:RelatesTo]->(b:Entity)
                   RETURN a.id, r.relation_type, b.id, r.attributes"""
            )
            edges = []
            while edge_results.has_next():
                edges.append(edge_results.get_next())

        changed = 0
        for node_id, raw_attrs in nodes:
            if not self._visible_attributes(raw_attrs):
                continue
            node_id = str(node_id)
            first_word = node_id.split()[0].lower() if node_id.split() else ""
            if (
                first_word in self._JUNK_ENTITY_PREFIXES
                or len(node_id.strip()) <= 2
                or re.match(r'^e\d+$', node_id.strip())
            ) and self._quarantine_node(node_id, "junk entity review"):
                changed += 1

        for source, relation, target, raw_attrs in edges:
            if not self._visible_attributes(raw_attrs):
                continue
            if not re.match(r"^[A-Za-z0-9_]+$", str(relation)):
                if self._quarantine_edge(
                    str(source), str(target), str(relation), "malformed relation label"
                ):
                    changed += 1

        if changed:
            self._flush()
        return changed

    def purge_junk_entities(self) -> int:
        """Compatibility alias; graph maintenance is now reversible quarantine."""
        return self.quarantine_junk_entities()

    # -- lifecycle ------------------------------------------------------------

    def close(self) -> None:
        """Decrement the shared connection ref count.

        Only closes the actual database when the last instance disconnects.
        """
        with KuzuGraphStore._shared_lock:
            shared = KuzuGraphStore._shared.get(self._db_key)
            if shared is None:
                self.conn = None
                self.database = None
                return
            database, conn, lock, ref_count = shared
            ref_count -= 1
            if ref_count <= 0:
                # Last instance — close the database.
                KuzuGraphStore._shared.pop(self._db_key, None)
                self.conn = None
                self.database = None
                logger.debug("Kuzu graph closed (last instance, ref_count=0)")
            else:
                # Other instances still using it — just decrement.
                KuzuGraphStore._shared[self._db_key] = (
                    database, conn, lock, ref_count
                )
                self.conn = None
                self.database = None
                logger.debug("Kuzu graph close (ref_count=%d remaining)", ref_count)
