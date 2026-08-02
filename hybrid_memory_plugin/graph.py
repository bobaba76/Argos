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

    def query_graph(self, entity_id: str) -> List[Dict[str, Any]]:
        """Find all outgoing edges from entity_id."""
        query = """
        MATCH (a:Entity {id: $id})-[r:RelatesTo]->(b:Entity)
        RETURN a.id AS source, r.relation_type AS relation,
               b.id AS target, b.entity_type AS target_type
        """
        with self._shared_conn_lock:
            results = self.conn.execute(query, parameters={"id": entity_id})
        edges: List[Dict[str, Any]] = []
        while results.has_next():
            row = results.get_next()
            edges.append({
                "source": row[0], "relation": row[1],
                "target": row[2], "target_type": row[3],
            })
        return edges

    def search_graph(self, term: str) -> List[Dict[str, Any]]:
        """Bidirectional fuzzy search — edges where term appears in source or target."""
        term_lower = term.lower()
        query = """
        MATCH (a:Entity)-[r:RelatesTo]->(b:Entity)
        RETURN a.id AS source, r.relation_type AS relation, b.id AS target
        """
        with self._shared_conn_lock:
            results = self.conn.execute(query)
        edges: List[Dict[str, Any]] = []
        while results.has_next():
            row = results.get_next()
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

    def purge_junk_entities(self) -> int:
        """Delete graph nodes (and their edges) whose id is a stop word or nonsense.

        Returns the count of nodes deleted.
        """
        with self._shared_conn_lock:
            results = self.conn.execute("MATCH (n:Entity) RETURN n.id AS id")
        ids_to_delete: List[str] = []
        while results.has_next():
            row = results.get_next()
            node_id = str(row[0])
            first_word = node_id.split()[0].lower() if node_id.split() else ""
            is_junk = (
                first_word in self._JUNK_ENTITY_PREFIXES
                or len(node_id.strip()) <= 2
                or re.match(r'^e\d+$', node_id.strip())
            )
            if is_junk:
                ids_to_delete.append(node_id)
        deleted = 0
        for node_id in ids_to_delete:
            try:
                with self._shared_conn_lock:
                    self.conn.execute(
                        "MATCH (a:Entity {id: $id})-[r:RelatesTo]->(b:Entity) DELETE r",
                        parameters={"id": node_id},
                    )
                    self.conn.execute(
                        "MATCH (a:Entity)-[r:RelatesTo]->(b:Entity {id: $id}) DELETE r",
                        parameters={"id": node_id},
                    )
                    self.conn.execute(
                        "MATCH (n:Entity {id: $id}) DELETE n",
                        parameters={"id": node_id},
                    )
                deleted += 1
            except Exception:
                pass
        if deleted:
            self._flush()
        return deleted

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
