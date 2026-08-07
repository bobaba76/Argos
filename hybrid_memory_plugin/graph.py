"""Kuzu graph layer for hybrid memory.

Stores entity and memory nodes (people, concepts, items, concepts,
events, organizations, places, and source memories) and RelatesTo edges
with relation_type (e.g. "uses", "married_to", "works_at",
"insight_about", "mentions"). Graph indexing is category-agnostic:
shared entity nodes link facts, preferences, insights, events, goals, and
context notes. Graph queries find connections that vector search alone
cannot surface.

Includes deterministic graph-pattern extraction, bounded bidirectional
traversal, per-memory evidence tracking, and ``purge_junk_entities()`` to
clean out stop-word / nonsense nodes that heuristic extraction may create.
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


_GRAPH_GENERIC_TAGS = frozenset({
    "personal_fact", "preference", "insight", "event", "relationship",
    "goal", "context_note", "identity", "self_observation", "ongoing",
    "transition", "habit",
})
_GRAPH_STOP_ENTITIES = frozenset({
    "i", "me", "my", "user", "the", "this", "that", "it", "thing",
    "something", "someone", "people", "today", "tomorrow", "yesterday",
    "now", "then", "here", "there", "nothing", "everything",
})
_GRAPH_RELATIONSHIP_WORDS = frozenset({
    "wife", "husband", "partner", "boyfriend", "girlfriend", "ex",
    "boss", "advisor", "doctor", "teacher", "mentor", "friend",
    "colleague", "manager", "supervisor", "sibling", "brother", "sister",
    "parent", "mother", "father", "son", "daughter", "child",
})
_GRAPH_TECH_TERMS = frozenset({
    "python", "javascript", "typescript", "rust", "go", "java", "react",
    "docker", "kubernetes", "vim", "neovim", "vscode", "git", "github",
    "duckdb", "kuzu", "stripe", "linux", "windows", "macos", "postgres",
    "postgresql", "redis", "sqlite", "aws", "gcp", "azure", "openai",
})


def _clean_graph_entity(value: str, max_length: int = 100) -> str:
    """Normalize an extracted entity while preserving readable casing."""
    value = re.sub(r"\s+", " ", str(value or "")).strip()
    value = value.strip(" \t\r\n.,;:!?\"'`()[]{}")
    value = re.sub(r"^(?:a|an|the)\s+", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+(?:daily|weekly|every day|right now|at the moment)$", "", value, flags=re.IGNORECASE)
    if len(value) > max_length:
        value = value[:max_length].rsplit(" ", 1)[0].rstrip(" ,;:")
    return value.strip()


def _valid_graph_entity(value: str) -> bool:
    cleaned = _clean_graph_entity(value)
    if len(cleaned) < 3 or len(cleaned.split()) > 8:
        return False
    if cleaned.casefold() in _GRAPH_STOP_ENTITIES:
        return False
    if cleaned.split()[0].casefold() in _GRAPH_STOP_ENTITIES:
        return False
    return bool(re.search(r"[A-Za-z0-9]", cleaned))


def _slug_relation(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return value or "related_to"


def _infer_graph_type(entity: str, relation: str, default: str = "concept") -> str:
    """Infer a useful node type for generic extracted targets."""
    relation = relation.lower()
    entity_lower = entity.casefold()
    if relation in {"uses", "uses"}:
        return "item"
    if relation in {"has_attribute", "has_attribute"}:
        return "attribute"
    if relation in {"works_at", "employed_by"}:
        return "organization"
    if relation in {"lives_in", "from", "based_in"}:
        return "place"
    if relation in {"has_event", "experienced"}:
        return "event"
    if relation in {"has_goal", "working_toward"}:
        return "goal"
    if relation in {"has_tool", "uses", "prefers", "dislikes"}:
        if entity_lower in _GRAPH_TECH_TERMS:
            return "technology"
        return "tool" if relation in {"has_tool", "uses"} else default
    if relation.startswith("has_") and relation[4:] in _GRAPH_RELATIONSHIP_WORDS:
        return "person"
    return default


def extract_graph_relations(
    content: str,
    category: str = "context_note",
    tags: List[str] | None = None,
) -> List[Dict[str, Any]]:
    """Extract typed, user-centered graph relations from a memory.

    This is deliberately deterministic and dependency-free. It handles
    explicit relation patterns first, then adds topical tag/proper-noun
    links so *every* memory category can participate in the graph. Each
    returned item has ``source``, ``source_type``, ``relation``, ``target``,
    ``target_type``, and ``attributes`` keys.
    """
    if not content or not content.strip():
        return []

    category = str(category or "context_note").lower()
    text = re.sub(r"\s+", " ", content).strip()
    raw_tags = [str(tag).strip().lower() for tag in (tags or []) if str(tag).strip()]
    relations: List[Dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    def add(
        source: str,
        source_type: str,
        relation: str,
        target: str,
        target_type: str | None = None,
        evidence: str = "pattern",
    ) -> None:
        source_clean = _clean_graph_entity(source)
        target_clean = _clean_graph_entity(target)
        relation_clean = _slug_relation(relation)
        if not _valid_graph_entity(target_clean):
            return
        if not source_clean:
            return
        key = (source_clean.casefold(), relation_clean, target_clean.casefold())
        if key in seen:
            return
        seen.add(key)
        relations.append({
            "source": source_clean,
            "source_type": source_type,
            "relation": relation_clean,
            "target": target_clean,
            "target_type": target_type or _infer_graph_type(target_clean, relation_clean),
            "attributes": {
                "category": category,
                "extractor": "graph_patterns",
                "evidence": evidence,
            },
        })

    # Relationships: "Pat is my role", "my advisor is Entity-B",
    # "I am related to Entity-B", and equivalent generated memory wording.
    relationship = re.compile(
        r"\b([A-Za-z][A-Za-z0-9'_-]*(?:\s+[A-Za-z][A-Za-z0-9'_-]*)?)"
        r"\s+is\s+(?:my|the\s+user'?s?)\s+([a-z][a-z_-]*)\b",
        re.IGNORECASE,
    )
    for match in relationship.finditer(text):
        name, relation = match.group(1), match.group(2).lower()
        if name.casefold() not in _GRAPH_STOP_ENTITIES:
            add("user", "person", f"has_{relation}", name, "person")

    my_relation = re.compile(
        r"\b(?:my|the\s+user'?s?)\s+([a-z][a-z_-]*)\s+is\s+"
        r"([A-Za-z][A-Za-z0-9'_-]*(?:\s+[A-Za-z][A-Za-z0-9'_-]*)?)",
        re.IGNORECASE,
    )
    for match in my_relation.finditer(text):
        relation, name = match.group(1).lower(), match.group(2)
        if relation in _GRAPH_RELATIONSHIP_WORDS:
            add("user", "person", f"has_{relation}", name, "person")

    direct_relationship = re.compile(
        r"\b(?:i\s+am\s+|i\s+)?(married|dating|seeing|friends?)"
        r"\s+(?:to|with)\s+([A-Za-z][A-Za-z0-9'_-]*(?:\s+[A-Za-z][A-Za-z0-9'_-]*)?)",
        re.IGNORECASE,
    )
    for match in direct_relationship.finditer(text):
        add("user", "person", _slug_relation(f"{match.group(1)}_with"), match.group(2), "person")

    # Work and location.
    work = re.search(
        r"\b(?:i|user)\s+(?:work|works)\s+(?:at|for|with)\s+(.+?)(?:[.,;]|$)",
        text,
        re.IGNORECASE,
    )
    if work:
        workplace = re.split(
            r"\s+and\s+(?:(?:i|user)\s+)?(?:use|uses|take|takes|live|lives|"
            r"work|works|prefer|prefers)\b",
            work.group(1),
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        add("user", "person", "works_at", workplace, "organization")

    location = re.search(
        r"(?:\b(?:i|user)\s+|\band\s+)(?:live|lives)\s+in\s+(.+?)(?:[.,;]|$)",
        text,
        re.IGNORECASE,
    )
    if location:
        place = re.split(r"\s+and\s+(?=(?:i|user)\b)", location.group(1), maxsplit=1, flags=re.IGNORECASE)[0]
        add("user", "person", "lives_in", place, "place")

    # Ongoing usage: "User has been using Docker" should become a tool
    # relation rather than a generic "has been using Docker" concept.
    ongoing_usage = re.finditer(
        r"\b(?:i|user)\s+(?:have|has|['’]?ve)\s+been\s+"
        r"(using|working\s+with|taking)\s+(.+?)(?:[.,;]|$)",
        text,
        re.IGNORECASE,
    )
    for ongoing in ongoing_usage:
        verb, thing = ongoing.group(1).lower(), ongoing.group(2)
        thing = re.split(r"\s+(?:for|because|when|to help)\s+", thing, maxsplit=1, flags=re.IGNORECASE)[0]
        relation = "uses" if verb == "taking" else "uses"
        target_type = "item" if relation == "uses" else "technology"
        add("user", "person", relation, thing, target_type)

    # Attributes and item/tool usage.
    attributeis = re.search(
        r"\b(?:i|user)\s+(?:was\s+attributeed\s+with|has|have)\s+"
        r"(?:a\s+attributeis\s+of\s+)?(.+?)(?:[.,;]|$)",
        text,
        re.IGNORECASE,
    )
    attributeis_context = re.search(
        r"\b(?:attribute|condition|depression|anxiety|adhd|bipolar|ptsd|ocd)\b",
        text,
        re.IGNORECASE,
    )
    if attributeis and attributeis_context:
        condition = re.split(r"\s+(?:and|but|for)\s+", attributeis.group(1), maxsplit=1, flags=re.IGNORECASE)[0]
        add("user", "person", "has_attribute", condition, "attribute")

    usage_pattern = re.compile(
        r"(?:\b(?:i|user)\s+|\band\s+)(take|takes|use|uses|own|owns|have|has|"
        r"am\s+using|is\s+using)\s+(.+?)(?=[.,;]|\s+and\s+|$)",
        re.IGNORECASE,
    )
    for usage in usage_pattern.finditer(text):
        verb, thing = usage.group(1).lower(), usage.group(2)
        if verb in {"has", "have"} and thing.lower().startswith("been "):
            continue
        thing = re.split(r"\s+(?:for|because|when|to help)\s+", thing, maxsplit=1, flags=re.IGNORECASE)[0]
        if verb.startswith("take"):
            add("user", "person", "uses", thing, "item")
        elif verb.startswith("use") or "using" in verb:
            add("user", "person", "uses", thing, _infer_graph_type(_clean_graph_entity(thing), "uses", "technology"))
        else:
            add("user", "person", "has", thing, "concept")

    # Preferences, including explicit comparisons.
    preference = re.search(
        r"\b(?:i|user)\s+(prefer|prefers|like|likes|love|loves|hate|hates|"
        r"enjoy|enjoys)\s+(.+?)(?:\s+over\s+(.+?))?(?:[.,;]|$)",
        text,
        re.IGNORECASE,
    )
    if preference:
        verb, preferred, alternative = preference.groups()
        relation = "dislikes" if verb.lower().startswith(("hate",)) else "prefers"
        add("user", "person", relation, preferred, "concept")
        if alternative:
            add(_clean_graph_entity(preferred), "concept", "preferred_over", alternative, "concept", "comparison")

    # Transitions, goals, events, and insight/context topical relations.
    transition = re.search(
        r"\b(?:i|user)\s+(?:switched|moved|migrated|transitioned)\s+from\s+"
        r"(.+?)\s+to\s+(.+?)(?:[.,;]|$)", text, re.IGNORECASE,
    )
    if transition:
        old, new = transition.groups()
        add("user", "person", "moved_away_from", old, "concept")
        add("user", "person", "moved_to", new, "concept")

    goal = re.search(
        r"(?:user\s+goal:|\b(?:i|user)\s+(?:want|wants|plan|plans|hope|hopes)\s+to)\s+"
        r"(.+?)(?:[.,;]|$)", text, re.IGNORECASE,
    )
    if goal:
        add("user", "person", "working_toward", goal.group(1), "goal")

    event = re.search(
        r"(?:life\s+event:\s*user\s+|\b(?:i|user)\s+)"
        r"(started|began|stopped|quit|resumed|finished|completed|launched)\s+"
        r"(.+?)(?:[.,;]|$)", text, re.IGNORECASE,
    )
    if event:
        add("user", "person", "experienced", event.group(2), "event")

    if category == "insight" or re.search(r"\b(?:insight|realiz|self-observation)", text, re.IGNORECASE):
        insight_target = re.sub(r"^user\s+self-observation:\s*", "", text, flags=re.IGNORECASE)
        # Keep the complete insight as a concept only when it is short enough;
        # topical tags/proper nouns below handle larger insight text.
        if len(insight_target.split()) <= 8:
            add("user", "person", "noticed_pattern", insight_target, "insight")

    # Topical tags make all categories graph-addressable. Date/category tags
    # are metadata, not entities.
    tag_relation = {
        "insight": "insight_about",
        "goal": "working_toward",
        "preference": "interested_in",
        "event": "related_to",
        "relationship": "related_to",
        "personal_fact": "related_to",
        "context_note": "context_about",
    }.get(category, "related_to")
    for tag in raw_tags:
        if tag in _GRAPH_GENERIC_TAGS or re.fullmatch(r"\d{4}-\d{2}-\d{2}", tag):
            continue
        add("user", "person", tag_relation, tag, "concept", "tag")

    # Proper nouns provide lightweight entity discovery for categories that
    # have no explicit relation pattern (e.g. an insight mentioning Entity-B).
    proper_nouns = re.findall(
        r"\b[A-Z][A-Za-z0-9'_-]*(?:\s+[A-Z][A-Za-z0-9'_-]*)*\b", text
    )
    proper_relation = {
        "insight": "insight_about", "goal": "working_toward",
        "event": "related_to", "preference": "interested_in",
        "context_note": "context_about", "personal_fact": "related_to",
        "relationship": "related_to",
    }.get(category, "related_to")
    explicit_targets = {item["target"].casefold() for item in relations}
    for entity in proper_nouns:
        if entity.casefold() in _GRAPH_STOP_ENTITIES or entity.casefold() == "user":
            continue
        if entity.casefold() in explicit_targets:
            continue
        add("user", "person", proper_relation, entity, "concept", "proper_noun")

    return relations


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
        incoming = dict(attributes or {})
        scope = user_scope or self.user_id
        with self._shared_conn_lock:
            existing_result = self.conn.execute(
                "MATCH (n:Entity {id: $id}) RETURN n.attributes",
                parameters={"id": node_id},
            )
            existing: Dict[str, Any] = {}
            if existing_result.has_next():
                existing = self._parse_attributes(existing_result.get_next()[0])
            merged = dict(existing)
            merged.update(incoming)
            query = """
            MERGE (n:Entity {id: $id})
            ON MATCH SET n.entity_type = $type, n.attributes = $attrs, n.user_scope = $scope
            ON CREATE SET n.entity_type = $type, n.attributes = $attrs, n.user_scope = $scope
            """
            self.conn.execute(query, parameters={
                "id": node_id, "type": node_type,
                "attrs": json.dumps(merged), "scope": scope,
            })

    def upsert_edge(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        attributes: Dict[str, Any] | None = None,
        user_scope: str | None = None,
    ) -> None:
        """Create or update an edge while preserving multi-memory evidence."""
        incoming = dict(attributes or {})
        scope = user_scope or self.user_id
        with self._shared_conn_lock:
            existing_result = self.conn.execute(
                """MATCH (a:Entity {id: $source})-[r:RelatesTo]->(b:Entity {id: $target})
                   WHERE r.relation_type = $rel_type
                   RETURN r.attributes""",
                parameters={
                    "source": source_id, "target": target_id,
                    "rel_type": relation_type,
                },
            )
            existing: Dict[str, Any] = {}
            if existing_result.has_next():
                raw = existing_result.get_next()[0]
                try:
                    parsed = json.loads(raw) if raw else {}
                    if isinstance(parsed, dict):
                        existing = parsed
                except Exception:
                    existing = {}

            # Keep a compact evidence index so several memories can share
            # one semantic edge without overwriting one another.
            memory_id = incoming.get("memory_id")
            memory_ids = existing.get("memory_ids", [])
            if not isinstance(memory_ids, list):
                memory_ids = [memory_ids] if memory_ids else []
            if memory_id and str(memory_id) not in {str(item) for item in memory_ids}:
                memory_ids.append(str(memory_id))
            if memory_ids:
                incoming["memory_ids"] = memory_ids
            merged = dict(existing)
            merged.update(incoming)
            if memory_id:
                # Re-indexing a previously updated/deleted memory should
                # reactivate its evidence edge after remove_memory() marked
                # the old edge quarantined.
                merged["status"] = "active"
                merged.pop("quarantine_reason", None)
                merged.pop("quarantined_at", None)
            attrs_json = json.dumps(merged)
            query = """
            MATCH (a:Entity {id: $source}), (b:Entity {id: $target})
            MERGE (a)-[r:RelatesTo {relation_type: $rel_type}]->(b)
            ON MATCH SET r.attributes = $attrs, r.user_scope = $scope
            ON CREATE SET r.attributes = $attrs, r.user_scope = $scope
            """
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

    def index_memory(
        self,
        memory_id: str,
        category: str,
        content: str,
        tags: List[str] | None = None,
        created_at: str | None = None,
    ) -> int:
        """Index one memory and its extracted entities in the graph.

        A memory node provides an explicit bridge back to the source record;
        shared entity nodes provide cross-memory linking. Re-indexing the
        same memory is safe because edges retain a memory-id evidence list.
        """
        if not memory_id or not content:
            return 0
        memory_node = f"memory:{memory_id}"
        memory_attrs = {
            "memory_id": str(memory_id),
            "category": str(category or "context_note"),
            "tags": list(tags or []),
            "content_preview": str(content)[:500],
            "created_at": created_at,
            "status": "active",
        }
        self.upsert_node(memory_node, "memory", memory_attrs)
        self.add_relationship(
            memory_node,
            "memory",
            "about_user",
            "user",
            "person",
            {"memory_id": str(memory_id), "category": category},
        )

        relations = extract_graph_relations(content, category, tags)
        for relation in relations:
            attributes = dict(relation.get("attributes") or {})
            attributes["memory_id"] = str(memory_id)
            self.add_relationship(
                relation["source"],
                relation["source_type"],
                relation["relation"],
                relation["target"],
                relation["target_type"],
                attributes,
            )
            # Link the source memory to the entity so graph traversal can
            # explain which stored memories support a relationship.
            self.add_relationship(
                memory_node,
                "memory",
                "mentions",
                relation["target"],
                relation["target_type"],
                {"memory_id": str(memory_id), "category": category},
            )
        self._flush()
        return len(relations)

    def remove_memory(self, memory_id: str) -> bool:
        """Remove one memory's graph evidence without deleting shared entities."""
        if not memory_id:
            return False
        memory_id = str(memory_id)
        memory_node = f"memory:{memory_id}"
        changed = False
        with self._shared_conn_lock:
            result = self.conn.execute(
                """MATCH (a:Entity)-[r:RelatesTo]->(b:Entity)
                   RETURN a.id, r.relation_type, b.id, r.attributes"""
            )
            edges = []
            while result.has_next():
                edges.append(result.get_next())
            for source, relation, target, raw_attrs in edges:
                try:
                    attrs = json.loads(raw_attrs) if raw_attrs else {}
                except Exception:
                    attrs = {}
                memory_ids = attrs.get("memory_ids", [])
                if not isinstance(memory_ids, list):
                    memory_ids = [memory_ids] if memory_ids else []
                if memory_id not in {str(item) for item in memory_ids} and source != memory_node:
                    continue
                remaining = [item for item in memory_ids if str(item) != memory_id]
                if remaining:
                    attrs["memory_ids"] = remaining
                    if str(attrs.get("memory_id")) == memory_id:
                        attrs.pop("memory_id", None)
                else:
                    attrs.pop("memory_ids", None)
                    if str(attrs.get("memory_id")) == memory_id:
                        attrs.pop("memory_id", None)
                    attrs["status"] = "quarantined"
                    attrs["quarantine_reason"] = "memory evidence removed"
                self.conn.execute(
                    """MATCH (a:Entity {id: $source})-[r:RelatesTo]->(b:Entity {id: $target})
                       WHERE r.relation_type = $relation
                       SET r.attributes = $attrs""",
                    parameters={
                        "source": source, "target": target,
                        "relation": relation, "attrs": json.dumps(attrs),
                    },
                )
                changed = True

            result = self.conn.execute(
                "MATCH (n:Entity {id: $id}) RETURN n.attributes",
                parameters={"id": memory_node},
            )
            if result.has_next():
                try:
                    attrs = json.loads(result.get_next()[0] or "{}")
                except Exception:
                    attrs = {}
                attrs["status"] = "quarantined"
                attrs["quarantine_reason"] = "memory removed"
                self.conn.execute(
                    "MATCH (n:Entity {id: $id}) SET n.attributes = $attrs",
                    parameters={"id": memory_node, "attrs": json.dumps(attrs)},
                )
                changed = True
        if changed:
            self._flush()
        return changed

    # -- read operations ------------------------------------------------------

    @staticmethod
    def _parse_attributes(raw: Any) -> Dict[str, Any]:
        try:
            attrs = json.loads(raw) if isinstance(raw, str) and raw else (raw or {})
        except Exception:
            attrs = {}
        return attrs if isinstance(attrs, dict) else {}

    @staticmethod
    def _visible_attributes(raw: Any) -> bool:
        attrs = KuzuGraphStore._parse_attributes(raw)
        return not attrs or attrs.get("status") != "quarantined"

    def query_graph(self, entity_id: str) -> List[Dict[str, Any]]:
        """Find visible outgoing edges from an entity id."""
        query = """
        MATCH (a:Entity {id: $id})-[r:RelatesTo]->(b:Entity)
        RETURN a.id AS source, a.entity_type AS source_type,
               r.relation_type AS relation, b.id AS target,
               b.entity_type AS target_type, a.attributes AS source_attrs,
               b.attributes AS target_attrs, r.attributes AS relation_attrs
        """
        with self._shared_conn_lock:
            results = self.conn.execute(query, parameters={"id": entity_id})
        edges: List[Dict[str, Any]] = []
        while results.has_next():
            row = results.get_next()
            if not all(self._visible_attributes(value) for value in row[5:8]):
                continue
            edges.append({
                "source": row[0], "source_type": row[1],
                "relation": row[2], "target": row[3],
                "target_type": row[4],
                "attributes": self._parse_attributes(row[7]),
            })
        return edges

    def search_graph(self, term: str) -> List[Dict[str, Any]]:
        """Bidirectional fuzzy search over visible entity edges."""
        term_lower = str(term or "").lower().strip()
        if not term_lower:
            return []
        query = """
        MATCH (a:Entity)-[r:RelatesTo]->(b:Entity)
        RETURN a.id AS source, a.entity_type AS source_type,
               r.relation_type AS relation, b.id AS target,
               b.entity_type AS target_type, a.attributes AS source_attrs,
               b.attributes AS target_attrs, r.attributes AS relation_attrs
        """
        with self._shared_conn_lock:
            results = self.conn.execute(query)
        edges: List[Dict[str, Any]] = []
        while results.has_next():
            row = results.get_next()
            if not all(self._visible_attributes(value) for value in row[5:8]):
                continue
            src, rel, tgt = row[0], row[2], row[3]
            if term_lower in str(src).lower() or term_lower in str(tgt).lower():
                edges.append({
                    "source": src, "source_type": row[1],
                    "relation": rel, "target": tgt,
                    "target_type": row[4],
                    "attributes": self._parse_attributes(row[7]),
                })
        return edges

    def traverse_graph(
        self,
        entity_id: str,
        depth: int = 2,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """Return a bounded bidirectional neighborhood around an entity.

        Kuzu's graph syntax varies across supported releases, so traversal is
        performed with one stable edge query followed by a small in-memory BFS.
        This keeps the RPC response bounded and supports incoming as well as
        outgoing links, which is useful for finding the memories that mention
        an entity.
        """
        requested = str(entity_id or "").strip()
        if not requested:
            return {"entity_id": requested, "depth": 0, "nodes": [], "edges": []}
        try:
            depth = max(1, min(int(depth), 4))
        except (TypeError, ValueError):
            depth = 2
        try:
            limit = max(1, min(int(limit), 250))
        except (TypeError, ValueError):
            limit = 100

        query = """
        MATCH (a:Entity)-[r:RelatesTo]->(b:Entity)
        RETURN a.id AS source, a.entity_type AS source_type,
               r.relation_type AS relation, b.id AS target,
               b.entity_type AS target_type, a.attributes AS source_attrs,
               b.attributes AS target_attrs, r.attributes AS relation_attrs
        """
        with self._shared_conn_lock:
            results = self.conn.execute(query)
        all_edges: List[Dict[str, Any]] = []
        node_data: Dict[str, Dict[str, Any]] = {}
        while results.has_next():
            row = results.get_next()
            if not all(self._visible_attributes(value) for value in row[5:8]):
                continue
            source, target = str(row[0]), str(row[3])
            node_data[source] = {
                "id": source,
                "entity_type": row[1],
                "attributes": self._parse_attributes(row[5]),
            }
            node_data[target] = {
                "id": target,
                "entity_type": row[4],
                "attributes": self._parse_attributes(row[6]),
            }
            all_edges.append({
                "source": source,
                "source_type": row[1],
                "relation": row[2],
                "target": target,
                "target_type": row[4],
                "attributes": self._parse_attributes(row[7]),
            })

        exact = [node for node in node_data if node.casefold() == requested.casefold()]
        if exact:
            seeds = exact[:1]
        else:
            seeds = [node for node in node_data if requested.casefold() in node.casefold()][:1]
        if not seeds:
            return {"entity_id": requested, "depth": depth, "nodes": [], "edges": []}

        adjacency: Dict[str, List[tuple[str, Dict[str, Any]]]] = {}
        for edge in all_edges:
            adjacency.setdefault(edge["source"], []).append((edge["target"], edge))
            adjacency.setdefault(edge["target"], []).append((edge["source"], edge))

        distances = {seeds[0]: 0}
        queue = [seeds[0]]
        selected_edges: List[Dict[str, Any]] = []
        selected_keys: set[tuple[str, str, str]] = set()
        while queue and len(distances) <= limit:
            current = queue.pop(0)
            current_depth = distances[current]
            if current_depth >= depth:
                continue
            for neighbor, edge in adjacency.get(current, []):
                key = (edge["source"], edge["relation"], edge["target"])
                if key not in selected_keys and len(selected_edges) < limit:
                    selected_keys.add(key)
                    selected_edges.append(edge)
                if neighbor not in distances and len(distances) < limit:
                    distances[neighbor] = current_depth + 1
                    queue.append(neighbor)

        selected_nodes = [node_data[node] for node in distances if node in node_data]
        return {
            "entity_id": seeds[0],
            "depth": depth,
            "nodes": selected_nodes[:limit],
            "edges": selected_edges[:limit],
        }

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
