"""Kuzu graph layer for hybrid memory.

Stores entity and memory nodes (people, concepts, items, tools,
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
from typing import Any, Dict, List, Optional

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
# Role words for "my X is Name" / "X is Name" alias extraction.
# Expanded seed list — user-extensible via config (role_words) and
# self-extending via LLM ambiguity gate (see _add_learned_role_word).
_DEFAULT_ROLE_WORDS = frozenset({
    "wife", "husband", "partner", "boyfriend", "girlfriend", "ex",
    "boss", "advisor", "doctor", "doc", "teacher", "mentor", "friend",
    "colleague", "manager", "supervisor", "sibling", "brother", "sister",
    "parent", "mother", "father", "son", "daughter", "child",
    "therapist", "shrink", "accountant", "lawyer", "coach", "physio",
    "physiotherapist", "landlord", "roommate", "sponsor", "counselor",
    "nurse", "carer", "caregiver", "midwife", "dentist", "pharmacist",
    "trainer", "tutor", "professor", "intern", "assistant", "secretary",
    "receptionist", "neighbor", "housemate", "flatmate",
})
# Config-driven override (set by __init__.py from hybrid_memory.json).
# Union of defaults + user-configured + LLM-learned words.
_role_words_override: set[str] = set()
_role_words_lock = threading.Lock()


def _get_role_words() -> frozenset[str]:
    """Return the active set of role words (defaults + config + learned)."""
    with _role_words_lock:
        if _role_words_override:
            return _DEFAULT_ROLE_WORDS | _role_words_override
        return _DEFAULT_ROLE_WORDS


def _is_role_word(word: str) -> bool:
    """Check if a word is a known person-role word (wife, therapist, etc.)."""
    return word.casefold() in _get_role_words()


def _add_learned_role_word(word: str) -> None:
    """Add a role word learned by the LLM ambiguity gate.

    Thread-safe. The caller is responsible for persisting the word to
    config so it survives restarts; this only updates the in-memory set.
    """
    w = word.casefold().strip()
    if not w or len(w) < 2:
        return
    with _role_words_lock:
        _role_words_override.add(w)


def _set_role_words_override(words: set[str]) -> None:
    """Replace the config-driven override set (called at init from config)."""
    with _role_words_lock:
        _role_words_override.clear()
        _role_words_override.update(
            w.casefold().strip() for w in words if w.strip()
        )
_GRAPH_TECH_TERMS = frozenset({
    "python", "javascript", "typescript", "rust", "go", "java", "react",
    "docker", "kubernetes", "vim", "neovim", "vscode", "git", "github",
    "duckdb", "kuzu", "stripe", "linux", "windows", "macos", "postgres",
    "postgresql", "redis", "sqlite", "aws", "gcp", "azure", "openai",
})
# Short extractor fragments that are never useful as graph entities. Keeping
# this gate before graph writes prevents known leaks instead of only hiding them
# later during quarantine maintenance.
_GRAPH_CURATED_JUNK_ENTITIES = frozenset({
    "location", "children and", "and", "the user", "a lot",
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
    words = cleaned.split()
    if len(cleaned) < 3 or len(words) > 8:
        return False
    if cleaned.casefold() in _GRAPH_STOP_ENTITIES:
        return False
    if cleaned.casefold() in _GRAPH_CURATED_JUNK_ENTITIES:
        return False
    # Allow role mentions like "my role", "my doctor", "the boss" through
    # even though their first word is a stopword. These are valid graph
    # entities that alias to canonical person names.
    if words[0].casefold() in _GRAPH_STOP_ENTITIES:
        if len(words) >= 2 and _is_role_word(words[1]):
            return True
        return False
    # Entity extraction should produce names or short noun phrases, not a
    # sentence/paragraph accidentally captured by a broad pattern or LLM.
    # Keep five-word goals such as "learn more about marine biology", but reject
    # longer payloads before they can create visible graph noise.
    if len(words) >= 6 or len(cleaned) >= 80:
        return False
    if re.search(
        r"\b(?:i|we|user|they)\s+(?:am|are|was|were|have|has|is|" \
        r"expecting|waiting|trying|working|thinking)\b",
        cleaned,
        re.IGNORECASE,
    ) and len(words) >= 4:
        return False
    return bool(re.search(r"[A-Za-z0-9]", cleaned))


def _slug_relation(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return value or "related_to"


def _infer_graph_type(entity: str, relation: str, default: str = "concept") -> str:
    """Infer a useful node type for generic extracted targets."""
    relation = relation.lower()
    entity_lower = entity.casefold()
    if relation in {"uses"}:
        return "item"
    if relation in {"has_attribute"}:
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
    if relation.startswith("has_") and _is_role_word(relation[4:]):
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

    # Relationships: "Entity-B is my role", "my advisor is Entity-B",
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
        r"\b(?i:(?:my|the\s+user'?s?))\s+([a-z][a-z_-]*)\s+is\s+"
        r"([A-Z][A-Za-z0-9'_-]*(?:\s+[A-Z][A-Za-z0-9'_-]*)?)",
    )
    for match in my_relation.finditer(text):
        relation, name = match.group(1).lower(), match.group(2)
        if _is_role_word(relation):
            add("user", "person", f"has_{relation}", name, "person")

    # Bare role-name pattern: "Role is Entity-A", "Contact is Entity-B"
    # (common in generated memories that drop the "my" prefix).
    # The name group MUST start with a capital letter to prevent matching
    # "boss is expecting", "doctor is happy", "ex is a director" etc. —
    # those are verbs/adjectives, not names. The role word is case-insensitive
    # (matches both "Role" and "role") but the name is NOT.
    # Broadened from a hardcoded alternation to any lowercase word — the
    # _is_role_word() gate filters non-role words so we don't need to
    # maintain a regex alternation in sync with the word set.
    bare_role = re.compile(
        r"\b(?i:([a-z][a-z_-]+))\s+is\s+"
        r"([A-Z][A-Za-z0-9'_-]*(?:\s+[A-Z][A-Za-z0-9'_-]*)?)",
    )
    for match in bare_role.finditer(text):
        role, name = match.group(1).lower(), match.group(2)
        if _is_role_word(role):
            add("user", "person", f"has_{role}", name, "person")

    direct_relationship = re.compile(
        r"\b(?:i\s+am\s+|i\s+)?(married|dating|seeing|friends?)"
        r"\s+(?:to|with)\s+([A-Za-z][A-Za-z0-9'_-]*(?:\s+[A-Za-z][A-Za-z0-9'_-]*)?)",
        re.IGNORECASE,
    )
    for match in direct_relationship.finditer(text):
        add("user", "person", _slug_relation(f"{match.group(1)}_with"), match.group(2), "person")

    # Role mentions without names: "my role", "my doc", "my advisor".
    # These create graph nodes that can be aliased to canonical names.
    # Without this, "my role" never enters the graph and searching for it
    # via the graph yields nothing — the alias system has no anchor.
    role_mention = re.finditer(
        r"\b(?:my|the\s+user'?s?)\s+([a-z][a-z_-]*)\b",
        text,
        re.IGNORECASE,
    )
    for match in role_mention:
        role = match.group(1).lower()
        if _is_role_word(role):
            add("user", "person", f"has_{role}", f"my {role}", "person", "role_mention")

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
    attribute = re.search(
        r"\b(?:i|user)\s+(?:has|have)\s+"
        r"(?:a\s+(?:trait|preference|skill|interest)\s+of\s+)?(.+?)(?:[.,;]|$)",
        text,
        re.IGNORECASE,
    )
    attribute_context = re.search(
        r"\b(?:attribute|trait|preference|skill|interest)\b",
        text,
        re.IGNORECASE,
    )
    if attribute and attribute_context:
        attr = re.split(r"\s+(?:and|but|for)\s+", attribute.group(1), maxsplit=1, flags=re.IGNORECASE)[0]
        add("user", "person", "has_attribute", attr, "attribute")

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
        add("user", "person", relation, preferred, evidence="preference")
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
    # Track which entities were already typed as "person" by explicit
    # relationship patterns, so proper-noun mentions of the same name
    # inherit the person type instead of defaulting to "concept".
    person_targets = {
        item["target"].casefold()
        for item in relations
        if item.get("target_type") == "person"
    }
    for entity in proper_nouns:
        if entity.casefold() in _GRAPH_STOP_ENTITIES or entity.casefold() == "user":
            continue
        if entity.casefold() in explicit_targets:
            continue
        # Inherit person type if this name was already seen in a relationship
        entity_type = "person" if entity.casefold() in person_targets else "concept"
        add("user", "person", proper_relation, entity, entity_type, "proper_noun")

    return relations


# ---------------------------------------------------------------------------
# LLM-assisted entity extraction (Stage 2 for graph)
# ---------------------------------------------------------------------------

_GRAPH_LLM_MIN_LENGTH = 60
_GRAPH_LLM_TIMEOUT = 15.0

# Relations that carry no traversal/typing signal. Regex produces these
# generically (mentions/related_to/context_about/insight_about/about_user/
# working_toward/interested_in) — they connect everything to everything.
# Used by the hybrid gate (count TYPED relations, not raw ones) and by
# traversal (never walk generic edges).
_GRAPH_GENERIC_RELATIONS = frozenset({
    "mentions", "about_user", "related_to", "context_about",
    "insight_about", "working_toward", "interested_in",
})

_GRAPH_LLM_PROMPT = """You are a knowledge-graph entity extractor. Given a stored memory, extract typed relationships between the user and entities mentioned.

Return a JSON array of objects with these keys:
- "source": entity name (usually "user")
- "source_type": "person", "memory", "concept", etc.
- "relation": snake_case relation (e.g. "works_at", "uses", "prefers", "has_attribute", "uses", "lives_in", "knows", "insight_about")
- "target": the related entity name
- "target_type": one of "person", "organization", "place", "item", "attribute", "technology", "tool", "concept", "event", "goal", "insight"

Only extract real, specific entities — not pronouns, stop words, or generic terms like "thing", "something", "people".
If the memory has no clear entities, return an empty array: []

Examples:
Memory: "User works at Stripe and uses Kubernetes"
Output: [{"source":"user","source_type":"person","relation":"works_at","target":"Stripe","target_type":"organization"},{"source":"user","source_type":"person","relation":"uses","target":"Kubernetes","target_type":"technology"}]

Memory: "I just realized shame shapes my work patterns"
Output: [{"source":"user","source_type":"person","relation":"insight_about","target":"shame","target_type":"concept"}]

Memory: "User prefers dark mode"
Output: [{"source":"user","source_type":"person","relation":"prefers","target":"dark mode","target_type":"concept"}]

Memory: "hey how are you"
Output: []
"""


def extract_graph_relations_llm(
    content: str,
    category: str = "context_note",
) -> List[Dict[str, Any]]:
    """LLM-assisted entity extraction for graph indexing.

    Uses the host's auxiliary LLM to extract typed relations. All results
    pass through the same _valid_graph_entity / _GRAPH_STOP_ENTITIES gate
    as the regex extractor, so junk is rejected before it reaches the graph.
    Never raises — returns [] on any failure.
    """
    if not content or len(content.strip()) < _GRAPH_LLM_MIN_LENGTH:
        return []
    from egress import gate as _egress_gate
    if not _egress_gate("graph_typing", content):
        return []


    try:
        from agent.auxiliary_client import call_llm
    except ImportError:
        logger.debug("Graph LLM extraction unavailable: agent.auxiliary_client not importable")
        return []
    except Exception as e:
        logger.debug("Graph LLM extraction unavailable: %s", e)
        return []

    messages = [
        {"role": "system", "content": _GRAPH_LLM_PROMPT},
        {"role": "user", "content": f"Memory ({category}): {content}"},
    ]

    try:
        response = call_llm(
            task="graph_entity_extraction",
            messages=messages,
            temperature=0.0,
            max_tokens=600,
            timeout=_GRAPH_LLM_TIMEOUT,
        )
    except Exception as e:
        logger.debug("Graph LLM extraction call failed: %s", e)
        return []

    if response is None:
        return []

    try:
        text = response.choices[0].message.content
    except (AttributeError, IndexError, KeyError):
        return []

    if not text or not text.strip():
        return []

    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        text = text.strip()

    try:
        raw_items = json.loads(text)
    except json.JSONDecodeError:
        logger.debug("Graph LLM extraction returned non-JSON: %s", text[:100])
        return []

    if not isinstance(raw_items, list):
        return []

    relations: List[Dict[str, Any]] = []
    seen: set = set()
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        source = _clean_graph_entity(str(item.get("source", "user")))
        target = _clean_graph_entity(str(item.get("target", "")))
        relation = _slug_relation(str(item.get("relation", "")))
        if not source or not _valid_graph_entity(target):
            continue
        key = (source.casefold(), relation, target.casefold())
        if key in seen:
            continue
        seen.add(key)
        target_type = str(item.get("target_type", "")).strip() or _infer_graph_type(target, relation)
        source_type = str(item.get("source_type", "person")).strip() or "person"
        relations.append({
            "source": source,
            "source_type": source_type,
            "relation": relation,
            "target": target,
            "target_type": target_type,
            "attributes": {
                "category": str(category or "context_note").lower(),
                "extractor": "llm",
                "evidence": "llm",
            },
        })
    return relations


def extract_graph_relations_hybrid(
    content: str,
    category: str = "context_note",
    tags: List[str] | None = None,
    use_llm: bool = True,
) -> List[Dict[str, Any]]:
    """Regex-first, LLM-supplemented entity extraction with junk gating.

    Stage 1: deterministic regex patterns (fast, zero cost).
    Stage 2: if regex found few relations and content is substantial,
    supplement with LLM extraction. All results — regex and LLM — pass
    through the same stop-word / validity gate.
    """
    regex_relations = extract_graph_relations(content, category, tags)
    if not use_llm:
        return regex_relations

    # Only invoke LLM when regex found little TYPED structure and the
    # message is substantial. Counting raw regex relations is wrong:
    # regex produces 3+ generic junk edges (related_to/context_about/
    # insight_about/mentions...) on content-rich memories, so the LLM
    # never fired on exactly the memories that need typing — the graph
    # slowly rotted back to concept soup. Count typed relations instead.
    typed_regex = [
        r for r in regex_relations
        if r.get("relation") not in _GRAPH_GENERIC_RELATIONS
    ]
    if len(typed_regex) >= 3 or len(content.strip()) < _GRAPH_LLM_MIN_LENGTH:
        return regex_relations

    llm_relations = extract_graph_relations_llm(content, category)
    if not llm_relations:
        return regex_relations

    # Merge, dedup by (source, relation, target). Regex takes priority.
    merged: List[Dict[str, Any]] = list(regex_relations)
    existing = {
        (r["source"].casefold(), r["relation"], r["target"].casefold())
        for r in regex_relations
    }
    for r in llm_relations:
        key = (r["source"].casefold(), r["relation"], r["target"].casefold())
        if key not in existing:
            merged.append(r)
            existing.add(key)
    return merged


_GRAPH_TYPED_RELATIONS = frozenset({
    # Relations that carry real semantic meaning for traversal. The generic
    # set (mentions/related_to/context_about/insight_about/about_user/
    # working_toward/interested_in) connects everything to everything and
    # must NOT be traversed — that noise is what made the old boost useless.
    "uses", "has", "works_on", "works_at", "works_with", "prefers",
    "has_goal", "has_attribute", "has_wife", "has_pet", "has_project",
    "has_trait", "lives_in", "knows", "develops", "plans_to_use",
    "goal_involves", "completed", "concerned_about", "experiences",
    "fears", "struggles_with", "includes", "attends", "married_to",
    "delegates_finances_to", "owns", "uses", "provided_by",
    "insured_by", "related_to_finance", "sold", "built", "recommends",
    "recommended", "manages", "leads", "joined", "left", "hired_by",
    "reports_to", "trained_in", "interested_in_work", "evaluates",
    "monitors", "involved_in", "part_of", "member_of", "cares_for",
    "communicates_with", "supports", "influenced_by", "caused_by",
    "affects", "replaces", "upgraded_to", "costs", "earns", "pays",
    "finances", "negotiates", "plans_to_buy", "plans_to_sell",
    "sold_out", "tracks", "measures", "solves", "pain_point_of",
})


class KuzuGraphStore:
    """Kuzu-backed relationship graph with thread-safe access.

    Uses a process-level shared database connection so that multiple
    ArgosProvider instances (Hermes creates one per agent/session)
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

    def _internal_id(self, node_id: str) -> str:
        """Scope graph IDs for non-default users while preserving legacy IDs."""
        node_id = str(node_id or "")
        if self.user_id == "default_user":
            return node_id
        prefix = f"{self.user_id}::"
        return node_id if node_id.startswith(prefix) else prefix + node_id

    def _external_id(self, node_id: str) -> str:
        node_id = str(node_id or "")
        if self.user_id == "default_user":
            return node_id
        prefix = f"{self.user_id}::"
        return node_id[len(prefix):] if node_id.startswith(prefix) else node_id

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
        node_id = self._internal_id(node_id)
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
            # If new evidence is being provided (incoming has memory_id or
            # memory_ids), clear any prior quarantine — a quarantined node
            # that receives fresh evidence is no longer junk.
            if (incoming.get("memory_id") or incoming.get("memory_ids")) and \
                    merged.get("status") == "quarantined":
                merged.pop("status", None)
                merged.pop("quarantine_reason", None)
                merged.pop("quarantined_at", None)
            # Never downgrade a person node to concept. If the node already
            # exists as "person" (from an explicit relationship pattern),
            # a later merge from a relation-free memory must not overwrite
            # it to "concept". "person" is a stronger type than "concept".
            type_result = self.conn.execute(
                "MATCH (n:Entity {id: $id}) RETURN n.entity_type",
                parameters={"id": node_id},
            )
            existing_type = None
            if type_result.has_next():
                existing_type = type_result.get_next()[0]
            effective_type = node_type
            if existing_type == "person" and node_type != "person":
                effective_type = "person"
            query = """
            MERGE (n:Entity {id: $id})
            ON MATCH SET n.entity_type = $type, n.attributes = $attrs, n.user_scope = $scope
            ON CREATE SET n.entity_type = $type, n.attributes = $attrs, n.user_scope = $scope
            """
            self.conn.execute(query, parameters={
                "id": node_id, "type": effective_type,
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
        source_id = self._internal_id(source_id)
        target_id = self._internal_id(target_id)
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
        """Convenience: upsert both nodes then the edge.

        Passes memory_id/memory_ids from the edge attributes to the node
        upserts so that re-indexing a memory clears any prior quarantine
        on the entity nodes (the quarantine-clear guard in upsert_node
        requires incoming memory evidence to fire).
        """
        node_attrs: Dict[str, Any] = {}
        if attributes:
            if "memory_id" in attributes:
                node_attrs["memory_id"] = attributes["memory_id"]
            if "memory_ids" in attributes:
                node_attrs["memory_ids"] = attributes["memory_ids"]
        self.upsert_node(source, source_type, node_attrs)
        self.upsert_node(target, target_type, node_attrs)
        self.upsert_edge(source, target, relation, attributes)

    def index_memory(
        self,
        memory_id: str,
        category: str,
        content: str,
        tags: List[str] | None = None,
        created_at: str | None = None,
        use_llm: bool = True,
    ) -> int:
        """Index one memory and its extracted entities in the graph.

        A memory node provides an explicit bridge back to the source record;
        shared entity nodes provide cross-memory linking. Re-indexing the
        same memory is safe because edges retain a memory-id evidence list.

        Extraction is regex-first, LLM-supplemented when regex finds few
        relations and the content is substantial. All entities pass through
        the stop-word / validity gate before reaching the graph.
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

        relations = extract_graph_relations_hybrid(content, category, tags, use_llm=use_llm)
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
        memory_node = self._internal_id(f"memory:{memory_id}")
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

    def memory_ids_for_query(self, query: str, limit: int = 100) -> List[str]:
        """Return memory IDs linked to graph entities mentioned in *query*.

        This is intentionally a bounded lexical bridge from normal memory
        search into the graph. It does not replace vector/text retrieval; it
        supplies graph-supported candidates and a stable relevance order.
        """
        stop_words = _GRAPH_STOP_ENTITIES | {
            "about", "and", "are", "does", "from", "have", "into", "more",
            "that", "the", "this", "what", "when", "where", "which", "with",
        }
        terms = []
        seen_terms = set()
        for term in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]{2,}", str(query or "").lower()):
            if term in stop_words or term in seen_terms:
                continue
            seen_terms.add(term)
            terms.append(term)
            if len(terms) >= 8:
                break
        if not terms:
            return []
        scores: Dict[str, int] = {}
        for term in terms:
            for edge in self.search_graph(term, limit=max(10, min(limit, 100))):
                attributes = edge.get("attributes") or {}
                memory_ids = attributes.get("memory_ids", [])
                if not isinstance(memory_ids, list):
                    memory_ids = [memory_ids] if memory_ids else []
                for memory_id in memory_ids:
                    if memory_id:
                        scores[str(memory_id)] = scores.get(str(memory_id), 0) + 2
                for endpoint in (edge.get("source"), edge.get("target")):
                    if str(endpoint).startswith("memory:"):
                        memory_id = str(endpoint)[len("memory:"):]
                        if memory_id:
                            scores[memory_id] = scores.get(memory_id, 0) + 1
        ordered = sorted(scores, key=lambda memory_id: (-scores[memory_id], memory_id))
        return ordered[:max(1, min(int(limit), 500))]

    def traversal_memory_ids(
        self,
        query: str,
        depth: int = 2,
        limit: int = 100,
        require_specific_seed: bool = True,
    ) -> List[str]:
        """Traversal-based retrieval: walk TYPED relations from seed entities.

        Unlike memory_ids_for_query (lexical term→edge matching), this
        actually walks the graph: ground query terms to seed entities,
        BFS up to `depth` hops over TYPED relations only (generic
        mentions/related_to edges are excluded — they connect everything
        and carry no traversal value), and collect memory IDs attached to
        the traversed edges. Weight: hop-1 = 2, hop-2 = 1.

        require_specific_seed=True (default): return [] unless at least one
        grounded seed is a NON-CONCEPT entity (person/organization/place/
        item/technology/tool/event/goal). Broad queries ("current
        relationships personal context") ground only to generic concepts —
        their traversal output is hub-adjacent noise that regresses
        precision, so the caller skips the boost floor for them.

        Returns memory IDs ordered by traversal weight (desc).
        """
        stop_words = _GRAPH_STOP_ENTITIES | {
            "about", "and", "are", "does", "from", "have", "into", "more",
            "that", "the", "this", "what", "when", "where", "which", "with",
            "who", "how", "why", "did", "was", "were", "been", "for", "our",
        }
        terms = []
        seen_terms = set()
        for term in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]{2,}", str(query or "").lower()):
            if term in stop_words or term in seen_terms:
                continue
            seen_terms.add(term)
            terms.append(term)
            if len(terms) >= 8:
                break
        if not terms:
            return []

        # Ground: find seed entities for each query term (fuzzy node match).
        # Exclude the "user" hub node — in a personal memory graph
        # EVERYTHING touches user, so traversing from it is meaningless.
        seeds = set()
        seed_types: Dict[str, str] = {}
        for term in terms:
            try:
                for edge in self.search_graph(term, limit=20):
                    for endpoint in (edge["source"], edge["target"]):
                        eid = str(endpoint)
                        if eid != "user" and not eid.startswith("memory:"):
                            seeds.add(eid)
                            node = self._query_node(eid)
                            if node:
                                seed_types[eid] = str(
                                    node.get("entity_type", "concept"))
            except Exception:
                continue
        if not seeds:
            return []
        if require_specific_seed:
            non_concept = sum(1 for t in seed_types.values() if t != "concept")
            if non_concept < 2:
                # Broad query: only generic-concept seeds or a single weak
                # non-concept hit (relationships, preferences, goals...).
                # Traversal output would be hub-adjacent noise — the boost
                # floor regresses precision on these.
                return []

        # BFS over meaningful relations only, hop-weighted. Meaningful =
        # typed relations (real semantics) OR LLM-extracted edges (entity
        # validity gated by the LLM). Regex generic edges (related_to,
        # context_about, insight_about, mentions...) connect everything to
        # everything and carry no traversal value.
        def _traversable_node(eid: str) -> bool:
            """Skip junk/path-like nodes that break Kuzu IN-list literals."""
            if len(eid) > 60 or any(ch in eid for ch in ("\\", "'", '"', ":", "/")):
                return False
            return not eid.startswith("memory:")

        scores: Dict[str, float] = {}
        frontier = [s for s in seeds if _traversable_node(s)]
        visited = set(frontier)
        for hop in range(1, depth + 1):
            if not frontier:
                break
            if len(frontier) > 40:  # keep per-hop queries bounded
                frontier = frontier[:40]
            edges = self._query_edges_for_nodes(frontier)
            next_frontier = []
            for edge in edges:
                rel = edge.get("relation")
                attrs = edge.get("attributes") or {}
                extractor = attrs.get("extractor", "graph_patterns")
                if rel not in _GRAPH_TYPED_RELATIONS and extractor != "llm":
                    continue
                src, tgt = edge.get("source"), edge.get("target")
                mem_ids = attrs.get("memory_ids", [])
                if not isinstance(mem_ids, list):
                    mem_ids = [mem_ids] if mem_ids else []
                weight = 2.0 if hop == 1 else 1.0
                for mid in mem_ids:
                    if mid:
                        scores[str(mid)] = scores.get(str(mid), 0.0) + weight
                # memory: nodes carry explicit memory evidence
                for endpoint in (src, tgt):
                    if str(endpoint).startswith("memory:"):
                        scores[str(endpoint)[len("memory:"):]] = (
                            scores.get(str(endpoint)[len("memory:"):], 0.0) + weight
                        )
                    elif (_traversable_node(str(endpoint))
                          and endpoint not in visited and len(visited) < 200):
                        visited.add(endpoint)
                        next_frontier.append(endpoint)
            frontier = next_frontier

        ordered = sorted(scores, key=lambda mid: (-scores[mid], mid))
        return ordered[:max(1, min(int(limit), 500))]

    def query_graph(self, entity_id: str) -> List[Dict[str, Any]]:
        """Find visible edges touching an entity id (bidirectional).

        Matches both outgoing (entity as source) and incoming (entity as
        target) edges, so concepts that only appear as targets (e.g.
        "shame" in a memory->concept mentions edge) are found. Without
        the incoming half, query_graph under-reports because the extractor
        mostly creates edges as memory -> concept.
        """
        query = """
        MATCH (a:Entity)-[r:RelatesTo]->(b:Entity)
        WHERE a.id = $id OR b.id = $id
        RETURN a.id AS source, a.entity_type AS source_type,
               r.relation_type AS relation, b.id AS target,
               b.entity_type AS target_type, a.attributes AS source_attrs,
               b.attributes AS target_attrs, r.attributes AS relation_attrs
        """
        internal_id = self._internal_id(entity_id)
        with self._shared_conn_lock:
            results = self.conn.execute(query, parameters={"id": internal_id})
        edges: List[Dict[str, Any]] = []
        while results.has_next():
            row = results.get_next()
            if not all(self._visible_attributes(value) for value in row[5:8]):
                continue
            edges.append({
                "source": self._external_id(row[0]), "source_type": row[1],
                "relation": row[2], "target": self._external_id(row[3]),
                "target_type": row[4],
                "attributes": self._parse_attributes(row[7]),
            })
        return edges

    def search_graph(self, term: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Bidirectional fuzzy search over visible entity edges.

        Filtering is pushed into Kuzu (WHERE CONTAINS + LIMIT) so the
        query is O(matching edges), not O(all edges). The quarantine
        visibility guard remains in Python because it inspects a JSON
        attribute field that Kuzu cannot filter natively.
        """
        term_lower = str(term or "").lower().strip()
        if not term_lower:
            return []
        try:
            limit = max(1, min(int(limit), 500))
        except (TypeError, ValueError):
            limit = 100
        # Kuzu CONTAINS is case-sensitive, so we can't use it directly for
        # case-insensitive matching. Instead we filter on lowercased ids
        # by checking both the original and a lowercased comparison. Kuzu
        # supports toLower() in Cypher.
        query = """
        MATCH (a:Entity)-[r:RelatesTo]->(b:Entity)
        WHERE (a.user_scope = $scope OR b.user_scope = $scope)
          AND (toLower(a.id) CONTAINS $term OR toLower(b.id) CONTAINS $term)
        RETURN a.id AS source, a.entity_type AS source_type,
               r.relation_type AS relation, b.id AS target,
               b.entity_type AS target_type, a.attributes AS source_attrs,
               b.attributes AS target_attrs, r.attributes AS relation_attrs
        LIMIT $limit
        """
        with self._shared_conn_lock:
            results = self.conn.execute(
                query,
                parameters={"term": term_lower, "scope": self.user_id, "limit": limit * 3},
            )
        edges: List[Dict[str, Any]] = []
        while results.has_next():
            row = results.get_next()
            if not all(self._visible_attributes(value) for value in row[5:8]):
                continue
            edges.append({
                "source": self._external_id(row[0]), "source_type": row[1],
                "relation": row[2], "target": self._external_id(row[3]),
                "target_type": row[4],
                "attributes": self._parse_attributes(row[7]),
            })
            if len(edges) >= limit:
                break
        return edges

    def _query_edges_for_nodes(
        self, node_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """Fetch all visible edges touching any of the given node ids.

        Uses a parameterized IN-list so Kuzu does the filtering, not Python.
        """
        if not node_ids:
            return []
        node_ids = [self._internal_id(node_id) for node_id in node_ids]
        # Kuzu doesn't support parameterized IN-lists reliably across
        # versions, so we build the list literal safely.
        safe_ids = [str(n).replace("'", "\\'") for n in node_ids]
        id_list = "[" + ", ".join(f"'{n}'" for n in safe_ids) + "]"
        query = f"""
        MATCH (a:Entity)-[r:RelatesTo]->(b:Entity)
        WHERE a.id IN {id_list} OR b.id IN {id_list}
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
            source, target = str(row[0]), str(row[3])
            edges.append({
                "source": self._external_id(source),
                "source_type": row[1],
                "relation": row[2],
                "target": self._external_id(target),
                "target_type": row[4],
                "attributes": self._parse_attributes(row[7]),
            })
        return edges

    def _query_node(self, entity_id: str) -> Dict[str, Any] | None:
        """Fetch a single node by exact or fuzzy id match within the scope."""
        internal_id = self._internal_id(entity_id)
        with self._shared_conn_lock:
            # Try exact match first.
            result = self.conn.execute(
                "MATCH (n:Entity {id: $id}) WHERE n.user_scope = $scope "
                "RETURN n.id, n.entity_type, n.attributes",
                parameters={"id": internal_id, "scope": self.user_id},
            )
            if result.has_next():
                row = result.get_next()
                attrs = self._parse_attributes(row[2])
                if self._visible_attributes(attrs):
                    return {"id": self._external_id(row[0]), "entity_type": row[1], "attributes": attrs}
            # Fuzzy match on substring.
            result = self.conn.execute(
                "MATCH (n:Entity) WHERE n.user_scope = $scope "
                "AND toLower(n.id) CONTAINS $term "
                "RETURN n.id, n.entity_type, n.attributes LIMIT 1",
                parameters={"scope": self.user_id, "term": entity_id.lower()},
            )
            if result.has_next():
                row = result.get_next()
                attrs = self._parse_attributes(row[2])
                if self._visible_attributes(attrs):
                    return {"id": self._external_id(row[0]), "entity_type": row[1], "attributes": attrs}
        return None

    def traverse_graph(
        self,
        entity_id: str,
        depth: int = 2,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """Return a bounded bidirectional neighborhood around an entity.

        Uses targeted per-hop queries (WHERE node IN frontier) instead of
        loading all edges. This keeps each query O(frontier edges) rather
        than O(all edges), making traversal practical as the graph grows.
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

        seed = self._query_node(requested)
        if seed is None:
            return {"entity_id": requested, "depth": depth, "nodes": [], "edges": []}

        # Verify the seed has at least one visible edge. A node with all
        # edges quarantined (e.g. after remove_memory) should not appear
        # in traversal results, matching the old edge-driven behavior.
        seed_edges = self._query_edges_for_nodes([seed["id"]])
        if not seed_edges:
            return {"entity_id": seed["id"], "depth": depth, "nodes": [], "edges": []}

        # BFS with targeted per-hop edge queries.
        node_data: Dict[str, Dict[str, Any]] = {seed["id"]: seed}
        distances: Dict[str, int] = {seed["id"]: 0}
        selected_edges: List[Dict[str, Any]] = []
        selected_keys: set = set()
        frontier: List[str] = [seed["id"]]

        for hop in range(depth):
            if not frontier or len(node_data) >= limit:
                break
            edges = self._query_edges_for_nodes(frontier)
            next_frontier: List[str] = []
            for edge in edges:
                key = (edge["source"], edge["relation"], edge["target"])
                if key not in selected_keys and len(selected_edges) < limit:
                    selected_keys.add(key)
                    selected_edges.append(edge)
                for endpoint in (edge["source"], edge["target"]):
                    if endpoint not in node_data:
                        # Fetch node details for newly discovered nodes.
                        node = self._query_node(endpoint)
                        if node and len(node_data) < limit:
                            node_data[endpoint] = node
                            distances[endpoint] = hop + 1
                            next_frontier.append(endpoint)
            frontier = next_frontier

        selected_nodes = [node_data[n] for n in distances if n in node_data]
        return {
            "entity_id": seed["id"],
            "depth": depth,
            "nodes": selected_nodes[:limit],
            "edges": selected_edges[:limit],
        }

    def list_nodes(self, node_type: str | None = None, limit: int = 100) -> List[Dict[str, Any]]:
        if node_type:
            query = "MATCH (n:Entity) WHERE n.user_scope = $scope AND n.entity_type = $type " \
                    "RETURN n.id, n.entity_type, n.attributes LIMIT $limit"
            params = {"scope": self.user_id, "type": node_type, "limit": limit}
        else:
            query = "MATCH (n:Entity) WHERE n.user_scope = $scope " \
                    "RETURN n.id, n.entity_type, n.attributes LIMIT $limit"
            params = {"scope": self.user_id, "limit": limit}
        with self._shared_conn_lock:
            results = self.conn.execute(query, parameters=params)
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
            nodes.append({"id": self._external_id(row[0]), "entity_type": row[1], "attributes": attrs})
        return nodes

    def count_nodes(self) -> int:
        with self._shared_conn_lock:
            results = self.conn.execute(
                "MATCH (n:Entity) WHERE n.user_scope = $scope RETURN COUNT(*)",
                parameters={"scope": self.user_id},
            )
            row = results.get_next()
            return int(row[0]) if row else 0

    def count_edges(self) -> int:
        with self._shared_conn_lock:
            results = self.conn.execute(
                "MATCH (a:Entity)-[r:RelatesTo]->(b:Entity) "
                "WHERE r.user_scope = $scope RETURN COUNT(*)",
                parameters={"scope": self.user_id},
            )
            row = results.get_next()
            return int(row[0]) if row else 0

    def clear_scope(self) -> tuple[int, int]:
        """Delete all nodes and edges for the current user_scope.

        Returns (remaining_nodes, remaining_edges) after deletion.
        This is the first phase of a graph rebuild: clear, then re-index
        from DuckDB.
        """
        with self._shared_conn_lock:
            self.conn.execute(
                "MATCH (a:Entity)-[r:RelatesTo]->(b:Entity) "
                "WHERE r.user_scope = $scope DELETE r",
                parameters={"scope": self.user_id},
            )
            self.conn.execute(
                "MATCH (n:Entity) WHERE n.user_scope = $scope DELETE n",
                parameters={"scope": self.user_id},
            )
        self._flush()
        return (self.count_nodes(), self.count_edges())

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

    # Exact entity ids known to be extractor leaks (short fragments that the
    # generic heuristics can't separate from legitimate names). Additions here
    # are quarantined reversibly; they can be restored if ever mis-tagged.
    _CURATED_JUNK_ENTITY_IDS = frozenset({
        "Location", "children and", "and", "the user", "a lot",
    })

    def _quarantine_node(self, node_id: str, reason: str) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        node_id = self._internal_id(node_id)
        with self._shared_conn_lock:
            result = self.conn.execute(
                "MATCH (n:Entity {id: $id}) WHERE n.user_scope = $scope RETURN n.attributes",
                parameters={"id": node_id, "scope": self.user_id},
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
                "MATCH (n:Entity {id: $id}) WHERE n.user_scope = $scope "
                "SET n.attributes = $attrs",
                parameters={"id": node_id, "scope": self.user_id, "attrs": json.dumps(attrs)},
            )
        return True

    def _quarantine_edge(self, source: str, target: str, relation: str, reason: str) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        source = self._internal_id(source)
        target = self._internal_id(target)
        with self._shared_conn_lock:
            result = self.conn.execute(
                """MATCH (a:Entity {id: $source})-[r:RelatesTo]->(b:Entity {id: $target})
                   WHERE r.relation_type = $relation AND r.user_scope = $scope
                   RETURN r.attributes""",
                parameters={"source": source, "target": target, "relation": relation, "scope": self.user_id},
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
                   WHERE r.relation_type = $relation AND r.user_scope = $scope
                   SET r.attributes = $attrs""",
                parameters={
                    "source": source, "target": target,
                    "relation": relation, "scope": self.user_id,
                    "attrs": json.dumps(attrs),
                },
            )
        return True

    def quarantine_junk_entities(self) -> int:
        """Hide obviously malformed nodes/edges without deleting graph data."""
        with self._shared_conn_lock:
            node_results = self.conn.execute(
                "MATCH (n:Entity) WHERE n.user_scope = $scope "
                "RETURN n.id AS id, n.attributes AS attributes, "
                "n.entity_type AS entity_type",
                parameters={"scope": self.user_id},
            )
            nodes = []
            while node_results.has_next():
                nodes.append(node_results.get_next())
            edge_results = self.conn.execute(
                """MATCH (a:Entity)-[r:RelatesTo]->(b:Entity)
                   WHERE r.user_scope = $scope
                   RETURN a.id, r.relation_type, b.id, r.attributes""",
                parameters={"scope": self.user_id},
            )
            edges = []
            while edge_results.has_next():
                edges.append(edge_results.get_next())

        changed = 0
        for node_id, raw_attrs, entity_type in nodes:
            if not self._visible_attributes(raw_attrs):
                continue
            # Skip nodes that have active memory evidence — they were indexed
            # from a real memory and validated by the extraction pipeline.
            # The junk heuristics (prefix matching, length) would re-quarantine
            # legitimate role-mention entities like "my role" (first word "my"
            # is in _JUNK_ENTITY_PREFIXES) on every startup sweep, undoing the
            # quarantine-clear that add_relationship performs at index time.
            attrs = self._parse_attributes(raw_attrs)
            if attrs.get("memory_id") or attrs.get("memory_ids"):
                continue
            node_id = self._external_id(str(node_id))
            first_word = node_id.split()[0].lower() if node_id.split() else ""
            word_count = len(node_id.split())
            # Whole-sentence / paragraph payloads are not valid entity names.
            # Real entities are short noun phrases (a few words); anything
            # running on for many words or a long string is extracted junk.
            sentence_payload = (
                word_count >= 6
                or len(node_id.strip()) >= 80
            )
            # Recognized extractor-leak payloads that the generic heuristics
            # can't isolate from legitimate short names (e.g. "Entity-A").
            curated_leak = str(node_id).strip() in self._CURATED_JUNK_ENTITY_IDS
            if (
                first_word in self._JUNK_ENTITY_PREFIXES
                or len(node_id.strip()) <= 2
                or re.match(r'^e\d+$', node_id.strip())
                or sentence_payload
                or curated_leak
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
