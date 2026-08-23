"""Two-stage fact extractor for auto-extraction after each turn.

Stage 1 — Generic syntactic regex patterns (fast, local, zero latency).
    Matches the *syntactic shape* of durable statements, not topic-specific
    keywords.  Works across ANY domain: personal life, work, tech, hobbies,
    finance, hobbies, etc.  Catches ~80% of durable facts.

Stage 2 — LLM fallback (higher recall, adds latency + token cost).
    If Stage 1 found zero or very few facts AND the user's message is
    substantial, sends the message to the host's LLM via
    ``agent.auxiliary_client.call_llm`` with a structured extraction prompt.
    Returns JSON facts.  Falls back gracefully if the LLM is unavailable.

The agent's ``memory_save`` tool is the primary, high-quality extraction path.
This auto-extractor is a bonus safety net that catches things the agent might
not explicitly save.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Minimum content length to bother extracting.
_MIN_LENGTH = 15
# Maximum sentences per turn to avoid extracting from walls of text.
_MAX_SENTENCES = 25
# LLM fallback triggers when regex finds fewer facts than this AND the
# message is substantial.  We use a ratio-based heuristic rather than a
# fixed count: a 200-word message with 1 regex fact probably has more
# durable facts the regex missed; a 20-word message with 1 fact is likely
# complete.  See _should_try_llm_fallback() for the full logic.
_LLM_FALLBACK_MIN_FACTS = 2  # Below this many facts, always try LLM (if message is long enough).
_LLM_FALLBACK_WORDS_PER_FACT = 80  # If words/facts > this ratio, regex likely missed things.
# Minimum user message length to justify an LLM call (avoid wasting tokens
# on short chit-chat).
_LLM_MIN_CONTENT_LENGTH = 60
# LLM call timeout in seconds.
_LLM_TIMEOUT = 15.0


# ---------------------------------------------------------------------------
# Junk filter — prevents low-quality memories from being stored
# ---------------------------------------------------------------------------

# Phrases that indicate the extracted text is actually agent output, not a
# user fact.  These appear when the regex accidentally matches text the
# assistant said (e.g. "I'll search authoritative sources").
_AGENT_SPEAK_PATTERNS = re.compile(
    r"(?:i'll\s+search|i'll\s+look|i'll\s+check|let me|authoritative\s+sources"
    r"|i'll\s+find|i'll\s+research|searching\s+for|looking\s+up"
    # Implementation-status reports: when a work summary/handoff is pasted
    # into the chat, the LLM extraction stage reads it as "user facts" and
    # mints the agent's self-report as memories (e.g. "Completed
    # implementation of evolution chains feature", "Memory provider now
    # includes 13 tools", "Chain-unfold configuration ships off by
    # default"). These are agent/dev-log narration, not user facts.
    r"|completed\s+implementation\s+of|now\s+includes\s+\d+\s+tools"
    r"|ships\s+off\s+by\s+default|deployment\s+reference\s+specifies"
    r"|handoff\s+doc\s+is\s+located|implementation\s+of\s+evolution"
    r"|work\s+items?\s+(?:from\s+the\s+handoff|complete)"
    r"|service\s+\(pids?\s+\d+)",
    re.IGNORECASE,
)

# Trivial facts that aren't worth storing — the agent can infer these from
# context or they add no value to future conversations.
_TRIVIAL_FACTS = frozenset({
    "literally glm 5", "literally glm5", "a human", "a person",
    "typing", "using a keyboard", "on a computer", "online",
    "here", "there", "ready", "done", "back",
})

# Content that looks like a sentence fragment (starts with lowercase,
# no subject, or is a partial clause).
_FRAGMENT_RE = re.compile(
    r'^(?:the\s+list|i\'ll\s|let me|so\s+now|that\'s\s+why'
    r'|which\s+means|because\s+|so\s+that|in\s+order\s+to)',
    re.IGNORECASE,
)


def _is_junk(fact: Dict[str, Any]) -> bool:
    """Return True if a fact is low-quality and should not be stored.

    Checks for:
    - Agent-speak fragments (text from the assistant's output, not the user)
    - Trivial facts that add no value
    - Sentence fragments with no complete meaning
    - Content that is too short to be useful
    """
    content = fact.get("content", "").strip()
    if not content or len(content) < 10:
        return True

    content_lower = content.lower()

    # Check for agent-speak — the extractor sometimes grabs the assistant's
    # output when it contains "I" statements.
    if _AGENT_SPEAK_PATTERNS.search(content):
        return True

    # Check for trivial facts.
    # "User is X" → check X part.
    if content_lower.startswith("user is "):
        detail = content_lower[8:].strip()
        if detail in _TRIVIAL_FACTS or detail.split()[0] in _TRIVIAL_FACTS:
            return True
    # "User uses/has: X" → check if X is trivial.
    if content_lower.startswith("user uses/has: "):
        thing = content_lower[14:].strip()
        if thing in _TRIVIAL_FACTS or len(thing) < 5:
            return True

    # Check for sentence fragments.
    if _FRAGMENT_RE.search(content):
        return True

    # Check for content that is just a partial clause ending abruptly.
    # E.g. "User uses/has: the item list, I'll search authoritative sources (e"
    if content.rstrip().endswith("(") or content.rstrip().endswith(","):
        return True
    # Unmatched parenthesis at end — sign of a truncated fragment.
    if content.count("(") > content.count(")"):
        return True

    return False


# These checks are intentionally advisory for proposals. A candidate can be
# retained for review without becoming active memory, while clearly malformed
# candidates are kept out of retrieval by the status gate in the store.
_META_REFERENCE_RE = re.compile(
    r"\b(?:previous session|new chat|this chat|this conversation|continue in|"
    r"chat dump|assistant response|you brought|symmetric with it)\b",
    re.IGNORECASE,
)
_SECOND_PERSON_RE = re.compile(r"\b(?:you|your|we|our)\b", re.IGNORECASE)
_FRAGMENT_START_RE = re.compile(r"^(?:or|and|but|because|so|which)\b", re.IGNORECASE)
_FRAGMENT_END_RE = re.compile(r"(?:\bto|\bwith|\bbecause|\band|\bbut|\bof|\bfor)$", re.IGNORECASE)
_WRONG_SUBJECT_RE = re.compile(
    r"^user\s+is\s+(?:an?\s+)?(?:ai|the assistant|the agent)\b",
    re.IGNORECASE,
)
_ASSISTANT_INSTRUCTION_RE = re.compile(
    r"\b(?:stop|help|ask|tell|remind)\s+you\b",
    re.IGNORECASE,
)
# Unanchored / unresolved subject: the memory names nobody, so it cannot be
# self-contained. Catches "the person being discussed...", "this person...",
# "the subject...", and facts opening on a bare third-person pronoun
# ("she is...", "he is...", "they are...") with no named referent.
_UNANCHORED_SUBJECT_RE = re.compile(
    r"\b(?:the\s+person(?:\s+being\s+discussed|\s+in\s+question|\s+referenced)?|"
    r"this\s+person|that\s+person|"
    r"the\s+subject(?:\s+being\s+discussed|\s+referenced|\s+in\s+question)?|"
    r"the\s+(?:man|woman|girl|boy|guy|lady)(?:\s+being\s+discussed|\s+referenced)?)\b",
    re.IGNORECASE,
)
_UNANCHORED_PRONOUN_START_RE = re.compile(r"^(?:she|he|they)\b", re.IGNORECASE)


def quality_flags_for_fact(fact: Dict[str, Any]) -> List[str]:
    """Return explainable quality flags without mutating the candidate."""
    content = str(fact.get("content", "")).strip()
    category = str(fact.get("category", "context_note"))
    flags: List[str] = []
    if _is_junk(fact):
        flags.append("syntax_junk")
    if _AGENT_SPEAK_PATTERNS.search(content):
        flags.append("assistant_language")
    if _META_REFERENCE_RE.search(content):
        flags.append("conversation_meta")
    if _WRONG_SUBJECT_RE.search(content):
        flags.append("wrong_subject")
    if _UNANCHORED_SUBJECT_RE.search(content) or _UNANCHORED_PRONOUN_START_RE.search(content):
        flags.append("unanchored_subject")
    if _ASSISTANT_INSTRUCTION_RE.search(content):
        flags.append("assistant_instruction")
    if _FRAGMENT_START_RE.search(content) or _FRAGMENT_END_RE.search(content):
        flags.append("sentence_fragment")
    if "?" in content or content.lower().startswith(("who ", "what ", "why ", "how ")):
        flags.append("question_or_request")
    if _SECOND_PERSON_RE.search(content):
        flags.append("second_person_reference")
    if category in {"context_note", "event"}:
        flags.append("short_lived_category")
    return sorted(set(flags))


def hard_quality_flags(flags: List[str]) -> List[str]:
    """Flags strong enough to keep a candidate out of active memory."""
    hard = {
        "syntax_junk", "assistant_language", "assistant_instruction",
        "wrong_subject", "sentence_fragment", "unanchored_subject",
    }
    return [flag for flag in flags if flag in hard]


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences, keeping it simple."""
    text = re.sub(r'\s+', ' ', text).strip()
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Stage 1: Generic syntactic patterns (topic-agnostic)
# ---------------------------------------------------------------------------
#
# These patterns match the *structural shape* of durable statements, not
# specific topics.  "I use React" and "I take FocusTool" both match the same
# "I use/take X" pattern.  This makes the extractor general-purpose.

# "I am/is a <something>" — identity, role, profession.
_IDENTITY_RE = re.compile(
    r'\b(?:i\s+am|i\'m)\s+(?:a\s+|an\s+)?(.+?)(?:\.|$)',
    re.IGNORECASE,
)

# "I have/own/use/take <something>" — possession, usage, tools.
_HAVE_USE_RE = re.compile(
    r'\b(?:i\s+(?:have|own|use|take|\'ve\s+got|\'m\s+using))\s+(.+?)(?:\.|$)',
    re.IGNORECASE,
)

# "I work at/for <something>" / "I'm a <job>" — work context.
_WORK_RE = re.compile(
    r'\b(?:i\s+work\s+(?:at|for|with)|i\'m\s+(?:working\s+at|employed\s+at))\s+(.+?)(?:\.|$)',
    re.IGNORECASE,
)

# "I prefer/like/love/hate/enjoy <something>" — preferences.
_PREFERENCE_RE = re.compile(
    r'\b(?:i\s+(?:prefer|like|love|hate|enjoy|can\'t\s+stand|don\'t\s+like)'
    r'|i\'d\s+rather|i\'m\s+a\s+fan\s+of)\s+(.+?)(?:\.|$)',
    re.IGNORECASE,
)

# "I always/never/usually <something>" — habits and patterns.
_HABIT_RE = re.compile(
    r'\b(?:i\s+(?:always|never|usually|typically|generally|rarely|often))\s+(.+?)(?:\.|$)',
    re.IGNORECASE,
)

# "I'm working on/trying to/going to/need to/want to <something>" — goals.
_GOAL_RE = re.compile(
    r'\b(?:i\'m\s+working\s+on|i\s+want\s+to|i\'m\s+trying\s+to'
    r'|i\'m\s+going\s+to|i\s+need\s+to|i\'m\s+planning\s+to'
    r'|i\'m\s+learning|i\'m\s+studying)\s+(.+?)(?:\.|$)',
    re.IGNORECASE,
)

# "I switched from X to Y" / "I moved from X to Y" — transitions.
_SWITCH_RE = re.compile(
    r'\b(?:i\s+(?:switched|moved|migrated|transitioned)\s+from\s+(.+?)\s+to\s+(.+?))(?:\.|$)',
    re.IGNORECASE,
)

# "I started/began/stopped/quit/resumed <something>" — events.
_EVENT_RE = re.compile(
    r'\bi\s+(started|began|stopped|quit|resumed|finished|completed|launched)\s+(.+?)(?:\.|$)',
    re.IGNORECASE,
)

# "I tend to/struggle with/realized/noticed" — insights, self-observations.
# "i've been noticing", "i think i", "i feel like i" are top-level alternatives.
_INSIGHT_RE = re.compile(
    r'\b(?:'
    r'i\s+(?:tend\s+to|struggle\s+with|realized|noticed|found\s+that|learned\s+that)'
    r'|i\'ve\s+been\s+noticing'
    r'|i\s+think\s+i'
    r'|i\s+feel\s+like\s+i'
    r')\s+'
    r'(.+?)(?:\.|$)',
    re.IGNORECASE,
)

# "X is my <relation>" — relationship introduction (any relationship).
_RELATIONSHIP_RE = re.compile(
    r'\b([A-Z][a-z]+)\s+is\s+(?:my|the)\s+(\w+(?:\s+\w+)?)\b',
)

# "My <thing> is <value>" — attributes, config, ownership.
_MY_X_IS_RE = re.compile(
    r'\bmy\s+(\w+(?:\s+\w+)?)\s+is\s+(.+?)(?:\.|$)',
    re.IGNORECASE,
)

# "I live in/am based in/am from <place>" — location.
_LOCATION_RE = re.compile(
    r'\b(?:i\s+live\s+in|i\'m\s+(?:based\s+in|from)|i\s+reside\s+in)\s+(.+?)(?:\.|$)',
    re.IGNORECASE,
)

# "I've been doing X for Y" / "I've been on X for Y" — ongoing states.
_ONGOING_RE = re.compile(
    r'\b(?:i\'ve\s+been)\s+(.+?)(?:\.|$)',
    re.IGNORECASE,
)

# "I have/own/use/take X" — generic attribute extraction.
_ATTRIBUTE_RE = re.compile(
    r'\b(?:i\s+(?:have|own|use|take)\s+)'
    r'(.+?)(?:[.,;]|$)',
    re.IGNORECASE,
)


# Words that are not valid names for relationship extraction.
_NOT_A_NAME = frozenset({
    "this", "that", "it", "there", "here", "what", "who", "where",
    "today", "tomorrow", "yesterday", "now", "then",
})


def _classify_sentence(sentence: str) -> Dict[str, Any] | None:
    """Classify a single sentence into a memory category, or None if not durable.

    Tries patterns in priority order. Returns the first match.
    """
    # Relationship: "Entity-B is my role" / "Entity-C is my manager"
    m = _RELATIONSHIP_RE.search(sentence)
    if m:
        name = m.group(1).strip()
        relation = m.group(2).strip().lower()
        if name.lower() not in _NOT_A_NAME:
            return {
                "category": "relationship",
                "content": f"{name} is the user's {relation}",
                "tags": ["relationship", name.lower(), relation],
                "payload": {"name": name, "relation": relation},
            }

    # Switch/transition: "I switched from X to Y"
    m = _SWITCH_RE.search(sentence)
    if m:
        old = m.group(1).strip().rstrip('.')
        new = m.group(2).strip().rstrip('.')
        if len(old) > 2 and len(new) > 2:
            return {
                "category": "event",
                "content": f"User switched from {old} to {new}",
                "tags": ["event", "transition"],
                "payload": {"old": old, "new": new, "event_type": "switch"},
            }

    # Attribute (checked before generic "I have" to get the more specific tag).
    m = _ATTRIBUTE_RE.search(sentence)
    if m:
        attribute = m.group(1).strip().rstrip('.')
        if len(attribute) > 3:
            return {
                "category": "personal_fact",
                "content": f"User has: {attribute}",
                "tags": ["personal_fact"],
                "payload": {"attribute": attribute, "fact_type": "attribute"},
            }

    # Work context: "I work at Google" / "I'm working at a startup"
    m = _WORK_RE.search(sentence)
    if m:
        workplace = m.group(1).strip().rstrip('.')
        if len(workplace) > 2:
            return {
                "category": "personal_fact",
                "content": f"User works at: {workplace}",
                "tags": ["personal_fact", "work"],
                "payload": {"workplace": workplace, "fact_type": "work"},
            }

    # Location: "I live in Berlin" / "I'm from Tokyo"
    m = _LOCATION_RE.search(sentence)
    if m:
        place = m.group(1).strip().rstrip('.')
        if len(place) > 2:
            return {
                "category": "personal_fact",
                "content": f"User location: {place}",
                "tags": ["personal_fact", "location"],
                "payload": {"location": place, "fact_type": "location"},
            }

    # Have/use/take: "I have a dog" / "I use Vim" / "I take FocusTool"
    m = _HAVE_USE_RE.search(sentence)
    if m:
        thing = m.group(1).strip().rstrip('.')
        if len(thing) > 3:
            return {
                "category": "personal_fact",
                "content": f"User uses/has: {thing}",
                "tags": ["personal_fact"],
                "payload": {"thing": thing, "fact_type": "have_use"},
            }

    # My X is Y: "My favorite editor is Vim" / "My role is Entity-B"
    m = _MY_X_IS_RE.search(sentence)
    if m:
        attr = m.group(1).strip().lower()
        value = m.group(2).strip().rstrip('.')
        if len(attr) > 2 and len(value) > 2:
            # If it looks like a relationship ("my role is Entity-B"), tag it.
            if attr in ("wife", "husband", "partner", "boyfriend", "girlfriend",
                        "boss", "advisor", "doctor", "teacher", "mentor",
                        "friend", "colleague", "manager", "supervisor"):
                return {
                    "category": "relationship",
                    "content": f"User's {attr} is {value}",
                    "tags": ["relationship", attr],
                    "payload": {"relation": attr, "name": value},
                }
            return {
                "category": "personal_fact",
                "content": f"User's {attr}: {value}",
                "tags": ["personal_fact", attr],
                "payload": {"attribute": attr, "value": value, "fact_type": "attribute"},
            }

    # Preference: "I prefer dark mode" / "I love Python"
    m = _PREFERENCE_RE.search(sentence)
    if m:
        pref = m.group(1).strip().rstrip('.')
        if len(pref) > 3:
            return {
                "category": "preference",
                "content": f"User prefers: {pref}",
                "tags": ["preference"],
                "payload": {"preference": pref},
            }

    # Habit: "I always test before deploying" / "I never push to main"
    m = _HABIT_RE.search(sentence)
    if m:
        habit = m.group(1).strip().rstrip('.')
        if len(habit) > 5:
            return {
                "category": "preference",
                "content": f"User habit: {habit}",
                "tags": ["preference", "habit"],
                "payload": {"habit": habit},
            }

    # Goal: "I'm working on a side project" / "I want to learn Rust"
    m = _GOAL_RE.search(sentence)
    if m:
        goal = m.group(1).strip().rstrip('.')
        if len(goal) > 5:
            return {
                "category": "goal",
                "content": f"User goal: {goal}",
                "tags": ["goal"],
                "payload": {"goal": goal},
            }

    # Insight / self-observation: "I tend to overthink" / "I realized I need breaks"
    m = _INSIGHT_RE.search(sentence)
    if m:
        insight = m.group(1).strip().rstrip('.')
        if len(insight) > 5:
            return {
                "category": "insight",
                "content": f"User self-observation: {insight}",
                "tags": ["insight", "self_observation"],
                "payload": {"insight": insight},
            }

    # Event: "I started a new job" / "I quit smoking" / "I launched the app"
    m = _EVENT_RE.search(sentence)
    if m:
        verb = m.group(1).strip().lower()
        event = m.group(2).strip().rstrip('.')
        if len(event) > 5:
            return {
                "category": "event",
                "content": f"Life event: user {verb} {event}",
                "tags": ["event"],
                "payload": {"event": event, "verb": verb},
            }

    # Ongoing: "I've been feeling anxious" / "I've been using Docker"
    m = _ONGOING_RE.search(sentence)
    if m:
        ongoing = m.group(1).strip().rstrip('.')
        if len(ongoing) > 5:
            return {
                "category": "context_note",
                "content": f"User has been {ongoing}",
                "tags": ["context_note", "ongoing"],
                "payload": {"ongoing": ongoing},
            }

    # Identity (last — most generic "I am X"): "I am a developer" / "I'm 34"
    m = _IDENTITY_RE.search(sentence)
    if m:
        detail = m.group(1).strip().rstrip('.')
        # Filter out transient states ("I am tired", "I am here", "I am ready",
        # "I am tired and hungry right now"). Check both exact match and
        # whether the detail starts with a transient word.
        _TRANSIENT_WORDS = frozenset({
            "tired", "hungry", "thirsty", "here", "there", "ready", "done",
            "fine", "ok", "okay", "good", "great", "busy", "free", "back",
            "sorry", "glad", "happy", "sad", "angry", "confused", "stuck",
            "not sure", "not certain", "not really", "not happy",
            "sick", "bored", "excited", "worried", "anxious", "stressed",
            "exhausted", "overwhelmed", "frustrated", "annoyed", "grateful",
        })
        detail_lower = detail.lower()
        first_word = detail_lower.split()[0] if detail_lower else ""
        is_transient = (
            detail_lower in _TRANSIENT_WORDS
            or first_word in _TRANSIENT_WORDS
        )
        if len(detail) > 5 and not is_transient:
            return {
                "category": "personal_fact",
                "content": f"User is {detail}",
                "tags": ["personal_fact", "identity"],
                "payload": {"detail": detail, "fact_type": "identity"},
            }

    return None


def _extract_facts_regex(user_content: str) -> List[Dict[str, Any]]:
    """Stage 1: Extract candidate facts using generic syntactic patterns."""
    facts: List[Dict[str, Any]] = []
    if not user_content or len(user_content.strip()) < _MIN_LENGTH:
        return facts

    sentences = _split_sentences(user_content)[:_MAX_SENTENCES]
    for sentence in sentences:
        if len(sentence) < _MIN_LENGTH:
            continue
        fact = _classify_sentence(sentence)
        if fact:
            fact.setdefault("source", "regex_extraction")
            fact.setdefault("confidence", 0.75)
            fact.setdefault(
                "durability",
                "temporary" if fact.get("category") in {"context_note", "event", "goal"} else "durable",
            )
            fact.setdefault("scope", "profile")
            facts.append(fact)
    return facts


# ---------------------------------------------------------------------------
# Stage 2: LLM fallback extraction
# ---------------------------------------------------------------------------

_LLM_SYSTEM_PROMPT = """You are a memory extraction assistant. Your job is to extract durable facts from the user's message that would be worth remembering for future conversations.

Extract facts that are:
- Durable (will still be true tomorrow, not transient)
- Self-contained (make sense without the original conversation)
- About the user or things in the user's life

Do NOT extract:
- Transient states ("I'm tired", "I'm busy right now")
- Questions or requests
- Opinions about the assistant
- Greetings or chit-chat

Return a JSON array of objects with these keys:
- "category": one of "personal_fact", "preference", "insight", "event", "relationship", "goal", "context_note"
- "content": a clear, self-contained statement of the fact
- "tags": array of 1-3 short lowercase tags
- "confidence": number from 0 to 1
- "durability": "permanent", "durable", or "temporary"
- "scope": "profile", "project", or "session"

If there are no durable facts, return an empty array: []

Examples:
User: "I just got a new job at Stripe, I'll be starting next Monday as a backend engineer"
Output: [{"category": "event", "content": "User got a new job at Stripe as a backend engineer, starting next Monday", "tags": ["work", "job", "stripe"]}]

User: "My contact suggested I try journaling every morning"
Output: [{"category": "context_note", "content": "User's contact suggested morning journaling", "tags": ["journaling"]}]

User: "I've been deploying with Kubernetes lately, it's way better than Docker Swarm for our scale"
Output: [{"category": "preference", "content": "User prefers Kubernetes over Docker Swarm for deployment at scale", "tags": ["devops", "kubernetes"]}, {"category": "personal_fact", "content": "User deploys with Kubernetes", "tags": ["devops", "kubernetes"]}]

User: "hey how are you"
Output: []
"""


def _extract_facts_llm(user_content: str, *, model: str = "", provider: str = "") -> List[Dict[str, Any]]:
    """Stage 2: Extract facts using the host's LLM via call_llm.

    Returns a list of fact dicts, or empty list on any failure.
    Never raises — all errors are caught and logged.
    """
    if not user_content or len(user_content.strip()) < _LLM_MIN_CONTENT_LENGTH:
        return []
    from egress import gate as _egress_gate
    if not _egress_gate("extractor", user_content):
        return []


    try:
        from agent.auxiliary_client import call_llm
    except ImportError:
        logger.debug("LLM fallback unavailable: agent.auxiliary_client not importable")
        return []
    except Exception as e:
        logger.debug("LLM fallback unavailable: %s", e)
        return []

    messages = [
        {"role": "system", "content": _LLM_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    try:
        response = call_llm(
            task="memory_extraction",
            messages=messages,
            temperature=0.0,
            max_tokens=800,
            timeout=_LLM_TIMEOUT,
            model=model or None,
            provider=provider or None,
        )
    except Exception as e:
        logger.debug("LLM extraction call failed: %s", e)
        return []

    if response is None:
        return []

    try:
        text = response.choices[0].message.content
    except (AttributeError, IndexError, KeyError):
        return []

    if not text or not text.strip():
        return []

    # Parse JSON — handle both raw JSON and JSON wrapped in markdown code fences.
    text = text.strip()
    if text.startswith("```"):
        # Strip markdown code fences.
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        text = text.strip()

    try:
        facts = json.loads(text)
    except json.JSONDecodeError:
        logger.debug("LLM extraction returned non-JSON: %s", text[:100])
        return []

    if not isinstance(facts, list):
        return []

    # Validate and normalize each fact.
    valid_categories = {
        "personal_fact", "preference", "insight",
        "event", "relationship", "goal", "context_note",
    }
    result: List[Dict[str, Any]] = []
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        category = fact.get("category", "context_note")
        if category not in valid_categories:
            category = "context_note"
        content = fact.get("content", "").strip()
        if not content or len(content) < 5:
            continue
        tags = fact.get("tags", [])
        if not isinstance(tags, list):
            tags = [str(tags)] if tags else []
        tags = [str(t).lower().strip() for t in tags if t][:5]
        try:
            confidence = max(0.0, min(1.0, float(fact.get("confidence", 0.45))))
        except (TypeError, ValueError):
            confidence = 0.45
        durability = str(fact.get("durability", "durable")).lower()
        if durability not in {"permanent", "durable", "temporary"}:
            durability = "durable"
        scope = str(fact.get("scope", "profile")).lower()
        if scope not in {"profile", "project", "session"}:
            scope = "profile"
        result.append({
            "category": category,
            "content": content,
            "tags": tags,
            "source": "llm_extraction",
            "confidence": confidence,
            "durability": durability,
            "scope": scope,
            "payload": {"source": "llm_extraction"},
        })

    return result


# ---------------------------------------------------------------------------
# LLM fallback decision + cross-stage dedup
# ---------------------------------------------------------------------------

def _should_try_llm_fallback(user_content: str, fact_count: int) -> bool:
    """Decide whether to invoke the LLM fallback after regex extraction.

    The LLM call costs latency + tokens, so we only invoke it when regex
    extraction looks insufficient for the message length:

    - If regex found 0 facts and the message is substantial: always try LLM.
    - If regex found fewer than _LLM_FALLBACK_MIN_FACTS facts: try LLM
      (the message might have more durable facts the regex missed).
    - If the words-per-fact ratio is high (message is long but regex found
      few facts): try LLM — regex likely missed things.
    - Otherwise (regex found several facts from a short message): skip LLM.
    """
    if not user_content or len(user_content.strip()) < _LLM_MIN_CONTENT_LENGTH:
        return False
    if fact_count == 0:
        return True
    if fact_count < _LLM_FALLBACK_MIN_FACTS:
        return True
    word_count = len(user_content.split())
    if word_count / max(1, fact_count) > _LLM_FALLBACK_WORDS_PER_FACT:
        return True
    return False


def _text_overlap(a: str, b: str) -> bool:
    """Check if two text strings are near-duplicates (high token overlap).

    Uses Jaccard similarity on word sets — fast and dependency-free.
    Threshold of 0.6 means 60% of words are shared (order-independent).
    Used to dedup LLM-extracted facts against regex-extracted facts so
    we don't store the same fact twice with different phrasing.
    """
    if not a or not b:
        return False
    words_a = set(a.split())
    words_b = set(b.split())
    if not words_a or not words_b:
        return False
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union) >= 0.6


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_from_turn(
    user_content: str,
    assistant_content: str,
    *,
    use_llm_fallback: bool = True,
    llm_model: str = "",
    llm_provider: str = "",
    shadow_diff: bool = False,
) -> List[Dict[str, Any]]:
    """Extract candidate facts from a completed turn.

    Two-stage extraction:
    1. Fast regex patterns (always run, zero latency).
    2. LLM fallback (only if Stage 1 found < threshold facts AND the
       message is substantial AND use_llm_fallback is True).

    Only extracts from the USER's content — the assistant's content is
    the agent's own output and not a source of durable user facts.

    When *shadow_diff* is True, always runs LLM extraction in parallel
    and logs the diff (what LLM found that regex didn't, and vice versa).
    The actual proposals are NOT changed — this is a validation mode for
    evaluating whether LLM-first extraction would improve recall. The diff
    is logged at INFO level with structured fields for offline analysis.
    """
    # Stage 1: regex patterns.
    facts = _extract_facts_regex(user_content)

    # Shadow-diff mode: always run LLM extraction and compare.
    # Does NOT change the returned facts — purely attributetic.
    if shadow_diff and use_llm_fallback and len(user_content.strip()) >= _LLM_MIN_CONTENT_LENGTH:
        try:
            llm_facts_shadow = _extract_facts_llm(user_content, model=llm_model, provider=llm_provider)
            _log_shadow_diff(user_content, facts, llm_facts_shadow)
        except Exception as exc:
            logger.debug("Shadow-diff LLM extraction failed: %s", exc)

    # Stage 2: LLM fallback if regex didn't find enough.
    # The LLM is expensive (latency + tokens), so we only call it when
    # regex extraction looks insufficient for the message length.
    if use_llm_fallback and _should_try_llm_fallback(user_content, len(facts)):
        llm_facts = _extract_facts_llm(user_content, model=llm_model, provider=llm_provider)
        if llm_facts:
            # Dedup LLM facts against regex facts by content similarity.
            existing_contents = [f["content"].lower() for f in facts]
            for lf in llm_facts:
                lf_lower = lf["content"].lower()
                # Skip if an existing regex fact is a near-duplicate.
                if not any(_text_overlap(lf_lower, ec) for ec in existing_contents):
                    facts.append(lf)

    # Filter only irrecoverably malformed candidates here. Other quality flags
    # travel with the proposal so a reviewer can make an informed decision.
    normalized: List[Dict[str, Any]] = []
    for fact in facts:
        if _is_junk(fact):
            continue
        payload = dict(fact.get("payload") or {})
        flags = quality_flags_for_fact(fact)
        if flags:
            payload["quality_flags"] = flags
        fact["payload"] = payload
        fact.setdefault("source", "regex_extraction")
        fact.setdefault("confidence", 0.75 if fact["source"] == "regex_extraction" else 0.45)
        fact.setdefault("durability", "durable")
        fact.setdefault("scope", "profile")
        normalized.append(fact)

    return normalized


def _log_shadow_diff(
    user_content: str,
    regex_facts: List[Dict[str, Any]],
    llm_facts: List[Dict[str, Any]],
) -> None:
    """Log the diff between regex and LLM extraction for offline analysis.

    Logs structured fields:
    - regex_count: number of regex-extracted facts
    - llm_count: number of LLM-extracted facts
    - llm_only: facts the LLM found that regex missed (potential recall gain)
    - regex_only: facts regex found that the LLM missed (potential precision gain)
    - content_preview: first 80 chars of the user message
    """
    regex_contents = {f.get("content", "").lower() for f in regex_facts}
    llm_contents = {f.get("content", "").lower() for f in llm_facts}

    llm_only = [f for f in llm_facts if f.get("content", "").lower() not in regex_contents]
    regex_only = [f for f in regex_facts if f.get("content", "").lower() not in llm_contents]

    # Use near-duplicate matching for more accurate diff
    llm_only_real: List[Dict[str, Any]] = []
    for lf in llm_facts:
        lf_lower = lf.get("content", "").lower()
        if not any(_text_overlap(lf_lower, rc) for rc in regex_contents):
            llm_only_real.append(lf)

    regex_only_real: List[Dict[str, Any]] = []
    for rf in regex_facts:
        rf_lower = rf.get("content", "").lower()
        if not any(_text_overlap(rf_lower, lc) for lc in llm_contents):
            regex_only_real.append(rf)

    logger.info(
        "SHADOW_DIFF regex=%d llm=%d llm_only=%d regex_only=%d preview=%.80s",
        len(regex_facts),
        len(llm_facts),
        len(llm_only_real),
        len(regex_only_real),
        user_content.replace("\n", " ")[:80],
    )
    for fact in llm_only_real:
        logger.info(
            "SHADOW_DIFF_LLM_ONLY content=%.120s category=%s",
            fact.get("content", "")[:120],
            fact.get("category", "?"),
        )
    for fact in regex_only_real:
        logger.info(
            "SHADOW_DIFF_REGEX_ONLY content=%.120s category=%s",
            fact.get("content", "")[:120],
            fact.get("category", "?"),
        )
