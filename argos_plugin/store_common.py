"""Shared store-level constants, sanitizers, trust helpers and the record model.

Extracted verbatim from store.py during the god-file split (behavior-neutral:
no renames, no fixes). Deliberately has no local imports: it loads identically
as ``argos_plugin.store_common`` (package form) and ``store_common`` (top-level
sibling form used by tests and the benchmark clone).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

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

# Shared tokenizer regex (issue #26): text search and phrase-lift must split
# text identically so contractions ("don't", "it's") tokenize the same in both
# paths. The apostrophe is included so contractions survive as single tokens;
# the prior [a-z0-9]+ regex split "don't" into "don" + "t", which silently
# limited phrase-lift recall on common contractions and made the two rankers
# disagree about what the text contains.
_TOKEN_RE = re.compile(r"[a-z0-9']+", re.IGNORECASE)


def _tokenize(text: str) -> List[str]:
    """Tokenize *text* into lowercase word tokens.

    Shared between BM25-lite text search and phrase-lift so both paths see
    the same token boundaries (issue #26).
    """
    return [m.group().lower() for m in _TOKEN_RE.finditer(text or "")]


# --- prompt-injection / hidden-content hardening (2026-08-27) ---------------
# Stored memory is later replayed verbatim into prompts (auto-injection and
# memory_search results). Content that mimics instructions must not enter the
# store silently. These are heuristic regexes — deliberately conservative:
# a false positive lands in quarantine (recoverable), never in the active
# store, and never makes an LLM call.
_HIDDEN_CHARS_RE = re.compile(r"[\u00ad\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]")
_FORMAT_SPACES_RE = re.compile(r"[\u2000-\u200a\u202f\u205f]")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

# (compiled pattern, human label). First match wins; the label is stored in
# quarantine_reason / raised in ValueError messages for audit. All patterns
# are case-insensitive — payloads arrive in any casing.
def _ci(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.IGNORECASE)

_INJECTION_PATTERNS: List[tuple] = [
    (_ci(r"ignore\s+((all|the|any)\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|messages?|commands?|rules)"), "instruction_override"),
    (_ci(r"disregard\s+(the\s+)?(previous|prior|above|earlier)"), "disregard_previous"),
    (_ci(r"do\s+not\s+(tell|reveal|show|mention|say)\s+(the\s+)?(user|them|him|her)"), "conceal_from_user"),
    (_ci(r"reveal\s+(your\s+)?(system\s+prompt|internal\s+instructions?|instructions?)"), "prompt_reveal"),
    (_ci(r"you\s+are\s+(now\s+)?(an?|no\s+longer)\b"), "identity_shift"),
    (_ci(r"repeat\s+(after\s+me|the\s+following)"), "repeat_after_me"),
    (_ci(r"\[system\s+note\s*:"), "fence_spoof"),
    (_ci(r"\bjailbreak\b"), "jailbreak_ref"),
    (_ci(r"\bDAN\b\s+(mode|activated|is)\b"), "dan_mode"),
    (_ci(r"(forget|erase|wipe)\s+(everything|all)\s+(you\s+(know|learned)|your\s+(memories?|instructions?))"), "memory_wipe"),
]


def sanitize_content(content: str) -> tuple:
    """Normalize content for storage and sniff for instruction-like text.

    Returns ``(cleaned_text, matched_label_or_None)``. Cleaning removes
    zero-width/format/control characters used to hide text from humans
    (white-on-white PDF footers, invisible CJK joiners, BOMs). A non-None
    label means callers must refuse (direct saves/updates) or quarantine
    (proposals) — never store the text as active memory.
    """
    if not content:
        return content, None
    clean = _HIDDEN_CHARS_RE.sub("", content)
    clean = _FORMAT_SPACES_RE.sub(" ", clean)
    clean = _CONTROL_CHARS_RE.sub("", clean)
    for pattern, label in _INJECTION_PATTERNS:
        if pattern.search(clean):
            return clean, label
    return clean, None

VALID_CATEGORIES = frozenset({
    "personal_fact",
    "preference",
    "insight",
    "event",
    "relationship",
    "goal",
    "context_note",
        "negative",
    })

_DEFAULT_TTL_DAYS = {
    "context_note": 30,
    "event": 180,
    "goal": 180,
}

# Sentinel for "parameter not provided" — distinguishes explicit None
# (clear/revive expiry) from "caller didn't pass this kwarg" (carry forward).
_NOT_PROVIDED = object()


# --- trust-model: provenance taint, grounding, rejection ledger (batch-2) ----
# Three record-level metadata fields that gate what a memory may do or become.
# Designed together (one schema pass) per the trust-model cluster consolidation
# (issues #43, #40, #39). Issue #35 (quote verification) feeds #40's grounding.

# Provenance origin (#43): binary label, fail-closed to the stricter class.
# Unknown/corrupt values parse to "external" so a missing or tampered label can
# never widen a memory's blast radius. Sanitization/redaction does NOT alter it.
PROVENANCE_INTERNAL = "internal"
PROVENANCE_EXTERNAL = "external"
_VALID_PROVENANCE = frozenset({PROVENANCE_INTERNAL, PROVENANCE_EXTERNAL})


def normalize_provenance(value: Any) -> str:
    """Fail-closed provenance parser. Unknown/corrupt -> external (stricter)."""
    v = str(value or "").strip().lower()
    if v == PROVENANCE_INTERNAL:
        return PROVENANCE_INTERNAL
    # Everything else (external, "", None, garbage) fails closed to external.
    return PROVENANCE_EXTERNAL


# Grounding (#40): how a record was obtained; caps how trusted it can become.
# Ladder (low -> high): speculative < inferred < extracted < observed.
# Promotion/confirmation may raise a record's class but never above what its
# grounding allows; recall counts are NOT verification and never raise class.
GROUNDING_SPECULATIVE = "speculative"   # hypothesis, unverified
GROUNDING_INFERRED = "inferred"         # model- or distill-derived
GROUNDING_EXTRACTED = "extracted"       # parsed/extracted from a user turn
GROUNDING_OBSERVED = "observed"         # direct user statement
_GROUNDING_ORDER = {
    GROUNDING_SPECULATIVE: 0,
    GROUNDING_INFERRED: 1,
    GROUNDING_EXTRACTED: 2,
    GROUNDING_OBSERVED: 3,
}
_VALID_GROUNDING = frozenset(_GROUNDING_ORDER)


def normalize_grounding(value: Any) -> str:
    """Fail-closed grounding parser. Unknown/corrupt -> speculative (strictest)."""
    v = str(value or "").strip().lower()
    if v in _GROUNDING_ORDER:
        return v
    return GROUNDING_SPECULATIVE


def grounding_rank(value: Any) -> int:
    return _GROUNDING_ORDER.get(normalize_grounding(value), 0)


# Trust-class ceiling per grounding (#40). A record may be promoted up to its
# ceiling but never past it via recall or auto-review. User confirmation raises
# the grounding (lifts the ceiling); recall counts do not.
# Status ladder (low -> high): quarantined/rejected < pending_user_confirmation
#   < reviewed_approved < approved.
_TRUST_CLASS_ORDER = {
    "quarantined": 0,
    "rejected": 0,
    "pending_user_confirmation": 1,
    "reviewed_approved": 2,
    "approved": 3,
}
_GROUNDING_CEILING = {
    GROUNDING_SPECULATIVE: "pending_user_confirmation",
    GROUNDING_INFERRED: "reviewed_approved",
    GROUNDING_EXTRACTED: "approved",
    GROUNDING_OBSERVED: "approved",
}


def trust_class_rank(status: Any) -> int:
    return _TRUST_CLASS_ORDER.get(str(status or "").strip().lower(), 0)


def grounding_allows_status(grounding: Any, status: Any) -> bool:
    """True if *grounding* permits reaching *status* (the ceiling check)."""
    ceiling = _GROUNDING_CEILING.get(
        normalize_grounding(grounding), "pending_user_confirmation"
    )
    return trust_class_rank(status) <= trust_class_rank(ceiling)


def default_grounding_for_write(*, source: str = "", external: bool = False,
                                explicit_grounding: Any = None) -> str:
    """Default grounding per write path (#40).

    - direct user statement (source=explicit, not external) -> observed
    - parsed/extracted (source=llm_extraction)              -> extracted
    - distill/model-derived (source=distillation)           -> inferred
    - external-origin ingest                                -> inferred
    - anything else                                         -> speculative
    An explicit grounding from the caller always wins (after normalization).
    """
    if explicit_grounding is not None:
        return normalize_grounding(explicit_grounding)
    src = str(source or "").strip().lower()
    if external:
        return GROUNDING_INFERRED
    if src in {"explicit", "user", "manual"}:
        return GROUNDING_OBSERVED
    if src in {"llm_extraction", "extraction"}:
        return GROUNDING_EXTRACTED
    if src in {"distillation", "distill"}:
        return GROUNDING_INFERRED
    return GROUNDING_SPECULATIVE


# Rejection ledger (#39): one-way trust ladder. A rejected value's identity
# (subject, predicate, scope) is recorded; no approval path may resurrect it
# without a NEW record passing the same gates. Keyed by (subject, predicate,
# scope) so paraphrased re-assertions of the same claim slot are also blocked.
def rejection_key(candidate: dict) -> tuple:
    """Derive a (subject, predicate, scope) identity from a candidate/record.

    subject:   the entity the fact is about (a named person from payload, else
               "user" for self-referential facts).
    predicate: category + the specific claim slot (attribute / fact_type /
               relation / preference / ...) so "user/age" and "user/location"
               are distinct slots.
    scope:     user_scope (defaults to default_user).
    """
    if not isinstance(candidate, dict):
        return ("", "", "default_user")
    payload = candidate.get("payload")
    payload = payload if isinstance(payload, dict) else {}
    category = str(candidate.get("category", "") or "").strip().lower()
    subject = (
        str(payload.get("name") or "").strip().lower()
        or str(payload.get("workplace") or "").strip().lower()
        or "user"
    )
    slot = (
        str(payload.get("attribute") or "").strip().lower()
        or str(payload.get("fact_type") or "").strip().lower()
        or str(payload.get("relation") or "").strip().lower()
        or str(payload.get("preference") or "").strip().lower()
        or str(payload.get("goal") or "").strip().lower()
        or str(payload.get("insight") or "").strip().lower()
        or str(payload.get("event") or "").strip().lower()
    )
    predicate = f"{category}:{slot}" if slot else category
    scope = str(
        candidate.get("user_scope") or payload.get("user_scope") or "default_user"
    ).strip().lower()
    return (subject, predicate, scope)


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
        "provenance_origin", "grounding",
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
        provenance_origin: str = PROVENANCE_INTERNAL,
        grounding: str = GROUNDING_OBSERVED,
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
        # Trust-model (batch-2): provenance taint (#43) and grounding (#40).
        # Both fail-closed at parse time; see normalize_provenance /
        # normalize_grounding. Sanitization never alters provenance_origin.
        self.provenance_origin = normalize_provenance(provenance_origin)
        self.grounding = normalize_grounding(grounding)

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
            "provenance_origin": self.provenance_origin,
            "grounding": self.grounding,
        }
