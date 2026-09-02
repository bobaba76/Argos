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
import threading
from pathlib import Path
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
# --- ReDoS guards (#89) -------------------------------------------------------
# Total content length cap: a turn longer than this is truncated before the
# regex stage so a 50k-char paste cannot feed unbounded input to the
# patterns. The LLM fallback still sees the full (egress-gated) content.
_MAX_REGEX_CONTENT_CHARS = 50_000
# Per-sentence length cap: naive sentence splitting (#15) treats a long
# unpunctuated run as one "sentence". Cap each sentence before regex
# matching so the bounded quantifiers below never see pathological input.
_MAX_SENTENCE_CHARS = 2_000
# Regex stage watchdog timeout (seconds). The bounded quantifiers + length
# caps make catastrophic backtracking unlikely, but a thread-timeout is
# defense-in-depth — a hang in the regex engine must never stall the turn.
_REGEX_STAGE_TIMEOUT_S = 5.0
# Module-level counter for extraction failures (surfaced in review).
_EXTRACTION_FAILURES = 0


def get_extraction_failure_stats() -> Dict[str, int]:
    """Counter for extraction-stage failures (surfaced in review, #89/#85)."""
    return {"extraction_failures": _EXTRACTION_FAILURES}


def _reset_extraction_failure_stats() -> None:
    """Test hook: reset the failure counter."""
    global _EXTRACTION_FAILURES
    _EXTRACTION_FAILURES = 0


# --- Pattern pack loader (#134) ---------------------------------------------
# Patterns and lexicons live in config data files
# (extractor_patterns/<locale>.json) so domain packs and locale packs can
# be added without code changes. Loaded once at module init (build-once,
# auditable). Falls back to inline defaults if the data file is missing or
# unreadable — extraction must never crash on a missing config file.
#
# ReDoS audit (#134): the patterns are stored as complete regex strings
# (not dynamically-built alternations), so the bounded {0,200} tails,
# per-sentence/per-content caps, and the _REGEX_STAGE_TIMEOUT_S watchdog
# all survive unchanged. Dynamic alternation from lexicons is #136, which
# must re.escape() every member and preserve the bounded tails.

_PATTERNS_DIR = Path(__file__).resolve().parent / "extractor_patterns"

_FLAG_MAP = {
    "IGNORECASE": re.IGNORECASE,
    "DOTALL": re.DOTALL,
    "MULTILINE": re.MULTILINE,
    "VERBOSE": re.VERBOSE,
}


def _compile_flags(flag_names: list[str]) -> int:
    """Convert flag-name list to a combined re flag integer."""
    flags = 0
    for name in flag_names:
        flags |= _FLAG_MAP.get(name, 0)
    return flags


def _load_pattern_pack(locale: str = "en") -> tuple[Dict[str, "re.Pattern"], Dict[str, frozenset]]:
    """Load compiled patterns and lexicons for *locale* from the data file.

    Returns ``(patterns, lexicons)`` where *patterns* maps name → compiled
    regex and *lexicons* maps name → frozenset. Falls back to inline
    defaults if the file is missing or unreadable (defense-in-depth).
    """
    path = _PATTERNS_DIR / f"{locale}.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Pattern pack %s not loaded (%s); using inline defaults", path, exc)
        return _inline_pattern_defaults()

    # E1: wrap pattern compilation + lexicon loading in try/except so a
    # malformed pack (invalid regex, missing key, bad lexicon type) falls
    # back to inline defaults instead of crashing the module import.
    # Blast radius of a crash: provider_session.py imports the extractor
    # inside try/except ImportError -- but re.error/KeyError/TypeError are
    # NOT ImportError, so they propagate and kill the provider import ->
    # the memory provider silently never loads.
    try:
        patterns: Dict[str, "re.Pattern"] = {}
        for name, spec in raw.get("patterns", {}).items():
            patterns[name] = re.compile(spec["pattern"], _compile_flags(spec.get("flags", [])))
        lexicons: Dict[str, frozenset] = {}
        for name, entries in raw.get("lexicons", {}).items():
            lexicons[name] = frozenset(entries)

        # Validate that all required keys are present. A partial pack
        # (missing a pattern or lexicon referenced by key at module level)
        # would crash with KeyError -- fall back instead.
        inline_p, inline_l = _inline_pattern_defaults()
        for key in inline_p:
            if key not in patterns:
                raise KeyError(f"pattern pack missing required pattern: {key}")
        for key in inline_l:
            if key not in lexicons:
                raise KeyError(f"pattern pack missing required lexicon: {key}")

        return patterns, lexicons
    except Exception as exc:
        logger.warning(
            "Pattern pack %s has malformed patterns/lexicons (%s); "
            "using inline defaults", path, exc,
        )
        return _inline_pattern_defaults()


def _inline_pattern_defaults() -> tuple[Dict[str, "re.Pattern"], Dict[str, frozenset]]:
    """Inline fallback patterns/lexicons, used when the JSON data file is
    unavailable. Kept in sync with ``extractor_patterns/en.json``."""
    patterns = {
        "IDENTITY_RE": re.compile(r'\b(?:i\s+am|i\'m)\s+(?:a\s+|an\s+)?(.+?)(?:\.|$)', re.IGNORECASE),
        "HAVE_USE_RE": re.compile(r'\b(?:i\s+(?:have|own|use|take|\'ve\s+got|\'m\s+using))\s+(.+?)(?:\.|$)', re.IGNORECASE),
        "WORK_RE": re.compile(r'\b(?:i\s+work\s+(?:at|for|with)|i\'m\s+(?:working\s+at|employed\s+at))\s+(.+?)(?:\.|$)', re.IGNORECASE),
        # PREFERENCE_RE and EVENT_RE are NOT here — they are built
        # dynamically by _build_preference_re() / _build_event_re() to
        # support runtime verb extension (#136). Do not add them to the
        # JSON pack or inline defaults — they are dead config that
        # misleads editors into thinking they're changing behavior.
        "ASSISTANT_DIRECTIVE_RE": re.compile(r'\b(?:(?:always|never)\s+(?:give|say|use|respond|reply|write|speak|talk|call|address|refer|summarise|summarize|explain|include|mention|answer)\b[^.!?]{0,200}|(?:do\s+not|don\'t)\s+(?:ever\s+|please\s+|just\s+|always\s+)?(?:give|say|use|respond|reply|write|speak|talk|call|address|refer|summarise|summarize|explain|include|mention|answer)\b[^.!?]{0,200}|call\s+me\s+[^.!?]{1,200}|stop\s+(?:doing|being|using|saying)\b[^.!?]{0,200})', re.IGNORECASE),
        "HABIT_RE": re.compile(r'\b(?:i\s+(?:always|never|usually|typically|generally|rarely|often))\s+(.+?)(?:\.|$)', re.IGNORECASE),
        "GOAL_RE": re.compile(r'\b(?:i\'m\s+working\s+on|i\s+want\s+to|i\'m\s+trying\s+to|i\'m\s+going\s+to|i\s+need\s+to|i\'m\s+planning\s+to|i\'m\s+learning|i\'m\s+studying)\s+(.+?)(?:\.|$)', re.IGNORECASE),
        "SWITCH_RE": re.compile(r'\b(?:i\s+(?:switched|moved|migrated|transitioned)\s+from\s+(.+?)\s+to\s+(.+?))(?:\.|$)', re.IGNORECASE),
        # EVENT_RE: built dynamically by _build_event_re() — see comment above.
        "INSIGHT_RE": re.compile(r'\b(?:i\s+(?:tend\s+to|struggle\s+with|realized|noticed|found\s+that|learned\s+that)|i\'ve\s+been\s+noticing|i\s+think\s+i|i\s+feel\s+like\s+i)\s+(.+?)(?:\.|$)', re.IGNORECASE),
        "RELATIONSHIP_RE": re.compile(r'\b([A-Z][a-z]+)\s+is\s+(?:my|the)\s+(\w+(?:\s+\w+)?)\b'),
        "MY_X_IS_RE": re.compile(r'\bmy\s+(\w+(?:\s+\w+)?)\s+is\s+(.+?)(?:\.|$)', re.IGNORECASE),
        "LOCATION_RE": re.compile(r'\b(?:i\s+live\s+in|i\'m\s+(?:based\s+in|from)|i\s+reside\s+in)\s+(.+?)(?:\.|$)', re.IGNORECASE),
        "ONGOING_RE": re.compile(r'\b(?:i\'ve\s+been)\s+(.+?)(?:\.|$)', re.IGNORECASE),
        "ATTRIBUTE_RE": re.compile(r'\b(?:i\s+(?:have|own)\s+)(.+?)(?:[.,;]|$)', re.IGNORECASE),
        "AGENT_SPEAK_PATTERNS": re.compile(r"(?:i'll\s+search|i'll\s+look|i'll\s+check|let me|authoritative\s+sources|i'll\s+find|i'll\s+research|searching\s+for|looking\s+up|completed\s+implementation\s+of|now\s+includes\s+\d+\s+tools|ships\s+off\s+by\s+default|deployment\s+reference\s+specifies|handoff\s+doc\s+is\s+located|implementation\s+of\s+evolution|work\s+items?\s+(?:from\s+the\s+handoff|complete)|service\s+\(pids?\s+\d+)", re.IGNORECASE),
    }
    lexicons = {
        "TRANSIENT_WORDS": frozenset({"tired", "hungry", "thirsty", "here", "there", "ready", "done", "fine", "ok", "okay", "good", "great", "busy", "free", "back", "sorry", "glad", "happy", "sad", "angry", "confused", "stuck", "not sure", "not certain", "not really", "not happy", "sick", "bored", "excited", "worried", "anxious", "stressed", "exhausted", "overwhelmed", "frustrated", "annoyed", "grateful"}),
        "NOT_A_NAME": frozenset({"this", "that", "it", "there", "here", "what", "who", "where", "today", "tomorrow", "yesterday", "now", "then"}),
        "BASE_ROLE_WORDS": frozenset({"wife", "husband", "partner", "boyfriend", "girlfriend", "boss", "advisor", "doctor", "teacher", "mentor", "friend", "colleague", "manager", "supervisor"}),
        "TRIVIAL_FACTS": frozenset({"literally glm 5", "literally glm5", "a human", "a person", "typing", "using a keyboard", "on a computer", "online", "here", "there", "ready", "done", "back"}),
        "ABBREVIATIONS": frozenset({"inc", "ltd", "pty", "corp", "co", "no", "st", "ave", "blvd", "vs", "e.g", "i.e", "etc", "approx", "mr", "mrs", "ms", "dr", "prof", "sr", "jr"}),
        "PREFERENCE_VERBS": frozenset({"prefer", "like", "love", "hate", "enjoy", "dislike"}),
        "EVENT_VERBS": frozenset({"started", "began", "stopped", "quit", "resumed", "finished", "completed", "launched"}),
    }
    return patterns, lexicons


# Load the default (en) pattern pack once at module init.
_PATTERNS, _LEXICONS = _load_pattern_pack("en")


# --- quote verification (#35, batch-2) ---------------------------------------
# The verbatim/quote label is LLM-claimed; nothing checked that the quote
# actually appears in the source conversation. This deterministic check greps
# the claimed quote against the source transcript; on miss, the item is
# downgraded from verbatim to inferred (landing on #40's grounding field) and
# the failure is logged with a countable counter surfaced in review. Cheap,
# zero LLM. Normalization: whitespace/case-insensitive substring match with a
# small tolerance for near-misses (punctuation/whitespace differences).
_QUOTE_MISSES = 0  # module-level counter; read via get_quote_verification_stats()


def get_quote_verification_stats() -> Dict[str, int]:
    """Counters for the quote-verification gate (surfaced in review)."""
    return {"quote_verification_misses": _QUOTE_MISSES}


def _reset_quote_verification_stats() -> None:
    """Test hook: reset the miss counter."""
    global _QUOTE_MISSES
    _QUOTE_MISSES = 0


def _normalize_for_quote_match(text: str) -> str:
    """Normalize text for quote matching: lowercase, collapse whitespace,
    strip common quote marks and surrounding punctuation."""
    if not text:
        return ""
    # Strip the quote characters an LLM might wrap a verbatim quote in, plus
    # all punctuation that varies between transcript and claim (trailing ! vs .,
    # commas, etc.). Keep alphanumerics and spaces only.
    stripped = re.sub(r"[`\"‘’“”«»„‟'']", " ", str(text))
    stripped = re.sub(r"[^\w\s]", " ", stripped, flags=re.UNICODE)
    stripped = re.sub(r"\s+", " ", stripped).strip().lower()
    return stripped


def verify_quote_against_source(quote: str, source_text: str,
                                *, min_quote_len: int = 8) -> bool:
    """Return True if *quote* can be found in *source_text*.

    Whitespace/case-insensitive substring match. A near-miss tolerance handles
    minor punctuation/whitespace divergence: the normalized quote must be a
    contiguous substring of the normalized source. Quotes shorter than
    *min_quote_len* characters (after normalization) are treated as found —
    a tiny fragment is not a meaningful verbatim claim to falsify.
    """
    q = _normalize_for_quote_match(quote)
    s = _normalize_for_quote_match(source_text)
    if not q or not s:
        return False
    if len(q) < min_quote_len:
        return True
    return q in s


def apply_quote_verification(fact: Dict[str, Any], source_text: str) -> Dict[str, Any]:
    """Verify a fact's claimed verbatim quote against the source (#35).

    If the fact carries a ``verbatim_quote`` (LLM-claimed direct quote) and it
    cannot be found in *source_text*, the fact is downgraded: its grounding
    drops to ``inferred`` (the structural home from #40) and the miss is
    counted. Facts without a verbatim_quote are untouched. Mutates and returns
    *fact*; never raises.
    """
    global _QUOTE_MISSES
    try:
        quote = fact.get("verbatim_quote") if isinstance(fact, dict) else None
        if not quote:
            return fact
        payload = fact.setdefault("payload", {})
        if not isinstance(payload, dict):
            payload = {}
            fact["payload"] = payload
        if verify_quote_against_source(str(quote), source_text or ""):
            payload["quote_verified"] = True
        else:
            payload["quote_verified"] = False
            # Downgrade verbatim -> inferred (lands on #40's grounding field).
            payload["grounding"] = "inferred"
            _QUOTE_MISSES += 1
            logger.info(
                "Quote verification miss: claimed verbatim quote not found in "
                "source; downgraded to inferred. fact=%r",
                (fact.get("content") or "")[:80],
            )
    except Exception as exc:
        logger.debug("Quote verification failed (fail-soft): %s", exc)
    return fact


# ---------------------------------------------------------------------------
# Junk filter — prevents low-quality memories from being stored
# ---------------------------------------------------------------------------

# Phrases that indicate the extracted text is actually agent output, not a
# user fact.  These appear when the regex accidentally matches text the
# assistant said (e.g. "I'll search authoritative sources").
# Loaded from the pattern pack (#134); see _inline_pattern_defaults for the
# fallback definition.
_AGENT_SPEAK_PATTERNS = _PATTERNS["AGENT_SPEAK_PATTERNS"]

# Trivial facts that aren't worth storing — the agent can infer these from
# context or they add no value to future conversations.
_TRIVIAL_FACTS = _LEXICONS["TRIVIAL_FACTS"]

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


# ---------------------------------------------------------------------------
# Memory-control command gate (#99)
# ---------------------------------------------------------------------------
# Imperative memory-management commands ("bin it", "discard that proposal",
# "forget this") are actions against the memory system, not durable facts.
# The extractor must not mint them as preference candidates. This gate runs
# before the regex stage so a discard command never becomes "User wants to
# bin (discard) memory."
#
# The gate is deliberately conservative: it only matches short imperative
# phrases directed at the memory system, not legitimate assistant-side
# directives ("always explain in plain English") or durable preferences
# ("I prefer concise answers"). A command must be (a) short (under ~120
# chars), (b) start with a memory-control verb, and (c) reference a
# memory/proposal/candidate/it/that/this — so "I always forget my keys"
# (a durable habit) is NOT matched (it has "I" as the subject, not an
# imperative verb at the start).
_MEMORY_CONTROL_VERBS = re.compile(
    r"^\s*(?:bin|discard|forget|delete|remove|reject|skip|ignore|drop|purge"
    r"|scrap|trash|cancel|abort|undo|revoke)\b",
    re.IGNORECASE,
)
_MEMORY_CONTROL_TARGETS = re.compile(
    r"\b(?:it|that|this|the\s+(?:memory|proposal|candidate|fact|entry|record"
    r"|item|reminder|note|preference|directive)|them|those|these)\b",
    re.IGNORECASE,
)
# Phrases that are clearly memory-control commands even without a generic
# target word ("bin the proposal about X", "forget what I just said").
_MEMORY_CONTROL_PHRASES = re.compile(
    r"^\s*(?:bin|discard|forget|delete|remove|reject|skip|drop|purge"
    r"|scrap|trash|cancel)\s+(?:the\s+)?(?:proposal|candidate|memory|fact"
    r"|entry|record|item|reminder|note|preference|directive|what\s+I\s+just"
    r"|that\s+(?:proposal|candidate|memory|fact|entry|record|item))\b",
    re.IGNORECASE,
)
# Maximum content length for the memory-control gate — a long message is
# unlikely to be a pure memory-control command (it probably contains
# durable facts too). The gate only short-circuits short, imperative-only
# messages.
_MEMORY_CONTROL_MAX_CHARS = 120


def is_memory_control_command(user_content: str) -> bool:
    """Return True if *user_content* is a memory-control imperative (#99).

    These are one-off commands directed at the memory system ("bin it",
    "discard that proposal", "forget this") — they must not be extracted
    as durable facts. The check is conservative: only short, imperative-
    shaped messages that start with a control verb and reference a
    memory/proposal target are matched. Legitimate directives ("always
    explain in plain English") and durable habits ("I always forget my
    keys") are NOT matched.
    """
    if not user_content:
        return False
    text = user_content.strip()
    if len(text) > _MEMORY_CONTROL_MAX_CHARS:
        return False
    # Must start with a memory-control verb (imperative shape).
    if not _MEMORY_CONTROL_VERBS.match(text):
        return False
    # Must reference a memory/proposal target OR match a known phrase.
    if _MEMORY_CONTROL_TARGETS.search(text):
        return True
    if _MEMORY_CONTROL_PHRASES.match(text):
        return True
    return False


# Abbreviations that drive extraction-quality false-splits (#135). Trimmed
# to business-relevant suffixes + titles that the regex patterns commonly
# over-capture (e.g. _WORK_RE grabbing "Inc" as a workplace), NOT a general
# NLP-style suppress set. Stored lowercased without the trailing dot; the
# merge step strips the dot before comparing. Loaded from the pattern pack
# (#134).
_ABBREVIATIONS = _LEXICONS["ABBREVIATIONS"]


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences, preserving line boundaries (#133) and
    rejoining abbreviation false-splits (#135).

    Newlines are treated as hard sentence boundaries — a pasted list or
    note keeps its line structure instead of being collapsed into one
    run-on sentence. Within each line, spaces/tabs are normalized and the
    sentence is split on ``[.!?]`` followed by whitespace as before. A
    post-filter then rejoins fragments where the split point was an
    abbreviation (``Inc.``, ``Dr.``, ``Ltd.`` …) so ``_WORK_RE`` does not
    grab ``Inc`` as a workplace. Abbreviation merging never crosses a
    newline boundary.

    ReDoS guard (#89): each sentence is capped at ``_MAX_SENTENCE_CHARS`` so
    a long unpunctuated run (which the naive splitter treats as one
    "sentence") cannot feed pathological input to the regex patterns.
    """
    parts: List[str] = []
    for line in text.split('\n'):
        # Normalize spaces/tabs within the line, but keep newlines as
        # boundaries (do not collapse '\n' into a space). An empty line is
        # a hard boundary — it contributes no sentence and prevents the
        # surrounding lines from merging.
        line = re.sub(r'[ \t]+', ' ', line).strip()
        if not line:
            continue
        line_parts = re.split(r'(?<=[.!?])\s+', line)
        # Merge fragments where the split point was an abbreviation (#135).
        # Done per-line so an abbreviation never merges across a newline.
        merged: List[str] = []
        for part in line_parts:
            if merged:
                prev = merged[-1].rstrip()
                # Last token of the previous fragment, trailing dot stripped.
                last_word = prev.rsplit(None, 1)[-1].rstrip('.') if prev else ""
                if last_word.lower() in _ABBREVIATIONS:
                    merged[-1] = prev + " " + part
                    continue
            merged.append(part)
        parts.extend(merged)
    return [p.strip()[:_MAX_SENTENCE_CHARS] for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Stage 1: Generic syntactic patterns (topic-agnostic)
# ---------------------------------------------------------------------------
#
# These patterns match the *structural shape* of durable statements, not
# specific topics.  "I use React" and "I take FocusTool" both match the same
# "I use/take X" pattern.  This makes the extractor general-purpose.

# "I am/is a <something>" — identity, role, profession.
_IDENTITY_RE = _PATTERNS["IDENTITY_RE"]

# "I have/own/use/take <something>" — possession, usage, tools.
_HAVE_USE_RE = _PATTERNS["HAVE_USE_RE"]

# "I work at/for <something>" / "I'm a <job>" — work context.
_WORK_RE = _PATTERNS["WORK_RE"]

# "I prefer/like/love/hate/enjoy <something>" — preferences.
# The simple verb alternation is built dynamically from the base lexicon +
# any extra verbs added via set_preference_verbs() (#136), so a domain pack
# can add "adore"/"detest" without code changes. The multi-word phrases
# ("can't stand", "a fan of", "keen on" …) are structural and stay fixed.
# ReDoS (#89/#136): every verb is re.escape()'d before joining; the
# bounded (.+?) tail and per-sentence cap are preserved.
_BASE_PREFERENCE_VERBS = _LEXICONS["PREFERENCE_VERBS"]
_extra_preference_verbs: set[str] = set()

# Fixed multi-word preference phrases (not simple verbs — structural).
_PREFERENCE_PHRASES = r"can\'t\s+stand|don\'t\s+like|don\'t\s+enjoy"


def _build_preference_re() -> "re.Pattern":
    verbs = _BASE_PREFERENCE_VERBS | _extra_preference_verbs
    verb_alt = "|".join(re.escape(v) for v in sorted(verbs))
    return re.compile(
        rf'\b(?:i\s+(?:{verb_alt}|{_PREFERENCE_PHRASES})'
        r"|i\'d\s+(?:rather|prefer)"
        r"|i\'m\s+(?:a\s+fan\s+of|not\s+a\s+fan\s+of|into|keen\s+on))\s+(.+?)(?:\.|$)",
        re.IGNORECASE,
    )


def set_preference_verbs(words: set[str] | frozenset[str] | None) -> None:
    """Extend the preference-verb set at runtime (#136).

    A domain pack can add verbs like "adore", "detest", "fancy" without
    editing extractor.py. The regex is rebuilt from the combined
    (base + extra) set with re.escape on every member.
    """
    global _extra_preference_verbs, _PREFERENCE_RE
    _extra_preference_verbs = {w.lower().strip() for w in (words or set()) if w}
    _PREFERENCE_RE = _build_preference_re()


_PREFERENCE_RE = _build_preference_re()

# Assistant-side style directives: "always give me the short version",
# "never use code comments", "don't call it that", "call me Mike",
# "stop using jargon" — rules the user sets for how the assistant behaves.
# ReDoS guard (#89): the [^.!?]* tails are bounded to {0,200} so a long
# unpunctuated run cannot cause catastrophic backtracking. Combined with the
# per-sentence length cap in _extract_facts_regex, this caps the worst-case
# regex work on any input.
_ASSISTANT_DIRECTIVE_RE = _PATTERNS["ASSISTANT_DIRECTIVE_RE"]

# "I always/never/usually <something>" — habits and patterns.
_HABIT_RE = _PATTERNS["HABIT_RE"]

# "I'm working on/trying to/going to/need to/want to <something>" — goals.
_GOAL_RE = _PATTERNS["GOAL_RE"]

# "I switched from X to Y" / "I moved from X to Y" — transitions.
_SWITCH_RE = _PATTERNS["SWITCH_RE"]

# "I started/began/stopped/quit/resumed <something>" — events.
# The verb alternation is built dynamically from the base lexicon + any
# extra verbs added via set_event_verbs() (#136), so a domain pack can add
# "deployed"/"shipped"/"approved" without code changes.
# ReDoS (#89/#136): every verb is re.escape()'d before joining; the bounded
# (.+?) tail and per-sentence cap are preserved.
_BASE_EVENT_VERBS = _LEXICONS["EVENT_VERBS"]
_extra_event_verbs: set[str] = set()


def _build_event_re() -> "re.Pattern":
    verbs = _BASE_EVENT_VERBS | _extra_event_verbs
    verb_alt = "|".join(re.escape(v) for v in sorted(verbs))
    return re.compile(
        rf"\bi\s+({verb_alt})\s+(.+?)(?:\.|$)",
        re.IGNORECASE,
    )


def set_event_verbs(words: set[str] | frozenset[str] | None) -> None:
    """Extend the event-verb set at runtime (#136).

    A domain pack can add verbs like "deployed", "shipped", "approved",
    "signed", "filed" without editing extractor.py. The regex is rebuilt
    from the combined (base + extra) set with re.escape on every member.
    """
    global _extra_event_verbs, _EVENT_RE
    _extra_event_verbs = {w.lower().strip() for w in (words or set()) if w}
    _EVENT_RE = _build_event_re()


_EVENT_RE = _build_event_re()

# "I tend to/struggle with/realized/noticed" — insights, self-observations.
# "i've been noticing", "i think i", "i feel like i" are top-level alternatives.
_INSIGHT_RE = _PATTERNS["INSIGHT_RE"]

# "X is my <relation>" — relationship introduction (any relationship).
_RELATIONSHIP_RE = _PATTERNS["RELATIONSHIP_RE"]

# "My <thing> is <value>" — attributes, config, ownership.
_MY_X_IS_RE = _PATTERNS["MY_X_IS_RE"]

# "I live in/am based in/am from <place>" — location.
_LOCATION_RE = _PATTERNS["LOCATION_RE"]

# "I've been doing X for Y" / "I've been on X for Y" — ongoing states.
_ONGOING_RE = _PATTERNS["ONGOING_RE"]

# "I have/own X" — possession attribute (scoped to have/own only so it
# doesn't shadow _HAVE_USE_RE for "I use/take X" — issue #32).
_ATTRIBUTE_RE = _PATTERNS["ATTRIBUTE_RE"]


# Words that are not valid names for relationship extraction.
_NOT_A_NAME = _LEXICONS["NOT_A_NAME"]

# Transient states for the identity gate ("I am tired", "I am ready"). A
# detail that is or starts with one of these is not a durable identity fact.
# The base set is loaded from the pattern pack (#134); extra words can be
# added at runtime via set_transient_words() (#136).
_BASE_TRANSIENT_WORDS = _LEXICONS["TRANSIENT_WORDS"]
_extra_transient_words: set[str] = set()


def set_transient_words(words: set[str] | frozenset[str] | None) -> None:
    """Extend the transient-word set at runtime (#136).

    A domain pack can add transient states specific to its context (e.g.
    "on leave", "in transit") without editing extractor.py.
    """
    global _extra_transient_words
    _extra_transient_words = {w.lower().strip() for w in (words or set()) if w}


def _all_transient_words() -> frozenset[str]:
    """Return the complete transient-word set (base + extra)."""
    return _BASE_TRANSIENT_WORDS | _extra_transient_words

# Base relationship role words for the "My X is Y" gate. Extended at runtime
# by set_role_words() with the graph's learned/configured role words so the
# extractor and graph converge on one lexicon (issue #14).
_BASE_ROLE_WORDS = _LEXICONS["BASE_ROLE_WORDS"]
_extra_role_words: set[str] = set()


def set_role_words(words: set[str] | frozenset[str] | None) -> None:
    """Update the extractor's role-word set from the graph's learned words.

    Called by the provider during initialize() so the extractor and graph
    converge on one lexicon (issue #14: previously the extractor had a
    private 14-word list that learning never updated).
    """
    global _extra_role_words
    _extra_role_words = {w.lower().strip() for w in (words or set()) if w}


def _all_role_words() -> frozenset[str]:
    """Return the complete role-word set (base + learned)."""
    return _BASE_ROLE_WORDS | _extra_role_words


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
            # Abbreviation guard (#135): a bare single-token corporate suffix
            # ("Inc", "Pty", "Ltd", "Corp" …) captured because the regex
            # stopped at the suffix's period is not a real workplace name.
            # A multi-token capture like "Acme Inc" is a real name + suffix
            # and is kept.
            if " " not in workplace and workplace.lower() in _ABBREVIATIONS:
                pass  # fall through to other patterns / no fact
            else:
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
            # Uses the dynamic role-word set (base + graph-learned words) so
            # a learned word like "doula" or "housemate" correctly categorizes
            # as a relationship (issue #14).
            if attr in _all_role_words():
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
            matched = m.group(0).lower()
            negated = any(k in matched for k in (" not ", "n't", "dislike", "hate", "can't stand"))
            return {
                "category": "preference",
                "content": f"User {'dislikes' if negated else 'prefers'}: {pref}",
                "tags": ["preference"],
                "payload": {"preference": pref},
            }

    # Assistant-side directive: "always give me the short version" /
    # "never use jargon" / "call me Mike" — stored as preference with
    # the assistant_side tag so preference-shaped queries surface it.
    m = _ASSISTANT_DIRECTIVE_RE.search(sentence)
    if m:
        directive = m.group(0).strip().rstrip('.!?')
        if len(directive) > 8:
            return {
                "category": "preference",
                "content": f"User directive: {directive}",
                "tags": ["preference", "assistant_side"],
                "payload": {"preference": directive, "assistant_side": True},
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
        detail_lower = detail.lower()
        first_word = detail_lower.split()[0] if detail_lower else ""
        is_transient = (
            detail_lower in _all_transient_words()
            or first_word in _all_transient_words()
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
    """Stage 1: Extract candidate facts using generic syntactic patterns.

    ReDoS guard (#89): the input is capped at ``_MAX_REGEX_CONTENT_CHARS``
    before sentence splitting, and the regex stage is wrapped in a thread
    watchdog (``_REGEX_STAGE_TIMEOUT_S``) so a hang in the regex engine
    never stalls the turn. On timeout or any exception, returns the facts
    found so far (fail-soft).
    """
    global _EXTRACTION_FAILURES
    facts: List[Dict[str, Any]] = []
    if not user_content or len(user_content.strip()) < _MIN_LENGTH:
        return facts

    # Total content cap: truncate before splitting so a huge paste cannot
    # feed unbounded input to the patterns.
    capped = user_content[:_MAX_REGEX_CONTENT_CHARS]
    sentences = _split_sentences(capped)[:_MAX_SENTENCES]

    def _classify_all() -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
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
                out.append(fact)
        return out

    # Watchdog: run the regex classification in a worker thread with a
    # timeout. The bounded quantifiers + length caps make catastrophic
    # backtracking unlikely, but this is defense-in-depth (#89).
    result_holder: List[Any] = []
    exc_holder: List[BaseException] = []

    def _worker() -> None:
        try:
            result_holder.extend(_classify_all())
        except BaseException as exc:  # noqa: BLE001 — watchdog must capture all
            exc_holder.append(exc)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join(timeout=_REGEX_STAGE_TIMEOUT_S)
    if thread.is_alive():
        # Timeout — the worker is still running. A daemon thread will be
        # reaped at process exit; we return what we have (possibly empty)
        # and count the failure for review.
        _EXTRACTION_FAILURES += 1
        logger.warning(
            "Regex extraction stage timed out after %.1fs (input %d chars); "
            "returning partial results",
            _REGEX_STAGE_TIMEOUT_S, len(user_content),
        )
        return facts
    if exc_holder:
        _EXTRACTION_FAILURES += 1
        logger.debug("Regex extraction stage raised: %s", exc_holder[0])
        return facts
    facts = result_holder
    return facts


# ---------------------------------------------------------------------------
# Stage 2: LLM fallback extraction
# ---------------------------------------------------------------------------

_LLM_SYSTEM_PROMPT = """You are a memory extraction assistant. Your job is to extract durable facts from the user's message that would be worth remembering for future conversations.

Extract facts that are:
- Durable (will still be true tomorrow, not transient)
- Self-contained (make sense without the original conversation)
- About the user or things in the user's life

DO extract preferences and style directives — these are high value:
- First-person likes/dislikes/wants ("I prefer window seats", "I hate meetings before 10")
- Style/workflow directives aimed at the assistant ("always give me the short
  version", "never use code comments", "call me Mike", "explain in plain English first")
- Endorsements of assistant behavior ("I like it when you push back",
  "don't be so formal with me")

Do NOT extract:
- Transient states ("I'm tired", "I'm busy right now")
- Questions or requests
- Greetings or chit-chat
- First-person statements inside quoted or pasted third-party content —
  only the user's own voice counts as the user's view

Return a JSON array of objects with these keys:
- "category": one of "personal_fact", "preference", "insight", "event", "relationship", "goal", "context_note"
- "content": a clear, self-contained statement of the fact
- "tags": array of 1-3 short lowercase tags
- "confidence": number from 0 to 1
- "durability": "permanent", "durable", or "temporary"
- "scope": "profile", "project", or "session"
- "verbatim_quote": OPTIONAL — the exact substring of the user's message that
  this fact is a direct quote of. Include this ONLY when the content is a
  word-for-word quote of the user; omit it for paraphrased or inferred facts.
  The quote MUST appear verbatim in the user's message or it will be rejected.

If there are no durable facts, return an empty array: []

Examples:
User: "I just got a new job at Stripe, I'll be starting next Monday as a backend engineer"
Output: [{"category": "event", "content": "User got a new job at Stripe as a backend engineer, starting next Monday", "tags": ["work", "job", "stripe"]}]

User: "My contact suggested I try journaling every morning"
Output: [{"category": "context_note", "content": "User's contact suggested morning journaling", "tags": ["journaling"]}]

User: "I've been deploying with Kubernetes lately, it's way better than Docker Swarm for our scale"
Output: [{"category": "preference", "content": "User prefers Kubernetes over Docker Swarm for deployment at scale", "tags": ["devops", "kubernetes"]}, {"category": "personal_fact", "content": "User deploys with Kubernetes", "tags": ["devops", "kubernetes"]}]

User: "Honestly I prefer short answers, and don't ever give me code without a plain-English explanation first"
Output: [{"category": "preference", "content": "User prefers short answers", "tags": ["preference", "communication"]}, {"category": "preference", "content": "User directive: don't ever give code without a plain-English explanation first", "tags": ["preference", "assistant_side"]}]

User: "I like it when you challenge my assumptions instead of just agreeing"
Output: [{"category": "preference", "content": "User likes the assistant to challenge their assumptions rather than just agree", "tags": ["preference", "assistant_side"]}]

User: "hey how are you"
Output: []
"""


def _extract_facts_llm(user_content: str, *, model: str = "", provider: str = "") -> List[Dict[str, Any]]:
    """Stage 2: Extract facts using the host's LLM via call_llm.

    Returns a list of fact dicts, or empty list on any failure.
    Never raises — all errors are caught and logged.
    """
    global _EXTRACTION_FAILURES
    if not user_content or len(user_content.strip()) < _LLM_MIN_CONTENT_LENGTH:
        return []
    # Egress gate (#85): the import and gate call must sit inside the try so
    # a malformed config / import failure fails soft (returns []) instead of
    # propagating through extract_from_turn() and crashing the turn. The
    # function's own contract is "Never raises".
    try:
        from egress import gate as _egress_gate
        if not _egress_gate("extractor", user_content):
            return []
    except Exception as e:
        _EXTRACTION_FAILURES += 1
        logger.debug("Egress gate unavailable for LLM extraction (fail-soft): %s", e)
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
        if category == "preference" and "preference" not in tags and len(tags) < 5:
            tags.append("preference")
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
        payload: Dict[str, Any] = {"source": "llm_extraction"}
        out_fact = {
            "category": category,
            "content": content,
            "tags": tags,
            "source": "llm_extraction",
            "confidence": confidence,
            "durability": durability,
            "scope": scope,
            "payload": payload,
        }
        # Carry an LLM-claimed verbatim quote through for verification (#35).
        vq = fact.get("verbatim_quote")
        if isinstance(vq, str) and vq.strip():
            out_fact["verbatim_quote"] = vq.strip()
        # Verify the claimed quote against the source transcript before
        # labelling the memory verbatim. On miss, downgrade to inferred (#40
        # grounding field) and count the failure for review (#35).
        apply_quote_verification(out_fact, user_content)
        result.append(out_fact)

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

    Never raises (#85/#89): a top-level guard catches any exception from
    either stage and returns the facts found so far (or [] if none),
    counting the failure for review. Extraction must never crash the turn.

    Memory-control commands (#99): a short imperative like "bin it" or
    "discard that proposal" is an action against the memory system, not a
    durable fact. The action/intent gate short-circuits before either
    extraction stage so a discard command never becomes a preference
    candidate.
    """
    global _EXTRACTION_FAILURES
    # Action/intent gate (#99): memory-control imperatives are not facts.
    if is_memory_control_command(user_content):
        logger.debug("Memory-control command detected, skipping extraction: %.80s", user_content)
        return []
    # Top-level guard (#85/#89): any uncaught exception in either stage
    # fails soft — return what we have (possibly empty) and count it.
    try:
        return _extract_from_turn_impl(
            user_content, assistant_content,
            use_llm_fallback=use_llm_fallback,
            llm_model=llm_model,
            llm_provider=llm_provider,
            shadow_diff=shadow_diff,
        )
    except BaseException as exc:
        _EXTRACTION_FAILURES += 1
        logger.debug("extract_from_turn top-level guard caught: %s", exc)
        return []


def _extract_from_turn_impl(
    user_content: str,
    assistant_content: str,
    *,
    use_llm_fallback: bool = True,
    llm_model: str = "",
    llm_provider: str = "",
    shadow_diff: bool = False,
) -> List[Dict[str, Any]]:
    """Implementation body of extract_from_turn (guarded by the caller)."""
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
            # E2: update existing_contents inside the loop so LLM facts
            # are deduped against other LLM facts, not just regex facts.
            existing_contents = [f["content"].lower() for f in facts]
            for lf in llm_facts:
                lf_lower = lf["content"].lower()
                # Skip if an existing fact (regex or LLM) is a near-duplicate.
                if not any(_text_overlap(lf_lower, ec) for ec in existing_contents):
                    facts.append(lf)
                    existing_contents.append(lf_lower)  # E2: update dedup set

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
