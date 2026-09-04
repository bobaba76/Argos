"""Inbound content security scanner (memory-poisoning defense).

Deterministic, zero-LLM screening for text arriving from EXTERNAL/untrusted
channels (email, web imports, tickets, feeds) before it can influence
memory. Domain-neutral port of the attack classes measured by
eval/probes/probe_poisoning.py:

  - injection_override  : "ignore previous instructions", "you are now",
                          prompt-reveal, jailbreak-style directives
  - stealth_suppression : "don't mention this email", "keep secret from",
                          skip-approval, hide-from-admin
  - memory_mutation     : directives that mutate durable state from external
                          content ("decision recorded", "effective
                          immediately", "update your knowledge base",
                          suppression overrides, price-list changes)
  - sql_code / web_code : classic SQL / script / exfil patterns

The scanner is a gate, not a judge: a BLOCKED verdict means "route to a
human", never "act on this". It is cheap (regex-only) and safe to run on
every inbound message.

Integrations (importers, feed handlers) call scan_inbound_text() on raw
inbound content, and may tag candidates with external=True so the review
gate re-scans the evidence and enforces the confirmation policy.
"""
from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Set

# (category, pattern_name, regex) — all matched case-insensitively on the
# lowercased input. Patterns are written to be specific rather than broad:
# false positives here are cheap (→ human review), false negatives are not.
_NEG = r"(?:do\s+not|don'?t|dont|do\s+n['\u2019]?t)"  # "do not" / "don't"

# ---------------------------------------------------------------------------
# Confusable homoglyph mapping (IS2)
# ---------------------------------------------------------------------------
# NFKD decomposition only handles compatibility characters (fullwidth,
# superscript), NOT cross-script confusables (Cyrillic 'а' → Latin 'a').
# This table covers the most common Cyrillic and Greek confusables that
# look like ASCII letters. Source: Unicode confusables.txt (curated subset
# for the security scanner's ASCII keyword patterns).
_CONFUSABLES: dict[str, str] = {}
for _src, _dst in [
    # Cyrillic → Latin
    ("а", "a"), ("е", "e"), ("о", "o"), ("р", "p"), ("с", "c"),
    ("у", "y"), ("х", "x"), ("А", "A"), ("В", "B"), ("Е", "E"),
    ("К", "K"), ("М", "M"), ("Н", "H"), ("О", "O"), ("Р", "P"),
    ("С", "C"), ("Т", "T"), ("Х", "X"), ("і", "i"), ("І", "I"),
    ("ј", "j"), ("Ј", "J"), ("ѕ", "s"), ("Ѕ", "S"),
    # Greek → Latin
    ("α", "a"), ("β", "b"), ("ε", "e"), ("ζ", "z"), ("η", "h"),
    ("θ", "0"), ("ι", "i"), ("κ", "k"), ("ν", "v"), ("ο", "o"),
    ("ρ", "p"), ("τ", "t"), ("υ", "u"), ("χ", "x"), ("ω", "w"),
    ("Α", "A"), ("Β", "B"), ("Ε", "E"), ("Ζ", "Z"), ("Η", "H"),
    ("Ι", "I"), ("Κ", "K"), ("Μ", "M"), ("Ν", "N"), ("Ο", "O"),
    ("Ρ", "P"), ("Τ", "T"), ("Υ", "Y"), ("Χ", "X"),
]:
    _CONFUSABLES[_src] = _dst
_CONFUSABLE_TABLE = str.maketrans(_CONFUSABLES)

_PATTERNS: List[tuple] = [
    # ---- instruction-override / roleplay injections ----------------------
    ("injection_override", "ignore_previous_instructions",
     r"ignore\s+(?:all\s+)?(?:previous|prior)\s+(?:\w+\s+)*(?:instructions?|guidelines?|rules?|directives?|content)"),
    ("injection_override", "disregard_previous",
     r"disregard\s+(?:all\s+)?(?:previous|prior|the\s+above)"),
    ("injection_override", "forget_everything",
     r"forget\s+(?:everything|all\s+(?:prior|previous))"),
    ("injection_override", "you_are_now",
     r"you\s+are\s+now\s+(?:an?|the)\s+"),
    ("injection_override", "act_as",
     r"act\s+as\s+(?:if\s+you|an?|a)\s+"),
    ("injection_override", "reveal_prompt",
     r"(?:reveal|print|show|output|repeat)\s+(?:your|the|system)\s+(?:system\s+)?prompt"),
    ("injection_override", "jailbreak_mode",
     r"jailbreak|developer\s+mode|dan\s+mode|do\s+anything\s+now"),
    # ---- stealth / suppression -------------------------------------------
    ("stealth_suppression", "do_not_mention",
     rf"{_NEG}\s+mention\s+this"),
    ("stealth_suppression", "keep_secret",
     r"keep\s+(?:this\s+)?secret\s+from"),
    ("stealth_suppression", "do_not_notify",
     rf"{_NEG}\s+notify"),
    ("stealth_suppression", "do_not_tell",
     rf"{_NEG}\s+(?:tell|alert)\s+(?:the\s+)?\w+"),
    ("stealth_suppression", "do_not_log",
     rf"{_NEG}\s+(?:log|record|store)\s+this"),
    ("stealth_suppression", "no_human_approval",
     r"(?:skip|bypass|without|no\s+need\s+for)\s+(?:human\s+)?(?:approval|review|confirmation|checks?)"),
    ("stealth_suppression", "between_us",
     r"keep\s+this\s+between\s+(?:us|you\s+and\s+me|the\s+two\s+of\s+us)"),
    ("stealth_suppression", "hide_from",
     r"hide\s+(?:this\s+)?from\s+(?:admin|management|your\s+\w+)"),
    # ---- durable-state mutation directives --------------------------------
    ("memory_mutation", "update_knowledge",
     r"update\s+(?:your|the)\s+(?:knowledge\s+)?(?:base|knowledge|memory|records)"),
    ("memory_mutation", "record_decision",
     r"(?:record(?:ed|s)?|note(?:d)?|log(?:ged)?|store(?:d)?)\s+(?:this\s+)?(?:decision|fact|policy|change|instruction)"),
    ("memory_mutation", "decision_recorded",
     r"decision\s+recorded"),
    ("memory_mutation", "effective_now",
     r"effective\s+(?:now|immediately)"),
    ("memory_mutation", "final_approved",
     r"final\s+approved"),
    ("memory_mutation", "from_now_on",
     r"from\s+now\s+on[,\s]+(?:this|the\s+above|all|every)\s+(?:is|are|applies?|rules?|polic(?:y|ies)|guidelines?|instructions?|directives?)"),
    ("memory_mutation", "overrides_previous",
     r"(?:this|that|the\s+above)\s+overrides?|overrides?\s+(?:all|any|previous)"),
    ("memory_mutation", "remove_suppression",
     r"remove\s+the\s+(?:suppression|marketing\s+suppression|block|flag)"),
    ("memory_mutation", "resume_targeting",
     r"resume\s+(?:all\s+)?(?:marketing|targeting|campaigns?|communication)"),
    ("memory_mutation", "price_change_directive",
     r"(?:new\s+)?(?:price|pricing|rates?)\s+(?:list|change|update|effective)"),
    # ---- SQL / web injection ----------------------------------------------
    ("sql_code", "sql_select",
     r"select\s+\*?\s*from\s+(?!the\b|a\b|an\b|my\b|your\b|his\b|her\b|our\b|their\b|this\b|that\b|these\b|those\b|all\b|some\b|any\b|each\b|every\b)\w+"),
    ("sql_code", "sql_drop",
     r"drop\s+table"),
    ("sql_code", "sql_delete",
     r"delete\s+from\s+(?!the\b|a\b|an\b|my\b|your\b|his\b|her\b|our\b|their\b|this\b|that\b|these\b|those\b|all\b|some\b|any\b|each\b|every\b)\w+"),
    ("sql_code", "sql_union",
     r"union\s+select"),
    ("web_code", "script_tag",
     r"<script|onerror\s*=|javascript:"),
    ("web_code", "exfil_pattern",
     r"curl\s+-|wget\s+|eval\s*\(|exec\s*\(|base64\s+--decode|data:text/html"),
]


@dataclass
class ScanMatch:
    category: str
    pattern: str
    snippet: str


@dataclass
class ScanResult:
    blocked: bool = False
    matches: List[ScanMatch] = field(default_factory=list)

    def categories(self) -> Set[str]:
        return {m.category for m in self.matches}

    def summary(self) -> str:
        if not self.matches:
            return "clean"
        return "; ".join(
            f"{m.category}:{m.pattern}"
            for m in self.matches[:8]
        ) + (f" (+{len(self.matches) - 8} more)" if len(self.matches) > 8 else "")


def _normalize_for_scan(text: str) -> str:
    """Normalize text before regex scanning to catch evasion (#19).

    The scanner is regex-only, which has a known ceiling: paraphrasing,
    non-English text, and character-level evasion bypass it. This
    normalization layer addresses the character-level evasion class
    without adding LLM-based detection:

    1. Decode HTML entities (&#x69; = 'i', &lt; = '<') — a common
       obfuscation technique that hides patterns from regex. This runs
       FIRST so that entity-encoded zero-width characters (e.g.
       ``&#8203;``, ``&#x200b;``, ``&ZeroWidthSpace;``) are decoded
       before the zero-width replacement in step 2 (#75).
    2. Replace zero-width characters (U+200B, U+200C, U+200D, U+FEFF)
       with a SPACE (IS1) -- attackers insert these between keywords to
       break regex matches while keeping the text visually identical.
       Replacing with space (not stripping to nothing) ensures that
       ``ignore\\u200bprevious`` -> ``ignore previous`` -> matches.
       Running AFTER html.unescape catches entity-encoded zero-width
       chars (#75). The whitespace collapse in step 5 cleans up any
       double spaces this may introduce.
    3. Map confusable homoglyphs (IS2) -- NFKD alone does NOT decompose
       cross-script confusables (Cyrillic 'a' stays U+0430, not Latin
       'a' U+0061). A curated mapping table translates the most common
       Cyrillic/Greek lookalike characters to their ASCII equivalents
       BEFORE NFKD runs, so patterns match regardless of the input
       script.
    4. NFKD normalization -- decomposes compatibility characters
       (fullwidth, superscript) that the confusable table doesn't cover.
    5. Collapse whitespace -- multiple spaces/tabs between words can
       break patterns that expect single spaces. Also cleans up the
       double spaces introduced by zero-width -> space replacement.

    This keeps the zero-LLM property: the normalization is deterministic
    and cheap. The original text is preserved for the snippet; only the
    scanned copy is normalized.
    """
    if not text:
        return ""
    # 1. Decode HTML entities (handles named, decimal, and hex).
    # Must run before zero-width strip so entity-encoded zero-width
    # chars (&#8203; &#x200b; &ZeroWidthSpace;) are caught (#75).
    result = html.unescape(text)
    # 2. Replace zero-width characters with a space (IS1, #75).
    # Replacing with space (not stripping) ensures "ignore\u200bprevious"
    # becomes "ignore previous" (matches) instead of "ignoreprevious" (no match).
    result = re.sub(r"[\u200B\u200C\u200D\uFEFF]", " ", result)
    # 3. Map confusable homoglyphs to ASCII (IS2).
    # NFKD alone does NOT decompose cross-script confusables (Cyrillic 'a'
    # stays U+0430). This table handles the common Cyrillic/Greek lookalikes.
    result = result.translate(_CONFUSABLE_TABLE)
    # 4. NFKD normalization: decompose compatibility characters (fullwidth, etc.).
    result = unicodedata.normalize("NFKD", result)
    # 5. Collapse whitespace (cleans up double spaces from zero-width replacement).
    result = re.sub(r"[ \t]+", " ", result)
    return result


def scan_inbound_text(text: str) -> ScanResult:
    """Scan one inbound text. BLOCKED = route to a human; never act on it.

    The text is normalized before scanning (#19) to catch character-level
    evasion (zero-width chars, HTML entities, homoglyphs). The original
    text is used for snippet extraction so the evidence is readable.
    """
    result = ScanResult()
    if not text or not text.strip():
        return result
    # Normalize for scanning (catches evasion), but keep original for snippets.
    normalized = _normalize_for_scan(text)
    lower = normalized.lower()
    for category, name, regex in _PATTERNS:
        m = re.search(regex, lower)
        if m:
            start = max(0, m.start() - 30)
            snippet = normalized[start:m.end() + 30].replace("\n", " ")
            result.matches.append(
                ScanMatch(category=category, pattern=name, snippet=snippet)
            )
    result.blocked = bool(result.matches)
    return result


def scan_inbound_or_raise(text: str, *, content_label: str = "content") -> ScanResult:
    """Ingestion-time enforcement helper (#19): scan external content at the boundary.

    Unlike ``scan_inbound_text`` (which just returns a result), this function
    raises ``InboundSecurityError`` if the scan blocks, so the caller can
    refuse the write — the content never enters the store.

    **Usage note (IS5):** The production store paths (``remember()``,
    ``save_candidate()``) call ``scan_inbound_text`` directly and handle
    the blocked result themselves (refuse vs. quarantine). This helper is
    available for importers and feed handlers that prefer the raise-on-
    block pattern. It is not dead code — it is an alternative API for
    callers that want exception-based enforcement rather than result-based.

    Args:
        text: The inbound content to scan.
        content_label: A short label for error messages (e.g. "candidate",
                       "memory").

    Returns:
        ScanResult with blocked=False if the content is clean.

    Raises:
        InboundSecurityError: If the scan blocks (content matches a
                              poisoning/injection pattern).
    """
    result = scan_inbound_text(text)
    if result.blocked:
        raise InboundSecurityError(
            f"Inbound security scan blocked {content_label}: {result.summary()}"
        )
    return result


class InboundSecurityError(Exception):
    """Raised when inbound content is blocked by the security scanner.

    The content matched a poisoning/injection pattern and must not enter
    the store. Route to a human for review.
    """