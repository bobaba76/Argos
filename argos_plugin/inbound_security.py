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

import re
from dataclasses import dataclass, field
from typing import List, Set

# (category, pattern_name, regex) — all matched case-insensitively on the
# lowercased input. Patterns are written to be specific rather than broad:
# false positives here are cheap (→ human review), false negatives are not.
_NEG = r"(?:do\s+not|don'?t|dont|do\s+n['\u2019]?t)"  # "do not" / "don't"

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
     r"from\s+now\s+on"),
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
     r"select\s+\*?\s*from"),
    ("sql_code", "sql_drop",
     r"drop\s+table"),
    ("sql_code", "sql_delete",
     r"delete\s+from"),
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


def scan_inbound_text(text: str) -> ScanResult:
    """Scan one inbound text. BLOCKED = route to a human; never act on it."""
    result = ScanResult()
    if not text or not text.strip():
        return result
    lower = text.lower()
    for category, name, regex in _PATTERNS:
        m = re.search(regex, lower)
        if m:
            start = max(0, m.start() - 30)
            snippet = text[start:m.end() + 30].replace("\n", " ")
            result.matches.append(
                ScanMatch(category=category, pattern=name, snippet=snippet)
            )
    result.blocked = bool(result.matches)
    return result