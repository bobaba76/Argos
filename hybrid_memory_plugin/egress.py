"""Egress inventory and gate for LLM-bound auxiliary calls.

The hybrid-memory plugin makes its own LLM calls beyond the host's normal
answering loop (extraction fallback, review, graph typing, query
expansion, the temporal sub-call, distillation). This module is the
single place that (a) inventories every such site — what it sends, when,
and which config flag gates it — and (b) enforces the egress gate before
any call.

Gate semantics
--------------
* ``local_only=true`` in the plugin config: every LLM egress site is
  refused and falls back to its built-in fail-soft behavior. No stored or
  conversational memory data leaves the machine (embeddings are already
  computed by a local model; the only remaining network traffic would be
  the host's own answering loop and the weather provider's city-name
  lookup, which carries no memory data).
* Sensitive-identifier payloads: conversation-derived payloads
  (extractor, reviewer, query expansion, role-word, temporal question)
  are refused when they contain PII identifiers (emails, phone numbers,
  ID/card digit runs) — the call site then fails soft (no proposal, no
  review, no expansion, original results, empty hint).
* Store-derived payloads (graph typing, distillation) are governed by
  their own config flags plus ``local_only``; the sensitive-identifier
  gate does not apply to them, because stored memories legitimately
  contain identifiers the user chose to keep, and these features are
  explicitly configured on/off.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Static inventory: every plugin-owned LLM egress site.
# ---------------------------------------------------------------------------

SITES = [
    {
        "kind": "extractor",
        "file": "extractor.py",
        "trigger": "auto-extraction on a turn where regex patterns missed",
        "payload": "recent turn text (potentially personal)",
        "gate": "llm_fallback",
        "default": True,
    },
    {
        "kind": "reviewer",
        "file": "reviewer.py",
        "trigger": "new pending proposal with auto_review enabled",
        "payload": "proposal text + evidence quote",
        "gate": "auto_review",
        "default": True,
    },
    {
        "kind": "graph_typing",
        "file": "graph.py",
        "trigger": "memory save with <3 typed regex relations, >=60 chars",
        "payload": "stored memory content (category + text)",
        "gate": "llm_fallback",
        "default": True,
    },
    {
        "kind": "query_expansion",
        "file": "query_expander.py",
        "trigger": "weak top search hit (below similarity floor)",
        "payload": "the search query",
        "gate": "query_expansion_enabled",
        "default": True,
    },
    {
        "kind": "temporal_subcall",
        "file": "temporal_subcall.py",
        "trigger": "temporal/multi-hop intent route",
        "payload": "question + up to 8 dated memory snippets",
        "gate": "router_subcall_enabled",
        "default": True,
    },
    {
        "kind": "role_word",
        "file": "__init__.py",
        "trigger": "'my X is Name' ambiguity (rare, single word)",
        "payload": "one candidate role word",
        "gate": "none (always-on, rare)",  # n/a
        "default": True,
    },
    {
        "kind": "distillation",
        "file": "distillation.py",
        "trigger": "session-end pass (novelty gate + cooldown + call budget)",
        "payload": "sampled memory packs (stored memories)",
        "gate": "distillation_enabled",
        "default": False,
    },
]

# Non-memory network callers (context providers) — listed for completeness;
# they never carry stored or conversational memory data.
CONTEXT_NETWORK = [
    {
        "kind": "weather",
        "file": "hermes_weather.py",
        "trigger": "ambient context injection",
        "payload": "configured city name (Open-Meteo geocoding + forecast)",
        "note": "no API key; no memory data",
    },
    {
        "kind": "location",
        "file": "hermes_location.py",
        "trigger": "ambient context injection",
        "payload": "none (local resolution)",
        "note": "no network",
    },
]

# ---------------------------------------------------------------------------
# Sensitive-identifier gate (applies to conversation-derived payloads).
# ---------------------------------------------------------------------------

SENSITIVE_PATTERNS = [
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "email address"),
    (
        re.compile(r"(?<!\d)(?:\+?27|0)\d{9}(?!\d)"),
        "South African phone number",
    ),
    (re.compile(r"\b\d{13}\b"), "13-digit ID number"),
    (
        re.compile(r"\b(?:\d[ -]?){15,16}\b"),
        "16-digit card-number run",
    ),
]

# Kinds whose conversation-derived payload gets the sensitive-identifier gate.
SENSITIVE_KINDS = {
    "extractor",
    "reviewer",
    "query_expansion",
    "role_word",
    "temporal_subcall",
}


def contains_sensitive(text: str) -> str | None:
    """Return the matched identifier label if the text carries PII, else None."""
    if not text:
        return None
    for pattern, label in SENSITIVE_PATTERNS:
        if pattern.search(text):
            return label
    return None


# Grouping for the report: conversation-derived kinds get the
# sensitive-identifier gate; store-derived kinds are governed by their
# own config flags plus local_only.
GROUPS = [
    (
        "Conversation-derived (sensitive-identifier gated)",
        ["extractor", "reviewer", "query_expansion", "role_word", "temporal_subcall"],
    ),
    ("Store-derived (config-gated)", ["graph_typing", "distillation"]),
]

# ---------------------------------------------------------------------------
# Live configuration.
# ---------------------------------------------------------------------------

def _hermes_home() -> Path:
    env = os.environ.get("HERMES_HOME")
    if env:
        return Path(env)
    return Path.home() / "AppData" / "Local" / "hermes"


def load_config() -> dict:
    """Load the live plugin configuration (hybrid_memory.json).

    Mirrors temporal_subcall's loader: checks the plugins directory and the
    hermes home root, and layers the env-var overrides on top.
    """
    cfg: dict = {}
    home = _hermes_home()
    candidates = [
        home / "plugins" / "hybrid_memory" / "hybrid_memory.json",
        home / "hybrid_memory.json",
    ]
    for path in candidates:
        try:
            if path.is_file():
                import json

                cfg.update(json.loads(path.read_text(encoding="utf-8")))
                break
        except Exception:
            logger.debug("egress: could not read %s", path, exc_info=True)
    for key, raw in (("LLM_MODEL", "llm_model"), ("LLM_PROVIDER", "llm_provider")):
        env = os.environ.get(f"HERMES_HYBRID_{key}")
        if env:
            cfg[raw] = env
    return cfg


def _flag(cfg: dict, key: str, default: bool) -> bool:
    val = cfg.get(key, "true" if default else "false")
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("true", "1", "yes", "on")


def local_only(cfg: dict | None = None) -> bool:
    """True when the plugin must not make any LLM auxiliary calls."""
    return _flag(cfg if cfg is not None else load_config(), "local_only", False)


def gate(kind: str, text: str = "", cfg: dict | None = None) -> bool:
    """True = the call may proceed. False = refuse (fail soft at the caller).

    Refuses when ``local_only`` is on (all kinds), or when a
    conversation-derived payload contains a PII identifier.
    """
    if kind not in {site["kind"] for site in SITES}:
        logger.warning("egress gate: unknown kind %r (defaulting to allowed)", kind)
        return True
    cfg = cfg if cfg is not None else load_config()
    if local_only(cfg):
        return False
    if kind in SENSITIVE_KINDS:
        label = contains_sensitive(text)
        if label is not None:
            logger.info(
                "egress gate: refusing %s call (sensitive payload: %s)",
                kind,
                label,
            )
            return False
    return True


# ---------------------------------------------------------------------------
# Report.
# ---------------------------------------------------------------------------

def site_live(site: dict, cfg: dict | None = None) -> str:
    """'blocked' (local_only), 'ON', or 'OFF' for one egress site."""
    cfg = cfg if cfg is not None else load_config()
    if local_only(cfg):
        return "blocked"
    return "ON" if _flag(cfg, site["gate"], site["default"]) else "OFF"


def report(cfg: dict | None = None) -> str:
    """Render the 'what leaves this machine' report for the live config."""
    cfg = cfg if cfg is not None else load_config()
    lo = local_only(cfg)
    lines = [
        "hybrid-memory egress report",
        "=========================",
        f"local_only: {lo}",
        "",
        "LLM-bound auxiliary calls (plugin-owned):",
    ]
    by_kind = {site["kind"]: site for site in SITES}
    for group_name, kinds in GROUPS:
        lines.append(f"  [{group_name}]")
        for kind in kinds:
            site = by_kind[kind]
            lines.append(
                f"    {kind:<17} {site['payload'][:32]:<33} "
                f"{site['gate'][:31]:<32} {site_live(site, cfg)}"
            )
    lines.append("")
    lines.append("Context providers (no memory data):")
    for site in CONTEXT_NETWORK:
        lines.append(
            f"  {site['kind']:<17} {site['payload'][:32]:<33} {site['note']}"
        )
    lines.append("")
    lines.append("Embeddings: local model only (no egress).")
    if lo:
        lines.append(
            "Verdict: local_only is ON — no plugin-owned LLM call can run; "
            "all sites fail soft."
        )
    else:
        on_kinds = [kind for _, kinds in GROUPS for kind in kinds
                    if _flag(cfg, by_kind[kind]["gate"], by_kind[kind]["default"])]
        lines.append(
            "Verdict: "
            + (", ".join(on_kinds) if on_kinds else "no plugin-owned LLM calls")
            + " may send data outside this machine."
        )
    return "\n".join(lines)