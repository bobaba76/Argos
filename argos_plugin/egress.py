"""Egress inventory and gate for LLM-bound auxiliary calls.

The argos plugin makes its own LLM calls beyond the host's normal
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
import types
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
        # E5: ``None`` means "always ON when not local_only". The old value
        # ``"none (always-on, rare)"`` was a descriptive string that
        # ``_flag`` looked up as a config key, never found, and silently
        # fell back to the default — confusing and misleading.
        "gate": None,
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
    {
        "kind": "watcher_extraction",
        "file": "watcher.py",
        "trigger": "hot-doc fact extraction (spec-07 D5)",
        "payload": "document text (PDF/XLSX/CSV/DOCX extraction input)",
        "gate": "llm_fallback",
        "default": True,
    },
    {
        "kind": "memory_rollup",
        "file": "rollup.py",
        "trigger": "long-horizon rollup pass (P5.1 Phase 3)",
        "payload": "stored memory content (oldest active low-retrieval records)",
        "gate": "rollup_enabled",
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
    # E4: obfuscated email ("user at example dot com") — common evasion.
    (
        re.compile(
            r"\b[A-Za-z0-9._%+-]+\s+(?:at|@)\s+[A-Za-z0-9.-]+\s+(?:dot|\.|\(dot\))\s+[A-Za-z]{2,}\b",
            re.IGNORECASE,
        ),
        "obfuscated email address",
    ),
    # E4: SA phone with optional separators (spaces/dashes). The old
    # ``\d{9}`` required 9 consecutive digits and missed "+27 82 123 4567"
    # and "082-123-4567".
    (
        re.compile(r"(?<!\d)(?:\+?27|0)[\s-]?\d{2}[\s-]?\d{3}[\s-]?\d{4}(?!\d)"),
        "South African phone number",
    ),
    # E4: international phone format — a leading + followed by 7-15 digits
    # with optional separators. Catches US/UK/EU numbers that the SA-only
    # pattern missed.
    (
        re.compile(r"(?<!\d)\+\d{1,3}(?:[\s-]?\d){6,14}(?!\d)"),
        "international phone number",
    ),
    # E4: 13-digit ID. The old ``\b\d{13}\b`` also matched Unix timestamps
    # in milliseconds (e.g. 1700000000000). Require the SA-ID checksum
    # shape: YYMMDD followed by 7 digits, and exclude pure-millisecond
    # timestamps by requiring the first 6 digits to be a plausible date
    # (month 01-12, day 01-31). This is a heuristic filter, not a full
    # Luhn check, but it removes the most common false positives.
    (
        re.compile(r"(?<!\d)(\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{7}(?!\d)"),
        "13-digit ID number",
    ),
    # E4: US Social Security Number (XXX-XX-XXXX).
    (
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "US Social Security number",
    ),
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
    "watcher_extraction",
}


def contains_sensitive(text: str) -> str | None:
    """Return the matched identifier label if the text carries PII, else None.

    E8: previously returned on the first match, so a text with both an email
    and a phone only reported the email. Now all matches are collected and
    logged (for audit/diagnostic value) before the first label is returned
    (the call is blocked either way, so the return value stays compatible
    with existing callers that only need a yes/no + first label).
    """
    if not text:
        return None
    found: list[str] = []
    for pattern, label in SENSITIVE_PATTERNS:
        if pattern.search(text):
            found.append(label)
    if not found:
        return None
    if len(found) > 1:
        logger.info(
            "egress: multiple sensitive identifiers found (%s); blocking on first",
            ", ".join(found),
        )
    return found[0]


def all_sensitive_labels(text: str) -> list[str]:
    """Return labels for ALL sensitive identifiers in *text* (E8).

    Unlike ``contains_sensitive`` (which returns the first label for
    backward compatibility), this returns every match for audit/diagnostic
    use.
    """
    if not text:
        return []
    return [label for pattern, label in SENSITIVE_PATTERNS if pattern.search(text)]


# Grouping for the report: conversation-derived kinds get the
# sensitive-identifier gate; store-derived kinds are governed by their
# own config flags plus local_only.
GROUPS = [
    (
        "Conversation-derived (sensitive-identifier gated)",
        ["extractor", "reviewer", "query_expansion", "role_word", "temporal_subcall"],
    ),
    ("Store-derived (config-gated)", ["graph_typing", "distillation"]),
    ("Document-derived (sensitive-identifier gated)", ["watcher_extraction"]),
    ("Lifecycle-derived (config-gated)", ["memory_rollup"]),
]

# ---------------------------------------------------------------------------
# Live configuration.
# ---------------------------------------------------------------------------

def _hermes_home() -> Path:
    """Return the Hermes home directory (E1: cross-platform).

    Previously this hardcoded the Windows path ``~/AppData/Local/hermes``,
    which created a nonsensical path on macOS/Linux and caused load_config
    to silently default to empty config (all flags default) on non-Windows.
    Now mirrors provider_core.py: try ``hermes_constants.get_hermes_home()``,
    fall back to ``$HERMES_HOME`` then ``~/.hermes``.
    """
    try:
        from hermes_constants import get_hermes_home
        return Path(get_hermes_home())
    except Exception:
        return Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")))


# E6: config cache with mtime-based invalidation. ``gate`` is called on the
# hot path (every auxiliary LLM call); previously each call (when cfg was
# None) did a file read + JSON parse. Negligible next to LLM latency, but
# free and matches provider_core._load_config's caching approach.
_config_cache: dict | None = None
_config_cache_mtime: float = 0.0
_config_cache_path: str = ""


def load_config() -> dict:
    """Load the live plugin configuration (hybrid_memory.json).

    Mirrors temporal_subcall's loader: checks the plugins directory and the
    hermes home root, and layers the env-var overrides on top.

    E6: the parsed config is cached and reused until the file's mtime changes
    (like provider_core._load_config), so the hot-path ``gate`` calls don't
    re-read + re-parse the JSON on every invocation.
    """
    global _config_cache, _config_cache_mtime, _config_cache_path
    home = _hermes_home()
    # E2: the live plugin directory is ``plugins/hybrid_memory`` (confirmed
    # by scripts/deploy.py), not ``plugins/argos``. The old first candidate
    # never matched, making it dead code that fell through to the second.
    candidates = [
        home / "plugins" / "hybrid_memory" / "hybrid_memory.json",
        home / "hybrid_memory.json",
    ]
    found_path: Path | None = None
    for path in candidates:
        try:
            if path.is_file():
                found_path = path
                break
        except Exception:
            logger.debug("egress: could not stat %s", path, exc_info=True)
    if found_path is None:
        # No config file on disk — return env-overlaid empty config (cached
        # briefly so we don't re-stat on every call).
        cfg: dict = {}
        _apply_env_overrides(cfg)
        # EG1: return an immutable view so callers can't mutate the cache.
        return types.MappingProxyType(cfg)
    path_str = str(found_path)
    try:
        mtime = found_path.stat().st_mtime
    except OSError:
        mtime = 0.0
    # E6: serve from cache if the path + mtime are unchanged.
    if _config_cache is not None and _config_cache_path == path_str and _config_cache_mtime == mtime:
        return _config_cache
    try:
        import json

        cfg = json.loads(found_path.read_text(encoding="utf-8"))
    except Exception:
        logger.debug("egress: could not read %s", found_path, exc_info=True)
        cfg = {}
    _apply_env_overrides(cfg)
    # EG1: wrap in MappingProxyType so callers get a read-only view.
    # One mutating caller could otherwise poison every subsequent gate
    # decision (e.g. flipping local_only or a site flag with no error).
    _config_cache = types.MappingProxyType(cfg)
    _config_cache_mtime = mtime
    _config_cache_path = path_str
    return _config_cache


def _apply_env_overrides(cfg: dict) -> None:
    """Layer HERMES_HYBRID_* env-var overrides onto *cfg* in place."""
    for key, raw in (("LLM_MODEL", "llm_model"), ("LLM_PROVIDER", "llm_provider")):
        env = os.environ.get(f"HERMES_HYBRID_{key}")
        if env:
            cfg[raw] = env


def _reset_config_cache() -> None:
    """Test hook: clear the config cache so a new file is re-read."""
    global _config_cache, _config_cache_mtime, _config_cache_path
    _config_cache = None
    _config_cache_mtime = 0.0
    _config_cache_path = ""


def _flag(cfg: dict, key: str, default: bool) -> bool:
    val = cfg.get(key, "true" if default else "false")
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("true", "1", "yes", "on")


def local_only(cfg: dict | None = None) -> bool:
    """True when the plugin must not make any LLM auxiliary calls."""
    return _flag(cfg if cfg is not None else load_config(), "local_only", False)


# E7: the set of known kinds is rebuilt on every gate call; hoist it to a
# module-level constant so the set comprehension is not reconstructed each
# time (negligible for 9 sites, but free and clearer).
_KNOWN_KINDS = {site["kind"] for site in SITES}
# E3: map kind → site dict for O(1) lookup of the per-site gate flag in
# ``gate()`` (previously only ``site_live`` consulted the per-site flag,
# creating an inconsistency where the report showed OFF but the gate allowed
# the call).
_SITES_BY_KIND = {site["kind"]: site for site in SITES}


def gate(kind: str, text: str = "", cfg: dict | None = None) -> bool:
    """True = the call may proceed. False = refuse (fail soft at the caller).

    Refuses when ``local_only`` is on (all kinds), or when a
    conversation-derived payload contains a PII identifier.

    E3: also enforces the per-site config flag (``site["gate"]``). Previously
    only callers checked their own flag, so a caller that forgot to check it
    could make a call the report showed as OFF. Now the gate is the single
    enforcement point — a site whose flag is OFF is refused here regardless
    of what the caller does.
    """
    if kind not in _KNOWN_KINDS:
        logger.warning("egress gate: unknown kind %r (defaulting to blocked)", kind)
        return False
    cfg = cfg if cfg is not None else load_config()
    if local_only(cfg):
        return False
    # E3: enforce the per-site flag. A ``None`` gate means "always ON when
    # not local_only" (e.g. the rare role-word site).
    site = _SITES_BY_KIND[kind]
    site_gate = site.get("gate")
    if site_gate is not None and not _flag(cfg, site_gate, site["default"]):
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
    # E5: a ``None`` gate means "always ON when not local_only" (e.g. the
    # rare role-word site). Avoids treating a descriptive string as a key.
    if site.get("gate") is None:
        return "ON"
    return "ON" if _flag(cfg, site["gate"], site["default"]) else "OFF"


def report(cfg: dict | None = None) -> str:
    """Render the 'what leaves this machine' report for the live config."""
    cfg = cfg if cfg is not None else load_config()
    lo = local_only(cfg)
    lines = [
        "argos egress report",
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
            # E5: gate may be None (always-on sites); render as "always-on".
            gate_str = site["gate"] if site["gate"] is not None else "always-on"
            lines.append(
                f"    {kind:<17} {site['payload'][:32]:<33} "
                f"{gate_str[:31]:<32} {site_live(site, cfg)}"
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