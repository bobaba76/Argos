"""Argos temporal sub-call — the price-engineered answerer (P2A option 2).

Problem: the intent router's original design switched the WHOLE answerer to
the smart model, which sent the entire ~124k-token context (full history +
all injected memories + tool defs) to the expensive model on every temporal
query — ~R2.50 each.

Fix: when a genuine temporal/multi-hop query is detected, instead of
switching the whole turn, Argos makes ONE small dedicated call with a
TRIMMED context — just the question plus a handful of dated memory records
(~a few k tokens) — and returns a short answer. The cheap Flash answerer
relays it. Cost drops to ~1/10th while keeping the temporal date-math
quality the smart model measured +26pp better at.

Provider: the call goes through the host's auxiliary LLM client
(agent.auxiliary_client.call_llm), which honors the configured provider and
model — no hard-coded vendor endpoint. The optional ``llm_model`` /
``llm_provider`` keys in hybrid_memory.json override the host defaults when
set.

Fail-soft: any error returns "" so the turn is never broken (Flash just
answers as best it can).
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_MAX_EVIDENCE_CHARS = 5000
_MAX_TOKENS = 260
_TIMEOUT_SECONDS = 30

_SYSTEM_PROMPT = (
    "You are a precise temporal fact-checker. Using ONLY the provided stored "
    "facts, answer the user's date/time/sequence question in one or two short "
    "sentences. State dates clearly (e.g. '10 August 2026'). If the facts are "
    "insufficient, say you cannot determine it from the stored facts. Never "
    "invent facts. Do not use tools."
)


def _hermes_home() -> Path:
    try:
        from hermes_constants import get_hermes_home
        return Path(get_hermes_home())
    except Exception:
        return Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")))


def _config_overrides() -> tuple[str | None, str | None]:
    """Optional llm_model / llm_provider from hybrid_memory.json (empty = host defaults)."""
    try:
        config_path = _hermes_home() / "hybrid_memory.json"
        if config_path.exists():
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            return cfg.get("llm_model") or None, cfg.get("llm_provider") or None
    except Exception:
        pass
    return None, None


def temporal_answer(question: str, evidence_text: str) -> str:
    """One trimmed call via the host's configured LLM client; '' fail-soft."""
    question = (question or "").strip()
    if not question:
        return ""
    try:
        from agent.auxiliary_client import call_llm
    except Exception as exc:
        logger.warning("temporal_subcall: host LLM client unavailable: %s", exc)
        return ""

    model, provider = _config_overrides()
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Stored facts (with dates):\n{evidence_text[: _MAX_EVIDENCE_CHARS]}"
            ),
        },
    ]
    try:
        response = call_llm(
            task="temporal_subcall",
            messages=messages,
            temperature=0.1,
            max_tokens=_MAX_TOKENS,
            timeout=_TIMEOUT_SECONDS,
            model=model,
            provider=provider,
        )
    except Exception as exc:
        logger.warning("temporal_subcall failed: %s", exc)
        return ""
    if response is None:
        return ""
    text = response
    if hasattr(response, "choices"):
        try:
            text = response.choices[0].message.content
        except Exception:
            return ""
    return str(text or "").strip()


def format_evidence(records) -> str:
    """Turn dated MemoryRecords into a compact evidence block."""
    lines = []
    for r in records or []:
        ts = getattr(r, "created_at", "") or ""
        content = (getattr(r, "content", "") or "").strip()
        if content:
            lines.append(f"[{ts}] {content}")
    return "\n".join(lines)