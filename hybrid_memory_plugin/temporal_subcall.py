"""Argos temporal sub-call — the price-engineered answerer (P2A option 2).

Problem: the intent router's original design switched the WHOLE answerer to
the smart model (deepseek-v4-pro-0813 on openrouter), which sent the entire
~124k-token context (full history + all injected memories + tool defs) to the
expensive model on every temporal query — ~R2.50 each.

Fix: when a genuine temporal/multi-hop query is detected, instead of
switching the whole turn, Argos makes ONE small dedicated call to the smart
model with a TRIMMED context — just the question plus a handful of dated
memory records (~a few k tokens) — and returns a short answer.  The cheap
Flash answerer relays it.  Cost drops to ~1/10th while keeping the temporal
date-math quality the smart model measured +26pp better at.

Fail-soft: any error returns "" so the turn is never broken (Flash just
answers as best it can).
"""
from __future__ import annotations

import json
import logging
import os
import urllib.request
import urllib.error
from pathlib import Path

logger = logging.getLogger(__name__)

_SMART_MODEL = "deepseek/deepseek-v4-pro-0813"
_BASE_URL = "https://openrouter.ai/api/v1"
_MAX_EVIDENCE_CHARS = 5000
_MAX_TOKENS = 260
_TIMEOUT_SECONDS = 30


def _hermes_home() -> Path:
    try:
        from hermes_constants import get_hermes_home
        return Path(get_hermes_home())
    except Exception:
        return Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")))


def _resolve_key() -> str:
    env = _hermes_home() / ".env"
    if env.exists():
        try:
            for line in env.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if line.startswith("OPENROUTER_API_KEY="):
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if val:
                        return val
        except Exception:
            pass
    return os.environ.get("OPENROUTER_API_KEY", "")


def temporal_answer(question: str, evidence_text: str, api_key: str = "") -> str:
    """One trimmed call to the smart model; returns a short answer or ''."""
    question = (question or "").strip()
    if not question:
        return ""
    if not api_key:
        api_key = _resolve_key()
    if not api_key:
        logger.warning("temporal_subcall: no OPENROUTER_API_KEY available")
        return ""

    system = (
        "You are a precise temporal fact-checker. Using ONLY the provided stored "
        "facts, answer the user's date/time/sequence question in one or two short "
        "sentences. State dates clearly (e.g. '10 August 2026'). If the facts are "
        "insufficient, say you cannot determine it from the stored facts. Never "
        "invent facts. Do not use tools."
    )
    user = (
        f"Question: {question}\n\n"
        f"Stored facts (with dates):\n{evidence_text[: _MAX_EVIDENCE_CHARS]}"
    )

    body = {
        "model": _SMART_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": _MAX_TOKENS,
        "temperature": 0.1,
        # Turn off deep-thinking to keep the sub-call fast and cheap.
        "reasoning": {"enabled": False, "effort": "low"},
    }
    req = urllib.request.Request(
        f"{_BASE_URL}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://hermes.local",
            "X-Title": "Hermes-Argos",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_SECONDS) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return ((data.get("choices") or [{}])[0].get("message") or {}).get(
            "content", ""
        ).strip()
    except urllib.error.HTTPError as exc:
        logger.warning("temporal_subcall HTTP %s: %s", exc.code, exc.read()[:300])
        return ""
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("temporal_subcall failed: %s", exc)
        return ""


def format_evidence(records) -> str:
    """Turn dated MemoryRecords into a compact evidence block."""
    lines = []
    for r in records or []:
        ts = getattr(r, "created_at", "") or ""
        content = (getattr(r, "content", "") or "").strip()
        if content:
            lines.append(f"[{ts}] {content}")
    return "\n".join(lines)