"""Shared LLM JSON parsing utility.

Extracts the code-fence stripping and prose-wrapped JSON extraction
logic that was duplicated across reviewer.py, distillation.py, and
rollup.py (issues R4, D2, D6).

All three parsers handle:
1. Fenced JSON: ``\\`\\`\\`json {...} \\`\\`\\``` → fences stripped.
2. Pure JSON: ``{...}`` or ``[...]`` → unchanged.
3. Prose-wrapped JSON: ``Here is the JSON: {...} Done.`` → extracts
   the first balanced ``{...}`` or ``[...]`` block via the JSON
   decoder's ``raw_decode``.

This module provides two functions:
- ``parse_llm_json_object(text)`` → ``dict | None`` (for reviewer/distillation)
- ``parse_llm_json_array(text)`` → ``list | None`` (for rollup)
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

# Strip a single pair of leading/trailing markdown code fences.
_FENCE_START_RE = re.compile(r"^```(?:json)?\s*")
_FENCE_END_RE = re.compile(r"\s*```$")


def _strip_fences(text: str) -> str:
    """Strip a single pair of leading/trailing markdown code fences."""
    if text.startswith("```"):
        text = _FENCE_START_RE.sub("", text)
        text = _FENCE_END_RE.sub("", text).strip()
    return text


def parse_llm_json_object(text: str) -> Optional[dict]:
    """Parse an LLM response as a JSON object (dict).

    Returns ``None`` if the text is empty, not valid JSON, or not a dict.
    Handles code-fenced and prose-wrapped JSON.
    """
    if not text or not text.strip():
        return None
    text = _strip_fences(text.strip())
    # Try a direct parse first (the common case).
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        # Fallback: extract the first balanced {...} object.
        start = text.find("{")
        if start == -1:
            return None
        try:
            value, _end = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError:
            return None
    if not isinstance(value, dict):
        return None
    return value


def parse_llm_json_array(text: str) -> Optional[list]:
    """Parse an LLM response as a JSON array (list).

    Returns ``None`` if the text is empty, not valid JSON, or not a list.
    Handles code-fenced and prose-wrapped JSON.
    """
    if not text or not text.strip():
        return None
    text = _strip_fences(text.strip())
    # Try a direct parse first (the common case).
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        # Fallback: extract the first balanced [...] block.
        start = text.find("[")
        if start == -1:
            return None
        try:
            value, _end = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError:
            return None
    if not isinstance(value, list):
        return None
    return value


def parse_llm_response(response: Any) -> Optional[dict]:
    """Extract text from an LLM response object and parse as JSON object.

    Handles the ``response.choices[0].message.content`` extraction common
    to reviewer.py and distillation.py. Returns ``None`` on any extraction
    or parse failure.
    """
    try:
        text = response.choices[0].message.content
    except (AttributeError, IndexError, KeyError, TypeError):
        return None
    return parse_llm_json_object(text)
