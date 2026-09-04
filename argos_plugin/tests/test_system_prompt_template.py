"""Test that the system prompt template is byte-stable and matches
the historical constant (#247).

The prompt text MUST remain byte-identical to the original f-string
for prompt-cache stability. This test asserts that the template file
produces the exact same output as the historical constant.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))


# The exact historical constant — the original f-string output with
# graph_status="available". This is the byte-stable reference.
_HISTORICAL_AVAILABLE = (
    "# Argos (Local)\n"
    "Active. Relationship graph: available.\n"
    "You have persistent memory of this user from past conversations — "
    "any topic: personal life, work, tech, hobbies, relationships. "
    "Relevant memories are auto-injected before each turn. For deeper or "
    "multi-hop lookups, call memory_search with different wording.\n"
    "Categories: personal_fact (stable facts), preference (how they like things), "
    "insight (self-observations, realizations), event (life events, milestones), "
    "relationship (people in their life), goal (what they're working toward), "
    "context_note (situational context).\n"
    "When the user states a durable fact, preference, or insight — about "
    "ANY topic — call memory_save immediately — don't wait to be asked. "
    "Automatic extraction creates pending proposals, not active memories. "
    "Review them with memory_candidate_list and memory_candidate_review; "
    "never approve a proposal merely because another model produced it.\n"
    "\n"
    "## Save reasoning, not just conclusions\n"
    "When you work through a non-trivial topic with the user — technical reasoning, "
    "analytical reasoning, trade-off analysis, decision-making, important "
    "decisions — save the REASONING CHAIN, not just the final conclusion. "
    "A bare fact like 'Fact-A might be Fact-B' is far less useful than the full "
    "reasoning: what evidence supports it, what was considered and ruled out, "
    "what the uncertainty level is, and what would confirm or deny it. "
    "Use the content field to store a self-contained reasoning summary "
    "(200-800 chars is fine — the system handles long content). "
    "This ensures future sessions can reconstruct WHY a conclusion was reached, "
    "not just WHAT it was.\n"
    "\n"
    "## Quality over quantity\n"
    "Don't save trivial facts the agent could infer from context ('user uses "
    "a keyboard', 'user is typing'). Don't save fragments of your own output. "
    "Don't save the same fact in slightly different wording. One rich, "
    "well-reasoned memory is worth ten shallow flashcards.\n"
    "Use memory_graph_search to find relationships between people, tools, "
    "and concepts in the user's life."
)

_HISTORICAL_UNAVAILABLE = _HISTORICAL_AVAILABLE.replace(
    "Active. Relationship graph: available.",
    "Active. Relationship graph: unavailable.",
)


def test_template_file_exists():
    """#247: the template file exists alongside the module."""
    template_path = _plugin_dir / "system_prompt_template.txt"
    assert template_path.is_file(), f"Template file not found at {template_path}"


def test_template_has_graph_status_placeholder():
    """#247: the template has exactly one {graph_status} placeholder."""
    from provider_retrieval import _load_system_prompt
    template = _load_system_prompt()
    assert "{graph_status}" in template
    assert template.count("{graph_status}") == 1


def test_template_matches_historical_available():
    """#247: template.format(graph_status='available') matches the
    historical constant byte-for-byte."""
    from provider_retrieval import _load_system_prompt
    rendered = _load_system_prompt().format(graph_status="available")
    assert rendered == _HISTORICAL_AVAILABLE, (
        "Template output does not match historical constant! "
        f"Template length: {len(rendered)}, historical: {len(_HISTORICAL_AVAILABLE)}"
    )


def test_template_matches_historical_unavailable():
    """#247: template.format(graph_status='unavailable') matches the
    historical constant byte-for-byte."""
    from provider_retrieval import _load_system_prompt
    rendered = _load_system_prompt().format(graph_status="unavailable")
    assert rendered == _HISTORICAL_UNAVAILABLE


def test_system_prompt_block_uses_template():
    """#247: system_prompt_block() loads from the template file, not
    an inline f-string."""
    import inspect
    from provider_retrieval import ProviderRetrievalMixin
    src = inspect.getsource(ProviderRetrievalMixin.system_prompt_block)
    assert "_load_system_prompt" in src
    # Should NOT have the old inline f-string.
    assert "f\"Active. Relationship graph" not in src


def test_template_is_cached():
    """#247: _load_system_prompt is cached (lru_cache)."""
    from provider_retrieval import _load_system_prompt
    # Two calls should return the same object (cached).
    a = _load_system_prompt()
    b = _load_system_prompt()
    assert a is b  # identity check — same cached object
