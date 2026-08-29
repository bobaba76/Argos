"""Compile-to-handoff consumer for project proposals (#48).

Reads a project's approved proposals + active memories and renders a
copy-paste-ready markdown handoff block:

- ISSUE — the project's tracking issue URL, if one is set
- STATE — active facts relevant to the project
- TODO — approved-but-unconsumed proposals
- GOTCHAS — resolved conflicts / tombstone notes

Consumer-side only: no write-path changes to the store or reviewer.
Output goes to stdout; optional file write behind a flag.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


def compile_handoff(
    store,
    project_id: str | None = None,
    *,
    issue_url: str = "",
    max_state_items: int = 20,
    max_todo_items: int = 20,
    max_gotchas_items: int = 10,
) -> str:
    """Compile a handoff block from a project's store state.

    Args:
        store: A DuckDBMemoryStore (or compatible) instance.
        project_id: The project to compile. None for global scope.
        issue_url: Optional tracking issue URL for the project.
        max_state_items: Max active facts to include in STATE.
        max_todo_items: Max approved proposals to include in TODO.
        max_gotchas_items: Max conflict/tombstone notes in GOTCHAS.

    Returns:
        A markdown string with the handoff block.
    """
    parts: list[str] = []

    # ISSUE — tracking issue URL.
    if issue_url:
        parts.append(f"## ISSUE\n{issue_url}\n")

    # STATE — active facts relevant to the project.
    state_items = _collect_state(store, project_id, max_state_items)
    parts.append("## STATE")
    if state_items:
        for item in state_items:
            parts.append(f"- {item}")
    else:
        parts.append("- (no active facts for this project)")
    parts.append("")

    # TODO — approved-but-unconsumed proposals.
    todo_items = _collect_todo(store, project_id, max_todo_items)
    parts.append("## TODO")
    if todo_items:
        for item in todo_items:
            parts.append(f"- {item}")
    else:
        parts.append("- (no pending proposals for this project)")
    parts.append("")

    # GOTCHAS — resolved conflicts / tombstone notes.
    gotchas_items = _collect_gotchas(store, project_id, max_gotchas_items)
    parts.append("## GOTCHAS")
    if gotchas_items:
        for item in gotchas_items:
            parts.append(f"- {item}")
    else:
        parts.append("- (no known gotchas)")
    parts.append("")

    return "\n".join(parts)


def _collect_state(
    store, project_id: str | None, limit: int,
) -> list[str]:
    """Collect active facts for the project (STATE section)."""
    try:
        results = store.search(
            "*",
            limit=limit,
            project_id=project_id,
            suppress_retrieval=True,
        )
        return [
            f"[{r.category}] {r.content[:120]}"
            for r in results
            if hasattr(r, "content") and hasattr(r, "category")
        ]
    except Exception:
        return []


def _collect_todo(
    store, project_id: str | None, limit: int,
) -> list[str]:
    """Collect pending proposals for the project (TODO section)."""
    try:
        digest = store.project_digest(project_id=project_id, status="pending", limit=limit)
        items: list[str] = []
        for cand in digest.get("global_candidates", []):
            content = (cand.get("content") or "")[:120]
            cat = cand.get("category", "")
            items.append(f"[{cat}] {content}")
        for proj in digest.get("projects", []):
            for cand in proj.get("candidates", []):
                content = (cand.get("content") or "")[:120]
                cat = cand.get("category", "")
                items.append(f"[{cat}] {content}")
        return items[:limit]
    except Exception:
        return []


def _collect_gotchas(
    store, project_id: str | None, limit: int,
) -> list[str]:
    """Collect resolved conflicts / tombstone notes (GOTCHAS section)."""
    try:
        # Tombstones — deleted/superseded content that should not re-appear.
        tombstones = store.list_tombstones(limit=limit)
        return [
            f"Deleted: {t.get('reason', 'unknown')} ({t.get('category', '')})"
            for t in tombstones
            if isinstance(t, dict)
        ]
    except Exception:
        return []


def compile_handoff_to_file(
    store,
    output_path: str,
    project_id: str | None = None,
    *,
    issue_url: str = "",
    **kwargs,
) -> bool:
    """Compile a handoff block and write it to a file.

    Returns True if the file was written successfully.
    """
    try:
        content = compile_handoff(store, project_id, issue_url=issue_url, **kwargs)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    except Exception:
        return False
