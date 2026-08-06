"""Storage routing rules for hybrid memory provider instances."""
from __future__ import annotations

from typing import Tuple

# Hermes uses these values for local conversation surfaces. In particular,
# the desktop chat surface is tagged "desktop", not "cli".
LOCAL_PLATFORMS = frozenset({"cli", "desktop", "tui", "local"})


def resolve_storage_names(
    platform: str | None,
    database_filename: str,
    graph_dirname: str,
) -> Tuple[str, str]:
    """Return the database/graph names for a Hermes platform."""
    platform_name = (platform or "cli").strip().lower()
    if platform_name in LOCAL_PLATFORMS:
        return database_filename, graph_dirname

    if not database_filename.endswith("_gateway.duckdb"):
        database_filename = database_filename.replace(".duckdb", "_gateway.duckdb")
    if not graph_dirname.endswith("_gateway"):
        graph_dirname = graph_dirname + "_gateway"
    return database_filename, graph_dirname
