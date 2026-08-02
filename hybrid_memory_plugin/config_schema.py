"""Declarative config schema for the hybrid_memory memory provider.

Surfaces configuration in the Hermes desktop UI config panel. This file
only imports from plugins.memory.config_schema (pure data, no agent runtime).
"""
from __future__ import annotations

from plugins.memory.config_schema import (
    ProviderConfigSchema,
    ProviderField,
    ProviderFieldOption,
    KIND_TEXT,
    KIND_SELECT,
    KIND_SECRET,
    KIND_BOOL,
    KIND_NUMBER,
    STORAGE_FLAT_JSON,
)

CONFIG_SCHEMA = ProviderConfigSchema(
    name="hybrid_memory",
    label="Hybrid Memory (Local)",
    storage=STORAGE_FLAT_JSON,
    fields=(
        ProviderField(
            key="local_embedding_model",
            label="Embedding model",
            kind=KIND_TEXT,
            default="sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
            description="Sentence-transformers model for local embeddings (~90MB).",
            info="If the model fails to load, the plugin falls back to text search automatically.",
            inline=True,
            group="Embeddings",
        ),
        ProviderField(
            key="max_injected_items",
            label="Max injected memories",
            kind=KIND_NUMBER,
            default="8",
            description="Maximum memories auto-injected as context before each turn.",
            inline=True,
            group="Retrieval",
        ),
        ProviderField(
            key="auto_extract",
            label="Auto-extract facts",
            kind=KIND_BOOL,
            default="true",
            description="Automatically extract durable facts from each conversation turn.",
            info="When enabled, generic syntactic patterns scan each turn for durable statements (any topic) and save them. If patterns miss facts, an LLM fallback can extract them. The agent can also save explicitly via memory_save.",
            inline=True,
            group="Extraction",
        ),
        ProviderField(
            key="llm_fallback",
            label="LLM fallback extraction",
            kind=KIND_BOOL,
            default="true",
            description="Use the host LLM to extract facts when regex patterns miss them.",
            info="Adds ~1-3s latency and token cost per turn, but catches facts the patterns miss. Disable for zero-cost extraction (regex only).",
            inline=True,
            group="Extraction",
        ),
        ProviderField(
            key="database_filename",
            label="DuckDB filename",
            kind=KIND_TEXT,
            default="hybrid_memory.duckdb",
            description="DuckDB database filename (stored in HERMES_HOME).",
            group="Storage",
        ),
        ProviderField(
            key="graph_dirname",
            label="Kuzu graph directory",
            kind=KIND_TEXT,
            default="hybrid_memory_kuzu",
            description="Kuzu graph database directory (stored in HERMES_HOME).",
            group="Storage",
        ),
    ),
)
