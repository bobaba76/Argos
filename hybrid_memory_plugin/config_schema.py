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
            default="BAAI/bge-small-en-v1.5",
            description="Sentence-transformers model for local embeddings (~130MB).",
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
            description="Extract memory proposals from each conversation turn.",
            info="When enabled, generic patterns and optional LLM fallback create pending proposals. They are not active memory until reviewed; the agent can save explicitly via memory_save.",
            inline=True,
            group="Extraction",
        ),
        ProviderField(
            key="llm_fallback",
            label="LLM fallback extraction",
            kind=KIND_BOOL,
            default="true",
            description="Use the host LLM to create proposals when patterns miss facts.",
            info="Adds latency and token cost per turn, but proposals remain pending until reviewed. Disable for regex-only proposals.",
            inline=True,
            group="Extraction",
        ),
        ProviderField(
            key="auto_review",
            label="Automatic proposal review",
            kind=KIND_BOOL,
            default="true",
            description="Have the auxiliary LLM review new proposals.",
            info="Obvious junk is quarantined automatically; sensitive or contextless proposals stay pending for confirmation.",
            inline=True,
            group="Extraction",
        ),
        ProviderField(
            key="storage_mode",
            label="Storage mode",
            kind=KIND_SELECT,
            default="shared_service",
            description="Use one local memory service for all Hermes surfaces.",
            info="Shared service prevents desktop/gateway split-brain and DuckDB writer locks. Direct mode is for attributetics only.",
            options=(
                ProviderFieldOption("shared_service", "Shared service"),
                ProviderFieldOption("local", "Direct files"),
            ),
            group="Storage",
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
