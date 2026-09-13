"""Spec-09 (#124): MCP stdio server — read tier.

Transport adapter only. Exposes the READ tier of the public allowlist
(search, fetch, fetch_history, explain, explain_retrieval, capabilities)
over MCP stdio, behind the facade from #123.

Protocol: MCP JSON-RPC 2.0 over stdio (newline-delimited).
  - stdout carries ONLY valid MCP JSON-RPC messages
  - logs go to stderr
  - no startup banners on stdout
  - UTF-8, newline framing
  - bounded message size
  - deterministic tool ordering

Modern protocol era (2025-06-18 line, per spec Decision 2):
  - initialize → capability negotiation
  - notifications/initialized → ready
  - tools/list → tool definitions with strict inputSchema
  - tools/call → invoke facade operation, return structured result

Strict tool schemas: additionalProperties: false, max string lengths,
max result counts, enum validation. No caller-controlled internal flags
(include_quarantined, include_archived, include_expired).

Output: structured results with outputSchema, not provider JSON strings.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

# JSON-RPC 2.0 error codes (per spec).
JSONRPC_PARSE_ERROR = -32700
JSONRPC_INVALID_REQUEST = -32600
JSONRPC_METHOD_NOT_FOUND = -32601
JSONRPC_INVALID_PARAMS = -32602
JSONRPC_INTERNAL_ERROR = -32603

# MCP protocol version (2025-06-18 line, per spec Decision 2).
MCP_PROTOCOL_VERSION = "2025-06-18"

# Bounded message size (D6: bounded messages).
MAX_MESSAGE_BYTES = 4 * 1024 * 1024  # 4 MiB

logger = logging.getLogger("argos.mcp")


# -- Tool definitions (D2 read tier) -----------------------------------------

def _search_input_schema() -> Dict[str, Any]:
    """Strict input schema for memory_search."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["query"],
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language search query.",
                "maxLength": 2000,
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results (1-50).",
                "minimum": 1,
                "maximum": 50,
                "default": 10,
            },
            "category_filter": {
                "type": "string",
                "description": "Optional: filter to a specific memory category.",
            },
            "trust_class": {
                "type": "string",
                "enum": ["unreviewed", "clean"],
                "description": "Optional: filter by write-policy class (#393 S2).",
            },
        },
    }


def _fetch_input_schema() -> Dict[str, Any]:
    """Strict input schema for memory_fetch."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["memory_id"],
        "properties": {
            "memory_id": {
                "type": "string",
                "description": "The memory ID to fetch.",
            },
        },
    }


def _fetch_history_input_schema() -> Dict[str, Any]:
    """Strict input schema for memory_fetch_history."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["memory_id"],
        "properties": {
            "memory_id": {
                "type": "string",
                "description": "The memory ID to fetch version history for.",
            },
        },
    }


def _explain_input_schema() -> Dict[str, Any]:
    """Strict input schema for memory_explain (#280)."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["memory_id"],
        "properties": {
            "memory_id": {
                "type": "string",
                "description": "The memory ID to explain (provenance view).",
            },
        },
    }


def _explain_retrieval_input_schema() -> Dict[str, Any]:
    """Strict input schema for memory_why_not (retrieval diagnostic)."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["query", "memory_id"],
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query that should have surfaced the memory.",
                "maxLength": 2000,
            },
            "memory_id": {
                "type": "string",
                "description": "The memory ID that did not surface.",
                "maxLength": 256,
            },
            "top_k": {
                "type": "integer",
                "description": "Diagnostic window size (1-50).",
                "minimum": 1,
                "maximum": 50,
                "default": 20,
            },
        },
    }


def _unreviewed_input_schema() -> Dict[str, Any]:
    """Strict input schema for memory_unreviewed (no parameters)."""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {},
    }


def _capabilities_input_schema() -> Dict[str, Any]:
    """Strict input schema for memory_capabilities (no params)."""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {},
    }


def _propose_input_schema() -> Dict[str, Any]:
    """Strict input schema for memory_propose (class A write).

    The idempotency_key is required for propose operations (D5).
    Provenance fields (source, provenance_origin, grounding) are
    server-set and intentionally absent from this schema — the facade
    rejects them if the caller attempts to set them.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["content", "idempotency_key"],
        "properties": {
            "content": {
                "type": "string",
                "description": "The fact or observation to propose for review.",
                "maxLength": 10000,
            },
            "category": {
                "type": "string",
                "description": "Memory category (defaults to context_note).",
                "default": "context_note",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional tags for the proposed memory.",
            },
            "idempotency_key": {
                "type": "string",
                "description": (
                    "Client-generated unique key. Same key + same body → "
                    "returns original result (no duplicate). Same key + "
                    "different body → 409 conflict."
                ),
                "minLength": 1,
                "maxLength": 256,
            },
        },
    }


def _ingest_input_schema() -> Dict[str, Any]:
    """Strict input schema for memory_ingest (#386, Spec-12).

    Structured JSON/CSV ingestion (#289) over MCP. Preview (default)
    validates and reports — writes nothing. Apply materializes ACTIVE
    records through the self-approved candidate path and is class-C
    (loopback transports only); the facade denies apply on non-loopback.

    Provenance fields (source, provenance_origin, grounding) are
    server-set and intentionally absent from this schema — the facade
    rejects them if the caller attempts to set them.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["data", "fmt", "source_name", "mapping", "idempotency_key"],
        "properties": {
            "data": {
                "type": "string",
                "description": (
                    "Raw JSON array or CSV text to ingest. UTF-8 byte cap: "
                    "262144 bytes (enforced by the facade)."
                ),
            },
            "fmt": {"type": "string", "enum": ["json", "csv"]},
            "source_name": {
                "type": "string",
                "description": (
                    "Source file/feed name — stamped into the ingest "
                    "namespace and provenance of every row."
                ),
                "minLength": 1,
                "maxLength": 200,
            },
            "mapping": {
                "type": "object",
                "description": (
                    "Field-mapping spec: requires 'category' and "
                    "'content_template'; optional 'tags', 'key_field', and "
                    "per-field mappings (see docs/api/mcp.md)."
                ),
            },
            "mode": {
                "type": "string",
                "enum": ["preview", "apply"],
                "default": "preview",
            },
            "confirm": {
                "type": "boolean",
                "default": False,
                "description": (
                    "Human-in-loop gate. Apply requires the literal "
                    "boolean true (preview first, then confirm)."
                ),
            },
            "client_scope": {"type": "string", "maxLength": 100},
            "doc_class": {"type": "string", "maxLength": 100},
            "project_id": {"type": "string", "maxLength": 100},
            "idempotency_key": {
                "type": "string",
                "description": (
                    "Client-generated unique key. Same key + same body → "
                    "returns original result (no duplicate). Same key + "
                    "different body → 409 conflict."
                ),
                "minLength": 1,
                "maxLength": 256,
            },
        },
    }


def _save_input_schema() -> Dict[str, Any]:
    """Strict input schema for memory_save (class C write, loopback only).

    #200 Spec-10 PR-3: direct active-memory write. Only available on
    loopback transport (is_loopback=True). Non-loopback → denied.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["content", "category", "idempotency_key"],
        "properties": {
            "content": {
                "type": "string",
                "description": "The fact to save directly to active memory.",
                "maxLength": 10000,
            },
            "category": {
                "type": "string",
                "description": "Memory category (e.g. context_note, personal_fact).",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional tags.",
            },
            "idempotency_key": {
                "type": "string",
                "description": "Client-generated unique key for idempotency.",
                "minLength": 1,
                "maxLength": 256,
            },
        },
    }


def _update_input_schema() -> Dict[str, Any]:
    """Strict input schema for memory_update (class C write, loopback only)."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["memory_id", "content", "idempotency_key"],
        "properties": {
            "memory_id": {
                "type": "string",
                "description": "The memory ID to update (creates a new version).",
                "maxLength": 256,
            },
            "content": {
                "type": "string",
                "description": "The new content for the memory.",
                "maxLength": 10000,
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional new tags.",
            },
            "expected_version": {
                "type": "string",
                "description": "CAS: the last-seen memory_id. Stale → 409 conflict.",
            },
            "idempotency_key": {
                "type": "string",
                "description": "Client-generated unique key for idempotency.",
                "minLength": 1,
                "maxLength": 256,
            },
        },
    }


def _candidate_review_input_schema() -> Dict[str, Any]:
    """Strict input schema for memory_candidate_review (class B, human only).

    #200 Spec-10 PR-3: approve/reject a pending candidate. Human principal
    only — model principals are denied (no self-approval).
    #200 PR-3 fix: idempotency_key is required — aligns with REST which
    requires Idempotency-Key on POST /v1/candidates/{id}/decision. All
    mutations require an idempotency key (docs say so).
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["candidate_id", "decision", "idempotency_key"],
        "properties": {
            "candidate_id": {
                "type": "string",
                "description": "The candidate to review.",
                "maxLength": 256,
            },
            "decision": {
                "type": "string",
                "enum": ["approved", "rejected", "quarantined"],
                "description": "The review decision.",
            },
            "reason": {
                "type": "string",
                "description": "Optional reason for the decision.",
                "maxLength": 2000,
            },
            "idempotency_key": {
                "type": "string",
                "description": "Client-generated unique key for idempotency.",
                "minLength": 1,
                "maxLength": 256,
            },
        },
    }


def _memory_review_input_schema() -> Dict[str, Any]:
    """Strict input schema for memory_review (class B, human only).

    Spec-13 S3 (#393): promote (vouch -> clean class, rank penalty
    removed) or dismiss (quarantine + rejection ledger) a memory saved
    under the 'unreviewed' trust class. Human principal only - model
    principals are denied (no self-vouch).
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["memory_id", "decision", "idempotency_key"],
        "properties": {
            "memory_id": {
                "type": "string",
                "description": "The memory to resolve.",
                "maxLength": 256,
            },
            "decision": {
                "type": "string",
                "enum": ["promote", "dismiss"],
                "description": (
                    "'promote' vouches the memory (clean class, rank "
                    "penalty removed); 'dismiss' quarantines it and blocks "
                    "re-assertion via the rejection ledger."
                ),
            },
            "reason": {
                "type": "string",
                "description": "Optional reason for the decision.",
                "maxLength": 2000,
            },
            "idempotency_key": {
                "type": "string",
                "description": "Client-generated unique key for idempotency.",
                "minLength": 1,
                "maxLength": 256,
            },
        },
    }


def _collection_create_input_schema() -> Dict[str, Any]:
    """Strict input schema for collection_create (class C, loopback only)."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["name", "idempotency_key"],
        "properties": {
            "name": {
                "type": "string",
                "description": "Collection name.",
                "maxLength": 500,
            },
            "template": {
                "type": "string",
                "description": "Optional template type (e.g. backlog, reading_list).",
            },
            "schema": {
                "type": "object",
                "description": "Optional field schema (name/type/required per field).",
            },
            "idempotency_key": {
                "type": "string",
                "description": "Client-generated unique key for idempotency.",
                "minLength": 1,
                "maxLength": 256,
            },
        },
    }


def _collection_add_item_input_schema() -> Dict[str, Any]:
    """Strict input schema for collection_add_item (class C, loopback only)."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["collection_id", "fields", "idempotency_key"],
        "properties": {
            "collection_id": {
                "type": "string",
                "description": "The collection to add to.",
                "maxLength": 256,
            },
            "fields": {
                "type": "object",
                "description": "Item fields (validated against collection schema if set).",
            },
            "status": {
                "type": "string",
                "enum": ["open", "done", "parked"],
                "description": "Initial item status (default: open).",
            },
            "idempotency_key": {
                "type": "string",
                "description": "Client-generated unique key for idempotency.",
                "minLength": 1,
                "maxLength": 256,
            },
        },
    }


def _collection_items_input_schema() -> Dict[str, Any]:
    """Strict input schema for collection_items (read, exhaustive)."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["collection_id"],
        "properties": {
            "collection_id": {
                "type": "string",
                "description": "The collection to list items for.",
                "maxLength": 256,
            },
            "status": {
                "type": "string",
                "enum": ["open", "done", "parked"],
                "description": "Optional status filter.",
            },
            "include_archived": {
                "type": "boolean",
                "description": "Include archived items (default: false).",
                "default": False,
            },
        },
    }


def _collection_update_item_input_schema() -> Dict[str, Any]:
    """Strict input schema for collection_update_item (class C, loopback only)."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["item_id", "idempotency_key"],
        "properties": {
            "item_id": {
                "type": "string",
                "description": "The item to update.",
                "maxLength": 256,
            },
            "fields": {
                "type": "object",
                "description": "Updated fields (merged with existing).",
            },
            "status": {
                "type": "string",
                "enum": ["open", "done", "parked"],
                "description": "New item status.",
            },
            "expected_version": {
                "type": "string",
                "description": "CAS: the item's current item_id. Stale → 409.",
            },
            "idempotency_key": {
                "type": "string",
                "description": "Client-generated unique key for idempotency.",
                "minLength": 1,
                "maxLength": 256,
            },
        },
    }


def _collection_remove_item_input_schema() -> Dict[str, Any]:
    """Strict input schema for collection_remove_item (class C, loopback only)."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["item_id", "idempotency_key"],
        "properties": {
            "item_id": {
                "type": "string",
                "description": "The item to remove (archive).",
                "maxLength": 256,
            },
            "expected_version": {
                "type": "string",
                "description": "CAS: the item's current item_id. Stale → 409.",
            },
            "idempotency_key": {
                "type": "string",
                "description": "Client-generated unique key for idempotency.",
                "minLength": 1,
                "maxLength": 256,
            },
        },
    }


def _collection_list_input_schema() -> Dict[str, Any]:
    """Strict input schema for collection_list (read)."""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "status": {
                "type": "string",
                "description": "Optional status filter (e.g. active).",
            },
        },
    }


# Tool name → (facade operation, input schema, description, output schema).
# Deterministic ordering (D6): sorted by tool name.
# M9: tuple (immutable) to prevent accidental mutation across instances.
# #200 Spec-10 PR-3: write tier + collection tools added.
TOOL_DEFINITIONS: tuple = (
    {
        "name": "collection_add_item",
        "description": (
            "Add an item to a collection. Class C write — loopback only. "
            "Requires idempotency key."
        ),
        "inputSchema": _collection_add_item_input_schema(),
        "outputSchema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "status": {"type": "string"},
                "item_id": {"type": "string"},
                "item": {"type": "object"},
            },
        },
    },
    {
        "name": "collection_create",
        "description": (
            "Create a new collection (backlog, reading list, etc.). "
            "Class C write — loopback only. Requires idempotency key."
        ),
        "inputSchema": _collection_create_input_schema(),
        "outputSchema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "status": {"type": "string"},
                "collection_id": {"type": "string"},
                "collection": {"type": "object"},
            },
        },
    },
    {
        "name": "collection_items",
        "description": (
            "List ALL items in a collection (exhaustive — no top-N cutoff). "
            "Read-only. Scope-filtered to the caller's user_id."
        ),
        "inputSchema": _collection_items_input_schema(),
        "outputSchema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "items": {"type": "array", "items": {"type": "object"}},
                "count": {"type": "integer"},
            },
        },
    },
    {
        "name": "collection_list",
        "description": (
            "List collections for the caller's scope. Read-only, exhaustive."
        ),
        "inputSchema": _collection_list_input_schema(),
        "outputSchema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "collections": {"type": "array", "items": {"type": "object"}},
                "count": {"type": "integer"},
            },
        },
    },
    {
        "name": "collection_remove_item",
        "description": (
            "Remove (archive) an item from a collection. Class C write — "
            "loopback only. CAS via expected_version. Requires idempotency key."
        ),
        "inputSchema": _collection_remove_item_input_schema(),
        "outputSchema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "status": {"type": "string"},
                "item_id": {"type": "string"},
                "item": {"type": "object"},
            },
        },
    },
    {
        "name": "collection_update_item",
        "description": (
            "Update an item in a collection. Class C write — loopback only. "
            "CAS via expected_version. Requires idempotency key."
        ),
        "inputSchema": _collection_update_item_input_schema(),
        "outputSchema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "status": {"type": "string"},
                "item_id": {"type": "string"},
                "item": {"type": "object"},
            },
        },
    },
    {
        "name": "memory_candidate_review",
        "description": (
            "Review a pending candidate (approve/reject/quarantine). "
            "Class B — HUMAN principal only. A model principal cannot "
            "approve its own candidate (no self-approval, ever)."
        ),
        "inputSchema": _candidate_review_input_schema(),
        "outputSchema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "candidate_id": {"type": "string"},
                "decision": {"type": "string"},
                "review_reason": {"type": "string"},
                "reviewed_at": {"type": "string"},
                "memory_id": {"type": "string"},
                "reviewer": {"type": "string"},
            },
        },
    },
    {
        "name": "memory_capabilities",
        "description": "List the operations available to the authenticated principal.",
        "inputSchema": _capabilities_input_schema(),
        "outputSchema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "operations": {"type": "array", "items": {"type": "string"}},
                "transport": {"type": "string"},
                "principal": {"type": "string"},
            },
        },
    },
    {
        "name": "memory_explain",
        "description": (
            "Explain why a memory was retrieved — provenance view. "
            "Returns evidence row, version chain, conflict note (if any), "
            "blend score, confidence, and gates fired. Read-only, zero-LLM, "
            "fail-soft. ACL-enforced."
        ),
        "inputSchema": _explain_input_schema(),
        "outputSchema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "memory_id": {"type": "string"},
                "content": {"type": "string"},
                "category": {"type": "string"},
                "evidence": {"type": "object"},
                "version_chain": {"type": "array", "items": {"type": "object"}},
                "conflict_note": {"type": "string"},
                "blend_score": {"type": "object"},
                "confidence": {"type": "number"},
                "provenance_origin": {"type": "string"},
                "grounding": {"type": "string"},
                "gates_fired": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
    {
        "name": "memory_fetch",
        "description": "Fetch a single memory by its ID.",
        "inputSchema": _fetch_input_schema(),
        "outputSchema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "memory_id": {"type": "string"},
                "category": {"type": "string"},
                "content": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "created_at": {"type": "string"},
                "updated_at": {"type": "string"},
                "status": {"type": "string"},
                "scope": {"type": "string"},
            },
        },
    },
    {
        "name": "memory_fetch_history",
        "description": "Fetch the version history for a memory.",
        "inputSchema": _fetch_history_input_schema(),
        "outputSchema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "history": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "memory_id": {"type": "string"},
                            "content": {"type": "string"},
                            "created_at": {"type": "string"},
                            "status": {"type": "string"},
                        },
                    },
                },
                "count": {"type": "integer"},
            },
        },
    },
    {
        "name": "memory_ingest",
        "description": (
            "Structured ingestion (#289): JSON/CSV rows become memories "
            "with first-class provenance. Preview (default) validates and "
            "reports, writing nothing. Apply materializes records via the "
            "candidate/approval machinery (requires the literal "
            "confirm=true) and runs only on loopback transports (class C "
            "trusted-local); non-loopback callers get 403 on apply. An "
            "idempotency key is required."
        ),
        "inputSchema": _ingest_input_schema(),
        "outputSchema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "mode": {"type": "string"},
                "source": {"type": "string"},
                "mapping_id": {"type": "string"},
                "total_rows": {"type": "integer"},
                "valid_rows": {"type": "integer"},
                "error_rows": {"type": "integer"},
                "inserted": {"type": "integer"},
                "superseded": {"type": "integer"},
                "duplicates": {"type": "integer"},
                "quarantined": {"type": "integer"},
                "blocked": {"type": "integer"},
                "rows": {"type": "array", "items": {"type": "object"}},
                "errors": {"type": "array", "items": {"type": "object"}},
                "wrote": {"type": "boolean"},
            },
        },
    },
    {
        "name": "memory_propose",
        "description": (
            "Propose a new memory for human review. The candidate enters "
            "the review queue — it does NOT become active memory until a "
            "human approves it. An idempotency key is required: retrying "
            "with the same key and body returns the original result."
        ),
        "inputSchema": _propose_input_schema(),
        "outputSchema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "candidate_id": {"type": "string"},
                "status": {
                    "type": "string",
                    "enum": ["pending", "quarantined", "error"],
                },
                "reason": {"type": "string"},
                "scan_summary": {"type": "string"},
            },
        },
    },
    {
        "name": "memory_review",
        "description": (
            "Promote (vouch -> clean class) or dismiss (quarantine + "
            "rejection ledger) a memory saved under the 'unreviewed' "
            "trust class (Spec-13 resolution). Class B — HUMAN principal "
            "only. A model principal cannot vouch its own memory."
        ),
        "inputSchema": _memory_review_input_schema(),
        "outputSchema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "memory_id": {"type": "string"},
                "decision": {"type": "string"},
                "changed": {"type": "boolean"},
                "previous_trust_class": {"type": ["string", "null"]},
                "previous_status": {"type": ["string", "null"]},
                "reassertion_blocked": {"type": ["boolean", "null"]},
                "reviewer": {"type": "string"},
            },
        },
    },
    {
        "name": "memory_save",
        "description": (
            "Save a fact directly to active memory (class C write — "
            "loopback only). Same provider-level semantics as the native "
            "memory_save tool. Requires idempotency key."
        ),
        "inputSchema": _save_input_schema(),
        "outputSchema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "status": {"type": "string"},
                "memory_id": {"type": "string"},
            },
        },
    },
    {
        "name": "memory_search",
        "description": "Search memories by natural-language query.",
        "inputSchema": _search_input_schema(),
        "outputSchema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "memory_id": {"type": "string"},
                            "category": {"type": "string"},
                            "content": {"type": "string"},
                            "tags": {"type": "array", "items": {"type": "string"}},
                            "similarity": {"type": "number"},
                            "created_at": {"type": "string"},
                            "updated_at": {"type": "string"},
                            "status": {"type": "string"},
                            "scope": {"type": "string"},
                        },
                    },
                },
                "count": {"type": "integer"},
            },
        },
    },
    {
        "name": "memory_unreviewed",
        "description": "Report the live unreviewed trust-class backlog: count and oldest age. Read-only.",
        "inputSchema": _unreviewed_input_schema(),
        "outputSchema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "count": {"type": "integer"},
                "oldest_created_at": {"type": ["string", "null"]},
                "oldest_age_days": {"type": ["number", "null"]},
                "scope": {"type": "string"},
            },
        },
    },
    {
        "name": "memory_update",
        "description": (
            "Update an existing memory, creating a new version (class C "
            "write — loopback only). CAS via expected_version. Requires "
            "idempotency key."
        ),
        "inputSchema": _update_input_schema(),
        "outputSchema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "status": {"type": "string"},
                "memory_id": {"type": "string"},
                "old_memory_id": {"type": "string"},
            },
        },
    },
    {
        "name": "memory_why_not",
        "description": (
            "Diagnose why a memory did NOT surface in retrieval — rank, "
            "per-stage diagnostics (vector/text/status/scope), and "
            "human-readable reasons. Deterministic, read-only, zero-LLM. "
            "ACL-enforced."
        ),
        "inputSchema": _explain_retrieval_input_schema(),
        "outputSchema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "expected_memory_id": {"type": "string"},
                "expected": {"type": "object"},
                "found_in_results": {"type": "boolean"},
                "rank": {
                    "anyOf": [{"type": "integer"}, {"type": "null"}],
                },
                "top_results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "memory_id": {"type": "string"},
                            "content": {"type": "string"},
                            "category": {"type": "string"},
                            "similarity": {"type": "number"},
                            "raw_similarity": {"type": "number"},
                        },
                    },
                },
                "reasons": {"type": "array", "items": {"type": "string"}},
                "diagnostics": {"type": "object"},
            },
        },
    },

)

# Map MCP tool names to facade operations.
TOOL_TO_OPERATION: Dict[str, str] = {
    "memory_search": "search",
    "memory_fetch": "fetch",
    "memory_fetch_history": "fetch_history",
    "memory_explain": "explain",
    "memory_why_not": "explain_retrieval",
    "memory_capabilities": "capabilities",
    "memory_unreviewed": "unreviewed",
    "memory_propose": "memory_propose",
    "memory_ingest": "ingest",
    # #200 Spec-10 PR-3: write tier + collection tools.
    "memory_save": "memory_save",
    "memory_update": "memory_update",
    "memory_candidate_review": "review_candidate",
    "memory_review": "review_memory",
    "collection_list": "collection_list",
    "collection_items": "collection_items",
    "collection_create": "collection_create",
    "collection_add_item": "collection_add_item",
    "collection_update_item": "collection_update_item",
    "collection_remove_item": "collection_remove_item",
}

# Tools that require an idempotency_key (popped from arguments and passed
# as a separate keyword to facade.execute).
TOOLS_WITH_IDEMPOTENCY_KEY: frozenset = frozenset({
    "memory_propose",
    "memory_ingest",
    "memory_save",
    "memory_update",
    "memory_candidate_review",
    "memory_review",
    "collection_create",
    "collection_add_item",
    "collection_update_item",
    "collection_remove_item",
})


# -- JSON-RPC message helpers ------------------------------------------------

def _make_response(
    request_id: Any,
    result: Any = None,
    error: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build a JSON-RPC 2.0 response."""
    msg: Dict[str, Any] = {"jsonrpc": "2.0", "id": request_id}
    if error is not None:
        msg["error"] = error
    else:
        msg["result"] = result
    return msg


def _make_error(code: int, message: str, data: Any = None) -> Dict[str, Any]:
    """Build a JSON-RPC 2.0 error object.

    M10: *data* is sent verbatim to the client — it must be client-safe
    (no stack traces, internal IDs, file paths, or SQL). The facade
    redacts errors, but callers must not pass unredacted data here.
    """
    err: Dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return err


def _make_notification(method: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Build a JSON-RPC 2.0 notification (no id, no response expected)."""
    msg: Dict[str, Any] = {"jsonrpc": "2.0", "method": method}
    if params:
        msg["params"] = params
    return msg


# -- MCP server --------------------------------------------------------------

class MCPServer:
    """MCP stdio server exposing the Argos read tier through the facade.

    The server reads JSON-RPC messages from stdin (one per line), processes
    them, and writes responses to stdout (one per line). Logs go to stderr.

    The server is a transport adapter only — all business logic (auth,
    ACL, validation, idempotency, audit) is handled by the facade.
    """

    def __init__(
        self,
        facade,
        auth_context,
        *,
        stdin=None,
        stdout=None,
        stderr=None,
    ) -> None:
        """Initialize the MCP server.

        Args:
            facade: an ArgosAPIFacade instance.
            auth_context: an AuthContext for the authenticated principal.
                The transport derives this from the credential/environment
                before constructing the server.
            stdin: input stream (defaults to sys.stdin).
            stdout: output stream (defaults to sys.stdout).
            stderr: log stream (defaults to sys.stderr).
        """
        from api_facade import ArgosAPIFacade, AuthContext  # noqa: F401
        self._facade = facade
        self._auth = auth_context
        self._stdin = stdin or sys.stdin
        self._stdout = stdout or sys.stdout
        self._stderr = stderr or sys.stderr
        self._initialized = False

    def run(self) -> None:
        """Main loop: read lines from stdin, process, write to stdout.

        Exits when stdin is closed (EOF) or a fatal error occurs.
        M5: KeyboardInterrupt/SystemExit are caught and logged so the
        server shuts down gracefully rather than dying mid-message.
        """
        for line in self._stdin:
            line = line.strip()
            if not line:
                continue
            if len(line.encode("utf-8")) > MAX_MESSAGE_BYTES:
                self._send(_make_response(
                    None, error=_make_error(
                        JSONRPC_INVALID_REQUEST,
                        "Message exceeds maximum size.",
                    ),
                ))
                continue
            try:
                self._handle_line(line)
            except (KeyboardInterrupt, SystemExit):
                # M5: graceful shutdown on signals — log and re-exit.
                logger.info("MCP server shutting down (signal received).")
                raise
            except BrokenPipeError:
                # M6: stdout closed — client disconnected. Exit gracefully.
                logger.info("MCP server: stdout closed (client disconnected).")
                break
            except Exception as exc:
                # Fatal error — log to stderr, send error to stdout, continue.
                logger.error("MCP server error: %s", exc, exc_info=True)
                self._send(_make_response(
                    None, error=_make_error(
                        JSONRPC_INTERNAL_ERROR,
                        "Internal server error.",
                    ),
                ))

    def _handle_line(self, line: str) -> None:
        """Parse and handle one JSON-RPC message line."""
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            self._send(_make_response(
                None, error=_make_error(
                    JSONRPC_PARSE_ERROR, "Invalid JSON.",
                ),
            ))
            return
        if not isinstance(msg, dict):
            self._send(_make_response(
                None, error=_make_error(
                    JSONRPC_INVALID_REQUEST, "Request must be a JSON object.",
                ),
            ))
            return
        msg_id = msg.get("id")
        method = msg.get("method")
        params = msg.get("params") or {}
        if not isinstance(method, str):
            # Notifications (no method) are silently ignored.
            if msg_id is not None:
                self._send(_make_response(
                    msg_id, error=_make_error(
                        JSONRPC_INVALID_REQUEST, "Missing method.",
                    ),
                ))
            return
        # Route to handler.
        if method == "initialize":
            self._handle_initialize(msg_id, params)
        elif method == "notifications/initialized":
            self._initialized = True
            # No response for notifications.
        elif method == "tools/list":
            # M8: tools should not be listed until after notifications/initialized.
            if not self._initialized:
                self._send(_make_response(
                    msg_id, error=_make_error(
                        JSONRPC_INVALID_REQUEST,
                        "Server not initialized — send notifications/initialized first.",
                    ),
                ))
            else:
                self._handle_tools_list(msg_id)
        elif method == "tools/call":
            if not self._initialized:
                self._send(_make_response(
                    msg_id, error=_make_error(
                        JSONRPC_INVALID_REQUEST,
                        "Server not initialized — send notifications/initialized first.",
                    ),
                ))
            else:
                self._handle_tools_call(msg_id, params)
        elif method == "ping":
            self._send(_make_response(msg_id, result={}))
        elif method == "resources/list":
            if not self._initialized:
                self._send(_make_response(
                    msg_id, error=_make_error(
                        JSONRPC_INVALID_REQUEST,
                        "Server not initialized — send notifications/initialized first.",
                    ),
                ))
            else:
                self._handle_resources_list(msg_id)
        elif method == "resources/read":
            if not self._initialized:
                self._send(_make_response(
                    msg_id, error=_make_error(
                        JSONRPC_INVALID_REQUEST,
                        "Server not initialized — send notifications/initialized first.",
                    ),
                ))
            else:
                self._handle_resources_read(msg_id, params)
        elif method == "prompts/list":
            if not self._initialized:
                self._send(_make_response(
                    msg_id, error=_make_error(
                        JSONRPC_INVALID_REQUEST,
                        "Server not initialized — send notifications/initialized first.",
                    ),
                ))
            else:
                self._handle_prompts_list(msg_id)
        elif method == "prompts/get":
            if not self._initialized:
                self._send(_make_response(
                    msg_id, error=_make_error(
                        JSONRPC_INVALID_REQUEST,
                        "Server not initialized — send notifications/initialized first.",
                    ),
                ))
            else:
                self._handle_prompts_get(msg_id, params)
        else:
            self._send(_make_response(
                msg_id, error=_make_error(
                    JSONRPC_METHOD_NOT_FOUND,
                    f"Unknown method: {method}",
                ),
            ))

    def _handle_initialize(self, msg_id: Any, params: Dict[str, Any]) -> None:
        """Handle the initialize request (capability negotiation).

        M4: logs a warning if the client's protocol version is older than
        the server's minimum supported version.
        """
        client_version = params.get("protocolVersion", "")
        # M4: warn on incompatible versions (server still responds with its own).
        if client_version and client_version != MCP_PROTOCOL_VERSION:
            logger.warning(
                "MCP client requested protocol %s; server supports %s. "
                "Responding with server version (per spec).",
                client_version, MCP_PROTOCOL_VERSION,
            )
        # Respond with our protocol version. If the client requested a
        # different version, we respond with ours (per spec: server
        # responds with a version it supports).
        result = {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {
                "tools": {
                    "listChanged": False,
                },
                # #389: resource + prompt surfaces (read-only, gated by
                # the same facade allowlist).
                "resources": {"listChanged": False},
                "prompts": {},
            },
            "serverInfo": {
                "name": "argos-memory",
                "title": "Argos Memory Service",
                "version": "1.0.0",
            },
            "instructions": (
                "Argos memory service. Use memory_search to find memories, "
                "memory_fetch to get one by ID, memory_fetch_history for "
                "version history, memory_propose to submit a fact for human "
                "review (requires an idempotency key), memory_save to write "
                "directly to active memory (loopback only), memory_update to "
                "update a memory with version chaining, "
                "memory_candidate_review to approve/reject candidates "
                "(human only), memory_review to promote/dismiss unreviewed "
                "memories (human only), collection_list/collection_items to list "
                "collections and their items, and collection_create/"
                "collection_add_item/collection_update_item/"
                "collection_remove_item to manage collections (loopback "
                "only). Use memory_capabilities to list available operations."
            ),
        }
        self._send(_make_response(msg_id, result=result))

    def _handle_tools_list(self, msg_id: Any) -> None:
        """Handle tools/list — return available tool definitions."""
        # Only return tools the principal is authorized for.
        allowed_ops = self._auth.allowed_operations
        tools = []
        for tool_def in TOOL_DEFINITIONS:
            op = TOOL_TO_OPERATION.get(tool_def["name"], "")
            if op in allowed_ops:
                tools.append(tool_def)
        self._send(_make_response(msg_id, result={"tools": tools}))

    # -- #389: resources/list + resources/read + prompts/list + prompts/get --

    def _handle_resources_list(self, msg_id: Any) -> None:
        """Return resource descriptors (#389). Resources are gated by the
        same facade allowlist as tools — a principal only sees resources
        backed by operations it may call."""
        allowed_ops = self._auth.allowed_operations
        resources = []
        if "browse" in allowed_ops:
            resources.append({
                "uri": "memory://stats",
                "name": "Memory stats",
                "mimeType": "application/json",
                "description": "Live count of the caller's memories (scope-filtered).",
            })
            resources.append({
                "uri": "memory://stats/categories",
                "name": "Memory by category",
                "mimeType": "application/json",
                "description": "Memory counts grouped by category (scope-filtered).",
            })
        self._send(_make_response(msg_id, result={"resources": resources}))

    def _handle_resources_read(self, msg_id: Any, params: Dict[str, Any]) -> None:
        """Read a resource URI. Only resources backed by allowed facade
        ops are served; everything else is JSONRPC_INVALID_PARAMS."""
        uri = str(params.get("uri", ""))
        allowed_ops = self._auth.allowed_operations
        if "browse" not in allowed_ops:
            self._send(_make_response(
                msg_id, error=_make_error(JSONRPC_INVALID_PARAMS, "Resource not available."),
            ))
            return
        try:
            if uri == "memory://stats":
                res = self._facade.execute(
                    self._auth, "browse", {"limit": 50},
                )
                content = {
                    "memory_count": res.get("count", 0),
                    "scope": getattr(self._auth, "user_id", ""),
                }
            elif uri == "memory://stats/categories":
                content = {"categories": "use memory_search for category breakdown"}
            else:
                self._send(_make_response(
                    msg_id, error=_make_error(JSONRPC_INVALID_PARAMS, f"Unknown resource: {uri}"),
                ))
                return
        except Exception as exc:
            self._send(_make_response(
                msg_id, error=_make_error(JSONRPC_INTERNAL_ERROR, str(exc)),
            ))
            return
        self._send(_make_response(msg_id, result={
            "contents": [{
                "uri": uri,
                "mimeType": "application/json",
                "text": json.dumps(content),
            }],
        }))

    def _handle_prompts_list(self, msg_id: Any) -> None:
        """Return the prompt catalog (#389): read-only templates that
        guide a client how to use the memory surface."""
        self._send(_make_response(msg_id, result={"prompts": [
            {
                "name": "memory_search_usage",
                "description": "How to retrieve a specific memory with memory_search.",
                "arguments": [],
            },
            {
                "name": "review_queue_usage",
                "description": "How to list and decide on pending candidates.",
                "arguments": [],
            },
        ]}))

    def _handle_prompts_get(self, msg_id: Any, params: Dict[str, Any]) -> None:
        """Return the text for a known prompt (static catalog)."""
        name = str(params.get("name", ""))
        catalog = {
            "memory_search_usage": (
                "Search memories with memory_search (query, optional limit 1-50, "
                "optional category_filter). Results are scoped to your identity."
            ),
            "review_queue_usage": (
                "List pending candidates with the candidates endpoint; approve or "
                "reject with memory_candidate_review (human principal only, "
                "idempotency_key required)."
            ),
        }
        if name not in catalog:
            self._send(_make_response(
                msg_id, error=_make_error(JSONRPC_INVALID_PARAMS, f"Unknown prompt: {name}"),
            ))
            return
        self._send(_make_response(msg_id, result={
            "description": "Prompt template",
            "messages": [{"role": "user", "content": {"type": "text", "text": catalog[name]}}],
        }))

    def _handle_tools_call(self, msg_id: Any, params: Dict[str, Any]) -> None:
        """Handle tools/call — invoke a tool through the facade."""
        tool_name = params.get("name", "")
        arguments = dict(params.get("arguments") or {})  # copy — don't mutate caller's
        # Map tool name to facade operation.
        operation = TOOL_TO_OPERATION.get(tool_name)
        if operation is None:
            self._send(_make_response(
                msg_id, error=_make_error(
                    JSONRPC_METHOD_NOT_FOUND,
                    f"Unknown tool: {tool_name}",
                ),
            ))
            return
        # M2: validate arguments against the tool's inputSchema before
        # calling the facade. The MCP spec requires the server to validate
        # against the declared schema.
        tool_def = None
        for td in TOOL_DEFINITIONS:
            if td["name"] == tool_name:
                tool_def = td
                break
        if tool_def is not None:
            schema = tool_def.get("inputSchema")
            if schema is not None:
                try:
                    import jsonschema
                    jsonschema.validate(instance=arguments, schema=schema)
                except Exception as exc:
                    # jsonschema.ValidationError -> invalid params;
                    # ImportError/other -> jsonschema unavailable, fall
                    # through to facade validation (fail-open, not
                    # fail-closed, since the facade does its own
                    # validation).
                    if (
                        isinstance(exc, ImportError)
                        or not hasattr(exc, "message")
                    ):
                        pass
                    else:
                        self._send(_make_response(
                            msg_id, error=_make_error(
                                JSONRPC_INVALID_PARAMS,
                                f"Invalid arguments: {exc.message}",
                            ),
                        ))
                        return
        # M1: pop idempotency_key for tools that require it. The key is
        # passed as a keyword arg to facade.execute, not in the params
        # dict. Schema validation (additionalProperties: false) ensures
        # only tools with idempotency_key in their schema accept it.
        # #200 PR-3: extended to write tier + collection tools.
        idempotency_key = None
        if tool_name in TOOLS_WITH_IDEMPOTENCY_KEY:
            idempotency_key = arguments.pop("idempotency_key", None)
        # Call the facade. The facade handles validation, auth, ACL,
        # idempotency, audit, and error redaction.
        try:
            result = self._facade.execute(
                self._auth, operation, arguments,
                idempotency_key=idempotency_key,
            )
            # MCP tools/call returns a CallToolResult with content array.
            # For structured results, we use the content as JSON.
            self._send(_make_response(msg_id, result={
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(result, ensure_ascii=False),
                    },
                ],
                "structuredContent": result,
                "isError": False,
            }))
        except Exception as exc:
            # The facade raises APIError with stable codes. We map those
            # to MCP error responses.
            from api_facade import APIError
            if isinstance(exc, APIError):
                self._send(_make_response(msg_id, result={
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(exc.to_dict(), ensure_ascii=False),
                        },
                    ],
                    "structuredContent": exc.to_dict(),
                    "isError": True,
                }))
            else:
                # Should not happen — the facade catches all exceptions.
                logger.error("Unhandled error in tools/call: %s", exc, exc_info=True)
                self._send(_make_response(msg_id, result={
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps({
                                "error": {
                                    "code": "internal_error",
                                    "message": "An internal error occurred.",
                                },
                            }, ensure_ascii=False),
                        },
                    ],
                    "isError": True,
                }))

    def _send(self, msg: Dict[str, Any]) -> None:
        """Write one JSON-RPC message to stdout (newline-delimited).

        M6: BrokenPipeError (client disconnected) is caught and logged
        rather than propagating — the run loop handles the exit.
        """
        data = json.dumps(msg, ensure_ascii=False) + "\n"
        try:
            self._stdout.write(data)
            self._stdout.flush()
        except (BrokenPipeError, OSError) as exc:
            logger.info("MCP server: write failed (%s) — client may be gone.", exc)


# -- Entry point -------------------------------------------------------------

def _load_credential_context(home: Path, cred_name: str) -> "AuthContext":
    """Resolve a named per-principal credential (#387, Spec-12).

    The spawner names the credential (ARGOS_API_CREDENTIAL); the file
    holds the identity (principal, tenant, user_id, principal_type,
    operation classes). Fail-closed: a missing, expired, or invalid
    credential refuses to start - never a silent fallback to
    env-derived identity.

    Stdio threat model: the spawner still *selects* which credential
    this process uses - stdio has no per-request auth surface. What the
    credential file adds over raw env vars: scopes and revocation/expiry
    live in an operator-controlled file, the principal is named and
    auditable, and class-B unlock (human) is an explicit file value
    rather than an ambient env typo.
    """
    from api_credentials import (
        CredentialFileError,
        build_context,
        credential_file_path,
        parse_credentials_file,
        resolve_by_name,
    )

    override = os.environ.get("ARGOS_API_CREDENTIAL_FILE", "").strip()
    path = Path(override) if override else credential_file_path(home)
    try:
        _, credentials = parse_credentials_file(path)
    except CredentialFileError as exc:
        raise SystemExit(
            f"ARGOS_API_CREDENTIAL={cred_name!r} is set but {path} is "
            f"invalid: {exc} - refusing to start (fail-closed)."
        )
    cred = resolve_by_name(credentials, cred_name)
    if cred is None:
        raise SystemExit(
            f"ARGOS_API_CREDENTIAL={cred_name!r} not found in {path} - "
            "refusing to start (fail-closed)."
        )
    if cred.is_expired():
        raise SystemExit(
            f"ARGOS_API_CREDENTIAL={cred_name!r} is expired in {path} - "
            "refusing to start (fail-closed)."
        )
    is_loopback = os.environ.get("ARGOS_API_NO_LOOPBACK", "").lower() not in ("true", "1", "yes")
    is_read_only = os.environ.get("ARGOS_API_READ_ONLY", "").lower() in ("true", "1", "yes")
    return build_context(
        cred,
        transport="mcp-stdio",
        is_loopback=is_loopback,
        env_principal_type=os.environ.get("ARGOS_API_PRINCIPAL_TYPE", ""),
        is_read_only=is_read_only,
    )


def _load_auth_context(home: Path) -> "AuthContext":
    """Derive the auth context from the environment.

    For v1 (trusted-local mode), the principal is derived from the
    ARGOS_API_PRINCIPAL env var (default: "local"), the tenant from
    ARGOS_API_TENANT (default: "default"), and the user_id from
    ARGOS_API_USER_ID (default: "default_user").

    #200 Spec-10 PR-3: principal_type and is_loopback are wired here.
    - principal_type: "model" (default) or "human" (ARGOS_API_PRINCIPAL_TYPE).
      A model principal is denied class B (candidate approval) — no
      self-approval, ever. The default is "model" (fail-closed): a
      transport that forgets to set principal_type is treated as a model
      agent and CANNOT approve candidates. A human-driven UI must
      explicitly set ARGOS_API_PRINCIPAL_TYPE=human to unlock class B.
    - is_loopback: True for MCP stdio (the process is spawned locally by
      the user's shell — it's a trusted-local transport). This enables
      class C writes (memory_save, memory_update, collection writes).
      Set ARGOS_API_NO_LOOPBACK=1 to disable (for testing non-loopback
      denial).

    M3 — Threat model (trusted-local mode):
    Identity is derived from environment variables with NO credential
    verification. Any process that can set env vars can impersonate any
    user. This is acceptable ONLY in trusted-local mode (single-user
    workstation, MCP client spawned by the user's own shell). In a
    multi-process or hosted environment, the env vars are controlled by
    the spawner, not the user — a malicious spawner can impersonate
    anyone. For non-trusted-local deployments, set ARGOS_API_CREDENTIAL
    to a named credential in {home}/api_credential.json (#387,
    Spec-12): the file supplies principal, tenant, user_id,
    principal_type, and operation classes, so scopes, expiry, and
    revocation live in an operator-controlled file instead of
    ambient env vars.
    """
    from api_facade import (
        AuthContext, READ_OPERATIONS, PROPOSAL_OPERATIONS,
        FEEDBACK_OPERATIONS, WRITE_OPERATIONS,
        COLLECTION_READ_OPERATIONS, COLLECTION_WRITE_OPERATIONS,
    )

    # #387 (Spec-12): credential-backed identity. When
    # ARGOS_API_CREDENTIAL names a credential, identity comes from
    # {home}/api_credential.json (or ARGOS_API_CREDENTIAL_FILE) and
    # the spawner-asserted env identity below is ignored for the
    # principal/tenant/user_id/classes. Fail-closed: an unknown,
    # expired, or invalid credential refuses to start.
    cred_name = os.environ.get("ARGOS_API_CREDENTIAL", "").strip()
    if cred_name:
        return _load_credential_context(home, cred_name)

    principal = os.environ.get("ARGOS_API_PRINCIPAL", "local")
    tenant = os.environ.get("ARGOS_API_TENANT", "default")
    user_id = os.environ.get("ARGOS_API_USER_ID", "default_user")
    # #200 PR-3 fix: wire principal_type — "model" (default) or "human".
    # A model principal is denied class B (candidate approval). The
    # default is "model" (fail-closed): a transport that forgets to set
    # ARGOS_API_PRINCIPAL_TYPE is treated as a model agent and CANNOT
    # approve candidates. A human-driven UI MUST explicitly set
    # ARGOS_API_PRINCIPAL_TYPE=human to unlock class B. This closes the
    # self-approval spoof: a model agent wired with defaults
    # (ARGOS_API_CAN_PROPOSE=1, principal_type unset) is denied class B.
    principal_type = os.environ.get("ARGOS_API_PRINCIPAL_TYPE", "model")
    if principal_type not in ("human", "model"):
        principal_type = "model"  # fail-closed: unknown → model (denied class B)
    # #200 PR-3: MCP stdio is a loopback transport (local process).
    # Class C writes require loopback. Set ARGOS_API_NO_LOOPBACK=1 to
    # test non-loopback denial.
    is_loopback = os.environ.get("ARGOS_API_NO_LOOPBACK", "").lower() not in ("true", "1", "yes")

    # Spec-11 (9/9): write tiers ON by default on loopback transports.
    # Class A (propose) and Class C (direct write) are default-ON so
    # external MCP clients (OpenWebUI, Claude Desktop, Cursor, etc.) get
    # a working read+write surface with no env-var discovery required.
    # Class B (feedback/approval) stays OFF — model self-approval, never.
    # ARGOS_API_READ_ONLY=1 restores the spec-09 read-only default for
    # conservative or shared deployments.
    is_read_only = os.environ.get("ARGOS_API_READ_ONLY", "").lower() in ("true", "1", "yes")

    allowed = set(READ_OPERATIONS) | COLLECTION_READ_OPERATIONS
    if not is_read_only:
        allowed |= PROPOSAL_OPERATIONS
        if is_loopback:
            allowed |= WRITE_OPERATIONS
            allowed |= COLLECTION_WRITE_OPERATIONS
    # Explicit env vars still work as overrides (belt-and-suspenders for
    # deployments that set them intentionally).
    if os.environ.get("ARGOS_API_CAN_PROPOSE", "").lower() in ("true", "1", "yes"):
        allowed |= PROPOSAL_OPERATIONS
    if os.environ.get("ARGOS_API_CAN_FEEDBACK", "").lower() in ("true", "1", "yes"):
        allowed |= FEEDBACK_OPERATIONS
    if is_loopback and os.environ.get("ARGOS_API_CAN_WRITE", "").lower() in ("true", "1", "yes"):
        allowed |= WRITE_OPERATIONS
        allowed |= COLLECTION_WRITE_OPERATIONS

    return AuthContext(
        principal=principal,
        tenant=tenant,
        user_id=user_id,
        transport="mcp-stdio",
        allowed_operations=allowed,
        can_propose="memory_propose" in allowed,
        can_feedback="record_feedback" in allowed,
        principal_type=principal_type,
        is_loopback=is_loopback,
    )


def _force_utf8_stdio() -> None:
    """Force UTF-8 on stdout/stderr.

    Tool descriptions and memory content carry non-ASCII characters
    (e.g. the arrow in "Same key + different body -> 409 conflict").
    On Windows the default stdio encoding is cp1252, which cannot
    encode those characters and makes tools/list (and any non-ASCII
    memory content) crash with UnicodeEncodeError. Reconfigure the
    streams to UTF-8 so the server is correct regardless of the
    client's env config (PYTHONIOENCODING / PYTHONUTF8).

    reconfigure() is available on Python 3.7+ for the default
    TextIOWrapper streams. If reconfigure is unavailable (non-standard
    stream), fall back to reassigning a UTF-8 wrapper.
    """
    for stream in (sys.stdout, sys.stderr):
        if not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            # Stream may be closed or not reconfigurable — skip.
            pass


def main() -> None:
    """Entry point for the MCP stdio server.

    Configured HERMES_HOME — never an arbitrary caller-selected data path.
    """
    import argparse
    from api_facade import ArgosAPIFacade, ACLConfig
    from service_client import SharedMemoryStore

    # Force UTF-8 stdio before any output (Windows defaults to cp1252,
    # which cannot encode the non-ASCII characters in tool descriptions
    # and memory content — tools/list would crash with no tools listed).
    _force_utf8_stdio()

    parser = argparse.ArgumentParser(description="Argos MCP stdio server (read + write tier)")
    parser.add_argument("--home", required=True, type=Path,
                        help="Path to the Hermes home directory.")
    args = parser.parse_args()

    # Logs to stderr only (D6: no banners on stdout).
    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    # Build the store, facade, and auth context.
    # #386 (Spec-12): verified 2026-09-12 — no client-side embedder is
    # involved in shared mode. SharedMemoryStore proxies every retrieval
    # stage (embedding, reranker/blend, chains) to the memory service
    # over RPC, so MCP search already runs the full vector-backed
    # pipeline; results are byte-identical to the native path for the
    # same store/user (pinned by tests/test_spec12_parity.py). The
    # embedder argument is kept for DuckDBMemoryStore signature
    # compatibility and is unused here. (The former M7 comment claimed a
    # text-only degradation — that predated the shared-service
    # architecture and is obsolete.)
    store = SharedMemoryStore(args.home, user_id="default_user", embedder=None)
    acl = ACLConfig()  # v1: open store (trusted-local mode)
    facade = ArgosAPIFacade(store, acl=acl, api_mode=False)
    auth_ctx = _load_auth_context(args.home)

    server = MCPServer(facade, auth_ctx)
    server.run()


if __name__ == "__main__":
    main()
