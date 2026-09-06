"""#200 Spec-10 PR 2/3: Collections store mixin — exhaustive, structural, no ranking.

Collection CRUD methods for DuckDBMemoryStore. These are plain filtered SQL
queries — no embeddings, no ranking, no top-N cutoff. Every matching item
is returned, every time.

Isolation:
  - user_scope: same pattern as memory_records — ``AND (user_scope IS NULL
    OR user_scope = ?)`` with ``self.user_id`` (set by set_user_scope).
  - tenant: each tenant has its own DuckDB file (Cells #131), so tenant
    isolation is filesystem-level. The tenant column is for audit/debug.

ID minting:
  - All IDs are server-minted (uuid4 hex). Client-supplied IDs are never
    used.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Valid item statuses (v1 — configurable later).
_VALID_ITEM_STATUSES = frozenset({"open", "done", "parked"})
_VALID_COLLECTION_STATUSES = frozenset({"active", "archived"})


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _mint_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:16]}"


def _validate_fields_against_schema(
    fields: Dict[str, Any], schema: Dict[str, Any] | None,
) -> None:
    """Minimal schema validation (v1): name/type/required check.

    schema is a list of field defs: [{"name": "title", "type": "string",
    "required": true}, ...]. If schema is None or empty, no validation
    runs (free-form JSON).
    """
    if not schema:
        return
    if not isinstance(schema, list):
        raise ValueError("schema must be a list of field definitions")
    for field_def in schema:
        name = field_def.get("name")
        if not name:
            raise ValueError("schema field def missing 'name'")
        expected_type = field_def.get("type")
        required = field_def.get("required", False)
        value = fields.get(name)
        if required and value is None:
            raise ValueError(f"field {name!r} is required")
        if value is not None and expected_type:
            _check_type(name, value, expected_type)


def _check_type(name: str, value: Any, expected_type: str) -> None:
    """Check a value against a minimal type string."""
    type_map = {
        "string": str,
        "text": str,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    expected = type_map.get(expected_type)
    if expected is None:
        return  # unknown type — don't validate (v1 minimal)
    if not isinstance(value, expected):
        raise ValueError(
            f"field {name!r} expected type {expected_type!r}, "
            f"got {type(value).__name__}"
        )


class StoreCollectionsMixin:
    """Collection CRUD methods for DuckDBMemoryStore.

    All methods assume ``self.connection`` and ``self.user_id`` are set
    (via set_user_scope). All SQL uses the standard scope filter:
    ``AND (user_scope IS NULL OR user_scope = ?)``.
    """

    # -- Reads (exhaustive, no ranking) ----------------------------------

    def list_collections(
        self,
        *,
        status: str | None = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """List collections for the current user_scope. EXHAUSTIVE (no
        top-N cutoff beyond the safety limit). Filter by status only."""
        with self._state.lock:
            assert self.connection is not None
            sql = (
                "SELECT collection_id, name, template, schema, status, "
                "       user_scope, tenant, created_at, updated_at "
                "FROM collections "
                "WHERE (user_scope IS NULL OR user_scope = ?)"
            )
            params: list = [self.user_id]
            if status:
                sql += " AND status = ?"
                params.append(status)
            sql += " ORDER BY created_at DESC"
            if limit and limit > 0:
                sql += f" LIMIT {int(limit)}"
            rows = self.connection.execute(sql, params).fetchall()
        result = []
        for row in rows:
            result.append({
                "collection_id": row[0],
                "name": row[1],
                "template": row[2],
                "schema": json.loads(row[3]) if row[3] else None,
                "status": row[4],
                "user_scope": row[5],
                "tenant": row[6],
                "created_at": row[7],
                "updated_at": row[8],
            })
        return result

    def list_collection_items(
        self,
        *,
        collection_id: str,
        status: str | None = None,
        include_archived: bool = False,
        limit: int = 0,
    ) -> List[Dict[str, Any]]:
        """List items in a collection. EXHAUSTIVE — every matching item,
        no top-N cutoff (limit=0 means no limit). Filter by status/scope
        only. No ranking, no similarity."""
        with self._state.lock:
            assert self.connection is not None
            sql = (
                "SELECT item_id, collection_id, fields, status, "
                "       user_scope, tenant, created_at, updated_at, archived_at "
                "FROM collection_items "
                "WHERE collection_id = ? "
                "  AND (user_scope IS NULL OR user_scope = ?)"
            )
            params: list = [collection_id, self.user_id]
            if status:
                sql += " AND status = ?"
                params.append(status)
            if not include_archived:
                sql += " AND archived_at IS NULL"
            sql += " ORDER BY created_at ASC"
            if limit and limit > 0:
                sql += f" LIMIT {int(limit)}"
            rows = self.connection.execute(sql, params).fetchall()
        result = []
        for row in rows:
            result.append({
                "item_id": row[0],
                "collection_id": row[1],
                "fields": json.loads(row[2]) if row[2] else {},
                "status": row[3],
                "user_scope": row[4],
                "tenant": row[5],
                "created_at": row[6],
                "updated_at": row[7],
                "archived_at": row[8],
            })
        return result

    def count_collection_items(
        self,
        *,
        collection_id: str,
        status: str | None = None,
    ) -> int:
        """Count items in a collection (scope-filtered). For isolation
        checks — tenant B's count of tenant A's collection is 0."""
        with self._state.lock:
            assert self.connection is not None
            sql = (
                "SELECT COUNT(*) FROM collection_items "
                "WHERE collection_id = ? "
                "  AND (user_scope IS NULL OR user_scope = ?)"
                "  AND archived_at IS NULL"
            )
            params: list = [collection_id, self.user_id]
            if status:
                sql += " AND status = ?"
                params.append(status)
            result = self.connection.execute(sql, params).fetchone()
        return int(result[0]) if result else 0

    def get_collection(self, collection_id: str) -> Dict[str, Any] | None:
        """Get a single collection by ID (scope-filtered)."""
        with self._state.lock:
            assert self.connection is not None
            row = self.connection.execute(
                "SELECT collection_id, name, template, schema, status, "
                "       user_scope, tenant, created_at, updated_at "
                "FROM collections "
                "WHERE collection_id = ? "
                "  AND (user_scope IS NULL OR user_scope = ?)",
                [collection_id, self.user_id],
            ).fetchone()
        if not row:
            return None
        return {
            "collection_id": row[0],
            "name": row[1],
            "template": row[2],
            "schema": json.loads(row[3]) if row[3] else None,
            "status": row[4],
            "user_scope": row[5],
            "tenant": row[6],
            "created_at": row[7],
            "updated_at": row[8],
        }

    # -- Writes (class C loopback only in v1) ----------------------------

    def create_collection(
        self,
        *,
        name: str,
        template: str | None = None,
        schema: Dict[str, Any] | None = None,
        tenant: str | None = None,
        confirm: bool = False,
    ) -> Dict[str, Any]:
        """Create a new collection. Server-mints the collection_id.

        ``confirm`` is a capability marker for the RPC seam
        (memory_service.py dispatch) — ignored in direct-store mode.
        See #200 PR-2 fix: raw RPC collection writes without
        confirm=True are denied at the service dispatch.
        """
        if not name or not name.strip():
            raise ValueError("collection name is required")
        collection_id = _mint_id("col")
        now = _now_iso()
        schema_json = json.dumps(schema) if schema else None
        with self._state.lock:
            assert self.connection is not None
            self.connection.execute(
                "INSERT INTO collections "
                "(collection_id, name, template, schema, status, "
                " user_scope, tenant, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, 'active', ?, ?, ?, ?)",
                [collection_id, name.strip(), template, schema_json,
                 self.user_id, tenant, now, now],
            )
        return {
            "collection_id": collection_id,
            "name": name.strip(),
            "template": template,
            "schema": schema,
            "status": "active",
            "user_scope": self.user_id,
            "tenant": tenant,
            "created_at": now,
            "updated_at": now,
        }

    def add_collection_item(
        self,
        *,
        collection_id: str,
        fields: Dict[str, Any],
        status: str = "open",
        tenant: str | None = None,
        confirm: bool = False,
    ) -> Dict[str, Any]:
        """Add an item to a collection. Server-mints the item_id.

        Validates fields against the collection's schema if one is set
        (minimal name/type/required check). Free-form JSON when no schema.

        ``confirm`` is a capability marker for the RPC seam — ignored
        in direct-store mode. See #200 PR-2 fix.
        """
        if not collection_id:
            raise ValueError("collection_id is required")
        if not isinstance(fields, dict):
            raise ValueError("fields must be a dict")
        if status not in _VALID_ITEM_STATUSES:
            raise ValueError(
                f"status must be one of {sorted(_VALID_ITEM_STATUSES)}"
            )
        # Verify the collection exists and is in scope.
        collection = self.get_collection(collection_id)
        if collection is None:
            raise ValueError(f"Collection not found: {collection_id}")
        # Validate fields against collection schema (if set).
        _validate_fields_against_schema(fields, collection.get("schema"))
        item_id = _mint_id("item")
        now = _now_iso()
        fields_json = json.dumps(fields)
        with self._state.lock:
            assert self.connection is not None
            self.connection.execute(
                "INSERT INTO collection_items "
                "(item_id, collection_id, fields, status, "
                " user_scope, tenant, created_at, updated_at, archived_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)",
                [item_id, collection_id, fields_json, status,
                 self.user_id, tenant, now, now],
            )
        return {
            "item_id": item_id,
            "collection_id": collection_id,
            "fields": fields,
            "status": status,
            "user_scope": self.user_id,
            "tenant": tenant,
            "created_at": now,
            "updated_at": now,
            "archived_at": None,
        }

    def update_collection_item(
        self,
        *,
        item_id: str,
        fields: Dict[str, Any] | None = None,
        status: str | None = None,
        expected_version: str | None = None,
        confirm: bool = False,
    ) -> Dict[str, Any]:
        """Update a collection item. CAS via expected_version: if provided,
        must match the item's current item_id (compare+update in one call).

        Returns the updated item dict. Raises ValueError on not found or
        CAS conflict.

        ``confirm`` is a capability marker for the RPC seam — ignored
        in direct-store mode. See #200 PR-2 fix.
        """
        if not item_id:
            raise ValueError("item_id is required")
        if status is not None and status not in _VALID_ITEM_STATUSES:
            raise ValueError(
                f"status must be one of {sorted(_VALID_ITEM_STATUSES)}"
            )
        now = _now_iso()
        with self._state.lock:
            assert self.connection is not None
            # CAS check + fetch in one query (under the lock — no TOCTOU).
            row = self.connection.execute(
                "SELECT item_id, collection_id, fields, status, "
                "       user_scope, tenant, created_at, archived_at "
                "FROM collection_items "
                "WHERE item_id = ? "
                "  AND (user_scope IS NULL OR user_scope = ?) "
                "  AND archived_at IS NULL",
                [item_id, self.user_id],
            ).fetchone()
            if not row:
                raise ValueError(f"Item not found: {item_id}")
            if expected_version is not None and row[0] != expected_version:
                raise ValueError(
                    f"CAS conflict: expected_version={expected_version} "
                    f"does not match item_id={row[0]}"
                )
            # Validate fields against collection schema if set.
            if fields is not None:
                collection = self.get_collection(row[1])
                if collection:
                    _validate_fields_against_schema(
                        fields, collection.get("schema"),
                    )
            # Build the UPDATE — SET clause params first, then WHERE params.
            set_parts: list[str] = []
            set_params: list = []
            if fields is not None:
                set_parts.append("fields = ?")
                set_params.append(json.dumps(fields))
            if status is not None:
                set_parts.append("status = ?")
                set_params.append(status)
            set_parts.append("updated_at = ?")
            set_params.append(now)
            sql = (
                f"UPDATE collection_items SET {', '.join(set_parts)} "
                f"WHERE item_id = ? "
                f"  AND (user_scope IS NULL OR user_scope = ?) "
                f"  AND archived_at IS NULL"
            )
            # WHERE params: item_id first (matches SQL order), then user_scope
            self.connection.execute(
                sql, set_params + [item_id, self.user_id],
            )
        return {
            "item_id": item_id,
            "collection_id": row[1],
            "fields": fields if fields is not None else (
                json.loads(row[2]) if row[2] else {}
            ),
            "status": status if status is not None else row[3],
            "user_scope": row[4],
            "tenant": row[5],
            "created_at": row[6],
            "updated_at": now,
            "archived_at": None,
        }

    def remove_collection_item(
        self,
        *,
        item_id: str,
        expected_version: str | None = None,
        confirm: bool = False,
    ) -> Dict[str, Any]:
        """Remove (archive) a collection item. Sets archived_at; does NOT
        delete the row (audit trail). CAS via expected_version.

        Returns the archived item dict. Raises ValueError on not found or
        CAS conflict.

        ``confirm`` is a capability marker for the RPC seam — ignored
        in direct-store mode. See #200 PR-2 fix.
        """
        if not item_id:
            raise ValueError("item_id is required")
        now = _now_iso()
        with self._state.lock:
            assert self.connection is not None
            # CAS check + archive in one query (under the lock).
            row = self.connection.execute(
                "SELECT item_id, collection_id, fields, status, "
                "       user_scope, tenant, created_at "
                "FROM collection_items "
                "WHERE item_id = ? "
                "  AND (user_scope IS NULL OR user_scope = ?) "
                "  AND archived_at IS NULL",
                [item_id, self.user_id],
            ).fetchone()
            if not row:
                raise ValueError(f"Item not found: {item_id}")
            if expected_version is not None and row[0] != expected_version:
                raise ValueError(
                    f"CAS conflict: expected_version={expected_version} "
                    f"does not match item_id={row[0]}"
                )
            self.connection.execute(
                "UPDATE collection_items SET archived_at = ?, status = 'parked' "
                "WHERE item_id = ? "
                "  AND (user_scope IS NULL OR user_scope = ?) "
                "  AND archived_at IS NULL",
                [now, item_id, self.user_id],
            )
        return {
            "item_id": item_id,
            "collection_id": row[1],
            "status": "parked",
            "archived_at": now,
        }
