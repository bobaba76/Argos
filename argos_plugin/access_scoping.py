"""Spec-06 (#69): per-user, per-client access control inside a practice tenant.

The inside-tenant wall. Cells (#49) is the between-tenant wall; this spec
governs who can see what *within* a single tenant.

The one master rule: **facts inherit access from their source document.**
A fact extracted from a payroll PDF inside Client X's folder carries that
folder's access. Never more permissive than the source.

Model (D1):
  - **Allow mask**: each staff role holds a set of client scopes it may
    query. Principals/partners default to **wheel** (all client scopes).
  - **Deny list**: a per-user or per-role deny list for content that must
    not exist for that user. **Precedence: deny > allow > wheel.**
  - **Hidden deny**: excluded content never appears — not in results, not
    as a hint. The access event IS recorded in the audit log.

Deterministic, zero LLM calls, no new deps. Pure functions — the store
and provider call these on retrieval results before ranking.

Backward compatible: **absence of an ACL config = today's open-store
behaviour.** No config file, no enforcement, no filter.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

logger = logging.getLogger(__name__)

# Reserved doc_class for practice-internal documents (payroll, partner
# profit-share). Principals-only by default; no staff, ever.
PRACTICE_INTERNAL = "practice-internal"


class ACLConfig:
    """Loaded ACL configuration for a tenant.

    A null/empty config (no file, empty JSON) means **open store** —
    today's behaviour. Enforcement only activates when at least one role
    or deny entry is present.
    """

    def __init__(
        self,
        *,
        roles: Dict[str, Dict[str, Any]] | None = None,
        user_roles: Dict[str, str] | None = None,
        deny_lists: Dict[str, List[Dict[str, Any]]] | None = None,
        enforcement_on: bool = False,
        parse_error: bool = False,
    ) -> None:
        # roles: {role_name: {"client_scopes": ["acme", "beta"], "wheel": False}}
        self.roles = roles or {}
        # user_roles: {user_id: role_name}
        self.user_roles = user_roles or {}
        # deny_lists: {user_id|role_name: [{"client_scope": "acme", "doc_class": "practice-internal"}]}
        self.deny_lists = deny_lists or {}
        # Enforcement flips on when the second staff user joins (D6).
        # Pilot (single-user) = audit skeleton only, no enforcement.
        self.enforcement_on = enforcement_on
        # True when an ACL config FILE existed but could not be loaded
        # (JSON error, unreadable, structurally invalid). Distinguishes
        # "no ACL configured" (user's choice, open store) from "ACL
        # configured but corrupted" (must fail closed in API mode).
        # See ArgosAPIFacade (D3).
        self.parse_error = parse_error

    @classmethod
    def from_file(cls, path: str | Path) -> "ACLConfig":
        """Load from a JSON sidecar file. Returns an empty (open-store)
        config if the file doesn't exist or is empty."""
        p = Path(path)
        if not p.exists():
            return cls()
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("ACL config %s unreadable: %s — defaulting to open store", p, exc)
            # Fail-closed marker: the file EXISTS but is corrupted. API
            # mode refuses to start on this (see ArgosAPIFacade D3); the
            # internal/intra-process path keeps today's open-store fallback.
            return cls(parse_error=True)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ACLConfig":
        """Build from a parsed dict (used by tests and the service loader).

        Structurally invalid data (wrong container types) is reported
        via ``parse_error`` instead of silently becoming an open store.
        """
        if not data:
            return cls()
        # Absent keys are valid (defaults apply); only WRONG TYPES are
        # structurally invalid.
        roles = data.get("roles")
        user_roles = data.get("user_roles")
        deny_lists = data.get("deny_lists")
        bad = (
            (roles is not None and not isinstance(roles, dict))
            or (user_roles is not None and not isinstance(user_roles, dict))
            or (deny_lists is not None and not isinstance(deny_lists, dict))
            or not isinstance(data.get("enforcement_on", False), bool)
        )
        # AS3: validate that each role's client_scopes is a list of strings.
        # A string "acme" instead of ["acme"] would produce set("acme") =
        # {'a','c','m','e'} — corrupted mask, denied own scope.
        if isinstance(roles, dict):
            for role_name, role_def in roles.items():
                if isinstance(role_def, dict):
                    cs = role_def.get("client_scopes")
                    if cs is not None:
                        if isinstance(cs, str):
                            logger.warning(
                                "ACL role %r has client_scopes as a string %r, "
                                "not a list — wrapping in a list (AS3)",
                                role_name, cs,
                            )
                            role_def["client_scopes"] = [cs]
                        elif not isinstance(cs, list) or not all(
                            isinstance(s, str) for s in cs
                        ):
                            bad = True
        if bad:
            logger.warning("ACL config structurally invalid — defaulting to open store (parse_error)")
            return cls(parse_error=True)
        config = cls(
            roles=data.get("roles") or {},
            user_roles=data.get("user_roles") or {},
            deny_lists=data.get("deny_lists") or {},
            enforcement_on=bool(data.get("enforcement_on", False)),
        )
        # AS4: warn on dangling user_roles references to non-existent roles.
        for uid, role_name in config.user_roles.items():
            if role_name not in config.roles:
                logger.warning(
                    "ACL user_roles: user %r assigned to role %r which does "
                    "not exist in roles config — user will be deny-all (AS4)",
                    uid, role_name,
                )
        # AS5: warn on empty deny entries (wildcard deny footgun).
        for key, entries in config.deny_lists.items():
            for entry in entries:
                if not entry.get("client_scope") and not entry.get("doc_class"):
                    logger.warning(
                        "ACL deny_lists: entry for %r has no client_scope or "
                        "doc_class — this denies ALL content for that scope "
                        "(AS5): %r",
                        key, entry,
                    )
        return config

    @property
    def is_open_store(self) -> bool:
        """True when no ACL config is active (today's behaviour)."""
        if self.enforcement_on:
            return False
        return not self.roles and not self.deny_lists

    def role_for(self, user_id: str) -> str | None:
        """Return the role name for a user, or None if unassigned."""
        return self.user_roles.get(user_id)

    def allow_mask(self, user_id: str) -> Set[str] | None:
        """Return the set of client scopes the user may query.

        Returns None for **wheel** (all scopes, no filter).
        Returns an empty set for a user with no role and enforcement on
        (deny-all — fails closed).
        """
        if self.is_open_store:
            return None  # open store — no filter
        role_name = self.role_for(user_id)
        if role_name is None:
            # Unassigned user under enforcement: deny-all.
            return set()
        role = self.roles.get(role_name) or {}
        if role.get("wheel"):
            return None  # wheel = all scopes
        return set(role.get("client_scopes") or [])

    def deny_entries(self, user_id: str) -> List[Dict[str, Any]]:
        """Return the deny list entries for a user (user-level + role-level)."""
        entries: List[Dict[str, Any]] = []
        # User-level deny list.
        entries.extend(self.deny_lists.get(user_id) or [])
        # Role-level deny list.
        role_name = self.role_for(user_id)
        if role_name:
            entries.extend(self.deny_lists.get(role_name) or [])
        return entries

    def is_denied(
        self,
        user_id: str,
        *,
        client_scope: str | None,
        doc_class: str | None,
    ) -> bool:
        """Check whether a specific (client_scope, doc_class) pair is denied.

        Deny entries match on client_scope (None = wildcard) and/or
        doc_class (None = wildcard). An entry matches if all its non-null
        fields match the record's fields.
        """
        for entry in self.deny_entries(user_id):
            entry_cs = entry.get("client_scope")
            entry_dc = entry.get("doc_class")
            cs_match = entry_cs is None or entry_cs == client_scope
            dc_match = entry_dc is None or entry_dc == doc_class
            if cs_match and dc_match:
                return True
        return False

    def can_see(
        self,
        user_id: str,
        *,
        client_scope: str | None,
        doc_class: str | None,
    ) -> bool:
        """Full mask evaluation: deny > practice-internal > allow > wheel.

        Returns True if the user may see a record with the given
        client_scope and doc_class. Precedence (AS6 — corrected to match
        the actual code order):
        1. Deny list — if matched, False (hidden, logged).
        2. Practice-internal — principals-only (wheel), staff denied.
        3. Allow mask — if client_scope not in mask, False.
        4. Default: allowed.
        """
        if self.is_open_store:
            return True
        # 1. Deny beats everything.
        if self.is_denied(user_id, client_scope=client_scope, doc_class=doc_class):
            return False
        # 3. Practice-internal: only wheel (principals) may see these.
        if doc_class == PRACTICE_INTERNAL:
            mask = self.allow_mask(user_id)
            # mask is None only for wheel or open store. Open store already
            # returned True above. So None here = wheel.
            if mask is not None:
                return False
            return True
        # 2. Allow mask.
        mask = self.allow_mask(user_id)
        if mask is None:
            return True  # wheel or open store
        # client_scope NULL = global; visible inside any client query
        # (same convention as spec-05).
        if client_scope is None:
            return True
        return client_scope in mask


def filter_records_by_access(
    records: List[Any],
    config: ACLConfig,
    user_id: str,
) -> Tuple[List[Any], int]:
    """Filter retrieval results by the access mask. Returns (visible, denied_count).

    Records that fail the mask are silently dropped (hidden deny). The
    caller is responsible for writing the audit row (including the
    denied_count). This function is pure — no side effects.
    """
    if config.is_open_store:
        return records, 0
    visible: List[Any] = []
    denied = 0
    for r in records:
        cs = getattr(r, "client_scope", None)
        dc = getattr(r, "doc_class", None)
        if config.can_see(user_id, client_scope=cs, doc_class=dc):
            visible.append(r)
        else:
            denied += 1
    return visible, denied


def filter_graph_neighbours(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    config: ACLConfig,
    user_id: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    """Post-traversal graph filter: drop cross-scope neighbours.

    The graph is the leak vector — a shared director between Client A and
    Client B must not surface B's facts through the relationship graph
    when the user only has A in their mask.

    Uses ``config.can_see()`` (AS1) so deny lists, practice-internal
    doc_class, and the allow mask are all enforced consistently — the
    same gate as ``filter_records_by_access``. Nodes without an ``id``
    field are skipped (AS8) — they can't be safely tracked and could
    share the empty-string ID, making all of them visible if one is.

    Returns (filtered_nodes, filtered_edges, dropped_count).
    """
    if config.is_open_store:
        return nodes, edges, 0
    # AS1: use can_see() for full deny-list + doc_class + mask check.
    # Do NOT early-return for wheel (mask is None) — deny lists must
    # still be enforced even for wheel users.
    mask = config.allow_mask(user_id)

    def _node_attrs(node: Dict[str, Any]) -> Dict[str, Any]:
        return node.get("attributes") or {}

    # Determine which nodes are visible.
    visible_node_ids: Set[str] = set()
    dropped = 0
    for node in nodes:
        # AS8: skip nodes without an id field — they can't be safely
        # tracked and sharing "" would make all id-less nodes visible.
        node_id = node.get("id")
        if not node_id:
            dropped += 1
            continue
        attrs = _node_attrs(node)
        cs = attrs.get("client_scope")
        dc = attrs.get("doc_class")
        # AS1: use can_see() for full deny-list + doc_class + mask check.
        if config.can_see(user_id, client_scope=cs, doc_class=dc):
            visible_node_ids.add(node_id)
        else:
            dropped += 1

    filtered_nodes = [n for n in nodes if n.get("id") in visible_node_ids]
    # Drop edges where either endpoint was filtered out.
    filtered_edges = [
        e for e in edges
        if e.get("source") in visible_node_ids
        and e.get("target") in visible_node_ids
    ]
    return filtered_nodes, filtered_edges, dropped
