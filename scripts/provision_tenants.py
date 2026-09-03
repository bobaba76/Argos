#!/usr/bin/env python3
"""Tenant provisioning and inspection tool (#130).

Usage:
  python scripts/provision_tenants.py <home> status
  python scripts/provision_tenants.py <home> validate
  python scripts/provision_tenants.py <home> add <name> [--db FILE] [--graph DIR]
      [--allowed-user-ids u1,u2] [--credential TOKEN:USER_ID]
      [--review-mode confirm|auto] [--local-only true|false]
  python scripts/provision_tenants.py <home> migrate-legacy
  python scripts/provision_tenants.py <home> backup <tenant> [--dst-root DIR]
  python scripts/provision_tenants.py <home> list-backups <tenant> [--dst-root DIR]

Commands:
  status          Show all configured tenants, resolved paths, and mode.
  validate        Validate the config without starting the service.
  add             Add a new tenant to the config (validates first).
  migrate-legacy  Migrate a single-tenant config to an explicit 'default' cell.
  backup          Back up a specific tenant (requires explicit tenant name).
  list-backups    List snapshots for a tenant.

No live data or credentials are committed. The tool reads/writes
hybrid_memory.json in <home> and validates all inputs before applying.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

# Allow running from the repo root or scripts/ dir.
_repo = Path(__file__).resolve().parent.parent
_plugin = _repo / "argos_plugin"
if str(_plugin) not in sys.path:
    sys.path.insert(0, str(_plugin))

_TENANT_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")


def _load_config(home: Path) -> dict:
    path = home / "hybrid_memory.json"
    if not path.exists():
        return {}
    try:
        val = json.loads(path.read_text(encoding="utf-8"))
        return val if isinstance(val, dict) else {}
    except Exception as exc:
        print(f"ERROR: malformed config {path}: {exc}", file=sys.stderr)
        sys.exit(1)


def _save_config(home: Path, config: dict) -> None:
    path = home / "hybrid_memory.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(config, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _validate_tenant_name(name: str) -> None:
    if not name or not _TENANT_NAME_RE.match(name):
        print(
            f"ERROR: invalid tenant name {name!r} — must be alphanumeric, _, or -, "
            f"1-64 chars, start with alphanumeric",
            file=sys.stderr,
        )
        sys.exit(1)


def _validate_path(path: str, field: str, tenant: str) -> None:
    if not path:
        print(f"ERROR: tenant {tenant!r}: {field} is empty", file=sys.stderr)
        sys.exit(1)
    p = Path(path)
    if p.is_absolute():
        print(f"ERROR: tenant {tenant!r}: {field}={path!r} must be relative", file=sys.stderr)
        sys.exit(1)
    if ".." in p.parts:
        print(f"ERROR: tenant {tenant!r}: {field}={path!r} contains '..'", file=sys.stderr)
        sys.exit(1)
    if len(path) >= 2 and path[1] == ":":
        print(f"ERROR: tenant {tenant!r}: {field}={path!r} has drive letter", file=sys.stderr)
        sys.exit(1)
    if path.startswith("\\\\") or path.startswith("//"):
        print(f"ERROR: tenant {tenant!r}: {field}={path!r} is UNC path", file=sys.stderr)
        sys.exit(1)


def cmd_status(home: Path, args) -> None:
    """Show all configured tenants, resolved paths, and mode."""
    config = _load_config(home)
    tenants = config.get("tenants")
    if not isinstance(tenants, dict) or not tenants:
        # Legacy single-tenant.
        print("Mode: legacy single-tenant (no 'tenants' key)")
        print(f"Home: {home}")
        db = config.get("database_filename", "hybrid_memory.duckdb")
        graph = config.get("graph_dirname", "hybrid_memory_kuzu")
        print(f"  Database: {db}")
        print(f"  Graph:    {graph}")
        print()
        print("Use 'migrate-legacy' to convert to explicit 'default' cell.")
        return

    has_creds = any(
        isinstance(t.get("credentials"), list) and t["credentials"]
        for t in tenants.values()
    )
    mode = "multi-user" if has_creds else "trusted-local"
    print(f"Mode: {mode}")
    print(f"Home: {home}")
    print(f"Tenants: {len(tenants)}")
    print()

    for name, entry in tenants.items():
        entry = entry if isinstance(entry, dict) else {}
        db = entry.get("database_filename", config.get("database_filename", "hybrid_memory.duckdb"))
        graph = entry.get("graph_dirname", config.get("graph_dirname", "hybrid_memory_kuzu"))
        allowed = entry.get("allowed_user_ids", [])
        creds = entry.get("credentials", [])
        overlay = entry.get("config", {})
        review_mode = overlay.get("review_mode", "confirm")
        local_only = overlay.get("local_only", "false")
        is_default = name == "default" or name == min(tenants.keys())

        print(f"  [{name}]{' (default)' if is_default else ''}")
        print(f"    Database:       {db}")
        print(f"    Graph:          {graph}")
        print(f"    Allowed users:  {len(allowed) if isinstance(allowed, list) else 0}")
        print(f"    Credentials:    {len(creds) if isinstance(creds, list) else 0}")
        print(f"    Review mode:    {review_mode}")
        print(f"    Local only:     {local_only}")
        print()


def cmd_validate(home: Path, args) -> None:
    """Validate the config without starting the service."""
    config = _load_config(home)
    tenants = config.get("tenants")
    if not isinstance(tenants, dict) or not tenants:
        print("OK: legacy single-tenant config (no 'tenants' key)")
        return

    errors: list[str] = []
    seen_dbs: dict = {}
    seen_graphs: dict = {}
    seen_users: dict = {}

    for name, entry in tenants.items():
        # Name validation.
        if not name or not _TENANT_NAME_RE.match(name):
            errors.append(f"Invalid tenant name: {name!r}")
            continue
        entry = entry if isinstance(entry, dict) else {}

        # Path validation.
        db = entry.get("database_filename", config.get("database_filename", "hybrid_memory.duckdb"))
        graph = entry.get("graph_dirname", config.get("graph_dirname", "hybrid_memory_kuzu"))
        try:
            p = Path(str(db))
            if p.is_absolute() or ".." in p.parts:
                errors.append(f"Tenant {name!r}: invalid database_filename {db!r}")
            elif db in seen_dbs:
                errors.append(f"database_filename collision: {db!r} in {seen_dbs[db]!r} and {name!r}")
            else:
                seen_dbs[db] = name
        except Exception as exc:
            errors.append(f"Tenant {name!r}: database_filename error: {exc}")

        try:
            p = Path(str(graph))
            if p.is_absolute() or ".." in p.parts:
                errors.append(f"Tenant {name!r}: invalid graph_dirname {graph!r}")
            elif graph in seen_graphs:
                errors.append(f"graph_dirname collision: {graph!r} in {seen_graphs[graph]!r} and {name!r}")
            else:
                seen_graphs[graph] = name
        except Exception as exc:
            errors.append(f"Tenant {name!r}: graph_dirname error: {exc}")

        # User ID duplicate check.
        allowed = entry.get("allowed_user_ids")
        if isinstance(allowed, list):
            for uid in allowed:
                uid_s = str(uid)
                if uid_s in seen_users and seen_users[uid_s] != name:
                    errors.append(
                        f"allowed_user_ids conflict: {uid_s!r} in both "
                        f"{seen_users[uid_s]!r} and {name!r}"
                    )
                else:
                    seen_users[uid_s] = name

    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print(f"OK: {len(tenants)} tenant(s) validated successfully")


def cmd_add(home: Path, args) -> None:
    """Add a new tenant to the config."""
    _validate_tenant_name(args.name)
    config = _load_config(home)
    if "tenants" not in config or not isinstance(config.get("tenants"), dict):
        print(f"ERROR: no 'tenants' map in config. Use 'migrate-legacy' first.", file=sys.stderr)
        sys.exit(1)

    tenants = config["tenants"]
    if args.name in tenants:
        print(f"ERROR: tenant {args.name!r} already exists", file=sys.stderr)
        sys.exit(1)

    # Build the tenant entry.
    entry: dict = {}
    if args.db:
        _validate_path(args.db, "database_filename", args.name)
        entry["database_filename"] = args.db
    else:
        entry["database_filename"] = f"{args.name}.duckdb"
    if args.graph:
        _validate_path(args.graph, "graph_dirname", args.name)
        entry["graph_dirname"] = args.graph
    else:
        entry["graph_dirname"] = f"{args.name}_kuzu"

    if args.allowed_user_ids:
        entry["allowed_user_ids"] = [u.strip() for u in args.allowed_user_ids.split(",") if u.strip()]

    if args.credential:
        creds = []
        for c in args.credential:
            parts = c.split(":", 1)
            if len(parts) != 2:
                print(f"ERROR: credential must be TOKEN:USER_ID, got {c!r}", file=sys.stderr)
                sys.exit(1)
            creds.append({"token": parts[0], "user_id": parts[1]})
        entry["credentials"] = creds

    overlay: dict = {}
    if args.review_mode:
        overlay["review_mode"] = args.review_mode
    if args.local_only:
        overlay["local_only"] = args.local_only
    if overlay:
        entry["config"] = overlay

    # Validate the full config with the new tenant before saving.
    config["tenants"][args.name] = entry
    # Re-validate by running the validation logic.
    old_argv = sys.argv
    sys.argv = ["provision", str(home), "validate"]
    try:
        # Don't exit — just check.
        config_copy = dict(config)
        tenants_copy = config_copy.get("tenants", {})
        seen_dbs = {}
        for n, e in tenants_copy.items():
            db = e.get("database_filename", "hybrid_memory.duckdb")
            if db in seen_dbs:
                print(f"ERROR: database_filename collision: {db!r}", file=sys.stderr)
                sys.exit(1)
            seen_dbs[db] = n
    finally:
        sys.argv = old_argv

    _save_config(home, config)
    print(f"OK: tenant {args.name!r} added")
    print(f"  Database: {entry['database_filename']}")
    print(f"  Graph:    {entry['graph_dirname']}")
    if args.allowed_user_ids:
        print(f"  Users:    {args.allowed_user_ids}")
    if args.credential:
        print(f"  Creds:    {len(args.credential)}")
    print()
    print("Restart the memory service to apply changes.")


def cmd_migrate_legacy(home: Path, args) -> None:
    """Migrate a single-tenant config to an explicit 'default' cell."""
    config = _load_config(home)
    if "tenants" in config and isinstance(config["tenants"], dict) and config["tenants"]:
        print("ERROR: config already has a 'tenants' map — nothing to migrate", file=sys.stderr)
        sys.exit(1)

    # Build the default tenant from the global config.
    db = config.get("database_filename", "hybrid_memory.duckdb")
    graph = config.get("graph_dirname", "hybrid_memory_kuzu")

    default_entry = {
        "database_filename": db,
        "graph_dirname": graph,
    }

    # Preserve any global settings that should be in the overlay.
    overlay_keys = [
        "review_mode", "max_injected_items", "inject_content_char_cap",
        "external_sources_require_confirmation", "local_only",
        "reranker_top_n", "phrase_lift_alpha", "phrase_lift_pool",
    ]
    overlay = {}
    for k in overlay_keys:
        if k in config:
            overlay[k] = config[k]
    if overlay:
        default_entry["config"] = overlay

    config["tenants"] = {"default": default_entry}
    _save_config(home, config)
    print("OK: migrated legacy config to explicit 'default' cell")
    print(f"  Database: {db}")
    print(f"  Graph:    {graph}")
    print()
    print("The existing database and graph files are unchanged.")
    print("Restart the memory service to apply changes.")


def cmd_backup(home: Path, args) -> None:
    """Back up a specific tenant (requires explicit tenant name)."""
    if not args.tenant:
        print("ERROR: --tenant is required for backup (no silent default)", file=sys.stderr)
        sys.exit(1)
    _validate_tenant_name(args.tenant)
    config = _load_config(home)
    tenants = config.get("tenants", {})
    if args.tenant not in tenants:
        print(f"ERROR: tenant {args.tenant!r} not found in config", file=sys.stderr)
        print(f"Available: {list(tenants.keys())}", file=sys.stderr)
        sys.exit(1)

    # Import and run the backup.
    sys.path.insert(0, str(_plugin))
    from backup import backup_store, list_snapshots

    entry = tenants[args.tenant]
    db_name = entry.get("database_filename", "hybrid_memory.duckdb")
    db_path = home / db_name

    dst_root = args.dst_root or str(home / "backups" / "memory")
    print(f"Backing up tenant {args.tenant!r} from {db_path} to {dst_root}")

    # We need a connection — use the store directly.
    from store import DuckDBMemoryStore
    store = DuckDBMemoryStore(db_path, user_id="default_user")
    try:
        manifest = backup_store(
            store.connection, dst_root,
            source_db_path=db_path,
        )
        manifest["tenant"] = args.tenant
        print(json.dumps(manifest, indent=2))
    finally:
        store.close()


def cmd_list_backups(home: Path, args) -> None:
    """List snapshots for a tenant."""
    if not args.tenant:
        print("ERROR: --tenant is required", file=sys.stderr)
        sys.exit(1)
    sys.path.insert(0, str(_plugin))
    from backup import list_snapshots

    dst_root = args.dst_root or str(home / "backups" / "memory")
    snapshots = list_snapshots(dst_root)
    print(json.dumps({"tenant": args.tenant, "snapshots": snapshots}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Tenant provisioning tool (#130)")
    parser.add_argument("home", help="Hermes home directory")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="Show all configured tenants")
    sub.add_parser("validate", help="Validate config without starting service")

    add_p = sub.add_parser("add", help="Add a new tenant")
    add_p.add_argument("name", help="Tenant name")
    add_p.add_argument("--db", help="Database filename (relative to home)")
    add_p.add_argument("--graph", help="Graph directory name (relative to home)")
    add_p.add_argument("--allowed-user-ids", help="Comma-separated user IDs")
    add_p.add_argument("--credential", action="append", help="TOKEN:USER_ID (can repeat)")
    add_p.add_argument("--review-mode", choices=["confirm", "auto"])
    add_p.add_argument("--local-only", choices=["true", "false"])

    sub.add_parser("migrate-legacy", help="Migrate single-tenant to explicit default cell")

    bk_p = sub.add_parser("backup", help="Back up a specific tenant")
    bk_p.add_argument("--tenant", required=True, help="Tenant name (required)")
    bk_p.add_argument("--dst-root", help="Backup destination root")

    lb_p = sub.add_parser("list-backups", help="List snapshots for a tenant")
    lb_p.add_argument("--tenant", required=True, help="Tenant name (required)")
    lb_p.add_argument("--dst-root", help="Backup destination root")

    args = parser.parse_args()
    home = Path(args.home).resolve()

    if args.command == "status":
        cmd_status(home, args)
    elif args.command == "validate":
        cmd_validate(home, args)
    elif args.command == "add":
        cmd_add(home, args)
    elif args.command == "migrate-legacy":
        cmd_migrate_legacy(home, args)
    elif args.command == "backup":
        cmd_backup(home, args)
    elif args.command == "list-backups":
        cmd_list_backups(home, args)


if __name__ == "__main__":
    main()
