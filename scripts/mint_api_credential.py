"""Mint a per-principal API credential (#387, Spec-12).

Usage:

    python mint_api_credential.py --home <hermes-home> --name simone-laptop \\
        --user-id simone --classes read,propose [--principal-type human] \\
        [--expires-in-days 90] [--tenant default] [--file <path>]

The plaintext token is printed ONCE; only its SHA-256 is written to
``{home}/api_credential.json`` (or --file). Store the token in the
client's environment (e.g. the ``Authorization: Bearer`` header for
REST, or hand it to the operator for the MCP side).

Revocation: delete the entry from the credential file (REST applies it
on the next request; MCP resolves at startup). Expiry: --expires-in-days
bakes an expires_at into the entry; expired entries never match.

Classes (finer-grained than the env tier gates; destructive members are
NOT implied by 'propose'):

    read             READ + collection reads
    propose          memory_propose
    ingest           structured ingestion (apply is confirm-gated)
    erase            erase_request (destructive; strict confirm gate)
    review           review_candidate + review_memory (class B: the
                     principal must ALSO be --principal-type human)
    write            direct writes (loopback posture required)
    feedback         record_feedback
    collection_read  collection reads
    collection_write collection writes (loopback posture required)
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_plugin_dir = Path(__file__).resolve().parent.parent / "argos_plugin"
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))

from api_credentials import (  # noqa: E402
    CLASS_TO_OPERATIONS,
    CredentialFileError,
    credential_file_path,
    write_credential,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mint a per-principal Argos API credential (#387).",
    )
    parser.add_argument("--home", required=True, type=Path,
                        help="Path to the Hermes home directory.")
    parser.add_argument("--name", required=True,
                        help="Principal name (audit identity; unique in the file).")
    parser.add_argument("--user-id", required=True,
                        help="Store user scope this credential is bound to.")
    parser.add_argument("--tenant", default="default",
                        help="Tenant scope (default: 'default').")
    parser.add_argument("--principal-type", default="model",
                        choices=["model", "human"],
                        help="'human' unlocks class B (review/approve) for "
                             "HUMAN-driven UIs only. Default: model.")
    parser.add_argument("--classes", default="read",
                        help="Comma list of classes: "
                             + ", ".join(sorted(CLASS_TO_OPERATIONS))
                             + " (default: read).")
    parser.add_argument("--expires-in-days", type=int, default=None,
                        help="Optional expiry, in days from now.")
    parser.add_argument("--file", default=None,
                        help="Credential file override "
                             "(default: {home}/api_credential.json).")
    args = parser.parse_args()

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    unknown = sorted(set(classes) - set(CLASS_TO_OPERATIONS))
    if unknown:
        print(f"error: unknown class(es): {', '.join(unknown)}; "
              f"valid: {', '.join(sorted(CLASS_TO_OPERATIONS))}", file=sys.stderr)
        return 2

    expires_at = None
    if args.expires_in_days is not None:
        if args.expires_in_days <= 0:
            print("error: --expires-in-days must be positive", file=sys.stderr)
            return 2
        expires_at = datetime.now(timezone.utc) + timedelta(days=args.expires_in_days)

    path = Path(args.file) if args.file else credential_file_path(args.home)
    try:
        token, cred = write_credential(
            path,
            name=args.name,
            tenant=args.tenant,
            user_id=args.user_id,
            principal_type=args.principal_type,
            allowed_classes=classes,
            expires_at=expires_at,
        )
    except CredentialFileError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print("Credential minted. The token below is shown ONCE — store it securely.")
    print()
    print(f"  name:            {cred.name}")
    print(f"  user_id:         {cred.user_id}")
    print(f"  tenant:          {cred.tenant}")
    print(f"  principal_type:  {cred.principal_type}")
    print(f"  classes:         {', '.join(sorted(cred.allowed_classes))}")
    print(f"  expires_at:      {cred.expires_at.isoformat() if cred.expires_at else '(none)'}")
    print(f"  file:            {path}")
    print()
    print(f"  TOKEN: {token}")
    print()
    print("REST:  Authorization: Bearer <token>")
    print("MCP:   ARGOS_API_CREDENTIAL=" + cred.name + " (spawner env)")
    print("Revoke: delete the entry from the credential file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
