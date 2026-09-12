"""Spec-12 (#387): credential-backed identity for the MCP/REST transports.

Implements the transport side of #129 for the MCP stdio and REST
surfaces: a principal proves itself with a per-principal bearer
credential registered in ``{home}/api_credential.json`` instead of the
spawner-asserted env vars (``ARGOS_API_PRINCIPAL`` / ``_USER_ID`` /
``_TENANT``). ``ARGOS_REST_TOKEN`` keeps working unchanged as the legacy
transport token.

File shape (version 1) — the SAME file the REST transport token has
used since spec-09, so legacy ``{"token": "..."}`` files stay valid:

    {
      "token": "<legacy transport token>",            # optional
      "version": 1,
      "credentials": [
        {
          "name": "simone-laptop",                    # principal id (audit)
          "token_sha256": "<64 hex>",                 # canonical (minted)
          "token": "<plaintext>",                     # alt: hashed at load
          "tenant": "default",
          "user_id": "simone",
          "principal_type": "model",                  # or "human" (class B)
          "allowed_classes": ["read", "propose"],
          "expires_at": "2026-10-12T00:00:00+00:00",  # optional
          "created_at": "2026-09-12T14:00:00+00:00"   # optional
        }
      ]
    }

Rules
-----
- Tokens match by SHA-256 with ``hmac.compare_digest`` (constant time).
  A minted token exists in plaintext only at mint time; the file stores
  the hash.
- Revocation = remove the entry (REST re-reads the file per request,
  so revocation applies immediately; MCP stdio has no per-request auth
  surface — it resolves once at startup and refuses to start if the
  named credential is missing or expired).
- Expiry: expired entries never match (REST returns 401 "expired";
  MCP refuses to start).
- Identity is NEVER widened: a credential's classes grant exactly their
  operation sets; class C (write) additionally requires the loopback
  posture; env vars can only narrow (``principal_type`` human -> model,
  ``ARGOS_API_READ_ONLY`` intersects down to read); env identity vars
  (``ARGOS_API_USER_ID`` etc.) are ignored in credential mode; and the
  facade's identity enforcement (D3) keeps rejecting client-supplied
  ``user_id``/``tenant``/``scope`` exactly as before.
- Classes are finer-grained than the env tier gates, because the facade
  authorizes per operation set: destructive members of the proposal
  tier (``erase_request``) and the review ops are NOT implied by
  ``propose`` — they are explicit classes (see the erase_request
  comment in api_facade.py):

      read             -> READ + COLLECTION_READ
      propose          -> memory_propose
      ingest           -> ingest (structured ingestion; apply confirm-gated)
      erase            -> erase_request (destructive; strict confirm gate)
      review           -> review_candidate + review_memory (class B; the
                          facade additionally requires a human principal)
      write            -> WRITE (loopback posture required)
      feedback         -> record_feedback
      collection_read  -> COLLECTION_READ
      collection_write -> COLLECTION_WRITE (loopback posture required)

Why not HMAC-signed tokens (#387 sketch item 3): the verifier always
holds this file locally, and the service's boot-time ``gate_secret``
rotates on every service restart — signed tokens would silently
invalidate on restart while adding no revocation benefit beyond a file
entry removal. Signed tokens remain the right mechanism for genuinely
remote/hosted transports (#277/#391), where the verifier may not share
a filesystem; this module is structured so a signed variant can slot in
behind the same resolve/build_context seam.

No secret values are ever logged by this module.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from api_facade import (
    AuthContext,
    COLLECTION_READ_OPERATIONS,
    COLLECTION_WRITE_OPERATIONS,
    FEEDBACK_OPERATIONS,
    READ_OPERATIONS,
    WRITE_OPERATIONS,
)

CREDENTIAL_FILE_NAME = "api_credential.json"
TOKEN_PREFIX = "argos_"

#: Class name -> facade operation set. Granular on purpose: destructive
#: proposal-tier members (ingest/erase) and the review ops require
#: explicit classes; the facade's class-B (human-only) check still
#: guards review ops at execute time.
CLASS_TO_OPERATIONS = {
    "read": frozenset(READ_OPERATIONS | COLLECTION_READ_OPERATIONS),
    "propose": frozenset({"memory_propose"}),
    "ingest": frozenset({"ingest"}),
    "erase": frozenset({"erase_request"}),
    "review": frozenset({"review_candidate", "review_memory"}),
    "write": frozenset(WRITE_OPERATIONS),
    "feedback": frozenset(FEEDBACK_OPERATIONS),
    "collection_read": frozenset(COLLECTION_READ_OPERATIONS),
    "collection_write": frozenset(COLLECTION_WRITE_OPERATIONS),
}

VALID_CLASSES = frozenset(CLASS_TO_OPERATIONS)
VALID_PRINCIPAL_TYPES = frozenset({"model", "human"})


class CredentialFileError(ValueError):
    """The credential file is malformed. Fail closed, never guess."""


def sha256_hex(token: str) -> str:
    """SHA-256 hex digest of a token (the at-rest representation)."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def mint_token() -> str:
    """Generate a fresh 256-bit bearer token (plaintext, shown once)."""
    return TOKEN_PREFIX + secrets.token_urlsafe(32)


@dataclass(frozen=True)
class Credential:
    """One per-principal credential entry."""

    name: str
    token_sha256: str
    tenant: str = "default"
    user_id: str = "default_user"
    principal_type: str = "model"
    allowed_classes: frozenset = frozenset()
    expires_at: Optional[datetime] = None
    created_at: Optional[str] = None

    def is_expired(self, now: Optional[datetime] = None) -> bool:
        if self.expires_at is None:
            return False
        current = now or datetime.now(timezone.utc)
        return current >= self.expires_at


def credential_file_path(home: Path) -> Path:
    """Default credential file location for a Hermes home."""
    return Path(home) / CREDENTIAL_FILE_NAME


# -- Parsing ------------------------------------------------------------------

def _parse_expires(raw: Any, name: str) -> Optional[datetime]:
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw.strip():
        raise CredentialFileError(
            f"credential {name!r}: expires_at must be an ISO-8601 string"
        )
    text = raw.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise CredentialFileError(
            f"credential {name!r}: expires_at is not valid ISO-8601: {raw!r}"
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _parse_entry(raw: Any, seen_names: set, seen_hashes: set) -> Credential:
    if not isinstance(raw, dict):
        raise CredentialFileError("each credential entry must be an object")

    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise CredentialFileError("credential entry is missing a non-empty 'name'")
    name = name.strip()
    if name in seen_names:
        raise CredentialFileError(f"duplicate credential name {name!r}")
    seen_names.add(name)

    token_sha256 = raw.get("token_sha256")
    plaintext = raw.get("token")
    if token_sha256 is not None and plaintext is not None:
        raise CredentialFileError(
            f"credential {name!r}: set exactly one of 'token_sha256' or 'token'"
        )
    if token_sha256 is not None:
        if not isinstance(token_sha256, str) or len(token_sha256) != 64 or not all(
            ch in "0123456789abcdef" for ch in token_sha256.lower()
        ):
            raise CredentialFileError(
                f"credential {name!r}: token_sha256 must be 64 hex characters"
            )
        token_sha256 = token_sha256.lower()
    elif isinstance(plaintext, str) and plaintext:
        token_sha256 = sha256_hex(plaintext)
    else:
        raise CredentialFileError(
            f"credential {name!r}: missing 'token_sha256' (or 'token')"
        )
    if token_sha256 in seen_hashes:
        raise CredentialFileError(
            f"credential {name!r}: token hash collides with another entry "
            "(same token registered twice?)"
        )
    seen_hashes.add(token_sha256)

    principal_type = raw.get("principal_type", "model")
    if principal_type not in VALID_PRINCIPAL_TYPES:
        raise CredentialFileError(
            f"credential {name!r}: principal_type must be one of "
            f"{sorted(VALID_PRINCIPAL_TYPES)}"
        )

    raw_classes = raw.get("allowed_classes", [])
    if not isinstance(raw_classes, list) or not all(
        isinstance(c, str) for c in raw_classes
    ):
        raise CredentialFileError(
            f"credential {name!r}: allowed_classes must be a list of strings"
        )
    classes = frozenset(raw_classes)
    unknown = classes - VALID_CLASSES
    if unknown:
        raise CredentialFileError(
            f"credential {name!r}: unknown class(es) {sorted(unknown)}; "
            f"valid: {sorted(VALID_CLASSES)}"
        )

    tenant = raw.get("tenant", "default")
    user_id = raw.get("user_id", "default_user")
    if not isinstance(tenant, str) or not tenant:
        raise CredentialFileError(f"credential {name!r}: tenant must be a string")
    if not isinstance(user_id, str) or not user_id:
        raise CredentialFileError(f"credential {name!r}: user_id must be a string")

    created_at = raw.get("created_at")
    if created_at is not None and not isinstance(created_at, str):
        raise CredentialFileError(f"credential {name!r}: created_at must be a string")

    return Credential(
        name=name,
        token_sha256=token_sha256,
        tenant=tenant,
        user_id=user_id,
        principal_type=principal_type,
        allowed_classes=classes,
        expires_at=_parse_expires(raw.get("expires_at"), name),
        created_at=created_at,
    )


def _parse_dict(data: Any) -> tuple[Optional[str], list[Credential]]:
    if not isinstance(data, dict):
        raise CredentialFileError("credential file must be a JSON object")

    transport_token = data.get("token")
    if transport_token is not None and (
        not isinstance(transport_token, str) or not transport_token
    ):
        raise CredentialFileError("'token' must be a non-empty string")

    raw_list = data.get("credentials", [])
    if not isinstance(raw_list, list):
        raise CredentialFileError("'credentials' must be a list")

    seen_names: set = set()
    seen_hashes: set = set()
    credentials = [
        _parse_entry(entry, seen_names, seen_hashes) for entry in raw_list
    ]
    return (transport_token or None, credentials)


def parse_credentials_file(path: Path) -> tuple[Optional[str], list[Credential]]:
    """Load ``(legacy_transport_token, credentials)`` from *path*.

    A missing file yields ``(None, [])`` (legacy env behavior). A
    malformed file raises :class:`CredentialFileError` — fail closed,
    never silently drop a broken credential (that would fail open).
    """
    path = Path(path)
    if not path.exists():
        return (None, [])
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise CredentialFileError(f"cannot read {path}: {exc}") from exc
    return _parse_dict(data)


# -- Resolution ---------------------------------------------------------------

def resolve_by_token(
    credentials: Iterable[Credential], token: str
) -> tuple[Optional[Credential], bool]:
    """Find the credential matching *token*.

    Returns ``(credential, expired_match)``. Hash comparison is
    constant-time; duplicate hashes are rejected at parse time, so at
    most one entry can match.
    """
    if not token:
        return (None, False)
    digest = sha256_hex(token)
    for cred in credentials:
        if hmac.compare_digest(cred.token_sha256, digest):
            if cred.is_expired():
                return (None, True)
            return (cred, False)
    return (None, False)


def resolve_by_name(
    credentials: Iterable[Credential], name: str
) -> Optional[Credential]:
    for cred in credentials:
        if cred.name == name:
            return cred
    return None


# -- Context building ---------------------------------------------------------

def build_context(
    cred: Credential,
    *,
    transport: str,
    is_loopback: bool,
    env_principal_type: str = "",
    is_read_only: bool = False,
) -> AuthContext:
    """Build an AuthContext for *cred* (never wider than the credential).

    - Class mapping is exact (see CLASS_TO_OPERATIONS).
    - Class C (write) additionally requires the loopback posture; a
      non-loopback transport never gets write ops even if the
      credential lists them.
    - ``env_principal_type="model"`` narrows human -> model (env can
      only narrow); anything else leaves the credential's value.
    - ``is_read_only`` intersects the set down to read ops.
    """
    allowed: set = set()
    for cls in cred.allowed_classes:
        allowed |= CLASS_TO_OPERATIONS[cls]

    if not is_loopback:
        allowed -= WRITE_OPERATIONS
        allowed -= COLLECTION_WRITE_OPERATIONS
    if is_read_only:
        allowed &= READ_OPERATIONS | COLLECTION_READ_OPERATIONS

    principal_type = cred.principal_type
    if env_principal_type == "model":
        principal_type = "model"

    return AuthContext(
        principal=cred.name,
        tenant=cred.tenant,
        user_id=cred.user_id,
        transport=transport,
        allowed_operations=allowed,
        can_propose="memory_propose" in allowed,
        can_feedback="record_feedback" in allowed,
        principal_type=principal_type,
        is_loopback=is_loopback,
    )


# -- Minting ------------------------------------------------------------------

def write_credential(
    path: Path,
    *,
    name: str,
    token: Optional[str] = None,
    tenant: str = "default",
    user_id: str = "default_user",
    principal_type: str = "model",
    allowed_classes: Iterable[str] = ("read",),
    expires_at: Optional[datetime] = None,
) -> tuple[str, Credential]:
    """Append a credential to *path* (atomically; preserves other keys).

    Returns ``(plaintext_token, credential)``. The plaintext token is
    NOT stored — only its SHA-256. Validation runs on the merged
    document before anything is written, so a bad entry can never
    clobber a working file.
    """
    path = Path(path)
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise CredentialFileError(
                f"refusing to modify unreadable credential file {path}: {exc}"
            ) from exc
        if not isinstance(raw, dict):
            raise CredentialFileError(
                f"refusing to modify credential file {path}: not a JSON object"
            )
    else:
        raw = {}

    plaintext = token or mint_token()
    entry = {
        "name": name,
        "token_sha256": sha256_hex(plaintext),
        "tenant": tenant,
        "user_id": user_id,
        "principal_type": principal_type,
        "allowed_classes": sorted(set(allowed_classes)),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if expires_at is not None:
        entry["expires_at"] = expires_at.astimezone(timezone.utc).isoformat()

    merged = dict(raw)
    merged.setdefault("version", 1)
    existing = merged.get("credentials", [])
    if not isinstance(existing, list):
        raise CredentialFileError(
            f"refusing to modify credential file {path}: 'credentials' not a list"
        )
    merged["credentials"] = existing + [entry]

    # Validate the merged document BEFORE touching the file.
    _parse_dict(merged)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    try:
        os.chmod(path, 0o600)  # best effort (POSIX); no-op on Windows
    except OSError:
        pass

    transport_token, credentials = _parse_dict(merged)
    written = resolve_by_name(credentials, name)
    assert written is not None  # just parsed
    return (plaintext, written)
