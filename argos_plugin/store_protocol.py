"""Shared RPC protocol constants for the local memory service (#246).

Both ``service_client.py`` (client) and ``memory_service.py`` (server)
import ``_PROTOCOL_VERSION`` from here so they can't drift — a single
source of truth for the wire format version.

The version is a monotonic integer.  When the request/response envelope
changes in a backwards-incompatible way, bump it.  The server rejects
mismatched versions with a structured ``VersionMismatch`` error; the
client self-heals by respawning the stale service.
"""
from __future__ import annotations

_PROTOCOL_VERSION = 1
