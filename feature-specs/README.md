# Argos feature specs — implemented archive

All six specs are **built and shipped**. These documents are the historical
design record (intent, safety invariants, test plans). Shipped behavior is
described in [../MEMORY_SYSTEM.md](../MEMORY_SYSTEM.md) and
[../CONFIG_REFERENCE.md](../CONFIG_REFERENCE.md). Do not hand this folder to a
fresh session as "build me these" — they are built.

| Spec | Feature | Status |
|---|---|---|
| `spec-01-ttl-expiry.md` | Best-before dates (TTL expiry tiers) | **IMPLEMENTED** (`expires_at`, `include_expired`, `as_of` fix) |
| `spec-02-why-not.md` | `memory_why_not` tool | **IMPLEMENTED** (tool + `why_not_cli.py`) |
| `spec-03-self-corpus-eval.md` | Self-corpus retrieval eval | **IMPLEMENTED** (evolved into the snapshot/gold/gate toolchain — see [../SYNC_HANDOFF.md](../SYNC_HANDOFF.md)) |
| `P4.1-semantic-merge.md` | Semantic merge (dedup upgrade) | **IMPLEMENTED** (`tests/test_semantic_dedup.py`) |
| `P4.2-distillation.md` | Distillation pass ("the dream") | **IMPLEMENTED** (gated, proposals-only; enabled in the maintainer's prod config) |
| `P4.3-live-backup.md` | True backup (cross-platform, no downtime) | **IMPLEMENTED** via `EXPORT DATABASE (FORMAT PARQUET)` — supersedes the VSS design (`argos_plugin/backup.py`, commit `91a492c`). VSS retained as a forensic alternative. |

Line numbers and repo paths inside the specs date from their writing time
(pre-`argos_plugin` rename, pre-`shared_service` RPC) — historical, not current.
`P4.3-live-backup.md` is now **superseded** — the backup shipped via a
different approach than the spec proposed. The spec is retained as a
historical design record; see its status banner for details.
