# Argos feature specs — implemented archive + pending

Shipped specs below are **built and shipped** — historical design record (intent,
safety invariants, test plans); shipped behavior is described in
[../MEMORY_SYSTEM.md](../MEMORY_SYSTEM.md) and
[../CONFIG_REFERENCE.md](../CONFIG_REFERENCE.md). Do not hand the shipped half of
this folder to a fresh session as "build me these" — they are built.

The **P5.x** specs are DRAFT design records for open board issues — read them as
the plan, not the shipped behavior.

| Spec | Feature | Status |
|---|---|---|
| `spec-01-ttl-expiry.md` | Best-before dates (TTL expiry tiers) | **IMPLEMENTED** (`expires_at`, `include_expired`, `as_of` fix) |
| `spec-02-why-not.md` | `memory_why_not` tool | **IMPLEMENTED** (tool + `why_not_cli.py`) |
| `spec-03-self-corpus-eval.md` | Self-corpus retrieval eval | **IMPLEMENTED** (evolved into the snapshot/gold/gate toolchain — see [../SYNC_HANDOFF.md](../SYNC_HANDOFF.md)) |
| `P4.1-semantic-merge.md` | Semantic merge (dedup upgrade) | **IMPLEMENTED** (`tests/test_semantic_dedup.py`) |
| `P4.2-distillation.md` | Distillation pass ("the dream") | **IMPLEMENTED** (gated, proposals-only; enabled in the maintainer's prod config) |
| `P4.3-live-backup.md` | True backup (cross-platform, no downtime) | **IMPLEMENTED** via `EXPORT DATABASE (FORMAT PARQUET)` — supersedes the VSS design (`argos_plugin/backup.py`, commit `91a492c`). VSS retained as a forensic alternative. |
| `P5.1-memory-lifecycle.md` | Memory lifecycle: archival tier, forgetting, long-horizon rollups | **DRAFT** — board issue pending |
| `P5.2-deploy-tooling.md` | One-command repo→live plugin deploy | **DRAFT** — board issue pending |
| `spec-04-trust-model.md` | Trust-model cluster (provenance taint, grounding, one-way ladder, quote verify) | **IMPLEMENTED** (batch-2: #43, #40, #39, #35) |
| `spec-05-doc-fact-namespace.md` | Doc-fact namespace: domain separation for document-sourced facts | **APPROVED** 30/8 — implementation folds into the god-file refactor window (#67) |
| `spec-06-access-scoping.md` | Access scoping: per-user, per-client ACL inside a practice tenant | **APPROVED** 30/8 — implementation parked behind Cells (#49) + spec-05 (#69) |
| `spec-07-watcher-catalog.md` | The watcher: document catalog, extraction & freshness | **APPROVED** 30/8 — implementation parked in the Cells window (#71); closes #68, #70 on landing |

Specs marked **APPROVED** are signed-off design records awaiting implementation
(parked by design — see each spec's Gate section); do not hand them to a fresh
session as "build me these" before their gate opens.

Line numbers and repo paths inside the specs date from their writing time
(pre-`argos_plugin` rename, pre-`shared_service` RPC) — historical, not current.
`P4.3-live-backup.md` is now **superseded** — the backup shipped via a
different approach than the spec proposed. The spec is retained as a
historical design record; see its status banner for details.
