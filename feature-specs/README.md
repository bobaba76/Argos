# Argos feature specs — implemented archive + pending

Shipped specs below are **built and shipped** — historical design record (intent,
safety invariants, test plans); shipped behavior is described in
[../MEMORY_SYSTEM.md](../MEMORY_SYSTEM.md) and
[../CONFIG_REFERENCE.md](../CONFIG_REFERENCE.md). Do not hand the shipped half of
this folder to a fresh session as "build me these" — they are built.

As of **2026-09-02 every spec in this index is IMPLEMENTED** — the status column
below is the live record (commit SHAs from `git log`).

| Spec | Feature | Status |
|---|---|---|
| `spec-01-ttl-expiry.md` | Best-before dates (TTL expiry tiers) | **IMPLEMENTED** (`expires_at`, `include_expired`, `as_of` fix) |
| `spec-02-why-not.md` | `memory_why_not` tool | **IMPLEMENTED** (tool + `why_not_cli.py`) |
| `spec-03-self-corpus-eval.md` | Self-corpus retrieval eval | **IMPLEMENTED** (evolved into the snapshot/gold/gate toolchain — see [../SYNC_HANDOFF.md](../SYNC_HANDOFF.md)) |
| `P4.1-semantic-merge.md` | Semantic merge (dedup upgrade) | **IMPLEMENTED** (`tests/test_semantic_dedup.py`) |
| `P4.2-distillation.md` | Distillation pass ("the dream") | **IMPLEMENTED** (gated, proposals-only; enabled in the maintainer's prod config) |
| `P4.3-live-backup.md` | True backup (cross-platform, no downtime) | **SUPERSEDED** — shipped via `EXPORT DATABASE (FORMAT PARQUET)` (`argos_plugin/backup.py`, commit `91a492c`); VSS design retained as a forensic alternative |
| `P5.1-memory-lifecycle.md` | Memory lifecycle: archival tier, forgetting, long-horizon rollups | **IMPLEMENTED 2026-09-01** (batch-F, #6 — `fb41594`) |
| `P5.2-deploy-tooling.md` | One-command repo→live plugin deploy | **IMPLEMENTED 2026-09-01** (`scripts/deploy.py` sync tool + marker READMEs, #7/#22 — `f39000d`) |
| `spec-04-trust-model.md` | Trust-model cluster (provenance taint, grounding, one-way ladder, quote verify) | **IMPLEMENTED** (batch-2: #43, #40, #39, #35) |
| `spec-05-doc-fact-namespace.md` | Doc-fact namespace: domain separation for document-sourced facts | **IMPLEMENTED 2026-09-01** (batch-12, #67 — `namespace_partition.py`, `62d85d6`) |
| `spec-06-access-scoping.md` | Access scoping: per-user, per-client ACL inside a practice tenant | **IMPLEMENTED 2026-09-01** (batch-12, #69 — `access_scoping.py`, `e020a80`/`351b73f`) |
| `spec-07-watcher-catalog.md` | The watcher: document catalog, extraction & freshness | **IMPLEMENTED 2026-09-01** (batch-12, #71 — `watcher.py`, `93d7fab`/`57c25ba`; closes #68, #70; companion #10 stale-review sweep `bc77853`) |
| `spec-08` (no file — see note) | POPIA & deployment mode | **IMPLEMENTED 2026-09-01** (#72 — provider abstraction + [../docs/annexure-a-processing-annex.md](../docs/annexure-a-processing-annex.md), `58f82c2`/`c364335`) |
| `spec-09-external-api.md` | External API: MCP server + REST + MCP client behind a trust-boundary facade | **DRAFT 2026-09-02** — plan, not shipped (issues #123–#126) |

> **Note on spec-08:** the POPIA/deployment-mode spec (#72) shipped without a
> `feature-specs/` file — its design record is `docs/annexure-a-processing-annex.md`
> plus the provider-abstraction commit. A future external-API spec filed here should
> take the next free number (**spec-09**).

Spec files are historical design records; **the status column above is the live
record**. Where a spec's own header still says "APPROVED — parked", that text
dates from before 2026-09-01 and is superseded by this table.

Line numbers and repo paths inside the specs date from their writing time
(pre-`argos_plugin` rename, pre-`shared_service` RPC) — historical, not current.
`P4.3-live-backup.md` is now **superseded** — the backup shipped via a
different approach than the spec proposed. The spec is retained as a
historical design record; see its status banner for details.