# Changelog

All notable changes to Argos. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), semver-less (internal tooling).

## [Unreleased]

### Added

- **Memory lifecycle** (#6, P5.1): archival tier, forgetting, long-horizon rollups. Three independent phases, all ship OFF by default. Phase 1: `archive_enabled` — records older than `archive_after_days` (180) with no retrievals/feedback are tiered to `archived` (out of injection pool, searchable via `include_archived=True`). Phase 2: `forget_enabled` — auto-quarantine (reversible, never delete) of `context_note`/`event`/`goal` older than `forget_after_days` (365). Phase 3: `rollup_enabled` — monthly LLM pass emitting profile-style proposals only (reuses P4.2 distillation seam). New `tier` column (zero-migration). Config: new "Lifecycle" group in the UI. 26 tests.
- **Stale-review sweep** (#10): wired the four `stale_review_*` config keys (previously parsed but never consumed) to a periodic daemon thread that re-reviews proposals stranded in `pending`. Min-age filter, batch cap, fail-soft, no auto-promotion. 26 tests.
- **deploy.py** (#7, P5.2): one-command repo → live plugin sync. `--check` reports per-file sha256 drift + repo HEAD vs last-deployed HEAD (exit 0 clean / 1 drift, gate-usable); copy mode byte-verifies every copied file, writes `.bak-<ts>` before overwrite, and appends an auditable `deploy_state.json` (timestamp, source HEAD, hashes). `--prune` is opt-in and never touches protected live artifacts (`skills/`, `*.duckdb`, `hybrid_memory_service.json`, backups). `--restart-service` kills stale `memory_service` processes per SYNC_HANDOFF.md.
- **Marker READMEs** (#22): `tool_compression/` and `ambient_context/` now state they are reference copies of patched Hermes core — NOT runtime code, NOT on the sync path.
- **Batch-4** (#36, #41, #42, #47, #48, #16): transition-only supersession, explicit conflict resolution outcomes, structural-loss guard on rewrites, project-scoped proposals, compile-to-handoff consumer, live procedure-outcome records with tripwatch.
- **Batch-3** (#21, #38): shared verdict thresholds for the self-corpus gate with per-probe timeout; rank-1 survival guard in RRF fusion + loss probe (`eval/probes/probe_rank1_loss.py`).
- **Eval-harness resume/checkpointing** (#44, benchmark clone `fix/issue-44-resume-v2`): session-level `--resume` for the shared MemConflict harness (`benchmark/eval_common.py` + `argosvault/eval_argos.py`) — progress sidecar written after each session, completed sessions skipped (no re-ingest / re-answer), per-session output appends so a killed run keeps its answers, schema-fingerprint guard refuses resume on embedder/store change, DB wipe retained for fresh runs. 13 tests (`benchmark/test_resume_checkpoint.py`).

### Changed

- **SYNC_HANDOFF.md**: Step 3 now references `scripts/deploy.py` as the only sync path (manual md5/cp loop replaced).

### Fixed

- **BM25-lite** (#26): substring token counting replaced with exact word-boundary token counting; text search and phrase-lift share one tokenizer regex.

## [2026-09-04]

### Added

- **RPC wire versioning** (#246, de2171b): v:1 envelope on RPC messages; stale-service self-heal (reject + respawn-once). The shared service and its clients negotiate a protocol version; a stale service is detected and respawned once automatically.
- **StoreMixinState refactor** (#249, c7d602f): cross-mixin shared state extracted into a `StoreMixinState` dataclass. Documented, no behavior change. Reduces implicit coupling between StoreCoreMixin, StoreWriteMixin, StoreRetrievalMixin, and StoreMaintenanceMixin.
- **Config model** (#244, e44fa29): Pydantic-backed `MemoryConfig` replaces the per-attribute slurp in `provider_core.initialize()`. `extra="forbid"`, fail-soft clamping, bool coercion, backward-compat `.get()`. Follow-up (#285, 86001dc): 19 model-only keys declared as `_INTERNAL_KEYS`; schema⊆model parity + internal-keys allowlist (T1a/T1b tests).
- **Hygiene batch A** (#247 #248, 459115a): test-suite hygiene and cleanup.
- **Audit batches 4–8** (#208 #214 #213, #226, #227, #223, #264 #265 #266): store core SC1-SC7, store write SW1-SW12, store retrieval SR1-SR12, provider core, memory service, and additional audit findings across the codebase.

### Fixed

- **Store retrieval** (#245, 1096625): WHERE-clause builder extracted and tested.

## [2026-09-05]

### Added

- **Facade hardening** (#311, 480ac34): fixes #222 #299 #300 #301 #303 — API facade auth-context + ACL + validation + audit hardening.
- **RPC audit-path hardening** (#313, 26d95bd): fix #312 — `write_access_audit` + `export_access_audit` threaded through `SharedMemoryStore` RPC proxy.
- **Repo hardening** (#314, 8f4c860): fixes #304 #305 #307 #308 #309 — collapse dual-branch mixin re-export shells in store.py (#304), periodic DuckDB-Kuzu reconciliation probe for graph drift (#305), split test_hybrid_memory.py mega-file + retire run_tests.py (#307), deploy atomic swap / versioned rollback (#308), clean untracked strays + .gitignore (#309).

### Fixed

- **Config parity canary** (#274): schema/model/loader parity tests + realistic-fixture canary — CI fails if schema/model/loader drift or any silent config wipe.
- **Liveness probes** (#275): startup self-smoke test (LP1), per-feature hit counters (LP2), config fingerprint (LP3) — silent feature death is now a boot ERROR, not a silent degradation.
- **Stale docs** (#306): README test count, CHANGELOG, CLAIMS-AUDIT pin, .project_readme, MEMORY_SYSTEM updated to match master.
- **CONFIG_REFERENCE** (#310): full config surface documented, per-key descriptions expanded, schema/reference parity test.
