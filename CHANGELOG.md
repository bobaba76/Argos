# Changelog

All notable changes to Argos. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), semver-less (internal tooling).

## [Unreleased]

### Added

- **Spec-12: external human-review surface — console credential auth + docs truth-up** (#390): the admin console now accepts per-principal credentials (#387). A human credential (`principal_type: human` + the `review` class) unlocks review actions; a model credential is refused class B by the facade and sees no buttons; the legacy env path is preserved as the explicitly-local trusted UI, and an explicit `ARGOS_API_PRINCIPAL_TYPE=model` now actually narrows it (previously ignored on the console). Review buttons render only for human principals — the queue message no longer tells users to set `ARGOS_API_READ_ONLY` to "enable review actions" and the README start command drops the stale `ARGOS_API_CAN_PROPOSE=1`. Docs: the Class B story leads with the human credential (env = local-trusted only) and the console is presented as the reference human click surface (docs/integration.md).

- **Spec-12: credential-backed identity for MCP/REST** (#387, implements #129's transport side): external principals now prove themselves with per-principal bearer credentials in `api_credential.json` instead of spawner-asserted env vars. File shape (legacy `{"token": ...}` files stay valid): `credentials[]` with `name`, `token_sha256` (hash-at-rest; tokens minted via `scripts/mint_api_credential.py` are stored hashed), `tenant`, `user_id`, `principal_type`, `allowed_classes`, optional `expires_at`. Granular classes — `read`, `propose`, `ingest`, `erase`, `review`, `write`, `feedback`, `collection_read`, `collection_write` — so destructive members of the proposal tier (`erase_request`) and the review ops are explicit grants, not implied by `propose`. Never-widen posture: class C (write) still requires loopback; env vars can only narrow (principal_type human→model, read-only intersect); credential `user_id`/`tenant` beat env identity; client-supplied identity fields stay rejected by the facade (D3). Revocation = entry removal (REST re-reads per request, so it applies on the next call; MCP resolves at startup and refuses to start on missing/expired/invalid — fail-closed). REST maps unknown tokens through the credential path with `401 "Credential expired."` / `401 "Invalid credentials."` and `500 invalid_credential_config` for a malformed file; the legacy `ARGOS_REST_TOKEN` env path is byte-for-byte unchanged. 57 tests (`tests/test_api_credentials.py`), hermetic.

- **Spec-12: external parity — structured ingestion over both transports** (#386): the facade's `ingest` op (#289 structured JSON/CSV ingestion) is now reachable from outside clients — MCP `memory_ingest` tool and `POST /v1/ingest` (idempotency key required on both). Verified-before-wiring: apply mode materializes ACTIVE records via the self-approved candidate path (`review_source="tool"`), so it is class-C posture — the facade now denies apply on non-loopback transports (`write_requires_loopback`, fail-closed); preview writes nothing and stays available at proposal tier. Retrieval parity pinned: MCP/REST search returns byte-identical results to the direct store path for the same store/user (`tests/test_spec12_parity.py`); the obsolete "text-only MCP search" premise is retired — retrieval has run service-side (embedding + reranker + chains) since the shared-service architecture, no client embedder involved. 12 tests.

- **Spec-11: harness-agnostic read+write surface** (#384): MCP/REST transports now boot **write-enabled by default** — Class A (propose → human review queue) and Class C (direct write, loopback-only) are ON out of the box, no env vars required. Class B (candidate approval) stays OFF (model self-approval, never). `ARGOS_API_READ_ONLY=1` restores the spec-09 read-only default for conservative/shared deployments. `ARGOS_API_NO_LOOPBACK=1` still denies class C regardless of defaults. Existing explicit env vars (`CAN_PROPOSE`, `CAN_WRITE`) still work as belt-and-suspenders overrides.

- **External API: explain_retrieval (why-not)**: the Spec-2 retrieval diagnostic (``memory_why_not``) is now reachable end-to-end through the public facade — ``explain_retrieval`` op on the ACL allowlist, ``POST /v1/memory/explain-retrieval``, MCP ``memory_why_not`` tool, and RPC proxy/dispatch threading (the diagnostic previously had zero external call sites outside the provider/CLI). Deterministic, read-only, zero-LLM; ACL-scoped (cross-user → not_found).

- **Memory lifecycle** (#6, P5.1): archival tier, forgetting, long-horizon rollups. Three independent phases, all ship OFF by default. Phase 1: `archive_enabled` — records older than `archive_after_days` (180) with no retrievals/feedback are tiered to `archived` (out of injection pool, searchable via `include_archived=True`). Phase 2: `forget_enabled` — auto-quarantine (reversible, never delete) of `context_note`/`event`/`goal` older than `forget_after_days` (365). Phase 3: `rollup_enabled` — monthly LLM pass emitting profile-style proposals only (reuses P4.2 distillation seam). New `tier` column (zero-migration). Config: new "Lifecycle" group in the UI. 26 tests.
- **Stale-review sweep** (#10): wired the four `stale_review_*` config keys (previously parsed but never consumed) to a periodic daemon thread that re-reviews proposals stranded in `pending`. Min-age filter, batch cap, fail-soft, no auto-promotion. 26 tests.
- **deploy.py** (#7, P5.2): one-command repo → live plugin sync. `--check` reports per-file sha256 drift + repo HEAD vs last-deployed HEAD (exit 0 clean / 1 drift, gate-usable); copy mode byte-verifies every copied file, writes `.bak-<ts>` before overwrite, and appends an auditable `deploy_state.json` (timestamp, source HEAD, hashes). `--prune` is opt-in and never touches protected live artifacts (`skills/`, `*.duckdb`, `hybrid_memory_service.json`, backups). `--restart-service` kills stale `memory_service` processes per SYNC_HANDOFF.md.
- **Marker READMEs** (#22): `tool_compression/` and `ambient_context/` now state they are reference copies of patched Hermes core — NOT runtime code, NOT on the sync path.
- **Batch-4** (#36, #41, #42, #47, #48, #16): transition-only supersession, explicit conflict resolution outcomes, structural-loss guard on rewrites, project-scoped proposals, compile-to-handoff consumer, live procedure-outcome records with tripwatch.
- **Batch-3** (#21, #38): shared verdict thresholds for the self-corpus gate with per-probe timeout; rank-1 survival guard in RRF fusion + loss probe (`eval/probes/probe_rank1_loss.py`).
- **Eval-harness resume/checkpointing** (#44, benchmark clone `fix/issue-44-resume-v2`): session-level `--resume` for the shared MemConflict harness (`benchmark/eval_common.py` + `argosvault/eval_argos.py`) — progress sidecar written after each session, completed sessions skipped (no re-ingest / re-answer), per-session output appends so a killed run keeps its answers, schema-fingerprint guard refuses resume on embedder/store change, DB wipe retained for fresh runs. 13 tests (`benchmark/test_resume_checkpoint.py`).

### Changed

- **SYNC_HANDOFF.md**: Step 3 now references `scripts/deploy.py` as the only sync path (manual md5/cp loop replaced).

### Removed

- **Dead config knobs** (#359): `backup_enabled`, `backup_dst_root`, `backup_retention_snapshots` and `router_default_provider` were advertised in `config_schema.py` / `CONFIG_REFERENCE.md` but read by zero code (backup uses the nested `backup` dict; the router only reads `router_default_model`). Removed from schema, model, docs and parity tests. Any of these keys left in a live `hybrid_memory.json` are harmless — the production load paths (`provider_core._load_config`, `memory_service._load_config`) filter unknown keys before validation — and can be deleted at leisure.

### Fixed

- **consolidation_auto_apply default mismatch** (#361): the safe-default contract (dry-run/report-only unless explicit opt-in) was contradicted by every real default — `MemoryConfig` default `True`, session inline fallback `"true"`, doc row `true`. Enabling `consolidation_enabled` would silently trigger an irreversible auto-quarantine at session end. All three sites now default to `false`; explicit `"consolidation_auto_apply": "true"` still wins.
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
