# Changelog

All notable changes to Argos. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), semver-less (internal tooling).

## [Unreleased]

### Added

- **deploy.py** (#7, P5.2): one-command repo → live plugin sync. `--check` reports per-file sha256 drift + repo HEAD vs last-deployed HEAD (exit 0 clean / 1 drift, gate-usable); copy mode byte-verifies every copied file, writes `.bak-<ts>` before overwrite, and appends an auditable `deploy_state.json` (timestamp, source HEAD, hashes). `--prune` is opt-in and never touches protected live artifacts (`skills/`, `*.duckdb`, `hybrid_memory_service.json`, backups). `--restart-service` kills stale `memory_service` processes per SYNC_HANDOFF.md.
- **Marker READMEs** (#22): `tool_compression/` and `ambient_context/` now state they are reference copies of patched Hermes core — NOT runtime code, NOT on the sync path.
- **Batch-4** (#36, #41, #42, #47, #48, #16): transition-only supersession, explicit conflict resolution outcomes, structural-loss guard on rewrites, project-scoped proposals, compile-to-handoff consumer, live procedure-outcome records with tripwatch.
- **Batch-3** (#21, #38): shared verdict thresholds for the self-corpus gate with per-probe timeout; rank-1 survival guard in RRF fusion + loss probe (`eval/probes/probe_rank1_loss.py`).
- **Eval-harness resume/checkpointing** (#44, benchmark clone `fix/issue-44-resume-v2`): session-level `--resume` for the shared MemConflict harness (`benchmark/eval_common.py` + `argosvault/eval_argos.py`) — progress sidecar written after each session, completed sessions skipped (no re-ingest / re-answer), per-session output appends so a killed run keeps its answers, schema-fingerprint guard refuses resume on embedder/store change, DB wipe retained for fresh runs. 13 tests (`benchmark/test_resume_checkpoint.py`).

### Changed

- **SYNC_HANDOFF.md**: Step 3 now references `scripts/deploy.py` as the only sync path (manual md5/cp loop replaced).

### Fixed

- **BM25-lite** (#26): substring token counting replaced with exact word-boundary token counting; text search and phrase-lift share one tokenizer regex.
