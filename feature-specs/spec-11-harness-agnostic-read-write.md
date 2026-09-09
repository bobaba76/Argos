# Spec 11 — Harness-agnostic read+write surface (MCP/REST write-enabled by default)

Status: **DRAFT 2026-09-09** — supersedes spec-09's "no speculative REST writes" stance. Implementation is tracked as issue #384 (one PR, Devin) and lands behind this spec's acceptance criteria. Contents are read as the plan, never as a description of live behavior — today's live default is still read-only until the PR merges.

## Pivot (owned explicitly)

- spec-09 (2/9) deliberately shipped read-first defaults: "no speculative REST writes" + "loopback-first." Rationale was sound for v1: transports are trust boundaries, fail-closed first.

- **9/9 decision (user):** harness-agnostic support is the product. The majority of worldwide users are NOT Hermes users; a read-only-by-default MCP/REST surface means an external harness's store stays empty forever → uninstall. REST writes become v1, not deferred. This is a change of requirements — recorded as such, so the old read-first stance reads as history, not as a bug.oor an accident.



## Problem (adoption framedy

- Any MCP/REST client (OpenWebUI, Claude Desktop, Cursor, etc.) following the documented flow gets a **read-only toolset by default**. The store stays empty forever; writing requires discovering undocumented env gates. Adoption killer for non-Hermes harnesses.



## Decisions

1. **Write-enabled by default on both transports.** Both MCP stdio and REST are loopback transports, already token/bind-protected:
   - `ARGOS_API_CAN_PROPOSE` — default **ON** (class A: propose → human review queue)。
   - `ARGOS_API_CAN_WRITE` — default **ON** (class C: direct active-memory writes, loopback-required。
   - `ARGOS_API_CAN_FEEDBACK` — stays **OFF** (class B: approvals. Model self-approval must keep failing closed)。
   - `ARGOS_API_PRINCIPAL_TYPE` — default stays `"model"` (fail-closed: unset = model = denied class B). Unchanged from spec-09「no self-approval, ever」。

   - NEW: `ARGOS_API_READ_ONLY=1` — forces the old default (write tiers off),the paranoid escape hatch for conservative or shared deployments. `ARGOS_API_NO_LOOPBACK=1` unchanged (explicit non-loopback deployments stay honest about itた。

2. **Hermes is unaffected.** Hermes's memory path is the plugin provider directly, never `mcp_server.py` / `rest_server.py`. MCP/REST consumers are exclusively external harnesses — the default flip costs Hermes instances nothing and introduces no new attack surface on existing installs.oor

3. **Class B invariant does not move.** No env-gate combination grants model-principal approval — `PRINCIPAL_TYPE` defaults to `"model"` and any explicit `"human"` requires the transport to be an actually-human-driven UI. Idempotency, compare-and-set, collection/HMAC gates, graph isolation → all unchanged (spec-09/10 gates stay green).



## Implementation scope (one PR, Devin)

- `argos_plugin/mcp_server.py` — `_load_auth_context`: default `allowed = READ | COLLECTION_READ | PROPOSAL | WRITE | COLLECTION_WRITE`; `ARGOS_API_READ_ONLY=1` flips back to read tier only.oor
- `argos_plugin/rest_server.py` — same flip in its auth-context builder.oor
- `argos_plugin/admin_console.py` — same flip (its builder copies the same pattern。パラ
- Tests (`test_spec10_transports.py`, `test_admin_console.py`, plus new edge cases): existing "read-only by default" assertions flip to explicitly set `ARGOS_API_READ_ONLY=1`; new assertions: (a) no-env spawn exposes write ops (b) `READ_ONLY=1` restores the old surface;; (c) `NO_LOOPBACK=1` + `CAN_WRITE` still denies class C writes;; (d) `CAN_FEEDBACK=1` + model principal still denies class B;; (e) forged write flag ona raw RPC caller (gate-spoof class #341/#44) stays denied. 프로젝트
- Docs:: `docs/quickstart.md` — new "**Argos works with any MCP/REST client — reads AND writes**" section:: OpenWebUI / Claude Desktop / Cursor registration JSON (no required env vars now), trust-model table (propose vs direct-write vs approve — what's on by default, when you'd turn it off), "make it read-only" one-liner (`ARGOS_API_READ_ONLY=1`). `docs/api/mcp.md` — env-table update to match. `README.md` — replace the `ARGOS_API_CAN_WRITE=1 ...` example with the new default story「+ READ_ONLY escape hatch」; verify no "read-only by default" phrasing remains anywhere (grep audit in the PR)。パラ
- `feature-specs/spec-09-external-api.md` — add prose note at top: "Amended by spec-11 (9/9): write tiers (A/C) now default-ON on loopback transports; REST writes = v1。" Keep spec-09 body neutral (it describes the facade spine, still accurate)。パラ
- `CHANGELOG.md` — entry。パ

## Acceptance criteria (verifiable, live证

1. Spawn `mcp_server` with **no env vars** → `memory_capabilities` lists include write ops (`memory_save` etc.)。パラ
2. `memory_propose` succeeds with content + idempotency key → candidate lands in review queue (not active memory,listable via candidate listing. パラ
3. `memory_save` succeeds (loopback)→ record is active + graph-indexed; verify via `memory_search` immediately after.パラ
4. `ARGOS_API_READ_ONLY=1` spawn → capabilities = read tier exactly (as today's default).パラ
5. `ARGOS_API_NO_LOOPBACK=1` + `CAN_WRITE` → `memory_save` denied (class C loopback gate intact).パラ
6. Model-principal approval: `PRINCIPAL_TYPE=model` + `CAN_FEEDBACK=1` → approve op denied, same via REST (no self-approval.パラ
7. `git grep -i "read.only.by.default"` on docs → zero stale phrasing (except explicit mentions of the old stance as history」.パ

## Why this is not a security regression

- The write surface already existed (spec-10/PR-2 #344[]; we are flipping gates, not adding capability — every write still passesthe facade (auth→ACL→validation→idempotency→audit→redaction), idempotency keys required, collection/update gates unchanged, graph isolation unchanged (spec-09 gate,且AND Hermes never routes through this surface (plugin provider path unchanged.パラ
- The trust boundary that matters most — "model cannot approve itself" — is untouched and explicitly re-tested (criterion 6).パ

## History

- 2/9 spec-09:: reads-first, "no speculative REST writes," loopback-first (v1 posture)
- 9/9 spec-11:: harness-agnostic read+write surface — writes ON by default on loopback transports;; READ_ONLY escape hatch;; REST writes promoted to v1.パラ