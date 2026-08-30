# GOD-FILE REFACTOR PLAN — Argos (scout complete 2026-08-29, execution deferred)

**Status: READY TO EXECUTE — do not start until the open PR/issue work lands on `master`**
(feat/20-rpc-hardening #20 + multitenant cells #49). Then cut a fresh branch from `master`.

Michael reversed the 27/8 "low-ROI" judgement on 29/8 after the files grew.
Scope: **only `argos_plugin/store.py` + `argos_plugin/__init__.py`**. `graph.py` (77 KB) is
NOT in scope unless asked. Work on a NEW branch (`refactor/god-file-split`) — never on a
feature branch with open PRs; never commit the in-flight `memory_service.py` `_Tenant`
work or `tests/test_multitenant_cells.py`.

## Measured state (2026-08-29, branch feat/20-rpc-hardening, HEAD 7ab7fce, 1 ahead of master)

| File | Size | Lines | Contents |
|---|---|---|---|
| `store.py` | 196,936 B | 4,306 | module helpers (78-248), `MemoryRecord` (284), god-class `DuckDBMemoryStore` (403-4306) |
| `__init__.py` | 181,556 B | 3,950 | big module-level clusters (config 123-623, ambient 3499-3732, commands 3813-3950) + god-class `ArgosProvider` (646-3497) |

### Public surface that MUST survive (re-export shells)

From `store`: `DuckDBMemoryStore`, `MemoryRecord`, `VALID_CATEGORIES`, `sanitize_content`,
`_INJECTION_PATTERNS`, `default_grounding_for_write`, `GROUNDING_OBSERVED`,
`normalize_provenance`, `PROVENANCE_EXTERNAL`, `PROVENANCE_INTERNAL`, `rejection_key`.
Also top-level `import store as store_mod` (why_not_cli.py) — keep the try/except top-level
sibling-import pattern in every new module (same as the existing `.retriever`/`.value_extractor`).

From package root (`argos`/`argos_plugin`): `ArgosProvider`, `register`, plus
`_freshness_marker_for`, `_DATE_ANCHOR_RE`, `_build_file_activity_hint`,
`_build_location_hint`, `_build_weather_hint` (imported directly by tests).

Private attrs set externally (must stay plain instance attrs): `store._phrase_lift_alpha`,
`store._phrase_lift_pool`, `store._reranker_top_n` (memory_service.py `_Tenant`).

## Method (surgical rules — non-negotiable)

1. **Verbatim moves.** Bytes move identical into new modules. No renames, no reformatting,
   no drive-by fixes. Any genuine bug found → separate issue ticket, not folded in.
2. **Mixins, not rewrites.** `class DuckDBMemoryStore(StoreCoreMixin, StoreWriteMixin,
   StoreRetrievalMixin, StoreMaintenanceMixin)` and `class ArgosProvider(MemoryProvider,
   ProviderCoreMixin, ProviderRetrievalMixin, ProviderSessionMixin)` — every `self.x()`
   call site resolves via MRO unchanged.
3. **Bare-name audit — THE pitfall.** Moved code referencing module-level names
   (`_TEXT_STOPWORDS`, `_INJECTION_PATTERNS`, `logger`, config constants…) must resolve IN
   THE TARGET MODULE after the move. This exact class caused the 21/8 phrase-lift outage
   (`_PHRASE_STOPWORDS` bare-name NameError → all retrieval down). Audit every moved body
   for bare names; class attributes must be referenced via `self.`/`cls.` as today.
4. **AST name-equality proof.** Before the first move and after each stage: dump module-level
   names + class method sets (ast + dir), diff → must be identical (except `__module__`
   of classes, expected).
5. **One commit per cluster, suite green after each.** ~8 commits, each bisectable.
   Conventional style: `refactor(store): extract retrieval cluster → store_retrieval.py (behavior-neutral)`.
   Treat ONLY a final-state re-run as truth (mid-run edits contaminate: 88 phantom failures lesson).

## Target module map

```
argos_plugin/
  store.py               shell: imports + class composition + re-exports (~200 ln)
  store_common.py        _TEXT_STOPWORDS, injection regexes, _ci, sanitize_content,
                         provenance/grounding helpers, rejection_key, MemoryRecord,
                         VALID_CATEGORIES                                        (~370)
  store_core.py          DuckDBMemoryStore.__init__/_connect/_init_db/set_user_scope,
                         _is_lock_error/_now/_is_expired/_matches_scope, class consts (~400)
  store_retrieval.py     _row_to_record..search, _record_scale_metric..set_retriever,
                         _hybrid_search, _content_exists, scale metrics           (~760)
  store_write.py         remember, supersession trio, candidates (save/list/review),
                         evidence, quarantine/restore/feedback, update_memory,
                         history, chain membership, delete, tombstones, rejections (~1,140)
  store_maintenance.py   aliases, list_recent/by_category/memories, get_insights,
                         count, cleanup_junk, semantic dedup, consolidate,
                         explain_retrieval, state kv, eligible/ high-signal loads, close (~1,100)
  __init__.py            shell: imports + ArgosProvider composition + register() (~250)
  provider_core.py       _load_config cluster + _flag + ArgosProvider.__init__/name/
                         is_available/get_config_schema/save_config/initialize   (~600)
  provider_retrieval.py  system_prompt_block, referential/enrich/expand/record_*,
                         _search_memories, prefetch trio, chain-unfold cluster    (~850)
  provider_session.py    _review_candidate, sync_turn, sync worker, on_session_end,
                         distillation, on_session_switch, tools, backup, shutdown (~950)
  provider_ambient.py    freshness markers, ambient hint builders, _on_pre_llm_call,
                         _build_temporal_hint, /ilog /neg /revisit handlers       (~500)
```

## Test + gate recipes (validated 29/8)

Baseline: **617 tests collect** (598 prior + 19 new multitenant). CUDA recipe (validated):

### Baseline record (2026-08-29, branch feat/20-rpc-hardening HEAD 7ab7fce, NO changes made)

`pytest tests/ -q -n 4` (venv-cuda, HF_HUB_OFFLINE=1, fork-root PYTHONPATH): **610 passed, 7 failed in 21:22**.
All 7 failures = shared-service cluster socket timeouts (`SharedMemoryServiceError: ... timed out`,
`endpoint is unavailable` at service_client.py:203):
- test_backup.py::test_service_coordinated_backup
- test_candidate_review_integration.py::test_shared_memory_tool_delete_works_with_shared_client
- test_rpc_hardening.py::TestServerLocks::test_in_flight_counter_drains
- test_multitenant_cells.py × 4 (test_count_isolation, test_tombstone_isolation,
  test_unknown_user_id_falls_back_to_default, test_per_tenant_backup)

Classification: parallel xdist workers racing service spawns + the in-flight uncommitted
`memory_service.py` `_Tenant` work (#49). NOT caused by any refactor — recorded as the
pre-state. **Refactor pass criteria = the 610-pass set stays green; the 7 known failures
may persist (they are owned by the #20/#49 PR work, not this refactor).** Re-run the suite
with `-n 1` for the service cluster if the 7 need bisection (suite has a known single-process
deadlock ~70 tests in — see test-env-pythonpath-duality notes).

CUDA recipe (validated):

```bash
cd argos_plugin
PYTHONPATH="C:/Users/michael/AppData/Local/hermes/hermes-agent" HF_HUB_OFFLINE=1 \
  "C:/Users/michael/AppData/Local/hermes/hermes-agent/venv-cuda/Scripts/python.exe" \
  -m pytest tests/ -q -n 4 > "$LOCALAPPDATA/Temp/argos_tests_<stage>.log" 2>&1
```

MSYS rules: NEVER background a pipeline (`| tail`); use log redirection + `echo EXIT=$?`.
Never kill a background pytest job (orphans the python worker) — inspect with
`scripts/list-python-cmdlines.ps1` and kill only `-m pytest` cmdlines. No LLM calls in
tests when `agent` is stubbed; keep `HF_HUB_OFFLINE=1` for embeddings.

Self-corpus gate (REQUIRED before any live sync — store.py change rule; zero API spend):

```bash
cd argos_plugin
PYTHONPATH=… HF_HUB_OFFLINE=1 venv-cuda/Scripts/python.exe eval/run_gate.py \
  --snapshot eval/snapshots/20260826_170626_224920_4d612e0f \
  --gold eval/gold/gold_v1.jsonl \
  --out "$LOCALAPPDATA/Temp/gate_refactor.json" \
  --compare eval/snapshots/20260826_170626_224920_4d612e0f/gate_baseline.json
```

Gate assets are gitignored (personal content). Baseline: recall@5 .9618 / @20 .997 / @96 1.0,
MRR .8847, 995 probes, seed 42. PASS = no category recall@<max-k> drop > 1pp, overall ≤ 0.5pp,
MRR ≤ 0.01. Snapshot run required NO new snapshot — the frozen 26/8 pair is the ruler.

## Deploy (per SYNC_HANDOFF.md; live dir is %LOCALAPPDATA%\hermes\plugins\hybrid_memory)

1. Commit + push branch (origin is public — no personal data; plan doc stays untracked local).
2. md5-diff repo vs live; copy ALL drifted runtime modules INCLUDING the 9 new ones
   (new modules must land in live or the store/package import breaks on next restart).
3. Clear `__pycache__` in live. Do NOT copy tests/eval/utility scripts; preserve
   `skills/`, `*.duckdb`, `hybrid_memory_service.json`, `_mh_analysis.txt`.
4. Stop Hermes app → kill stale memory-service processes (cmdline match
   `hybrid_memory\memory_service.py`, BOTH venv + system pythons) → restart app.
5. Verify: md5 parity; service endpoint rewritten (fresh CreationDate); pytest against live
   copy; **real `store.search()` through the RPC path** (bare-name bugs are invisible to
   parity/health checks — the 21/8 lesson); recalled-memories smoke in a chat.
6. Rollback: re-copy previous files from git (branch commit) + restart.

## Deferred decisions (ask Michael)

- Deploy timing (tonight vs later) — app restart closes his live chat session.
- Whether the 27/8 "only justifiable cut = ambient-hint block" view changes anything — now moot: full plan greenlit.
- graph.py (77 KB) stays out of scope unless requested.