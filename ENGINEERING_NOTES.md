# Argos Engineering Notes — exploration log

> Started 2026-08-23 (session: chief code engineer familiarization).
> Log everything seen + things worth investigating. Append as we go.

## Topology (measured live, 2026-08-23 — SYNC_HANDOFF.md is STALE on this)

Places this code lives:

| Tag | Path / URL | Reality (measured) |
|-----|-----------|--------------------|
| A (dev fork) | `C:\Users\<user>\Documents\Github\Hermes\hybrid_memory_plugin` | Has `.git` at Hermes root (recheck why `git -C` silently failed). Plugin files are mostly **Aug 6–9 vintage** — the handoff's "A is where commits happen" no longer matches: recent commits (Aug 22–23) were made in the Argos repo itself. |
| B (remote) | `github.com/bobaba76/Argos` (branch master) | Origin of the local Argos repo. |
| B-local (dev, ACTIVE) | `C:\Users\<user>\Documents\Github\Argos` | **The actual dev repo now.** All Aug 22–23 commits live here. Ahead of origin by 1 (see below). |
| C (live install) | `%LOCALAPPDATA%\hermes\plugins\hybrid_memory` | Plain copy, no git. Mixed state: 12 files = current repo state (mtimes **Aug 23 16:30**), 17 files = Aug 6–9 vintage, 2 legacy test files present only here. |

### Active sync happening during exploration
- 16:30 today: 12 runtime files copied into live **from the Argos repo** (not from the fork, per SYNC_HANDOFF's Step 3 — handoff procedure is stale).
- 16:38 today: commit `44eca22` (LICENSE → LICENSE.md rename, BSL 1.1) landed in the Argos repo **while** I was reading it. Another agent/process is actively working — be careful not to clobber.

### Drift inventory (md5, 2026-08-23 ~16:45)
- **Repo vs live**: 17 files differ + 2 live-only legacy tests (`run_tests.py`, `test_hybrid_memory.py`).
  - Differing: backfill_graph, cleanup_memories, confirmation, date_anchor, dump_memories, embeddings, hermes_file_activity, intent_router, memory_service, migrate_gateway, plugin.yaml, rebuild_graph, reembed_memories, retriever, review_pending, routing, why_not_cli.
  - Matching (i.e. already live, copied 16:30): __init__, config_schema, distillation, egress, extractor, graph, hermes_weather, query_expander, reviewer, service_client, store, temporal_subcall.
- **Fork vs live**: exactly the complement — the 12 repo-matching files differ fork↔live, the 17 differ repo↔live match fork↔live. So live == fork on the 17, == repo on the 12.
- **Repo vs fork**: ALL 30 files differ. Fork is a much older snapshot + has `_mh_analysis.txt` (a PowerShell error log from a duckdb one-liner — diagnostic leftover, also in live; not committed).

### Fork mystery resolved (17:00)
- The fork IS a git repo at `b3afce7` (clean tree, only untracked `SYNC_HANDOFF.md`). Earlier `git -C` failure was the MSYS path-conversion quirk — native git needs `C:/...` paths, not `/c/...`.
- `b3afce7..9772f71` (fork HEAD → origin/master): **33 files changed, +2336/−212** — one big sweep: egress gate (11 files), Perseus P0 fixes (10), benchmark appendix (9), evidence trail, config_schema fixes, 800-char comments. That's the whole story: **the fork is a stale checkout of the SAME history** (7 commits behind origin/master, 8 behind the Argos repo), NOT a divergent code line.
- Live's 17 stale files == fork's b3afce7 versions (copied Aug 6–9, unchanged until the sweep). Today 16:30 someone copied the OTHER 12 from the repo line.
- `git rev-list origin/master...HEAD` in fork = `0 0` — fork's local `origin/master` ref is stale (never fetched); don't trust it.

### Resolved TODOs
1. ~~"Fork = second code line"~~ → false; same history, stale checkout. Note in SYNC_HANDOFF rewrite: dev = Argos repo; fork is 8 commits behind and its plugin dir must NOT be used as a copy source.
2. ~~git -C failure~~ → MSYS path quirk, nothing wrong with the repo.
3. Live mid-sync: 17 files still stale vs repo (list above). Complete sync repo→live, then restart Hermes.
4. `44eca22` committed but **unpushed** — origin/master is 1 behind.

## Doc drift found while reading
- `MEMORY_SYSTEM.md` says "Provider: 12 tools" (twice: architecture header + Files listing) — README correctly says **fourteen** tools. Table lists 14. Fix doc.
- README links `[LICENSE](LICENSE)` → file was renamed to `LICENSE.md` in 44eca22. Broken link (git handles rename; GitHub URL does not).
- SYNC_HANDOFF "Current known delta" (5 modules) is outdated: store.py, graph.py, query_expander.py, hermes_weather.py now match; the real delta is 17 modules (list above).

## Repo shape (stats)
- `hybrid_memory_plugin/` ≈ 26.6k LOC Python, ~30 runtime modules + tests/ + eval/.
- Recent commit stream (Aug 22–23): BSL license, 800-char cap comments, update_memory chain fix, config_schema UI/runtime default alignment, optional config_file, manifest removal from tree, evidence trail (Perseus review pt 2), egress gate + report (pt 6), approval-boundary invariant (pt 5), benchmark reproducibility appendix, P0 fixes, docs sync, dream on session-switch, P4.2 audit fixes, P4.2 distillation, ...
- So a "Perseus review" recently drove fixes (P0 → pt2/pt5/pt6). Possibly more review points outstanding — check PerseusVault / review threads for remaining items.

## General docs read
- README.md: 14 tools; honest-numbers section (93% chain-unfold; 99.6% answer-bearing recall to top-96; 82% temporal; 70.4% LongMemEval_S; 2nd behind Perseus in builder's own cross-system compare); "no native local-LLM yet" limitation; BSL 1.1.
- MEMORY_SYSTEM.md: architecture (provider → retrieval pipeline → DuckDB + Kùzu), config reference, storage modes (shared_service RPC default vs direct), extraction pipeline, distillation gates, ambient context, prefetch.

## Next steps
- [ ] Recheck fork git state (stderr visible).
- [x] Read `__init__.py` (provider surface) — hooks, tools, retrieval pipeline entry.
- [ ] Read `store.py`, `retriever.py`, `embeddings.py` (the 16:30-synced set + the stale set).
- [ ] Skim tests + eval harness; check CI/test status headroom.
- [ ] Read skill `hybrid-memory-engineering` reference list (known pitfalls).
- [ ] Check `_mh_analysis.txt` origin (leave in place; live-only artifact).

# P0 FINDING (verified empirically 2026-08-23 ~17:30) — search signature drift breaks search+prefetch on next restart

**Bug:** `SharedMemoryStore.search()` (service_client.py:178) has a CLOSED signature —
`(query, limit, exclude_categories, category_filter, project_id, as_of, suppress_retrieval)` —
**no `include_expired` param, no `**kwargs`**. The provider's `_search_memories()`
(__init__.py:1611) **always passes `include_expired=include_expired`**, and it's used by
- the `memory_search` tool (handle_tool_call ~2598) and
- `_start_prefetch` (~1897) — i.e. the per-turn memory injection itself.

**Consequence:** in `storage_mode: shared_service` (the production default — live config
confirms it), every search → `TypeError: SharedMemoryStore.search() got an unexpected
keyword argument 'include_expired'`.

**Why it works today:** the running gateway (PID 41876/30112, started 14:49) loaded the
plugin before the 16:30 partial sync; the pre-sync client/provider combination predates
`include_expired`. The 16:30 sync put the new `__init__.py` + `service_client.py` into
live → **next Hermes restart = memory_search tool AND injected memories both break.**

**Introduced by:** `b872aa4` "feat: TTL expiry tiers, memory_why_not diagnostic, self-corpus
eval" — `git log -S include_expired` shows it was added to `__init__.py` and NEVER to
`service_client.py`.

**Why tests missed it:** `tests/test_shared_service.py` has ONE search test:
`store.search("shared memory service", limit=5)` — bare args, no kwargs. Store-level
tests use `DuckDBMemoryStore` (open `*args/**kwargs`), so the closed RPC-client signature
is never exercised with the provider's real calling convention. → same class of bug as
the skill's `references/rpc-client-search-signature-drift.md`.

**Direct evidence:**
- probe (live client, read-only): `search("...", limit=3)` OK n=3; `search(..., include_expired=False)` → TypeError; `include_expired=True` → TypeError.
- probe to running service (raw socket with `component:"store"`): server accepts all three shapes OK (including include_expired) — server side is FINE in repo code (memory_service.py:135 forwards it).
- DuckDBMemoryStore direct: `search()` has `*args/**kwargs` → forwards to retriever → `_hybrid_search(include_expired=...)` ✓ works.

**Fix (small, do before syncing live/restarting):**
1. Add `include_expired: bool = False` to `SharedMemoryStore.search()` and forward it in the `_rpc.call("store", "search", ...)` args.
2. Add a shared-service test that searches with the provider's full kwarg set (include_expired at minimum).
3. Then complete the 17-file live sync + restart (the restart is what exposes the bug).

**Other RPC client calls verified OK** (checked call sites against client sigs):
`_query_side_chain_lookup` (suppress_retrieval ✓), `_expand_and_merge` (category_filter/project_id/suppress_retrieval ✓), `_build_temporal_hint` (limit ✓), memory_* tools all use client methods that exist. `include_expired` is the only missing kwarg.

## Code-reading log (provider, __init__.py 479-3467)
- Config loading: `_load_config()` with per-key try/except clamps; chain-unfold knobs, phrase lift, reranker (lazy load), shared vs direct store wiring. `initialize` logs store count, graph on/off, embeddings status ("pending" if embedder unavailable — suspicious label, look into `is_available` semantics later).
- `system_prompt_block()`: STATIC text only (byte-stable for prompt caching) — dynamic state goes through prefetch. Good practice, keep.
- `_is_referential_query` + `_enrich_query_with_context`: pronoun/context prepend; patterns list `\bthat\b`, `\bit\b` etc.
- `_expand_and_merge`: RRF k=20 merge of sub-queries; `rrf_scores` normalized to 0-1 into `similarity`.
- `_search_memories`: candidate_limit = min(512, max(limit, limit*4)); gates query expansion on `raw_similarity` (good — avoids importance-boost contamination); graph boost alive only when `graph_aware_retrieval`; traversal disabled in prod config (`graph_traversal_enabled: "false"` — matches eval A/B).
- **Smell:** leftover debug scaffolding in `_search_memories`: `TRAVERSAL-DBG` logs + hardcoded `mem-4d83cf06` startswith checks (lines ~1721, 1817-1821) — per-record string compare on every graph pass. Remove or gate behind a debug flag.
- **Smell:** `_build_coder_directive` hardcodes user-machine repo names (salesdash, miser, codebrain, "documents\\github"...) in public code; docstring admits "keep in sync". Consider moving to config.
- `_build_temporal_hint` (pre_llm_call): creates a NEW SharedMemoryStore RPC client per turn `SharedMemoryStore(home, user_id="default_user")` — hardcoded user_id (multi-user risk) + new client per turn (RPC ensure_service + health check per turn?). Check cost; consider caching client on the provider.
- Hooks/register: clean fail-soft structure; /ilog + /revisit commands; insight-log skill registration; pre_llm_call returns {"context": ...} + optional {"model","provider"} for the router (P2A). Router sub-call path (`router_subcall_enabled`) = trimmed temporal sub-call, stays on cheap answerer — matches earlier cost lesson.
- `handle_tool_call`: all 14 tools; memory_save → remember + graph index; memory_update graph re-index on new version + removes old id; memory_delete promotes predecessor; candidate_review supersede → graph cleanup; feedback `incorrect` → graph detach; maintenance → consolidate + graph cleanup of quarantined; chain modes arc/versions/diff (difflib), evidence joined via get_evidence_batch. Clean.
- `prefetch`: `_run()` = confirmation block + `_search_memories` + optional chronological/date-anchor re-sort (P2B/P2B2 flags) + content cap (800) + `[id: ...]` exposure for memory_fetch_full. Fail-soft.

## Store reading log (store.py)
- `search()` → retriever seam (default DuckDBRetriever = pure delegation to `_hybrid_search`). Scale metrics recorded every call (rolling p95 → HYBRID_MEMORY_SCALE warning at thresholds).
- `_hybrid_search`: vector (via `_vector_search_raw`) + text (`_text_search_raw`, ILIKE) in parallel → RRF → optional cross-encoder blend (0.8/0.2) → raw_similarity snapshot → optional exact-phrase lift (alpha; on in prod: 0.25/pool 200) → feedback+recency → P2C dedup demote → truncate. `suppress_retrieval` respected ("eval self-pollution" fix).
- `_content_exists` (dedup): exact → substring → semantic (string-cast vector param — the DuckDB per-row bind pitfall documented; they use the fixed CAST form ✓).
- `remember()`: dedup default on; TTL via `expires_at` semantics (_NOT_PROVIDED/None/ISO); durability default: context_note/event/goal → temporary.
- Scale-metric plumbing only on direct store; provider reads via RPC `get_scale_metrics` (service-side store instance tracks its own).
- Retriever Protocol: `Retriever` docstring says implementations must honor scope/status/validity/expiry semantics.

## Still to read
- store.py update_memory / review_candidate / consolidate / explain_retrieval (chain semantics — recent fix 0f210a2)
- graph.py (1696 lines), egress.py, distillation.py, intent_router.py, date_anchor.py, temporal_subcall.py, extractor.py, reviewer.py, query_expander.py, config_schema.py, plugin.yaml
- tests (test_hybrid_memory 3835 lines — spot-check), eval harness (run_eval_provider 280, eval_self_corpus 858, dry_run_distillation 126)
- skill hybrid-memory-engineering (103KB, over cap — memory says curator must trim) + its references (rpc-client-search-signature-drift.md is THE doc for the P0 above)
- Run test_shared_service.py under venv python to confirm current pass state (HF_HUB_OFFLINE=1)

# WORKLOG 2026-08-23 (evening) — remaining work executed

## 1. P0 FIXED — include_expired client signature drift
- `service_client.py`: `SharedMemoryStore.search()` gained `include_expired: bool = False` AND forwards it in `_rpc.call("store","search",...)`.
- `tests/test_shared_service.py`: added `test_shared_service_search_forwards_provider_kwargs` — saves 1 durable + 1 expired memory, searches with the provider's EXACT calling convention (`category_filter=None, project_id=None, suppress_retrieval=True, include_expired=False|True`), asserts active-only vs active+expired membership. NOTE: the file was fully rewritten (write_file) after two fuzzy-patch manglings — verify the whole file reads cleanly before judging diffs.
- Guard for future drift: the skill's preventive habit (test BOTH search impls with identical kwargs) is now followed for the shared path.

## 2. Debug scaffolding removed (`__init__.py` `_search_memories`)
- `TRAVERSAL-DBG` ×2 → generic `logger.debug("traversal: ...")` / `"graph injectable ids: ..."`.
- `mem-4d83cf06` startswith checks (the INDWE inject/reject logs) → removed; the `sim >= floor` logic untouched. Grep now returns 0 hits.

## 3. Doc drift fixed
- `MEMORY_SYSTEM.md`: "12 tools" → "14 tools" (architecture diagram + Files listing).
- `README.md`: `[LICENSE](LICENSE)` → `[LICENSE.md](LICENSE.md)` (rename landed in 44eca22).

## 4. SYNC_HANDOFF.md rewritten (6.9KB)
- New reality: D = Argos repo (canonical), R = origin, F = Hermes fork LEGACY (never a copy source), C = live install (copy, not git).
- Includes: the include_expired rule ("never copy service_client.py unless tests green"), exact md5-diff + targeted-copy procedure, do-NOT-copy list (tests/eval/utility scripts), live-only artifacts to preserve (skills/, *.duckdb, hybrid_memory_service.json, _mh_analysis.txt), **the service-process-outlives-restart gotcha** + the PowerShell kill command, and verify steps.

## 5. Skill reference updated
- `hybrid-memory-engineering/references/rpc-client-search-signature-drift.md`: added the Windows gotcha — the memory_service PROCESS survives app restarts (measured: 8/22 19:44 service still serving 8/23); kill stale `memory_service.py` processes after app exit; verify endpoint file rewritten. The skill already documented this bug class (even named include_expired) — my fix matches its prescription exactly.
- Skill SKILL.md review: 102,662 bytes (>100k cap), 49 references, inline Common Pitfalls 1-53 largely duplicate reference docs → curator trim candidate (move inline sections to refs, keep index + core invariants). NOT trimmed in this session (large editorial job).

## 6. Reading pass completed (remaining modules)
- graph.py: `memory_ids_for_query` (lexical bridge, term→edge→memory_id scoring) + `traversal_memory_ids` (BFS hop-weighted 2/1, typed relations only, excludes generic mentions/related_to, `require_specific_seed` needs ≥2 non-concept seeds, hub "user" node excluded, junk-node guard for Kuzu IN-list literals). Well-guarded, matches eval A/B decisions.
- distillation.py `run_distillation`: gated on `distillation_enabled`, dry-run flags, cluster-size floor, per-cluster merge via provider boundary... solid.
- egress.py (full read): SITES-gated LLM call sites, local_only blocks all, PII-identifier gate → pending_user_confirmation (fail-soft). Report script reads LIVE config. Solid.
- temporal_subcall.py: trimmed sub-call via `agent.auxiliary_client.call_llm`, evidence = created_at-prefixed records. Note: for VERSIONED memories the DATE that matters may be valid_from/updated_at, not created_at — potential nuance, low priority.
- intent_router `route_answerer`: P2A, over-routing guarded per 21/8 lessons.
- plugin.yaml: hooks on_turn_start/sync_turn/on_session_end/on_session_switch; pre_llm_call registered at runtime.
- eval harness + tests: NOT yet run this session — full pytest suite is running now (was still running at notes time).

## 7. Live sync — 17 stale runtime files copied repo→live + md5 parity verified
- ALL 17 copied (backfill_graph, cleanup_memories, confirmation, date_anchor, dump_memories, embeddings, hermes_file_activity, intent_router, memory_service, migrate_gateway, plugin.yaml, rebuild_graph, reembed_memories, retriever, review_pending, routing, why_not_cli). **0/17 md5 mismatches** after the fix below.
- **New MSYS trap found:** `md5sum "$LOCALAPPDATA/hermes/..."` (backslash path) mangles → hash printed with `\` prefix + `\\`-doubled path, every file false-DIFFs. Fix: `LIVE="$(cygpath -m "$LOCALAPPDATA")/hermes/plugins/hybrid_memory"`. Documented in SYNC_HANDOFF.
- Live-only artifacts preserved: skills/, duckdb, kuzu (actually kuzu = single ~90MB FILE at hermes root), hybrid_memory_service.json ×2, _mh_analysis.txt, legacy run_tests.py/test_hybrid_memory.py (left as-is).

## 8. Tests + commit
- `tests/test_shared_service.py` standalone: **4/4 PASSED (34s)** incl. new `test_shared_service_search_forwards_provider_kwargs` — expired memory hidden with include_expired=False, visible with True, through the REAL RPC service. P0 fix empirically verified end-to-end.
- Committed `ee0cd85` (6 files: service_client, test_shared_service, __init__, MEMORY_SYSTEM, README, SYNC_HANDOFF). ENGINEERING_NOTES.md stays untracked.
- **Full pytest suite: 390 passed in 577s (9m36s), ZERO failures** — includes the new regression test.
- **Pushed to origin**: `44eca22..ee0cd85 master -> master` (license commit + fix set now on GitHub).
- ⚠️ While working: `hybrid_memory_plugin/intent_router.py` became `M` (uncommitted) — the OTHER agent is editing it live. Do NOT touch/commit; it's in-progress work. (It was already synced to live earlier today from the committed state.)

## Still open / next session
- [x] Full pytest suite green: **390 passed / 0 failed** (9m36s).
- [x] Fix set committed (`ee0cd85`) + **pushed** (with license commit `44eca22`); origin/master clean.
- [x] Restart #1 (17:09): endpoint rewritten, fresh service PID 49904 ✓ — BUT app still threw the TypeError in-app. Root cause: the app started BEFORE the ~17:21 copy of edited `service_client.py`/`__init__.py` into live (they were in the "already synced 16:30" set my drift list skipped, yet I edited them after). Disk parity re-established at ~17:21: full sweep ALL 30 runtime modules = 0 mismatches.
- [ ] **Restart #2 REQUIRED**: running app holds the old client class in memory; disk is fixed but process isn't. After restart: verify memory_search works in-app.
- LESSON for SYNC_HANDOFF: a drift list computed BEFORE edits goes stale AFTER edits — always finish with a full-parity sweep across ALL runtime modules and re-copy any file touched since.
- [ ] Curator: trim SKILL.md to <100k (move inline pitfalls to references).
- [ ] `_mh_analysis.txt` origin check (left in place).
- [ ] Consider `_build_temporal_hint` per-turn SharedMemoryStore client → cache on provider; hardcoded `user_id="default_user"` (multi-user).

## Post-restart verification log
- Restart #1 (17:09): service PID 49904 fresh (CreationDate 17:09:36) ✓, endpoint rewritten ✓, but in-app `memory_search` TypeError'd → app loaded plugin before the late copy of the two edited files. Fixed on disk (~17:21), awaiting restart #2.