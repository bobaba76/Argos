# Argos feature specs — handoff for a fresh session

Three independent feature specs for the **Argos** memory plugin (internal name: `hybrid_memory`), a persistent-memory `MemoryProvider` for the Hermes agent. Each spec is self-contained; you do not need any prior context. Build them in any order — they touch mostly separate files, except **Spec 1 (TTL expiry)** which edits the core `store.py`, so land its changes before merging other work into that file.

| Spec | File | Feature |
|---|---|---|
| 1 | `spec-01-ttl-expiry.md` | Best-before dates (TTL expiry tiers) for memories |
| 2 | `spec-02-why-not.md` | `memory_why_not` tool — "why didn't you remember X?" |
| 3 | `spec-03-self-corpus-eval.md` | `eval_self_corpus.py` — regression-test retrieval on the real store, no API spend |

---

## Codebase map (read this first)

**Two copies of the plugin exist. Do not confuse them:**

1. **DEV / fork** — `C:\Users\<user>\Documents\Github\Hermes\hybrid_memory_plugin\`
   You build, edit, and test here. The test suite and eval harness import this copy.
2. **LIVE / deployed** — `%LOCALAPPDATA%\hermes\plugins\hybrid_memory\`
   The running Hermes desktop app executes THIS copy. **Edits to the fork are not live until copied here** and the app is restarted. Verify the deployed copy after deploying (grep for your new code, or compare file hashes) — claiming a fix is live without this check has burned us before.

**Repo hygiene (hard rule):** `github.com/bobaba76/Argos` is a PUBLIC, sanitized repo. Never commit personal names (use `Alex` / `Sam` as fixture names), never commit real user memory dumps, never `git add -A` without scanning the staged diff. Keep generated eval outputs (`*.jsonl`) out of any public repo. This spec folder lives in the private working fork — you may commit the specs there freely (they contain no personal data).

**Config system:** `config_schema.py` declares the config surface (rendered in the Hermes desktop UI panel); the runtime reads/writes `hybrid_memory.json`. Follow the existing `ProviderField(key, label, kind, default, description, info, inline, group)` pattern exactly. Existing groups: Embeddings, Retrieval, Extraction, Maintenance, LLM, Storage, Chains, Routing, Temporal. Add an "Expiry" group for Spec 1.

**Core files:**
- `store.py` (~2,660 lines) — DuckDB-backed store; the only file with SQL. Key anchors (verified):
  - `memory_records` DDL: `expires_at VARCHAR` column exists (line ~223), plus `valid_from` / `valid_to` / `superseded_by` (versioning), `status DEFAULT 'active'`
  - `_DEFAULT_TTL_DAYS = {"context_note": 30, "event": 180, "goal": 180}` (line ~50)
  - `_is_expired(expires_at)` (line ~428)
  - `_text_search_raw` (line ~500) and `_vector_search_raw` (line ~563) — both already filter `AND (expires_at IS NULL OR expires_at > ?)` in SQL and re-check `_is_expired` in Python
  - `_hybrid_search` (line ~1068) — the single retrieval entry point: parallel text+vector, RRF fusion (`_RRF_K = 20`), optional cross-encoder blend (0.8/0.2), phrase-lift (off), importance adjustment. **There is a retrieval-accounting side effect: `_record_retrieval` bumps `retrieval_count`; diagnostics/eval MUST pass `suppress_retrieval=True`** or they pollute ranking.
  - save path (~line 1310): INSERT already writes `expires_at`, and auto-applies a TTL when `durability == "temporary"` and no explicit expiry is given (this is the dormant TTL machinery)
  - `update_memory` (~line 1898): new-version INSERT already threads `expires_at` (explicit value or carries the old one forward)
  - Read paths at lines ~979, ~2363, ~2375, ~2593 already exclude expired records
- `__init__.py` (~3,170 lines) — provider; tool schemas (`SEARCH_SCHEMA` line ~145, `SAVE_SCHEMA` ~175, `UPDATE_SCHEMA` ~206, `DELETE_SCHEMA` ~227, `FEEDBACK_SCHEMA` ~335, plus GRAPH/CANDIDATE/RESTORE/MAINTENANCE/CHAIN/FETCH_FULL), `get_tool_schemas` (~2311), `handle_tool_call` dispatch (~2332, a long `if tool_name == ...` chain). Tool responses: `json.dumps(...)`, errors via `tool_error(...)`.
- `retriever.py` — a thin seam (`DuckDBRetriever` delegates to `_hybrid_search`); new retrieval features can also live behind this protocol.
- `date_anchor.py` — regex extraction of date expressions ("N days ago", "last weekday", "31 Dec") used by temporal retrieval; reuse its patterns, don't duplicate them.
- `query_expander.py` — lazy LLM query rewrite (fires only when top similarity < 0.3), SHA-256 cache, fail-soft. The pattern to copy for any new LLM-touching code.
- `reviewer.py` + `confirmation.py` — the proposal review flow (candidates → auto-review → user confirmation). Spec 1 hooks expiry suggestions here.
- `tests/run_tests.py` — plain-Python suite (1270 lines): run `python tests/run_tests.py` from the plugin dir, or `python -m pytest tests/test_hybrid_memory.py -v`. All tests must pass before deploy.
- `eval/` — LongMemEval harness (`run_eval_provider.py`), chain-unfold probes. Spec 3 lives here as `eval/eval_self_corpus.py`.
- `C:\Users\<user>\Documents\Github\LongMemEval\Evaluation\diagnose_ranking.py` — existing ranking-miss diagnostic on benchmark caches (hit ≤ cutoff / catchable / retriever-miss classification). Spec 3 mirrors its hit/miss classification but on the real store.

**Environment facts:**
- Windows; Python 3.11; DuckDB (single-writer file `hybrid_memory.duckdb` — diagnostics should open a second connection `read_only=True`, or copy the file, never write from a second process).
- Embedder: `BAAI/bge-small-en-v1.5` (384-dim, CUDA). Loading an HF embedder by name does a network HEAD-check that can hang for minutes — always load from the local cached snapshot path, exactly as the production harness already resolves models.
- Timestamps: ISO-8601 UTC strings (see `_now()` / `_is_expired`). Never local time in storage.
- Cost discipline: **no LLM calls by default** in any new code. Optional LLM features must be behind explicit flags and respect a day budget.

**Cross-cutting acceptance rules (all three specs):**
1. New behavior is OFF by default wherever the spec says so; with the feature off, retrieval results must be bit-identical to today.
2. No memory is ever hard-deleted by any new feature — everything is reversible/filter-at-read.
3. All new code passes `suppress_retrieval=True` on any search it performs internally.
4. Tests exist for every behavior and pass in the fork before deployment.
5. Deploy = copy changed files into `%LOCALAPPDATA%\hermes\plugins\hybrid_memory\`, restart the Hermes desktop app, verify the deployed copy actually contains the change, and describe what you verified.

---

## Suggested dev order
1. **Spec 3** (fully isolated, greenfield script, lowest risk)
2. **Spec 1** (core store edits — biggest regression surface, run full suite)
3. **Spec 2** (touches store.py but as an additive diagnostic; safe after 1 lands)