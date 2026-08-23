# Spec 2 — `memory_why_not`: "why didn't you remember X?"

## 1. Why this exists
When the agent fails to recall something the user is sure it knows ("I told you about the power bill!"), the user currently has no way to interrogate the failure. They want to ask **why** a specific memory did not surface and get a deterministic, honest answer about which stage of the retrieval pipeline dropped it — not a guess.

Deterministic and free are hard requirements: no LLM call, no API spend, instant. This mirrors the diagnostic philosophy already in `C:\Users\<user>\Documents\Github\LongMemEval\Evaluation\diagnose_ranking.py` (which classifies benchmark misses as hit / catchable / retriever-miss) but applies it per-query on the real store, from inside the running plugin.

## 2. Tool contract
New provider tool: **`memory_why_not`**

Parameters (JSON schema, add a `WHY_NOT_SCHEMA` next to the other schemas in `__init__.py`, register in `get_tool_schemas` (~line 2311), dispatch in `handle_tool_call` (~line 2332)):
- `query` (string, required) — the question the user believes should have surfaced the memory.
- `target` (string, optional) — one of:
  - a `memory_id` (e.g. `mem-abc123...`), or
  - a content fragment / phrase that the user remembers ("bond payoff", "last month's meeting")
  - if omitted: report the top-ranked results and why they ranked where they did (a "retrieval explain" mode).
- `top_k` (int, optional, default 20, max 50) — the window the user cares about (their search was presumably run at some K; the store's default injected window is 96, `memory_search` caps at 50).

Response (JSON string, matching the existing tool style): a **blameline** — one verdict per candidate matching the target, plus a human summary line.

## 3. Architecture — do NOT touch the production path
The prod pipeline (`_hybrid_search`, store.py ~line 1068) stays byte-identical. Implement explanation as a **parallel diagnostic runner** in `store.py`:

```python
def explain_retrieval(self, query, target=None, top_k=20, ...) -> dict
```
that re-runs the same stage functions with tracing, using `suppress_retrieval=True` **always** (never let a diagnostic inflate `retrieval_count` — that pollutes ranking; this exact bug class has burned the project before).

Pipeline stages to trace (mirror `_hybrid_search` exactly, same order, same parameters):
1. **Query input** — record what the query actually was after any provider-level touching (context-aware prepend of recent user messages, query expansion via `query_expander.py` if top similarity < 0.3, `date_anchor.py` rewriting). Note: if the production search you're explaining *does* apply those, apply the same here; if `memory_why_not` is called standalone, document that the trace shows raw-query behavior unless the caller tells you otherwise. Keep it simple: trace with the raw query **plus** optionally note "provider may have prepended context".
2. **Text leg** — call `_text_search_raw(query, pool, ...)`; for each returned record record: token overlap count/score, rank in text list.
3. **Vector leg** — call `_vector_search_raw(emb, pool, ...)`; record raw cosine (`similarity`) per record, rank.
   - If the embedder is unavailable (text-only fallback), record `"vector_leg": "unavailable"` (do not error).
4. **Fusion** — `_rrf_fuse` (RRF k=20); record fused rank.
5. **Gates** — for the target record specifically, check in order and report the FIRST gate that kills it:
   - `status != 'active'` (quarantined/pending) → `"gate": "status", "value": <status>`
   - `user_scope` / `project_id` mismatch → `"gate": "scope"` (the calling store's scope; note if the memory belongs to another profile/project)
   - `valid_to IS NOT NULL` → `"gate": "superseded"`, plus `"superseded_by": <new id>` — the memory is an old version of a fact that still exists
   - `_is_expired(expires_at)` → `"gate": "expired", "expires_at": <value>` (requires Spec 1's expiry feature to be present; if not yet merged, omit this gate)
   - category exclusion / `category_filter` mismatch → `"gate": "category"`
6. **Importance adjustment** — `_importance_adjustment(...)` delta (raw cosine → final similarity).
7. **Pool entry** — did the record enter the candidate pool at all? Pool size follows `_hybrid_search`'s logic (`max(limit, reranker_top_n, phrase_pool)`; for explanation use a wide `limit` = `WHY_NOT_POOL_DEPTH` constant, default 200). If never retrieved → `"stage": "not_in_pool"` (retriever miss — embedding/text both failed to attract it).
8. **Final cutoff** — fused rank after adjustment vs `top_k` (and note the production injection window 96 for context).
9. **Not in database** — if the target phrase matches zero rows even by loose token overlap: `"verdict": "not_in_db"` + the top-5 nearest neighbors (content + cosine) so the user can see what the store thought it had.

Target resolution (for `target` given as a fragment, not an id): normalize (strip, lowercase) and match against `content` by exact substring first, then token-Jaccard ≥ 0.7; case-insensitive substring on `[:400]` truncated forms is fine. If multiple rows match, report each.

Verdict enumeration (one per matching record):
- `FOUND` (rank N ≤ top_k)
- `BELOW_CUTOFF` (rank N > top_k; include rank and how far above cutoff) — the memory was retrieved but lost the ranking race. Add: `"needs": "better ranking or higher top_k"`
- `FILTERED` (with the gate name + value + how to unblock: e.g. "status=quarantined → review/restore via memory_restore"; "superseded → the current version is mem-…")
- `NOT_IN_POOL` (retriever miss; suggest query rewording — tokens that DID match vs missed)
- `NOT_IN_DB` (nearest neighbors included)

Also return a short `"summary"` string (1–2 sentences, plain language, e.g. `"Found 'power bill' at rank 47 — above your 20 window. It scored 0.61 cosine but 14 newer memories outranked it. Raising top_k or pinning it would fix this."`).

Fail-soft: any exception inside explanation returns `{"verdict": "explanation_unavailable", "reason": "<short msg>"}` — never raises to the agent.

## 4. Implementation notes
- Keep all explanation logic in `store.py` as a sibling of `_hybrid_search` (it reuses the private stage functions) — or behind the `retriever.py` protocol if cleaner; do not fork the SQL.
- The provider branch is thin: validate args, call `explain_retrieval`, `json.dumps` the dict.
- Also ship a **CLI wrapper** `argos_plugin/why_not_cli.py` (mirror the style of the existing `review_pending.py` / `cleanup_memories.py` standalone scripts) so the feature can be exercised without the desktop app: `python why_not_cli.py --query "..." --target "..." --db <path>`.
- Do not add config keys unless something genuinely needs tuning; `WHY_NOT_POOL_DEPTH = 200` as a module constant is enough.
- Timestamps/format: match existing tool responses (JSON, `tool_error` for missing required args).

## 5. Tests (`tests/test_why_not.py`, imported by `tests/run_tests.py`)
Build a fixture store (in-memory/temp DuckDB — follow existing test fixtures in `test_hybrid_memory.py`) with ~15 known records engineered to cover every verdict:
1. `FOUND` — top-ranked target.
2. `BELOW_CUTOFF` — a target that lands at rank 9 with `top_k=5`, rank reported correctly.
3. `FILTERED/status` — quarantined record: gate named, restore suggested.
4. `FILTERED/superseded` — old version of a fact: points at the current version.
5. `FILTERED/expired` — record with past `expires_at` (needs Spec 1's surface, or create via store API directly): gate named.
6. `FILTERED/scope` — record with a different `project_id`.
7. `NOT_IN_POOL` — topically distant record: reports token hits/misses.
8. `NOT_IN_DB` — gibberish target: nearest neighbors returned, no crash.
9. Fragment-target resolution (substring + token-Jaccard).
10. `suppress_retrieval=True` honored — `retrieval_count` unchanged for every record after explanation runs.
11. Fail-soft: explain with a closed store → `explanation_unavailable`, no raise.
12. Existing suite still passes; SQL paths unchanged (diff `_hybrid_search` against git before/after — it must not be modified by this feature).

## 6. Acceptance criteria
- From a chat session: `memory_why_not(query="...", target="...")` returns a truthful, stage-accurate blameline for a real miss on the live store, and `FOUND` with a rank for a known hit.
- Zero API cost per call; response well under a second on a store of a few thousand memories.
- Production retrieval code path is untouched (git diff shows only additive code).
- CLI wrapper works standalone against a copied DB.
- All tests pass; deployed + verified per README rules.

## 7. Out of scope
- No LLM-based "why" (no natural-language diagnosis), no graph-traversal tracing (graph legs are off in prod search), no suggestions beyond the gate unblock hints, no auto-fixes (e.g. auto-pinning) — this tool diagnoses only.