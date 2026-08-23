# Spec 1 — TTL expiry tiers ("best-before dates" on memories)

## 1. Why this exists
Some memories are only true for a while ("tenant moves out at the end of June", "promo runs until Friday", "project X is blocked this week"). Today those facts sit in the store forever and surface in retrieval long after they stopped being true, cluttering search results and occasionally making the agent answer with stale facts.

Goal: give memories a **best-before date** (expiry). Expired memories quietly stop being *read* (excluded from every recall path), are **never deleted**, and can be revived or inspected on demand. The user's design preference: expiry over deletion (deleting a true fact is quiet data loss; expiry is honest and reversible). Tiers = different shelf lives for different kinds of facts.

## 2. Current state — MOST OF THIS IS ALREADY BUILT. Do not rebuild it.
Verified in the fork at `C:\Users\<user>\Documents\Github\Hermes\argos_plugin\`:

Already exists:
- **Schema:** `memory_records.expires_at VARCHAR` column (store.py ~line 223). No migration needed.
- **Record field:** `MemoryRecord.expires_at` (store.py ~line 62/79).
- **Expiry predicate:** `_is_expired(expires_at)` (store.py ~line 428) — picks up ISO-8601 UTC, treats unparseable as not expired.
- **Retrieval filtering:** `_text_search_raw` (~line 534) and `_vector_search_raw` (~line 603) both filter in SQL: `AND (expires_at IS NULL OR expires_at > ?)` (param = `self._now()`), and re-check `_is_expired` in the Python loop. Other read paths (~lines 979, 2363, 2375, 2593) also exclude expired.
- **Save-path TTL machinery (dormant):** the memory INSERT (~line 1310) writes `expires_at` from `record_payload.get("expires_at")`, and when `durability == "temporary"` with no explicit expiry, auto-applies `_DEFAULT_TTL_DAYS` by category (store.py ~line 50: `{"context_note": 30, "event": 180, "goal": 180}`).
- **Update path:** `update_memory` (~line 1960) threads `expires_at` — explicit new value, else carries the old record's value forward.

Why it's still dormant: **no tool surface sets `durability`/`expires_at`** (the `memory_save` schema only exposes content/category/tags), **no config exposes the TTL map**, **no user confirmation flow exists**, and **nothing reports expiry state**. Also: the `as_of` (point-in-time version query) path applies the expiry filter against *now* instead of *as_of* — a correctness bug to fix (see §4).

## 3. What to build

### 3.1 Config surface — `config_schema.py`, new group `"Expiry"`
```python
ProviderField(key="expiry_enabled", label="Expiry enabled", kind=KIND_BOOL, default="false",
    description="Allow memories to carry a best-before date (expires_at). Off = current behavior exactly.",
    inline=True, group="Expiry"),
ProviderField(key="expiry_ttl_days", label="TTL days by category", kind=KIND_TEXT, default='{"context_note":30,"event":180,"goal":180}',
    description="Shelf life in days applied when a memory is saved/approved as temporary. JSON map category→days.",
    group="Expiry"),
ProviderField(key="expiry_default_days", label="Fallback TTL days", kind=KIND_NUMBER, default="90",
    description="Used when the saved category is not in the TTL map and the memory is temporary.",
    group="Expiry"),
ProviderField(key="expiry_auto_suggest", label="Suggest expiry on review", kind=KIND_BOOL, default="false",
    description="When the reviewer approves a context_note/event/goal candidate, propose an expiry (deterministic, no LLM). User confirms before it sticks.",
    inline=True, group="Expiry"),
```
Validation: `expiry_ttl_days` must parse as a JSON object of ints > 0 (fall back to the default on bad input, log a warning — match the codebase's fail-soft style). `expiry_default_days` clamped to [1, 3650]. All four keys also need their runtime defaults mirrored in the plugin's config-loading code (find where existing defaults are read, e.g. `max_injected_items`, and follow that pattern; the schema is the UI declaration — the runtime must also consult the values).

### 3.2 Tool schemas + dispatch — `__init__.py`
- `SAVE_SCHEMA` (~line 175): add optional params
  - `durability`: `"durable" | "temporary"` (default `"durable"` — current behavior unchanged; only `temporary` triggers the TTL map)
  - `expires_at`: ISO-8601 UTC string (or `null`). **Explicit `expires_at` wins over the TTL map.** Document this in the schema description.
- `UPDATE_SCHEMA` (~line 206): add optional `expires_at` (ISO string **or `null` to clear** — clearing revives the memory; this is the documented "uncancel" path).
- `SEARCH_SCHEMA` (~line 145): add optional `include_expired: bool` (default false).
- Wire them in `handle_tool_call` (~line 2332):
  - `memory_save` branch: pass `durability` and `expires_at` through to the store save call (find the exact call and the store method's existing `durability`/`payload` handling — the INSERT at ~1310 already honors `record_payload["expires_at"]`; make sure the tool args reach it).
  - `memory_update` branch: pass `expires_at` through (store already accepts it).
  - `memory_search` branch: pass `include_expired` through to `store.search` → `_hybrid_search` → both raw searches (add the keyword with default False; when True, drop the expiry clause/check).
- Do **not** add a fifth expiry-management tool; `memory_update(expires_at=...)` + `memory_search(include_expired=True)` cover set/clear/inspect.

### 3.3 Store wiring — `store.py`
- `_hybrid_search` (~1068), `search` (~981), `_text_search_raw` (~500), `_vector_search_raw` (~563): add `include_expired: bool = False` keyword (thread through; when True, omit the SQL expiry clause and skip the `_is_expired` post-check).
- Save path (~1310): no change needed (already writes it) — but READ it to confirm the `durability` value arriving from the tool falls through to the `temporary` branch.
- `update_memory` (~1898): confirm the `expires_at` parameter is already threaded (it is, per §2); only the schema/dispatch layer needs exposing.
- **Fix the `as_of` bug:** in both raw searches the expiry SQL uses `?` bound to `self._now()` even when `as_of` is set. Point-in-time queries should see a memory that expired *after* `as_of` but is expired *now* — history is history. Change the expiry comparison value to `as_of` when `as_of` is provided (`AND (expires_at IS NULL OR expires_at > ?)` with the as_of param), else `now`. Add a regression test proving a memory expired yesterday is still returned by `as_of` = 2 days ago.

### 3.4 Expiry suggestions at review time (only when `expiry_auto_suggest=true`)
- In the candidate auto-review path (`reviewer.py`), after a candidate is approved, if its category is in the configured TTL map and the content does not express a fixed historical date-past (reuse `date_anchor.py` to detect date expressions; if the content is a past-dated event like "valentine's 2026", do NOT suggest expiry — historical events stay), attach a **proposed** `expires_at = now + ttl_days[category]`.
- The proposal is **shown to the user for confirmation** (same flow as sensitive/contextless candidates today — see `confirmation.py`; the user confirms every mutation). Only on user confirmation does the approved memory get its `expires_at`. No silent expiry assignment.
- Deterministic only — no LLM involved.

### 3.5 Visibility via `memory_maintenance` (no new tool)
Extend the existing `memory_maintenance` output (`MAINTENANCE_SCHEMA` ~line 351 / its handler) to also report:
- `expired_count`: active-status rows with `expires_at <= now` (these are filtered from retrieval but still stored — the number tells the user the feature is doing something)
- `expiring_soon_count`: rows expiring within 7 days
- `expired_revivable_count`: same as expired_count (for clarity in the report)
Do not change what maintenance quarantines; expiry never auto-deletes, never auto-quarantines.

## 4. Behavior rules & edge cases
- **Non-destructive:** expiry only changes *read* behavior. Rows keep `status='active'`; nothing is deleted; `memory_chain`/`memory_fetch_full` still show expired versions (they are version-history viewers — verify they don't filter expiry; if they do, that's a bug: history is history).
- **Revive:** `memory_update(memory_id, expires_at=null)` clears the date. Document it in the schema.
- **Explicit vs TTL:** explicit `expires_at` > category TTL > nothing. `durability="temporary"` without map entry falls back to `expiry_default_days`.
- **Supersede interplay:** an expired *current* version is invisible to retrieval (correct); its predecessor versions stay `valid_to`-closed and equally invisible. Creating a new version via `memory_update` carries `expires_at` forward unless overridden (existing behavior — keep).
- **Category policy (defaults, off-by-default):** only `context_note` / `event` / `goal` get TTL suggestions. `personal_fact` / `preference` / `relationship` / `insight` never get auto-expiry. The explicit `expires_at` param can still be used on any category.
- **Timezones:** store UTC (`_now()` style); `_is_expired` already compares UTC.
- **Backfill:** none. Existing memories are untouched; expiry only applies to new/changed records. Do not write a sweep that assigns expiry to old rows.
- **Include-expired search:** returns expired rows ranked normally, so the user can audit "what did I know then".
- **Graph:** verify graph read paths (~2363/2375/2593) and graph-expansion during retrieval also skip expired (they already call `_is_expired` — keep it consistent when `include_expired=True` is... **do not** wire `include_expired` into graph legs; graph follows store semantics).

## 5. Tests (add to `tests/test_hybrid_memory.py` or a new `tests/test_expiry.py` imported by `tests/run_tests.py`)
1. Save with explicit `expires_at` → row stored with it; search excludes it after the date passes (use a past date to avoid sleeping).
2. Save `durability="temporary"` with no explicit expiry → `expires_at = now + ttl[category]`.
3. Explicit `expires_at` overrides the TTL map.
4. Category not in map + temporary → `expiry_default_days`.
5. `memory_update(expires_at=...)` sets on the new version; `expires_at=null` clears (revives).
6. `memory_search` default excludes expired; `include_expired=True` returns them.
7. `as_of` regression: expired-yesterday memory visible at `as_of` = two days ago; invisible at `as_of` = today.
8. `memory_maintenance` reports the three new counts.
9. Config off (`expiry_enabled=false`) + no new parameters used → the exact SQL/rank paths produced today (assert search result order identical to a stored expected list on the fixture).
10. Reviewer suggestion flow: `expiry_auto_suggest=true`, content "tenant moves out at the end of June" (future-dated event) → proposal created; user confirm → `expires_at` set. Content "valentine's 2026" (past event) → no suggestion.
11. Full existing suite still passes.

## 6. Acceptance criteria
- A user/agent can save a memory with a best-before date, a temporary fact auto-expires via the TTL map (when enabled), expired memories disappear from every retrieval path, reappear with `include_expired=True`, and are revived with `memory_update(expires_at=null)`.
- `expiry_enabled=false` (default) = byte-identical behavior to today.
- Nothing is ever deleted; maintenance reports counts; review-time suggestions require user confirmation.
- All tests pass; deployed copy verified (see README deploy rules).

## 7. Out of scope
- No automatic deletion, no cron sweeps, no bulk backfill, no graph-leg `include_expired`, no UI beyond the config panel, no per-memory UI calendar picker (a plain ISO string param is enough).