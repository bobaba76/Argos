# Handoff: sync the fork, the remote repo, and the live install

Three separate places hold this code, and only a manual step connects them.
Do all three so they stop disagreeing.

## Topology (get this right first)

| Tag | Path / URL | Type | Note |
|-----|-----------|------|------|
| **A. Fork (dev)** | `C:\Users\michael\Documents\Github\Hermes` | full git repo | origin = `github.com/bobaba76/Argos`. This is where all code changes and commits happen. |
| **B. Remote** | `github.com/bobaba76/Argos` (branch `master`) | GitHub repo | Source of truth on GitHub. Currently the code you're reading here. |
| **C. Live install** | `%LOCALAPPDATA%\hermes\plugins\hybrid_memory` | **NOT a git repo** | A plain file **copy** of A's `hybrid_memory_plugin/`. This is the copy the running Hermes app actually loads. Editing A on its own does nothing live until you re-copy to here and restart Hermes. |

There is **no automatic link** from A/B to C. C is a copy-by-hand directory
(no `.git`). Build in A → push to B → copy to C → restart Hermes.

## Current known delta (measured 2026-08-22)

C is stale on **5 runtime modules** (file hashes differ vs A):

- `store.py`      — includes the new `as_of` expiry fix (point-in-time history queries now filter expiry against the query date, not today)
- `graph.py`
- `hermes_location.py`
- `hermes_weather.py`
- `query_expander.py`

`__init__.py` and the rest already match. Don't trust this list forever — re-derive it (Step 3) before copying.

## Step 1 — A fork → make sure it's committed and pushed ("if the fork has been touched")

```bash
cd /c/Users/michael/Documents/Github/Hermes
git status --short        # if empty, A is clean — skip the commit below
git add <your changed files>
git commit -m "..."
git push origin master
```

**Right now A is dirty** with your (Devin's) in-progress spec-03 work:
- `hybrid_memory_plugin/eval/eval_self_corpus.py` (untracked)
- `hybrid_memory_plugin/tests/test_eval_self_corpus.py` (untracked)
- `.gitignore` (modified)
Decide if those are ready to commit; if yes, commit+push. Do NOT commit the other person's files that you didn't touch.

## Step 2 — B remote → confirm it's current

After Step 1's push, B is the source of truth:

```bash
git fetch origin && git status -sb   # expect "## master...origin/master" with nothing ahead/behind
```

Anything in A that isn't on B needs to be pushed (Step 1) before Step 3, because Step 3 copies **from A** to C.

## Step 3 — C live install → re-copy the plugin source from A

C is NOT a git repo, so this is a file copy — NOT git pull.

```bash
REPO="/c/Users/michael/Documents/Github/Hermes/hybrid_memory_plugin"
LIVE="$LOCALAPPDATA/hermes/plugins/hybrid_memory"
```

Re-derive drift, then copy the changed runtime modules:

```bash
(cd "$REPO" && md5sum *.py *.yaml) | sort -k2 > /tmp/repo.md5
(cd "$LIVE" && md5sum *.py *.yaml) | sort -k2 > /tmp/live.md5
join -j 2 -a 1 -a 2 -e MISSING -o '1.2,1.1,2.1' /tmp/repo.md5 /tmp/live.md5 \
  | awk '$2!=$3{print $1}'
# For every <file> listed, run:
cp "$REPO/<file>" "$LIVE/<file>"
```

**Do NOT** do a blanket `rsync -av --delete` of the whole directory. Rules:
- **Copy only the runtime modules** (top-level `.py` + `config_schema.py` + `plugin.yaml`) that differ.
- **Do NOT touch / delete live-only runtime artifacts:** `skills/`, `*.duckdb` (the live memory DB), `hybrid_memory_service.json`, `__pycache__/`.
- **Do NOT copy dev artifacts into live:** `tests/`, `eval/`, `*_backfill*.py`, `cleanup_memories.py`, `dump_memories.py`. These do not belong in the installed plugin.
- **Legacy test files already sitting in live root** (`run_tests.py`, `test_hybrid_memory.py` — leftovers from an older layout where the repo now uses `tests/`): leave them alone, they're harmless.

## Step 4 — Restart so the new code actually loads

Hermes loads the plugin modules at startup. Copying files under a running app
does **not** hot-reload them. A restart of the Hermes desktop app (or the
`serve` gateway process) is required for C to run the new modules.

## Step 5 — Verify

```bash
# hash parity after copy — these must now MATCH between REPO and LIVE:
md5sum "$REPO/store.py" "$LIVE/store.py"
# run the plugin's own tests against the live copy (from the LIVE dir):
cd "$LOCALAPPDATA/hermes/plugins/hybrid_memory"
python -m pytest   # expect 0 failures attributable to your change
```

Do not claim "live is synced" until Step 5's hashes match and the app has
restarted. Verify, don't assume.

## Compact version
**A** (commit+push your dirty eval files) → **B** (fetch, confirm current) → **C** (md5-diff, copy only the 5 drifted runtime modules, preserve `skills/`+`.duckdb`+`hybrid_memory_service.json`, don't copy `tests/`/`eval`/`*_backfill*`) → **restart Hermes** → **md5 parity + pytest**.