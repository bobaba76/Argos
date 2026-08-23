# Handoff: sync the dev repo, the remote, and the live install

Three places hold this code. Only a manual step connects them. Do all three
so they stop disagreeing. **This file was rewritten 2026-08-23** — the old
version's topology (dev = the Hermes fork) is stale and must not be used.

## Topology (measured 2026-08-23)

| Tag | Path / URL | Type | Note |
|-----|-----------|------|------|
| **D (dev repo, canonical)** | `C:\Users\<user>\Documents\Github\Argos` | full git repo | **All code changes and commits happen here.** origin = `github.com/bobaba76/Argos`. |
| **R (remote)** | `github.com/bobaba76/Argos` (branch `master`) | GitHub repo | Source of truth on GitHub. |
| **F (fork, LEGACY)** | `C:\Users\<user>\Documents\Github\Hermes` | git repo, stale | Old hermes-agent fork; **8 commits behind** D at last check (HEAD `b3afce7` vs D's `44eca22+`). Its `hybrid_memory_plugin/` files are old fork-era versions. **Never copy runtime modules from here. Do not commit here.** Keep only as reference. |
| **C (live install)** | `%LOCALAPPDATA%\hermes\plugins\hybrid_memory` | **NOT a git repo** | Plain file **copy** of D's `hybrid_memory_plugin/`. This is what the running Hermes app loads. Editing D alone does nothing live until you re-copy here and restart. |

There is **no automatic link** from D/R to C. C is a copy-by-hand directory
(no `.git`). Build in D → push to R → copy to C → restart Hermes.

**Do NOT use F as the copy source.** The old handoff's Step 3 copied from F;
that is exactly how C ended up with 17 stale fork-era files on 2026-08-23
while the repo had moved on. Copy from D only.

## CRITICAL — the include_expired signature-drift bug (fixed 2026-08-23)

`SharedMemoryStore.search()` (the RPC client) had a closed signature missing
`include_expired`, while the provider passes it on EVERY search (memory_search
tool + per-turn prefetch). Result: TypeError on every search in
`shared_service` mode **the moment a freshly-synced client loads** (i.e. after
the next restart). Introduced by `b872aa4`; fixed in the repo with a
regression test (`tests/test_shared_service.py::test_shared_service_search_forwards_provider_kwargs`).

**Rule: never copy `service_client.py` into live unless the repo state also
has the matching `__init__.py` provider conventions — run pytest first.** When
in doubt: `cd hybrid_memory_plugin && HF_HUB_OFFLINE=1 <venv-python> -m pytest tests/test_shared_service.py -q`.

## Step 1 — D: commit and push ("if the repo has been touched")

```bash
cd /c/Users/<user>/Documents/Github/Argos
git status --short        # if empty, skip the commit below
git add <your changed files>     # NEVER bare `git add -A` without reviewing
git commit -m "..."
git push origin master
```

Local `ENGINEERING_NOTES.md` (exploration log) is intentionally **untracked** —
never commit it (it's a working note, not a product doc).

## Step 2 — R: confirm it's current

```bash
git fetch origin && git status -sb   # expect "## master...origin/master" with nothing ahead/behind
```

## Step 3 — C: re-copy the plugin source from D (NOT from F)

C is NOT a git repo — file copy, not git pull:

```bash
REPO="/c/Users/<user>/Documents/Github/Argos/hybrid_memory_plugin"
LIVE="$(cygpath -m "$LOCALAPPDATA")/hermes/plugins/hybrid_memory"   # forward slashes, no backslashes!
```

> **MSYS trap (hit 2026-08-23):** `$LOCALAPPDATA` is `C:\Users\...` (backslashes).
> Passing `"$LOCALAPPDATA/hermes/..."` to `md5sum`/`cmp` mangles the path and
> produces bogus hash prefixes (`\c259...`), so every file reports DIFF even
> when the bytes are identical. Always `cygpath -m` the home path first.

Re-derive drift, then copy the changed runtime modules:

```bash
(cd "$REPO" && md5sum *.py *.yaml) | sort -k2 > /tmp/repo.md5
(cd "$LIVE" && md5sum *.py *.yaml) | sort -k2 > /tmp/live.md5
join -j 2 -a 1 -a 2 -e MISSING -o '1.2,1.1,2.1' /tmp/repo.md5 /tmp/live.md5 \
  | awk '$2!=$3{print $1}'
# For every <file> listed (runtime modules only), run:
cp "$REPO/<file>" "$LIVE/<file>"
```

**Do NOT** do a blanket `rsync -av --delete`. Rules:
- Copy only the **runtime modules** (top-level `.py` + `config_schema.py` + `plugin.yaml`) that differ.
- Do **NOT** touch/delete live-only runtime artifacts: `skills/`, `*.duckdb` (the live memory DB), `hybrid_memory_service.json` (both copies), `__pycache__/`, `_mh_analysis.txt`.
- Do **NOT** copy dev artifacts into live: `tests/`, `eval/`, `*_backfill*.py`, `cleanup_memories.py`, `dump_memories.py`, `migrate_gateway.py`, `rebuild_graph.py`, `reembed_memories.py`, `review_pending.py`, `why_not_cli.py`, `run_tests.py`. These do not belong in the installed plugin.
- Legacy files already sitting in live root (`run_tests.py`, `test_hybrid_memory.py` — leftovers from an older layout): leave them alone, they're harmless.

## Step 4 — Restart so the new code actually loads

Hermes loads the plugin modules at startup. Copying files under a running app
does **not** hot-reload them — the gateway process keeps running the OLD code
until restarted. A restart of the Hermes desktop app (or the `gateway run`
process) is required.

**The shared memory SERVICE is a separate process and survives app restarts.**
It was spawned on 2026-08-22 19:44 and will keep running the old
`memory_service.py`/`store.py`/`graph.py`/`embeddings.py` from memory even
after the app restarts (clients re-connect to whatever answers on the port).
To force it onto the new code, **after** the app has fully exited, kill the
stale service processes and let the next app start spawn fresh ones:

```powershell
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'hybrid_memory\\memory_service\.py' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

(There may be TWO matching processes — the venv python and the system python
run in parallel on this machine. Kill both.)

## Step 5 — Verify (verify, don't assume)

1. **Hash parity**: `md5sum "$REPO/store.py" "$LIVE/store.py"` etc. — the synced
   files must now MATCH between D and C.
2. **Service health**: after restart, the endpoint file
   `%LOCALAPPDATA%\hermes\hybrid_memory_service.json` gets rewritten — check
   that a new `memory_service.py` process is running (CreationDate = now).
3. **Plugin tests against the live copy** (from the LIVE dir):
   `cd "$LOCALAPPDATA/hermes/plugins/hybrid_memory" && python -m pytest -q 2>&1 | tail -5`
   (expect 0 failures; `tests/` lives in live because an earlier copy brought it in).
4. **Smoke test**: open a chat, ask "what do you remember about me?" — the
   Recalled Memories block must appear. Then `memory_search` via a tool call
   must return results (this is the call that used to TypeError).
5. **md5 the drifted set again**: re-run Step 3's diff — expect no runtime
   module differences.

## Compact version

**D** (commit+push; run shared-service tests first — include_expired drift
guard) → **R** (fetch, confirm) → **C** (md5-diff, copy only drifted runtime
modules from D, preserve `skills/`+`*.duckdb`+`hybrid_memory_service.json`+`_mh_analysis.txt`,
don't copy `tests/`/`eval`/utility scripts) → **kill stale memory_service
processes** (they outlive app restarts) → **restart Hermes** → **md5 parity +
service CreationDate fresh + pytest + memory_search smoke test**.