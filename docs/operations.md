# Operations

How to back up, verify, and restore the Argos memory store — and how to put the whole thing on a weekly schedule.

## How backups work

Argos ships a service-coordinated backup tool (`argos_plugin/backup_cli.py` + `backup.py`). It is a **logical** backup built on DuckDB's `EXPORT`/`IMPORT DATABASE` (FORMAT PARQUET) — no filesystem snapshots, no elevation, no platform-specific tricks. It works identically on Windows, macOS, and Linux.

- The export runs **inside the memory service** — the sole database writer — so a backup is consistent by construction. The service runs `CHECKPOINT` (flushing the write-ahead log into the main database), then exports the schema, one `.parquet` file per table, and a `manifest.json` (per-table row counts, schema, source size, timestamp, and DuckDB version).
- Every snapshot is **verified before it passes**: each parquet file is reopened in a fresh connection and re-counted against the manifest. The manifest is anchored by a SHA-256 sidecar file stored *outside* the snapshot directory, so a snapshot cannot forge its own integrity data.
- Old snapshots are pruned **after** the new one verifies — never before.
- Backups make **zero LLM calls and zero network calls**, and they **never stop or restart a running service**. If no service is running, the CLI starts one for the duration of the backup and stops only the one it started.
- The backup surface is deliberately local: it is not exposed through the MCP/REST facade. This CLI is the supported path.

## Running a backup

```bash
python backup_cli.py backup [--home PATH] [--dst-root PATH] [--retention N]
```

Run it with the same Python environment that runs the plugin (it needs the plugin's dependencies).

- `--home` overrides `HERMES_HOME` (default: the `HERMES_HOME` environment variable, else `~/.hermes`; on Windows `%LOCALAPPDATA%\hermes`).
- `--dst-root` sets the destination. Default: `$HERMES_HOME/backups/memory`. You can also set `backup.dst_root` in `hybrid_memory.json` (see [Configuration](configuration.md)).
- `--retention` keeps the N most recent snapshots (default `6`).

Each run writes a timestamped snapshot directory, for example:

```text
memory-20260910-140000/
  schema.sql
  memories.parquet
  memory_evidence.parquet
  ...
  manifest.json
memory-20260910-140000.manifest.sha256     ← integrity anchor (sibling of the snapshot dir)
```

Exit codes: `0` success, `1` error (the CLI fails loudly on any mismatch).

!!! tip "Point the destination somewhere durable"
    `dst_root` should live on storage that survives disk failure — an external drive or a network share. A same-disk snapshot protects you from mistakes and corruption, not from losing the disk.

## Scheduling a weekly snapshot

Any scheduler works. The requirements: run the backup on a weekly cadence, and alert when the exit code is not `0`.

**Linux / macOS (cron):**

```text
# Every Sunday at 03:00 — adjust paths to your install
0 3 * * 0 cd /path/to/argos_plugin && /path/to/python backup_cli.py backup --retention 8 >> /var/log/argos-backup.log 2>&1
```

**macOS (launchd):** create a LaunchAgent with a `StartCalendarInterval` entry (`Weekday` + `Hour`) that runs the same command, and set `StandardErrorPath` so failures are captured.

**Windows (Task Scheduler):** create a weekly trigger that runs the same command with your plugin's Python (`...\python.exe backup_cli.py backup --retention 8`).

On Hermes hosts you can alternatively run the command as a managed scheduled job (`hermes cron`) if you want the result delivered through the agent.

## Listing and verifying snapshots

```bash
python backup_cli.py list [--home PATH] [--dst-root PATH]
python backup_cli.py verify --snapshot-dir PATH
```

- `list` prints the snapshots (newest first) with timestamp, table count, and total rows.
- `verify` re-checks a snapshot **without restoring it** — manifest integrity, file hashes, and row counts. Run it after copying snapshots to cold storage, or periodically as a hygiene check.

## Restoring

!!! warning "The memory service must be stopped"
    Restore is standalone. It checks for a live service endpoint (and fails on a locked database), so stop the service — and any app that owns it — first. `--force` skips the endpoint check; use it only when you know the service is down.

```bash
# 1. Stop the memory service.
# 2. Restore the most recent snapshot:
python backup_cli.py restore --latest [--home PATH]

#    ...or restore a specific one:
python backup_cli.py restore --snapshot-dir PATH [--home PATH]

# 3. Rebuild the relationship graph (derived data):
python backfill_graph.py --home <HERMES_HOME>

# 4. Restart Hermes — the memory service respawns on demand.
```

What restore does: imports the snapshot into a temporary database, verifies row counts against the manifest, then **atomically swaps** it into place of the live database. Any mismatch fails loudly and leaves the live data untouched.

The Kùzu relationship graph is derived data — rebuild it from the restored memories with `backfill_graph.py`. The DuckDB snapshot includes `entity_aliases`, so the important manual graph data survives the round trip.

!!! tip "Practice the drill"
    A backup you have never restored is a hypothesis. Verify a snapshot, then do a restore drill on a spare machine (or a copy of the data) so the runbook is muscle memory before you actually need it.

## What is (and isn't) in a snapshot

- **In:** the DuckDB database — memories, evidence rows, version chains, aliases, and relational state — plus a manifest that describes it.
- **Not in:** the Kùzu graph files (derived — rebuild with `backfill_graph.py`), embedding model caches (re-downloaded on demand), and configuration such as `hybrid_memory.json` (version those settings separately).
