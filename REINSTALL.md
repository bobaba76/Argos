# Reinstall / Migration Guide

## Reinstalling the memory plugin

If you need to wipe and reinstall (corrupted DB, schema change, fresh start):

1. **Stop Hermes** (all instances — CLI, gateway, desktop app).

2. **Back up your data** (optional but recommended):
   ```bash
   cp -r ~/.hermes/hybrid_memory.duckdb ~/.hermes/hybrid_memory.duckdb.bak
   cp ~/.hermes/hybrid_memory_kuzu ~/.hermes/hybrid_memory_kuzu.bak
   cp ~/.hermes/hybrid_memory_kuzu.wal ~/.hermes/hybrid_memory_kuzu.wal.bak 2>/dev/null || true
   ```

3. **Delete the databases**:
   ```bash
   rm ~/.hermes/hybrid_memory.duckdb
   rm ~/.hermes/hybrid_memory.duckdb.wal
   rm -f ~/.hermes/hybrid_memory_kuzu ~/.hermes/hybrid_memory_kuzu.wal
   ```

4. **Delete the config** (to get fresh defaults):
   ```bash
   rm ~/.hermes/hybrid_memory.json
   ```

5. **Restart Hermes**. The plugin recreates everything on first run.

## Migrating to a new machine

1. **Install Hermes** on the new machine.

2. **Copy the plugin**:
   ```bash
   cp -r argos_plugin/ ~/.hermes/plugins/hybrid_memory/
   ```

3. **Copy your data**:
   ```bash
   cp ~/.hermes/hybrid_memory.duckdb  <new-machine>:~/.hermes/
   cp ~/.hermes/hybrid_memory_kuzu* <new-machine>:~/.hermes/
   cp ~/.hermes/hybrid_memory.json     <new-machine>:~/.hermes/
   ```

4. **Copy the embedding model** (or let it re-download):
   ```bash
   cp -r ~/.hermes/models/bge-small-en-v1.5/ <new-machine>:~/.hermes/models/
   ```

5. **Start Hermes** on the new machine. The plugin picks up the existing databases.

## Rebuilding the graph

If the Kuzu graph gets out of sync with the DuckDB store (e.g. after a manual DB edit). Note: in
current releases the graph is a **single file** named `hybrid_memory_kuzu` (plus a
`hybrid_memory_kuzu.wal` while the service holds it), not a directory:

```bash
python ~/.hermes/plugins/hybrid_memory/rebuild_graph.py --home ~/.hermes
```

This re-indexes all active memories into the graph. Supports `--dry-run` to preview without writing.

## Re-embedding memories

If you change the embedding model or need to regenerate vectors:

```bash
python ~/.hermes/plugins/hybrid_memory/reembed_memories.py --home ~/.hermes
```

## Switching storage modes

To switch between `direct` and `shared_service`:

1. Stop Hermes.
2. Edit `~/.hermes/hybrid_memory.json`:
   ```json
   { "storage_mode": "shared_service" }
   ```
   or
   ```json
   { "storage_mode": "direct" }
   ```
3. Restart Hermes.

The same DuckDB file is used in both modes — no data migration needed.
