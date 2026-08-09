#!/usr/bin/env python3
"""Re-embed all memory records with the current embedding model.

When you switch embedding models (e.g. from multi-qa-MiniLM-L6-cos-v1 to
bge-small-en-v1.5), existing memory rows still hold embeddings from the OLD
model.  Cosine similarity only works within one model's space, so mixing old
and new embeddings produces garbage rankings.  This script re-embeds every
row from its stored ``content`` text using the model currently configured in
``hybrid_memory.json`` (or the default).

Run with Hermes STOPPED (the shared memory service holds an exclusive lock
on the DuckDB file):

    # 1. Stop Hermes completely (close desktop app, kill any gateway).
    # 2. Update hybrid_memory.json with the new model name (or rely on default).
    # 3. Run the migration:
    python reembed_memories.py
    # 4. Start Hermes.

Options:
    --dry-run          Show what would be re-embedded without writing.
    --model NAME       Override the model from config (useful for testing).
    --batch-size N     Rows per embedding batch (default: 64).
    --db PATH          Override the DuckDB path (auto-detected by default).

Safety:
    - Backs up the database to <db>.pre-reembed.bak before writing.
    - Skips rows whose content is empty.
    - Reports progress every batch.
    - Never deletes rows; only updates the embedding column.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


def _get_hermes_home() -> Path:
    """Resolve HERMES_HOME the same way the plugin does."""
    try:
        sys.path.insert(
            0, str(Path(__file__).resolve().parent.parent.parent / "hermes-agent")
        )
        from hermes_constants import get_hermes_home
        return Path(get_hermes_home())
    except Exception:
        pass
    env = os.environ.get("HERMES_HOME")
    if env:
        return Path(env)
    local = Path(os.path.expandvars(r"%LOCALAPPDATA%\hermes"))
    if local.exists():
        return local
    return Path(os.path.expanduser("~/.hermes"))


def _load_model_name(home: Path) -> str:
    """Read the configured embedding model from hybrid_memory.json."""
    config_path = home / "hybrid_memory.json"
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            model = cfg.get("local_embedding_model")
            if model:
                return str(model)
        except Exception:
            pass
    return "BAAI/bge-small-en-v1.5"


def _resolve_db_path(home: Path, override: str | None = None) -> Path:
    if override:
        return Path(override)
    config_path = home / "hybrid_memory.json"
    db_name = "hybrid_memory.duckdb"
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            db_name = cfg.get("database_filename", db_name)
        except Exception:
            pass
    return home / db_name


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-embed all memory records.")
    parser.add_argument("--dry-run", action="store_true", help="Show counts without writing.")
    parser.add_argument("--model", default=None, help="Override the embedding model.")
    parser.add_argument("--batch-size", type=int, default=64, help="Rows per batch.")
    parser.add_argument("--db", default=None, help="Override the DuckDB path.")
    args = parser.parse_args()

    home = _get_hermes_home()
    db_path = _resolve_db_path(home, args.db)
    model_name = args.model or _load_model_name(home)

    if not db_path.exists():
        print(f"ERROR: Database not found at {db_path}")
        return 1

    print(f"Database:  {db_path}")
    print(f"Model:     {model_name}")
    print(f"Batch size: {args.batch_size}")
    print()

    # Import dependencies.
    try:
        import duckdb
    except ImportError:
        print("ERROR: duckdb is not installed. Run: pip install duckdb")
        return 1

    # Open the database (Hermes must be stopped).
    try:
        conn = duckdb.connect(str(db_path))
    except Exception as e:
        msg = str(e).lower()
        if "being used by another process" in msg or "cannot access" in msg:
            print(f"ERROR: Database is locked. Stop Hermes completely and retry.")
            print(f"       Detail: {e}")
            return 1
        print(f"ERROR: Cannot open database: {e}")
        return 1

    # Count rows.
    try:
        total = conn.execute("SELECT COUNT(*) FROM memory_records").fetchone()[0]
    except Exception as e:
        print(f"ERROR: Cannot query memory_records: {e}")
        conn.close()
        return 1

    with_emb = conn.execute(
        "SELECT COUNT(*) FROM memory_records WHERE embedding IS NOT NULL"
    ).fetchone()[0]
    print(f"Total memory rows: {total}")
    print(f"Rows with existing embeddings: {with_emb}")
    print()

    if total == 0:
        print("Nothing to re-embed. Exiting.")
        conn.close()
        return 0

    if args.dry_run:
        print(f"[DRY RUN] Would re-embed {total} rows with model '{model_name}'.")
        conn.close()
        return 0

    # Load the embedder.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from embeddings import LocalEmbedder
    except ImportError as e:
        print(f"ERROR: Cannot import LocalEmbedder: {e}")
        conn.close()
        return 1

    print(f"Loading embedding model '{model_name}'...")
    embedder = LocalEmbedder(model_name)
    # Force load by embedding a probe.
    probe = embedder.embed("dimension probe")
    if not probe:
        print(f"ERROR: Embedding model '{model_name}' failed to load.")
        print("       Check that sentence-transformers is installed and the model name is correct.")
        conn.close()
        return 1
    dim = len(probe)
    print(f"Model loaded successfully (dim={dim}).")
    print()

    # Backup.
    backup_path = db_path.with_suffix(
        db_path.suffix + f".pre-reembed-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.bak"
    )
    print(f"Backing up database to {backup_path}...")
    # DuckDB holds a Windows file handle for the open connection, so close it
    # before copying the database. Reopen it after the backup is complete.
    conn.close()
    shutil.copy2(str(db_path), str(backup_path))
    conn = duckdb.connect(str(db_path))
    print("Backup complete.")
    print()

    # Re-embed in batches.
    batch_size = max(1, args.batch_size)
    re_embedded = 0
    skipped = 0
    offset = 0

    while offset < total:
        rows = conn.execute(
            "SELECT memory_id, content FROM memory_records "
            "ORDER BY memory_id LIMIT ? OFFSET ?",
            [batch_size, offset],
        ).fetchall()
        if not rows:
            break

        # Collect non-empty contents.
        ids_to_update = []
        texts_to_embed = []
        for memory_id, content in rows:
            if content and content.strip():
                ids_to_update.append(memory_id)
                texts_to_embed.append(content)
            else:
                skipped += 1

        if texts_to_embed:
            # Embed content (is_query=False — these are stored documents).
            embeddings = embedder.embed_batch(texts_to_embed, is_query=False)
            for mid, emb in zip(ids_to_update, embeddings):
                if emb:
                    conn.execute(
                        "UPDATE memory_records SET embedding = ? WHERE memory_id = ?",
                        [emb, mid],
                    )
                    re_embedded += 1
                else:
                    skipped += 1

        offset += len(rows)
        print(f"  Progress: {min(offset, total)}/{total} rows processed "
              f"({re_embedded} re-embedded, {skipped} skipped)")

    print()
    print(f"=== RE-EMBED COMPLETE ===")
    print(f"Re-embedded: {re_embedded}")
    print(f"Skipped:     {skipped}")
    print(f"Backup at:   {backup_path}")
    print()
    print("You can now start Hermes. Search will use the new embedding space.")

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
