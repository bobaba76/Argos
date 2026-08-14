#!/usr/bin/env python3
"""Detect whether a hybrid_memory DB is pre- or post-embedding-model-upgrade.

The plugin's embedding model was upgraded from multi-qa-MiniLM-L6-cos-v1 to
BAAI/bge-small-en-v1.5. Stored vectors are only comparable within one model's
space, so a DB that was never re-embedded silently degrades memory search.

Run with Hermes STOPPED (the shared memory service locks the DuckDB file):

    python embedding-check.py [--home HERMES_HOME]

Never writes anything. Prints a VERDICT line:
    POST  -> vectors match bge-small-en-v1.5 (all good)
    PRE   -> vectors match the old model (re-embed needed)
"""
import argparse
import json
import os
import sys
from pathlib import Path

import duckdb
import numpy as np

BGE = "BAAI/bge-small-en-v1.5"
OLD = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"


def cos(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", default=None, help="HERMES_HOME (default: env or %%LOCALAPPDATA%%\\hermes)")
    args = ap.parse_args()

    home = Path(args.home) if args.home else Path(
        os.environ.get("HERMES_HOME") or (Path.home() / "AppData/Local/hermes")
    )
    cfg_path = home / "hybrid_memory.json"
    cfg = {}
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"WARNING: could not read {cfg_path}: {exc}")
    cfg_model = str(cfg.get("local_embedding_model") or "").strip()
    print(f"config local_embedding_model: {cfg_model or '(unset -> default BAAI/bge-small-en-v1.5)'}")

    db = home / str(cfg.get("database_filename", "hybrid_memory.duckdb"))
    if not db.exists():
        print("VERDICT: no-memory-db — fresh install, nothing to check")
        return 0
    try:
        con = duckdb.connect(str(db), read_only=True)
    except Exception as exc:
        print(f"VERDICT: cannot-open-db — is Hermes still running? ({exc})")
        return 2

    rows = con.execute(
        "SELECT content, embedding FROM memory_records "
        "WHERE content IS NOT NULL AND embedding IS NOT NULL LIMIT 2"
    ).fetchall()
    if not rows:
        print("VERDICT: no-embedded-rows — nothing to check")
        return 0

    from sentence_transformers import SentenceTransformer

    bge = SentenceTransformer(BGE)
    old = SentenceTransformer(OLD)
    scores = [(cos(v, bge.encode(c)), cos(v, old.encode(c))) for c, v in rows]
    avg_b = float(np.mean([s[0] for s in scores]))
    avg_o = float(np.mean([s[1] for s in scores]))
    print(f"cos vs bge-small  : {avg_b:.3f}")
    print(f"cos vs multi-qa   : {avg_o:.3f}")

    if avg_b > avg_o:
        print("VERDICT: POST-re-embedding — vectors match bge-small-en-v1.5. All good.")
        if cfg_model and "bge" not in cfg_model.lower():
            print(f"WARNING: config pins '{cfg_model}' but DB is bge — set "
                  f"local_embedding_model to '{BGE}' (or remove the key).")
        return 0

    print("VERDICT: PRE-re-embedding — vectors match the OLD model.")
    print("ACTION: with Hermes closed, re-embed now:")
    print(f'  "{sys.executable}" "{home / "plugins/hybrid_memory" / "reembed_memories.py"}"')
    if cfg_model and "bge" not in cfg_model.lower():
        print(f"  (first set local_embedding_model to '{BGE}' in {cfg_path}, or remove the key)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
