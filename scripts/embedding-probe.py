"""Probe which embedding model produced stored vectors (read-only).

Compares stored memory embeddings against fresh embeddings from the
candidate models via cosine similarity. High similarity to one model
identifies the embedding space the DB was built with.
"""
import duckdb
import numpy as np

DB = os.path.expandvars(r"%LOCALAPPDATA%\hermes\hybrid_memory.duckdb")
BGE = os.path.expandvars(r"%LOCALAPPDATA%\hermes\models\bge-small-en-v1.5")
MQA = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"

con = duckdb.connect(DB, read_only=True)
cols = [r[0] for r in con.execute("SELECT column_name FROM information_schema.columns WHERE table_name='memory_records'").fetchall()]
print("columns:", cols)
if "embedding" not in cols:
    raise SystemExit("no embedding column")

rows = con.execute(
    "SELECT content, embedding FROM memory_records "
    "WHERE content IS NOT NULL AND embedding IS NOT NULL LIMIT 3"
).fetchall()

from sentence_transformers import SentenceTransformer
bge = SentenceTransformer(BGE)
mqa = SentenceTransformer(MQA)

def cos(a, b):
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

for content, vec in rows:
    vb = bge.encode(content)
    vm = mqa.encode(content)
    print(f"content: {content[:50]!r}")
    print(f"  cos vs bge-small   : {cos(vec, vb):.3f}")
    print(f"  cos vs multi-qa    : {cos(vec, vm):.3f}")
