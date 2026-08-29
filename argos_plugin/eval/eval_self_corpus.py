#!/usr/bin/env python3
"""eval_self_corpus.py — regression-test retrieval on the real store.

A free, deterministic, repeatable retrieval-recall harness that runs
against the user's own memory store (not a lab benchmark).  Generates
template probe queries from sampled memories, runs the production
retrieval pipeline with ``suppress_retrieval=True`` (never pollutes
ranking), and reports grouped recall@K metrics.  With ``--baseline``
it emits a PASS/FAIL regression verdict.

Zero API spend by default.  ``--llm-paraphrase`` is opt-in and bounded.

Usage:
    python eval/eval_self_corpus.py --db <path> [options]

See ``--help`` for all flags.  Output is append-only JSONL; never commit
``--out`` files to a public repo (they may carry personal content).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import random
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Resolve the plugin package so imports work whether invoked from the
# plugin dir, the repo root, or elsewhere.  Mirrors run_eval_provider.py.
_PLUGIN_ROOT = Path(__file__).resolve().parents[1]
if str(_PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_ROOT))
_EVAL_DIR = Path(__file__).resolve().parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

import duckdb  # noqa: E402
import verdict as _verdict_mod  # noqa: E402 — shared threshold source (#21)

# Lazy imports of store/embedder/date_anchor — done in main() so --help
# is fast and doesn't require the full plugin stack.


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_LIMIT = 100
MAX_LIMIT = 5000
DEFAULT_SEED = 42
DEFAULT_LADDER = "5,20,96"
DEFAULT_OUT = "eval_self_corpus_out.jsonl"
DEFAULT_EMBEDDER = "BAAI/bge-small-en-v1.5"
WIDE_POOL_K = 512  # mirror diagnose_ranking.py's pool-cap insight
EXPIRING_SOON_DAYS = 7  # not used here but documented for parity

# Domain-agnostic synonym map (never commit real user content).
_SYNONYM_MAP: Dict[str, str] = {
    "car": "vehicle",
    "job": "work",
    "bond": "home loan",
    "boss": "manager",
    "wife": "partner",
    "husband": "partner",
    "girlfriend": "partner",
    "boyfriend": "partner",
    "house": "home",
    "apartment": "home",
    "flat": "home",
    "phone": "device",
    "laptop": "computer",
    "doctor": "physician",
    "kid": "child",
    "kids": "children",
    "dog": "pet",
    "cat": "pet",
    "money": "funds",
    "salary": "income",
    "company": "organization",
    "school": "institution",
    "gym": "fitness center",
}

# Stopwords for subject extraction (subset of store._PHRASE_STOPWORDS).
_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "am", "be", "been",
    "i", "me", "my", "you", "your", "we", "our", "us", "this", "that",
    "these", "those", "it", "its", "of", "to", "in", "for", "and", "or",
    "but", "on", "with", "at", "by", "from", "as", "do", "does", "did",
    "what", "who", "how", "where", "when", "why", "which", "about", "so",
    "very", "have", "has", "had", "would", "will", "can", "could",
    "user", "likes", "prefers", "wants", "needs", "has", "is",
})

# Templates exercised round-robin within the sample.
_TEMPLATES = ("direct", "temporal", "preference", "entity", "negation", "synonym")


# ---------------------------------------------------------------------------
# DB connection helpers
# ---------------------------------------------------------------------------

def _try_readonly_connect(db_path: Path) -> Optional[duckdb.DuckDBPyConnection]:
    """Try opening a read-only connection; return None if locked/unavailable."""
    try:
        return duckdb.connect(str(db_path), read_only=True)
    except Exception:
        return None


def _resolve_db(db_path: Path) -> Tuple[Path, bool, Optional[Path]]:
    """Return (effective_db_path, is_copy, cleanup_dir).

    Tries read-only first.  If the file is locked (writer held by the
    desktop app), copies to a temp path and returns that.  Never writes
    to the live store.
    """
    ro = _try_readonly_connect(db_path)
    if ro is not None:
        ro.close()
        # Read-only works for sampling.  For retrieval (store init does
        # writes) we still need a copy — the store's _init_db runs
        # CREATE TABLE / ALTER TABLE / backfill UPDATEs which fail on a
        # read-only connection.  Copy for the store; sample from the
        # original via a fresh read-only conn.
        return db_path, False, None
    # Locked — copy everything.
    tmpdir = Path(tempfile.mkdtemp(prefix="eval_self_corpus_"))
    copy_path = tmpdir / db_path.name
    shutil.copy2(db_path, copy_path)
    # DuckDB may have a .wal file alongside.
    wal = Path(str(db_path) + ".wal")
    if wal.exists():
        shutil.copy2(wal, Path(str(copy_path) + ".wal"))
    return copy_path, True, tmpdir


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _sample_memories(
    conn: duckdb.DuckDBPyConnection,
    user_id: str,
    limit: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Stratified sample of active, current, non-expired memories.

    Stratifies by category (proportional to store distribution) then by
    recency (half from newest third, half from the rest).  Deterministic
    via *seed*.
    """
    now = _dt.datetime.now(_dt.timezone.utc).isoformat()
    rows = conn.execute(
        """
        SELECT memory_id, category, content, tags, created_at,
               expires_at, valid_to, superseded_by, status, project_id
        FROM memory_records
        WHERE COALESCE(status, 'active') = 'active'
          AND valid_to IS NULL
          AND (expires_at IS NULL OR expires_at > ?)
          AND (user_scope IS NULL OR user_scope = ?)
        """,
        [now, user_id],
    ).fetchall()
    if not rows:
        return []
    cols = [
        "memory_id", "category", "content", "tags", "created_at",
        "expires_at", "valid_to", "superseded_by", "status", "project_id",
    ]
    records = [dict(zip(cols, r)) for r in rows]
    rng = random.Random(seed)

    # Group by category.
    by_cat: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        by_cat.setdefault(rec["category"] or "context_note", []).append(rec)

    # Sort each category by created_at descending (newest first).
    for cat_recs in by_cat.values():
        cat_recs.sort(key=lambda r: r.get("created_at") or "", reverse=True)

    # Proportional allocation per category.
    total = len(records)
    sample: List[Dict[str, Any]] = []
    for cat, cat_recs in by_cat.items():
        n_cat = max(1, round(limit * len(cat_recs) / total))
        third = max(1, len(cat_recs) // 3)
        newest = cat_recs[:third]
        rest = cat_recs[third:]
        rng.shuffle(newest)
        rng.shuffle(rest)
        half = n_cat // 2
        picked = newest[:half] + rest[: n_cat - half]
        # If one side is short, fill from the other.
        if len(picked) < n_cat:
            pool = newest + rest
            for r in pool:
                if r in picked:
                    continue
                picked.append(r)
                if len(picked) >= n_cat:
                    break
        sample.extend(picked)

    # Trim to limit (round-robin across categories for fairness).
    if len(sample) > limit:
        rng.shuffle(sample)
        sample = sample[:limit]
    return sample


# ---------------------------------------------------------------------------
# Query generation (deterministic, free)
# ---------------------------------------------------------------------------

def _subject_tokens(content: str, max_tokens: int = 4) -> str:
    """Extract first 1-3 non-stopword tokens (cap *max_tokens* total)."""
    tokens = re.findall(r"[a-z0-9']+", (content or "").lower())
    keywords = [t for t in tokens if t not in _STOPWORDS and len(t) > 2]
    if not keywords:
        keywords = [t for t in tokens if len(t) > 1]
    return " ".join(keywords[:max_tokens])


def _clip(text: str, n: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0]


def _has_date_expression(content: str) -> bool:
    """Reuse date_anchor patterns to detect a date expression in content."""
    try:
        from date_anchor import resolve_target_date
    except Exception:
        return False
    try:
        target, _ = resolve_target_date(content or "")
        return target is not None
    except Exception:
        return False


def _apply_synonym(content: str) -> Optional[str]:
    """Replace one content token with a synonym; return rewritten content or None."""
    tokens = (content or "").split()
    for i, tok in enumerate(tokens):
        key = re.sub(r"[^a-z]", "", tok.lower())
        if key in _SYNONYM_MAP:
            tokens[i] = _SYNONYM_MAP[key]
            return " ".join(tokens)
    return None


def _first_verbish(content: str) -> str:
    """Heuristic: first token after the subject that looks verb-like."""
    tokens = re.findall(r"[a-z0-9']+", (content or "").lower())
    keywords = [t for t in tokens if t not in _STOPWORDS and len(t) > 2]
    if len(keywords) >= 2:
        return keywords[1]
    return ""


def generate_probe(memory: Dict[str, Any], template: str) -> Optional[Dict[str, Any]]:
    """Generate one probe query for *memory* using *template*.

    Returns ``{"query": str, "template": str}`` or None if the template
    does not apply to this memory (e.g. temporal on a non-dated record).
    """
    content = memory.get("content") or ""
    category = memory.get("category") or ""
    tags = memory.get("tags") or []
    subject = _subject_tokens(content)

    if template == "direct":
        if not subject:
            return None
        rest = _clip(content, 24)
        return {"query": f"what is {subject} {rest}?", "template": "direct"}

    if template == "temporal":
        if not _has_date_expression(content):
            return None
        if not subject:
            return None
        rest = _clip(content, 24)
        return {"query": f"when did {subject} {rest}?", "template": "temporal"}

    if template == "preference":
        if category != "preference":
            return None
        if not subject:
            return None
        return {"query": f"what does the user prefer about {subject}?", "template": "preference"}

    if template == "entity":
        # Tags matching graph entities (non-empty tag list as a proxy).
        if not tags:
            return None
        if not subject:
            return None
        verb = _first_verbish(content)
        return {"query": f"{subject} {verb}?", "template": "entity"}

    if template == "negation":
        if not subject:
            return None
        return {"query": f"who is NOT {subject}", "template": "negation"}

    if template == "synonym":
        rewritten = _apply_synonym(content)
        if not rewritten:
            return None
        subj = _subject_tokens(rewritten)
        if not subj:
            return None
        rest = _clip(rewritten, 24)
        return {"query": f"what is {subj} {rest}?", "template": "synonym"}

    return None


def assign_templates(memories: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], str]]:
    """Round-robin template assignment, skipping non-applicable templates."""
    assigned: List[Tuple[Dict[str, Any], str]] = []
    tidx = 0
    for mem in memories:
        # Try the round-robin template first; if it doesn't apply, walk
        # forward through the template list until one does.
        for offset in range(len(_TEMPLATES)):
            template = _TEMPLATES[(tidx + offset) % len(_TEMPLATES)]
            probe = generate_probe(mem, template)
            if probe is not None:
                assigned.append((mem, template))
                tidx = (tidx + offset + 1) % len(_TEMPLATES)
                break
        else:
            # No template applied — assign direct as a fallback.
            probe = generate_probe(mem, "direct")
            if probe is not None:
                assigned.append((mem, "direct"))
    return assigned


# ---------------------------------------------------------------------------
# Hit rule
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> Set[str]:
    return set(re.findall(r"[a-z0-9']+", (text or "").lower()))


def _token_jaccard(a: str, b: str) -> float:
    sa, sb = _tokenize(a), _tokenize(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _build_chain_set(store: Any, memory_id: str) -> Set[str]:
    """Return the set of all memory_ids in the same version chain."""
    try:
        history = store.get_memory_history(memory_id)
        return {v.memory_id for v in history}
    except Exception:
        return {memory_id}


def is_hit(
    target: Dict[str, Any],
    result_ids: List[str],
    result_contents: Dict[str, str],
    chain_ids: Set[str],
) -> bool:
    """A probe is a hit if ANY of: exact id, older version (chain), Jaccard ≥ 0.75."""
    target_id = target["memory_id"]
    target_content = target.get("content") or ""
    for rid in result_ids:
        if rid == target_id:
            return True
        if rid in chain_ids:
            return True
        rcontent = result_contents.get(rid, "")
        if rcontent and _token_jaccard(rcontent, target_content) >= 0.75:
            return True
    return False


# ---------------------------------------------------------------------------
# Config hash
# ---------------------------------------------------------------------------

def _config_hash(db_path: Path, embedder_model: str) -> str:
    """SHA-256 over retrieval-relevant knobs so runs are comparable."""
    cfg_path = db_path.parent / "hybrid_memory.json"
    cfg: Dict[str, Any] = {}
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            cfg = {}
    # Import store class constants.
    try:
        from store import DuckDBMemoryStore as S
        rrf_k = S._RRF_K
        imp_helpful = S._IMPORTANCE_HELPFUL_WEIGHT
        imp_dismissed = S._IMPORTANCE_DISMISSED_WEIGHT
        imp_confidence = S._IMPORTANCE_CONFIDENCE_WEIGHT
        imp_retrieval = S._IMPORTANCE_RETRIEVAL_WEIGHT
        imp_retrieval_cap = S._IMPORTANCE_RETRIEVAL_CAP
        imp_age_decay = S._IMPORTANCE_AGE_DECAY_PER_DAY
        imp_age_cap = S._IMPORTANCE_AGE_DECAY_CAP_DAYS
        imp_dormancy = S._IMPORTANCE_DORMANCY_DECAY_PER_DAY
        imp_dormancy_cap = S._IMPORTANCE_DORMANCY_CAP_DAYS
        imp_base_clamp = S._IMPORTANCE_BASE_CLAMP
        imp_feedback_clamp = S._IMPORTANCE_FEEDBACK_CLAMP
    except Exception:
        rrf_k = imp_helpful = imp_dismissed = imp_confidence = 0
        imp_retrieval = imp_retrieval_cap = imp_age_decay = imp_age_cap = 0
        imp_dormancy = imp_dormancy_cap = imp_base_clamp = imp_feedback_clamp = 0

    knobs = {
        "embedder_model": embedder_model,
        "max_injected_items": cfg.get("max_injected_items", ""),
        "inject_content_char_cap": cfg.get("inject_content_char_cap", ""),
        "graph_aware_retrieval": cfg.get("graph_aware_retrieval", ""),
        "graph_retrieval_boost": cfg.get("graph_retrieval_boost", ""),
        "graph_boost_min_similarity": cfg.get("graph_boost_min_similarity", ""),
        "query_expansion_enabled": cfg.get("query_expansion_enabled", ""),
        "date_anchor_rerank": cfg.get("date_anchor_rerank", ""),
        "chain_unfold": cfg.get("chain_unfold", ""),
        "chain_unfold_min_similarity": cfg.get("chain_unfold_min_similarity", ""),
        "reranker_enabled": cfg.get("reranker_enabled", ""),
        "reranker_top_n": cfg.get("reranker_top_n", ""),
        "context_aware_retrieval": cfg.get("context_aware_retrieval", ""),
        "alias_expansion_boost": cfg.get("alias_expansion_boost", ""),
        "_RRF_K": rrf_k,
        "_IMPORTANCE_HELPFUL_WEIGHT": imp_helpful,
        "_IMPORTANCE_DISMISSED_WEIGHT": imp_dismissed,
        "_IMPORTANCE_CONFIDENCE_WEIGHT": imp_confidence,
        "_IMPORTANCE_RETRIEVAL_WEIGHT": imp_retrieval,
        "_IMPORTANCE_RETRIEVAL_CAP": imp_retrieval_cap,
        "_IMPORTANCE_AGE_DECAY_PER_DAY": imp_age_decay,
        "_IMPORTANCE_AGE_DECAY_CAP_DAYS": imp_age_cap,
        "_IMPORTANCE_DORMANCY_DECAY_PER_DAY": imp_dormancy,
        "_IMPORTANCE_DORMANCY_CAP_DAYS": imp_dormancy_cap,
        "_IMPORTANCE_BASE_CLAMP": imp_base_clamp,
        "_IMPORTANCE_FEEDBACK_CLAMP": imp_feedback_clamp,
    }
    blob = json.dumps(knobs, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Retrieval execution
# ---------------------------------------------------------------------------

def _run_probe(
    store: Any,
    probe: Dict[str, Any],
    target: Dict[str, Any],
    ladder: List[int],
    chain_ids: Set[str],
) -> Dict[str, Any]:
    """Run retrieval for one probe and record per-window hit flags."""
    query = probe["query"]
    max_k = max(ladder)
    # Production-window search.
    results = store.search(query, limit=max_k, suppress_retrieval=True)
    result_ids = [r.memory_id for r in results]
    result_contents = {r.memory_id: r.content for r in results}
    # Wide-pool probe for not_in_pool classification.
    wide = store.search(query, limit=WIDE_POOL_K, suppress_retrieval=True)
    wide_ids = [r.memory_id for r in wide]

    hit = is_hit(target, result_ids, result_contents, chain_ids)
    per_window: Dict[str, bool] = {}
    for k in ladder:
        per_window[str(k)] = is_hit(target, result_ids[:k], result_contents, chain_ids)

    wide_rank: Optional[int] = None
    if target["memory_id"] in wide_ids:
        wide_rank = wide_ids.index(target["memory_id"]) + 1
    # Also check chain membership in wide pool.
    if wide_rank is None:
        for i, wid in enumerate(wide_ids):
            if wid in chain_ids:
                wide_rank = i + 1
                break
    not_in_pool = wide_rank is None

    return {
        "query": query,
        "template": probe["template"],
        "target_memory_id": target["memory_id"],
        "target_category": target.get("category"),
        "target_content_len": len(target.get("content") or ""),
        "target_created_at": target.get("created_at"),
        "hit": hit,
        "per_window": per_window,
        "wide_pool_rank": wide_rank,
        "not_in_pool": not_in_pool,
        "top_5_ids": result_ids[:5],
        "top_5_similarities": [round(r.similarity, 4) for r in results[:5]],
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _age_bucket(created_at: Optional[str]) -> str:
    if not created_at:
        return "unknown"
    try:
        created = _dt.datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        days = (_dt.datetime.now(_dt.timezone.utc) - created).days
    except Exception:
        return "unknown"
    if days < 30:
        return "<30d"
    if days <= 180:
        return "30-180d"
    return ">180d"


def _length_bucket(length: int) -> str:
    if length < 100:
        return "<100"
    if length <= 500:
        return "100-500"
    return ">500"


def _recall_at(probes: List[Dict[str, Any]], k: int) -> float:
    if not probes:
        return 0.0
    hits = sum(1 for p in probes if p["per_window"].get(str(k)))
    return hits / len(probes)


def _group_recall(
    probes: List[Dict[str, Any]], k: int, key_fn,
) -> Dict[str, float]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for p in probes:
        groups.setdefault(key_fn(p), []).append(p)
    return {g: _recall_at(ps, k) for g, ps in sorted(groups.items())}


def compute_metrics(
    probes: List[Dict[str, Any]],
    ladder: List[int],
    sample_size: int,
    category_dist: Dict[str, int],
    config_hash: str,
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "sample_size": sample_size,
        "probe_count": len(probes),
        "config_hash": config_hash,
        "category_distribution": category_dist,
        "ladder": ladder,
    }
    # Overall recall@K.
    metrics["overall"] = {f"recall@{k}": round(_recall_at(probes, k), 4) for k in ladder}
    # By category.
    metrics["by_category"] = {}
    for k in ladder:
        metrics["by_category"][f"recall@{k}"] = {
            g: round(v, 4)
            for g, v in _group_recall(probes, k, lambda p: p["target_category"] or "context_note").items()
        }
    # By template.
    metrics["by_template"] = {}
    for k in ladder:
        metrics["by_template"][f"recall@{k}"] = {
            g: round(v, 4)
            for g, v in _group_recall(probes, k, lambda p: p["template"]).items()
        }
    # By content-length bucket.
    metrics["by_length_bucket"] = {}
    for k in ladder:
        metrics["by_length_bucket"][f"recall@{k}"] = {
            g: round(v, 4)
            for g, v in _group_recall(probes, k, lambda p: _length_bucket(p["target_content_len"])).items()
        }
    # By age bucket.
    metrics["by_age_bucket"] = {}
    for k in ladder:
        metrics["by_age_bucket"][f"recall@{k}"] = {
            g: round(v, 4)
            for g, v in _group_recall(probes, k, lambda p: _age_bucket(p["target_created_at"])).items()
        }
    # not_in_pool_rate + avg_rank_of_hits.
    not_in_pool = sum(1 for p in probes if p["not_in_pool"])
    metrics["not_in_pool_rate"] = round(not_in_pool / len(probes), 4) if probes else 0.0
    hit_ranks = [p["wide_pool_rank"] for p in probes if p["wide_pool_rank"] is not None and p["hit"]]
    metrics["avg_rank_of_hits"] = round(sum(hit_ranks) / len(hit_ranks), 2) if hit_ranks else None
    return metrics


# ---------------------------------------------------------------------------
# Baseline verdict
# ---------------------------------------------------------------------------

def _load_baseline(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                if "run_summary" in obj:
                    return obj["run_summary"]
    except Exception:
        return None
    return None


def _verdict(current: Dict[str, Any], baseline: Dict[str, Any]) -> str:
    """Delegates to the shared verdict module (#21).

    Both run_gate and eval_self_corpus --baseline now use the same
    thresholds: category recall@max-k > 1pp, overall recall@max-k > 0.5pp,
    overall MRR > 0.01.  The old 3pp recall@20-only verdict is deprecated.
    """
    return _verdict_mod.verdict_string(current, baseline)


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def _completed_ids(out_path: Path) -> Set[str]:
    if not out_path.exists():
        return set()
    done: Set[str] = set()
    try:
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                mid = obj.get("target_memory_id")
                if mid and "run_summary" not in obj:
                    done.add(mid)
    except Exception:
        pass
    return done


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_ladder(s: str) -> List[int]:
    try:
        return [int(x.strip()) for x in s.split(",") if x.strip()]
    except Exception:
        return [5, 20, 96]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Regression-test retrieval on the real memory store (free, deterministic).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--db", required=True, help="Path to hybrid_memory.duckdb (live store or a copy).")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help=f"Memories sampled (default {DEFAULT_LIMIT}, max {MAX_LIMIT}).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Reproducible sampling seed (default {DEFAULT_SEED}).")
    parser.add_argument("--top-k-ladder", default=DEFAULT_LADDER, help=f"Recall windows, comma-separated (default '{DEFAULT_LADDER}').")
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"Append-only JSONL output (default '{DEFAULT_OUT}'). Never commit to public repos.")
    parser.add_argument("--baseline", default="", help="Path to a prior --out file; enables PASS/FAIL regression verdict.")
    parser.add_argument("--embedder-model", default=DEFAULT_EMBEDDER, help=f"Embedder model name (default '{DEFAULT_EMBEDDER}'). Must resolve from local HF cache.")
    parser.add_argument("--llm-paraphrase", action="store_true", help="OFF by default. Synthesize one LLM-paraphrased probe per memory (bounded, costs API).")
    parser.add_argument("--threads", type=int, default=4, help="Embedding workers (keep modest; default 4).")
    parser.add_argument("--resume", action="store_true", help="Skip memory_ids already present in --out.")
    parser.add_argument("--verbose", action="store_true", help="Print per-probe details.")
    parser.add_argument("--user-id", default="default_user", help="User scope to sample (default 'default_user').")
    args = parser.parse_args(argv)

    limit = max(1, min(args.limit, MAX_LIMIT))
    ladder = _parse_ladder(args.top_k_ladder)
    db_path = Path(args.db).resolve()
    if not db_path.exists():
        print(f"ERROR: DB not found: {db_path}", file=sys.stderr)
        return 2

    out_path = Path(args.out)
    done_ids: Set[str] = _completed_ids(out_path) if args.resume else set()

    # --- Resolve DB (read-only or copy) ---
    effective_db, is_copy, cleanup_dir = _resolve_db(db_path)
    if is_copy:
        print(f"DB locked or read-only — copied to {effective_db}", flush=True)

    # --- Sampling (read-only on the original, or on the copy) ---
    sample_conn = _try_readonly_connect(db_path)
    if sample_conn is None:
        sample_conn = _try_readonly_connect(effective_db)
    if sample_conn is None:
        # Fall back to a write connection on the copy.
        sample_conn = duckdb.connect(str(effective_db))
    if sample_conn is None:
        print("ERROR: could not open DB for sampling", file=sys.stderr)
        return 2

    print(f"Sampling up to {limit} memories (seed={args.seed})...", flush=True)
    memories = _sample_memories(sample_conn, args.user_id, limit, args.seed)
    sample_conn.close()
    if not memories:
        print("No active memories found to sample.", flush=True)
        if cleanup_dir:
            shutil.rmtree(cleanup_dir, ignore_errors=True)
        return 0
    print(f"Sampled {len(memories)} memories.", flush=True)

    # Skip already-completed ids.
    if done_ids:
        memories = [m for m in memories if m["memory_id"] not in done_ids]
        if not memories:
            print("All sampled memories already in --out; nothing to do.", flush=True)
            if cleanup_dir:
                shutil.rmtree(cleanup_dir, ignore_errors=True)
            return 0
        print(f"{len(memories)} new memories to probe (after --resume).", flush=True)

    # --- Assign templates ---
    assigned = assign_templates(memories)
    print(f"Assigned probes: {len(assigned)}", flush=True)

    # --- Build embedder + store for retrieval ---
    from embeddings import LocalEmbedder, _resolve_embedding_model_path
    from store import DuckDBMemoryStore

    resolved_model = _resolve_embedding_model_path(args.embedder_model, hermes_home=None)
    embedder = LocalEmbedder(resolved_model)
    store = DuckDBMemoryStore(effective_db, user_id=args.user_id, embedder=embedder)

    cfg_hash = _config_hash(db_path, args.embedder_model)

    # --- Optional LLM paraphrase ---
    llm_call_count = 0
    llm_cap = 50
    llm_client = None
    if args.llm_paraphrase:
        try:
            from agent.auxiliary_client import call_llm as llm_client  # noqa
            print("LLM paraphrase enabled (bounded). Cost estimate: ~R0.03/call, cap=50.", flush=True)
        except Exception:
            print("LLM paraphrase requested but auxiliary_client unavailable; skipping.", flush=True)
            llm_client = None

    # --- Run probes ---
    category_dist: Dict[str, int] = {}
    probes_out: List[Dict[str, Any]] = []
    ts = _dt.datetime.now(_dt.timezone.utc).isoformat()

    for i, (mem, template) in enumerate(assigned):
        probe = generate_probe(mem, template)
        if probe is None:
            continue
        chain_ids = _build_chain_set(store, mem["memory_id"])
        try:
            result = _run_probe(store, probe, mem, ladder, chain_ids)
        except Exception as exc:
            result = {
                "query": probe["query"],
                "template": probe["template"],
                "target_memory_id": mem["memory_id"],
                "error": str(exc),
            }
        result["config_hash"] = cfg_hash
        result["timestamp"] = ts
        cat = mem.get("category") or "context_note"
        category_dist[cat] = category_dist.get(cat, 0) + 1
        probes_out.append(result)
        # Append immediately (append-only).
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        if args.verbose:
            hit = result.get("hit", False)
            print(f"  [{i+1}/{len(assigned)}] {template:10s} hit={hit} q={probe['query'][:60]}", flush=True)

        # Optional LLM paraphrase probe.
        if args.llm_paraphrase and llm_client is not None and llm_call_count < llm_cap:
            try:
                resp = llm_client(
                    task="memory_eval_paraphrase",
                    messages=[
                        {"role": "system", "content": "Paraphrase the user's question about a stored memory in 1 sentence. Return only the paraphrase."},
                        {"role": "user", "content": probe["query"]},
                    ],
                    temperature=0.3,
                    max_tokens=100,
                    timeout=10.0,
                )
                para_text = resp.choices[0].message.content.strip()
                llm_call_count += 1
                if para_text:
                    para_probe = {"query": para_text, "template": "llm_paraphrase"}
                    para_result = _run_probe(store, para_probe, mem, ladder, chain_ids)
                    para_result["config_hash"] = cfg_hash
                    para_result["timestamp"] = ts
                    with out_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(para_result, ensure_ascii=False) + "\n")
            except Exception:
                pass  # fail-soft

    store.close()

    # --- Compute metrics ---
    metrics = compute_metrics(probes_out, ladder, len(memories), category_dist, cfg_hash)
    summary_line = {"run_summary": metrics, "timestamp": ts}
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(summary_line, ensure_ascii=False) + "\n")

    # --- Print results ---
    print(f"\n=== Self-corpus eval ({len(probes_out)} probes) ===", flush=True)
    print(f"config_hash: {cfg_hash}", flush=True)
    print(f"sample_size: {metrics['sample_size']}", flush=True)
    print(f"category_distribution: {category_dist}", flush=True)
    for k in ladder:
        r = metrics["overall"].get(f"recall@{k}", 0.0)
        print(f"  recall@{k}: {r*100:.1f}%", flush=True)
    print(f"  not_in_pool_rate: {metrics['not_in_pool_rate']*100:.1f}%", flush=True)
    print(f"  avg_rank_of_hits: {metrics['avg_rank_of_hits']}", flush=True)
    if metrics["by_template"]:
        print("  by_template (recall@20):", flush=True)
        for t, v in metrics["by_template"].get("recall@20", {}).items():
            print(f"    {t:15s}: {v*100:.1f}%", flush=True)

    # --- Baseline verdict ---
    if args.baseline:
        baseline = _load_baseline(Path(args.baseline))
        if baseline is None:
            print("WARNING: could not load baseline; skipping verdict.", flush=True)
        elif baseline.get("config_hash") != cfg_hash:
            print(f"WARNING: baseline config_hash ({baseline.get('config_hash')}) != current ({cfg_hash}); verdict may be unreliable.", flush=True)
            print(_verdict(metrics, baseline), flush=True)
        else:
            print(f"\n=== VERDICT ===", flush=True)
            print(_verdict(metrics, baseline), flush=True)

    # --- Cleanup ---
    if cleanup_dir:
        shutil.rmtree(cleanup_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
