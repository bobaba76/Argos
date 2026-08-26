#!/usr/bin/env python3
"""personal_bench.py — personal-domain answerer benchmark.

Retrieval is measured by the gate (eval/run_gate.py). This harness measures
the ANSWERING half: for each gold probe it retrieves top-k from the frozen
SNAPSHOT (never live), asks an answerer model to answer using ONLY the
retrieved notes, then a judge model (same model by default — internal
ruler, swap with --judge-model) verifies the answer against the gold fact.

Both models default to deepseek-v4-flash on opencode-go — the production
assistant model. Internal benchmarking only; outputs carry personal memory
content and live under eval/bench/ (gitignored, never commit).

Pipeline per probe:
    [--paraphrase] -> retrieve top-k (suppress_retrieval=True)
    -> answerer(query, notes) -> judge(query, gold fact, answer) -> YES/NO

Usage:
    python eval/personal_bench.py --snapshot eval/snapshots/<id> \
        --gold eval/gold/gold_v1.jsonl --limit 100 --paraphrase
    # append-only JSONL resume: rerun with same --out continues.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("HF_HUB_OFFLINE", "1")

_PLUGIN_ROOT = Path(__file__).resolve().parents[1]
if str(_PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_ROOT))
_EVAL_DIR = Path(__file__).resolve().parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

import build_gold  # noqa: E402
import eval_self_corpus as esc  # noqa: E402

DEFAULT_EMBEDDER = "BAAI/bge-small-en-v1.5"
DEFAULT_MODEL = "deepseek-v4-flash"
BASE_URL = "https://opencode.ai/zen/go/v1/chat/completions"
DEFAULT_OUT = "eval/bench/bench_out.jsonl"
CONCURRENCY = 8
RETRY_ONCE_CODES = {429, 500, 502, 503, 504}

_ATTRIBUTION = {
    "HTTP-Referer": "https://hermes-agent.nousresearch.com",
    "X-Title": "Hermes Agent",
    "User-Agent": "HermesAgent/personal-bench",
}

ANSW_SYSTEM = (
    "You answer questions about facts stored in a personal memory system. "
    "Use the memory notes to answer; when the notes support an answer, give "
    "it in one short sentence. Only if NO note supports an answer, reply "
    "exactly: NOT IN NOTES."
)
JUDGE_SYSTEM = (
    "You are a strict factual judge. Decide whether the ANSWER correctly "
    "states the fact given as the GOLD FACT, for the QUESTION. An answer of "
    "'NOT IN NOTES' is INCORRECT if the gold fact appears in the notes the "
    "answerer was given; you only see the gold fact here, so treat 'NOT IN "
    "NOTES' as incorrect. Reply with exactly YES or NO."
)
PARA_SYSTEM = (
    "Produce ONE natural, standalone question a person might ask their "
    "personal assistant, such that the EXPECTED ANSWER is exactly the FACT "
    "below. Derive the question from the FACT, not from the template query — "
    "keep the fact's direction (never invert/negate it, never invent people "
    "or entities), keep all names/numbers/specifics. If the FACT is about "
    "the user, the question may ask about 'the user' or use second person. "
    "Output only the question, no preamble."
)


def _api_key() -> str:
    env = os.environ.get("OPENCODE_GO_API_KEY")
    if env:
        return env
    home = os.environ.get("HERMES_HOME") or str(Path.home() / "AppData/Local/hermes")
    for line in (Path(home) / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("OPENCODE_GO_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("OPENCODE_GO_API_KEY not found")


_HTTPX: Any = None  # module-level persistent client (one TLS handshake, not per call)


def _client() -> Any:
    global _HTTPX
    if _HTTPX is None:
        import httpx
        _HTTPX = httpx.Client(timeout=httpx.Timeout(180.0, connect=15.0))
    return _HTTPX


def _call(messages: List[Dict[str, str]], max_tokens: int, key: str) -> Dict[str, Any]:
    body = {
        "model": DEFAULT_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        **_ATTRIBUTION,
    }
    last: Optional[Exception] = None
    for attempt in range(2):
        try:
            resp = _client().post(BASE_URL, json=body, headers=headers)
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            out = resp.json()
            return {
                "text": out["choices"][0]["message"].get("content", "").strip(),
                "usage": out.get("usage", {}),
            }
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise last  # type: ignore[misc]


def _notes_block(records: List[Any]) -> str:
    return "\n".join(f"{i+1}. {r.content}" for i, r in enumerate(records))


def _yesno(text: str) -> Optional[bool]:
    t = text.strip().upper()
    if t.startswith("YES"):
        return True
    if t.startswith("NO"):
        return False
    return None


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snapshot", required=True, help="Snapshot dir (manifest.json + duckdb)")
    parser.add_argument("--gold", required=True, help="Frozen gold JSONL (approved lines)")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=0, help="0 = all approved")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=20, help="Retrieval window (prod-ish)")
    parser.add_argument("--paraphrase", action="store_true",
                        help="Paraphrase template queries first (1 call/probe); "
                             "writes eval/gold/paraphrase_<limit>.jsonl")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--user-id", default="default_user")
    args = parser.parse_args(argv)

    snap = Path(args.snapshot)
    manifest = json.loads((snap / "manifest.json").read_text(encoding="utf-8"))
    gold_lines = [l for l in build_gold.load_gold(Path(args.gold))
                  if l.get("status") == "approved"]
    if args.limit > 0:
        rng = random.Random(args.seed)
        rng.shuffle(gold_lines)
        gold_lines = gold_lines[:args.limit]
    print(f"probes: {len(gold_lines)}  snapshot: {manifest['snapshot_id']}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done_ids.add(json.loads(line)["gold_id"])
    todo = [l for l in gold_lines if l["memory_id"] not in done_ids]
    print(f"resume: {len(done_ids)} done, {len(todo)} to run")

    if not todo:
        print("nothing to do")
        return 0

    key = _api_key()
    tmpdir = Path(tempfile.mkdtemp(prefix="bench_"))
    db_copy = tmpdir / manifest["db_filename"]
    shutil.copy2(snap / manifest["db_filename"], db_copy)
    try:
        from embeddings import LocalEmbedder, _resolve_embedding_model_path
        from store import DuckDBMemoryStore

        embedder = LocalEmbedder(_resolve_embedding_model_path(DEFAULT_EMBEDDER, hermes_home=None))
        store = DuckDBMemoryStore(db_copy, user_id=args.user_id, embedder=embedder)

        para_out = None
        if args.paraphrase:
            para_path = Path("eval/gold") / f"paraphrase_{args.limit or 'all'}.jsonl"
            para_out = para_path
            existing = set()
            if para_path.exists():
                for line in para_path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        existing.add(json.loads(line)["memory_id"])

        def run_one(line: Dict[str, Any]) -> Dict[str, Any]:
            q = line["query"]
            if args.paraphrase and line["memory_id"] not in existing:
                pr = _call([{"role": "system", "content": PARA_SYSTEM},
                            {"role": "user",
                             "content": f"FACT: {line['content']}\n\n"
                                        f"TEMPLATE QUERY (ignore unless it helps): {q}\n\n"
                                        f"Question:"}], 160, key)
                q = pr["text"] or q
            t0 = time.time()
            results = store.search(q, limit=args.k, suppress_retrieval=True)
            search_ms = int((time.time() - t0) * 1000)
            notes = _notes_block(results)
            hit = line["memory_id"] in {r.memory_id for r in results}
            ans = _call([{"role": "system", "content": ANSW_SYSTEM},
                         {"role": "user",
                          "content": f"Memory notes:\n{notes}\n\nQuestion: {q}\n\nAnswer:"}],
                        300, key)
            jdg = _call([{"role": "system", "content": JUDGE_SYSTEM},
                         {"role": "user",
                          "content": f"GOLD FACT: {line['content']}\nQUESTION: {q}\n"
                                     f"ANSWER: {ans['text']}\n\nCorrect? (YES/NO)"}],
                        24, key)
            correct = _yesno(jdg["text"])
            rec = {
                "gold_id": line["memory_id"],
                "category": line.get("category"),
                "gold_content": line["content"],
                "query": q,
                "paraphrased": args.paraphrase and line["memory_id"] not in existing,
                "gold_in_topk": hit,
                "answer": ans["text"],
                "judge": jdg["text"],
                "correct": correct,
                "search_ms": search_ms,
                "usage": {"ans": ans["usage"], "judge": jdg["usage"]},
            }
            return rec

        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futs = {pool.submit(run_one, l): l for l in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    rec = fut.result()
                except Exception as exc:  # noqa: BLE001
                    print(f"  [{i}/{len(todo)}] FAILED {futs[fut]['memory_id']}: {exc}")
                    continue
                with out_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if para_out is not None and rec["paraphrased"]:
                    with para_out.open("a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "memory_id": rec["gold_id"],
                            "category": rec["category"],
                            "content": futs[fut]["content"],
                            "query": rec["query"],
                            "template": "paraphrase",
                            "status": "approved",
                        }, ensure_ascii=False) + "\n")
                if i % 10 == 0 or i == len(todo):
                    print(f"  [{i}/{len(todo)}] done")
        store.close()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # Summary
    rows = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    judged = [r for r in rows if r["correct"] is not None]
    acc = sum(1 for r in judged if r["correct"]) / len(judged) if judged else 0.0
    hit = sum(1 for r in judged if r["gold_in_topk"]) / len(judged) if judged else 0.0
    print("\n=== bench summary ===")
    print(f"probes judged: {len(judged)}/{len(rows)}  accuracy: {acc*100:.1f}%  "
          f"gold_in_topk: {hit*100:.1f}%")
    by_cat: Dict[str, List[bool]] = {}
    for r in judged:
        by_cat.setdefault(r["category"], []).append(bool(r["correct"]))
    for c, vals in sorted(by_cat.items()):
        print(f"  {c:14s} {sum(vals)/len(vals)*100:5.1f}%  (n={len(vals)})")
    tok = sum(r["usage"]["ans"].get("total_tokens", 0) + r["usage"]["judge"].get("total_tokens", 0)
              for r in rows)
    print(f"tokens: {tok} (opencode-go meter)")
    return 0


if __name__ == "__main__":
    sys.exit(main())