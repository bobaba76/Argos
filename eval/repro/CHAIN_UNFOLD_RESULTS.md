# Chain-unfold: repro artifact (2026-08-27)

Committed reference run of `argos_plugin/eval/eval_chain_unfold.py` (sanitized
harness). This is the artifact backing §10 of BENCHMARK_REPRODUCIBILITY.md and
the "Chain-unfold" row of its claim map.

## Run metadata

| Field | Value |
|---|---|
| Date | 2026-08-27 |
| Repo | `bobaba76/Argos` @ `16f4f528c1f6311ddb76d171f38ab89d36ff98b0` (master) |
| Python | 3.11.9 (Hermes `hermes-agent/venv`) |
| Embedder | `bge-small-en-v1.5` (local cache, offline — `HF_HUB_OFFLINE=1`) |
| Corpus snapshot | `%LOCALAPPDATA%\hermes\memory-eval-clean.duckdb` (sha256 prefix `861118e6f7ac98da`) |
| Mode | provider-layer, direct storage, temp home per variant (never touches live data); seeds sanitized (generic names/employers/meds) |
| Cost | 0 LLM calls — precision/recall is deterministic vs expected flags; token figures are `len(arc)//4` estimates, not billed calls |

## Results

### Trade-off (3 production-shaped variants)

| variant | precision | recall | tp | fp | fn | tn | tokens |
|---|---|---|---|---|---|---|---|
| `top1_baseline` (top_k=1, no fallback) | 100% | 60% | 3 | 0 | 2 | 5 | 67 |
| `top3` (top_k=3, no fallback) | 100% | 60% | 3 | 0 | 2 | 5 | 67 |
| `top3_fallback` (top_k=3, fallback) | 80% | 80% | 4 | 1 | 1 | 4 | 117 |

### Arc-floor sweep (top_k=3 + fallback + arc gate)

Identical `4/1/1/4` (80%/80%) at **every** floor: 0.00, 0.10, 0.15, 0.20, 0.25,
0.30, 0.35. The gate selected nothing — no floor met the ≥90% / ≥90% band.

### Per-query truth table (top3_fallback)

| query | expected | unfolded | correct | class |
|---|---|---|---|---|
| why did I stop using Spotify | yes | yes | yes | TP |
| why did I switch music services | yes | yes | yes | TP |
| what changed with my property plan | yes | **no** | — | **FN** |
| when did I change my medication | yes | yes | yes | TP (fallback-rescued) |
| why did I stop using Topiramate | yes | yes | yes | TP |
| what changed in the weather today | no | **yes** | no | **FP** |
| why did the dog food brand change | no | no | — | TN |
| what music do I like | no | no | — | TN |
| tell me about my dog | no | no | — | TN |
| how much budget do I have | no | no | — | TN |

`get_chain_unfold_stats()`: top1/top3 → `count=3, tokens=67`; fallback →
`count=5, tokens=117`.

## Findings

1. **The archived 20/8 claim (~93% recall / ~93% precision) does NOT reproduce
   on current code.** Best measured here: 80% / 80%. The number in §10 of
   BENCHMARK_REPRODUCIBILITY.md and the MEMORY_SYSTEM.md bullet was stale
   relative to repo HEAD.
2. **The arc-similarity gate is inert on this query set** — identical counts at
   0.00 and 0.35. The archived framing "`Arc(0.15)` + `anchor(0.30)` are pure
   precision gates with zero recall cost" does not describe current behavior.
3. **Recall leak is retrieval-shaped, not gate-shaped:** "what changed with my
   property plan" never unfolds in any config, including with query-fallback —
   the chain's current version is not surfaced (RETRIEVAL-BURIED per the §10
   taxonomy), so no unfold decision is even reached.
4. **Precision leak is matcher/retrieval-shaped, not threshold-shaped:** "what
   changed in the weather today" fires only when fallback is on, and passes the
   arc gate at 0.35 — no cosine floor separates it.
5. The 20/8 run used the same protocol on then-current code; matcher and
   retrieval changes since (include_closed work, tombstones, temporal
   hardening) likely explain the delta — **not diagnosed in this pass.**

## Status

- Artifact **committed**: harness (`argos_plugin/eval/eval_chain_unfold.py`) +
  this results file. Repro: `env -u PYTHONPATH HF_HUB_OFFLINE=1 <hermes-venv-python> argos_plugin/eval/eval_chain_unfold.py`.
- The 93% claim stays **archived/directional** until the intent matcher +
  retrieval geometry are re-examined (the private probe files
  `probe_chain_miss.py` / `probe_fp.py` etc. remain dev-tree-only by rule).
  Do not quote 93% as current.