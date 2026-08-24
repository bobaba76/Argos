#!/usr/bin/env bash
# verify_repro.sh — reproducibility gate for BENCHMARK_REPRODUCIBILITY.md.
#
# Re-derives every headline number from the committed judged artifacts
# (this repo's eval/repro/) plus the external run artifacts (default:
# sibling ../LongMemEval checkout), and fails loudly on any drift.
#
# Usage:
#   ./verify_repro.sh                 # artifacts in ../LongMemEval
#   ./verify_repro.sh --artifacts DIR # point at a different checkout
#
# Exit codes: 0 = every claim verified; 1 = at least one check failed.
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPRO_DIR="$REPO_ROOT/eval/repro"
ARTIFACTS="${REPO_ROOT}/../LongMemEval"

if [[ "${1:-}" == "--artifacts" ]]; then
    ARTIFACTS="$2"
fi

PY="$(command -v python || command -v python3)"
if [[ -z "$PY" ]]; then
    echo "FAIL  no python interpreter found" >&2
    exit 1
fi
# Native Windows python cannot open MSYS-style /c/... paths.
win() { cygpath -w "$1" 2>/dev/null || echo "$1"; }

FAILS=0
check() { # check <name> <ok> <detail>
    local name="$1" ok="$2" detail="$3"
    if [[ "$ok" == "1" ]]; then
        printf "PASS  %-34s %s\n" "$name" "$detail"
    else
        printf "FAIL  %-34s %s\n" "$name" "$detail"
        FAILS=$((FAILS + 1))
    fi
}

count_correct() { # count_correct <judged.jsonl>
    local jf
    jf="$(win "$1")"
    "$PY" -c "
import json, sys
n = corr = 0
with open(sys.argv[1], encoding='utf-8') as fh:
    for line in fh:
        if not line.strip():
            continue
        rec = json.loads(line)
        n += 1
        corr += bool(rec['autoeval_label']['label'])
print(f'{corr} {n}')
" "$jf"
}

expected_cf() { # expected_cf <correct> <total>
    [[ "$1" -eq "$2" && "$3" -eq "$4" ]]
}

# --- 1. Dataset identity (§2) ------------------------------------------------
DS="$ARTIFACTS/data/longmemeval_s_cleaned.json"
DS_SHA_EXPECTED="d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
if [[ -f "$DS" ]]; then
    DS_SHA_ACTUAL="$(sha256sum "$DS" | awk '{print $1}')"
    if [[ "$DS_SHA_ACTUAL" == "$DS_SHA_EXPECTED" ]]; then
        check "dataset sha256 (§2)" 1 "${DS_SHA_ACTUAL:0:12}… ok"
    else
        check "dataset sha256 (§2)" 0 "expected ${DS_SHA_EXPECTED:0:12}… got ${DS_SHA_ACTUAL:0:12}…"
    fi
else
    check "dataset sha256 (§2)" 0 "missing $DS"
fi

# --- 2. Headline 70.4% (§1/§8) -----------------------------------------------
J="$REPRO_DIR/judged_capexp_c1500_k96_gpt4o.jsonl"
read -r C T <<<"$(count_correct "$J")"
if expected_cf "$C" 352 "$T" 500; then
    check "headline 70.4% ($C/$T)" 1 "352/500 == 0.7040"
else
    check "headline 70.4% ($C/$T)" 0 "expected 352/500"
fi

# --- 3. Temporal slices (§1) -------------------------------------------------
J="$REPRO_DIR/judged_temporal_flash_flat_75_gpt4o.jsonl"
read -r C T <<<"$(count_correct "$J")"
expected_cf "$C" 52 "$T" 75 && check "temporal 52/75" 1 "52/75" \
    || check "temporal 52/75" 0 "got $C/$T"

J="$REPRO_DIR/judged_temporal_flash_flat_58_gpt4o.jsonl"
read -r C T <<<"$(count_correct "$J")"
expected_cf "$C" 57 "$T" 58 && check "temporal 57/58" 1 "57/58" \
    || check "temporal 57/58" 0 "got $C/$T"

# --- 4. Stronger-answerer probe (§1) -----------------------------------------
J="$REPRO_DIR/judged_temporal_dsv4pro_55_gpt4o.jsonl"
read -r C T <<<"$(count_correct "$J")"
expected_cf "$C" 43 "$T" 55 && check "dsv4-pro probe 43/55" 1 "43/55" \
    || check "dsv4-pro probe 43/55" 0 "got $C/$T"

# --- 5. Cost axes (§9) -------------------------------------------------------
CACHE="$ARTIFACTS/cache_capexp_k96.jsonl.phaseA.jsonl"
if [[ -f "$CACHE" ]]; then
    NOFLOOR="$("$PY" "$(win "$REPRO_DIR/cost_axes.py")" "$(win "$CACHE")" --k 96 --cap 1500 --floor 0.0 \
        | awk '/context/ {print $3}' | tr -d ',')"
    FLOORED="$("$PY" "$(win "$REPRO_DIR/cost_axes.py")" "$(win "$CACHE")" --k 96 --cap 1500 --floor 0.30 \
        | awk '/context/ {print $3}' | tr -d ',')"
    check "cost axes no-floor (§9)" \
        "$([[ -n "${NOFLOOR:-}" ]] && awk -v a="$NOFLOOR" 'BEGIN {exit !(a>=83564 && a<=83584)}' && echo 1 || echo 0)" \
        "chars/q=$NOFLOOR (expect 83,574)"
    check "cost axes +floor (§9)" \
        "$([[ -n "${FLOORED:-}" ]] && awk -v a="$FLOORED" 'BEGIN {exit !(a>=58440 && a<=58460)}' && echo 1 || echo 0)" \
        "chars/q=$FLOORED (expect 58,450)"
else
    check "cost axes (§9)" 0 "missing $CACHE"
fi

# --- Verdict ----------------------------------------------------------------
echo
if [[ "$FAILS" -eq 0 ]]; then
    echo "REPRODUCIBILITY VERIFIED — every claim in BENCHMARK_REPRODUCIBILITY.md reproduced."
    exit 0
else
    echo "REPRODUCIBILITY BROKEN — $FAILS check(s) failed; investigate before quoting numbers."
    exit 1
fi