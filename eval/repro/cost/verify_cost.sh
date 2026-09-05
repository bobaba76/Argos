#!/usr/bin/env bash
# verify_cost.sh — reproducibility wrapper for the cost benchmark suite.
#
# Checks prerequisites, runs both arms, generates the report, and
# asserts outputs exist. This script SPENDS OpenRouter credits (~$2-5).
# It is intentionally separate from ../verify_repro.sh (which verifies
# accuracy claims and should not make paid API calls).
#
# Usage:
#   export OPENROUTER_API_KEY="sk-or-..."
#   bash eval/repro/cost/verify_cost.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Cost Benchmark Reproducibility Check ==="
echo ""

# 1. Check API key
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "ERROR: OPENROUTER_API_KEY not set. Export it before running."
    echo "  export OPENROUTER_API_KEY=\"sk-or-...\""
    exit 1
fi
echo "[1/5] API key present"

# 2. Check Python + requests
python -c "import requests" 2>/dev/null || {
    echo "ERROR: 'requests' package not installed. Run: pip install requests"
    exit 1
}
echo "[2/5] Python + requests available"

# 3. Dry-run validation
echo "[3/5] Dry-run validation..."
python run_token_arm.py --dry-run
python run_agentic_arm.py --dry-run
echo ""

# 4. Run both arms
echo "[4/5] Running token-economics arm..."
python run_token_arm.py
echo ""
echo "[4/5] Running agentic arm..."
python run_agentic_arm.py
echo ""

# 5. Generate report
echo "[5/5] Generating report..."
python report.py
echo ""

# Assert outputs
assert_file() {
    if [ ! -s "$1" ]; then
        echo "ASSERT FAILED: $1 is missing or empty"
        exit 1
    fi
}

assert_file "COST_REPORT.md"
assert_file "results/token_arm_deepseek-v4-flash_raw.jsonl"
assert_file "results/token_arm_glm-5.3-flash_raw.jsonl"
assert_file "results/agentic_arm_deepseek-v4-flash_raw.jsonl"
assert_file "results/agentic_arm_glm-5.3-flash_raw.jsonl"

echo ""
echo "=== PASS: All outputs generated ==="
echo "Report: $SCRIPT_DIR/COST_REPORT.md"
