#!/usr/bin/env bash
# =============================================================================
# review-pr.sh — Hermes review of a Devin PR via OpenRouter
#
# Reads:  env vars PR_NUMBER, OPENROUTER_KEY
# Calls:  OpenRouter API with deepseek-v4-flash + the PR diff
# Does:   Posts review comment, adds label, optionally auto-merges
# =============================================================================
set -euo pipefail

# Guard
if [ -z "${OPENROUTER_KEY:-}" ]; then
  echo "::error::OPENROUTER_API_KEY secret not set"
  exit 1
fi

# Fetch PR details
echo "--- Fetching PR #${PR_NUMBER} ---"
PR_JSON=$(gh pr view "$PR_NUMBER" --json title,body,headRefName,baseRefName,additions,deletions,changedFiles 2>/dev/null || echo '{}')
PR_TITLE=$(echo "$PR_JSON" | jq -r '.title // "Untitled"')
PR_BODY=$(echo "$PR_JSON" | jq -r '.body // ""' | head -c 10000)
PR_BRANCH=$(echo "$PR_JSON" | jq -r '.headRefName // "unknown"')
PR_BASE=$(echo "$PR_JSON" | jq -r '.baseRefName // "main"')
PR_ADDITIONS=$(echo "$PR_JSON" | jq -r '.additions // 0')
PR_DELETIONS=$(echo "$PR_JSON" | jq -r '.deletions // 0')
PR_FILES=$(echo "$PR_JSON" | jq -r '.changedFiles // 0')

# Get the diff (truncated to 30k chars)
echo "--- Fetching diff ---"
DIFF=$(gh pr diff "$PR_NUMBER" 2>/dev/null | head -c 30000)

if [ -z "$DIFF" ]; then
  echo "::warning::Empty diff for PR #${PR_NUMBER}"
  DIFF="(no diff available — merge conflict or empty PR)"
fi

# Get changed files list
CHANGED_FILES=$(gh pr view "$PR_NUMBER" --json files --jq '.files[].path' 2>/dev/null | head -c 2000)

echo "--- Files changed ---"
echo "$CHANGED_FILES"
echo "--- Stats: +${PR_ADDITIONS} / -${PR_DELETIONS} across ${PR_FILES} files ---"

# Build the review prompt
SYSTEM_PROMPT=$(cat <<'SYSEOF'
You are Hermes, a senior engineer reviewing pull requests for the Argos project.

Argos is a hybrid memory system for Hermes Agent with multi-version support,
graph relationships, and LLM-based retrieval augmentation.

Codebase conventions:
- Python 3.11+, type hints required on all function signatures
- Tests in tests/ directory (pytest, parametrized where applicable)
- Error handling: specific exception types, never bare except
- Logging via structlog, not print
- DB operations use sqlite3 with row_factory = dict_factory
- DuckDB for analytics queries
- KuzuDB for graph operations (graph.py)

Your job: review the PR diff and decide if it is correct and ready to merge.

Focus on:
1. Logic errors, missing edge cases, race conditions
2. Correctness of DB operations (transactions, rollback on error)
3. Test coverage — does the PR add/update tests?
4. API compatibility (if changing existing interfaces)
5. Consistency with surrounding code patterns
6. Security issues (SQL injection, unchecked user input)

Output your verdict in this exact format (one of two):

## VERDICT: APPROVED
## REASONING: <2-3 sentence why>
## CONCERNS: <optional minor concerns>

## VERDICT: CHANGES_REQUESTED
## REASONING: <specific, actionable issues>
## CONCERNS: <what needs to change>

Be specific. Reference exact line numbers or code patterns. If tests are missing
or insufficient, flag it clearly. If the PR is straightforward and correct,
APPROVED is fine.
SYSEOF
)

USER_PROMPT="Review this PR against ${PR_BASE}:

Title: ${PR_TITLE}

Description:
${PR_BODY}

Stats: +${PR_ADDITIONS}/-${PR_DELETIONS} across ${PR_FILES} files

Files changed:
${CHANGED_FILES}

Diff:
\`\`\`diff
${DIFF}
\`\`\`

---

Evaluate: Is this correct? Any bugs, edge cases, missing tests, or style issues?"

# Escape for JSON
SYSTEM_JSON=$(echo "$SYSTEM_PROMPT" | jq -Rs '.')
USER_JSON=$(echo "$USER_PROMPT" | jq -Rs '.')

# Call OpenRouter
echo "--- Calling OpenRouter ---"
HTTP_RESPONSE=$(curl -s -w '\n%{http_code}' -X POST https://openrouter.ai/api/v1/chat/completions \
  -H "Authorization: Bearer $OPENROUTER_KEY" \
  -H "Content-Type: application/json" \
  -H "HTTP-Referer: https://github.com/bobaba76/Argos" \
  -H "X-Title: Argos PR Review" \
  -d "{
    \"model\": \"deepseek/deepseek-v4-flash\",
    \"messages\": [
      {\"role\": \"system\", \"content\": $SYSTEM_JSON},
      {\"role\": \"user\", \"content\": $USER_JSON}
    ],
    \"max_tokens\": 2048,
    \"temperature\": 0.3,
    \"reasoning\": {\"enabled\": true, \"effort\": \"max\"}
  }")

HTTP_CODE=$(echo "$HTTP_RESPONSE" | tail -1)
API_BODY=$(echo "$HTTP_RESPONSE" | sed '$d')

if [ "$HTTP_CODE" != "200" ]; then
  echo "::error::OpenRouter returned $HTTP_CODE"
  echo "$API_BODY" | jq '.' 2>/dev/null || echo "$API_BODY"
  exit 1
fi

REVIEW_TEXT=$(echo "$API_BODY" | jq -r '.choices[0].message.content // empty')
MODEL_NAME=$(echo "$API_BODY" | jq -r '.model // "deepseek/deepseek-v4-flash"')

if [ -z "$REVIEW_TEXT" ]; then
  echo "::error::Empty response from OpenRouter"
  exit 1
fi

echo "--- Review received ---"
echo "$REVIEW_TEXT"

# Parse verdict
if echo "$REVIEW_TEXT" | grep -q "VERDICT: APPROVED"; then
  VERDICT="APPROVED"
  EMOJI="✅"
else
  VERDICT="CHANGES_REQUESTED"
  EMOJI="🔧"
fi

# Extract reasoning and concerns
REASONING=$(echo "$REVIEW_TEXT" | awk '/^## REASONING:/{found=1; next} /^## CONCERNS:/{exit} found{print}' | head -c 2000)
CONCERNS=$(echo "$REVIEW_TEXT" | awk '/^## CONCERNS:/{found=1; next} /^## /{if(found) exit} found{print}' | head -c 4000)

# Ensure the verdict label exists, so a deleted/missing label cannot fail the run.
if [ "$VERDICT" = "APPROVED" ]; then
  NEW_LABEL="review-approved"
  LABEL_COLOR="0E8A16"
  LABEL_DESCRIPTION="Hermes approved this PR"
else
  NEW_LABEL="changes-requested"
  LABEL_COLOR="D93F0B"
  LABEL_DESCRIPTION="Hermes requested PR changes"
fi
gh label create "$NEW_LABEL" --color "$LABEL_COLOR" --description "$LABEL_DESCRIPTION" 2>/dev/null || true

# Add label
gh pr edit "$PR_NUMBER" --add-label "$NEW_LABEL" 2>/dev/null || true

# Build comment
COMMENT="## ${EMOJI} Hermes PR Review: ${VERDICT}

**PR:** [#${PR_NUMBER} ${PR_TITLE}](https://github.com/${GITHUB_REPOSITORY}/pull/${PR_NUMBER})

**Reasoning:**
${REASONING}

**Concerns:**
${CONCERNS}

---
_Reviewed via ${MODEL_NAME}_"

# Post comment
gh pr comment "$PR_NUMBER" --body "$COMMENT"

# Export outputs
echo "verdict=$VERDICT" >> "$GITHUB_OUTPUT"

# Auto-merge if approved (merge queue style — squash merge)
if [ "$VERDICT" = "APPROVED" ]; then
  echo "--- Auto-merging PR #${PR_NUMBER} ---"
  # Enable auto-merge (squash)
  gh pr merge "$PR_NUMBER" --squash --auto --subject "PR #${PR_NUMBER}: ${PR_TITLE}" 2>/dev/null \
    && echo "Auto-merge enabled for PR #${PR_NUMBER}" \
    || echo "::warning::Auto-merge not available (may need review approval or branch protection)"
fi

echo "Verdict: $VERDICT"