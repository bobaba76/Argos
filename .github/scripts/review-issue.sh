#!/usr/bin/env bash
# =============================================================================
# review-issue.sh — Hermes review of a GitHub issue via OpenRouter
#
# Reads:  env vars ISSUE_NUMBER, ISSUE_TITLE, ISSUE_BODY, OPENROUTER_KEY
# Calls:  OpenRouter API with deepseek-v4-flash
# Does:   Removes "needs-review" label, adds verdict label, posts comment
# =============================================================================
set -euo pipefail

# Guard
if [ -z "${OPENROUTER_KEY:-}" ]; then
  echo "::error::OPENROUTER_API_KEY secret not set"
  exit 1
fi

BODY="${ISSUE_BODY:0:25000}"

# Build the system prompt (single-line JSON-safe via jq)
SYSTEM_PROMPT=$(cat <<'SYSEOF'
You are Hermes, a senior engineer reviewing issues for the Argos project.

Argos is a hybrid memory system for Hermes Agent. It stores, retrieves, and manages
persistent memories with multi-version support, graph relationships, and LLM-based
retrieval augmentation.

Key codebase areas:
- store_write.py — memory write path (ingest, candidates, reviews, conflicts, tombstones)
- store_retrieval.py — memory search (embedding, graph boost, chain unfold, phrase lift, reranker)
- store_maintenance.py — maintenance, garbage collection, stats
- store_core.py / store_common.py — shared DB operations
- provider_session.py / provider_retrieval.py — session-level logic
- graph.py — KuzuDB knowledge graph integration
- rest_server.py / mcp_server.py — external API (REST + MCP)
- api_facade.py — auth, ACL, validation, audit

Your job: review each issue and decide if it is well-scoped, correctly identifies
a real problem, and is ready for Devin to implement.

Output your verdict in this exact format (one of three):

## VERDICT: APPROVED
## REASONING: <2-3 sentence why>
## FEEDBACK: <optional implementation notes>

## VERDICT: CHANGES_NEEDED
## REASONING: <what is missing or wrong>
## FEEDBACK: <specific suggestions>

## VERDICT: REJECTED
## REASONING: <why not valid>
## FEEDBACK: <alternative if applicable>

Be specific. Reference files or patterns you expect to see changed.
If the issue is vague or lacks detail, mark CHANGES_NEEDED.
SYSEOF
)

USER_PROMPT="Review this Argos issue:

Title: ${ISSUE_TITLE}

Body:
${BODY}

---

Evaluate: Is this issue well-scoped, does it identify a real problem, and is it ready for implementation?"

# Escape for JSON
SYSTEM_JSON=$(echo "$SYSTEM_PROMPT" | jq -Rs '.')
USER_JSON=$(echo "$USER_PROMPT" | jq -Rs '.')

# Call OpenRouter
echo "--- Calling OpenRouter ---"
HTTP_RESPONSE=$(curl -s -w '\n%{http_code}' -X POST https://openrouter.ai/api/v1/chat/completions \
  -H "Authorization: Bearer $OPENROUTER_KEY" \
  -H "Content-Type: application/json" \
  -H "HTTP-Referer: https://github.com/bobaba76/Argos" \
  -H "X-Title: Argos Issue Review" \
  -d "{
    \"model\": \"deepseek/deepseek-v4-flash\",
    \"messages\": [
      {\"role\": \"system\", \"content\": $SYSTEM_JSON},
      {\"role\": \"user\", \"content\": $USER_JSON}
    ],
    \"max_tokens\": 1024,
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
  NEW_LABEL="devin-ready"
  EMOJI="✅"
elif echo "$REVIEW_TEXT" | grep -q "VERDICT: REJECTED"; then
  VERDICT="REJECTED"
  NEW_LABEL="rejected"
  EMOJI="❌"
else
  VERDICT="CHANGES_NEEDED"
  NEW_LABEL="needs-revision"
  EMOJI="🔧"
fi

# Extract reasoning and feedback
REASONING=$(echo "$REVIEW_TEXT" | awk '/^## REASONING:/{found=1; next} /^## FEEDBACK:/{exit} found{print}' | head -c 2000)
FEEDBACK=$(echo "$REVIEW_TEXT" | awk '/^## FEEDBACK:/{found=1; next} /^## /{if(found) exit} found{print}' | head -c 4000)

# Ensure the verdict label exists, so a deleted/missing label cannot fail the run.
case "$NEW_LABEL" in
  devin-ready)
    LABEL_COLOR="0E8A16"
    LABEL_DESCRIPTION="Hermes approved: ready for Devin"
    ;;
  needs-revision)
    LABEL_COLOR="FBCA04"
    LABEL_DESCRIPTION="Hermes requested issue changes"
    ;;
  rejected)
    LABEL_COLOR="B60205"
    LABEL_DESCRIPTION="Hermes rejected this issue"
    ;;
esac
gh label create "$NEW_LABEL" --color "$LABEL_COLOR" --description "$LABEL_DESCRIPTION" 2>/dev/null || true

# Remove needs-review label
gh issue edit "$ISSUE_NUMBER" --remove-label "needs-review" 2>/dev/null || true

# Add verdict label
gh issue edit "$ISSUE_NUMBER" --add-label "$NEW_LABEL"

# Build comment
COMMENT="## ${EMOJI} Hermes Review: ${VERDICT}

**Issue:** [#${ISSUE_NUMBER} ${ISSUE_TITLE}](https://github.com/${GITHUB_REPOSITORY}/issues/${ISSUE_NUMBER})

**Reasoning:**
${REASONING}

**Feedback:**
${FEEDBACK}

---
_Reviewed via ${MODEL_NAME}_"

# Post Hermes review comment
gh issue comment "$ISSUE_NUMBER" --body "$COMMENT"

# The devin-ready label is the sole Devin trigger. Do not post a /devin
# comment here; that would require a second automation and could duplicate runs.

# Export for downstream steps
echo "verdict=$VERDICT" >> "$GITHUB_OUTPUT"
echo "review<<EOF_REVIEW" >> "$GITHUB_OUTPUT"
echo "$REVIEW_TEXT" >> "$GITHUB_OUTPUT"
echo "EOF_REVIEW" >> "$GITHUB_OUTPUT"

echo "Verdict: $VERDICT — ${REASONING:0:100}"