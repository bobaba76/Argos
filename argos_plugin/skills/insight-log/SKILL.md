---
name: insight-log
description: "Use when user shares a realization/insight; log it."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [insight, memory, self-reflection, personal, capture]
    category: memory
---

# Insight Log

Capture the user's personal realizations and insights automatically, then
surface them contextually in future sessions. The goal is zero-friction:
the user just talks, and durable insights get saved without any manual
filing, review ritual, or app to open.

## When to capture

Trigger when the user shares a **deep thought**, **realization**,
**insight**, or **big self-discovery**. Signal phrases include:

- "I just realised..." / "I just realized..."
- "deep thought:" / "big thought:"
- "that makes sense now..."
- "it just clicked that..."
- "I've been thinking about..." (when followed by a revelation)
- A paragraph that starts with a personal revelation about themselves,
  their patterns, their relationships, or their life

The user's self-monitoring guard is typically down during
these moments — often late at night or in a relaxed state. This is when
the best material surfaces. **Do not interrupt the flow.**

## How to capture

1. **Save verbatim-ish, raw, uncensored.** Their guard is down; do NOT
   clean it up, summarize it, judge it, or rephrase it into "proper"
   language. Preserve the user's own framing and energy. If they said it
   in slang, keep the slang. If they rambled, keep the ramble — just
   trim obvious repetition.

2. **Make it self-contained.** Someone reading it later with zero prior
   context should understand what was being said. Add minimal connecting
   words if needed, but do not rewrite.

3. **Save to memory** using the `memory` tool (or `memory_add`):
   - **category**: `insight`
   - **content**: the realization, verbatim-ish
   - **tags**: `["insight", "<YYYY-MM-DD>"]` plus any topical tags
     (e.g. `work`, `ex`, `shame`, `anxiety`, `identity`, `relationships`)

4. **Do NOT attach moral judgment.** Never say "that's concerning" or
   "you should talk to someone about that." Never discourage the user
   mid-flow. This is a capture zone, not a structured session. Acknowledge
   briefly ("got it, saving that") and let them continue.

5. **Do NOT require `update_memory`.** The insight log is append-only.
   If the user refines an insight later, save the new version as a new
   entry — do not try to edit the old one.

## When to recall

In future sessions, when the active conversation topic overlaps a saved
insight's tags or content, **proactively surface it**:

1. Use `memory_search` with the current topic to find relevant insights.
2. Or use `/insights` internally to fetch recent insights by tag.
3. When you find a match, bring it up naturally:

   > "You had a related thought on 2024-03-15: *'I redirect credit away
   > from myself because I'm afraid of being seen as arrogant, but
   > actually people respect confidence more than false humility.'*
   > That connects to what you're saying now about..."

4. **Do not force it.** Only surface insights when genuinely relevant to
   the current conversation. Don't dump the log unprompted. One
   well-timed recall is worth ten forced ones.

5. **Quote the original wording.** The user's own phrasing is more
   powerful than a summary. Use italics for the quoted insight.

## Slash commands

The user can also explicitly interact with the log:

- **`/ilog`** — list all saved insights, newest first. Shows date,
  tags, and first line of content. Useful for browsing. (Note: `/ilog`
  is used instead of `/insights` to avoid conflicting with the built-in
  usage-analytics `/insights` command.)
- **`/revisit`** — pick a random older insight (not surfaced recently)
  and bring it up conversationally. Surprise-recall mode: "Hey, you had
  this thought back in March..." Good for re-engaging with old
  realizations that might have new relevance.

## What counts as an insight (vs. a regular fact)

| Insight | Regular fact |
|---------|-------------|
| "I deflect praise when I'm unsure of myself" | "I use Vim as my editor" |
| "I over-prepare before big meetings" | "I take a daily medication" |
| "I take on other people\'s problems as my own" | "I work at a small studio" |
| "Shame is the through-line of my patterns" | "I live in Springfield" |

Insights are **self-observations about patterns, motivations, fears, and
realizations**. They're the "why" beneath the "what." Regular facts are
the "what." When in doubt, save it as `insight` — the category filter
keeps them from cluttering regular recall.