# Argos

Persistent memory for AI agents, running on your own machine.

## What Argos is

Argos is a plugin for Hermes Agent that gives your agent a real memory. Instead of forgetting everything the moment a conversation ends, it keeps track of who you are, what you care about, and what you've told it before, and brings that back up when it's relevant. It's named after the hundred-eyed watchman from Greek myth who never slept and never missed a thing. Under the hood it combines a vector store, a knowledge graph, and local embeddings, but you don't need to know any of that to use it. You just talk to your agent, and it starts remembering.

## What it does for you

- **Remembers facts across sessions.** Tell it once that you're allergic to shellfish or that your team ships on Fridays, and it doesn't need you to say it again. Ask "what do you know about my work schedule?" three weeks later and it'll still know.
- **Tracks how things change over time.** If your job title changes or your address moves, Argos doesn't erase the old fact, it versions it. Ask "what changed about my role this year?" and it can walk you through the history instead of just giving you the latest snapshot.
- **Finds things by meaning, not just keywords.** Ask "what did we agree about the budget in March?" and it searches by intent, fuses that with keyword matches, and surfaces the memory even if you never used the word "budget" when you first mentioned it.
- **Understands relationships between things it remembers.** Ask "who works with my sister?" and it can traverse a graph of people, places, and concepts instead of just doing flat text lookup.
- **Knows the context of right now.** It's aware of the current time in your timezone, roughly where you are, the weather, and files you've recently touched, so it can answer things like "should I bring an umbrella later?" without you spelling out where "later" and "here" mean.
- **Catches your own realizations.** When you say something like "I just realized I've been avoiding this project because I'm scared of the scope," it logs that as an insight and can bring it back up later if the topic resurfaces. You can browse them with `/ilog` or pull one back into the conversation with `/revisit`.
- **Learns from its own feedback over time.** Once a day, gated and cost-capped, Argos looks over what it has remembered — and what you've marked helpful or dismissed — and *proposes* distilled patterns: insights, contradictions, and guardrails ("this project keeps slipping whenever X happens"). Nothing it proposes lands in memory without your approval; it all passes through the same suggestion tray you already approve.
- **Cleans up after itself, reversibly.** Old or duplicate memories get quarantined rather than deleted, and you can restore them if it turns out you needed them after all.

## How it earns trust

Memory systems fail when they either forget things that matter or remember things you never agreed to. Argos is built around not doing either.

Nothing becomes a memory silently. Every conversation turn gets mined for durable facts, first with regex, then with an LLM fallback if needed, but what comes out of that is a *proposal*, not a stored memory. An auxiliary review step quarantines obvious junk automatically, and everything else sits pending until you explicitly approve it. Argos never puts words in your mouth and calls them memories.

Updating a fact never destroys the old version. When something changes, Argos chains a new version onto the old one instead of overwriting it. You can look at the arc, compare versions, or just ask "what changed?" and get a compact history injected automatically. Your past self doesn't get erased just because your present self said something different.

Cleanup is reversible too. Maintenance can quarantine stale or duplicate memories, but quarantine isn't deletion. If the cleanup was wrong, you restore it and move on.

The distillation pass ("the dream") follows the same rule one level up. It can suggest patterns the store never stated outright, but it can only *propose*: its output lands in the same suggestion tray your extracted memories pass through, and nothing becomes active memory until you approve it. It doesn't edit, merge, or delete anything on its own — ever.

And underneath all of this is a habit worth mentioning: every feature in Argos is gated by an actual measurement in the eval harness. Several ideas that sounded good on paper, plain chronological ordering, a stronger graph boost, wider context injection, got turned off because the numbers said they didn't help. That's not a failure, it's the point. Features earn their place with evidence, not intuition.

## The tools

Argos exposes twelve tools your agent can call directly:

| Tool | What it's for |
|---|---|
| `memory_search` | Hybrid vector + keyword search |
| `memory_save` | Store a new memory |
| `memory_update` | Version an existing memory |
| `memory_delete` | Delete a memory; the previous version takes its place |
| `memory_chain` | Walk a memory's version history (arc, versions, diff) |
| `memory_graph_search` | Search the relationship graph |
| `memory_graph_query` | Run a direct graph traversal |
| `memory_candidate_list` | See pending extraction proposals |
| `memory_candidate_review` | Approve or reject a proposal |
| `memory_restore` | Bring a quarantined memory back |
| `memory_feedback` | Tag a memory helpful, dismissed, or incorrect |
| `memory_maintenance` | Run cleanup and dedup passes |

## The honest numbers

Argos's headline measurements, each under a stated protocol:

- **Chain-unfold, change-intent questions: ~93% recall / ~93% precision**
  (2026-08-20, its own eval harness). The residual false positives sit just
  inside the true-positive similarity band — one sits at 0.548 — so ~93%
  precision is a *diagnosed ceiling*: no cosine threshold separates them.
  Recall on this category moved via the intent matcher, not the thresholds.
- **99.6% of answer-bearing memories reach the top-96 candidates.** Recall
  (delivery) isn't the weak point.
- **Temporal questions: 82% correct** (133 questions, one uniform protocol),
  with date-anchored retrieval (time-expression re-ranking, added
  2026-08-21) improving that bucket further.
- **Overall: 70.4% on LongMemEval_S** (500 questions, judged by gpt-4o,
  default answerer) — and head-to-head runs showed the *answerer*, not the
  memory layer, is the bigger lever (a stronger answerer measured ~80+).
  The benchmark runs on synthetic conversation data; real conversations are
  messier and your results may differ.
- In the builder's own comparison against other open memory systems Argos
  came in second, behind Perseus (73.8) and ahead of Zep (63.8) — but those
  are vendor-published numbers under different protocols, so treat the
  cross-system gap as directional, not lab-controlled.

No claim here is "we beat vendor X." The numbers are there so you can judge
for yourself, not so we can win an argument.

## What it can't do yet

Be direct about this one: your memory data, embeddings, and graph all live locally, as flat files on your machine, and never go to a hosted memory vendor. But the LLM calls Argos makes for fact extraction, review, and the optional distillation pass currently go through whatever cloud model you've configured in Hermes. There's no native local-LLM support yet. If you want a fully offline setup, you're not there today, only the embedding step is offline right now.

## Quick start

1. Copy the plugin folder to your Hermes plugins directory (on Windows: `%LOCALAPPDATA%\hermes\plugins\hybrid_memory`).
2. Restart Hermes.
3. Run `hermes tools` and confirm the twelve `memory_*` tools show up.
4. Configure it in `hybrid_memory.json`, or through the settings UI under Memory -> Argos (Local).

Full walkthrough: [SETUP_GUIDE.md](SETUP_GUIDE.md).

One honest note: Argos is developed and tested on the maintainer's own build of Hermes. It only uses stock Hermes plugin APIs (the memory provider, the pre-call hook, the user-context injection path), so it's expected to work on a plain install, but a plain upstream build hasn't been through the test suite yet. If you hit something on a stock build, an issue is the fastest way to get it fixed.

## License

Argos is licensed under the **Business Source License 1.1** (BSL 1.1): free to read, modify, and self-host for personal and non-production use; production or commercial use requires a license from the author. It converts to **Apache 2.0** automatically on August 21, 2030. MIT-licensed components from the Hermes Agent base stay MIT. Full terms and the commercial-licensing contact path: [LICENSE](LICENSE).

## More docs

Settings and every configuration option live in [CONFIG_REFERENCE.md](CONFIG_REFERENCE.md), not in this file. For how the memory system actually works under the hood, see [MEMORY_SYSTEM.md](MEMORY_SYSTEM.md). If something breaks or you need to migrate, see [REINSTALL.md](REINSTALL.md). Tests live under `hybrid_memory_plugin`, run them with `python -m pytest tests/ -v`.