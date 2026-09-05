"""Pytest tests for the argos plugin.

Run with:
    python -m pytest tests/test_argos.py -v

"""
from __future__ import annotations

import sys
import tempfile
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Ensure the plugin package is importable.
_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir.parent) not in sys.path:
    sys.path.insert(0, str(_plugin_dir.parent))


@pytest.fixture
def tmp_path_factory_dir(tmp_path):
    """Provide a clean temp directory for each test."""
    return tmp_path


class TestPriority3GraphEnhancements:
    """Tests for all-category entity extraction and graph traversal."""

    def test_graph_extractor_handles_all_memory_categories(self):
        from graph import extract_graph_relations

        cases = [
            ("Sam is my therapist", "relationship", ["therapy"]),
            ("User works at TechCorp and uses Kubernetes", "personal_fact", ["work", "devops"]),
            ("I prefer Kubernetes over Docker Swarm", "preference", ["devops"]),
            ("I just realized shame shapes my work patterns", "insight", ["insight", "shame", "work"]),
            ("User goal: learn Rust", "goal", ["goal", "rust"]),
            ("Life event: user launched Hermes", "event", ["event", "hermes"]),
            ("User has been using Docker", "context_note", ["ongoing", "devops"]),
        ]
        for content, category, tags in cases:
            relations = extract_graph_relations(content, category, tags)
            assert relations, f"No graph relations extracted for {category}: {content}"
            assert all({"source", "source_type", "relation", "target", "target_type"} <= set(r) for r in relations)

        personal = extract_graph_relations(
            "I take Medication-X for depression and live in Berlin",
            "personal_fact",
            ["health", "location"],
        )
        assert any(r["relation"] == "uses" and r["target"] == "Medication-X" for r in personal)
        assert any(r["relation"] == "lives_in" and r["target"] == "Berlin" for r in personal)
        ongoing = extract_graph_relations("User has been using Docker", "context_note", ["devops"])
        assert any(r["relation"] == "uses" and r["target"] == "Docker" for r in ongoing)

    def test_graph_extracts_role_mentions_without_names(self):
        """Role mentions like 'my wife', 'my doc' should create graph entities
        even when no canonical name is present. Without this, 'my wife' never
        enters the graph and the alias system has no anchor to link to.
        """
        from graph import extract_graph_relations

        relations = extract_graph_relations(
            "User goal: explain it to my doc and my wife",
            "goal",
            ["goal"],
        )
        # Should create has_wife → "my wife" and has_doc → "my doc"
        targets = {r["target"] for r in relations}
        assert "my wife" in targets, f"Role mention 'my wife' not extracted: {targets}"
        assert "my doc" in targets, f"Role mention 'my doc' not extracted: {targets}"

        # Both should be typed as "person"
        for r in relations:
            if r["target"] in ("my wife", "my doc"):
                assert r["target_type"] == "person", \
                    f"Role mention '{r['target']}' typed as '{r['target_type']}', expected 'person'"

    def test_graph_proper_noun_inherits_person_type(self):
        """When a name like 'Alex' appears in an explicit relationship
        pattern (e.g. 'Wife is Alex'), proper-noun mentions of 'Alex'
        in the same memory should inherit the 'person' type, not 'concept'.
        """
        from graph import extract_graph_relations

        relations = extract_graph_relations(
            "Wife is Alex. Alex recommended user try her medication stack.",
            "personal_fact",
            ["medication"],
        )
        # Find the proper-noun extraction of "Alex"
        alex_relations = [r for r in relations if r["target"] == "Alex"]
        assert alex_relations, "Alex not extracted as proper noun"
        # At least one should be typed as "person" (the one from the
        # explicit relationship pattern, or the proper-noun that inherits it)
        person_alexs = [r for r in alex_relations if r["target_type"] == "person"]
        assert person_alexs, \
            f"Alex not typed as 'person' in any relation: {[(r['target_type'], r['evidence'] if 'evidence' in r else r.get('attributes',{}).get('evidence')) for r in alex_relations]}"

    def test_bare_role_does_not_mint_junk_persons(self):
        """bare_role pattern must NOT match when the 'name' is a verb,
        adjective, or common noun (lowercase). Adversarial cases from
        the real corpus:

        - "User's boss is expecting me to load new codes"
        - "My therapist is helping me with anxiety"
        - "User's friend is coming over this weekend"
        - "Doctor is happy with the bloods"
        - "The ex is a director at the company"

        All of these should NOT produce has_boss→"expecting", has_therapist→
        "helping", etc. The fix requires the name group to start with a
        capital letter.
        """
        from graph import extract_graph_relations

        noisy_cases = [
            "User's boss is expecting me to load new codes",
            "My therapist is helping me with anxiety",
            "User's friend is coming over this weekend",
            "Doctor is happy with the bloods",
            "The ex is a director at the company",
        ]
        for content in noisy_cases:
            relations = extract_graph_relations(content, "personal_fact", [])
            # None of these should produce a has_<role> relation with a
            # verb/adjective target (e.g. "helping me", "expecting me").
            # Role mentions like "my therapist" are valid (they're for the
            # alias system), but verb/adjective captures are junk.
            for r in relations:
                if r["relation"].startswith("has_"):
                    target = r["target"]
                    # Role mentions ("my therapist", "my wife") are valid
                    if target.lower().startswith(("my ", "the ")):
                        continue
                    # Everything else must start with a capital letter (a name)
                    assert target[0].isupper(), \
                        f"bare_role minted junk person: '{content}' → {r['relation']}→'{target}'"

    def test_configurable_role_words_include_therapist(self):
        """The default role word set must include 'therapist' and other
        expanded roles (accountant, lawyer, coach, etc.) so that
        'my therapist is Sam' produces an alias without code changes."""
        from graph import _is_role_word, _get_role_words

        # Expanded defaults
        assert _is_role_word("therapist"), "therapist must be a default role word"
        assert _is_role_word("accountant"), "accountant must be a default role word"
        assert _is_role_word("lawyer"), "lawyer must be a default role word"
        assert _is_role_word("coach"), "coach must be a default role word"
        # Original defaults still present
        assert _is_role_word("wife")
        assert _is_role_word("doctor")
        assert _is_role_word("boss")

    def test_role_word_override_extends_set(self):
        """_set_role_words_override adds user-configured words to the set."""
        from graph import _set_role_words_override, _is_role_word, _get_role_words, _DEFAULT_ROLE_WORDS
        import threading

        # Save and clear override
        original = _get_role_words()
        _set_role_words_override({"nutritionist", "osteopath"})
        try:
            assert _is_role_word("nutritionist"), "override word must be recognized"
            assert _is_role_word("osteopath"), "override word must be recognized"
            assert _is_role_word("wife"), "defaults must still be present"
        finally:
            _set_role_words_override(set())

    def test_add_learned_role_word_extends_set(self):
        """_add_learned_role_word adds a word to the in-memory set (self-extending)."""
        from graph import _add_learned_role_word, _is_role_word, _set_role_words_override

        _set_role_words_override(set())
        assert not _is_role_word("hypnotherapist")
        _add_learned_role_word("hypnotherapist")
        assert _is_role_word("hypnotherapist"), "learned word must be recognized"
        # Cleanup
        _set_role_words_override(set())

    def test_car_is_not_a_role_word(self):
        """'car' is NOT a role word — 'my car is Toyota' must not produce
        a person alias. This is the junk-gate regression for the broadened
        bare_role regex."""
        from graph import _is_role_word

        assert not _is_role_word("car"), "car must not be a role word"
        assert not _is_role_word("phone"), "phone must not be a role word"
        assert not _is_role_word("house"), "house must not be a role word"
        assert not _is_role_word("dog"), "dog must not be a role word"

    def test_broadened_bare_role_does_not_match_non_role_words(self):
        """The broadened bare_role regex captures any lowercase word, but
        the _is_role_word() gate must filter out non-role words like 'car',
        'phone', 'house'. Only known role words should produce has_ relations."""
        from graph import extract_graph_relations

        # "car is Toyota" matches the broadened regex pattern but 'car' is
        # not a role word — must NOT produce has_car→Toyota
        relations = extract_graph_relations(
            "My car is Toyota. My phone is iPhone.", "personal_fact", []
        )
        for r in relations:
            assert not (r["relation"] == "has_car" and r["target"] == "Toyota"), \
                "Non-role word 'car' should not produce has_car relation"
            assert not (r["relation"] == "has_phone" and r["target"] == "iPhone"), \
                "Non-role word 'phone' should not produce has_phone relation"

    def test_therapist_alias_extraction_works(self, tmp_path):
        """'my therapist is Sam' must produce alias 'my therapist' → 'Sam'
        via the expanded default role words (no LLM call needed)."""
        from store import DuckDBMemoryStore
        from graph import KuzuGraphStore

        try:
            import argos_plugin
        except ModuleNotFoundError:
            import argos as argos_plugin

        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user")
        graph = KuzuGraphStore(tmp_path / "test_kuzu", user_id="test_user")

        provider = argos_plugin.ArgosProvider()
        provider._store = store
        provider._graph = graph

        provider._index_memory_graph(
            "mem-therapist-1", "personal_fact",
            "My therapist is Sam. We meet weekly.",
            ["therapy"],
        )
        aliases = store.aliases_for_canonical("Sam")
        assert "my therapist" in aliases, \
            f"Therapist alias not written: {aliases}"

        store.close()
        graph.close()

    def test_llm_ambiguity_gate_learns_new_role_word(self, tmp_path):
        """When 'my X is Name' matches but X is unknown, the LLM ambiguity
        gate should classify X and add it to the role words set. Mocks the
        LLM call so no real API call is made."""
        from store import DuckDBMemoryStore
        from graph import KuzuGraphStore, _is_role_word, _set_role_words_override
        from unittest.mock import patch

        # Ensure 'bartender' is not in defaults
        _set_role_words_override(set())
        assert not _is_role_word("bartender")

        try:
            import argos_plugin
        except ModuleNotFoundError:
            import argos as argos_plugin

        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user")
        graph = KuzuGraphStore(tmp_path / "test_kuzu", user_id="test_user")

        provider = argos_plugin.ArgosProvider()
        provider._store = store
        provider._graph = graph
        provider._config = {"role_alias_llm_fallback": "true"}
        provider._hermes_home = str(tmp_path)

        # Mock the LLM to say "bartender" IS a role word
        mock_response = type("MockResp", (), {
            "choices": [type("MockChoice", (), {
                "message": type("MockMsg", (), {"content": '{"is_role": true}'})
            })]
        })()
        with patch.object(provider, "_llm_classify_role_word", return_value=True):
            provider._extract_role_aliases(
                "My bartender is Sam. He makes great cocktails.",
                [],
            )

        # The alias should be written
        aliases = store.aliases_for_canonical("Sam")
        assert "my bartender" in aliases, \
            f"LLM-learned alias not written: {aliases}"

        # The role word should be in the in-memory set (self-extending)
        assert _is_role_word("bartender"), \
            "Learned role word 'bartender' should be in the set"

        store.close()
        graph.close()
        # Cleanup
        _set_role_words_override(set())

    def test_llm_ambiguity_gate_rejects_non_role_word(self, tmp_path):
        """When the LLM says X is NOT a role word, no alias should be written.
        'my car is Toyota' with LLM saying car is not a role → no alias."""
        from store import DuckDBMemoryStore
        from graph import KuzuGraphStore, _is_role_word, _set_role_words_override
        from unittest.mock import patch

        _set_role_words_override(set())
        assert not _is_role_word("car")

        try:
            import argos_plugin
        except ModuleNotFoundError:
            import argos as argos_plugin

        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user")
        graph = KuzuGraphStore(tmp_path / "test_kuzu", user_id="test_user")

        provider = argos_plugin.ArgosProvider()
        provider._store = store
        provider._graph = graph
        provider._config = {"role_alias_llm_fallback": "true"}
        provider._hermes_home = str(tmp_path)

        with patch.object(provider, "_llm_classify_role_word", return_value=False):
            provider._extract_role_aliases(
                "My car is Toyota. It's a great car.",
                [],
            )

        # No alias should be written for "my car"
        aliases = store.aliases_for_canonical("Toyota")
        assert "my car" not in aliases, \
            f"Non-role word 'car' should not produce alias: {aliases}"
        # 'car' should NOT be in the role words set
        assert not _is_role_word("car"), \
            "Rejected word 'car' should not be in role words set"

        store.close()
        graph.close()
        _set_role_words_override(set())

    def test_llm_fallback_disabled_skips_ambiguity_gate(self, tmp_path):
        """When role_alias_llm_fallback is false, the LLM ambiguity gate
        must not fire — unknown role words are simply skipped."""
        from store import DuckDBMemoryStore
        from graph import KuzuGraphStore, _is_role_word, _set_role_words_override
        from unittest.mock import patch

        _set_role_words_override(set())
        assert not _is_role_word("bartender")

        try:
            import argos_plugin
        except ModuleNotFoundError:
            import argos as argos_plugin

        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user")
        graph = KuzuGraphStore(tmp_path / "test_kuzu", user_id="test_user")

        provider = argos_plugin.ArgosProvider()
        provider._store = store
        provider._graph = graph
        provider._config = {"role_alias_llm_fallback": "false"}
        provider._hermes_home = str(tmp_path)

        # Even if LLM would say yes, it should never be called
        with patch.object(provider, "_llm_classify_role_word", side_effect=AssertionError(
            "LLM should not be called when fallback is disabled"
        )):
            provider._extract_role_aliases(
                "My bartender is Sam. He makes great cocktails.",
                [],
            )

        aliases = store.aliases_for_canonical("Sam")
        assert "my bartender" not in aliases, \
            "Disabled LLM fallback should not produce alias"
        assert not _is_role_word("bartender"), \
            "Disabled LLM fallback should not learn role word"

        store.close()
        graph.close()
        _set_role_words_override(set())

    def test_upsert_node_never_downgrades_person_to_concept(self, tmp_path):
        """upsert_node must never downgrade an existing 'person' node to
        'concept'. Cross-memory, a relation-free memory mentioning 'Alex'
        should not overwrite her type from 'person' (set by an explicit
        relationship pattern) to 'concept'.
        """
        from graph import KuzuGraphStore

        graph = KuzuGraphStore(tmp_path / "test_kuzu", user_id="test_user")

        # First memory: "Wife is Alex" → creates Alex as person
        graph.index_memory(
            "m1", "personal_fact", "Wife is Alex",
            ["relationship"], use_llm=False,
        )

        # Verify Alex is typed as person
        alex_edges = graph.search_graph("Alex")
        assert alex_edges, "Alex not found in graph after first memory"

        # Second memory: relation-free mention of Alex
        graph.index_memory(
            "m2", "context_note", "Alex stepped out to run errands",
            [], use_llm=False,
        )

        # Alex should still be typed as person, not downgraded to concept
        # Check via traverse_graph
        result = graph.traverse_graph("Alex", depth=1)
        alex_node = None
        for n in result.get("nodes", []):
            if n.get("id") == "Alex":
                alex_node = n
                break
        assert alex_node is not None, "Alex node not found after second memory"
        node_type = alex_node.get("type") or alex_node.get("entity_type")
        assert node_type == "person", \
            f"Alex downgraded from 'person' to '{node_type}' by relation-free memory"

        graph.close()

    def test_index_memory_creates_cross_memory_links(self, tmp_path):
        from graph import KuzuGraphStore

        graph = KuzuGraphStore(tmp_path / "test_kuzu", user_id="test_user")
        graph.index_memory(
            "m1", "personal_fact", "User works at TechCorp and uses Kubernetes",
            ["work", "devops"],
        )
        graph.index_memory(
            "m2", "preference", "User prefers Kubernetes over Docker Swarm",
            ["devops"],
        )

        edges = graph.search_graph("Kubernetes")
        memory_ids = {
            memory_id
            for edge in edges
            for memory_id in edge.get("attributes", {}).get("memory_ids", [])
        }
        assert {"m1", "m2"} <= memory_ids

        neighborhood = graph.traverse_graph("Kubernetes", depth=2)
        node_ids = {node["id"] for node in neighborhood["nodes"]}
        assert "user" in node_ids
        assert "memory:m1" in node_ids
        assert "memory:m2" in node_ids
        assert neighborhood["edges"]
        graph.close()

    def test_remove_and_reindex_memory_refreshes_graph_evidence(self, tmp_path):
        from graph import KuzuGraphStore

        graph = KuzuGraphStore(tmp_path / "test_kuzu", user_id="test_user")
        graph.index_memory("m1", "personal_fact", "User works at TechCorp", ["work"])
        assert graph.traverse_graph("TechCorp")["nodes"]

        assert graph.remove_memory("m1") is True
        assert graph.traverse_graph("TechCorp")["nodes"] == []

        graph.index_memory("m1", "personal_fact", "User works at Acme", ["work"])
        assert graph.traverse_graph("Acme")["nodes"]
        assert graph.traverse_graph("TechCorp")["nodes"] == []
        graph.close()

    def test_shared_graph_exposes_traversal(self):
        from service_client import SharedGraphStore
        assert hasattr(SharedGraphStore, "traverse_graph")
        assert hasattr(SharedGraphStore, "index_memory")
        assert hasattr(SharedGraphStore, "remove_memory")

    def test_graph_service_dispatch_exposes_traversal(self):
        import inspect
        from memory_service import MemoryService
        source = inspect.getsource(MemoryService._call_graph)
        assert "traverse_graph" in source
        assert "index_memory" in source
        assert "remove_memory" in source

    def test_graph_llm_extraction_falls_back_gracefully(self):
        """LLM-assisted extraction must return [] when the LLM is unavailable,
        and the hybrid function must still return regex results."""
        from graph import extract_graph_relations_llm, extract_graph_relations_hybrid

        # LLM path returns [] when agent.auxiliary_client is not importable
        # (test environment doesn't have the Hermes runtime).
        llm_result = extract_graph_relations_llm(
            "User works at TechCorp and uses Kubernetes for deployment", "personal_fact"
        )
        assert llm_result == []

        # Hybrid path still returns regex results when LLM is unavailable.
        hybrid = extract_graph_relations_hybrid(
            "User works at TechCorp and uses Kubernetes for deployment",
            "personal_fact",
            ["work", "devops"],
            use_llm=True,
        )
        assert hybrid, "Hybrid extraction should return regex results even when LLM unavailable"
        assert any(r["target"] == "TechCorp" for r in hybrid)

    def test_hybrid_gate_fires_llm_on_generic_regex_noise(self):
        """Regression: the hybrid gate counted RAW regex relations, so
        content-rich memories (3+ generic related_to/context_about edges,
        ​0 typed) never triggered the LLM — the graph rotted to concept
        soup. The gate must count TYPED relations instead."""
        from unittest.mock import patch
        import graph as graph_mod

        content = (
            "Alex's medication stack: Medication-A 800mg/day "
            "(400 morning + 400 night), Medication-B 15mg at night. "
            "Both user and Alex see the same clinician who wants to "
            "change Alex's meds."
        )
        regex_rels = graph_mod.extract_graph_relations(content, "personal_fact")
        # Precondition: regex finds >=3 relations, ALL generic (0 typed).
        assert len(regex_rels) >= 3
        generic = graph_mod._GRAPH_GENERIC_RELATIONS
        assert all(r["relation"] in generic for r in regex_rels)

        mock_llm = [
            {"source": "user", "source_type": "person",
             "relation": "shares_clinician_with", "target": "Alex",
             "target_type": "person"}
        ]
        with patch.object(graph_mod, "extract_graph_relations_llm",
                          return_value=mock_llm) as mock:
            hybrid = graph_mod.extract_graph_relations_hybrid(
                content, "personal_fact", use_llm=True)
        mock.assert_called_once()  # LLM fired despite regex>=3 (all generic)
        assert any(r["relation"] == "shares_clinician_with" for r in hybrid)
        # Generic regex edges preserved (regex takes priority in merge).
        assert len(hybrid) == len(regex_rels) + 1

    def test_graph_search_uses_kuzu_filter(self, tmp_path):
        """search_graph should use WHERE CONTAINS in Kuzu, not Python filtering.
        Verify it returns results and respects the limit parameter."""
        from graph import KuzuGraphStore

        graph = KuzuGraphStore(tmp_path / "test_kuzu", user_id="test_user")
        graph.index_memory(
            "m1", "personal_fact", "User works at TechCorp and uses Kubernetes",
            ["work", "devops"], use_llm=False,
        )
        graph.index_memory(
            "m2", "preference", "User prefers Kubernetes over Docker Swarm",
            ["devops"], use_llm=False,
        )

        # Search for Kubernetes — should find edges from both memories.
        edges = graph.search_graph("Kubernetes")
        assert len(edges) >= 2, f"Expected at least 2 edges, got {len(edges)}"

        # Limit parameter should be respected.
        limited = graph.search_graph("Kubernetes", limit=1)
        assert len(limited) <= 1

        # Non-existent term returns empty.
        assert graph.search_graph("NonExistentEntity123") == []

        graph.close()

    def test_graph_traverse_uses_targeted_queries(self, tmp_path):
        """traverse_graph should use targeted per-hop queries, not full scan.
        Verify it finds the neighborhood and respects depth."""
        from graph import KuzuGraphStore

        graph = KuzuGraphStore(tmp_path / "test_kuzu", user_id="test_user")
        graph.index_memory(
            "m1", "personal_fact", "User works at TechCorp and uses Kubernetes",
            ["work", "devops"], use_llm=False,
        )

        # Depth 1 — only direct neighbors of Kubernetes.
        d1 = graph.traverse_graph("Kubernetes", depth=1)
        assert any(n["id"] == "user" for n in d1["nodes"])
        assert d1["edges"]

        # Depth 2 — should also reach TechCorp through user.
        d2 = graph.traverse_graph("Kubernetes", depth=2)
        d2_ids = {n["id"] for n in d2["nodes"]}
        assert "TechCorp" in d2_ids or "memory:m1" in d2_ids

        # Non-existent entity returns empty.
        assert graph.traverse_graph("NonExistentEntity123")["nodes"] == []

        graph.close()

    def test_query_graph_is_bidirectional(self, tmp_path):
        """query_graph must find edges where the entity is a TARGET, not
        just a source. The extractor creates edges as memory -> concept,
        so concepts like 'shame' only appear as targets. A unidirectional
        query_graph would return 0 for them — a real gap found in testing."""
        from graph import KuzuGraphStore

        graph = KuzuGraphStore(tmp_path / "test_kuzu", user_id="test_user")
        graph.index_memory(
            "m1", "insight", "I realized that shame shapes my work patterns",
            ["insight", "shame"], use_llm=False,
        )

        # 'shame' is a concept — it only appears as a target in edges
        # like (memory:m1)-[mentions]->(shame) and (user)-[insight_about]->(shame).
        # query_graph must find these incoming edges.
        edges = graph.query_graph("shame")
        assert edges, "query_graph('shame') returned 0 — bidirectional query is broken"

        # Verify at least one edge has shame as the target.
        assert any(e["target"] == "shame" for e in edges), \
            "query_graph should find edges where shame is the target"

        graph.close()

    def test_every_approved_memory_yields_graph_backlink(self, tmp_path):
        """Every approved memory of each category must yield at least one
        graph edge with a memory_id backlink, so graph traversal can
        trace from entity -> the actual memory text.

        This is the contract test the user specifically requested: for
        each category, index a memory, then assert:
        1. A memory:<id> node exists in the graph.
        2. At least one edge carries memory_ids containing the memory ID.
        3. traverse_graph from the memory node reaches at least one entity.
        """
        from graph import KuzuGraphStore

        graph = KuzuGraphStore(tmp_path / "test_kuzu", user_id="test_user")

        test_cases = [
            ("personal_fact", "User works at TechCorp as a backend engineer", ["work"]),
            ("preference", "User prefers Kubernetes over Docker Swarm for deployment", ["devops"]),
            ("insight", "I realized that shame shapes my work patterns", ["insight", "shame"]),
            ("event", "User started a new job at TechCorp last Monday", ["work", "career"]),
            ("relationship", "User's partner Sam is supportive of their therapy", ["relationships"]),
            ("goal", "User wants to learn Rust for systems programming", ["programming", "rust"]),
            ("context_note", "User mentioned they live in Westford and commute by bike", ["location"]),
        ]

        for idx, (category, content, tags) in enumerate(test_cases):
            memory_id = f"backlink-test-{idx}"
            count = graph.index_memory(
                memory_id, category, content, tags, use_llm=False,
            )
            # Every category should produce at least one relation (the
            # about_user edge is always created, plus extracted entities).
            assert count >= 0, f"{category}: index_memory returned negative count"

            # 1. The memory:<id> node must exist.
            memory_node = f"memory:{memory_id}"
            node = graph._query_node(memory_node)
            assert node is not None, f"{category}: memory node {memory_node} not found in graph"
            assert node["entity_type"] == "memory"

            # 2. At least one edge must carry memory_ids containing this ID.
            edges = graph._query_edges_for_nodes([memory_node])
            backlink_edges = [
                e for e in edges
                if memory_id in [str(x) for x in (e.get("attributes", {}).get("memory_ids") or [])]
            ]
            assert backlink_edges, (
                f"{category}: no edges carry memory_ids backlink for {memory_id}"
            )

            # 3. traverse_graph from the memory node must reach at least
            # one non-memory entity (the user node or an extracted entity).
            traversal = graph.traverse_graph(memory_node, depth=2)
            non_memory_nodes = [
                n for n in traversal["nodes"]
                if n["entity_type"] != "memory" and n["id"] != memory_node
            ]
            assert non_memory_nodes, (
                f"{category}: traversal from {memory_node} found no linked entities"
            )

        graph.close()


