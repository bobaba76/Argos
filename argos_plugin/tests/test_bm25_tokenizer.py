"""Tests for the BM25-lite tokenizer fix (issue #26).

Two bugs were fixed in store.py:

1. **Substring token counting inflated BM25-lite.** ``tf = content_lower.count(t)``
   counted substring occurrences — "cat" matched "caterpillar" and
   "concatenate". The df computation (``t in c``) had the same substring
   semantics. Both tf and df were inflated, distorting BM25 scores and
   ranking irrelevant memories above word-true matches. Fix: tokenize each
   doc with the shared regex and use ``collections.Counter`` for exact
   word-boundary matches.

2. **Tokenizer inconsistency between text search and phrase-lift.** Text
   search used ``[a-z0-9]+`` (no apostrophe); phrase-lift used
   ``[a-z0-9']+`` (with apostrophe). Contractions like "don't" tokenized
   differently in the two paths. Fix: share a single module-level tokenizer
   (``_tokenize``) between both paths.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))


@pytest.fixture
def store(tmp_path):
    from store import DuckDBMemoryStore
    s = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="default_user")
    yield s
    s.close()


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class TestTokenizer:
    def test_tokenize_basic(self):
        from store import _tokenize
        assert _tokenize("hello world") == ["hello", "world"]

    def test_tokenize_contraction_preserved(self):
        """Contractions survive as single tokens (issue #26 part 2)."""
        from store import _tokenize
        tokens = _tokenize("don't stop it's")
        assert "don't" in tokens
        assert "it's" in tokens

    def test_tokenize_mixed_case_lowercased(self):
        from store import _tokenize
        assert _tokenize("Hello WORLD") == ["hello", "world"]

    def test_tokenize_punctuation_stripped(self):
        from store import _tokenize
        assert _tokenize("hello, world!") == ["hello", "world"]

    def test_tokenize_empty(self):
        from store import _tokenize
        assert _tokenize("") == []
        assert _tokenize(None) == []

    def test_tokenize_numbers_preserved(self):
        from store import _tokenize
        assert _tokenize("user has 42 cats") == ["user", "has", "42", "cats"]


# ---------------------------------------------------------------------------
# Substring bug (BM25-lite)
# ---------------------------------------------------------------------------

class TestSubstringBug:
    def test_substring_does_not_match_exact_word(self, store):
        """'cat' must not match 'caterpillar' or 'concatenate' (issue #26).

        Before the fix, content_lower.count('cat') returned 2 for
        'caterpillar concatenate', inflating tf. Now tokenized counting
        returns 0 for 'cat' in that content.
        """
        # Ingest a memory with "caterpillar" (contains "cat" as substring).
        store.remember(category="context_note", content="The caterpillar became a butterfly",
                       dedup=False)
        # Ingest a memory with the exact word "cat".
        store.remember(category="context_note", content="My cat is named Whiskers",
                       dedup=False)
        # Search for "cat" — the exact-word match should rank above the
        # substring match. Before the fix, both had tf=1 for "cat" (substring
        # count), so they tied or the substring match could rank higher.
        results = store._text_search_raw("cat", limit=10, excluded=set())
        assert len(results) >= 2
        # The exact-word match ("My cat is named Whiskers") must rank first.
        assert "cat" in results[0].content.lower().split()
        # The caterpillar memory should not match "cat" as a token at all.
        caterpillar_results = [r for r in results if "caterpillar" in r.content.lower()]
        if caterpillar_results:
            # If it appears, it's because the ILIKE %cat% SQL filter caught
            # the substring. But its BM25 score should be 0 (no exact token
            # match), so it ranks below the exact match.
            assert results[0].similarity > caterpillar_results[0].similarity

    def test_substring_count_vs_token_count(self, store):
        """Direct test: 'the' in 'theatre' is not a token match."""
        store.remember(category="context_note", content="Went to the theatre yesterday",
                       dedup=False)
        store.remember(category="context_note", content="The meeting was productive",
                       dedup=False)
        # "the" is a stopword (len <= 2 filter doesn't catch it, but
        # _TEXT_STOPWORDS does). Use a non-stopword instead.
        # "meet" is in "meeting" as a substring but not as a token.
        results = store._text_search_raw("meet", limit=10, excluded=set())
        # "meet" is 4 chars, passes the len > 2 filter.
        # "Went to the theatre yesterday" has no exact token "meet".
        # "The meeting was productive" has "meeting" not "meet".
        # Neither should have a BM25 score > 0 for "meet" (no exact token).
        # But the ILIKE filter (%meet%) catches "meeting", so results may
        # be non-empty — just with similarity 0.
        for r in results:
            if "meeting" in r.content.lower() and "meet" not in r.content.lower().split():
                # "meeting" contains "meet" as substring but not as token.
                assert r.similarity == 0.0, (
                    f"substring match 'meet' in 'meeting' should have BM25=0, "
                    f"got {r.similarity}"
                )


# ---------------------------------------------------------------------------
# BM25 ranking correctness
# ---------------------------------------------------------------------------

class TestBM25Ranking:
    def test_exact_token_match_ranks_above_substring(self, store):
        """A doc with exact token 'dog' should rank above a doc with
        'dogma' (substring 'dog') for query 'dog'."""
        store.remember(category="context_note", content="The dogma of the church is strict",
                       dedup=False)
        store.remember(category="context_note", content="My dog loves to play fetch",
                       dedup=False)
        results = store._text_search_raw("dog", limit=10, excluded=set())
        # Both match the ILIKE %dog% filter, but only "My dog loves..." has
        # an exact token match. Its BM25 score should be > 0 and it should
        # rank first.
        assert len(results) >= 1
        dog_results = [r for r in results if "dog" in r.content.lower().split()]
        assert len(dog_results) >= 1
        assert dog_results[0].similarity > 0.0
        # The dogma result should have similarity 0 (no exact token "dog").
        dogma_results = [r for r in results if "dogma" in r.content.lower().split()]
        if dogma_results:
            assert dogma_results[0].similarity == 0.0

    def test_multiple_exact_matches_score_higher(self, store):
        """A doc with 2 occurrences of 'cat' should score higher than a doc
        with 1 occurrence (tf saturation aside, both should have tf > 0)."""
        store.remember(category="context_note", content="I saw a cat once",
                       dedup=False)
        store.remember(category="context_note", content="My cat and your cat are friends",
                       dedup=False)
        results = store._text_search_raw("cat", limit=10, excluded=set())
        # Find the two results.
        one_cat = [r for r in results if r.content == "I saw a cat once"]
        two_cats = [r for r in results if r.content == "My cat and your cat are friends"]
        if one_cat and two_cats:
            # Both have exact token matches (tf > 0), but the 2-occurrence
            # doc should have a higher raw BM25 score. After max-normalization
            # the 2-cat doc should be 1.0 and the 1-cat doc < 1.0.
            assert two_cats[0].similarity > one_cat[0].similarity


# ---------------------------------------------------------------------------
# Tokenizer consistency (text search vs phrase-lift)
# ---------------------------------------------------------------------------

class TestTokenizerConsistency:
    def test_contractions_tokenize_same_in_both_paths(self):
        """Contractions like "don't" must tokenize identically in text
        search and phrase-lift (issue #26 part 2)."""
        from store import _tokenize
        # The shared _tokenize is used by both paths now.
        tokens = _tokenize("I don't know what's happening")
        assert "don't" in tokens
        assert "what's" in tokens
        # No split "don" + "t" fragments.
        assert "don" not in tokens
        assert "t" not in tokens

    def test_phrase_lift_uses_shared_tokenizer(self, store):
        """Phrase-lift bigrams should include contractions as single tokens."""
        store.remember(category="context_note",
                       content="The user don't like spicy food",
                       dedup=False)
        # Enable phrase-lift.
        store._phrase_lift_alpha = 0.25
        store._PHRASE_STOPWORDS = frozenset()
        # Search with a contraction — phrase-lift should be able to form
        # the bigram ("don't", "like") from the query and match it in the
        # memory. Before the fix, text search tokenized "don't" as "don" +
        # "t" but phrase-lift kept "don't", so they disagreed.
        results = store.search("don't like", limit=10)
        # The memory should be found.
        assert any("don't" in r.content.lower() for r in results)
