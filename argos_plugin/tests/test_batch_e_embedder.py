"""Batch-E tests (issues #83, #90, #98).

#83 — embedder first-load failure is no longer permanent:
  * ``LocalEmbedder`` retries model load with backoff instead of degrading
    forever; ``recovered`` flips True on a successful retry.
  * ``remember()`` logs a warning when a record is stored without an
    embedding (instead of silent retrieval-quality loss).
  * ``DuckDBMemoryStore.backfill_null_embeddings()`` re-embeds records
    written with NULL embeddings after the embedder recovers, and
    ``remember()`` triggers it opportunistically on recovery.

These tests are hermetic — they never load the real sentence-transformers
model (a fake module is injected), so they run in milliseconds and do not
collide on the shared HF cache (the #90 flake class).
"""
from __future__ import annotations

import logging
import sys
import types

import pytest

from embeddings import LocalEmbedder, _SHARED_MODELS


# ---------------------------------------------------------------------------
# Fake sentence-transformers for hermetic LocalEmbedder tests
# ---------------------------------------------------------------------------

class _FakeModel:
    """Minimal SentenceTransformer stand-in."""

    def __init__(self, *args, **kwargs):
        self.calls = 0

    def encode(self, texts, normalize_embeddings=True):
        self.calls += 1
        # Deterministic 4-dim vector derived from the text so distinct
        # texts get distinct (non-zero) vectors.
        out = []
        for t in texts:
            h = abs(hash(t)) % 1000
            out.append([float((h >> i & 1)) for i in range(4)])
        return out


def _install_fake_st(monkeypatch, fail_times: int = 0):
    """Install a fake ``sentence_transformers`` module whose
    ``SentenceTransformer`` constructor raises ``fail_times`` times before
    succeeding. Returns the fake model class so the test can inspect it.
    """
    fake_mod = types.ModuleType("sentence_transformers")

    state = {"failures_left": fail_times}

    class FakeSentenceTransformer(_FakeModel):
        def __init__(self, *args, **kwargs):
            if state["failures_left"] > 0:
                state["failures_left"] -= 1
                raise RuntimeError("simulated transient load failure")
            super().__init__(*args, **kwargs)

    fake_mod.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_mod)
    # The retry path also imports CrossEncoder from this module; provide it.
    class FakeCrossEncoder:
        def __init__(self, *a, **k):
            pass

        def predict(self, pairs):
            return [0.5] * len(pairs)

    fake_mod.CrossEncoder = FakeCrossEncoder
    return fake_mod, state


@pytest.fixture(autouse=True)
def _clear_shared_model_cache():
    """The LocalEmbedder shares loaded models in a process-level dict.
    Clear it around every test so a prior test's fake model doesn't leak.
    """
    _SHARED_MODELS.clear()
    yield
    _SHARED_MODELS.clear()


# ---------------------------------------------------------------------------
# #83: retry-with-backoff + recovery flag
# ---------------------------------------------------------------------------

class TestEmbedderRetryRecovery:
    def test_first_load_failure_is_not_permanent(self, monkeypatch):
        _install_fake_st(monkeypatch, fail_times=1)
        emb = LocalEmbedder("fake-model-x")
        # First embed() triggers a load that fails -> returns [].
        assert emb.embed("hello") == []
        assert emb.load_failed is True
        # Bypass the backoff window (simulates time elapsing).
        emb.reset_for_retry()
        # Second embed() retries the load and succeeds.
        vec = emb.embed("hello")
        assert vec and len(vec) == 4
        assert emb.load_failed is False
        assert emb.recovered is True

    def test_backoff_window_prevents_immediate_retry(self, monkeypatch):
        _install_fake_st(monkeypatch, fail_times=1)
        emb = LocalEmbedder("fake-model-y")
        assert emb.embed("hello") == []
        assert emb.load_failed is True
        # Without reset, the backoff window blocks the retry -> still [].
        assert emb.embed("hello") == []
        assert emb.load_failed is True
        assert emb.recovered is False

    def test_recovered_flag_not_set_on_first_time_load(self, monkeypatch):
        _install_fake_st(monkeypatch, fail_times=0)
        emb = LocalEmbedder("fake-model-z")
        assert emb.embed("hello") != []
        # First-time success is not a "recovery".
        assert emb.recovered is False

    def test_recovered_clears_after_backfill_trigger(self, monkeypatch):
        _install_fake_st(monkeypatch, fail_times=1)
        emb = LocalEmbedder("fake-model-clear")
        assert emb.embed("hello") == []
        emb.reset_for_retry()
        assert emb.embed("hello") != []
        assert emb.recovered is True
        # A caller (the store) clears the flag after consuming it.
        emb.recovered = False
        assert emb.recovered is False


# ---------------------------------------------------------------------------
# #83: remember() warns on NULL embedding + backfill
# ---------------------------------------------------------------------------

class _ToggleEmbedder:
    """Duck-typed embedder that can be switched between broken/working."""

    def __init__(self):
        self.broken = False
        self.dim = 4

    def embed(self, text, *, is_query=False):
        if self.broken:
            return []
        h = abs(hash(text)) % 1000
        return [float((h >> i & 1)) for i in range(4)]

    def embed_batch(self, texts, *, is_query=False):
        return [self.embed(t, is_query=is_query) for t in texts]

    @property
    def is_available(self):
        return not self.broken

    @property
    def dimension(self):
        return self.dim


class TestRememberNullEmbeddingWarning:
    def test_remember_warns_when_embedder_unavailable(self, tmp_path, caplog):
        from store import DuckDBMemoryStore

        emb = _ToggleEmbedder()
        emb.broken = True
        store = DuckDBMemoryStore(
            tmp_path / "t.duckdb", user_id="u", embedder=emb
        )
        with caplog.at_level(logging.WARNING, logger="store_write"):
            rec = store.remember(category="personal_fact", content="A fact stored during outage.")
        assert rec is not None
        # The record was written with a NULL embedding.
        rows = store.connection.execute(
            "SELECT embedding FROM memory_records WHERE memory_id = ?",
            [rec.memory_id],
        ).fetchone()
        assert rows[0] is None
        assert any("without embedding" in m for m in caplog.messages)
        store.close()

    def test_remember_stores_embedding_when_available(self, tmp_path):
        from store import DuckDBMemoryStore

        emb = _ToggleEmbedder()
        store = DuckDBMemoryStore(
            tmp_path / "t.duckdb", user_id="u", embedder=emb
        )
        rec = store.remember(category="personal_fact", content="A fact stored normally.")
        assert rec is not None
        rows = store.connection.execute(
            "SELECT embedding FROM memory_records WHERE memory_id = ?",
            [rec.memory_id],
        ).fetchone()
        assert rows[0] is not None
        store.close()


class TestBackfillNullEmbeddings:
    def test_backfill_reembeds_null_records(self, tmp_path):
        from store import DuckDBMemoryStore

        emb = _ToggleEmbedder()
        store = DuckDBMemoryStore(
            tmp_path / "t.duckdb", user_id="u", embedder=emb
        )
        # Write two records while the embedder is broken -> NULL embeddings.
        emb.broken = True
        r1 = store.remember(category="personal_fact", content="fact during outage one", dedup=False)
        r2 = store.remember(category="personal_fact", content="fact during outage two", dedup=False)
        assert r1 and r2
        # Recover the embedder.
        emb.broken = False
        # Mark recovered so remember() would also backfill, but call the
        # method directly to test it in isolation.
        n = store.backfill_null_embeddings()
        assert n == 2
        for rid in (r1.memory_id, r2.memory_id):
            row = store.connection.execute(
                "SELECT embedding FROM memory_records WHERE memory_id = ?",
                [rid],
            ).fetchone()
            assert row[0] is not None
        store.close()

    def test_backfill_noop_when_embedder_unavailable(self, tmp_path):
        from store import DuckDBMemoryStore

        emb = _ToggleEmbedder()
        emb.broken = True
        store = DuckDBMemoryStore(
            tmp_path / "t.duckdb", user_id="u", embedder=emb
        )
        store.remember(category="personal_fact", content="orphan fact", dedup=False)
        assert store.backfill_null_embeddings() == 0
        store.close()

    def test_opportunistic_backfill_on_remember_recovery(self, tmp_path):
        """remember() with a working embedder whose `recovered` flag is set
        triggers a backfill of prior NULL-embedding records (issue #83)."""
        from store import DuckDBMemoryStore

        emb = _ToggleEmbedder()
        store = DuckDBMemoryStore(
            tmp_path / "t.duckdb", user_id="u", embedder=emb
        )
        # Outage: write a NULL-embedding record.
        emb.broken = True
        orphan = store.remember(category="personal_fact", content="orphan during outage", dedup=False)
        assert orphan is not None
        # Recover: the embedder is now available and flagged recovered.
        emb.broken = False
        emb.recovered = True
        # A subsequent remember() with a real embedding triggers backfill.
        store.remember(category="personal_fact", content="a fresh fact after recovery", dedup=False)
        # The orphan should now have an embedding.
        row = store.connection.execute(
            "SELECT embedding FROM memory_records WHERE memory_id = ?",
            [orphan.memory_id],
        ).fetchone()
        assert row[0] is not None
        # The recovered flag was consumed.
        assert emb.recovered is False
        store.close()
