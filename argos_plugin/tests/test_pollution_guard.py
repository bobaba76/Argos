"""#459: the conftest import-state guard contains sys.modules fake leaks.

Regression pin for the full-suite failure class where a leaked fake
``sentence_transformers`` survived into later suites and every real
embedder load failed with "No module named 'sentence_transformers.models'".

The autouse ``_restore_import_state_after_test`` fixture in conftest.py
snapshots the embedding-stack module keys around every test; a fake
leaked WITHOUT monkeypatch (raw sys.modules assignment) must still be
contained. Test order inside this module is source order: the leak test
defines the poison, the next test proves it is gone.
"""
import sys
import types


def test_zzz_raw_fake_sentence_transformers_installed():
    """Leak a fake WITHOUT monkeypatch - the guard must contain it.

    Runs first: defines the poison that the next test proves is gone.
    """
    fake = types.ModuleType("sentence_transformers")
    sys.modules["sentence_transformers"] = fake  # deliberately leaked
    assert sys.modules["sentence_transformers"] is fake


def test_zzz_fake_did_not_survive_the_guard():
    """The raw leak above must not survive into this test.

    After restore, the key is either absent (never really imported) or the
    REAL package (has ``__file__``/``__path__``) - never a bare fake
    ModuleType, which is what breaks ``sentence_transformers.models``
    resolution for every later real embedder load.
    """
    st = sys.modules.get("sentence_transformers")
    if st is not None:
        assert getattr(st, "__file__", None) or getattr(st, "__path__", None), (
            "a leaked fake 'sentence_transformers' survived the "
            "import-state guard - real embedder loads will fail with "
            "'No module named sentence_transformers.models'"
        )
