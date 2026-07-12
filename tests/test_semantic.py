"""Tests for semantic (embedding-based) deduplication."""
import pytest

np = pytest.importorskip("numpy")

from sanitizer_pro.dedup import SemanticDeduper, make_deduper  # noqa: E402

# Deterministic fake embeddings: a few fixed directions in 8-d space.
_DIRS = {
    'cats': np.array([1, 0.1, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    'cats2': np.array([1, 0.18, 0.05, 0, 0, 0, 0, 0], dtype=np.float32),  # ~cats
    'finance': np.array([0, 0, 1, 0.1, 0, 0, 0, 0], dtype=np.float32),
    'weather': np.array([0, 0, 0, 0, 1, 0.1, 0, 0], dtype=np.float32),
}


def fake_embed(text: str):
    for key, vec in _DIRS.items():
        if key.rstrip('2') in text and ('2' in key) == ('again' in text):
            return vec
    return np.ones(8, dtype=np.float32)


def deduper(threshold=0.9):
    return SemanticDeduper(threshold=threshold, _embed_fn=fake_embed)


class TestSemanticDeduper:
    def test_exact_repeat_detected(self):
        d = deduper()
        assert not d.contains("all about cats")
        d.add("all about cats")
        assert d.contains("all about cats")

    def test_near_duplicate_detected(self):
        d = deduper(threshold=0.95)
        d.add("all about cats")
        # 'cats again' embeds to a vector ~0.996 cosine from 'cats'
        assert d.contains("all about cats again")

    def test_unrelated_not_flagged(self):
        d = deduper()
        d.add("all about cats")
        assert not d.contains("quarterly finance report")
        assert not d.contains("weather forecast tomorrow")

    def test_threshold_controls_sensitivity(self):
        strict = deduper(threshold=0.9999)
        strict.add("all about cats")
        assert not strict.contains("all about cats again")

    def test_empty_index_never_flags(self):
        assert not deduper().contains("anything at all")

    def test_close_clears_state(self):
        d = deduper()
        d.add("all about cats")
        d.close()
        assert not d.contains("all about cats")


def _model2vec_available() -> bool:
    try:
        import model2vec  # noqa: F401
        return True
    except ImportError:
        return False


def test_make_deduper_semantic_selects_backend():
    if _model2vec_available():
        assert isinstance(make_deduper('memory', semantic=True), SemanticDeduper)
    else:
        with pytest.raises(ImportError):
            make_deduper('memory', semantic=True)


@pytest.mark.skipif(not _model2vec_available(), reason="model2vec not installed")
def test_live_paraphrase_dedup():
    d = SemanticDeduper(threshold=0.8)
    d.add("The cat sat quietly on the warm mat near the window.")
    assert d.contains("A cat was sitting quietly on the warm mat by the window.")
    assert not d.contains("Quarterly revenue grew by twelve percent year over year.")
