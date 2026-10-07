"""MinHash LSH near-duplicate detection (rensa or datasketch signatures)."""
import random

import pytest

from sanitizer_pro.dedup import MinHashDeduper, _optimal_bands, is_durable



def _available(name):
    try:
        __import__(name)
        return True
    except ImportError:
        return False


BACKENDS = [b for b in ("rensa", "datasketch") if _available(b)]
pytestmark = pytest.mark.skipif(not BACKENDS, reason="needs rensa or datasketch")


def doc(seed, n=200):
    rng = random.Random(seed)
    return " ".join(f"w{rng.randrange(3000)}" for _ in range(n))


def edit(text, k, seed=0):
    rng = random.Random(seed)
    words = text.split()
    for _ in range(k):
        words[rng.randrange(len(words))] = f"x{rng.randrange(3000)}"
    return " ".join(words)


@pytest.fixture(params=BACKENDS)
def backend(request):
    return request.param


class TestDetection:
    def test_exact_and_near_duplicates(self, backend):
        d = MinHashDeduper(threshold=0.8, backend=backend)
        a = doc(1)
        d.add(a)
        assert d.contains(a)
        assert d.contains(edit(a, 2))           # J ~ 0.97
        assert not d.contains(doc(2))
        assert not d.contains(edit(a, 40))      # J ~ 0.5

    def test_case_and_whitespace_insensitive(self, backend):
        d = MinHashDeduper(backend=backend)
        d.add(doc(3))
        assert d.contains("  " + doc(3).upper().replace(" ", "\n "))

    def test_short_texts(self, backend):
        d = MinHashDeduper(backend=backend)
        d.add("hello world")
        assert d.contains("Hello  world")
        assert not d.contains("goodbye world")
        d.add("")
        assert d.contains("")

    def test_backends_agree(self):
        if len(BACKENDS) < 2:
            pytest.skip("needs both backends")
        from sanitizer_pro.dedup import _shingles
        docs = [doc(i) for i in range(30)]
        probes = [(docs[i % 30], edit(docs[i % 30], i % 25, seed=i)) for i in range(90)]

        def jaccard(a, b):
            x, y = set(_shingles(a, 3)), set(_shingles(b, 3))
            return len(x & y) / len(x | y)

        # Different hash functions: decisions may differ only near the threshold.
        clear = [p for orig, p in probes if abs(jaccard(orig, p) - 0.8) > 0.1]
        assert len(clear) > 40
        results = []
        for b in BACKENDS:
            d = MinHashDeduper(threshold=0.8, backend=b)
            for t in docs:
                d.add(t)
            results.append([d.contains(p) for p in clear])
        assert results[0] == results[1]


class TestSQLiteStore:
    def test_persists_across_instances(self, backend, tmp_path):
        db = str(tmp_path / "fz.db")
        d = MinHashDeduper(backend=backend, store="sqlite", db_path=db)
        d.add(doc(1))
        assert d.contains(edit(doc(1), 1))      # found while still buffered
        d.close()
        d2 = MinHashDeduper(backend=backend, store="sqlite", db_path=db)
        assert d2.contains(edit(doc(1), 1))
        assert not d2.contains(doc(9))
        d2.close()

    def test_settings_mismatch_is_rejected(self, tmp_path):
        db = str(tmp_path / "fz.db")
        MinHashDeduper(threshold=0.8, backend=BACKENDS[0], store="sqlite", db_path=db).close()
        with pytest.raises(ValueError, match="different settings"):
            MinHashDeduper(threshold=0.5, backend=BACKENDS[0], store="sqlite", db_path=db)

    def test_rollback_forgets_later_records(self, backend, tmp_path):
        d = MinHashDeduper(backend=backend, store="sqlite", db_path=str(tmp_path / "fz.db"))
        d.add(doc(1))
        mark = d.high_water_mark()
        d.add(doc(2))
        assert d.rollback_to(mark) == 1
        assert d.contains(doc(1)) and not d.contains(doc(2))
        assert is_durable(d)
        assert not is_durable(MinHashDeduper(backend=backend))


class TestAccuracy:
    def test_recall_and_false_positives(self):
        from benchmarks.fuzzy_recall import evaluate, make_pairs
        r = evaluate(make_pairs(800, seed=3), 0.85, BACKENDS[0], "memory")
        assert r["recall"] >= 0.95
        assert r["fp_rate_far_below"] == 0.0

    def test_bands_match_datasketch_optimizer(self):
        lsh = pytest.importorskip("datasketch.lsh")
        for t in (0.5, 0.8, 0.9):
            assert _optimal_bands(t, 128) == lsh._optimal_param(t, 128, 0.5, 0.5)


def test_pipeline_fuzzy_dedup(backend):
    from sanitizer_pro import Sanitizer, SanitizerConfig
    cfg = SanitizerConfig(fuzzy_dedup=True, fuzzy_backend=backend, min_chars=0, min_words=0)
    recs = [{"text": doc(1)}, {"text": edit(doc(1), 1)}, {"text": doc(2)}]
    with Sanitizer(cfg) as s:
        out = list(s.process(recs))
    assert len(out) == 2 and s.stats.deduplicated == 1
