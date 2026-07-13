"""Tests for benchmark decontamination (n-gram overlap detection)."""
import json

import pytest

from sanitizer_pro.decontam import (
    KNOWN_BENCHMARKS, NGramIndex, build_index, iter_reference_file_texts,
    normalize_for_ngrams, resolve_benchmark_names,
)
from sanitizer_pro.utils import ConfigurationError

BENCH_Q = ("Natalia sold clips to 48 of her friends in April, and then she sold "
           "half as many clips in May. How many clips did Natalia sell altogether "
           "in April and May?")


class TestNormalization:
    def test_lowercase_and_punctuation(self):
        assert normalize_for_ngrams("Hello, World! It's FINE.") == \
            ['hello', 'world', 'it', 's', 'fine']

    def test_unicode_words_kept(self):
        assert normalize_for_ngrams("café über naïve") == ['café', 'über', 'naïve']


class TestNGramIndex:
    def test_exact_copy_detected(self):
        idx = NGramIndex(n=8)
        idx.add_reference(BENCH_Q)
        assert idx.is_contaminated(f"Solve this: {BENCH_Q} Show your work.")

    def test_case_and_punctuation_insensitive(self):
        idx = NGramIndex(n=8)
        idx.add_reference(BENCH_Q)
        assert idx.is_contaminated(BENCH_Q.upper().replace(",", " ;"))

    def test_unrelated_text_clean(self):
        idx = NGramIndex(n=8)
        idx.add_reference(BENCH_Q)
        assert not idx.is_contaminated(
            "A completely unrelated paragraph about ocean currents and their "
            "role in regulating coastal climates across the globe over decades.")

    def test_partial_overlap_below_ngram_clean(self):
        idx = NGramIndex(n=8)
        idx.add_reference(BENCH_Q)
        # shares a few words but never 8 consecutive ones
        assert not idx.is_contaminated(
            "Natalia sold lemonade at the fair and counted her earnings carefully "
            "before donating half to the school fund in June.")

    def test_min_hits_threshold(self):
        idx = NGramIndex(n=4, min_hits=3)
        idx.add_reference("alpha beta gamma delta epsilon")
        # only two colliding 4-grams available -> below min_hits of 3
        assert idx.contamination_hits("alpha beta gamma delta epsilon", max_hits=10) == 2
        assert not idx.is_contaminated("alpha beta gamma delta epsilon zeta" * 1)

    def test_short_reference_indexed_whole(self):
        idx = NGramIndex(n=8)
        idx.add_reference("What is the boiling point of water")  # 7 words < n
        assert idx.is_contaminated(
            "Answer briefly: what is the boiling point of water at sea level?")

    def test_too_short_reference_ignored(self):
        idx = NGramIndex(n=8)
        idx.add_reference("the answer is")  # 3 words < min_ref_words
        assert idx.ref_count == 0
        assert not idx.is_contaminated("of course the answer is forty two here")

    def test_empty_index_never_flags(self):
        idx = NGramIndex(n=8)
        assert not idx.is_contaminated(BENCH_Q)

    def test_invalid_params(self):
        with pytest.raises(ConfigurationError):
            NGramIndex(n=1)
        with pytest.raises(ConfigurationError):
            NGramIndex(min_hits=0)


class TestBenchmarkRegistry:
    def test_specs_are_complete(self):
        for name, spec in KNOWN_BENCHMARKS.items():
            assert spec.repo and '/' in spec.repo, name
            assert spec.parts and all(len(p) == 2 for p in spec.parts), name
            assert spec.fields, name

    def test_resolve_names(self):
        assert resolve_benchmark_names("mmlu, gsm8k") == ['mmlu', 'gsm8k']

    def test_resolve_all(self):
        assert set(resolve_benchmark_names("all")) == set(KNOWN_BENCHMARKS)

    def test_resolve_unknown(self):
        with pytest.raises(ConfigurationError, match="Unknown benchmark"):
            resolve_benchmark_names("mmlu,nosuchbench")



class TestReferenceFiles:
    def test_txt_refs(self, tmp_path):
        p = tmp_path / "refs.txt"
        p.write_text(f"{BENCH_Q}\nanother benchmark question with enough words here\n")
        texts = list(iter_reference_file_texts(str(p)))
        assert len(texts) == 2

    def test_jsonl_refs_extract_nested_strings(self, tmp_path):
        p = tmp_path / "refs.jsonl"
        p.write_text(json.dumps({"question": BENCH_Q, "meta": {"src": "gsm8k test split"}}) + '\n')
        texts = list(iter_reference_file_texts(str(p)))
        assert BENCH_Q in texts and "gsm8k test split" in texts

    def test_missing_file(self):
        with pytest.raises(ConfigurationError, match="not found"):
            list(iter_reference_file_texts("/nonexistent/refs.txt"))

    def test_build_index_from_files(self, tmp_path):
        p = tmp_path / "refs.txt"
        p.write_text(BENCH_Q + '\n')
        idx = build_index(ref_files=[str(p)], ngram=8)
        assert idx.ref_count == 1
        assert idx.is_contaminated(BENCH_Q)

    def test_build_index_requires_some_refs(self, tmp_path):
        p = tmp_path / "empty.txt"
        p.write_text("")
        with pytest.raises(ConfigurationError, match="no reference texts"):
            build_index(ref_files=[str(p)])
