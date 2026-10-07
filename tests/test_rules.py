"""Tests for Gopher/C4/FineWeb rule filters and the classifier scorers."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from sanitizer_pro import Sanitizer, SanitizerConfig
from sanitizer_pro.rules import (
    c4_quality, check_rules, fineweb_quality, gopher_quality, gopher_repetition,
    resolve_rule_sets, tokenize,
)
from sanitizer_pro.scoring import FastTextScorer, FineWebEduScorer, make_scorer
from sanitizer_pro.utils import ConfigurationError

REPO = Path(__file__).resolve().parent.parent

SENTENCES = [
    "The river carried sediment from the mountains to the delta over thousands of years.",
    "Farmers in the valley learned to plant their crops after the seasonal floods receded.",
    "Historians have studied the irrigation records to understand how the economy grew.",
    "Each generation improved the canals and shared that knowledge with their neighbours.",
    "Today the region remains one of the most productive agricultural areas in the world.",
    "Researchers continue to measure how climate change affects the timing of the floods.",
]
GOOD_DOC = "\n".join(SENTENCES)


class TestTokenizer:
    def test_spacy_like_tokens(self):
        assert tokenize("It's 3.5 -- ok...") == ["It", "'s", "3.5", "--", "ok", "..."]

    def test_cjk_per_character(self):
        assert tokenize("中文 ok") == ["中", "文", "ok"]


class TestRules:
    def test_good_document_passes_everything(self):
        assert check_rules(GOOD_DOC, resolve_rule_sets(['all'])) is None

    @pytest.mark.parametrize("text,reason", [
        ("too short to be a document", "gopher_short_doc"),
        ("\n".join("- " + s for s in SENTENCES), "gopher_too_many_bullets"),
        ("\n".join(s + "..." for s in SENTENCES), "gopher_too_many_end_ellipsis"),
        (GOOD_DOC + " #" * 40, "gopher_too_many_hashes"),
        (" ".join(["xylophone quartz"] * 40), "gopher_enough_stop_words"),
    ])
    def test_gopher(self, text, reason):
        assert gopher_quality(text) == reason

    def test_gopher_repetition(self):
        assert gopher_repetition(GOOD_DOC + "\n\n" + GOOD_DOC) == "dup_para_frac"
        assert gopher_repetition(GOOD_DOC + "\n" + GOOD_DOC) == "dup_line_frac"
        spam = " ".join(["buy cheap watches now"] * 30)
        assert gopher_repetition(spam).startswith(("top_", "duplicated_"))

    def test_c4(self):
        assert c4_quality(GOOD_DOC + "\nfunction f() { return 1; } is shown here.") == "curly_bracket"
        assert c4_quality(GOOD_DOC + "\nLorem ipsum dolor sit amet, consectetur.") == "lorem_ipsum"
        assert c4_quality("\n".join(SENTENCES[:3])) == "too_few_sentences"
        # policy/javascript lines are excluded from the sentence count, not fatal
        assert c4_quality(GOOD_DOC + "\nPlease enable JavaScript to view the site.") is None

    def test_fineweb(self):
        assert fineweb_quality(GOOD_DOC) is None
        assert fineweb_quality("\n".join(s.rstrip('.') for s in SENTENCES)) == "line_punct_ratio"
        assert fineweb_quality("\n".join(["Menu.", "Home.", "About us.", "Contact."] * 3)) \
            in ("short_line_ratio", "char_dup_ratio")

    def test_unknown_rule_set(self):
        with pytest.raises(ConfigurationError, match="Unknown quality rule"):
            resolve_rule_sets(['gopher', 'refinedweb'])


class TestRulesInPipeline:
    def test_reason_and_stats(self):
        cfg = SanitizerConfig(min_chars=10, min_words=3, min_unique_ratio=0.0,
                              quality_rules=['gopher', 'c4'])
        with Sanitizer(cfg) as s:
            assert s.process_record({"text": GOOD_DOC}).kept
            r = s.process_record({"text": "\n".join("- " + x for x in SENTENCES)})
        assert not r.kept and r.reason == 'rules:gopher:gopher_too_many_bullets'
        assert s.stats.filtered_rules == 1
        assert s.stats.to_dict()['rule_failures'] == {'gopher:gopher_too_many_bullets': 1}

    def test_rules_see_text_beyond_8k(self):
        long_doc = (GOOD_DOC + "\n") * 40 + "Lorem ipsum dolor sit amet, consectetur adipiscing."
        cfg = SanitizerConfig(max_chars=10**6, min_unique_ratio=0.0, quality_rules=['c4'])
        with Sanitizer(cfg) as s:
            assert s.process_record({"text": long_doc}).reason == 'rules:c4:lorem_ipsum'

    def test_cli_summary_and_report(self, tmp_path):
        inp, out, rep = tmp_path / "in.jsonl", tmp_path / "out.jsonl", tmp_path / "r.html"
        inp.write_text(json.dumps({"text": GOOD_DOC}) + "\n" +
                       json.dumps({"text": "\n".join("- " + x for x in SENTENCES)}) + "\n")
        r = subprocess.run([sys.executable, '-m', 'sanitizer_pro', '--input', str(inp),
                            '--output', str(out), '--quality-rules', 'all', '--report', str(rep),
                            '--no-progress'], capture_output=True, text=True, cwd=str(REPO))
        assert r.returncode == 0, r.stderr
        assert 'Filtered (rules)        : 1' in r.stderr
        assert 'gopher:gopher_too_many_bullets=1' in r.stderr
        assert 'Quality rule failures' in rep.read_text()


class FakeFastText:
    def predict(self, line, k=-1):
        hq = 0.9 if "learn" in line else 0.05
        return ('__label__hq', '__label__cc'), [hq, 1 - hq]


class TestScorers:
    def test_fineweb_edu_mapping(self):
        s = FineWebEduScorer(_score_fn=lambda t: {'a': 3.7, 'b': -0.4, 'c': 6.2}[t])
        assert (s.score('a'), s.score('b'), s.score('c')) == (0.74, 0.0, 1.0)
        assert s.raw_score('a') == 3.7 and s.score('  ') == 0.0

    def test_fasttext_positive_label(self):
        s = FastTextScorer('dclm', _model=FakeFastText())
        assert s.score("we learn things") == 0.9 and s.score("spam") == 0.05

    def test_generic_fasttext_needs_label_and_model(self):
        with pytest.raises(ConfigurationError, match="quality-label"):
            make_scorer('fasttext')
        with pytest.raises(ConfigurationError, match="quality-model"):
            make_scorer('fasttext', label='hq')

    def test_label_prefix_added(self):
        assert FastTextScorer('fasttext', label='hq', _model=FakeFastText()).label == '__label__hq'

    def test_unknown_scorer(self):
        with pytest.raises(ConfigurationError):
            SanitizerConfig(quality_scorer='gpt').validate()


def _cached(repo, filename):
    try:
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache(repo, filename), str)
    except ImportError:
        return False


EDU = ("Photosynthesis is the process by which green plants use sunlight to synthesize food "
       "from carbon dioxide and water. In this lesson, students will learn how chlorophyll "
       "absorbs light energy and why oxygen is released as a by-product.")
SPAM = "BUY NOW!!! best cheap watches free shipping click here click here limited offer"


@pytest.mark.skipif(not _cached('HuggingFaceFW/fineweb-edu-classifier', 'model.safetensors'),
                    reason="FineWeb-Edu classifier not downloaded")
def test_live_fineweb_edu():
    pytest.importorskip("torch")
    s = make_scorer('fineweb-edu')
    assert s.score(EDU) > 0.6 > 0.1 > s.score(SPAM)


@pytest.mark.skipif(not _cached('mlfoundations/fasttext-oh-eli5',
                                'openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin'),
                    reason="DCLM fastText classifier not downloaded")
def test_live_dclm():
    pytest.importorskip("fasttext")
    s = make_scorer('dclm')
    assert s.score(EDU) > 0.9 and s.score(SPAM) < 0.1
