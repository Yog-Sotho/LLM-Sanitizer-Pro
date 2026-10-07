"""Tests for language identification backends and code matching."""
import json
import os
import subprocess
import sys
import unicodedata
from pathlib import Path

import pytest

from sanitizer_pro import Sanitizer, SanitizerConfig
from sanitizer_pro.langid import (
    FastTextIdentifier, language_aliases, make_language_identifier, matches, normalize_filter,
)
from sanitizer_pro.utils import ConfigurationError

REPO = Path(__file__).resolve().parent.parent


class FakeFastText:
    """Stands in for a loaded fastText LID model."""

    def __init__(self, table):
        self.table, self.seen = table, []

    def predict(self, line, k=1):
        self.seen.append(line)
        for needle, (label, prob) in self.table.items():
            if needle in line:
                return (f'__label__{label}',), [prob]
        return ('__label__und_Zyyy',), [0.2]


class TestAliases:
    @pytest.mark.parametrize("detected,wanted,ok", [
        ('eng_Latn', 'en', True), ('eng_Latn', 'eng', True), ('eng_Latn', 'eng_latn', True),
        ('cmn_Hani', 'zh', True), ('yue_Hani', 'zh', True), ('arz_Arab', 'ar', True),
        ('pes_Arab', 'fa', True), ('nob_Latn', 'no', True), ('zh-cn', 'zh', True),
        ('en', 'en', True), ('fra_Latn', 'en', False), ('jpn_Jpan', 'zh', False),
        (None, 'en', False),
    ])
    def test_matches(self, detected, wanted, ok):
        assert matches(detected, normalize_filter([wanted])) is ok

    def test_aliases(self):
        assert language_aliases('__label__cmn_Hani') == {'cmn_hani', 'cmn', 'zh'}


class TestFastTextIdentifier:
    def test_one_line_truncated_input(self):
        fake = FakeFastText({'Bonjour': ('fra_Latn', 0.97)})
        ident = FastTextIdentifier('glotlid', _model=fake)
        assert ident.predict("Bonjour\ntout le monde") == ('fra_Latn', 0.97)
        assert fake.seen[-1] == "Bonjour tout le monde"
        ident.predict("x " * 5000)
        assert len(fake.seen[-1]) <= 2000

    def test_probability_clamped(self):
        ident = FastTextIdentifier('glotlid', _model=FakeFastText({'a': ('eng_Latn', 1.00001)}))
        assert ident.predict("a b c")[1] == 1.0

    def test_empty_text(self):
        assert FastTextIdentifier('x', _model=FakeFastText({})).predict("   ") == (None, 0.0)


class TestPipelineIntegration:
    GOOD = {
        'en': "The committee reviewed the proposal in detail and approved the budget plan.",
        'fr': "Le comité a examiné la proposition en détail et a approuvé le plan budgétaire.",
        'zh': "委员会详细审查了这项提案，并批准了预算计划，会议持续了整整一个下午的时间。",
    }

    def _sanitizer(self, monkeypatch, **kw):
        fake = FakeFastText({'committee': ('eng_Latn', 0.99), 'comité': ('fra_Latn', 0.98),
                             '委员会': ('cmn_Hani', 0.99)})
        import sanitizer_pro.core as core
        monkeypatch.setattr(core, 'make_language_identifier',
                            lambda backend, model: FastTextIdentifier(backend, _model=fake))
        monkeypatch.setattr('sanitizer_pro.langid.language_backend_available', lambda b: True)
        cfg = SanitizerConfig(min_chars=10, min_words=3, min_unique_ratio=0.0, **kw)
        return Sanitizer(cfg)

    def test_filter_by_iso1_codes_with_iso3_model(self, monkeypatch):
        with self._sanitizer(monkeypatch, lang_filter=['en', 'zh'], lang_backend='glotlid') as s:
            kept = {lang for lang, t in self.GOOD.items() if s.process_record({"text": t}).kept}
        assert kept == {'en', 'zh'}
        assert s.stats.lang_dist == {'eng_Latn': 1, 'cmn_Hani': 1}
        assert s.stats.filtered_lang == 1

    def test_confidence_gate(self, monkeypatch):
        with self._sanitizer(monkeypatch, lang_filter=['fr'], lang_confidence=0.985) as s:
            assert not s.process_record({"text": self.GOOD['fr']}).kept


def test_unknown_backend_rejected():
    with pytest.raises(ConfigurationError, match="lang_backend"):
        SanitizerConfig(lang_backend='cld3').validate()
    with pytest.raises(ConfigurationError):
        make_language_identifier('cld3')


def _glotlid_cached() -> bool:
    try:
        import fasttext  # noqa: F401
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache('cis-lmu/glotlid', 'model.bin'), str)
    except ImportError:
        return False


@pytest.mark.skipif(not _glotlid_cached(), reason="GlotLID model not downloaded")
def test_live_glotlid_cli(tmp_path):
    inp, out = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    texts = list(TestPipelineIntegration.GOOD.values()) + [
        "Привет, это совершенно нормальное предложение на русском языке для проверки."]
    inp.write_text('\n'.join(json.dumps({"text": t}) for t in texts) + '\n')
    env = dict(os.environ, HF_HUB_OFFLINE='1')
    r = subprocess.run([sys.executable, '-m', 'sanitizer_pro', '--input', str(inp), '--output',
                        str(out), '--lang-filter', 'en,zh', '--lang-backend', 'glotlid',
                        '--min-chars', '10', '--min-words', '3', '--no-progress', '--quiet'],
                       capture_output=True, text=True, cwd=str(REPO), env=env)
    assert r.returncode == 0, r.stderr
    kept = [json.loads(line)['text'] for line in out.read_text().splitlines()]
    # clean_text applies NFKC, which also folds fullwidth CJK punctuation
    assert kept == [unicodedata.normalize('NFKC', t) for t in (texts[0], texts[2])]
