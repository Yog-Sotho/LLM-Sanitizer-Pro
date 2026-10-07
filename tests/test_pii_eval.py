"""PII accuracy gate and GLiNER backend tests."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from sanitizer_pro.evaluation import (
    REGEX_KINDS, EvalRecord, combine, evaluate, load_eval_set, regex_detector,
)
from sanitizer_pro.ner import EntitySpan, NERRedactor, _plausible_place
from sanitizer_pro.pii import _PII_PATTERNS, find_pii_spans, redact_pii
from sanitizer_pro.utils import ConfigurationError

REPO = Path(__file__).resolve().parent.parent
RECORDS = load_eval_set()


class TestRegexAccuracyGate:
    """Fails CI if a PII regex change regresses precision or recall."""

    def test_per_kind_f1(self):
        results = evaluate(regex_detector(), RECORDS, set(REGEX_KINDS))
        for kind in REGEX_KINDS:
            assert results[kind]['f1'] >= 0.99, (kind, results[kind])

    def test_no_false_positives_on_hard_negatives(self):
        negatives = [r for r in RECORDS if not r.spans]
        assert len(negatives) >= 150
        hits = [(r.text, find_pii_spans(r.text)) for r in negatives if find_pii_spans(r.text)]
        assert hits == []

    def test_spans_match_what_redaction_removes(self):
        tokens = {kind: token for _, token, kind in _PII_PATTERNS}
        for r in RECORDS:
            text = r.text
            for start, end, kind in sorted(find_pii_spans(text), reverse=True):
                text = text[:start] + tokens[kind] + text[end:]
            assert text == redact_pii(r.text), r.text


class TestEvaluate:
    def test_exact_vs_overlap(self):
        recs = [EvalRecord("Maria Jensen called", [(0, 12, 'person')])]
        det = lambda t: [(0, 5, 'person')]  # noqa: E731
        assert evaluate(det, recs, {'person'})['person']['f1'] == 0.0
        assert evaluate(det, recs, {'person'}, match='overlap')['person']['f1'] == 1.0

    def test_combine_dedupes(self):
        det = combine([lambda t: [(0, 1, 'x')], lambda t: [(0, 1, 'x'), (2, 3, 'y')]])
        assert det("abc") == [(0, 1, 'x'), (2, 3, 'y')]

    def test_cli_json(self):
        r = subprocess.run([sys.executable, '-m', 'sanitizer_pro.evaluation', '--json'],
                           capture_output=True, text=True, cwd=str(REPO))
        assert r.returncode == 0, r.stderr
        out = json.loads(r.stdout)
        assert out['backend'] == 'regex' and out['results']['micro']['f1'] >= 0.99


def fake_gliner(text):
    spans = []
    for word, kind in (("Maria Jensen", 'person'), ("Maria", 'person'),
                       ("12 Elm Street", 'address'), ("127.0.0.1", 'address'),
                       ("X1234567", 'id_number')):
        i = text.find(word)
        if i >= 0:
            spans.append(EntitySpan(i, i + len(word), kind))
    return spans


class TestGlinerBackend:
    TEXT = "Maria Jensen lives at 12 Elm Street; passport X1234567."

    def test_detect_resolves_overlaps(self):
        ner = NERRedactor(entities=['all'], _detector=fake_gliner)
        assert [(s.start, s.end, s.kind) for s in ner.detect(self.TEXT)] == \
            [(0, 12, 'person'), (22, 35, 'address'), (46, 54, 'id_number')]

    def test_new_kinds_redacted(self):
        ner = NERRedactor(entities=['person', 'address', 'id_number'], _detector=fake_gliner)
        assert ner.redact(self.TEXT) == \
            "[PII_PERSON] lives at [PII_ADDRESS]; passport [PII_ID]."

    def test_structured_kinds_need_gliner(self):
        with pytest.raises(ConfigurationError, match="gliner"):
            NERRedactor(backend='spacy', entities=['person', 'id_number'])

    @pytest.mark.parametrize("value,ok", [
        ("12 Elm Street", True), ("Toronto", True), ("127.0.0.1", False), ("0.0.0.0", False),
        ("2001:db8::1", False), ("NL35 ABNA 2591 8876 54", False), ("12345", False),
    ])
    def test_plausible_place(self, value, ok):
        assert _plausible_place(value) is ok


def _gliner_cached():
    try:
        import gliner2  # noqa: F401
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache('fastino/gliner2-privacy-filter-PII-multi',
                                                 'model.safetensors'), str)
    except Exception:
        return False


@pytest.mark.skipif(not _gliner_cached(), reason="GLiNER2-PII model not downloaded")
def test_live_gliner_redaction():
    ner = NERRedactor(backend='gliner', entities=['person', 'address', 'id_number'])
    out = ner.redact("Please contact Maria Jensen at 12 Elm Street. Her passport is X1234567. "
                     "The service runs on 127.0.0.1.")
    assert "Maria Jensen" not in out and "X1234567" not in out and "12 Elm Street" not in out
    assert "127.0.0.1" in out  # loopback is not a postal address
