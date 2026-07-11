"""Tests for NER-backed PII detection."""
import pytest

from sanitizer_pro.ner import EntitySpan, NERRedactor, _mask_entity
from sanitizer_pro.pii import PseudoRegistry
from sanitizer_pro.utils import ConfigurationError

TEXT = "Barack Obama met Angela Merkel in Berlin while working for Microsoft."
SPANS = [
    EntitySpan(0, 12, 'person'),    # Barack Obama
    EntitySpan(17, 30, 'person'),   # Angela Merkel
    EntitySpan(34, 40, 'location'), # Berlin
    EntitySpan(59, 68, 'org'),      # Microsoft
]


def stub_redactor(spans=SPANS, entities=('person', 'location', 'org')):
    return NERRedactor(entities=entities, _detector=lambda text: list(spans))


class TestNERRedactor:
    def test_token_replacement(self):
        out = stub_redactor().redact(TEXT)
        assert out == ("[PII_PERSON] met [PII_PERSON] in [PII_LOCATION] "
                       "while working for [PII_ORG].")

    def test_entity_kind_filtering(self):
        out = stub_redactor(entities=('person',)).redact(TEXT)
        assert 'Berlin' in out and 'Microsoft' in out
        assert 'Barack' not in out and 'Merkel' not in out

    def test_masking(self):
        out = stub_redactor(entities=('person',)).redact(TEXT, mask=True)
        assert 'B*** O***' in out and 'A*** M***' in out

    def test_pseudonymization_stable(self):
        reg = PseudoRegistry()
        r = stub_redactor()
        out1 = r.redact(TEXT, pseudo_registry=reg)
        out2 = r.redact(TEXT, pseudo_registry=reg)
        assert out1 == out2
        assert 'Person_0001' in out1 and 'Person_0002' in out1
        assert 'Place_0001' in out1 and 'Org_0001' in out1

    def test_overlapping_spans_keep_first(self):
        spans = [EntitySpan(0, 12, 'person'), EntitySpan(7, 17, 'org')]
        out = stub_redactor(spans=spans).redact(TEXT)
        assert out.startswith('[PII_PERSON]')
        assert '[PII_ORG]' not in out

    def test_no_spans_returns_text_unchanged(self):
        assert stub_redactor(spans=[]).redact(TEXT) == TEXT

    def test_empty_text(self):
        assert stub_redactor().redact("") == ""

    def test_invalid_entity_kind(self):
        with pytest.raises(ConfigurationError, match="Invalid NER entity"):
            NERRedactor(entities=('person', 'wizard'), _detector=lambda t: [])

    def test_no_entities_rejected(self):
        with pytest.raises(ConfigurationError):
            NERRedactor(entities=(), _detector=lambda t: [])

    def test_unknown_backend(self):
        with pytest.raises(ConfigurationError, match="Unknown NER backend"):
            NERRedactor(backend='wizardry')


def test_mask_entity():
    assert _mask_entity("Barack Obama") == "B*** O***"
    assert _mask_entity("X") == "*"


# --- Live tests: run only when spaCy + en_core_web_sm are installed ----------

def _spacy_available() -> bool:
    try:
        import spacy
        spacy.load('en_core_web_sm')
        return True
    except Exception:
        return False


needs_spacy = pytest.mark.skipif(not _spacy_available(),
                                 reason="spacy en_core_web_sm not installed")


@needs_spacy
def test_spacy_backend_detects_names():
    r = NERRedactor(backend='spacy', entities=('person', 'location', 'org'))
    out = r.redact(TEXT)
    assert '[PII_PERSON]' in out and '[PII_LOCATION]' in out and '[PII_ORG]' in out
    assert 'Obama' not in out and 'Berlin' not in out


@needs_spacy
def test_spacy_backend_clean_text_untouched():
    r = NERRedactor(backend='spacy', entities=('person',))
    text = "The recipe requires flour, sugar, and two eggs mixed thoroughly."
    assert r.redact(text) == text
