"""NER-backed PII detection for person names, locations, and organizations.

Regex patterns cannot detect names ("please email Sarah Connor about the
invoice") — that requires a named-entity model. This module wraps two optional
backends behind one interface:

  * **spacy** (preferred): fast CPU pipeline, ``en_core_web_sm`` or any
    installed spaCy model with an NER component.
  * **transformers**: HF token-classification pipeline (``dslim/bert-base-NER``
    by default); heavier, needs torch.

Detected spans are replaced with ``[PII_PERSON]`` / ``[PII_LOCATION]`` /
``[PII_ORG]`` tokens, partially masked ("Barack Obama" → "B*** O***"), or
pseudonymized ("Person_0001") via the shared PseudoRegistry, matching the
behavior of the regex-based redactor.
"""
import logging
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

from sanitizer_pro.pii import PseudoRegistry
from sanitizer_pro.utils import ConfigurationError

VALID_ENTITY_KINDS = ('person', 'location', 'org')

# Backend label → internal kind
_LABEL_KIND = {
    'PERSON': 'person', 'PER': 'person',
    'GPE': 'location', 'LOC': 'location', 'LOCATION': 'location', 'FAC': 'location',
    'ORG': 'org', 'ORGANIZATION': 'org',
}
_KIND_TOKEN = {'person': '[PII_PERSON]', 'location': '[PII_LOCATION]', 'org': '[PII_ORG]'}

_SPACY_DEFAULT_MODEL = 'en_core_web_sm'
_HF_DEFAULT_MODEL = 'dslim/bert-base-NER'
_HF_CHUNK_CHARS = 1500  # keep well under BERT's 512-token limit


@dataclass(frozen=True)
class EntitySpan:
    start: int
    end: int
    kind: str


Detector = Callable[[str], List[EntitySpan]]


def _load_spacy_detector(model: Optional[str] = None) -> Detector:
    import spacy
    name = model or _SPACY_DEFAULT_MODEL
    # Only tok2vec + ner are needed; excluding the rest roughly halves latency.
    nlp = spacy.load(name, exclude=['tagger', 'parser', 'attribute_ruler', 'lemmatizer'])
    if 'ner' not in nlp.pipe_names:
        raise ConfigurationError(f"spaCy model '{name}' has no NER component.")

    def detect(text: str) -> List[EntitySpan]:
        spans = []
        for ent in nlp(text).ents:
            kind = _LABEL_KIND.get(ent.label_)
            if kind:
                spans.append(EntitySpan(ent.start_char, ent.end_char, kind))
        return spans

    return detect


def _load_transformers_detector(model: Optional[str] = None) -> Detector:
    from transformers import pipeline
    pipe = pipeline('token-classification', model=model or _HF_DEFAULT_MODEL,
                    aggregation_strategy='simple')

    def detect(text: str) -> List[EntitySpan]:
        spans = []
        offset = 0
        while offset < len(text):
            end = min(offset + _HF_CHUNK_CHARS, len(text))
            if end < len(text):  # cut on whitespace so entities aren't split
                ws = text.rfind(' ', offset + 1, end)
                if ws > offset:
                    end = ws
            chunk = text[offset:end]
            for ent in pipe(chunk):
                kind = _LABEL_KIND.get(ent.get('entity_group', ''))
                if kind:
                    spans.append(EntitySpan(offset + int(ent['start']),
                                            offset + int(ent['end']), kind))
            offset = end
        return spans

    return detect


def _mask_entity(value: str) -> str:
    return ' '.join((w[0] + '***') if len(w) > 1 else '*' for w in value.split())


class NERRedactor:
    """Detect and redact named-entity PII spans in text."""

    def __init__(self, backend: str = 'auto', entities: Sequence[str] = ('person',),
                 model: Optional[str] = None, _detector: Optional[Detector] = None) -> None:
        self.entities = {e.strip().lower() for e in entities if e.strip()}
        invalid = self.entities - set(VALID_ENTITY_KINDS)
        if invalid or not self.entities:
            raise ConfigurationError(
                f"Invalid NER entity kind(s): {sorted(invalid) or '(none)'}. "
                f"Valid: {', '.join(VALID_ENTITY_KINDS)}.")
        if _detector is not None:
            self._detect, self.backend_name = _detector, 'custom'
            return
        self._detect, self.backend_name = self._load_backend(backend, model)
        logging.info(f"NER PII backend ready: {self.backend_name} "
                     f"(entities: {', '.join(sorted(self.entities))})")

    @staticmethod
    def _load_backend(backend: str, model: Optional[str]):
        if backend not in ('auto', 'spacy', 'transformers'):
            raise ConfigurationError(f"Unknown NER backend '{backend}'.")
        errors = []
        if backend in ('auto', 'spacy'):
            try:
                return _load_spacy_detector(model if backend == 'spacy' else None), 'spacy'
            except ConfigurationError:
                raise
            except Exception as exc:
                errors.append(f"spacy: {exc}")
        if backend in ('auto', 'transformers'):
            try:
                return (_load_transformers_detector(model if backend == 'transformers' else None),
                        'transformers')
            except Exception as exc:
                errors.append(f"transformers: {exc}")
        raise ImportError(
            "--pii-ner needs an NER backend. Install one of:\n"
            f"  pip install spacy && pip install {_SPACY_DEFAULT_MODEL} "
            "(or from the HF mirror: pip install "
            f"'en_core_web_sm @ https://huggingface.co/spacy/{_SPACY_DEFAULT_MODEL}"
            f"/resolve/main/{_SPACY_DEFAULT_MODEL}-any-py3-none-any.whl')\n"
            "  pip install transformers torch\n"
            "Errors: " + '; '.join(errors))

    def redact(self, text: str, mask: bool = False,
               pseudo_registry: Optional[PseudoRegistry] = None) -> str:
        if not text:
            return text
        spans = [s for s in self._detect(text) if s.kind in self.entities]
        if not spans:
            return text
        # Sort and drop overlaps (keep the earliest span), then replace
        # right-to-left so earlier offsets stay valid.
        spans.sort(key=lambda s: (s.start, -s.end))
        kept: List[EntitySpan] = []
        last_end = -1
        for s in spans:
            if s.start >= last_end:
                kept.append(s)
                last_end = s.end
        for s in reversed(kept):
            value = text[s.start:s.end]
            if pseudo_registry is not None:
                replacement = pseudo_registry.get_or_create(value, s.kind)
            elif mask:
                replacement = _mask_entity(value)
            else:
                replacement = _KIND_TOKEN[s.kind]
            text = text[:s.start] + replacement + text[s.end:]
        return text
