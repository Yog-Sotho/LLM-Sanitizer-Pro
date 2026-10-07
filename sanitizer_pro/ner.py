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
import ipaddress
import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from sanitizer_pro.pii import _IBAN_RE, PseudoRegistry
from sanitizer_pro.utils import ConfigurationError

# Classic NER (spaCy, transformers) only knows names, places and organizations;
# GLiNER PII models also find structured PII that regexes cannot pin down.
CLASSIC_KINDS = ('person', 'location', 'org')
VALID_ENTITY_KINDS = CLASSIC_KINDS + (
    'address', 'date_of_birth', 'id_number', 'financial', 'username', 'credential')

# GLiNER2-PII label set (42 types) grouped into the redactor's kinds.
GLINER_LABELS: Dict[str, Tuple[str, ...]] = {
    'person': ('person', 'full_name', 'first_name', 'middle_name', 'last_name'),
    'location': ('city', 'state_or_region', 'country'),
    'org': ('organization',),
    'address': ('address', 'street_address', 'postal_code'),
    'date_of_birth': ('date_of_birth',),
    'id_number': ('government_id', 'national_id_number', 'passport_number',
                  'drivers_license_number', 'license_number', 'tax_id', 'tax_number'),
    'financial': ('bank_account', 'account_number', 'routing_number', 'iban',
                  'payment_card', 'card_number', 'card_expiry', 'card_cvv'),
    'username': ('username', 'account_id', 'sensitive_account_id'),
    'credential': ('password', 'secret', 'api_key', 'access_token', 'recovery_code'),
}
_GLINER_DEFAULT_MODEL = 'fastino/gliner2-privacy-filter-PII-multi'

# Backend label → internal kind
_LABEL_KIND = {
    'PERSON': 'person', 'PER': 'person',
    'GPE': 'location', 'LOC': 'location', 'LOCATION': 'location', 'FAC': 'location',
    'ORG': 'org', 'ORGANIZATION': 'org',
}
_KIND_TOKEN = {'person': '[PII_PERSON]', 'location': '[PII_LOCATION]', 'org': '[PII_ORG]',
               'address': '[PII_ADDRESS]', 'date_of_birth': '[PII_DOB]',
               'id_number': '[PII_ID]', 'financial': '[PII_FINANCIAL]',
               'username': '[PII_USERNAME]', 'credential': '[PII_CREDENTIAL]'}

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


def _chunks(text: str, size: int = _HF_CHUNK_CHARS) -> List[Tuple[int, str]]:
    """(offset, chunk) pieces of at most `size` chars, cut on whitespace so
    entities are not split (unless a chunk has no whitespace at all)."""
    out = []
    offset = 0
    while offset < len(text):
        end = min(offset + size, len(text))
        if end < len(text):
            ws = max(text.rfind(' ', offset + 1, end), text.rfind('\n', offset + 1, end))
            if ws > offset:
                end = ws
        out.append((offset, text[offset:end]))
        offset = end
    return out


def _load_transformers_detector(model: Optional[str] = None) -> Detector:
    from transformers import pipeline
    pipe = pipeline('token-classification', model=model or _HF_DEFAULT_MODEL,
                    aggregation_strategy='simple')

    def detect(text: str) -> List[EntitySpan]:
        spans = []
        for offset, chunk in _chunks(text):
            for ent in pipe(chunk):
                kind = _LABEL_KIND.get(ent.get('entity_group', ''))
                if kind:
                    spans.append(EntitySpan(offset + int(ent['start']),
                                            offset + int(ent['end']), kind))
        return spans

    return detect


def _plausible_place(value: str) -> bool:
    """GLiNER's 'address' label also fires on network addresses and account
    numbers ('127.0.0.1', 'NL35 ABNA ...'). Postal addresses and places always
    contain letters and are never IPs or IBANs; those kinds are left to the
    regex layer, which validates them properly."""
    v = value.strip()
    if not any(c.isalpha() for c in v):
        return False
    try:
        ipaddress.ip_address(v)
        return False
    except ValueError:
        pass
    return _IBAN_RE.fullmatch(v) is None


def _load_gliner_detector(model: Optional[str], kinds: Sequence[str],
                          threshold: float) -> Detector:
    """GLiNER2 PII model (default fastino/gliner2-privacy-filter-PII-multi,
    Apache-2.0, 42 labels, 7 languages). Uses exact character spans."""
    try:
        from gliner2 import GLiNER2
    except ImportError:
        raise ImportError("--pii-ner-backend gliner needs: pip install "
                          "'llm-sanitizer-pro[gliner]'") from None
    net = GLiNER2.from_pretrained(model or _GLINER_DEFAULT_MODEL)
    label_kind = {label: kind for kind in kinds for label in GLINER_LABELS[kind]}
    labels = list(label_kind)

    def detect(text: str) -> List[EntitySpan]:
        spans = []
        for offset, chunk in _chunks(text):
            if not chunk.strip():
                continue
            result = net.extract_entities(chunk, labels, threshold=threshold,
                                          include_confidence=True, include_spans=True)
            for label, found in (result.get('entities') or {}).items():
                kind = label_kind.get(label)
                if kind is None:
                    continue
                for ent in found:
                    if not (isinstance(ent, dict) and 'start' in ent and 'end' in ent):
                        continue
                    start, end = offset + int(ent['start']), offset + int(ent['end'])
                    if kind in ('address', 'location') and not _plausible_place(text[start:end]):
                        continue
                    spans.append(EntitySpan(start, end, kind))
        return spans

    return detect


def _mask_entity(value: str) -> str:
    return ' '.join((w[0] + '***') if len(w) > 1 else '*' for w in value.split())


class NERRedactor:
    """Detect and redact named-entity PII spans in text."""

    def __init__(self, backend: str = 'auto', entities: Sequence[str] = ('person',),
                 model: Optional[str] = None, _detector: Optional[Detector] = None,
                 threshold: float = 0.5) -> None:
        requested = {e.strip().lower() for e in entities if e.strip()}
        supported = VALID_ENTITY_KINDS if backend in ('gliner',) or _detector else CLASSIC_KINDS
        if 'all' in requested:
            requested = set(supported)
        self.entities = requested
        invalid = self.entities - set(supported)
        if invalid or not self.entities:
            hint = (" (address, date_of_birth, id_number, financial, username and "
                    "credential need --pii-ner-backend gliner)"
                    if invalid & set(VALID_ENTITY_KINDS) else '')
            raise ConfigurationError(
                f"Invalid NER entity kind(s) for backend '{backend}': "
                f"{sorted(invalid) or '(none)'}. Valid: {', '.join(supported)}{hint}.")
        if _detector is not None:
            self._detect, self.backend_name = _detector, 'custom'
            return
        if backend == 'gliner':
            self._detect = _load_gliner_detector(model, sorted(self.entities), threshold)
            self.backend_name = 'gliner'
        else:
            self._detect, self.backend_name = self._load_backend(backend, model)
        logging.info(f"NER PII backend ready: {self.backend_name} "
                     f"(entities: {', '.join(sorted(self.entities))})")

    @staticmethod
    def _load_backend(backend: str, model: Optional[str]) -> Tuple[Detector, str]:
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

    def detect(self, text: str) -> List[EntitySpan]:
        """Non-overlapping entity spans of the requested kinds, in text order.

        Overlaps (e.g. GLiNER's full_name 'Maria Jensen' plus first_name
        'Maria') keep the earliest-starting, then longest span — exactly what
        redact() replaces."""
        if not text:
            return []
        spans = sorted((s for s in self._detect(text) if s.kind in self.entities),
                       key=lambda s: (s.start, -s.end))
        kept: List[EntitySpan] = []
        last_end = -1
        for s in spans:
            if s.start >= last_end:
                kept.append(s)
                last_end = s.end
        return kept

    def redact(self, text: str, mask: bool = False,
               pseudo_registry: Optional[PseudoRegistry] = None,
               counters: Optional[Dict[str, int]] = None) -> str:
        if not text:
            return text
        kept = self.detect(text)
        if not kept:
            return text
        # Replace right-to-left so earlier offsets stay valid.
        if counters is not None:
            for s in kept:
                counters[s.kind] = counters.get(s.kind, 0) + 1
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
