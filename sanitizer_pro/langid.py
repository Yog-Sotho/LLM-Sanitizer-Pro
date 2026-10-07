"""Language identification backends for --lang-filter.

  * **glotlid** (default when ``fasttext`` is installed): GlotLID v3, a fastText
    model covering 2,000+ language varieties (Apache-2.0), used by FineWeb-2.
  * **openlid**: OpenLID-v2, a fastText model for ~200 varieties (GPL-3.0;
    downloaded on demand, never bundled).
  * **langdetect**: the original pure-Python detector (55 languages).

fastText models label languages as ISO 639-3 + script (``cmn_Hani``), while
langdetect uses ISO 639-1 (``zh-cn``). Filters accept any of these forms:
``--lang-filter en,zh`` matches ``eng_Latn`` and ``cmn_Hani`` as well as ``en``
and ``zh-cn``, because every detected code is expanded to its aliases,
including the ISO 639-1 code of its macrolanguage (``arz`` -> ``ar``).
"""
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Protocol, Set, Tuple

from sanitizer_pro.utils import ConfigurationError

LANG_BACKENDS = ('auto', 'glotlid', 'openlid', 'langdetect')
FASTTEXT_MODELS: Dict[str, Tuple[str, str]] = {
    'glotlid': ('cis-lmu/glotlid', 'model.bin'),      # GlotLID v3
    'openlid': ('laurievb/OpenLID-v2', 'model.bin'),
}
_MAX_CHARS = 2000  # language is settled long before this; keeps long docs cheap

# ISO 639-3 (individual or macrolanguage) -> ISO 639-1. Individual languages
# map to their macrolanguage's code (cmn/yue -> zh, arz/ary -> ar, pes -> fa).
_ISO3_TO_1: Dict[str, str] = {
    'afr': 'af', 'als': 'sq', 'amh': 'am', 'ara': 'ar', 'arb': 'ar', 'acm': 'ar', 'acq': 'ar',
    'aeb': 'ar', 'ajp': 'ar', 'apc': 'ar', 'apd': 'ar', 'ars': 'ar', 'ary': 'ar', 'arz': 'ar',
    'asm': 'as', 'aze': 'az', 'azb': 'az', 'azj': 'az', 'bak': 'ba', 'bel': 'be', 'ben': 'bn',
    'bod': 'bo', 'bos': 'bs', 'bre': 'br', 'bul': 'bg', 'cat': 'ca', 'ces': 'cs', 'ckb': 'ku',
    'cmn': 'zh', 'cos': 'co', 'cym': 'cy', 'dan': 'da', 'deu': 'de', 'div': 'dv', 'dzo': 'dz',
    'ekk': 'et', 'ell': 'el', 'eng': 'en', 'epo': 'eo', 'est': 'et', 'eus': 'eu', 'fao': 'fo',
    'fas': 'fa', 'fij': 'fj', 'fin': 'fi', 'fra': 'fr', 'fry': 'fy', 'gaz': 'om', 'gla': 'gd',
    'gle': 'ga', 'glg': 'gl', 'grn': 'gn', 'guj': 'gu', 'hat': 'ht', 'hau': 'ha', 'heb': 'he',
    'hin': 'hi', 'hrv': 'hr', 'hun': 'hu', 'hye': 'hy', 'ibo': 'ig', 'ind': 'id', 'isl': 'is',
    'ita': 'it', 'jav': 'jv', 'jpn': 'ja', 'kan': 'kn', 'kat': 'ka', 'kaz': 'kk', 'khk': 'mn',
    'khm': 'km', 'kin': 'rw', 'kir': 'ky', 'kmr': 'ku', 'kor': 'ko', 'kur': 'ku', 'lao': 'lo',
    'lat': 'la', 'lav': 'lv', 'lim': 'li', 'lit': 'lt', 'ltz': 'lb', 'lug': 'lg', 'lvs': 'lv',
    'mal': 'ml', 'mar': 'mr', 'mkd': 'mk', 'mlg': 'mg', 'mlt': 'mt', 'mon': 'mn', 'mri': 'mi',
    'msa': 'ms', 'mya': 'my', 'nep': 'ne', 'nld': 'nl', 'nno': 'nn', 'nob': 'nb', 'nor': 'no',
    'npi': 'ne', 'nya': 'ny', 'oci': 'oc', 'orm': 'om', 'ory': 'or', 'pan': 'pa', 'pbt': 'ps',
    'pes': 'fa', 'plt': 'mg', 'pol': 'pl', 'por': 'pt', 'prs': 'fa', 'pus': 'ps', 'que': 'qu',
    'quy': 'qu', 'roh': 'rm', 'ron': 'ro', 'rus': 'ru', 'san': 'sa', 'sin': 'si', 'slk': 'sk',
    'slv': 'sl', 'smo': 'sm', 'sna': 'sn', 'snd': 'sd', 'som': 'so', 'sot': 'st', 'spa': 'es',
    'sqi': 'sq', 'srp': 'sr', 'sun': 'su', 'swa': 'sw', 'swe': 'sv', 'swh': 'sw', 'tam': 'ta',
    'tat': 'tt', 'tel': 'te', 'tgk': 'tg', 'tgl': 'tl', 'tha': 'th', 'tir': 'ti', 'ton': 'to',
    'tsn': 'tn', 'tuk': 'tk', 'tur': 'tr', 'uig': 'ug', 'ukr': 'uk', 'urd': 'ur', 'uzb': 'uz',
    'uzn': 'uz', 'vie': 'vi', 'wol': 'wo', 'xho': 'xh', 'ydd': 'yi', 'yid': 'yi', 'yor': 'yo',
    'yue': 'zh', 'wuu': 'zh', 'zho': 'zh', 'zsm': 'ms', 'zul': 'zu',
}
# Norwegian Bokmål and Nynorsk are both 'no' (the macrolanguage) too.
_EXTRA_ALIASES: Dict[str, Set[str]] = {'nb': {'no'}, 'nn': {'no'}}


def language_aliases(code: str) -> Set[str]:
    """All forms a detected code can be matched by (lowercase).

    'cmn_Hani' -> {'cmn_hani', 'cmn', 'zh'}; 'zh-cn' -> {'zh-cn', 'zh'}."""
    code = code.strip().lower().replace('__label__', '')
    aliases = {code}
    base = code.split('_')[0].split('-')[0]
    aliases.add(base)
    if base in _ISO3_TO_1:
        aliases.add(_ISO3_TO_1[base])
    for a in list(aliases):
        aliases |= _EXTRA_ALIASES.get(a, set())
    return aliases


def normalize_filter(codes: Iterable[str]) -> Set[str]:
    return {c.strip().lower() for c in codes if c.strip()}


def matches(detected: Optional[str], wanted: Set[str]) -> bool:
    return detected is not None and bool(language_aliases(detected) & wanted)


class LanguageIdentifier(Protocol):
    name: str

    def predict(self, text: str) -> Tuple[Optional[str], float]: ...


class LangDetectIdentifier:
    name = 'langdetect'

    def __init__(self) -> None:
        from sanitizer_pro.quality import LANGDETECT_AVAILABLE
        if not LANGDETECT_AVAILABLE:
            raise ImportError("Language backend 'langdetect' needs: pip install langdetect")

    def predict(self, text: str) -> Tuple[Optional[str], float]:
        from sanitizer_pro.quality import detect_language
        return detect_language(text)


class FastTextIdentifier:
    """GlotLID / OpenLID (or any fastText LID model with __label__ outputs)."""

    def __init__(self, name: str, model_path: Optional[str] = None,
                 _model: Optional[Any] = None) -> None:
        self.name = name
        if _model is not None:
            self._model = _model
            return
        try:
            import fasttext
        except ImportError:
            raise ImportError(
                f"Language backend '{name}' needs fastText: pip install 'llm-sanitizer-pro[lang]'"
            ) from None
        path = model_path or str(resolve_model_file(*FASTTEXT_MODELS[name]))
        self._model = fasttext.load_model(path)

    def predict(self, text: str) -> Tuple[Optional[str], float]:
        line = ' '.join(text.split())[:_MAX_CHARS]  # fastText predicts on one line
        if not line:
            return None, 0.0
        labels, probs = self._model.predict(line, k=1)
        if not labels:
            return None, 0.0
        return str(labels[0]).replace('__label__', ''), min(1.0, float(probs[0]))


def resolve_model_file(repo: str, filename: str, revision: str = 'main') -> Path:
    """Local path of a model file from the Hub: huggingface_hub's cache when
    that library is installed, else this package's own cache and downloader."""
    try:
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(repo, filename, revision=revision))
    except ImportError:
        pass
    from sanitizer_pro.hub import http_download
    cache = Path(os.path.expanduser(os.path.join(
        '~', '.cache', 'llm-sanitizer-pro', 'models', repo.replace('/', '__'), revision)))
    dest = cache / filename
    if not dest.exists():
        cache.mkdir(parents=True, exist_ok=True)
        logging.info(f"Downloading {repo}/{filename} (one-time) …")
        http_download(f"https://huggingface.co/{repo}/resolve/{revision}/{filename}", dest)
    return dest


def fasttext_available() -> bool:
    try:
        import fasttext  # noqa: F401
        return True
    except ImportError:
        return False


def make_language_identifier(backend: str = 'auto',
                             model: Optional[str] = None) -> LanguageIdentifier:
    """Build the requested backend. 'auto' prefers GlotLID (fastText) and
    falls back to langdetect when fastText or the model is unavailable."""
    if backend not in LANG_BACKENDS:
        raise ConfigurationError(f"Unknown language backend '{backend}' {LANG_BACKENDS}.")
    if backend == 'langdetect':
        return LangDetectIdentifier()
    if backend in ('glotlid', 'openlid'):
        return FastTextIdentifier(backend, model)
    if fasttext_available():
        try:
            return FastTextIdentifier('glotlid', model)
        except Exception as exc:
            logging.warning(f"GlotLID unavailable ({exc}); falling back to langdetect.")
    return LangDetectIdentifier()


def language_backend_available(backend: str) -> bool:
    from sanitizer_pro.quality import LANGDETECT_AVAILABLE
    if backend == 'langdetect':
        return LANGDETECT_AVAILABLE
    if backend in ('glotlid', 'openlid'):
        return fasttext_available()
    return fasttext_available() or LANGDETECT_AVAILABLE
