"""The single configuration object for the sanitization pipeline.

`SanitizerConfig` holds every pipeline setting, its default and its
validation. The CLI derives its flag defaults from it and builds one from the
parsed arguments (`SanitizerConfig.from_namespace`); library users construct it
directly. Field names match the CLI flags (``--min-chars`` -> ``min_chars``).
"""
import argparse
import re
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Set, Tuple

from sanitizer_pro.utils import _MAX_DEPTH_DEFAULT, ConfigurationError

PiiPattern = Tuple['re.Pattern[str]', str, str]          # (compiled regex, token, kind)
FieldOps = Tuple[Dict[str, str], Set[str], Set[str], Set[str]]  # renames, drops, pii_only, no_clean

DEDUP_BACKENDS = ('memory', 'sqlite')
FUZZY_BACKENDS = ('auto', 'rensa', 'datasketch')
SEMANTIC_INDEXES = ('auto', 'usearch', 'lsh')
QUALITY_SCORERS = ('heuristic', 'perplexity', 'fineweb-edu', 'dclm', 'fasttext')
NER_BACKENDS = ('auto', 'spacy', 'transformers', 'gliner')


@dataclass
class SanitizerConfig:
    """Typed pipeline configuration (same names and defaults as the CLI flags)."""

    # Cleaning & PII
    clean_html: bool = False
    remove_pii: bool = False
    pii_mask: bool = False
    pii_pseudonymize: bool = False
    # HMAC key for pseudonyms that are stable across worker processes and runs
    # (kept out of repr so it never lands in logs).
    pseudo_key: Optional[str] = field(default=None, repr=False)
    pii_ner: bool = False
    pii_ner_backend: str = 'auto'
    pii_ner_entities: Tuple[str, ...] = ('person',)
    pii_ner_model: Optional[str] = None
    pii_ner_threshold: float = 0.5        # GLiNER confidence threshold
    redact_secrets: bool = False
    extra_pii_patterns: Optional[List[PiiPattern]] = None

    # Quality gates
    min_chars: int = 50
    max_chars: int = 20000
    min_words: int = 8
    min_ascii_ratio: float = 0.0          # 0 = off (non-English text is kept)
    min_unique_ratio: float = 0.25
    reject_allcaps: bool = False
    allcaps_min_len: int = 50
    allcaps_min_alpha: int = 10
    reject_code: bool = False
    reject_profanity: bool = False
    text_fields: Optional[List[str]] = None
    require_fields: Optional[List[str]] = None
    max_depth: int = _MAX_DEPTH_DEFAULT
    text_fields_depth: int = 20
    quality_script: Optional[str] = None  # path to a module defining quality_check(record)
    quality_rules: Optional[List[str]] = None  # gopher, gopher-repetition, c4, fineweb, all

    # Language
    lang_filter: Optional[List[str]] = None   # ISO 639-1/-3 codes, e.g. ['en', 'zh']
    lang_confidence: float = 0.0
    lang_backend: str = 'auto'                # auto | glotlid | openlid | langdetect
    lang_model: Optional[str] = None          # local fastText model path override

    # Deduplication
    deduplicate: bool = False
    fuzzy_dedup: bool = False
    fuzzy_threshold: float = 0.8
    fuzzy_backend: str = 'auto'
    semantic_dedup: bool = False
    semantic_threshold: float = 0.9
    semantic_model: str = 'minishlab/potion-base-8M'
    semantic_index: str = 'auto'
    dedup_backend: str = 'memory'
    dedup_db_path: Optional[str] = None
    dedup_fields: Optional[List[str]] = None
    dedup_normalize: bool = False

    # Decontamination
    decontaminate: Optional[List[str]] = None   # benchmark names, or ['all']
    decontam_refs: Optional[List[str]] = None   # local reference files
    decontam_ngram: int = 8
    decontam_min_hits: int = 1
    decontam_cache: Optional[str] = None
    encoding: str = 'utf-8'                     # text encoding of reference files

    # Chat validation
    validate_chat: bool = False
    chat_lenient: bool = False
    chat_max_tokens: Optional[int] = None
    chat_roles: Tuple[str, ...] = ('system', 'user', 'assistant')

    # Quality scoring
    quality_scorer: str = 'heuristic'           # see QUALITY_SCORERS
    quality_model: Optional[str] = None
    quality_label: Optional[str] = None         # positive label for 'fasttext'
    quality_min_score: Optional[float] = None
    keep_top_percent: Optional[float] = None
    quality_score_field: Optional[str] = None

    # Sampling: keep a deterministic, content-addressed fraction of survivors
    sample: Optional[float] = None
    seed: Optional[int] = None

    # Formatting & truncation
    format_chatml: bool = False
    format_instruct: bool = False
    max_tokens: Optional[int] = None
    tokenizer: str = 'whitespace'

    # Audit report: include verbatim (unredacted) samples. Off by default
    # because the report would then contain raw PII/secrets.
    report_raw_samples: bool = False

    # Field-level operations: (renames, drops, pii_only, no_clean)
    field_ops: Optional[FieldOps] = None

    # -- validation -------------------------------------------------------------

    def validate(self) -> None:
        """Raise ConfigurationError for any invalid or contradictory setting."""
        def _require(ok: bool, msg: str) -> None:
            if not ok:
                raise ConfigurationError(msg)

        _require(self.quality_min_score is None or 0 <= self.quality_min_score <= 1,
                 "quality_min_score must be in [0, 1].")
        _require(self.keep_top_percent is None or 0 < self.keep_top_percent <= 100,
                 "keep_top_percent must be in (0, 100].")
        _require(self.sample is None or 0 < self.sample <= 1, "sample must be in (0, 1].")
        _require(0 < self.fuzzy_threshold <= 1, "fuzzy_threshold must be in (0, 1].")
        _require(0 < self.semantic_threshold <= 1, "semantic_threshold must be in (0, 1].")
        _require(not (self.semantic_dedup and self.fuzzy_dedup),
                 "semantic_dedup and fuzzy_dedup are mutually exclusive "
                 "(both compare quality text; pick one).")
        _require(self.semantic_index in SEMANTIC_INDEXES,
                 f"semantic_index must be one of {list(SEMANTIC_INDEXES)}.")
        _require(self.fuzzy_backend in FUZZY_BACKENDS,
                 f"fuzzy_backend must be one of {list(FUZZY_BACKENDS)}.")
        _require(self.dedup_backend in DEDUP_BACKENDS,
                 f"dedup_backend must be one of {list(DEDUP_BACKENDS)}.")
        _require(self.quality_scorer in QUALITY_SCORERS,
                 f"quality_scorer must be one of {list(QUALITY_SCORERS)}.")
        _require(self.pii_ner_backend in NER_BACKENDS,
                 f"pii_ner_backend must be one of {list(NER_BACKENDS)}.")
        _require(0 < self.pii_ner_threshold <= 1, "pii_ner_threshold must be in (0, 1].")
        _require(self.chat_max_tokens is None or self.chat_max_tokens >= 1,
                 "chat_max_tokens must be >= 1.")
        _require(not self.validate_chat or any(r.strip() for r in self.chat_roles),
                 "chat_roles must name at least one role.")
        _require(self.max_tokens is None or self.max_tokens >= 1, "max_tokens must be >= 1.")
        if self.quality_rules:
            from sanitizer_pro.rules import resolve_rule_sets
            self.quality_rules = resolve_rule_sets(self.quality_rules)
        _require(self.decontam_ngram >= 2, "decontam_ngram must be >= 2.")
        _require(self.decontam_min_hits >= 1, "decontam_min_hits must be >= 1.")
        from sanitizer_pro.langid import LANG_BACKENDS, language_backend_available
        _require(self.lang_backend in LANG_BACKENDS,
                 f"lang_backend must be one of {list(LANG_BACKENDS)}.")
        if self.lang_filter:
            _require(language_backend_available(self.lang_backend),
                     f"lang_filter needs a language-ID backend ({self.lang_backend}): "
                     "pip install 'llm-sanitizer-pro[lang]' (GlotLID, fastText) or "
                     "pip install langdetect; without one every record would be filtered out.")

    def warnings(self) -> List[str]:
        """Settings that are valid but have no effect."""
        notes = []
        for flag in ('pii_mask', 'pii_pseudonymize', 'pii_ner'):
            if getattr(self, flag) and not self.remove_pii and not (
                    flag != 'pii_ner' and self.redact_secrets):
                notes.append(f"{flag} has no effect without remove_pii.")
        if self.format_chatml and self.format_instruct:
            notes.append("format_chatml and format_instruct both set; format_chatml wins.")
        return notes

    # -- construction from the CLI ----------------------------------------------

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> 'SanitizerConfig':
        """Build a config from parsed CLI arguments (after profile/config merge).

        Comma-separated string flags become lists, and the file-based options
        (--pii-patterns-file, --field-config) are loaded and validated here."""
        from sanitizer_pro.config import (
            build_field_ops, load_custom_pii_patterns, load_field_config,
        )
        ns = vars(args)
        values: Dict[str, Any] = {f.name: ns[f.name] for f in fields(cls) if f.name in ns}
        for name in ('text_fields', 'require_fields', 'dedup_fields', 'decontam_refs',
                     'quality_rules'):
            values[name] = as_list(ns.get(name))
        values['lang_filter'] = [x.lower() for x in as_list(ns.get('lang_filter')) or []] or None
        values['decontaminate'] = as_list(ns.get('decontaminate'))
        values['chat_roles'] = tuple(as_list(ns.get('chat_roles')) or ())
        values['pii_ner_entities'] = tuple(as_list(ns.get('pii_ner_entities')) or ())
        if ns.get('pii_patterns_file'):
            values['extra_pii_patterns'] = load_custom_pii_patterns(ns['pii_patterns_file'])
        if ns.get('field_config'):
            values['field_ops'] = build_field_ops(load_field_config(ns['field_config']))
        return cls(**values)


def as_list(v: Any) -> Optional[List[str]]:
    """'a, b' or ['a', 'b'] -> ['a', 'b']; empty -> None."""
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        return [str(x).strip() for x in v if str(x).strip()] or None
    return [f.strip() for f in str(v).split(',') if f.strip()] or None


DEFAULTS = SanitizerConfig()
