"""First-class Python API for LLM Dataset Sanitizer.

The CLI is a thin orchestration layer; services (e.g. a SaaS backend) should
use this module instead of shelling out::

    from sanitizer_pro import Sanitizer, SanitizerConfig

    config = SanitizerConfig(remove_pii=True, deduplicate=True, min_chars=30)
    with Sanitizer(config) as s:
        clean = list(s.process(records))       # iterable of dicts in, dicts out
        report = s.stats.to_dict()             # same stats schema as --stats-file

Per-record introspection is available via :meth:`Sanitizer.process_record`,
which returns a :class:`ProcessResult` explaining exactly why a record was
kept or dropped — the building block for audit UIs.
"""
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from sanitizer_pro.core import TokenTruncator, get_record_hash, sanitize_record
from sanitizer_pro.dedup import make_deduper
from sanitizer_pro.pii import PseudoRegistry
from sanitizer_pro.stats import RunStats
from sanitizer_pro.utils import ConfigurationError, FilterReason, _MAX_DEPTH_DEFAULT


@dataclass
class SanitizerConfig:
    """Typed configuration mirroring the CLI flags (same names, same defaults)."""

    # Cleaning & PII
    clean_html: bool = False
    remove_pii: bool = False
    pii_mask: bool = False
    pii_pseudonymize: bool = False
    pii_ner: bool = False
    pii_ner_backend: str = 'auto'
    pii_ner_entities: Tuple[str, ...] = ('person',)
    pii_ner_model: Optional[str] = None
    redact_secrets: bool = False
    extra_pii_patterns: Optional[List] = None   # (compiled_regex, token, kind) triples

    # Quality gates
    min_chars: int = 50
    max_chars: int = 20000
    min_words: int = 8
    min_ascii_ratio: float = 0.85
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

    # Language
    lang_filter: Optional[List[str]] = None
    lang_confidence: float = 0.0

    # Deduplication
    deduplicate: bool = False
    fuzzy_dedup: bool = False
    fuzzy_threshold: float = 0.8
    semantic_dedup: bool = False
    semantic_threshold: float = 0.9
    semantic_model: str = 'minishlab/potion-base-8M'
    dedup_backend: str = 'memory'
    dedup_db_path: Optional[str] = None
    dedup_fields: Optional[List[str]] = None
    dedup_normalize: bool = False

    # Decontamination
    decontaminate: Optional[List[str]] = None   # benchmark names
    decontam_refs: Optional[List[str]] = None   # local reference files
    decontam_ngram: int = 8
    decontam_min_hits: int = 1
    decontam_cache: Optional[str] = None

    # Chat validation
    validate_chat: bool = False
    chat_lenient: bool = False
    chat_max_tokens: Optional[int] = None
    chat_roles: Tuple[str, ...] = ('system', 'user', 'assistant')

    # Quality scoring
    quality_scorer: str = 'heuristic'
    quality_model: Optional[str] = None
    quality_min_score: Optional[float] = None
    keep_top_percent: Optional[float] = None
    quality_score_field: Optional[str] = None

    # Formatting & truncation
    format_chatml: bool = False
    format_instruct: bool = False
    max_tokens: Optional[int] = None
    tokenizer: str = 'whitespace'

    # Field-level operations: (renames, drops, pii_only, no_clean)
    field_ops: Optional[Tuple[Dict[str, str], set, set, set]] = None

    def validate(self) -> None:
        if self.quality_min_score is not None and not (0 <= self.quality_min_score <= 1):
            raise ConfigurationError("quality_min_score must be in [0, 1].")
        if self.keep_top_percent is not None and not (0 < self.keep_top_percent <= 100):
            raise ConfigurationError("keep_top_percent must be in (0, 100].")
        if not (0 < self.fuzzy_threshold <= 1):
            raise ConfigurationError("fuzzy_threshold must be in (0, 1].")
        if not (0 < self.semantic_threshold <= 1):
            raise ConfigurationError("semantic_threshold must be in (0, 1].")
        if self.semantic_dedup and self.fuzzy_dedup:
            raise ConfigurationError("semantic_dedup and fuzzy_dedup are mutually exclusive.")
        if self.lang_filter:
            from sanitizer_pro.quality import LANGDETECT_AVAILABLE
            if not LANGDETECT_AVAILABLE:
                raise ConfigurationError(
                    "lang_filter requires langdetect (pip install langdetect); "
                    "without it every record would be filtered out.")


@dataclass
class ProcessResult:
    """Outcome of processing one record, with the reason it was dropped (if any)."""
    record: Optional[Dict[str, Any]]
    kept: bool
    reason: Optional[str] = None      # 'quality' | 'language' | 'require_fields' |
                                      # 'code' | 'profanity' | 'malformed' | 'duplicate' |
                                      # 'contaminated' | 'chat:<detail>' | 'low_score'
    score: Optional[float] = None
    lang: Optional[str] = None


class _ArgsView:
    """Duck-typed argparse.Namespace over a SanitizerConfig for core.sanitize_record."""

    def __init__(self, config: SanitizerConfig) -> None:
        self._c = config

    def __getattr__(self, name: str) -> Any:
        return getattr(self._c, name)


class Sanitizer:
    """Reusable sanitization engine. Stateful (dedup, pseudonyms) across calls;
    use one instance per dataset and close() it (or use as a context manager)."""

    def __init__(self, config: Optional[SanitizerConfig] = None) -> None:
        self.config = config or SanitizerConfig()
        self.config.validate()
        c = self.config
        self.stats = RunStats()
        from sanitizer_pro.report import AuditSampleCollector
        self.audit_samples = AuditSampleCollector()
        self._args = _ArgsView(c)

        self._lang_filter = {x.lower() for x in c.lang_filter} if c.lang_filter else None
        self._truncator = TokenTruncator(c.max_tokens, c.tokenizer) if c.max_tokens else None
        self.pseudo_registry = PseudoRegistry() if c.pii_pseudonymize else None

        self._ner = None
        if c.pii_ner and c.remove_pii:
            from sanitizer_pro.ner import NERRedactor
            self._ner = NERRedactor(backend=c.pii_ner_backend, entities=c.pii_ner_entities,
                                    model=c.pii_ner_model)

        self._deduper = None
        if c.deduplicate or c.fuzzy_dedup or c.semantic_dedup:
            self._deduper = make_deduper(
                c.dedup_backend, c.dedup_db_path, fuzzy=c.fuzzy_dedup,
                fuzzy_threshold=c.fuzzy_threshold, semantic=c.semantic_dedup,
                semantic_threshold=c.semantic_threshold, semantic_model=c.semantic_model)

        self._contamination = None
        if c.decontaminate or c.decontam_refs:
            from sanitizer_pro.decontam import build_index
            self._contamination = build_index(
                benchmarks=c.decontaminate, ref_files=c.decontam_refs,
                cache_dir=c.decontam_cache, ngram=c.decontam_ngram,
                min_hits=c.decontam_min_hits)

        self._chat_validator = None
        if c.validate_chat:
            from sanitizer_pro.chat import ChatValidator, make_token_counter
            self._chat_validator = ChatValidator(
                allowed_roles=c.chat_roles, lenient=c.chat_lenient,
                max_tokens=c.chat_max_tokens,
                token_counter=make_token_counter(c.tokenizer) if c.chat_max_tokens else None)

        self._scorer = None
        if (c.quality_min_score is not None or c.keep_top_percent is not None
                or c.quality_score_field):
            from sanitizer_pro.scoring import make_scorer
            self._scorer = make_scorer(c.quality_scorer, model=c.quality_model)

    # -- record level ---------------------------------------------------------

    def process_record(self, record: Any) -> ProcessResult:
        """Run one record through the full pipeline (dedup state is shared
        across calls). Does NOT apply keep_top_percent — that needs the whole
        stream; use process() for it."""
        outcome = self._evaluate(record)
        if isinstance(outcome, ProcessResult):
            return outcome
        sanitized, quality_text, lang, score = outcome
        return ProcessResult(self._emit(sanitized, quality_text, lang, score),
                             True, None, score=score, lang=lang)

    def _evaluate(self, record: Any):
        """Run all filters. Returns a dropped ProcessResult, or the survivor
        tuple (sanitized, quality_text, lang, score) awaiting emission."""
        c = self.config
        self.stats.total += 1

        if not isinstance(record, dict):
            self.stats.malformed += 1
            return ProcessResult(None, False, 'malformed')

        pii_before = sum(self.stats.pii_counts.values())
        sanitized, freason, quality_text, lang = sanitize_record(
            record, self._args, text_fields=c.text_fields,
            extra_pii_patterns=c.extra_pii_patterns, lang_filter=self._lang_filter,
            field_ops=c.field_ops, truncator=self._truncator,
            pseudo_registry=self.pseudo_registry, require_fields=c.require_fields,
            quality_fn=None, ner_redactor=self._ner,
            pii_counters=self.stats.pii_counts)
        if (sanitized is not None and self.audit_samples.wants_pii_diffs
                and sum(self.stats.pii_counts.values()) > pii_before):
            self.audit_samples.add_pii_diff(record, sanitized)

        if sanitized is None:
            if freason == FilterReason.LANGUAGE:
                self.stats.filtered_lang += 1
            elif freason == FilterReason.REQUIRE:
                self.stats.filtered_require += 1
            elif freason == FilterReason.CODE:
                self.stats.filtered_code += 1
            elif freason == FilterReason.PROFANITY:
                self.stats.filtered_profanity += 1
            else:
                self.stats.filtered_quality += 1
            reason = (freason or FilterReason.QUALITY).value
            self.audit_samples.add_dropped(reason, record)
            return ProcessResult(None, False, reason)

        if self._chat_validator is not None:
            chat_reason = self._chat_validator.check(sanitized)
            if chat_reason:
                self.stats.filtered_chat += 1
                self.stats.chat_invalid_reasons[chat_reason] = \
                    self.stats.chat_invalid_reasons.get(chat_reason, 0) + 1
                self.audit_samples.add_dropped('chat', sanitized)
                return ProcessResult(None, False, f'chat:{chat_reason}')

        if self._contamination is not None and self._contamination.is_contaminated(quality_text):
            self.stats.filtered_contaminated += 1
            self.audit_samples.add_dropped('contaminated', sanitized)
            return ProcessResult(None, False, 'contaminated')

        score: Optional[float] = None
        if self._scorer is not None:
            score = self._scorer.score(quality_text)
            if c.quality_min_score is not None and score < c.quality_min_score:
                self.stats.filtered_low_score += 1
                self.audit_samples.add_dropped('low_score', sanitized)
                return ProcessResult(None, False, 'low_score', score=score)

        if self._deduper is not None:
            key = quality_text if (c.fuzzy_dedup or c.semantic_dedup) else \
                get_record_hash(sanitized, c.dedup_fields, c.dedup_normalize)
            if self._deduper.contains(key):
                self.stats.deduplicated += 1
                return ProcessResult(None, False, 'duplicate', score=score, lang=lang)
            self._deduper.add(key)

        return sanitized, quality_text, lang, score

    def _emit(self, sanitized: Dict[str, Any], quality_text: str,
              lang: Optional[str], score: Optional[float]) -> Dict[str, Any]:
        if score is not None:
            self.stats.record_score(score)
            if self.config.quality_score_field:
                sanitized[self.config.quality_score_field] = score
        self.stats.record_kept(quality_text, lang=lang)
        return sanitized

    # -- stream level ---------------------------------------------------------

    def process(self, records: Iterable[Any]) -> Iterator[Dict[str, Any]]:
        """Process an iterable of records, yielding the kept ones.

        With keep_top_percent set, survivors are buffered and the best P% are
        yielded (in input order) once the input is exhausted."""
        c = self.config
        if c.keep_top_percent is None:
            for record in records:
                result = self.process_record(record)
                if result.kept and result.record is not None:
                    yield result.record
            return

        buffer: List[Tuple[Dict[str, Any], str, Optional[str], Optional[float]]] = []
        for record in records:
            outcome = self._evaluate(record)
            if not isinstance(outcome, ProcessResult):
                buffer.append(outcome)
        if not buffer:
            return
        n_keep = max(1, round(len(buffer) * c.keep_top_percent / 100))
        ranked = sorted(range(len(buffer)), key=lambda i: buffer[i][3], reverse=True)
        keep_idx = set(ranked[:n_keep])
        self.stats.filtered_low_score += len(buffer) - n_keep
        for i, (sanitized, quality_text, lang, score) in enumerate(buffer):
            if i in keep_idx:
                yield self._emit(sanitized, quality_text, lang, score)

    # -- lifecycle ------------------------------------------------------------

    def report_html(self, meta: Optional[Dict[str, Any]] = None) -> str:
        """Render the HTML audit report for everything processed so far."""
        from sanitizer_pro.report import generate_report_html
        return generate_report_html(self.stats.to_dict(), self.audit_samples, meta)

    def write_report(self, path: str, meta: Optional[Dict[str, Any]] = None) -> None:
        Path(path).write_text(self.report_html(meta), encoding='utf-8')

    def export_pseudonym_map(self, path: str) -> None:
        if self.pseudo_registry is None:
            raise ConfigurationError("Pseudonymization is not enabled in this config.")
        Path(path).write_text(
            json.dumps(self.pseudo_registry.to_dict(), indent=2, ensure_ascii=False),
            encoding='utf-8')

    def close(self) -> None:
        if self._deduper is not None:
            try:
                self._deduper.close()
            except Exception as exc:
                logging.warning(f"Deduper close failed: {exc}")

    def __enter__(self) -> 'Sanitizer':
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
