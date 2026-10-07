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
from typing import Any, Dict, Iterable, Iterator, List, NamedTuple, Optional, Union

from sanitizer_pro.core import RecordTransformer, Transformed, get_record_hash
from sanitizer_pro.decontam import full_text_for_decontam
from sanitizer_pro.dedup import make_deduper
from sanitizer_pro.pii import PseudoRegistry
from sanitizer_pro.sampling import keep_in_sample
from sanitizer_pro.settings import SanitizerConfig
from sanitizer_pro.stats import RunStats
from sanitizer_pro.utils import ConfigurationError


@dataclass
class ProcessResult:
    """Outcome of processing one record, with the reason it was dropped (if any)."""
    record: Optional[Dict[str, Any]]
    kept: bool
    reason: Optional[str] = None      # 'quality' | 'language' | 'require_fields' |
                                      # 'code' | 'profanity' | 'malformed' | 'duplicate' |
                                      # 'contaminated' | 'chat:<detail>' | 'low_score' |
                                      # 'sampled_out'
    score: Optional[float] = None
    lang: Optional[str] = None


class Survivor(NamedTuple):
    """A record that passed every filter and awaits emission."""
    record: Dict[str, Any]
    quality_text: str
    lang: Optional[str]
    score: Optional[float]


Outcome = Union[ProcessResult, Survivor]


class Sanitizer:
    """The sanitization engine, shared by the CLI and library callers.

    Pipeline per record: RecordTransformer (clean, redact, gates) -> chat
    validation -> decontamination -> quality score -> sampling -> dedup ->
    (top-P% buffer) -> emit. Stateful across calls (dedup, pseudonyms, stats);
    use one instance per dataset and close() it (or use it as a context manager).

    Streaming entry points: `feed(record)` returns the records to write now,
    `finish()` drains the --keep-top-percent buffer, and `process(records)`
    combines both. `feed_transformed()` accepts output of the per-record stage
    computed elsewhere (worker processes)."""

    def __init__(self, config: Optional[SanitizerConfig] = None, *,
                 stats: Optional[RunStats] = None,
                 pseudo_registry: Optional[PseudoRegistry] = None) -> None:
        self.config = config or SanitizerConfig()
        self.config.validate()
        c = self.config
        self.stats = stats if stats is not None else RunStats()
        if pseudo_registry is None and c.pii_pseudonymize:
            pseudo_registry = PseudoRegistry()
        self.pseudo_registry = pseudo_registry
        self.transformer = RecordTransformer(c)

        from sanitizer_pro.report import AuditSampleCollector
        self.audit_samples = AuditSampleCollector(
            raw=c.report_raw_samples, redact=self.transformer.report_redactor())

        self.deduper: Optional[Any] = None
        if c.deduplicate or c.fuzzy_dedup or c.semantic_dedup:
            self.deduper = make_deduper(
                c.dedup_backend, c.dedup_db_path, fuzzy=c.fuzzy_dedup,
                fuzzy_threshold=c.fuzzy_threshold, semantic=c.semantic_dedup,
                semantic_threshold=c.semantic_threshold, semantic_model=c.semantic_model)

        self._contamination: Optional[Any] = None
        if c.decontaminate or c.decontam_refs:
            from sanitizer_pro.decontam import build_index, resolve_benchmark_names
            names = c.decontaminate
            if names:  # validates names and expands 'all'
                names = resolve_benchmark_names(
                    names if isinstance(names, str) else ','.join(names))
            self._contamination = build_index(
                benchmarks=names, ref_files=c.decontam_refs,
                cache_dir=c.decontam_cache, ngram=c.decontam_ngram,
                min_hits=c.decontam_min_hits, encoding=c.encoding)

        self._chat_validator: Optional[Any] = None
        if c.validate_chat:
            from sanitizer_pro.chat import ChatValidator, make_counters
            text_counter, conversation_counter = (
                make_counters(c.tokenizer) if c.chat_max_tokens else (None, None))
            self._chat_validator = ChatValidator(
                allowed_roles=c.chat_roles, lenient=c.chat_lenient,
                max_tokens=c.chat_max_tokens, token_counter=text_counter,
                conversation_counter=conversation_counter)

        if self.transformer.scorer is not None:
            logging.info(f"Quality scorer ready: {self.transformer.scorer.backend_name}")

        self._topk: Optional[List[Survivor]] = None
        if c.keep_top_percent is not None:
            self._topk = []
            logging.info(f"keep_top_percent {c.keep_top_percent}: surviving records are "
                         "buffered in memory until the end of the input.")

    # -- record level ---------------------------------------------------------

    def process_record(self, record: Any) -> ProcessResult:
        """Run one record through the full pipeline (dedup state is shared
        across calls). Does NOT apply keep_top_percent, which needs the whole
        stream; use feed()/finish() or process() for it."""
        outcome = self._evaluate(record)
        if isinstance(outcome, ProcessResult):
            return outcome
        return ProcessResult(self._emit(outcome), True, None, score=outcome.score,
                             lang=outcome.lang)

    def _evaluate(self, record: Any) -> Outcome:
        self.stats.total += 1
        if not isinstance(record, dict):
            self.stats.malformed += 1
            return ProcessResult(None, False, 'malformed')
        pii_before = sum(self.stats.pii_counts.values())
        transformed = self.transformer.transform(record, self.pseudo_registry,
                                                 self.stats.pii_counts)
        if (transformed.record is not None and self.audit_samples.wants_pii_diffs
                and sum(self.stats.pii_counts.values()) > pii_before):
            self.audit_samples.add_pii_diff(record, transformed.record)
        return self._admit(record, transformed)

    def _admit(self, original: Optional[Dict[str, Any]], t: Transformed) -> Outcome:
        """Every stage after the per-record transform. `original` (the raw
        input, when available) only feeds redacted audit samples."""
        c, stats = self.config, self.stats
        if t.record is None:
            reason = (t.reason.value if t.reason is not None else 'quality')
            counter = {'language': 'filtered_lang', 'require_fields': 'filtered_require',
                       'code': 'filtered_code', 'profanity': 'filtered_profanity',
                       'rules': 'filtered_rules'}
            attr = counter.get(reason, 'filtered_quality')
            setattr(stats, attr, getattr(stats, attr) + 1)
            if t.detail and reason == 'rules':
                stats.rule_failures[t.detail] = stats.rule_failures.get(t.detail, 0) + 1
            if original is not None:
                self.audit_samples.add_dropped(reason, original)
            return ProcessResult(None, False, f'rules:{t.detail}' if t.detail else reason)
        sanitized = t.record

        if self._chat_validator is not None:
            chat_reason = self._chat_validator.check(sanitized)
            if chat_reason:
                stats.filtered_chat += 1
                stats.chat_invalid_reasons[chat_reason] = \
                    stats.chat_invalid_reasons.get(chat_reason, 0) + 1
                logging.debug(f"Chat validation rejected record: {chat_reason}")
                self.audit_samples.add_dropped('chat', sanitized, redacted=True)
                return ProcessResult(None, False, f'chat:{chat_reason}')

        bench = self._contamination.match(full_text_for_decontam(sanitized)) \
            if self._contamination is not None else None
        if bench is not None:
            stats.filtered_contaminated += 1
            stats.contaminated_by[bench] = stats.contaminated_by.get(bench, 0) + 1
            self.audit_samples.add_dropped('contaminated', sanitized, redacted=True)
            return ProcessResult(None, False, 'contaminated')

        score = t.score
        if score is not None:
            if c.quality_min_score is not None and score < c.quality_min_score:
                stats.filtered_low_score += 1
                self.audit_samples.add_dropped('low_score', sanitized, redacted=True)
                return ProcessResult(None, False, 'low_score', score=score)

        if c.sample is not None and not keep_in_sample(sanitized, c.sample, c.seed):
            stats.sampled_out += 1
            return ProcessResult(None, False, 'sampled_out', score=score, lang=t.lang)

        if self.deduper is not None:
            key = t.quality_text if (c.fuzzy_dedup or c.semantic_dedup) else \
                get_record_hash(sanitized, c.dedup_fields, c.dedup_normalize)
            if self.deduper.contains(key):
                stats.deduplicated += 1
                return ProcessResult(None, False, 'duplicate', score=score, lang=t.lang)
            self.deduper.add(key)

        return Survivor(sanitized, t.quality_text, t.lang, score)

    def _emit(self, s: Survivor) -> Dict[str, Any]:
        if s.score is not None:
            self.stats.record_score(s.score)
            if self.config.quality_score_field:
                s.record[self.config.quality_score_field] = s.score
        self.stats.record_kept(s.quality_text, lang=s.lang)
        return s.record

    def _route(self, outcome: Outcome) -> List[Dict[str, Any]]:
        if isinstance(outcome, ProcessResult):
            return []
        if self._topk is not None:
            self._topk.append(outcome)
            return []
        return [self._emit(outcome)]

    # -- stream level ---------------------------------------------------------

    def feed(self, record: Any) -> List[Dict[str, Any]]:
        """Process one input record; return the records to write now (none
        while keep_top_percent buffers survivors)."""
        return self._route(self._evaluate(record))

    def feed_transformed(self, transformed: Transformed,
                         pii_counts: Optional[Dict[str, int]] = None) -> List[Dict[str, Any]]:
        """Like feed(), for a record whose per-record stage already ran
        elsewhere (a worker process)."""
        self.stats.total += 1
        self.stats.merge_pii_counts(pii_counts)
        return self._route(self._admit(None, transformed))

    def finish(self) -> List[Dict[str, Any]]:
        """End of input: release the best keep_top_percent of the buffered
        survivors, in input order."""
        buffer, self._topk = self._topk, ([] if self._topk is not None else None)
        if not buffer:
            return []
        assert self.config.keep_top_percent is not None
        n_keep = max(1, round(len(buffer) * self.config.keep_top_percent / 100))
        ranked = sorted(range(len(buffer)), key=lambda i: buffer[i].score or 0.0, reverse=True)
        keep_idx = set(ranked[:n_keep])
        self.stats.filtered_low_score += len(buffer) - n_keep
        return [self._emit(s) for i, s in enumerate(buffer) if i in keep_idx]

    def process(self, records: Iterable[Any]) -> Iterator[Dict[str, Any]]:
        """Process an iterable of records, yielding the kept ones. With
        keep_top_percent set, the best P% are yielded (in input order) once
        the input is exhausted."""
        for record in records:
            yield from self.feed(record)
        yield from self.finish()

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
        if self.deduper is not None:
            try:
                self.deduper.close()
            except Exception as exc:
                logging.warning(f"Deduper close failed: {exc}")

    def __enter__(self) -> 'Sanitizer':
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


__all__ = ['ProcessResult', 'Sanitizer', 'SanitizerConfig', 'Survivor']
