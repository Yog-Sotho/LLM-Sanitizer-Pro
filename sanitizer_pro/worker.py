"""Multiprocessing workers: each runs the per-record stage (RecordTransformer)
built from the pipeline config; the parent runs the stateful stages.

Two task shapes: a single record sent by the parent (any input format), or
a byte-range chunk of a JSONL file that the worker reads and parses itself
(sources.read_chunk), which also moves parsing off the parent process.

Each record comes back as a WorkerResult carrying what the parent cannot
recompute without the original record: PII counts, audit-report samples
(redacted here, capped per worker) and pseudonyms created here (keyed, so
consistent across workers; merged by the parent for --pseudo-map-file)."""
import logging
import sys
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from sanitizer_pro.core import RecordTransformer, Transformed
from sanitizer_pro.io.readers import MalformedRecord
from sanitizer_pro.io.sources import Chunk, read_chunk
from sanitizer_pro.pii import PseudoRegistry
from sanitizer_pro.settings import SanitizerConfig

_transformer: Optional[RecordTransformer] = None
_registry: Optional[PseudoRegistry] = None
_samples: Optional[Any] = None          # report.AuditSampleCollector
_encoding = 'utf-8'

Sample = Tuple[str, Any]                # ('dropped', (reason, snippet)) | ('diff', (before, after))


class WorkerResult(NamedTuple):
    transformed: Optional[Transformed]  # None for malformed input
    pii_counts: Dict[str, int]
    problem: Optional[str] = None       # why the input was malformed
    samples: Tuple[Sample, ...] = ()
    pseudonyms: Tuple[Tuple[str, str], ...] = ()


def _worker_init(config: SanitizerConfig, log_level: str = 'WARNING',
                 encoding: str = 'utf-8', want_samples: bool = False) -> None:
    global _transformer, _registry, _samples, _encoding
    logging.basicConfig(level=getattr(logging, log_level, logging.WARNING),
                        format='%(asctime)s | %(levelname)s | %(message)s',
                        handlers=[logging.StreamHandler(sys.stderr)], force=True)
    # NER models and quality scripts are not picklable: each worker loads its own.
    _transformer = RecordTransformer(config)
    _registry = (PseudoRegistry(key=config.pseudo_key, track_new=True)
                 if config.pii_pseudonymize else None)
    _samples = None
    if want_samples:
        from sanitizer_pro.report import AuditSampleCollector
        _samples = AuditSampleCollector(raw=config.report_raw_samples,
                                        redact=_transformer.report_redactor())
    _encoding = encoding


def _process(record: Any, where: str) -> WorkerResult:
    assert _transformer is not None, "worker not initialized"
    if isinstance(record, MalformedRecord):
        return WorkerResult(None, {}, f"{record.location}: {record.error}")
    if not isinstance(record, dict):
        return WorkerResult(None, {}, f"{where}: not an object ({type(record).__name__})")
    pii_counts: Dict[str, int] = {}
    t = _transformer.transform(record, pseudo_registry=_registry, pii_counters=pii_counts)
    samples: List[Sample] = []
    if _samples is not None:
        if t.record is None:
            reason = t.reason.value if t.reason is not None else 'quality'
            bucket = _samples.dropped.get(reason, [])
            before = len(bucket)
            _samples.add_dropped(reason, record)
            bucket = _samples.dropped.get(reason, [])
            if len(bucket) > before:
                samples.append(('dropped', (reason, bucket[-1])))
        elif pii_counts and _samples.wants_pii_diffs:
            before = len(_samples.pii_diffs)
            _samples.add_pii_diff(record, t.record)
            if len(_samples.pii_diffs) > before:
                samples.append(('diff', _samples.pii_diffs[-1]))
    new = tuple(_registry.drain_new()) if _registry is not None else ()
    return WorkerResult(t, pii_counts, None, tuple(samples), new)


def _worker_fn(record: Any) -> WorkerResult:
    return _process(record, 'input')


def _worker_chunk(chunk: Chunk) -> List[WorkerResult]:
    return [_process(item, chunk.path) for item in read_chunk(chunk, _encoding)]
