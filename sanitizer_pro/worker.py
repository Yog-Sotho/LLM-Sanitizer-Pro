"""Multiprocessing workers: each runs the per-record stage (RecordTransformer)
built from the pipeline config; the parent runs the stateful stages.

Two task shapes: a single record sent by the parent (any input format), or
a byte-range chunk of a JSONL file that the worker reads and parses itself
(sources.read_chunk), which also moves parsing off the parent process."""
import logging
import sys
from typing import Any, Dict, List, Optional, Tuple

from sanitizer_pro.core import RecordTransformer, Transformed
from sanitizer_pro.io.readers import MalformedRecord
from sanitizer_pro.io.sources import Chunk, read_chunk
from sanitizer_pro.settings import SanitizerConfig

_transformer: Optional[RecordTransformer] = None
_encoding = 'utf-8'

# (transformed or None, PII counts, malformed-input description or None)
ChunkItem = Tuple[Optional[Transformed], Dict[str, int], Optional[str]]


def _worker_init(config: SanitizerConfig, log_level: str = 'WARNING',
                 encoding: str = 'utf-8') -> None:
    global _transformer, _encoding
    logging.basicConfig(level=getattr(logging, log_level, logging.WARNING),
                        format='%(asctime)s | %(levelname)s | %(message)s',
                        handlers=[logging.StreamHandler(sys.stderr)], force=True)
    # NER models and quality scripts are not picklable: each worker loads its own.
    _transformer = RecordTransformer(config)
    _encoding = encoding


def _worker_fn(record: Dict[str, Any]) -> Tuple[Transformed, Dict[str, int]]:
    assert _transformer is not None, "worker not initialized"
    pii_counts: Dict[str, int] = {}
    return _transformer.transform(record, pseudo_registry=None, pii_counters=pii_counts), pii_counts


def _worker_chunk(chunk: Chunk) -> List[ChunkItem]:
    assert _transformer is not None, "worker not initialized"
    out: List[ChunkItem] = []
    for item in read_chunk(chunk, _encoding):
        if isinstance(item, MalformedRecord):
            out.append((None, {}, f"{item.location}: {item.error}"))
        elif not isinstance(item, dict):
            out.append((None, {}, f"{chunk.path}: not an object ({type(item).__name__})"))
        else:
            pii_counts: Dict[str, int] = {}
            out.append((_transformer.transform(item, pseudo_registry=None,
                                               pii_counters=pii_counts), pii_counts, None))
    return out
