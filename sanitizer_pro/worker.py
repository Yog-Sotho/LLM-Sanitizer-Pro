"""Multiprocessing workers: each runs the per-record stage (RecordTransformer)
built from the pipeline config; the parent runs the stateful stages."""
import logging
import sys
from typing import Any, Dict, Optional, Tuple

from sanitizer_pro.core import RecordTransformer, Transformed
from sanitizer_pro.settings import SanitizerConfig

_transformer: Optional[RecordTransformer] = None


def _worker_init(config: SanitizerConfig, log_level: str = 'WARNING') -> None:
    global _transformer
    logging.basicConfig(level=getattr(logging, log_level, logging.WARNING),
                        format='%(asctime)s | %(levelname)s | %(message)s',
                        handlers=[logging.StreamHandler(sys.stderr)], force=True)
    # NER models and quality scripts are not picklable: each worker loads its own.
    _transformer = RecordTransformer(config)


def _worker_fn(record: Dict[str, Any]) -> Tuple[Transformed, Dict[str, int]]:
    assert _transformer is not None, "worker not initialized"
    pii_counts: Dict[str, int] = {}
    return _transformer.transform(record, pseudo_registry=None, pii_counters=pii_counts), pii_counts
