"""Multiprocessing worker initialization and execution."""
import argparse
import logging
import sys
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from sanitizer_pro.core import sanitize_record, TokenTruncator, FieldOps
from sanitizer_pro.utils import FilterReason
from sanitizer_pro.config import load_quality_script

_w_args: Optional[argparse.Namespace] = None
_w_extra_pii: Optional[List] = None
_w_lang_filter: Optional[Set[str]] = None
_w_field_ops: Optional[FieldOps] = None
_w_req_fields: Optional[List[str]] = None
_w_text_fields: Optional[List[str]] = None
_w_truncator: Optional[TokenTruncator] = None
_w_quality_fn: Optional[Callable] = None
_w_ner: Optional[Any] = None

def _worker_init(
    args_ns: argparse.Namespace, extra_pii: Optional[List], lang_filter: Optional[Set[str]],
    field_ops: Optional[FieldOps], req_fields: Optional[List[str]], text_fields: Optional[List[str]]
) -> None:
    global _w_args, _w_extra_pii, _w_lang_filter, _w_field_ops, _w_req_fields, _w_text_fields, _w_truncator, _w_quality_fn, _w_ner
    _w_args, _w_extra_pii, _w_lang_filter = args_ns, extra_pii, lang_filter
    _w_field_ops, _w_req_fields, _w_text_fields = field_ops, req_fields, text_fields

    log_level = getattr(args_ns, 'log_level', 'WARNING')
    logging.basicConfig(level=getattr(logging, log_level, logging.WARNING),
                        format='%(asctime)s | %(levelname)s | %(message)s',
                        handlers=[logging.StreamHandler(sys.stderr)], force=True)

    if getattr(args_ns, 'max_tokens', None):
        _w_truncator = TokenTruncator(args_ns.max_tokens, args_ns.tokenizer)
    if getattr(args_ns, 'quality_script', None):
        _w_quality_fn = load_quality_script(args_ns.quality_script)
    if getattr(args_ns, 'pii_ner', False) and getattr(args_ns, 'remove_pii', False):
        # NER models are not picklable, so each worker loads its own copy.
        from sanitizer_pro.ner import NERRedactor
        _w_ner = NERRedactor(
            backend=getattr(args_ns, 'pii_ner_backend', 'auto'),
            entities=str(getattr(args_ns, 'pii_ner_entities', 'person')).split(','),
            model=getattr(args_ns, 'pii_ner_model', None),
        )

def _worker_fn(record: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[FilterReason], str, Optional[str]]:
    return sanitize_record(
        record, _w_args, text_fields=_w_text_fields, extra_pii_patterns=_w_extra_pii,
        lang_filter=_w_lang_filter, field_ops=_w_field_ops, truncator=_w_truncator,
        pseudo_registry=None, require_fields=_w_req_fields, quality_fn=_w_quality_fn,
        ner_redactor=_w_ner
    )
