"""Core sanitization logic, recursive traversal, and LLM formatting."""
import argparse
import json
import hashlib
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from sanitizer_pro.utils import FilterReason, _MAX_DEPTH_DEFAULT
from sanitizer_pro.pii import clean_text, redact_pii, PseudoRegistry
from sanitizer_pro.quality import extract_text_for_quality, _check_quality_reason, detect_language, is_code_heuristic, contains_profanity

FieldOps = Tuple[Dict[str, str], Set[str], Set[str], Set[str]]

class TokenTruncator:
    def __init__(self, max_tokens: int, tokenizer_name: str = 'whitespace') -> None:
        self.max_tokens = max_tokens
        self._hf = None
        if tokenizer_name != 'whitespace':
            try:
                from transformers import AutoTokenizer
                self._hf = AutoTokenizer.from_pretrained(tokenizer_name)
            except Exception as exc:
                import logging
                logging.warning(
                    f"Could not load tokenizer '{tokenizer_name}' ({exc}); "
                    "falling back to whitespace tokenization.")

    def truncate(self, text: str) -> str:
        if not text or self.max_tokens <= 0: return text
        if self._hf:
            ids = self._hf.encode(text, add_special_tokens=False)
            return self._hf.decode(ids[:self.max_tokens], skip_special_tokens=True) if len(ids) > self.max_tokens else text
        words = text.split()
        return ' '.join(words[:self.max_tokens]) if len(words) > self.max_tokens else text

def _sanitize_value(
    v: Any, *, remove_html: bool, remove_pii: bool, pii_mask: bool,
    extra_pii: Optional[List], pseudo_registry: Optional[PseudoRegistry],
    field_pii_only: bool, field_no_clean: bool, max_depth: int,
    truncator: Optional[TokenTruncator], ner_redactor: Optional[Any] = None,
    pii_counters: Optional[Dict[str, int]] = None, _depth: int = 0
) -> Any:
    if _depth > max_depth: return v
    kw = dict(remove_html=remove_html, remove_pii=remove_pii, pii_mask=pii_mask,
              extra_pii=extra_pii, pseudo_registry=pseudo_registry,
              field_pii_only=field_pii_only, field_no_clean=field_no_clean,
              max_depth=max_depth, truncator=truncator, ner_redactor=ner_redactor,
              pii_counters=pii_counters, _depth=_depth + 1)

    if isinstance(v, str):
        if field_no_clean: return v
        if field_pii_only:
            if not remove_pii: return v
            # NER runs first: the model should see natural text, not [PII_*] tokens.
            if ner_redactor: v = ner_redactor.redact(v, mask=pii_mask, pseudo_registry=pseudo_registry, counters=pii_counters)
            return redact_pii(v, mask=pii_mask, extra_patterns=extra_pii, pseudo_registry=pseudo_registry, counters=pii_counters)
        cleaned = clean_text(v, remove_html)
        if remove_pii:
            if ner_redactor:
                cleaned = ner_redactor.redact(cleaned, mask=pii_mask, pseudo_registry=pseudo_registry, counters=pii_counters)
            cleaned = redact_pii(cleaned, mask=pii_mask, extra_patterns=extra_pii, pseudo_registry=pseudo_registry, counters=pii_counters)
        if truncator: cleaned = truncator.truncate(cleaned)
        return cleaned
    if isinstance(v, dict): return {k: _sanitize_value(val, **kw) for k, val in v.items()}
    if isinstance(v, list): return [_sanitize_value(item, **kw) for item in v]
    return v

def sanitize_record(
    record: Any, args: argparse.Namespace, text_fields: Optional[List[str]] = None,
    extra_pii_patterns: Optional[List] = None, lang_filter: Optional[Set[str]] = None,
    field_ops: Optional[FieldOps] = None, truncator: Optional[TokenTruncator] = None,
    pseudo_registry: Optional[PseudoRegistry] = None, require_fields: Optional[List[str]] = None,
    quality_fn: Optional[Callable] = None, ner_redactor: Optional[Any] = None,
    pii_counters: Optional[Dict[str, int]] = None
) -> Tuple[Optional[Dict[str, Any]], Optional[FilterReason], str, Optional[str]]:
    if not isinstance(record, dict):
        return None, FilterReason.QUALITY, '', None
        
    renames, drops, pii_only, no_clean = field_ops if field_ops else ({}, set(), set(), set())
    if drops: record = {k: v for k, v in record.items() if k not in drops}
    if renames: record = {renames.get(k, k): v for k, v in record.items()}

    sanitized: Dict[str, Any] = {
        fname: _sanitize_value(
            val, remove_html=args.clean_html, remove_pii=args.remove_pii, pii_mask=args.pii_mask,
            extra_pii=extra_pii_patterns, pseudo_registry=pseudo_registry,
            field_pii_only=(fname in pii_only), field_no_clean=(fname in no_clean),
            max_depth=getattr(args, 'max_depth', _MAX_DEPTH_DEFAULT), truncator=truncator,
            ner_redactor=ner_redactor, pii_counters=pii_counters
        ) for fname, val in record.items()
    }

    if require_fields:
        for rf in require_fields:
            v = sanitized.get(rf)
            is_empty = v is None or (isinstance(v, str) and not v.strip()) or (not isinstance(v, (bool, int, float)) and not v)
            if is_empty: return None, FilterReason.REQUIRE, '', None

    quality_text = extract_text_for_quality(sanitized, text_fields=text_fields, max_depth=getattr(args, 'text_fields_depth', 20))
    
    if getattr(args, 'reject_code', False) and is_code_heuristic(quality_text):
        return None, FilterReason.CODE, '', None
    if getattr(args, 'reject_profanity', False) and contains_profanity(quality_text):
        return None, FilterReason.PROFANITY, '', None

    reason_str = _check_quality_reason(quality_text, args)
    if reason_str: return None, FilterReason.QUALITY, '', None

    if quality_fn and not quality_fn(sanitized):
        return None, FilterReason.QUALITY, '', None

    detected_lang: Optional[str] = None
    if lang_filter:
        detected_lang, conf = detect_language(quality_text, min_confidence=getattr(args, 'lang_confidence', 0.0))
        if detected_lang not in lang_filter: return None, FilterReason.LANGUAGE, '', None

    # LLM Formatting
    if getattr(args, 'format_chatml', False):
        sanitized = format_chatml(sanitized)
    elif getattr(args, 'format_instruct', False):
        sanitized = format_instruct(sanitized)

    return sanitized, None, quality_text, detected_lang

def format_chatml(record: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(record.get("messages"), list):
        return {"messages": record["messages"]}  # already conversational
    messages = []
    if record.get("system"): messages.append({"role": "system", "content": str(record["system"])})
    user_content = str(record.get("instruction") or record.get("prompt") or record.get("question") or "")
    if record.get("input"): user_content = f"{user_content}\n{record['input']}".strip()
    if user_content: messages.append({"role": "user", "content": user_content.strip()})
    assistant = record.get("output") or record.get("response") or record.get("completion") or record.get("answer")
    if assistant is not None: messages.append({"role": "assistant", "content": str(assistant)})
    if not messages and record.get("text"):
        messages.append({"role": "user", "content": str(record["text"])})
    return {"messages": messages}

def format_instruct(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "instruction": str(record.get("instruction") or record.get("prompt") or record.get("question") or ""),
        "input": str(record.get("input") or ""),
        "output": str(record.get("output") or record.get("response") or record.get("completion") or record.get("answer") or "")
    }

def get_record_hash(record: Dict[str, Any], dedup_fields: Optional[List[str]] = None, normalize: bool = False) -> str:
    target = {k: record.get(k) for k in dedup_fields} if dedup_fields else record
    if normalize:
        def _norm(v: Any) -> Any:
            if isinstance(v, str): return re.sub(r'\s+', ' ', v.lower().strip())
            if isinstance(v, dict): return {k2: _norm(v2) for k2, v2 in v.items()}
            if isinstance(v, list): return [_norm(i) for i in v]
            return v
        target = _norm(target)
    serialized = json.dumps(target, sort_keys=True, ensure_ascii=False,
                            separators=(',', ':'), default=str)
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()
