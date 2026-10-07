"""Core sanitization logic, recursive traversal, and LLM formatting."""
import json
import hashlib
import re
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set

from sanitizer_pro.settings import FieldOps, SanitizerConfig
from sanitizer_pro.utils import FilterReason, _MAX_DEPTH_DEFAULT
from sanitizer_pro.pii import clean_text, redact_pii, PseudoRegistry
from sanitizer_pro.secrets import redact_secrets as _redact_secrets_fn
from sanitizer_pro.quality import extract_text_for_quality, _check_quality_reason, detect_language, is_code_heuristic, contains_profanity

__all__ = ['FieldOps', 'RecordTransformer', 'TokenTruncator', 'Transformed', 'format_chatml',
           'format_instruct', 'get_record_hash', 'make_report_redactor', 'sanitize_record']
_TOKEN_RE = re.compile(r'\S+')

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
        # Cut after the Nth whitespace-delimited token, keeping the original
        # spacing and newlines (code/markdown structure) of what remains.
        for i, m in enumerate(_TOKEN_RE.finditer(text), 1):
            if i == self.max_tokens:
                return text[:m.end()] if text[m.end():].strip() else text
        return text

def _sanitize_value(
    v: Any, *, remove_html: bool, remove_pii: bool, pii_mask: bool,
    extra_pii: Optional[List], pseudo_registry: Optional[PseudoRegistry],
    field_pii_only: bool, field_no_clean: bool, max_depth: int,
    truncator: Optional[TokenTruncator], ner_redactor: Optional[Any] = None,
    pii_counters: Optional[Dict[str, int]] = None, redact_secrets: bool = False,
    _depth: int = 0
) -> Any:
    if _depth > max_depth: return v
    kw = dict(remove_html=remove_html, remove_pii=remove_pii, pii_mask=pii_mask,
              extra_pii=extra_pii, pseudo_registry=pseudo_registry,
              field_pii_only=field_pii_only, field_no_clean=field_no_clean,
              max_depth=max_depth, truncator=truncator, ner_redactor=ner_redactor,
              pii_counters=pii_counters, redact_secrets=redact_secrets, _depth=_depth + 1)

    if isinstance(v, str):
        if field_no_clean: return v
        if field_pii_only:
            if not (remove_pii or redact_secrets): return v
            # Secrets first, then NER (natural text), then regex PII.
            if redact_secrets: v = _redact_secrets_fn(v, mask=pii_mask, pseudo_registry=pseudo_registry, counters=pii_counters)
            if remove_pii:
                if ner_redactor: v = ner_redactor.redact(v, mask=pii_mask, pseudo_registry=pseudo_registry, counters=pii_counters)
                v = redact_pii(v, mask=pii_mask, extra_patterns=extra_pii, pseudo_registry=pseudo_registry, counters=pii_counters)
            return v
        cleaned = clean_text(v, remove_html)
        if redact_secrets:
            cleaned = _redact_secrets_fn(cleaned, mask=pii_mask, pseudo_registry=pseudo_registry, counters=pii_counters)
        if remove_pii:
            if ner_redactor:
                cleaned = ner_redactor.redact(cleaned, mask=pii_mask, pseudo_registry=pseudo_registry, counters=pii_counters)
            cleaned = redact_pii(cleaned, mask=pii_mask, extra_patterns=extra_pii, pseudo_registry=pseudo_registry, counters=pii_counters)
        if truncator: cleaned = truncator.truncate(cleaned)
        return cleaned
    if isinstance(v, dict): return {k: _sanitize_value(val, **kw) for k, val in v.items()}
    if isinstance(v, list): return [_sanitize_value(item, **kw) for item in v]
    return v

def make_report_redactor(
    remove_pii: bool, redact_secrets: bool, extra_pii: Optional[List] = None,
    ner_redactor: Optional[Any] = None, max_depth: int = _MAX_DEPTH_DEFAULT,
) -> Optional[Callable[[Any], Any]]:
    """Redactor for audit-report samples of dropped records.

    Uses plain token replacement (no masking, no pseudonym registry) so it
    neither leaks partial values nor mutates the run's pseudonym map. Returns
    None when the run redacts nothing (samples then mirror the data as-is)."""
    if not (remove_pii or redact_secrets):
        return None

    def _redact(record: Any) -> Any:
        return _sanitize_value(
            record, remove_html=False, remove_pii=remove_pii, pii_mask=False,
            extra_pii=extra_pii, pseudo_registry=None, field_pii_only=True,
            field_no_clean=False, max_depth=max_depth, truncator=None,
            ner_redactor=ner_redactor if remove_pii else None, pii_counters=None,
            redact_secrets=redact_secrets)

    return _redact


class Transformed(NamedTuple):
    """Result of the per-record stage: the cleaned record, or None plus the
    reason it failed a gate. `quality_text` is the text the gates saw."""
    record: Optional[Dict[str, Any]]
    reason: Optional[FilterReason]
    quality_text: str
    lang: Optional[str]


def sanitize_record(
    record: Any, config: SanitizerConfig, *,
    pseudo_registry: Optional[PseudoRegistry] = None,
    pii_counters: Optional[Dict[str, int]] = None,
    truncator: Optional[TokenTruncator] = None,
    ner_redactor: Optional[Any] = None,
    lang_filter: Optional[Set[str]] = None,
    quality_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
) -> Transformed:
    """Clean, redact and gate one record (no cross-record state besides the
    optional pseudonym registry). Optional resources are normally supplied by
    RecordTransformer, which builds them from the config."""
    if not isinstance(record, dict):
        return Transformed(None, FilterReason.QUALITY, '', None)
    c = config

    renames, drops, pii_only, no_clean = c.field_ops if c.field_ops else ({}, set(), set(), set())
    if drops: record = {k: v for k, v in record.items() if k not in drops}
    if renames: record = {renames.get(k, k): v for k, v in record.items()}

    sanitized: Dict[str, Any] = {
        fname: _sanitize_value(
            val, remove_html=c.clean_html, remove_pii=c.remove_pii, pii_mask=c.pii_mask,
            extra_pii=c.extra_pii_patterns, pseudo_registry=pseudo_registry,
            field_pii_only=(fname in pii_only), field_no_clean=(fname in no_clean),
            max_depth=c.max_depth, truncator=truncator,
            ner_redactor=ner_redactor, pii_counters=pii_counters,
            redact_secrets=c.redact_secrets
        ) for fname, val in record.items()
    }

    for rf in c.require_fields or ():
        v = sanitized.get(rf)
        is_empty = v is None or (isinstance(v, str) and not v.strip()) or (not isinstance(v, (bool, int, float)) and not v)
        if is_empty: return Transformed(None, FilterReason.REQUIRE, '', None)

    quality_text = extract_text_for_quality(sanitized, text_fields=c.text_fields,
                                            max_depth=c.text_fields_depth)

    if c.reject_code and is_code_heuristic(quality_text):
        return Transformed(None, FilterReason.CODE, '', None)
    if c.reject_profanity and contains_profanity(quality_text):
        return Transformed(None, FilterReason.PROFANITY, '', None)
    if _check_quality_reason(quality_text, c):
        return Transformed(None, FilterReason.QUALITY, '', None)
    if quality_fn and not quality_fn(sanitized):
        return Transformed(None, FilterReason.QUALITY, '', None)

    detected_lang: Optional[str] = None
    if lang_filter:
        detected_lang, _conf = detect_language(quality_text, min_confidence=c.lang_confidence)
        if detected_lang not in lang_filter:
            return Transformed(None, FilterReason.LANGUAGE, '', None)

    # LLM Formatting
    if c.format_chatml:
        sanitized = format_chatml(sanitized)
    elif c.format_instruct:
        sanitized = format_instruct(sanitized)

    return Transformed(sanitized, None, quality_text, detected_lang)


class RecordTransformer:
    """The per-record stage of the pipeline, built from a SanitizerConfig.

    Owns the stage's resources (token truncator, NER model, language filter,
    quality script). It keeps no cross-record state, so worker processes build
    their own copy from the (picklable) config; the stateful stages live in
    Sanitizer."""

    def __init__(self, config: SanitizerConfig, ner_redactor: Optional[Any] = None) -> None:
        self.config = config
        self.truncator = TokenTruncator(config.max_tokens, config.tokenizer) \
            if config.max_tokens else None
        self.lang_filter: Optional[Set[str]] = \
            {x.lower() for x in config.lang_filter} if config.lang_filter else None
        self.quality_fn: Optional[Callable[[Dict[str, Any]], bool]] = None
        if config.quality_script:
            from sanitizer_pro.config import load_quality_script
            self.quality_fn = load_quality_script(config.quality_script)
        self.ner = ner_redactor
        if self.ner is None and config.pii_ner and config.remove_pii:
            from sanitizer_pro.ner import NERRedactor
            self.ner = NERRedactor(backend=config.pii_ner_backend,
                                   entities=config.pii_ner_entities, model=config.pii_ner_model)

    def transform(self, record: Any, pseudo_registry: Optional[PseudoRegistry] = None,
                  pii_counters: Optional[Dict[str, int]] = None) -> Transformed:
        return sanitize_record(
            record, self.config, pseudo_registry=pseudo_registry, pii_counters=pii_counters,
            truncator=self.truncator, ner_redactor=self.ner, lang_filter=self.lang_filter,
            quality_fn=self.quality_fn)

    def report_redactor(self) -> Optional[Callable[[Any], Any]]:
        c = self.config
        return make_report_redactor(c.remove_pii, c.redact_secrets, extra_pii=c.extra_pii_patterns,
                                    ner_redactor=self.ner, max_depth=c.max_depth)


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
