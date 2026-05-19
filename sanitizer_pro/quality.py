"""Quality scoring, language detection, and content filtering."""
import argparse
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

try:
    from langdetect import detect_langs, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

def extract_text_for_quality(
    record: Any, text_fields: Optional[List[str]] = None,
    max_depth: int = 20, _depth: int = 0, _budget: Optional[List[int]] = None
) -> str:
    """Recursively extract text for quality scoring with an 8192 char budget."""
    _MAX_QUALITY_CHARS = 8192
    if _budget is None:
        _budget = [_MAX_QUALITY_CHARS]
    if _depth > max_depth or _budget[0] <= 0:
        return ''
    if isinstance(record, str):
        chunk = record[:_budget[0]]
        _budget[0] -= len(chunk)
        return chunk
    if isinstance(record, dict):
        vals = [record.get(f, '') for f in text_fields] if text_fields and _depth == 0 else list(record.values())
        parts = [extract_text_for_quality(v, max_depth=max_depth, _depth=_depth + 1, _budget=_budget) for v in vals if _budget[0] > 0]
        return ' '.join(parts)
    if isinstance(record, list):
        parts = [extract_text_for_quality(i, max_depth=max_depth, _depth=_depth + 1, _budget=_budget) for i in record if _budget[0] > 0]
        return ' '.join(parts)
    return ''

def _check_quality_reason(text: str, args: argparse.Namespace) -> Optional[str]:
    if not text: return 'empty text'
    if len(text) < args.min_chars: return f'too short ({len(text)} < {args.min_chars})'
    if len(text) > args.max_chars: return f'too long ({len(text)} > {args.max_chars})'
    words = re.findall(r'\b\w+\b', text)
    if len(words) < args.min_words: return f'too few words ({len(words)} < {args.min_words})'
    ur = len(set(words)) / len(words) if words else 0.0
    if ur < args.min_unique_ratio: return f'low unique-word ratio ({ur:.3f})'
    ar = sum(1 for c in text if ord(c) < 128) / len(text)
    if ar < args.min_ascii_ratio: return f'low ASCII ratio ({ar:.3f})'
    
    if getattr(args, 'reject_allcaps', False):
        threshold = getattr(args, 'allcaps_min_len', 50)
        min_alpha = getattr(args, 'allcaps_min_alpha', 10)
        if len(text) > threshold:
            alpha_chars = [c for c in text if c.isalpha()]
            if len(alpha_chars) >= min_alpha and (sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars) >= 0.9):
                return 'all-caps'
    return None

def is_high_quality(text: str, args: argparse.Namespace) -> bool:
    return _check_quality_reason(text, args) is None

def detect_language(text: str, min_confidence: float = 0.0) -> Tuple[Optional[str], float]:
    if not LANGDETECT_AVAILABLE or not text:
        return None, 0.0
    try:
        results = detect_langs(text)
        if not results: return None, 0.0
        top = results[0]
        conf = float(top.prob)
        return (top.lang, conf) if conf >= min_confidence else (None, conf)
    except Exception:
        return None, 0.0

def is_code_heuristic(text: str) -> bool:
    """Fast heuristic for code detection based on symbol density and keywords."""
    if not text: return False
    code_chars = sum(1 for c in text if c in '{}[]();=<>')
    if code_chars / len(text) > 0.05: return True
    keywords = {'def ', 'function ', 'import ', 'require(', 'console.log', 'public static void'}
    return any(kw in text for kw in keywords)

_PROFANITY_REGEX = re.compile(r'\b(badword1|badword2)\b', re.IGNORECASE) # Placeholder regex
def contains_profanity(text: str) -> bool:
    return bool(_PROFANITY_REGEX.search(text))
