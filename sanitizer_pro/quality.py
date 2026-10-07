"""Quality scoring, language detection, and content filtering."""
import argparse
import re
from typing import Any, List, Optional, Tuple

try:
    from langdetect import detect_langs, DetectorFactory
    DetectorFactory.seed = 0  # langdetect is nondeterministic without a fixed seed
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

# Scripts written without spaces between words: each character counts as one
# word (Han, Hiragana/Katakana, Thai, Lao, Myanmar, Khmer). Otherwise a whole
# Chinese sentence is a single \w+ run and fails any word-count gate.
_UNSPACED = (r'\u0E00-\u0EFF\u1000-\u109F\u1780-\u17FF\u3040-\u30FF\u3400-\u4DBF'
             r'\u4E00-\u9FFF\uF900-\uFAFF')
_WORD_RE = re.compile(rf'[{_UNSPACED}]|(?:(?![{_UNSPACED}])\w)+')


def words_of(text: str) -> List[str]:
    """Script-aware word tokens used by the quality gates."""
    return _WORD_RE.findall(text)


def _check_quality_reason(text: str, args: argparse.Namespace) -> Optional[str]:
    if not text: return 'empty text'
    if len(text) < args.min_chars: return f'too short ({len(text)} < {args.min_chars})'
    if len(text) > args.max_chars: return f'too long ({len(text)} > {args.max_chars})'
    words = words_of(text)
    if len(words) < args.min_words: return f'too few words ({len(words)} < {args.min_words})'
    ur = len(set(words)) / len(words) if words else 0.0
    if ur < args.min_unique_ratio: return f'low unique-word ratio ({ur:.3f})'
    if args.min_ascii_ratio > 0:
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

_CODE_KEYWORD_REGEX = re.compile(
    r'(?:^|\n)\s*(?:def |class \w+[:(]|function\s*\w*\s*\(|import \w+|from \w+ import '
    r'|#include\s*<|public static void|const \w+\s*=|let \w+\s*=|var \w+\s*=)'
    r'|console\.log\(|require\([\'"]|=>\s*{|;\s*\n'
)

def is_code_heuristic(text: str) -> bool:
    """Fast heuristic for code detection based on symbol density and structure."""
    if not text: return False
    code_chars = sum(1 for c in text if c in '{}[]();=<>')
    if code_chars / len(text) > 0.05: return True
    return bool(_CODE_KEYWORD_REGEX.search(text))

# Compact default list targeting common English slurs/profanity. Kept intentionally
# small and high-precision; extend per-deployment via --pii-patterns-file style
# custom lists or a quality script for stricter policies.
_PROFANITY_WORDS = (
    'fuck', 'fucking', 'fucker', 'motherfucker', 'shit', 'bullshit', 'asshole',
    'bitch', 'bastard', 'cunt', 'dickhead', 'wanker', 'slut', 'whore', 'faggot',
    'nigger', 'nigga', 'retard', 'douchebag', 'jackass', 'prick', 'twat',
)
_PROFANITY_REGEX = re.compile(
    r'\b(?:' + '|'.join(re.escape(w) for w in _PROFANITY_WORDS) + r')\b', re.IGNORECASE)

def contains_profanity(text: str) -> bool:
    return bool(_PROFANITY_REGEX.search(text))
