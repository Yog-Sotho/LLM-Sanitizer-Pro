"""Rule-based document filters from published pretraining pipelines.

Dependency-free re-implementations of the heuristics behind Gopher, C4 and
FineWeb, with the thresholds and failure-reason names of the reference
implementation in HuggingFace datatrove (Apache-2.0):

  * ``gopher``             Gopher quality rules (Rae et al. 2021, §A1)
  * ``gopher-repetition``  Gopher repetition rules (duplicate lines,
                           paragraphs and n-grams; Table A1)
  * ``c4``                 C4 document rules (Raffel et al. 2020): curly
                           brackets, lorem ipsum, too few sentences once
                           disqualified lines are excluded
  * ``fineweb``            FineWeb quality rules (Penedo et al. 2024)

They target web-scale *pretraining* documents and are tuned for English
(Gopher's stop-word rule in particular) — apply them after a language filter.
Words are tokenized with a regex (words, single CJK characters, punctuation)
instead of NLTK, so token-level ratios can differ marginally from datatrove.
C4 is applied as a document-level decision; lines are not removed from records.
"""
import re
import unicodedata
from collections import Counter
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from sanitizer_pro.quality import _UNSPACED
from sanitizer_pro.utils import ConfigurationError

# Unicode Sentence_Terminal characters (as used by datatrove/FineWeb).
TERMINAL_PUNCTUATION = frozenset(
    "!.?։؝؞؟۔܀܁܂߹࠷࠹࠽࠾।॥၊။።፧፨᙮᜵᜶។៕៖៙៚᠃᠉᥄᥅᪨᪩᪪᪫᭚᭛᭞᭟᭽᭾᰻᰼᱾᱿‼‽⁇⁈⁉⸮⸼⹓⹔。꓿꘎꘏꛳꛷꡶꡷꣎꣏꤯꧈꧉꩝꩞꩟꫰꫱꯫﹒﹖﹗！．？｡"
    "𐩖𐩗𐽕𐽖𐽗𐽘𐽙𐾆𐾇𐾈𐾉𑁇𑁈𑂾𑂿𑃀𑃁𑅁𑅂𑅃𑇅𑇆𑇍𑇞𑇟𑈸𑈹𑈻𑈼𑊩𑑋𑑌𑗂𑗃𑗉𑗊𑗋𑗌𑗍𑗎𑗏𑗐𑗑𑗒𑗓𑗔𑗕𑗖𑗗𑙁𑙂𑜼𑜽𑜾𑥄𑥆𑩂𑩃𑪛𑪜𑱁𑱂𑻷𑻸𑽃𑽄"
    "𖩮𖩯𖫵𖬷𖬸𖭄𖺘𛲟𝪈")
_TERMINAL = tuple(TERMINAL_PUNCTUATION)

_W = rf'[^\W{_UNSPACED}]'  # \w minus unspaced scripts
# Mirrors spaCy's default tokenization closely enough for these ratios: a
# character per token for unspaced scripts, numbers with separators kept
# whole, apostrophe suffixes attached ('s), and punctuation runs as one token.
_TOKEN_RE = re.compile(rf"[{_UNSPACED}]|\d+(?:[.,:]\d+)+|['’]?{_W}+|[^\w\s]+")


def tokenize(text: str) -> List[str]:
    """Words, single characters of unspaced scripts, and punctuation runs."""
    return _TOKEN_RE.findall(text)


def _is_symbol(ch: str) -> bool:
    return unicodedata.category(ch)[0] in 'PSC'  # punctuation, symbols, controls


# -- Gopher quality ------------------------------------------------------------

GOPHER_STOP_WORDS = frozenset(["the", "be", "to", "of", "and", "that", "have", "with"])


def gopher_quality(text: str) -> Optional[str]:
    words = tokenize(text)
    if not words:
        return "gopher_short_doc"
    n_words = len(words)
    content_words = [w for w in words if any(not _is_symbol(ch) for ch in w)]
    if len(content_words) < 50:
        return "gopher_short_doc"
    if len(content_words) > 100_000:
        return "gopher_long_doc"
    avg_len = sum(len(w) for w in content_words) / len(content_words)
    if avg_len < 3:
        return "gopher_below_avg_threshold"
    if avg_len > 10:
        return "gopher_above_avg_threshold"
    if text.count("#") / n_words > 0.1:
        return "gopher_too_many_hashes"
    if (text.count("...") + text.count("…")) / n_words > 0.1:
        return "gopher_too_many_ellipsis"
    lines = text.splitlines() or [text]
    if sum(ln.lstrip().startswith(("•", "-")) for ln in lines) / len(lines) > 0.9:
        return "gopher_too_many_bullets"
    if sum(ln.rstrip().endswith(("...", "…")) for ln in lines) / len(lines) > 0.3:
        return "gopher_too_many_end_ellipsis"
    if sum(any(c.isalpha() for c in w) for w in words) / n_words < 0.8:
        return "gopher_below_alpha_threshold"
    if len(GOPHER_STOP_WORDS.intersection(words)) < 2:  # case-sensitive, as in datatrove
        return "gopher_enough_stop_words"
    return None


# -- Gopher repetition ---------------------------------------------------------

_PARA_RE = re.compile(r"\n{2,}")
_LINE_RE = re.compile(r"\n+")
_TOP_NGRAMS = ((2, 0.20), (3, 0.18), (4, 0.16))
_DUP_NGRAMS = ((5, 0.15), (6, 0.14), (7, 0.13), (8, 0.12), (9, 0.11), (10, 0.10))


def _find_duplicates(items: Iterable[str]) -> Tuple[int, int]:
    """(# duplicate elements, # characters in duplicate elements)."""
    seen = set()
    dup_elems = dup_chars = 0
    for item in items:
        if item in seen:
            dup_elems += 1
            dup_chars += len(item)
        else:
            seen.add(item)
    return dup_elems, dup_chars


def _all_duplicate_ngram_chars(words: List[str], n: int) -> int:
    seen = set()
    repeated = idx = 0
    while idx < len(words) - n + 1:
        gram = "".join(words[idx:idx + n])
        if gram in seen:
            repeated += len(gram)
            idx += n
        else:
            seen.add(gram)
            idx += 1
    return repeated


def gopher_repetition(text: str) -> Optional[str]:
    if not text:
        return "empty"
    paragraphs = _PARA_RE.split(text.strip())
    dup, dup_chars = _find_duplicates(paragraphs)
    if dup / len(paragraphs) > 0.3:
        return "dup_para_frac"
    if dup_chars / len(text) > 0.2:
        return "dup_para_char_frac"
    lines = _LINE_RE.split(text)
    dup, dup_chars = _find_duplicates(lines)
    if dup / len(lines) > 0.3:
        return "dup_line_frac"
    if dup_chars / len(text) > 0.2:
        return "dup_line_char_frac"
    words = tokenize(text)
    for n, frac in _TOP_NGRAMS:
        grams = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
        if grams:
            gram, count = Counter(grams).most_common(1)[0]
            if len(gram) * count / len(text) > frac:
                return f"top_{n}_gram"
    for n, frac in _DUP_NGRAMS:
        if _all_duplicate_ngram_chars(words, n) / len(text) > frac:
            return f"duplicated_{n}_n_grams"
    return None


# -- C4 --------------------------------------------------------------------------

_CITATION_RE = re.compile(r"\[\d*]|\[edit]|\[citation needed]")
_C4_END = (".", "?", "!", '"', "'")
_POLICY = ("terms of use", "privacy policy", "cookie policy", "uses cookies",
           "use of cookies", "use cookies")
_SENTENCE_END_RE = re.compile(r"[.!?]+(?=\s|$)")


def c4_quality(text: str) -> Optional[str]:
    sentences = 0
    for raw in text.splitlines():
        line = raw.strip()
        words = line.split()
        if any(len(w) > 1000 for w in words):
            continue
        line = _CITATION_RE.sub("", line)
        if not line.endswith(_C4_END) or line.endswith("..."):
            continue
        if len(words) < 3:
            continue
        lower = line.lower()
        if "lorem ipsum" in lower:
            return "lorem_ipsum"
        if "javascript" in lower:
            continue
        if "{" in line:
            return "curly_bracket"
        if any(p in lower for p in _POLICY):
            continue
        sentences += max(1, len(_SENTENCE_END_RE.findall(line)))
    if sentences < 5:
        return "too_few_sentences"
    return None


# -- FineWeb ---------------------------------------------------------------------

def fineweb_quality(text: str) -> Optional[str]:
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if not lines:
        return "empty"
    if sum(ln.endswith(_TERMINAL) for ln in lines) / len(lines) < 0.12:
        return "line_punct_ratio"
    if sum(len(ln) <= 30 for ln in lines) / len(lines) > 0.67:
        return "short_line_ratio"
    if _find_duplicates(lines)[1] / max(1, len(text.replace("\n", ""))) > 0.01:
        return "char_dup_ratio"
    words = tokenize(text)
    if words and text.count("\n") / len(words) > 0.3:
        return "list_ratio"
    return None


RULE_SETS: Dict[str, Callable[[str], Optional[str]]] = {
    'gopher': gopher_quality,
    'gopher-repetition': gopher_repetition,
    'c4': c4_quality,
    'fineweb': fineweb_quality,
}


def resolve_rule_sets(names: Iterable[str]) -> List[str]:
    names = [n.strip().lower() for n in names if n.strip()]
    if 'all' in names:
        return list(RULE_SETS)
    unknown = [n for n in names if n not in RULE_SETS]
    if unknown:
        raise ConfigurationError(
            f"Unknown quality rule set(s): {', '.join(unknown)}. "
            f"Available: {', '.join(RULE_SETS)}, or 'all'.")
    return names


def check_rules(text: str, names: Iterable[str]) -> Optional[str]:
    """First failing rule as '<set>:<reason>', or None when all pass."""
    for name in names:
        reason = RULE_SETS[name](text)
        if reason:
            return f"{name}:{reason}"
    return None
