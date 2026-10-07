"""Benchmark decontamination: n-gram overlap detection against eval test sets.

Removes training records that collide with benchmark data so fine-tuned models
are not evaluated on material they trained on. Follows the standard n-gram
collision approach used for GPT-3/Llama-style decontamination: a record is
contaminated when >= min_hits of its normalized word n-grams appear in the
reference index built from benchmark test sets.

References come from two sources, usable together:
  * Local files (``--decontam-refs``) in any supported input format.
  * Named benchmarks (``--decontaminate mmlu,gsm8k,...``) auto-downloaded from
    the Hugging Face Hub's parquet endpoints and cached locally.
"""
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

from sanitizer_pro.utils import ConfigurationError

_NORM_RE = re.compile(r'[\W_]+', re.UNICODE)


def normalize_for_ngrams(text: str) -> List[str]:
    """Lowercase, strip punctuation, and split into words."""
    return _NORM_RE.sub(' ', text.lower()).split()


class NGramIndex:
    """Index of word n-grams from benchmark reference texts.

    N-grams are stored as 64-bit hashes (not joined strings) mapped to the
    index of the source that first contributed them, so a hit can be
    attributed to a benchmark. Hashes use Python's hash(); the index lives in
    one process and is never persisted, so per-process salting is harmless."""

    def __init__(self, n: int = 8, min_hits: int = 1, min_ref_words: int = 4) -> None:
        if n < 2:
            raise ConfigurationError("--decontam-ngram must be >= 2.")
        if min_hits < 1:
            raise ConfigurationError("--decontam-min-hits must be >= 1.")
        self.n = n
        self.min_hits = min_hits
        self.min_ref_words = min_ref_words
        self._ngrams: Dict[int, int] = {}
        # Benchmark items shorter than n words are indexed whole, bucketed by
        # word count so lookups only slide windows for lengths that exist.
        self._short: Dict[int, Dict[int, int]] = {}
        self.sources: List[str] = []
        self.ref_count = 0
        self.ref_counts: Dict[str, int] = {}

    def _source_id(self, source: str) -> int:
        if source not in self.sources:
            self.sources.append(source)
        return self.sources.index(source)

    def add_reference(self, text: str, source: str = 'refs') -> None:
        words = normalize_for_ngrams(text)
        if len(words) < self.min_ref_words:
            return
        sid = self._source_id(source)
        self.ref_count += 1
        self.ref_counts[source] = self.ref_counts.get(source, 0) + 1
        if len(words) < self.n:
            self._short.setdefault(len(words), {}).setdefault(hash(' '.join(words)), sid)
            return
        grams = self._ngrams
        for i in range(len(words) - self.n + 1):
            grams.setdefault(hash(' '.join(words[i:i + self.n])), sid)

    def __len__(self) -> int:
        return len(self._ngrams) + sum(len(s) for s in self._short.values())

    def _hits(self, text: str, limit: int) -> List[int]:
        """Source ids of colliding n-grams, stopping after `limit` hits."""
        if not self._ngrams and not self._short:
            return []
        words = normalize_for_ngrams(text)
        found: List[int] = []
        for i in range(max(0, len(words) - self.n + 1)):
            sid = self._ngrams.get(hash(' '.join(words[i:i + self.n])))
            if sid is not None:
                found.append(sid)
                if len(found) >= limit:
                    return found
        for k, bucket in self._short.items():
            for i in range(len(words) - k + 1) if len(words) >= k else ():
                sid = bucket.get(hash(' '.join(words[i:i + k])))
                if sid is not None:
                    found.append(sid)
                    if len(found) >= limit:
                        return found
        return found

    def contamination_hits(self, text: str, max_hits: Optional[int] = None) -> int:
        """Count reference n-grams present in text (early exit at max_hits)."""
        return len(self._hits(text, max_hits if max_hits is not None else self.min_hits))

    def match(self, text: str) -> Optional[str]:
        """The benchmark a contaminated text overlaps (most hits), else None."""
        found = self._hits(text, self.min_hits)
        if len(found) < self.min_hits:
            return None
        return self.sources[max(set(found), key=found.count)]

    def is_contaminated(self, text: str) -> bool:
        return self.match(text) is not None


# =============================================================================
# Benchmark registry
# =============================================================================

@dataclass(frozen=True)
class BenchmarkSpec:
    repo: str
    parts: Tuple[Tuple[str, str], ...]  # (config, split) pairs; config '*' = all configs
    fields: Tuple[str, ...]             # indexed field paths (dotted; lists allowed)
    note: str = ''
    gated: bool = False                 # needs HF_TOKEN + accepted terms on the Hub


# Questions *and* answers/solutions/choices are indexed: training on a
# benchmark's solutions contaminates it as much as training on its questions.
# Values under 4 words (letters, numbers) are skipped by NGramIndex.
KNOWN_BENCHMARKS: Dict[str, BenchmarkSpec] = {
    'mmlu': BenchmarkSpec('cais/mmlu', (('all', 'test'),), ('question', 'choices'),
                          'MMLU test questions + choices (14k)'),
    'mmlu-pro': BenchmarkSpec('TIGER-Lab/MMLU-Pro', (('default', 'test'),),
                              ('question', 'options'), 'MMLU-Pro test (12k, 10 options)'),
    'gpqa': BenchmarkSpec('Idavidrein/gpqa', (('gpqa_extended', 'train'),),
                          ('Question', 'Correct Answer', 'Explanation'),
                          'GPQA extended (incl. Diamond); gated', gated=True),
    'gsm8k': BenchmarkSpec('openai/gsm8k', (('main', 'test'),), ('question', 'answer'),
                           'GSM8K test problems + solutions'),
    'gsm-plus': BenchmarkSpec('qintongli/GSM-Plus', (('default', 'test'),),
                              ('question', 'solution'), 'GSM-Plus perturbed GSM8K (10.5k)'),
    'math': BenchmarkSpec('EleutherAI/hendrycks_math', (('*', 'test'),),
                          ('problem', 'solution'), 'MATH test (5k, all subjects)'),
    'math-500': BenchmarkSpec('HuggingFaceH4/MATH-500', (('default', 'test'),),
                              ('problem', 'solution'), 'MATH-500 subset'),
    'aime-2024': BenchmarkSpec('HuggingFaceH4/aime_2024', (('default', 'train'),),
                               ('problem', 'solution'), 'AIME 2024 problems'),
    'aime-2025': BenchmarkSpec('yentinglin/aime_2025', (('default', 'train'),), ('problem',),
                               'AIME 2025 problems'),
    'ifeval': BenchmarkSpec('google/IFEval', (('default', 'train'),), ('prompt',),
                            'IFEval prompts (541)'),
    'bbh': BenchmarkSpec('lukaemon/bbh', (('*', 'test'),), ('input',),
                         'BIG-Bench Hard, all 27 tasks'),
    'musr': BenchmarkSpec('TAUR-Lab/MuSR', (('default', 'murder_mysteries'),
                                            ('default', 'object_placements'),
                                            ('default', 'team_allocation')),
                          ('narrative', 'question'), 'MuSR narratives'),
    'hle': BenchmarkSpec('cais/hle', (('default', 'test'),), ('question', 'answer'),
                         "Humanity's Last Exam; gated", gated=True),
    'simpleqa': BenchmarkSpec('basicv8vc/SimpleQA', (('default', 'test'),),
                              ('problem', 'answer'), 'SimpleQA questions'),
    'humaneval': BenchmarkSpec('openai/openai_humaneval', (('openai_humaneval', 'test'),),
                               ('prompt', 'canonical_solution'),
                               'HumanEval code generation problems'),
    'humaneval-plus': BenchmarkSpec('evalplus/humanevalplus', (('default', 'test'),),
                                    ('prompt', 'canonical_solution'), 'HumanEval+ (EvalPlus)'),
    'mbpp': BenchmarkSpec('google-research-datasets/mbpp', (('full', 'test'),),
                          ('text', 'code'), 'MBPP code problems'),
    'mbpp-plus': BenchmarkSpec('evalplus/mbppplus', (('default', 'test'),), ('prompt', 'code'),
                               'MBPP+ (EvalPlus)'),
    'arc': BenchmarkSpec('allenai/ai2_arc',
                         (('ARC-Challenge', 'test'), ('ARC-Easy', 'test')),
                         ('question', 'choices.text'), 'ARC Challenge + Easy test questions'),
    'hellaswag': BenchmarkSpec('Rowan/hellaswag', (('default', 'validation'),),
                               ('ctx', 'endings'), 'HellaSwag validation contexts + endings'),
    'truthfulqa': BenchmarkSpec('truthfulqa/truthful_qa', (('generation', 'validation'),),
                                ('question', 'best_answer'), 'TruthfulQA questions'),
    'winogrande': BenchmarkSpec('allenai/winogrande', (('winogrande_xl', 'validation'),),
                                ('sentence',), 'WinoGrande XL validation sentences'),
}

BENCHMARK_GROUPS: Dict[str, Tuple[str, ...]] = {
    # HF Open LLM Leaderboard v2 task set.
    'open-llm-v2': ('ifeval', 'bbh', 'math', 'gpqa', 'musr', 'mmlu-pro'),
}


def resolve_benchmark_names(spec: str) -> List[str]:
    """Names, groups ('open-llm-v2') and 'all' (every ungated benchmark)."""
    names: List[str] = []
    for raw in spec.split(','):
        name = raw.strip().lower()
        if not name:
            continue
        if name == 'all':
            expanded = [n for n, b in KNOWN_BENCHMARKS.items() if not b.gated]
        else:
            expanded = list(BENCHMARK_GROUPS.get(name, (name,)))
        names += [n for n in expanded if n not in names]
    unknown = [n for n in names if n not in KNOWN_BENCHMARKS]
    if unknown:
        raise ConfigurationError(
            f"Unknown benchmark(s): {', '.join(unknown)}. "
            f"Available: {', '.join(sorted(KNOWN_BENCHMARKS))}, groups "
            f"{', '.join(BENCHMARK_GROUPS)}, or 'all' (ungated).")
    return names


def iter_benchmark_texts(name: str, cache_dir: Optional[str] = None) -> Iterator[str]:
    """Yield the reference text of every record in a known benchmark."""
    from sanitizer_pro.hub import iter_parquet_texts
    spec = KNOWN_BENCHMARKS[name]
    try:
        yield from iter_parquet_texts(spec.repo, spec.parts, spec.fields, cache_dir=cache_dir)
    except ConfigurationError as exc:
        if spec.gated and ('401' in str(exc) or '403' in str(exc)):
            raise ConfigurationError(
                f"'{name}' ({spec.repo}) is gated: accept its terms on the Hugging Face Hub "
                "and set HF_TOKEN.") from None
        raise


def _iter_strings(value: Any, _depth: int = 0) -> Iterator[str]:
    if _depth > 20:
        return
    if isinstance(value, str):
        if value.strip():
            yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _iter_strings(v, _depth + 1)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _iter_strings(v, _depth + 1)


def full_text_for_decontam(record: Any) -> str:
    """Every string in the record, untruncated: contamination can sit in any
    field (e.g. the answer) and anywhere in a long document, not just in the
    8 KB quality-scoring slice."""
    return ' '.join(_iter_strings(record))


def iter_reference_file_texts(path: str, encoding: str = 'utf-8') -> Iterator[str]:
    """Yield reference texts from a local file in any supported input format."""
    from sanitizer_pro.io.readers import read_records
    if not os.path.exists(path):
        raise ConfigurationError(f"Decontamination reference file not found: {path}")
    for record in read_records(path, encoding=encoding):
        yield from _iter_strings(record)


def build_index(
    benchmarks: Optional[List[str]] = None,
    ref_files: Optional[List[str]] = None,
    cache_dir: Optional[str] = None,
    ngram: int = 8,
    min_hits: int = 1,
    encoding: str = 'utf-8',
) -> NGramIndex:
    """Build the contamination index from named benchmarks and/or local files."""
    index = NGramIndex(n=ngram, min_hits=min_hits)
    for name in benchmarks or []:
        before = index.ref_count
        for text in iter_benchmark_texts(name, cache_dir):
            index.add_reference(text, source=name)
        logging.info(f"Decontamination: indexed {index.ref_count - before:,} texts from '{name}'.")
    for path in ref_files or []:
        before = index.ref_count
        for text in iter_reference_file_texts(path, encoding=encoding):
            index.add_reference(text, source=os.path.basename(path))
        logging.info(f"Decontamination: indexed {index.ref_count - before:,} texts from {path}.")
    if index.ref_count == 0:
        raise ConfigurationError("Decontamination requested but no reference texts were indexed.")
    logging.info(f"Decontamination index ready: {index.ref_count:,} reference texts, "
                 f"{len(index):,} {index.n}-gram entries.")
    return index
