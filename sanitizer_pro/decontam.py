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
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from sanitizer_pro.utils import ConfigurationError

_NORM_RE = re.compile(r'[\W_]+', re.UNICODE)


def normalize_for_ngrams(text: str) -> List[str]:
    """Lowercase, strip punctuation, and split into words."""
    return _NORM_RE.sub(' ', text.lower()).split()


class NGramIndex:
    """Set-based index of word n-grams from benchmark reference texts."""

    def __init__(self, n: int = 8, min_hits: int = 1, min_ref_words: int = 4) -> None:
        if n < 2:
            raise ConfigurationError("--decontam-ngram must be >= 2.")
        if min_hits < 1:
            raise ConfigurationError("--decontam-min-hits must be >= 1.")
        self.n = n
        self.min_hits = min_hits
        self.min_ref_words = min_ref_words
        self._ngrams: Set[str] = set()
        # Benchmark items shorter than n words are indexed whole, bucketed by
        # word count so lookups only slide windows for lengths that exist.
        self._short: Dict[int, Set[str]] = {}
        self.ref_count = 0

    def add_reference(self, text: str) -> None:
        words = normalize_for_ngrams(text)
        if len(words) < self.min_ref_words:
            return
        self.ref_count += 1
        if len(words) < self.n:
            self._short.setdefault(len(words), set()).add(' '.join(words))
            return
        for i in range(len(words) - self.n + 1):
            self._ngrams.add(' '.join(words[i:i + self.n]))

    def __len__(self) -> int:
        return len(self._ngrams) + sum(len(s) for s in self._short.values())

    def contamination_hits(self, text: str, max_hits: Optional[int] = None) -> int:
        """Count reference n-grams present in text (early exit at max_hits)."""
        if not self._ngrams and not self._short:
            return 0
        limit = max_hits if max_hits is not None else self.min_hits
        words = normalize_for_ngrams(text)
        hits = 0
        for i in range(max(0, len(words) - self.n + 1)):
            if ' '.join(words[i:i + self.n]) in self._ngrams:
                hits += 1
                if hits >= limit:
                    return hits
        for k, bucket in self._short.items():
            if len(words) < k:
                continue
            for i in range(len(words) - k + 1):
                if ' '.join(words[i:i + k]) in bucket:
                    hits += 1
                    if hits >= limit:
                        return hits
        return hits

    def is_contaminated(self, text: str) -> bool:
        return self.contamination_hits(text, max_hits=self.min_hits) >= self.min_hits


# =============================================================================
# Benchmark registry
# =============================================================================

@dataclass(frozen=True)
class BenchmarkSpec:
    repo: str
    parts: Tuple[Tuple[str, str], ...]  # (config, split) pairs
    fields: Tuple[str, ...]             # record fields whose text is indexed
    note: str = ''


KNOWN_BENCHMARKS: Dict[str, BenchmarkSpec] = {
    'mmlu': BenchmarkSpec('cais/mmlu', (('all', 'test'),), ('question',),
                          'MMLU test questions (14k)'),
    'gsm8k': BenchmarkSpec('openai/gsm8k', (('main', 'test'),), ('question',),
                           'GSM8K grade-school math test set'),
    'humaneval': BenchmarkSpec('openai/openai_humaneval', (('openai_humaneval', 'test'),),
                               ('prompt', 'canonical_solution'),
                               'HumanEval code generation problems'),
    'arc': BenchmarkSpec('allenai/ai2_arc',
                         (('ARC-Challenge', 'test'), ('ARC-Easy', 'test')),
                         ('question',), 'ARC Challenge + Easy test questions'),
    'hellaswag': BenchmarkSpec('Rowan/hellaswag', (('default', 'validation'),), ('ctx',),
                               'HellaSwag validation contexts (eval split)'),
    'truthfulqa': BenchmarkSpec('truthfulqa/truthful_qa', (('generation', 'validation'),),
                                ('question',), 'TruthfulQA questions'),
    'winogrande': BenchmarkSpec('allenai/winogrande', (('winogrande_xl', 'validation'),),
                                ('sentence',), 'WinoGrande XL validation sentences'),
    'mbpp': BenchmarkSpec('google-research-datasets/mbpp', (('full', 'test'),),
                          ('text', 'code'), 'MBPP code problems'),
}


def resolve_benchmark_names(spec: str) -> List[str]:
    names = [s.strip().lower() for s in spec.split(',') if s.strip()]
    if 'all' in names:
        return list(KNOWN_BENCHMARKS)
    unknown = [n for n in names if n not in KNOWN_BENCHMARKS]
    if unknown:
        raise ConfigurationError(
            f"Unknown benchmark(s): {', '.join(unknown)}. "
            f"Available: {', '.join(sorted(KNOWN_BENCHMARKS))}, or 'all'.")
    return names


def iter_benchmark_texts(name: str, cache_dir: Optional[str] = None) -> Iterator[str]:
    """Yield the reference text of every record in a known benchmark."""
    from sanitizer_pro.hub import iter_parquet_texts
    spec = KNOWN_BENCHMARKS[name]
    yield from iter_parquet_texts(spec.repo, spec.parts, spec.fields, cache_dir=cache_dir)


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
            index.add_reference(text)
        logging.info(f"Decontamination: indexed {index.ref_count - before:,} texts from '{name}'.")
    for path in ref_files or []:
        before = index.ref_count
        for text in iter_reference_file_texts(path, encoding=encoding):
            index.add_reference(text)
        logging.info(f"Decontamination: indexed {index.ref_count - before:,} texts from {path}.")
    if index.ref_count == 0:
        raise ConfigurationError("Decontamination requested but no reference texts were indexed.")
    logging.info(f"Decontamination index ready: {index.ref_count:,} reference texts, "
                 f"{len(index):,} {index.n}-gram entries.")
    return index
