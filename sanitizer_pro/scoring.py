"""Model-based and heuristic quality scoring for training records.

Produces a quality score in [0, 1] (higher = better) per record, used three
ways by the CLI:

  * ``--quality-min-score X``   — reject records scoring below an absolute bar
  * ``--keep-top-percent P``    — keep only the best P% of surviving records
  * ``--quality-score-field F`` — annotate output records with their score

Two backends:

  * **heuristic** (default, no dependencies): a composite of the
    prose-quality signals used by C4/Gopher-style pipelines — stopword
    density, mean word length, alphabetic-word fraction, trigram and line
    repetition, and symbol density. Good English prose lands around 0.8+;
    keyboard mash, boilerplate spam, and symbol soup fall well below.
  * **perplexity** (optional, ``transformers`` + ``torch``): negative
    log-likelihood under a small causal LM (default ``distilgpt2``), mapped
    log-linearly onto [0, 1] (perplexity 10 → 1.0, 10 000 → 0.0).
"""
import math
from typing import Callable, List, Optional

from sanitizer_pro.utils import ConfigurationError

_STOPWORDS = frozenset(
    'the a an and or but if of to in on for with as is are was were be been being '
    'it its this that these those there here i you he she they we me him her them us '
    'my your his their our not no nor do does did done have has had having at by from '
    'up down out over under again then once so than too very can will just should now '
    'what which who whom when where why how all any both each few more most other some '
    'such only own same s t don about against between into through during before after '
    'above below off further while because until although'.split())


def _band(x: float, lo0: float, lo1: float, hi1: float, hi0: float) -> float:
    """1.0 inside [lo1, hi1], ramping linearly to 0.0 at lo0 / hi0."""
    if x <= lo0 or x >= hi0:
        return 0.0
    if x < lo1:
        return (x - lo0) / (lo1 - lo0)
    if x <= hi1:
        return 1.0
    return (hi0 - x) / (hi0 - hi1)


class HeuristicScorer:
    """Dependency-free prose-quality score built from C4/Gopher-style signals."""

    backend_name = 'heuristic'

    def score(self, text: str) -> float:
        words = text.split()
        if len(words) < 3 or not text.strip():
            return 0.0
        n = len(words)
        subs: List[float] = []

        lowered = [w.strip('.,!?;:"\'()[]').lower() for w in words]
        stopword_frac = sum(1 for w in lowered if w in _STOPWORDS) / n
        subs.append(_band(stopword_frac, 0.05, 0.18, 0.55, 0.75))

        mean_word_len = sum(len(w) for w in words) / n
        subs.append(_band(mean_word_len, 2.0, 3.2, 7.0, 12.0))

        alpha_frac = sum(1 for w in words if any(c.isalpha() for c in w)) / n
        subs.append(_band(alpha_frac, 0.4, 0.75, 1.0, 1.001))

        symbol_frac = sum(1 for c in text if not (c.isalnum() or c.isspace())) / len(text)
        subs.append(_band(symbol_frac, -0.001, 0.0, 0.06, 0.25))

        base = sum(subs) / len(subs)

        # Repetition is a multiplicative penalty, not an averaged signal: spam
        # that repeats a phrase can look perfect on every other axis.
        penalty = 1.0
        if n >= 5:
            trigrams = [' '.join(lowered[i:i + 3]) for i in range(n - 2)]
            dup_frac = 1.0 - len(set(trigrams)) / len(trigrams)
            penalty *= max(0.0, 1.0 - dup_frac * 2.0)
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if len(lines) > 1:
            line_dup_frac = 1.0 - len(set(lines)) / len(lines)
            penalty *= max(0.0, 1.0 - line_dup_frac * 2.0)

        return round(base * penalty, 4)


_PPL_BEST, _PPL_WORST = 10.0, 10_000.0


def perplexity_to_score(ppl: float) -> float:
    """Map perplexity log-linearly onto [0, 1]: 10 → 1.0, 10 000 → 0.0."""
    if ppl <= _PPL_BEST:
        return 1.0
    if ppl >= _PPL_WORST:
        return 0.0
    return round(1.0 - (math.log(ppl) - math.log(_PPL_BEST))
                 / (math.log(_PPL_WORST) - math.log(_PPL_BEST)), 4)


class PerplexityScorer:
    """Quality via causal-LM perplexity (transformers + torch required)."""

    backend_name = 'perplexity'

    def __init__(self, model: Optional[str] = None, max_length: int = 512,
                 _ppl_fn: Optional[Callable[[str], float]] = None) -> None:
        if _ppl_fn is not None:
            self._ppl = _ppl_fn
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "--quality-scorer perplexity needs: pip install transformers torch. "
                "The default 'heuristic' scorer has no dependencies.") from None
        name = model or 'distilgpt2'
        self._torch = torch
        self._tok = AutoTokenizer.from_pretrained(name)
        self._model = AutoModelForCausalLM.from_pretrained(name)
        self._model.eval()
        self._max_length = max_length

        def _ppl(text: str) -> float:
            ids = self._tok.encode(text, return_tensors='pt',
                                   truncation=True, max_length=self._max_length)
            if ids.shape[1] < 2:
                return _PPL_WORST
            with self._torch.no_grad():
                loss = self._model(ids, labels=ids).loss
            return float(self._torch.exp(loss))

        self._ppl = _ppl

    def perplexity(self, text: str) -> float:
        return self._ppl(text)

    def score(self, text: str) -> float:
        if not text.strip():
            return 0.0
        return perplexity_to_score(self._ppl(text))


def make_scorer(backend: str = 'heuristic', model: Optional[str] = None):
    if backend == 'heuristic':
        return HeuristicScorer()
    if backend == 'perplexity':
        return PerplexityScorer(model=model)
    raise ConfigurationError(f"Unknown quality scorer '{backend}' (heuristic|perplexity).")
