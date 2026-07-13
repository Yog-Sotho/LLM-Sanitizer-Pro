"""Run statistics shared by the CLI and the Python API."""
from typing import Any, Dict, Optional

_CHAR_BUCKETS = [0, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
_WORD_BUCKETS = [0, 5, 10, 25, 50, 100, 250, 500, 1000, 2500]


class RunStats:
    """Tracks processing metrics and histograms."""

    def __init__(self) -> None:
        self.total = 0
        self.kept = 0
        self.filtered_quality = 0
        self.filtered_lang = 0
        self.filtered_require = 0
        self.filtered_code = 0
        self.filtered_profanity = 0
        self.filtered_contaminated = 0
        self.filtered_chat = 0
        self.chat_invalid_reasons: Dict[str, int] = {}
        self.filtered_low_score = 0
        self.score_hist: Dict[int, int] = {}
        self.score_sum = 0.0
        self.score_count = 0
        self.deduplicated = 0
        self.malformed = 0
        self.sampled_out = 0
        self.char_hist: Dict[int, int] = {b: 0 for b in _CHAR_BUCKETS}
        self.word_hist: Dict[int, int] = {b: 0 for b in _WORD_BUCKETS}
        self.lang_dist: Dict[str, int] = {}
        self.pii_counts: Dict[str, int] = {}

    def merge_pii_counts(self, counts: Optional[Dict[str, int]]) -> None:
        if counts:
            for kind, n in counts.items():
                self.pii_counts[kind] = self.pii_counts.get(kind, 0) + n

    def record_kept(self, text: str, lang: Optional[str] = None) -> None:
        self.kept += 1
        n = len(text)
        bucket = next((b for b in reversed(_CHAR_BUCKETS) if n >= b), _CHAR_BUCKETS[0])
        self.char_hist[bucket] += 1
        w = len(text.split())
        wbucket = next((b for b in reversed(_WORD_BUCKETS) if w >= b), _WORD_BUCKETS[0])
        self.word_hist[wbucket] += 1
        if lang:
            self.lang_dist[lang] = self.lang_dist.get(lang, 0) + 1

    def record_score(self, score: float) -> None:
        bucket = min(9, int(score * 10))
        self.score_hist[bucket] = self.score_hist.get(bucket, 0) + 1
        self.score_sum += score
        self.score_count += 1

    def to_state(self) -> Dict[str, Any]:
        """Serializable snapshot for checkpointing (exact attribute round-trip)."""
        return {k: v for k, v in vars(self).items()}

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> 'RunStats':
        rs = cls()
        for k, v in state.items():
            if not hasattr(rs, k):
                continue
            # JSON turns int histogram keys into strings; convert them back.
            if k in ('char_hist', 'word_hist', 'score_hist') and isinstance(v, dict):
                v = {int(bk): bv for bk, bv in v.items()}
            setattr(rs, k, v)
        return rs

    def to_dict(self) -> Dict[str, Any]:
        total = self.total or 1
        return {
            'total': self.total,
            'kept': self.kept,
            'kept_pct': round(self.kept / total * 100, 4),
            'filtered_quality': self.filtered_quality,
            'filtered_language': self.filtered_lang,
            'filtered_require': self.filtered_require,
            'filtered_code': self.filtered_code,
            'filtered_profanity': self.filtered_profanity,
            'filtered_contaminated': self.filtered_contaminated,
            'filtered_chat_invalid': self.filtered_chat,
            'chat_invalid_reasons': dict(sorted(self.chat_invalid_reasons.items(),
                                                key=lambda x: -x[1])),
            'filtered_low_score': self.filtered_low_score,
            'quality_score_mean': (round(self.score_sum / self.score_count, 4)
                                   if self.score_count else None),
            'quality_score_histogram': {f"{b / 10:.1f}": self.score_hist[b]
                                        for b in sorted(self.score_hist)},
            'deduplicated': self.deduplicated,
            'malformed': self.malformed,
            'sampled_out': self.sampled_out,
            'char_length_histogram': {f">={k}": v for k, v in sorted(self.char_hist.items())},
            'word_count_histogram': {f">={k}": v for k, v in sorted(self.word_hist.items())},
            'language_distribution': dict(sorted(self.lang_dist.items(), key=lambda x: -x[1])),
            'pii_redactions': dict(sorted(self.pii_counts.items(), key=lambda x: -x[1])),
        }
