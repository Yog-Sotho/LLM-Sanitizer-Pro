"""PII detection accuracy: span-level precision / recall / F1 per kind.

    python -m sanitizer_pro.evaluation --backend regex
    python -m sanitizer_pro.evaluation --backend regex+gliner --entities all
    python -m sanitizer_pro.evaluation --backend spacy --data my_labeled.jsonl --json

Data: JSONL records ``{"text": ..., "spans": [[start, end, kind], ...]}``;
the default is the bundled synthetic set (sanitizer_pro/data/pii_eval.jsonl,
built by scripts/make_pii_eval.py). Kinds use the redactor's names (email,
url, phone, card, iban, ssn, ip, person, location, address, ...).

Matching is exact (same start, end and kind) by default; ``--match overlap``
counts any same-kind overlap, which is fairer to NER boundary differences.
Only kinds a backend can detect contribute to its micro average.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, NamedTuple, Optional, Sequence, Set, Tuple

from sanitizer_pro.pii import find_pii_spans

Span = Tuple[int, int, str]
Detector = Callable[[str], List[Span]]
DEFAULT_DATA = Path(__file__).resolve().parent / 'data' / 'pii_eval.jsonl'
REGEX_KINDS = frozenset({'email', 'url', 'phone', 'card', 'iban', 'ssn', 'ip'})


class EvalRecord(NamedTuple):
    text: str
    spans: List[Span]


def load_eval_set(path: Optional[str] = None) -> List[EvalRecord]:
    records = []
    with open(path or DEFAULT_DATA, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                raw = json.loads(line)
                spans = [(int(a), int(b), str(k)) for a, b, k in raw.get('spans', [])]
                records.append(EvalRecord(str(raw['text']), spans))
    return records


def regex_detector() -> Detector:
    return find_pii_spans


def ner_detector(backend: str, entities: Sequence[str] = ('all',), model: Optional[str] = None,
                 threshold: float = 0.5) -> Tuple[Detector, Set[str]]:
    from sanitizer_pro.ner import NERRedactor
    ner = NERRedactor(backend=backend, entities=entities, model=model, threshold=threshold)

    def detect(text: str) -> List[Span]:
        return [(s.start, s.end, s.kind) for s in ner.detect(text)]

    return detect, set(ner.entities)


def combine(detectors: Iterable[Detector]) -> Detector:
    dets = list(detectors)

    def detect(text: str) -> List[Span]:
        return sorted({span for d in dets for span in d(text)})

    return detect


def _matches(pred: Span, gold: Span, mode: str) -> bool:
    if pred[2] != gold[2]:
        return False
    if mode == 'exact':
        return pred[0] == gold[0] and pred[1] == gold[1]
    return pred[0] < gold[1] and gold[0] < pred[1]


def evaluate(detector: Detector, records: Iterable[EvalRecord], kinds: Set[str],
             match: str = 'exact') -> Dict[str, Dict[str, float]]:
    """Per-kind and micro-averaged precision/recall/F1 over `kinds`."""
    counts: Dict[str, List[int]] = {k: [0, 0, 0] for k in sorted(kinds)}  # tp, fp, fn
    for rec in records:
        gold = [s for s in rec.spans if s[2] in kinds]
        pred = [p for p in detector(rec.text) if p[2] in kinds]
        unmatched = list(gold)
        for p in pred:
            hit = next((g for g in unmatched if _matches(p, g, match)), None)
            if hit is not None:
                unmatched.remove(hit)
                counts[p[2]][0] += 1
            else:
                counts[p[2]][1] += 1
        for g in unmatched:
            counts[str(g[2])][2] += 1
    out: Dict[str, Dict[str, float]] = {}
    total = [0, 0, 0]
    for kind, (tp, fp, fn) in counts.items():
        out[kind] = _prf(tp, fp, fn)
        total = [total[0] + tp, total[1] + fp, total[2] + fn]
    out['micro'] = _prf(*total)
    return out


def _prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)
    r = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {'precision': round(p, 4), 'recall': round(r, 4), 'f1': round(f1, 4),
            'tp': tp, 'fp': fp, 'fn': fn}


def format_table(results: Dict[str, Dict[str, float]]) -> str:
    lines = [f"{'kind':<12}{'precision':>10}{'recall':>9}{'f1':>8}{'tp':>6}{'fp':>5}{'fn':>5}"]
    for kind, m in results.items():
        if kind == 'micro':
            lines.append('-' * len(lines[0]))
        lines.append(f"{kind:<12}{m['precision']:>10.3f}{m['recall']:>9.3f}{m['f1']:>8.3f}"
                     f"{m['tp']:>6}{m['fp']:>5}{m['fn']:>5}")
    return '\n'.join(lines)


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(prog='python -m sanitizer_pro.evaluation',
                                 description='Measure PII detection precision/recall/F1.')
    ap.add_argument('--backend', default='regex',
                    help="regex, spacy, transformers, gliner, or a '+'-joined combination "
                         "(e.g. regex+gliner).")
    ap.add_argument('--entities', default='all', help='NER entity kinds (default: all).')
    ap.add_argument('--model', default=None, help='NER model override.')
    ap.add_argument('--threshold', type=float, default=0.5, help='GLiNER threshold.')
    ap.add_argument('--data', default=None, help='Labeled JSONL (default: bundled set).')
    ap.add_argument('--match', choices=['exact', 'overlap'], default='exact')
    ap.add_argument('--json', action='store_true', help='Print JSON instead of a table.')
    args = ap.parse_args(argv)

    detectors: List[Detector] = []
    kinds: Set[str] = set()
    for part in args.backend.split('+'):
        if part == 'regex':
            detectors.append(regex_detector())
            kinds |= REGEX_KINDS
        else:
            det, ner_kinds = ner_detector(part, args.entities.split(','), args.model,
                                          args.threshold)
            detectors.append(det)
            kinds |= ner_kinds
    records = load_eval_set(args.data)
    present = {s[2] for r in records for s in r.spans}
    results = evaluate(combine(detectors), records, kinds & present, args.match)
    if args.json:
        print(json.dumps({'backend': args.backend, 'match': args.match, 'results': results},
                         indent=2))
    else:
        print(f"backend={args.backend} match={args.match} records={len(records)}")
        print(format_table(results))


if __name__ == '__main__':
    main(sys.argv[1:])
