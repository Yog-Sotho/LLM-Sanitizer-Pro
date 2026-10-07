"""Measure --fuzzy-dedup accuracy against exact Jaccard similarity.

Builds random documents and edited variants, computes the exact Jaccard
similarity of their word 3-shingle sets, indexes the originals and asks
whether each variant is flagged as a near-duplicate. Reports the detection
rate per similarity bin: recall above the threshold, false positives below.

    python -m benchmarks.fuzzy_recall --threshold 0.85
    python -m benchmarks.fuzzy_recall --backend datasketch --store sqlite
"""
import argparse
import random
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

from sanitizer_pro.dedup import MinHashDeduper, _shingles

VOCAB = [f"w{i}" for i in range(5000)]


def make_pairs(n: int, seed: int = 0) -> List[Tuple[str, str, float]]:
    """(original, variant, exact Jaccard) with similarities spread over 0.4-1."""
    rng = random.Random(seed)
    pairs = []
    for _ in range(n):
        words = [rng.choice(VOCAB) for _ in range(rng.randint(80, 400))]
        variant = list(words)
        # Each substitution breaks up to 3 shingles; spread edits over 0-12%.
        for _ in range(int(len(words) * rng.uniform(0, 0.12))):
            variant[rng.randrange(len(variant))] = rng.choice(VOCAB)
        a, b = set(_shingles(" ".join(words), 3)), set(_shingles(" ".join(variant), 3))
        pairs.append((" ".join(words), " ".join(variant), len(a & b) / len(a | b)))
    return pairs


def evaluate(pairs: Sequence[Tuple[str, str, float]], threshold: float, backend: str,
             store: str, db_path: Optional[str] = None) -> Dict[str, object]:
    d = MinHashDeduper(threshold=threshold, backend=backend, store=store, db_path=db_path)
    t0 = time.perf_counter()
    for original, _, _ in pairs:
        d.add(original)
    flagged = [d.contains(variant) for _, variant, _ in pairs]
    elapsed = time.perf_counter() - t0
    bins: Dict[str, List[int]] = {}
    above = [f for f, (_, _, j) in zip(flagged, pairs) if j >= threshold]
    far_below = [f for f, (_, _, j) in zip(flagged, pairs) if j < threshold - 0.15]
    for f, (_, _, j) in zip(flagged, pairs):
        lo = min(int(j * 20) / 20, 0.95)
        bins.setdefault(f"{lo:.2f}-{lo + 0.05:.2f}", [0, 0])
        bins[f"{lo:.2f}-{lo + 0.05:.2f}"][0] += f
        bins[f"{lo:.2f}-{lo + 0.05:.2f}"][1] += 1
    d.close()
    return {
        "backend": d.backend, "bands": f"{d.bands}x{d.rows}",
        "recall": sum(above) / len(above) if above else float("nan"),
        "fp_rate_far_below": sum(far_below) / len(far_below) if far_below else float("nan"),
        "docs_per_s": round(2 * len(pairs) / elapsed),
        "bins": {k: round(v[0] / v[1], 3) for k, v in sorted(bins.items())},
        "n_above": len(above),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m benchmarks.fuzzy_recall")
    ap.add_argument("--pairs", type=int, default=4000)
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--store", default="memory", choices=["memory", "sqlite"])
    ap.add_argument("--min-recall", type=float, default=None,
                    help="Exit non-zero when recall at >= threshold is lower.")
    a = ap.parse_args(argv)
    r = evaluate(make_pairs(a.pairs), a.threshold, a.backend, a.store)
    print(f"backend={r['backend']} bands={r['bands']} threshold={a.threshold} "
          f"store={a.store} docs/s={r['docs_per_s']}")
    print(f"recall (J >= {a.threshold}, n={r['n_above']}): {r['recall']:.3f}   "
          f"flagged at J < {a.threshold - 0.15:.2f}: {r['fp_rate_far_below']:.3f}")
    for k, v in r["bins"].items():  # type: ignore[attr-defined]
        print(f"  J {k}: {v:.3f} flagged")
    if a.min_recall is not None and r["recall"] < a.min_recall:  # type: ignore[operator]
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
