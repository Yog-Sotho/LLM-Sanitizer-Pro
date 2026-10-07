"""Measure the --semantic-dedup index against exact search.

The question an index answers is "is there an indexed vector with cosine
similarity >= threshold?". This compares each index's answer with brute
force on synthetic unit vectors (the embedding dimension of the default
model2vec model), with queries spread around the threshold.

    python -m benchmarks.semantic_recall --size 20000
    python -m benchmarks.semantic_recall --size 100000 --index usearch
"""
import argparse
import sys
import time
from typing import Dict, Optional, Sequence

import numpy as np

from sanitizer_pro.dedup import SemanticDeduper


def make_data(size: int, queries: int, dim: int = 256, threshold: float = 0.9, seed: int = 0):
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((size, dim)).astype(np.float32)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    # Half the queries are perturbed copies with cosine spread over
    # [threshold - 0.15, 1]; half are unrelated.
    near = base[rng.integers(0, size, queries // 2)]
    target = rng.uniform(threshold - 0.15, 1.0, queries // 2)
    noise = rng.standard_normal(near.shape).astype(np.float32)
    noise -= (noise * near).sum(1, keepdims=True) * near          # orthogonal to near
    noise /= np.linalg.norm(noise, axis=1, keepdims=True)
    q_near = near * target[:, None] + noise * np.sqrt(1 - target ** 2)[:, None]
    q_far = rng.standard_normal((queries - queries // 2, dim)).astype(np.float32)
    q = np.vstack([q_near, q_far]).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return base, q


def evaluate(index: str, size: int, queries: int, threshold: float) -> Dict[str, float]:
    base, q = make_data(size, queries, threshold=threshold)
    truth = (q @ base.T).max(axis=1) >= threshold
    vectors = {f"b{i}": v for i, v in enumerate(base)}
    vectors.update({f"q{i}": v for i, v in enumerate(q)})
    d = SemanticDeduper(threshold=threshold, index=index, _embed_fn=lambda t: vectors[t])
    t0 = time.perf_counter()
    for i in range(size):
        d.add(f"b{i}")
    t_add = time.perf_counter() - t0
    t0 = time.perf_counter()
    got = np.array([d.contains(f"q{i}") for i in range(len(q))])
    t_query = time.perf_counter() - t0
    tp = int((got & truth).sum())
    return {"index": d.index_kind, "size": size,
            "recall": tp / max(1, int(truth.sum())),
            "precision": tp / max(1, int(got.sum())),
            "adds_per_s": round(size / t_add), "queries_per_s": round(len(q) / t_query)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m benchmarks.semantic_recall")
    ap.add_argument("--size", type=int, default=20000)
    ap.add_argument("--queries", type=int, default=2000)
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--index", default="all", help="usearch, lsh, or all")
    a = ap.parse_args(argv)
    for index in (["usearch", "lsh"] if a.index == "all" else [a.index]):
        try:
            r = evaluate(index, a.size, a.queries, a.threshold)
        except ImportError as exc:
            print(f"{index}: skipped ({exc})")
            continue
        print(f"{r['index']:<8} size={r['size']:<8} recall={r['recall']:.3f} "
              f"precision={r['precision']:.3f} adds/s={r['adds_per_s']:,} "
              f"queries/s={r['queries_per_s']:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
