# Performance

Measured with `benchmarks/bench.py` on 100k synthetic instruction records (~525 bytes each, `benchmarks/corpus.py`), one x86 core unless noted.

| Scenario | rec/s | Peak RSS |
|---|---:|---:|
| no flags | 12.6k | 110 MB |
| `--remove-pii --redact-secrets` | 7.1k | 110 MB |
| … `--deduplicate` | 6.1k | 125 MB |
| … `--jobs 4` | 20.6k | 185 MB |
| `--fuzzy-dedup` (rensa) | 6.1k | 257 MB |
| `--fuzzy-dedup --dedup-backend sqlite` | 3.7k | 136 MB |
| `--fuzzy-dedup --fuzzy-backend datasketch` | 2.1k | 293 MB |

- **`--jobs`:** workers run the per-record stage, and read and parse the input themselves for plain JSONL. The parent process keeps the stateful stages (dedup, stats, writing), which caps throughput at about 20k rec/s for redaction + dedup.
- **Memory at scale:** at 500k records the in-memory fuzzy index peaks at 1.1 GB. The SQLite index stays at about 140 MB.

## Accuracy

| Component | Harness | Result |
|---|---|---|
| Regex PII | `python -m sanitizer_pro.evaluation` | P = R = 1.0 on the bundled 600-record set |
| Regex + GLiNER2 | `… --backend regex+gliner` | micro F1 0.92 (exact spans), 0.96 (overlap) |
| Fuzzy dedup | `python -m benchmarks.fuzzy_recall` | 97–98% recall at or above the threshold (t = 0.7–0.85); no pair 0.15 or more below it is flagged |
| Semantic index | `python -m benchmarks.semantic_recall` | usearch matched exact search on 99.9% of decisions on 20k real embeddings |
| Rule filters | parity with datatrove 0.10.1 | 98–100% identical keep/drop decisions on 3,000 docs |

## Running the benchmarks

```bash
python -m benchmarks.bench --records 100000              # all scenarios
python -m benchmarks.bench --scenarios regex,jobs --check  # fail on regressions
```

The nightly CI job runs `--check` against regression floors set from these measurements.
