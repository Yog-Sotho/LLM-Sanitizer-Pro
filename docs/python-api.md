# Python API

The CLI is a thin layer over a first-class library — services should embed it
directly instead of shelling out:

```python
from sanitizer_pro import Sanitizer, SanitizerConfig

config = SanitizerConfig(
    remove_pii=True, deduplicate=True,
    quality_min_score=0.5, quality_score_field="_q",
    decontaminate=["gsm8k", "mmlu"], validate_chat=False,
)

with Sanitizer(config) as s:
    clean = list(s.process(records))     # iterable of dicts in → clean dicts out
    report = s.stats.to_dict()           # same schema as --stats-file

    # Or inspect records one at a time (audit-UI building block):
    result = s.process_record({"text": "..."})
    result.kept        # bool
    result.reason      # 'quality' | 'duplicate' | 'contaminated' | 'chat:<detail>' | ...
    result.score       # quality score in [0, 1] when scoring is enabled
```

`SanitizerConfig` is the single configuration object: the CLI flags have the
same names, take their defaults from it, and a CLI run builds one and drives the
very same `Sanitizer` engine — so library and command-line results are identical.
Dedup and pseudonym state persist across calls on one `Sanitizer` instance.
For streaming, `s.feed(record)` returns the records to write now and
`s.finish()` releases the `keep_top_percent` buffer at the end (`process()`
combines both). Export pseudonym mappings with `s.export_pseudonym_map(path)`.

## Development

```bash
pip install -e .[dev]
pytest            # run the test suite
ruff check sanitizer_pro tests benchmarks scripts
mypy              # strict
pip install -e .[docs] && mkdocs serve           # docs site at localhost:8000
python scripts/gen_cli_reference.py              # after changing CLI options

# Throughput (rec/s, MB/s, peak RSS per scenario, --jobs scaling); nightly in CI
python -m benchmarks.bench --records 100000
python -m benchmarks.fuzzy_recall --threshold 0.85     # fuzzy dedup vs exact Jaccard
python -m benchmarks.semantic_recall --size 100000     # semantic index vs exact search
```
