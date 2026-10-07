# Provenance

Every run can leave a record of exactly how its output was produced.

```bash
sanitize --input raw/ --output clean.jsonl --remove-pii --deduplicate \
    --manifest run.json --dataset-card README.md --stats-file stats.json
```

## Run manifest (`--manifest`)

A JSON document (`schema: llm-sanitizer-pro/run-manifest`, `schema_version: 1`) with:

| Section | Contents |
|---|---|
| `tool` | package version, Python version, platform |
| `run` | start and end time (UTC), duration, `--jobs`, the command line |
| `config`, `config_sha256` | every setting, and a hash of them, so two runs with the same hash were configured identically |
| `models` | the models and reference data each stage loaded: language ID, quality scorer, NER, semantic dedup, tokenizer, decontamination benchmarks (with their Hub repos) and reference files |
| `dependencies` | installed versions of the optional packages that affect results |
| `inputs`, `outputs` | every file with its size and SHA-256 (`--manifest-no-digest` keeps sizes only); split and shard outputs are listed individually |
| `counts` | the per-stage counts, as in `--stats-file` |

The pseudonymization key is recorded only as set or unset. Its command-line value appears as `***`. The hash covers whether a key is set, not the key itself.

## Dataset card (`--dataset-card`)

A Hugging Face dataset card (`README.md`) rendered from the manifest:

- **Front matter:** size category, detected languages, tags.
- **Processing steps:** each step with its settings, in order.
- **Record funnel:** how many records each stage removed.
- **Counts:** redactions by kind (counts only, never values), decontamination by benchmark, and languages.
- **Models and files:** the models used, and file digests.
- **Reproduction:** the command that reproduces the run.

The sections only the dataset owner can write are left as `<!-- TODO -->` placeholders: summary, source, license and intended use.

## Comparing runs (`sanitize diff`)

```bash
sanitize diff old/run.json new/run.json
sanitize diff baseline.stats.json candidate.stats.json --json
sanitize diff baseline.json candidate.json --fail-on-change   # exit 1 on any difference
```

Accepts manifests or stats files. It shows:

- changed counts (absolute and relative), and per-key changes in breakdowns: PII kinds, rule failures, contamination by benchmark, histograms
- for manifests, also changes in configuration, models, dependency versions, and input/output files (compared by SHA-256)
