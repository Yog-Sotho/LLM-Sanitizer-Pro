# LLM Dataset Sanitizer PRO v3.0

Production-grade, modular dataset sanitization, PII redaction, and curation pipeline for LLM training and fine-tuning.

## 🚀 Features

- **Multi-Format Streaming**: JSONL, JSON (ijson streaming), CSV/TSV, TXT, Parquet, Excel, and gzip variants — from files or stdin/stdout.
- **Advanced PII Redaction**: Email, URL, phone, credit card, SSN, and IP detection with three modes: token replacement, partial masking (`--pii-mask`), and stable pseudonymization (`--pii-pseudonymize` + exportable mapping).
- **High-Performance Deduplication**:
  - Exact SHA-256 dedup (in-memory or disk-backed SQLite for huge datasets).
  - Fuzzy near-dedup via MinHash + LSH (`--fuzzy-dedup`, tunable `--fuzzy-threshold`).
- **LLM-Native Formatting**: Direct export to ChatML (`--format-chatml`) and Alpaca/Instruct (`--format-instruct`) schemas, with automatic key mapping (`prompt`/`question`/`response`/`completion`/…).
- **Quality & Content Filtering**: Length/word/uniqueness/ASCII gates, all-caps rejection, code detection, profanity filtering, language filtering with confidence gating, and pluggable Python quality scripts.
- **Dataset Splitting & Sharding**: `--split train=0.9,val=0.05,test=0.05` or fixed-size shards with `--shard-size`.
- **Crash-Safe I/O**: Atomic JSON writes (`.tmp` + `os.replace()`), safe HTML stripping via `html.parser`, structure-preserving text normalization (newlines kept for code/markdown data).
- **Parallel Processing**: `--jobs N` multiprocessing with accurate statistics.

## 📦 Installation

```bash
pip install llm-sanitizer-pro          # core (stdlib + tqdm)
pip install "llm-sanitizer-pro[all]"   # all format & feature extras
```

## 🔨 Usage

```bash
# Basic PII redaction + exact dedup
sanitize --input data.jsonl --output clean.jsonl --remove-pii --deduplicate

# Fuzzy dedup + ChatML formatting
sanitize --input data.jsonl --output chatml.jsonl --fuzzy-dedup --fuzzy-threshold 0.7 --format-chatml

# Parallel processing + SQLite dedup backend for datasets bigger than RAM
sanitize --input huge.jsonl --output clean.jsonl --jobs 8 --deduplicate --dedup-backend sqlite

# Train/val/test split with reproducible sampling
sanitize --input data.jsonl --output out.jsonl --split train=0.9,val=0.05,test=0.05 --seed 42

# Preview effects without writing output
sanitize --input data.jsonl --output out.jsonl --dry-run --stats-file report.json
```

Run `sanitize --help` for the full option reference, or `sanitize --generate-config yaml` to print a config template usable with `--config`.

## 🧪 Development

```bash
pip install -e .[dev]
pytest            # run the test suite
ruff check sanitizer_pro tests
```

## License

MIT
