# LLM Dataset Sanitizer PRO v3.0

Production-grade, modular dataset sanitization, PII redaction, and curation pipeline for LLM training and fine-tuning.

## 🚀 Features

- **Multi-Format Streaming**: JSON, JSONL, CSV, TXT, Parquet, Excel (with OOM-safe `xlsxwriter` streaming).
- **Advanced PII Redaction**: Regex + `ipaddress` subnet masking, safe `html.parser` HTML stripping, and stable pseudonymization.
- **High-Performance Deduplication**: 
  - Exact SHA-256 dedup (Memory or Batched SQLite).
  - Fuzzy Near-Dedup via MinHash + LSH (`datasketch`).
- **LLM-Native Formatting**: Direct export to ChatML (`--format-chatml`) and Alpaca/Instruct (`--format-instruct`) schemas.
- **Content Filtering**: Heuristic code detection, profanity filtering, and language confidence gating.
- **Crash-Safe I/O**: Atomic JSON writes via `.tmp` + `os.replace()`.

## 📦 Installation

```bash
pip install llm-sanitizer-pro[all]
```

## 🔨 Usage

```bash
# Basic PII redaction + Exact Dedup
sanitize --input data.jsonl --output clean.jsonl --remove-pii --deduplicate

# Fuzzy Dedup + ChatML Formatting
sanitize --input data.jsonl --output chatml.jsonl --fuzzy-dedup --format-chatml

# Parallel Processing + SQLite Backend
sanitize --input huge.jsonl --output clean.jsonl --jobs 8 --dedup-backend sqlite
```
