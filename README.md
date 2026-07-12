# LLM Dataset Sanitizer PRO v3.0

Production-grade, modular dataset sanitization, PII redaction, and curation pipeline for LLM training and fine-tuning.

## 🚀 Features

- **Multi-Format Streaming**: JSONL, JSON (ijson streaming), CSV/TSV, TXT, Parquet, Excel, and gzip variants — from files or stdin/stdout.
- **Hugging Face Hub Input** (`--input hf://owner/dataset[/config[/split]]`): sanitize a Hub dataset directly — shards are fetched via the Hub's parquet API (only `pyarrow` needed, no `datasets` library), cached locally, and streamed through the pipeline. Set `HF_TOKEN` for private/gated datasets.
- **Advanced PII Redaction**: Email, URL, phone, credit card, SSN, and IP detection with three modes: token replacement, partial masking (`--pii-mask`), and stable pseudonymization (`--pii-pseudonymize` + exportable mapping).
- **NER-Backed PII Detection** (`--pii-ner`): person names, locations, and organizations detected with a named-entity model (spaCy or transformers) — the PII that regexes fundamentally cannot catch. All three redaction modes apply ("Sarah Connor" → `[PII_PERSON]`, `S*** C***`, or a stable `Person_0001`).
- **High-Performance Deduplication** (three tiers):
  - Exact SHA-256 dedup (in-memory or disk-backed SQLite for huge datasets).
  - Fuzzy near-dedup via MinHash + LSH (`--fuzzy-dedup`, tunable `--fuzzy-threshold`).
  - Semantic near-dedup (`--semantic-dedup`): static embeddings (model2vec, ~30MB, no torch) + hyperplane LSH catch paraphrases that share no n-grams, verified with exact cosine similarity (`--semantic-threshold`).
- **LLM-Native Formatting**: Direct export to ChatML (`--format-chatml`) and Alpaca/Instruct (`--format-instruct`) schemas, with automatic key mapping (`prompt`/`question`/`response`/`completion`/…).
- **Chat Dataset Validation** (`--validate-chat`): lint `messages`-format records before they reach a trainer — role alternation, empty turns, missing assistant replies, multiple/misplaced system messages, unknown roles, and per-conversation token budgets (`--chat-max-tokens`), with a per-reason rejection breakdown in the report and stats file.
- **Quality & Content Filtering**: Length/word/uniqueness/ASCII gates, all-caps rejection, code detection, profanity filtering, language filtering with confidence gating, and pluggable Python quality scripts.
- **Quality Scoring** (`--quality-min-score`, `--keep-top-percent`, `--quality-score-field`): every record gets a [0, 1] quality score — a dependency-free heuristic (C4/Gopher-style prose signals with a multiplicative repetition penalty) or causal-LM perplexity (`--quality-scorer perplexity`). Filter by absolute bar, keep only the best P%, or just annotate records for downstream sorting; score histogram and mean land in the stats file.
- **Benchmark Decontamination**: n-gram overlap removal against eval test sets (`--decontaminate mmlu,gsm8k,humaneval,arc,hellaswag,truthfulqa,winogrande,mbpp`) — benchmarks are auto-downloaded from the Hugging Face Hub and cached, or supply your own reference files with `--decontam-refs`.
- **Dataset Splitting & Sharding**: `--split train=0.9,val=0.05,test=0.05` or fixed-size shards with `--shard-size`.
- **Crash-Safe I/O**: Atomic JSON writes (`.tmp` + `os.replace()`), safe HTML stripping via `html.parser`, structure-preserving text normalization (newlines kept for code/markdown data).
- **Resumable Runs** (`--resume`): progress is checkpointed to `<output>.checkpoint.json` every `--checkpoint-interval` records; after a crash or Ctrl-C, rerun the same command and the pipeline skips already-processed input, restores statistics and pseudonym state, and appends to the output. Pair with `--dedup-backend sqlite --dedup-db-path` for dedup state that also survives the restart.
- **Parallel Processing**: `--jobs N` multiprocessing with accurate statistics.
- **Audit Report** (`--report audit.html`): a self-contained HTML artifact per run — removal funnel, PII redaction counts by type, quality-score distribution, chat-failure breakdown, and before/after redaction samples. Light/dark aware, no external assets; archive it next to the dataset or attach it to a compliance ticket. Also available from the Python API via `s.write_report(path)`.

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

# Remove records that overlap with benchmark test sets (auto-downloads + caches them)
sanitize --input data.jsonl --output clean.jsonl --decontaminate gsm8k,mmlu,humaneval

# Decontaminate against your own held-out eval set
sanitize --input data.jsonl --output clean.jsonl --decontam-refs my_eval_set.jsonl

# List available built-in benchmarks
sanitize --decontaminate list

# NER-backed PII: redact person names too (see install note below)
sanitize --input data.jsonl --output clean.jsonl --remove-pii --pii-ner

# Pseudonymize everything consistently (names, emails, …) and keep the mapping
sanitize --input data.jsonl --output clean.jsonl --remove-pii --pii-ner \
    --pii-pseudonymize --pseudo-map-file mapping.json

# Convert instruction data to ChatML, then reject structurally invalid conversations
sanitize --input data.jsonl --output chat.jsonl --format-chatml --validate-chat

# Enforce a context-window budget per conversation (tokens counted with --tokenizer)
sanitize --input chat.jsonl --output fit.jsonl --validate-chat --chat-max-tokens 4096

# Multi-agent / tool traces: keep structural checks, relax ordering rules
sanitize --input traces.jsonl --output clean.jsonl --validate-chat --chat-lenient \
    --chat-roles system,user,assistant,tool

# Quality scoring: drop junk below an absolute bar, or keep only the best 30%
sanitize --input data.jsonl --output clean.jsonl --quality-min-score 0.5
sanitize --input data.jsonl --output best.jsonl --keep-top-percent 30

# Annotate records with their score instead of filtering (sort downstream)
sanitize --input data.jsonl --output scored.jsonl --quality-score-field quality

# Perplexity-based scoring with a causal LM (pip install transformers torch)
sanitize --input data.jsonl --output clean.jsonl --quality-scorer perplexity \
    --quality-model distilgpt2 --quality-min-score 0.4

# Full pipeline with an HTML audit report artifact
sanitize --input data.jsonl --output clean.jsonl --remove-pii --deduplicate \
    --quality-min-score 0.5 --report audit.html

# Sanitize a Hugging Face Hub dataset directly (cached under ~/.cache)
sanitize --input hf://openai/gsm8k/main/train --output clean.jsonl \
    --remove-pii --deduplicate --report audit.html

# Long job that survives crashes: checkpoint + durable dedup, rerun to continue
sanitize --input huge.jsonl --output clean.jsonl --resume \
    --deduplicate --dedup-backend sqlite --dedup-db-path dedup.db

# Semantic dedup: drop paraphrased near-duplicates (pip install model2vec)
sanitize --input data.jsonl --output clean.jsonl --semantic-dedup --semantic-threshold 0.85
```

### NER-backed PII install

`--pii-ner` auto-selects an installed backend (spaCy preferred, transformers fallback):

```bash
pip install "llm-sanitizer-pro[ner]"
pip install "en_core_web_sm @ https://huggingface.co/spacy/en_core_web_sm/resolve/main/en_core_web_sm-any-py3-none-any.whl"
```

Use `--pii-ner-entities person,location,org` to widen coverage (default: `person`) and
`--pii-ner-model` to swap in another spaCy model or HF token-classification model.

### How decontamination works

A record is flagged as contaminated when at least `--decontam-min-hits` (default 1) of its
normalized word 8-grams (`--decontam-ngram`) also appear in the reference index built from
benchmark test sets — the same n-gram collision approach used to decontaminate GPT-3 and
Llama training data. Benchmark items shorter than the n-gram size are matched whole.
Named benchmarks require `pyarrow`; `--decontam-refs` files work with any supported input
format and no extra dependencies.

Run `sanitize --help` for the full option reference, or `sanitize --generate-config yaml` to print a config template usable with `--config`.

## 🐍 Python API

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

`SanitizerConfig` mirrors the CLI flags with the same names and defaults.
Dedup and pseudonym state persist across calls on one `Sanitizer` instance;
`keep_top_percent` applies in `process()` (it needs the whole stream). Export
pseudonym mappings with `s.export_pseudonym_map(path)`.

## 🧪 Development

```bash
pip install -e .[dev]
pytest            # run the test suite
ruff check sanitizer_pro tests
```
---
## License

MIT

## Author

Yogsotho
