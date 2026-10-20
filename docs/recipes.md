# Recipes

Copy-paste commands for common jobs. Run `sanitize --help` for every option, or
`sanitize --generate-config yaml` to print a config template usable with `--config`.

```bash
# Basic PII redaction + exact dedup
sanitize --input data.jsonl --output clean.jsonl --remove-pii --deduplicate

# Fuzzy dedup + ChatML formatting
sanitize --input data.jsonl --output chatml.jsonl --fuzzy-dedup --fuzzy-threshold 0.7 --format-chatml

# Parallel processing + SQLite dedup backend for datasets bigger than RAM
sanitize --input huge.jsonl --output clean.jsonl --jobs 8 --deduplicate --dedup-backend sqlite

# A whole directory of shards, near-dedup on disk, 8 workers
sanitize --input 'shards/**/*.jsonl' --output clean.jsonl --jobs 8 \
    --fuzzy-dedup --dedup-backend sqlite --dedup-db-path minhash.db

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

# NER-backed PII: redact person names too (install: see Features → Installing NER-backed PII detection)
sanitize --input data.jsonl --output clean.jsonl --remove-pii --pii-ner

# Pseudonymize everything consistently (names, emails, …) and keep the mapping.
# With a key, pseudonyms are the same across runs and --jobs workers.
SANITIZE_PSEUDO_KEY=… sanitize --input data.jsonl --output clean.jsonl --remove-pii --pii-ner \
    --pii-pseudonymize --pseudo-map-file mapping.json --jobs 4

# Convert instruction data to ChatML, then reject structurally invalid conversations
sanitize --input data.jsonl --output chat.jsonl --format-chatml --validate-chat

# Enforce a context-window budget per conversation, counted through the
# model's chat template
sanitize --input chat.jsonl --output fit.jsonl --validate-chat --chat-max-tokens 4096 \
    --tokenizer Qwen/Qwen3-8B

# Tool-calling data (OpenAI tool_calls / tool results, or ShareGPT
# function_call / observation turns)
sanitize --input tools.jsonl --output clean.jsonl --validate-chat \
    --chat-roles system,user,assistant,tool

# Multi-agent traces: keep structural and tool-call checks, relax ordering rules
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

# One-flag preset for SFT data (PII + secrets + dedup + cleanup), report attached
sanitize --input data.jsonl --output clean.jsonl --profile fine-tune --report audit.html

# Redact leaked credentials from scraped code/logs
sanitize --input data.jsonl --output clean.jsonl --redact-secrets

# Structured JSON logs for a service pipeline
sanitize --input data.jsonl --output clean.jsonl --log-format json
```
