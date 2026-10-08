# Changelog

## 4.0.1 (2026-10-08)

### Fixed

- **Missing optional dependencies are reported before any work, in one line.** Parquet
  or Excel input/output without pyarrow, pandas/openpyxl or xlsxwriter used to fail
  with a Python traceback. Parquet output also failed only after the input had been
  read.
- **Unsupported input and output formats fail with the list of supported ones**
  instead of a traceback.
- **Undecodable input names the fix.** The error now suggests `--encoding`.
- **Other expected failures** (bad settings, I/O errors such as a full disk) print
  one line and exit 1. The traceback is shown only with `--log-level DEBUG`.
- **`--report`, `--stats-file`, `--manifest`, `--dataset-card` and
  `--pseudo-map-file` create missing parent directories**, like the main output.
- **A requested artifact that cannot be written now fails the run (exit status
  1).** Before, the run only logged a warning and exited 0, so CI pipelines could
  miss it.

## 4.0.0 (2026-10-07)

The five phases of the audit plan (`docs/AUDIT_2026-10.md`):
1. data-integrity, privacy and detection-quality fixes;
2. one shared pipeline engine;
3. current detection backends;
4. scale;
5. provenance and release quality.

### Behavior changes: read before upgrading

- **`--sample` and `--split` are content-addressed.** A salted hash of each record
  decides whether it is sampled and which split it lands in (`--seed` sets the
  salt), so results are reproducible across runs, resumes and filter changes, and
  identical records never straddle train/test. The selected records differ from
  3.0's random-stream choice. `SanitizerConfig` now supports `sample` and `seed`.
- **Non-English text is kept by default.** `--min-ascii-ratio` (and
  `SanitizerConfig.min_ascii_ratio`) now defaults to `0` (off) instead of `0.85`.
  Pass `--min-ascii-ratio 0.85` to keep the old English-only filtering. Word-based
  gates count each Han/kana/Thai/Lao/Khmer/Myanmar character as one word.
- **Audit reports no longer contain raw records.** Removed-record samples are
  redacted, and PII diffs mask the original values. `--report-raw-samples` /
  `report_raw_samples=True` restores verbatim samples.
- **Config files are validated.** Unknown keys are an error (with a did-you-mean hint).
  Values are type-checked like the matching CLI flag. Dashed keys (`remove-pii`) are
  accepted.
- **PII detection is stricter about what counts as PII.**
  - Card numbers must pass Luhn and SSNs must be valid.
  - NANP phone numbers need separators, so bare 10-digit runs are left alone.
  - Loopback, link-local and version-string "IPs" are left alone.
  - New kinds: `iban` (`[PII_IBAN]`) and IPv6 addresses.
- **Generic secret assignments keep their variable name:**
  `api_key = "[SECRET_GENERIC]"`. Low-entropy values and placeholders
  (`your-key-here`) are not redacted.
- **`--clean-html` only strips strings that look like HTML.** Inline tags no longer
  insert spaces, block tags become line breaks, and `<script>`/`<style>` bodies are
  dropped.
- **CSV and Parquet keep every column.** Outputs are staged on disk and written at the
  end with the union of all fields. Parquet columns with conflicting types are stored
  as strings.
- **Pseudonymized IPs are valid `10.x.y.z` addresses** (previously `0.0.0.N`).
- **Ctrl-C under `--resume` no longer writes a checkpoint.** The last periodic
  checkpoint is the restart point.
- **`--decontaminate all` means every *ungated* benchmark** (20 of the 22 now in the
  registry). Gated sets (`gpqa`, `hle`) must be named, with `HF_TOKEN` set.
- **`--validate-chat` reads more formats.** Records with ShareGPT `conversations`, content
  parts or `tool_calls` used to fail as `missing_messages` / `bad_message_schema`;
  they are now validated. A `tool` turn that follows a structured tool call must
  answer it (`orphan_tool_result`, `unanswered_tool_call`).
- **`--fuzzy-dedup` uses a new algorithm.** LSH finds candidates with high recall, and
  each candidate is checked against its stored MinHash signature. Rensa computes the
  signatures when installed. The records flagged differ from 3.0's datasketch LSH:
  more true near-duplicates are caught, fewer distinct records are dropped.
  `--dedup-backend sqlite` now also applies to fuzzy dedup and puts its index on disk.
- **`--semantic-dedup` uses a usearch HNSW index when usearch is installed.** It
  matched exact search on 99.9% of decisions, against 98% for the LSH index.
- **`--input` values that are directories or glob patterns are expanded** to the files
  they match. A path that exists is always read as-is.
- **Malformed input lines now count in `malformed`.** This shifts how input positions
  are counted. Finish any `--resume` run that was started with 3.0 *before* upgrading.

### Added

- **Language identification with GlotLID and OpenLID-v2** (`--lang-backend`,
  `--lang-model`; `pip install 'llm-sanitizer-pro[lang]'`). These are the fastText LID
  models behind FineWeb-2, with 2,000+ and ~200 varieties. `auto` prefers GlotLID and
  falls back to langdetect. `--lang-filter` accepts ISO 639-1, 639-3 or
  `code_Script` labels, so `en,zh` matches `eng_Latn` and `cmn_Hani`.
- **Pretraining rule filters** (`--quality-rules gopher,gopher-repetition,c4,fineweb`
  or `all`). These are dependency-free versions of the Gopher, C4 and FineWeb
  heuristics, using datatrove's thresholds and reason names. On 3,000 documents
  they agree with datatrove 0.10.1 on 98–100% of keep/drop decisions. Rejections
  are counted per rule in the summary, stats file and report.
- **Classifier quality scorers.** `--quality-scorer fineweb-edu` uses the FineWeb-Edu
  educational-value classifier. `dclm` uses the DCLM fastText classifier, and
  `fasttext` takes any fastText model with `--quality-model` and `--quality-label`.
- **GLiNER2 PII detection** (`--pii-ner-backend gliner`, `[gliner]` extra). It finds
  person, location, address, date of birth, ID and financial numbers, usernames and
  credentials in many languages. `--pii-ner-threshold` sets the cut-off.
- **PII accuracy harness.** `python -m sanitizer_pro.evaluation` reports span-level
  precision/recall/F1 per kind on a bundled 600-record synthetic set (919 spans, 200
  hard negatives) or your own labeled JSONL. Regex detectors score P = R = 1.0 on it;
  regex + GLiNER2 scores micro F1 0.92 with exact spans and 0.96 with overlap.
- **Decontamination covers current benchmarks.** New: MMLU-Pro, GPQA, MATH, MATH-500,
  AIME 2024/2025, IFEval, BBH, MuSR, HLE, SimpleQA, HumanEval+, MBPP+, GSM-Plus, plus
  the `open-llm-v2` group. Answer and choice fields are indexed alongside questions.
  Hits are attributed to the benchmark (`contaminated_by` in stats and report). The
  n-gram index stores hashes, so all 20 ungated benchmarks fit in ~300 MB.
- **Chat validation for tool-use, multimodal and ShareGPT data.**
  - Checks OpenAI `tool_calls`: a function name and JSON `arguments`.
  - Links tool results to their calls by `tool_call_id`.
  - Accepts content-part lists, including image-only turns.
  - `--format-chatml` converts ShareGPT `conversations`.
  - With an HF `--tokenizer` that has a chat template, `--chat-max-tokens` counts the
    rendered conversation.

- **Faster per-record path.**
  - No flags: 12.6k rec/s, up from 5.2k (100k synthetic records, one core).
  - `--remove-pii --redact-secrets`: 7.1k rec/s, up from 1.7k.
  - Each regex pattern now runs only when the text contains a literal the pattern
    needs, and those literals are found in one pass.
  - Output is byte-identical to the previous version.
- **Scale-out.**
  - `--input` takes directories and globs.
  - With `--jobs N` on JSONL, workers read and parse byte-range chunks themselves.
  - `--jobs 4` runs at 20.6k rec/s, up from 6.3k.
  - Output, stats, report samples and pseudonym maps are identical to `--jobs 1`.
- **Keyed pseudonyms** (`--pseudo-key` / `SANITIZE_PSEUDO_KEY`):
  - The id is HMAC-derived, so it is the same across worker processes and runs:
    `Person_3fa94c07e21b`.
  - Ids are 12 hex digits, not decimal. A long decimal id would contain a digit run
    that later patterns (cards) redact again.
  - `--pii-pseudonymize` now works with `--jobs > 1`.
- **Fuzzy dedup:**
  - `--fuzzy-backend {auto,rensa,datasketch}`, plus a `[fuzzy]` extra for rensa.
  - 6.1k rec/s, up from 455.
  - At thresholds 0.7–0.85: 97–98% recall at or above the threshold, and no false
    positives 0.15 or more below it.
  - The index can live on disk: flat ~140 MB peak memory at 500k records.
  - It survives `--resume` with `--dedup-db-path`.
- **`--semantic-index {auto,usearch,lsh}`**; usearch joins the `[semantic]` extra.
- **`benchmarks/`:**
  - throughput scenarios (rec/s, MB/s, peak RSS, `--jobs` scaling), run nightly in CI
    with regression targets
  - `fuzzy_recall.py`: fuzzy dedup against exact Jaccard
  - `semantic_recall.py`: the semantic index against exact search
- **Run manifest** (`--manifest run.json`):
  - tool, dependency and model versions;
  - the full configuration and its SHA-256;
  - every input and output file (splits and shards included) with size and SHA-256;
  - per-stage counts and timing.
  - The pseudonymization key is recorded only as set or unset.
- **Hugging Face dataset card** (`--dataset-card README.md`), rendered from the
  manifest:
  - front matter: size category, languages and tags;
  - the processing steps with their settings;
  - the record funnel, redaction counts (never values), decontamination results;
  - models, file digests and a reproduction command.
- **`sanitize diff A B`** compares two manifests or stats files: counts, breakdowns,
  configuration, models, dependencies and files. `--fail-on-change` exits 1 on any
  difference, which works as a CI gate.
- **Signed releases.**
  - The release workflow checks that the tag matches the built version.
  - It writes a CycloneDX SBOM.
  - Distributions are signed with Sigstore and get SLSA build provenance (GitHub
    attestations).
  - It publishes to PyPI with Trusted Publishing (PEP 740 attestations) and creates a
    GitHub release with the changelog notes, signatures and SBOM.
- **Documentation site** (MkDocs Material, GitHub Pages; `[docs]` extra):
  - a CLI reference generated from the argument parser (a test keeps it current);
  - pages on provenance, performance and accuracy, and verifying releases.
  - Every CLI option now has help text.

### Changed

- **One engine for the CLI and the library.** `SanitizerConfig` (now in
  `sanitizer_pro.settings`, still importable from `sanitizer_pro`) is the only
  configuration type: it validates every setting, the CLI's flag defaults come from
  it, and a CLI run builds one and drives the same `Sanitizer` the API uses
  (`feed()` / `feed_transformed()` / `finish()`). Worker processes build a
  `RecordTransformer` from the config. `core.sanitize_record` takes a
  `SanitizerConfig` and returns a `Transformed` named tuple.
- **Quality scoring runs in the per-record stage**, so `--jobs` parallelizes it.
- **The version has one source**, `sanitizer_pro.__version__`. `pyproject.toml`
  reads it, and so do the run summary, the stats file and the report. New
  `--version` flag.
- **The package passes `mypy --strict`**, and a CI job enforces it (Python 3.12).

### Fixed

- With `--jobs`, the pool's feeder thread could read the whole input into memory when
  the parent was slower than the workers. It also updated the malformed counters
  concurrently with the main thread.
- With `--jobs`, the audit report had no samples of records dropped by the per-record
  gates and no PII diffs.
- International phone numbers with more than four digit groups
  (`+33 1 42 68 53 07`) were only partly redacted.
- NER detectors could return overlapping spans for the same text, which mangled
  redaction.
- `--resume` after a hard crash (OOM-kill, SIGKILL) duplicated rows written after the
  last checkpoint. Checkpoints now record the durable output size and the SQLite dedup
  high-water mark. On resume the output is truncated and
  the dedup DB rolled back to the checkpoint.
- `--output x.json.gz` was written uncompressed.
- CSV, Parquet and Excel outputs silently dropped fields that first appeared after
  the first record. CSV wrote nested values as Python `repr`.
- Decontamination only scanned the first 8 KB of the `--text-fields` text. It now scans
  every string in the record.
- `--max-tokens` truncation flattened newlines and indentation.
- `HF_TOKEN` was sent to any URL containing `huggingface.co` and forwarded across
  redirects. Hub downloads now retry, stream to disk and are size-checked.
- `SanitizerConfig(decontaminate=['all'])` raised `KeyError`.
- Secrets that were not detected before:
  - Hugging Face, GitLab, npm, PyPI and Azure storage keys
  - AWS secret access keys
  - encrypted, PKCS#8 and PGP private keys, and truncated PEM blocks
- Connection-string matches no longer swallow trailing quotes and punctuation.
- The ruff rule set is pinned so new ruff releases can't break CI.
