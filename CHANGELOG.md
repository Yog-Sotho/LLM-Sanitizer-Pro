# Changelog

## Unreleased (planned 4.0.0)

Phase 1 of the audit plan (`docs/AUDIT_2026-10.md`): data-integrity, privacy and
detection-quality fixes.

### Behavior changes: read before upgrading

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
- **Malformed input lines now count in `malformed`.** This shifts how input positions
  are counted. Finish any `--resume` run that was started with 3.0 *before* upgrading.

### Fixed

- `--resume` after a hard crash (OOM-kill, SIGKILL) duplicated rows written after the
  last checkpoint. Checkpoints now record the durable output size and the SQLite dedup
  high-water mark, plus the sampling RNG state. On resume the output is truncated and
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
