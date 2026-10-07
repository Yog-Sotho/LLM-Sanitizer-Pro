"""Command Line Interface and main orchestration loop for LLM Dataset Sanitizer PRO."""
import argparse
import contextlib
import json
import logging
import multiprocessing
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

try:
    from tqdm import tqdm as _tqdm
    TQDM_AVAILABLE = True
except ImportError:
    _tqdm = None
    TQDM_AVAILABLE = False

try:
    import openpyxl as _openpyxl
    OPENPYXL_AVAILABLE = True
except ImportError:
    _openpyxl = None
    OPENPYXL_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False

from sanitizer_pro import __version__
from sanitizer_pro.api import Sanitizer
from sanitizer_pro.config import apply_config_to_args, collect_explicit_args, load_config_file
from sanitizer_pro.io.readers import read_records
from sanitizer_pro.langid import LANG_BACKENDS
from sanitizer_pro.io.writers import ShardedWriter, SplitWriter, StreamingWriter, parse_split_spec
from sanitizer_pro.pii import PseudoRegistry
from sanitizer_pro.settings import DEFAULTS as D
from sanitizer_pro.settings import (
    DEDUP_BACKENDS, NER_BACKENDS, QUALITY_SCORERS, SanitizerConfig, as_list,
)
from sanitizer_pro.stats import RunStats
from sanitizer_pro.utils import _EXCEL_WARN_MB_DEFAULT, _STDIN, _STDOUT, ConfigurationError, resolve_fmt
from sanitizer_pro.worker import _worker_fn, _worker_init

# =============================================================================
# Constants
# =============================================================================

BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ██████╗  █████╗ ████████╗ █████╗ ███████╗███████╗████████╗║
║   ██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗██╔════╝██╔════╝╚══██╔══╝║
║   ██║  ██║███████║   ██║   ███████║███████╗█████╗     ██║   ║
║   ██║  ██║██╔══██║   ██║   ██╔══██║╚════██║██╔══╝     ██║   ║
║   ██████╔╝██║  ██║   ██║   ██║  ██║███████║███████╗   ██║   ║
║   ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚══════╝   ╚═╝   ║
║                                                              ║
║                  S A N I T I Z E R   P R O                   ║
║                                                              ║
║   ▸ Multi-format  ▸ PII Redaction  ▸ Quality Filtering       ║
║   ▸ Fuzzy Dedup   ▸ Parallel Jobs  ▸ LLM-Ready Output        ║
║                                                              ║
║          Production-Grade Cleaner for LLM Training           ║
╚══════════════════════════════════════════════════════════════╝
"""

# =============================================================================
# Excel Sheet Resolution
# =============================================================================

def resolve_excel_sheet(
    sheet_name: Optional[str], sheet_index: Optional[int], input_path: Optional[str] = None
) -> Any:
    if sheet_name is not None and sheet_index is not None:
        raise ConfigurationError("--excel-sheet-name and --excel-sheet-index are mutually exclusive.")
    if sheet_index is not None and sheet_index < 0:
        raise ConfigurationError("--excel-sheet-index must be >= 0.")

    resolved: Any = sheet_name if sheet_name is not None else (sheet_index if sheet_index is not None else 0)
    
    if input_path and input_path not in {_STDIN}:
        try:
            if OPENPYXL_AVAILABLE:
                wb = _openpyxl.load_workbook(input_path, read_only=True, data_only=True)
                available = wb.sheetnames
                wb.close()
            elif PANDAS_AVAILABLE:
                xl = pd.ExcelFile(input_path)
                available = xl.sheet_names
                xl.close()
            else:
                return resolved

            if isinstance(resolved, str) and resolved not in available:
                raise ConfigurationError(f"Sheet '{resolved}' not found. Available: {available}")
            elif isinstance(resolved, int) and resolved >= len(available):
                raise ConfigurationError(f"Sheet index {resolved} out of range. Available: {available}")
        except ConfigurationError:
            raise
        except Exception as exc:
            logging.warning(f"Could not pre-validate Excel sheet: {exc}")
            
    return resolved

# =============================================================================
# Argument Parser
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"LLM Dataset Sanitizer PRO v{__version__} — Modular Production Cleaner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  sanitize --input data.jsonl --output clean.jsonl --deduplicate --remove-pii
  sanitize --input data.jsonl --output chatml.jsonl --fuzzy-dedup --format-chatml
  sanitize --input huge.jsonl --output clean.jsonl --jobs 8 --dedup-backend sqlite
"""
    )

    # Core I/O
    parser.add_argument('--version', action='version', version=f"sanitize {__version__}")
    parser.add_argument('--input', default=None, help="Input file or '-' for stdin.")
    parser.add_argument('--output', default=None, help="Output file or '-' for stdout.")
    parser.add_argument('--input-format', default=None, metavar='FMT', help="Override input format.")
    parser.add_argument('--output-format', default=None, metavar='FMT', help="Override output format.")
    parser.add_argument('--config', default=None, metavar='PATH', help="YAML or JSON config file.")
    parser.add_argument('--generate-config', default=None, const='yaml', nargs='?', choices=['yaml', 'json'], metavar='FMT', help="Print config template and exit.")
    parser.add_argument('--profile', default=None, metavar='NAME',
                        help="Apply a preset flag bundle: fine-tune, pretrain, rag "
                             "(or 'list' to show them). Explicit flags and --config override it.")

    # Quality Filters
    qg = parser.add_argument_group('Quality Filters')
    qg.add_argument('--min-chars', type=int, default=D.min_chars)
    qg.add_argument('--max-chars', type=int, default=D.max_chars)
    qg.add_argument('--min-words', type=int, default=D.min_words)
    qg.add_argument('--min-ascii-ratio', type=float, default=D.min_ascii_ratio,
                    help='Reject records whose ASCII-character share is below this '
                         '(0 = off, the default; e.g. 0.85 keeps mostly-English text).')
    qg.add_argument('--min-unique-ratio', type=float, default=D.min_unique_ratio)
    qg.add_argument('--text-fields', default='', help='Comma-separated fields for quality scoring.')
    qg.add_argument('--text-fields-depth', type=int, default=D.text_fields_depth)
    qg.add_argument('--reject-allcaps', action='store_true')
    qg.add_argument('--allcaps-min-len', type=int, default=D.allcaps_min_len)
    qg.add_argument('--allcaps-min-alpha', type=int, default=D.allcaps_min_alpha)
    qg.add_argument('--require-fields', default='')
    qg.add_argument('--quality-script', default=None, metavar='PATH')
    qg.add_argument('--max-depth', type=int, default=D.max_depth)
    qg.add_argument('--lang-filter', default='',
                    help='Keep only these languages, e.g. en,zh or eng,cmn (ISO 639-1 or -3).')
    qg.add_argument('--lang-confidence', type=float, default=D.lang_confidence)
    qg.add_argument('--lang-backend', default=D.lang_backend, choices=list(LANG_BACKENDS),
                    help='Language ID for --lang-filter: glotlid (fastText, 2000+ varieties; '
                         'default when fastText is installed), openlid, or langdetect.')
    qg.add_argument('--lang-model', default=None, metavar='PATH',
                    help='Local fastText language-ID model file (skips the Hub download).')
    qg.add_argument('--reject-code', action='store_true', help='Reject records detected as code snippets.')
    qg.add_argument('--reject-profanity', action='store_true', help='Reject records containing profanity.')
    qg.add_argument('--quality-scorer', default=D.quality_scorer, choices=list(QUALITY_SCORERS),
                    help='Scoring backend for --quality-min-score / --keep-top-percent / '
                         '--quality-score-field: heuristic (default, no dependencies), '
                         'perplexity, fineweb-edu (classifier, 0-5 mapped to 0-1), dclm '
                         '(fastText, P(high quality)), or fasttext (your own model).')
    qg.add_argument('--quality-model', default=None, metavar='NAME',
                    help='Model for the scorer: causal LM for perplexity (default distilgpt2), '
                         'HF model for fineweb-edu, or a fastText .bin for fasttext.')
    qg.add_argument('--quality-label', default=None, metavar='LABEL',
                    help="Positive label of a --quality-scorer fasttext model (e.g. __label__hq).")
    qg.add_argument('--quality-rules', default='', metavar='SETS',
                    help='Reject documents failing published pretraining rules: gopher, '
                         'gopher-repetition, c4, fineweb (comma-separated, or all). '
                         'English-tuned; thresholds as in datatrove.')
    qg.add_argument('--quality-min-score', type=float, default=None, metavar='X',
                    help='Reject records with quality score < X (scores are in [0, 1]).')
    qg.add_argument('--keep-top-percent', type=float, default=None, metavar='P',
                    help='Keep only the best P%% of surviving records by quality score. '
                         'Buffers survivors in memory; output order is preserved.')
    qg.add_argument('--quality-score-field', default=None, metavar='FIELD',
                    help='Annotate each output record with its quality score in FIELD.')

    # Features
    fg = parser.add_argument_group('Features')
    fg.add_argument('--deduplicate', action='store_true')
    fg.add_argument('--fuzzy-dedup', action='store_true', help='Use MinHash+LSH for near-duplicate detection.')
    fg.add_argument('--fuzzy-threshold', type=float, default=D.fuzzy_threshold, metavar='T',
                    help='Jaccard similarity threshold for --fuzzy-dedup (0-1, default 0.8).')
    fg.add_argument('--semantic-dedup', action='store_true',
                    help='Embedding-based near-dedup: drops paraphrases that share no '
                         'n-grams (pip install model2vec; ~30MB model, no torch).')
    fg.add_argument('--semantic-threshold', type=float, default=D.semantic_threshold, metavar='T',
                    help='Cosine similarity threshold for --semantic-dedup (default 0.9).')
    fg.add_argument('--semantic-model', default=D.semantic_model, metavar='NAME',
                    help='model2vec static embedding model for --semantic-dedup.')
    fg.add_argument('--dedup-fields', default='')
    fg.add_argument('--dedup-normalize', action='store_true')
    fg.add_argument('--dedup-backend', default=D.dedup_backend, choices=list(DEDUP_BACKENDS))
    fg.add_argument('--dedup-db-path', default=None, metavar='PATH')
    fg.add_argument('--remove-pii', action='store_true')
    fg.add_argument('--pii-mask', action='store_true')
    fg.add_argument('--pii-pseudonymize', action='store_true')
    fg.add_argument('--pseudo-map-file', default=None, metavar='PATH')
    fg.add_argument('--pii-patterns-file', default=None, metavar='PATH')
    fg.add_argument('--pii-ner', action='store_true',
                    help='Also detect PII with a named-entity model (person names by default). '
                         'Requires spacy (+en_core_web_sm) or transformers.')
    fg.add_argument('--pii-ner-backend', default=D.pii_ner_backend, choices=list(NER_BACKENDS))
    fg.add_argument('--pii-ner-entities', default='person', metavar='KINDS',
                    help='Comma-separated entity kinds to redact (default: person): person, '
                         'location, org; with --pii-ner-backend gliner also address, '
                         'date_of_birth, id_number, financial, username, credential; or all.')
    fg.add_argument('--pii-ner-threshold', type=float, default=D.pii_ner_threshold,
                    metavar='T', help='GLiNER confidence threshold (default 0.5).')
    fg.add_argument('--pii-ner-model', default=None, metavar='NAME',
                    help='Override the NER model (spaCy model name or HF model id).')
    fg.add_argument('--redact-secrets', action='store_true',
                    help='Detect and redact credentials — API keys, tokens, private keys, '
                         'connection strings. Works with or without --remove-pii; honors '
                         '--pii-mask / --pii-pseudonymize.')
    fg.add_argument('--clean-html', action='store_true')
    fg.add_argument('--paragraph-mode', action='store_true')
    fg.add_argument('--txt-fallback-field', default=None, metavar='FIELD')
    fg.add_argument('--field-config', default=None, metavar='PATH')
    fg.add_argument('--max-tokens', type=int, default=None)
    fg.add_argument('--tokenizer', default=D.tokenizer)
    fg.add_argument('--sample', type=float, default=None)
    fg.add_argument('--seed', type=int, default=None,
                    help='Salt for --sample/--split. Both are decided by a hash of each '
                         "record's content, so results are reproducible across runs.")
    fg.add_argument('--split', default=None, metavar='SPEC')
    fg.add_argument('--quick', action='store_true')
    fg.add_argument('--format-chatml', action='store_true', help='Format output as ChatML messages.')
    fg.add_argument('--format-instruct', action='store_true', help='Format output as Alpaca/Instruct schema.')

    # Decontamination
    dg = parser.add_argument_group('Benchmark Decontamination')
    dg.add_argument('--decontaminate', default=None, metavar='NAMES',
                    help="Comma-separated benchmark names to decontaminate against "
                         "(e.g. mmlu,gsm8k), 'all', or 'list' to show available benchmarks. "
                         "Test sets are downloaded from the Hugging Face Hub and cached.")
    dg.add_argument('--decontam-refs', default=None, metavar='PATHS',
                    help='Comma-separated local reference files (any supported input format) '
                         'whose text is treated as benchmark material.')
    dg.add_argument('--decontam-ngram', type=int, default=D.decontam_ngram, metavar='N',
                    help='Word n-gram size for overlap detection (default 8).')
    dg.add_argument('--decontam-min-hits', type=int, default=D.decontam_min_hits, metavar='N',
                    help='Minimum colliding n-grams to flag a record (default 1).')
    dg.add_argument('--decontam-cache', default=None, metavar='DIR',
                    help='Benchmark download cache dir (default ~/.cache/llm-sanitizer-pro/benchmarks).')

    # Chat validation
    cg = parser.add_argument_group('Chat Dataset Validation')
    cg.add_argument('--validate-chat', action='store_true',
                    help='Reject records whose "messages" structure is invalid for chat '
                         'fine-tuning (role alternation, empty turns, no assistant reply, …). '
                         'Combine with --format-chatml to convert first, then validate.')
    cg.add_argument('--chat-lenient', action='store_true',
                    help='Only structural checks (schema, known roles, non-empty content, '
                         'assistant present); skip ordering/alternation rules.')
    cg.add_argument('--chat-max-tokens', type=int, default=None, metavar='N',
                    help='Reject conversations whose total content exceeds N tokens '
                         '(counted with --tokenizer).')
    cg.add_argument('--chat-roles', default='system,user,assistant', metavar='ROLES',
                    help='Comma-separated allowed roles (default: system,user,assistant).')

    # CSV / Excel
    iog = parser.add_argument_group('CSV / Excel Options')
    iog.add_argument('--csv-delimiter', default=None, metavar='CHAR')
    iog.add_argument('--csv-no-header', action='store_true')
    iog.add_argument('--csv-columns', default='')
    iog.add_argument('--excel-sheet-name', default=None, metavar='NAME')
    iog.add_argument('--excel-sheet-index', type=int, default=None, metavar='N')
    iog.add_argument('--excel-warn-size', type=float, default=_EXCEL_WARN_MB_DEFAULT)

    # I/O
    io_g = parser.add_argument_group('I/O Options')
    io_g.add_argument('--encoding', default=D.encoding)
    io_g.add_argument('--shard-size', type=int, default=None, metavar='N')
    io_g.add_argument('--json-path', default='item', metavar='PATH')
    io_g.add_argument('--hf-cache', default=None, metavar='DIR',
                      help='Cache dir for hf:// dataset downloads '
                           '(default ~/.cache/llm-sanitizer-pro/datasets).')

    # Runtime
    rt = parser.add_argument_group('Runtime')
    rt.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    rt.add_argument('--log-format', default='text', choices=['text', 'json'],
                    help="Log line format on stderr: human text or JSON lines "
                         "(for structured ingestion). Default: text.")
    rt.add_argument('--quiet', action='store_true')
    rt.add_argument('--no-progress', action='store_true')
    rt.add_argument('--jobs', type=int, default=1)
    rt.add_argument('--chunk-size', type=int, default=64, metavar='N')
    rt.add_argument('--dry-run', action='store_true')
    rt.add_argument('--dry-run-size', type=int, default=10_000)
    rt.add_argument('--stats-only', action='store_true')
    rt.add_argument('--debug-records', action='store_true')
    rt.add_argument('--stats-file', default=None, metavar='PATH')
    rt.add_argument('--report', default=None, metavar='PATH',
                    help='Write a self-contained HTML audit report (removal funnel, PII '
                         'counts by type, sample diffs) to PATH. Samples are redacted.')
    rt.add_argument('--report-raw-samples', action='store_true',
                    help='Include verbatim (unredacted) record samples in --report. The report '
                         'then contains raw PII/secrets; handle it like the input data.')
    rt.add_argument('--resume', action='store_true',
                    help='Checkpoint progress to <output>.checkpoint.json and, when a '
                         'checkpoint exists, continue the run from where it stopped '
                         '(jsonl/txt/csv outputs, --jobs 1).')
    rt.add_argument('--checkpoint-interval', type=int, default=10_000, metavar='N',
                    help='Write a checkpoint every N input records (default 10000).')

    return parser

# =============================================================================
# Main Orchestration
# =============================================================================
#
# The pipeline itself lives in Sanitizer (api.py), shared with library users.
# This module only adds what a command-line run needs on top: argument and
# config-file merging, input/output resolution, resumable checkpoints,
# multiprocessing, progress bars, and the summary/stats/report artifacts.


class CliError(Exception):
    """A user-facing error: logged, then exit code 1."""


@dataclass
class IOPlan:
    input_fmt: str
    output_fmt: str
    is_hub_input: bool
    excel_sheet: Any
    split_spec: Optional[Dict[str, float]]
    no_output: bool


@dataclass
class ResumeState:
    skip: int = 0
    stats: Optional[RunStats] = None
    pseudo: Optional[PseudoRegistry] = None
    dedup_mark: Optional[int] = None


def _print_info_and_exit(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """--generate-config, --decontaminate list and --profile list."""
    if args.generate_config is not None:
        template = {a.dest: a.default for a in parser._actions
                    if a.dest not in {'help', 'generate_config', 'config'}
                    and a.default is not None and a.default is not argparse.SUPPRESS}
        if args.generate_config == 'yaml':
            try:
                import yaml as _yaml
                print(_yaml.safe_dump(template, sort_keys=True, default_flow_style=False))
            except ImportError:
                logging.warning("pyyaml not installed; emitting JSON instead.")
                print(json.dumps(template, indent=2))
        else:
            print(json.dumps(template, indent=2))
        sys.exit(0)
    if args.decontaminate and args.decontaminate.strip().lower() == 'list':
        from sanitizer_pro.decontam import BENCHMARK_GROUPS, KNOWN_BENCHMARKS
        for name, spec in sorted(KNOWN_BENCHMARKS.items()):
            print(f"{name:<15} {spec.repo:<36} {spec.note}")
        for group, members in BENCHMARK_GROUPS.items():
            print(f"{group:<15} (group) {', '.join(members)}")
        print("all             (every benchmark not marked gated; gated ones need HF_TOKEN)")
        sys.exit(0)
    if args.profile is not None:
        from sanitizer_pro.profiles import PROFILE_NAMES, describe_profiles
        if args.profile.strip().lower() == 'list':
            print(describe_profiles())
            sys.exit(0)
        if args.profile not in PROFILE_NAMES:
            parser.error(f"--profile must be one of {list(PROFILE_NAMES)} or 'list' "
                         f"(got '{args.profile}').")


def _merge_settings(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Precedence: explicit CLI flags > --config > --profile/--quick > defaults."""
    explicit_args = collect_explicit_args(parser)
    if args.profile:
        from sanitizer_pro.profiles import profile_settings
        for dest, val in profile_settings(args.profile).items():
            if dest not in explicit_args:
                setattr(args, dest, val)
    if args.config:
        try:
            apply_config_to_args(args, load_config_file(args.config), explicit_args, parser)
        except Exception as exc:
            print(f"ERROR loading config: {exc}", file=sys.stderr)
            sys.exit(1)
    if args.quick:
        for dest in ('remove_pii', 'deduplicate', 'clean_html', 'dedup_normalize'):
            if dest not in explicit_args:
                setattr(args, dest, True)
    if args.debug_records:
        args.log_level = 'DEBUG'


def _setup_logging(args: argparse.Namespace) -> None:
    eff_level = 'WARNING' if args.quiet else args.log_level
    handler = logging.StreamHandler(sys.stderr)
    if args.log_format == 'json':
        from sanitizer_pro.logutil import JsonLogFormatter
        handler.setFormatter(JsonLogFormatter())
    else:
        handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    logging.basicConfig(level=getattr(logging, eff_level), handlers=[handler], force=True)
    if (args.log_format != 'json' and not args.quiet
            and eff_level in {'DEBUG', 'INFO'} and args.output != _STDOUT):
        print(BANNER, file=sys.stderr)


def _plan_io(args: argparse.Namespace, config: SanitizerConfig) -> IOPlan:
    """Validate CLI-only options and resolve input/output formats."""
    if args.jobs < 1:
        raise CliError("--jobs must be >= 1.")
    if config.pii_pseudonymize and args.jobs > 1:
        raise CliError("--pii-pseudonymize is not supported with --jobs > 1.")
    if args.shard_size is not None and args.shard_size < 1:
        raise CliError("--shard-size must be >= 1.")
    split_spec = None
    if args.split:
        if args.shard_size:
            raise CliError("--split and --shard-size are mutually exclusive.")
        split_spec = parse_split_spec(args.split)

    is_hub_input = str(args.input).startswith('hf://')
    if is_hub_input:
        from sanitizer_pro.hub import parse_hf_uri
        parse_hf_uri(args.input)  # fail fast on malformed URIs
        input_fmt = 'hf'
    else:
        input_fmt = resolve_fmt(args.input, args.input_format)
        if not input_fmt:
            raise CliError("Cannot detect input format. Supply --input-format.")
    excel_sheet: Any = 0
    if input_fmt in {'.xlsx', '.xls'}:
        excel_sheet = resolve_excel_sheet(args.excel_sheet_name, args.excel_sheet_index,
                                          args.input if args.input != _STDIN else None)
    if args.input != _STDIN and not is_hub_input and not os.path.exists(args.input):
        raise CliError(f"Input file not found: {args.input}")

    if args.output not in {_STDOUT, '/dev/null'}:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.', exist_ok=True)
    no_output = args.dry_run or args.stats_only
    output_fmt = resolve_fmt(args.output, args.output_format)
    if not output_fmt:
        if not (no_output or args.output == '/dev/null'):
            raise CliError("Cannot detect output format. Supply --output-format.")
        output_fmt = input_fmt if input_fmt not in ('', 'hf') else '.jsonl'
    return IOPlan(input_fmt, output_fmt, is_hub_input, excel_sheet, split_spec, no_output)


def _prepare_resume(args: argparse.Namespace, config: SanitizerConfig, plan: IOPlan) -> ResumeState:
    """Validate --resume and restore the last checkpoint (truncating output
    written after it)."""
    if not args.resume:
        return ResumeState()
    from sanitizer_pro.checkpoint import (
        load_checkpoint, truncate_output_to_checkpoint, warn_about_volatile_state,
    )
    problems = []
    if args.input == _STDIN: problems.append("stdin input")
    if args.output in {_STDOUT, '/dev/null'}: problems.append("stdout//dev/null output")
    if plan.no_output: problems.append("--dry-run/--stats-only")
    if plan.split_spec or args.shard_size: problems.append("--split/--shard-size")
    if config.keep_top_percent is not None: problems.append("--keep-top-percent")
    if args.jobs > 1: problems.append("--jobs > 1")
    if plan.output_fmt not in {'.jsonl', '.txt', '.csv'}:
        problems.append(f"{plan.output_fmt} output (appendable formats: .jsonl/.txt/.csv)")
    if problems:
        raise CliError(f"--resume is not compatible with: {', '.join(problems)}")
    if args.checkpoint_interval < 1:
        raise CliError("--checkpoint-interval must be >= 1.")
    ckpt = load_checkpoint(args.output, args.input)
    if not ckpt:
        return ResumeState()
    discarded = truncate_output_to_checkpoint(args.output, ckpt.get('output_bytes'))
    if discarded:
        logging.info(f"Discarded {discarded:,} output bytes written after the last "
                     "checkpoint; those records are re-processed.")
    warn_about_volatile_state(args)
    state = ResumeState(skip=int(ckpt['records_read']),
                        stats=RunStats.from_state(ckpt['stats']),
                        pseudo=PseudoRegistry.from_state(ckpt['pseudo']) if ckpt.get('pseudo') else None,
                        dedup_mark=ckpt.get('dedup_mark'))
    logging.info(f"Resuming from checkpoint: skipping {state.skip:,} already-processed "
                 "input records, appending to output.")
    return state


def _open_writer(args: argparse.Namespace, plan: IOPlan, appending: bool) -> Any:
    if plan.no_output:
        return contextlib.nullcontext(None)
    if plan.split_spec:
        return SplitWriter(args.output, plan.output_fmt, args.encoding, plan.split_spec,
                           txt_fallback_field=args.txt_fallback_field, seed=args.seed)
    if args.shard_size:
        return ShardedWriter(args.output, plan.output_fmt, args.encoding, args.shard_size,
                             txt_fallback_field=args.txt_fallback_field)
    return StreamingWriter(args.output, plan.output_fmt, args.encoding,
                           txt_fallback_field=args.txt_fallback_field,
                           append=appending, durable=args.resume)


class _Checkpointer:
    """Writes a consistent checkpoint every N input records (single process)."""

    def __init__(self, args: argparse.Namespace, sanitizer: Sanitizer) -> None:
        from sanitizer_pro.dedup import SQLiteDeduper
        self.args, self.s = args, sanitizer
        # Only a named SQLite DB survives the process, so only it is rolled back.
        self.durable_dedup: Optional[SQLiteDeduper] = (
            sanitizer.deduper if args.resume and args.dedup_db_path
            and isinstance(sanitizer.deduper, SQLiteDeduper) else None)

    def rollback_dedup(self, mark: Optional[int]) -> None:
        if self.durable_dedup is None:
            return
        if mark is None:
            logging.warning("Checkpoint has no dedup mark; hashes recorded after it may "
                            "drop records as false duplicates.")
            return
        forgotten = self.durable_dedup.rollback_to(mark)
        if forgotten:
            logging.info(f"Dedup DB: forgot {forgotten:,} hashes recorded after the last checkpoint.")

    def maybe_save(self, writer: Any) -> None:
        """Called only between records, so every counted record is written."""
        if not self.args.resume or self.s.stats.total % self.args.checkpoint_interval != 0:
            return
        from sanitizer_pro.checkpoint import save_checkpoint
        # Make output and dedup state durable *before* the checkpoint that
        # references them: a crash in between leaves the previous checkpoint,
        # whose smaller marks make resume discard the newer rows and hashes.
        output_bytes = writer.durable_size() if writer is not None else None
        dedup_mark = self.durable_dedup.high_water_mark() if self.durable_dedup else None
        if self.s.deduper is not None and hasattr(self.s.deduper, 'flush'):
            self.s.deduper.flush()
        reg = self.s.pseudo_registry
        save_checkpoint(self.args.output, input_path=self.args.input,
                        records_read=self.s.stats.total, stats_state=self.s.stats.to_state(),
                        pseudo_state=reg.to_state() if reg else None,
                        output_bytes=output_bytes, dedup_mark=dedup_mark)


def _skip(it: Iterator[Any], n: int) -> Iterator[Any]:
    for i, rec in enumerate(it):
        if i >= n:
            yield rec


def _progress(it: Iterable[Any], args: argparse.Namespace, desc: str) -> Iterable[Any]:
    if TQDM_AVAILABLE and not args.no_progress and not args.quiet and args.input != _STDIN:
        wrapped: Iterable[Any] = _tqdm(it, desc=desc, unit="rec", dynamic_ncols=True,
                                       smoothing=0.1)
        return wrapped
    return it


def _run_pipeline(args: argparse.Namespace, sanitizer: Sanitizer, records: Iterator[Any],
                  writer: Any, ckpt: _Checkpointer) -> None:
    def write(out: List[Dict[str, Any]]) -> None:
        if writer is not None:
            for rec in out:
                writer.write(rec)

    limit = args.dry_run_size if args.dry_run else None
    if args.jobs == 1:
        for record in _progress(records, args, "Sanitizing"):
            if limit is not None and sanitizer.stats.total >= limit:
                break
            write(sanitizer.feed(record))
            ckpt.maybe_save(writer)
    else:
        def dispatchable() -> Iterator[Dict[str, Any]]:
            # Non-objects are counted here so `malformed` stays accurate.
            for rec in records:
                if isinstance(rec, dict):
                    yield rec
                else:
                    sanitizer.stats.total += 1
                    sanitizer.stats.malformed += 1

        pool = multiprocessing.Pool(processes=args.jobs, initializer=_worker_init,
                                    initargs=(sanitizer.config, args.log_level))
        stopped_early = False
        try:
            results = pool.imap(_worker_fn, dispatchable(), chunksize=args.chunk_size)
            for transformed, pii_counts in _progress(results, args, "Processing"):
                if limit is not None and sanitizer.stats.total >= limit:
                    stopped_early = True
                    break
                write(sanitizer.feed_transformed(transformed, pii_counts))
        except BaseException:
            stopped_early = True
            raise
        finally:
            if stopped_early:
                pool.terminate()
            else:
                pool.close()
            pool.join()
    write(sanitizer.finish())


def _print_summary(args: argparse.Namespace, stats: RunStats) -> None:
    if args.log_format == 'json':
        # Keep stderr a clean JSON-lines stream: emit the summary as one record.
        print(json.dumps({'event': 'complete', **stats.to_dict()}), file=sys.stderr)
        return
    total = stats.total
    kept_pct = (stats.kept / total * 100) if total > 0 else 0.0
    sep = '=' * 62
    rows = [
        ("Total records processed", total), ("Kept", None),
        ("Filtered (quality)", stats.filtered_quality),
        ("Filtered (rules)", stats.filtered_rules),
        ("Filtered (language)", stats.filtered_lang),
        ("Filtered (require)", stats.filtered_require),
        ("Filtered (code)", stats.filtered_code),
        ("Filtered (profanity)", stats.filtered_profanity),
        ("Filtered (contaminated)", stats.filtered_contaminated),
        ("Filtered (chat-invalid)", stats.filtered_chat),
        ("Filtered (low score)", stats.filtered_low_score),
        ("Deduplicated", stats.deduplicated), ("Malformed", stats.malformed),
        ("Sampled out", stats.sampled_out),
    ]
    lines = [f"\n{sep}", f"SANITIZATION COMPLETE — v{__version__}{_mode_tag(args)}", sep]
    for label, value in rows:
        shown = f"{stats.kept:,}  ({kept_pct:.2f}%)" if value is None else f"{value:,}"
        lines.append(f"{label:<24}: {shown}")
    if stats.contaminated_by:
        top = ', '.join(f"{k}={v}" for k, v in sorted(
            stats.contaminated_by.items(), key=lambda x: -x[1])[:8])
        lines.append(f"  contamination by benchmark: {top}")
    if stats.rule_failures:
        top = ', '.join(f"{k}={v}" for k, v in sorted(
            stats.rule_failures.items(), key=lambda x: -x[1])[:5])
        lines.append(f"  rule failures: {top}")
    if stats.chat_invalid_reasons:
        top = ', '.join(f"{k}={v}" for k, v in sorted(
            stats.chat_invalid_reasons.items(), key=lambda x: -x[1])[:5])
        lines.append(f"  chat-invalid breakdown: {top}")
    lines.append(sep)
    print('\n'.join(lines), file=sys.stderr)


def _mode_tag(args: argparse.Namespace) -> str:
    return ' [DRY RUN]' if args.dry_run else (' [STATS ONLY]' if args.stats_only else '')


def _write_artifacts(args: argparse.Namespace, config: SanitizerConfig, plan: IOPlan,
                     sanitizer: Sanitizer, duration: float) -> None:
    if args.stats_file:
        try:
            Path(args.stats_file).write_text(
                json.dumps({'version': __version__, **sanitizer.stats.to_dict()}, indent=2),
                encoding='utf-8')
        except Exception as exc:
            logging.warning(f"Could not write stats file: {exc}")
    if not args.report:
        return
    c = config
    features = [name for enabled, name in [
        (c.remove_pii, 'PII redaction' + (' + NER' if c.pii_ner else '')),
        (c.redact_secrets, 'secrets redaction'),
        (c.pii_pseudonymize, 'pseudonymization'),
        (c.deduplicate, f'exact dedup ({c.dedup_backend})'),
        (c.fuzzy_dedup, f'fuzzy dedup (t={c.fuzzy_threshold})'),
        (c.semantic_dedup, f'semantic dedup (t={c.semantic_threshold})'),
        (bool(c.decontaminate or c.decontam_refs),
         'decontamination' + (f" ({','.join(c.decontaminate)})" if c.decontaminate else '')),
        (c.validate_chat, 'chat validation'),
        (sanitizer.transformer.scorer is not None, f'quality scoring ({c.quality_scorer})'),
        (bool(c.quality_rules), f"quality rules ({','.join(c.quality_rules or [])})"),
        (c.clean_html, 'HTML stripping'),
        (bool(c.lang_filter), f"language filter ({','.join(c.lang_filter or [])})"),
    ] if enabled]
    meta = {
        'Input': f"{args.input} ({plan.input_fmt})",
        'Output': f"{args.output} ({plan.output_fmt}){_mode_tag(args)}",
        'Profile': args.profile or '—',
        'Active features': ', '.join(features) or 'none',
        'Duration': f"{duration:.1f}s | jobs={args.jobs}",
        'version': __version__,
    }
    try:
        sanitizer.write_report(args.report, meta)
        logging.info(f"Audit report written to {args.report}")
    except Exception as exc:
        logging.warning(f"Could not write audit report: {exc}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _print_info_and_exit(args, parser)
    if args.input is None or args.output is None:
        parser.error("--input and --output are required.")
    _merge_settings(args, parser)
    _setup_logging(args)

    try:
        config = SanitizerConfig.from_namespace(args)
        config.validate()
        plan = _plan_io(args, config)
        resume = _prepare_resume(args, config, plan)
    except (CliError, ConfigurationError) as exc:
        logging.error(str(exc)); sys.exit(1)
    except Exception as exc:
        logging.error(f"Failed to load auxiliary config: {exc}"); sys.exit(1)
    for note in config.warnings():
        logging.warning(note)
    if args.report_raw_samples and args.report:
        logging.warning("--report-raw-samples: the audit report will contain unredacted "
                        "records (PII/secrets). Handle it like the raw input data.")

    try:
        sanitizer = Sanitizer(config, stats=resume.stats, pseudo_registry=resume.pseudo)
    except (ConfigurationError, ImportError) as exc:
        logging.error(str(exc)); sys.exit(1)
    except Exception as exc:
        logging.error(f"Failed to initialize the pipeline: {exc}"); sys.exit(1)
    ckpt = _Checkpointer(args, sanitizer)
    if resume.stats is not None:
        ckpt.rollback_dedup(resume.dedup_mark)

    logging.info(f"Start: {args.input} ({plan.input_fmt}) → {args.output} ({plan.output_fmt}) "
                 f"| jobs={args.jobs}")
    started = time.monotonic()
    try:
        records: Iterator[Any] = read_records(
            args.input, encoding=args.encoding, paragraph_mode=args.paragraph_mode,
            csv_delimiter=args.csv_delimiter, csv_no_header=args.csv_no_header,
            csv_columns=as_list(args.csv_columns), excel_sheet=plan.excel_sheet,
            excel_warn_mb=args.excel_warn_size,
            input_format=None if plan.is_hub_input else plan.input_fmt,
            json_path=args.json_path, hf_cache=args.hf_cache, yield_malformed=True)
    except Exception as exc:
        logging.critical(f"Failed to open input: {exc}"); sys.exit(1)
    if resume.skip:
        records = _skip(records, resume.skip)

    writer_ctx: Any = None
    try:
        writer_ctx = _open_writer(args, plan, appending=resume.stats is not None)
        with writer_ctx as writer:
            _run_pipeline(args, sanitizer, records, writer, ckpt)
        if args.resume:
            from sanitizer_pro.checkpoint import clear_checkpoint
            clear_checkpoint(args.output)
    except KeyboardInterrupt:
        logging.warning("Interrupted — flushing output …")
        if writer_ctx is not None and hasattr(writer_ctx, 'flush'): writer_ctx.flush()
        if args.resume:
            # No checkpoint here: the interrupt may have landed mid-record, so
            # the last periodic checkpoint is the latest consistent state.
            from sanitizer_pro.checkpoint import checkpoint_path
            if os.path.exists(checkpoint_path(args.output)):
                logging.info("Rerun the same command with --resume to continue from the "
                             "last checkpoint.")
            else:
                logging.info("No checkpoint was reached yet; a rerun starts from the beginning.")
        sys.exit(130)
    except Exception as exc:
        logging.critical(f"Fatal error: {exc}", exc_info=True)
        sys.exit(1)
    finally:
        sanitizer.close()
        if sanitizer.pseudo_registry is not None and args.pseudo_map_file:
            try:
                sanitizer.export_pseudonym_map(args.pseudo_map_file)
            except Exception as exc:
                logging.warning(f"Could not write pseudonym map: {exc}")

    _print_summary(args, sanitizer.stats)
    _write_artifacts(args, config, plan, sanitizer, time.monotonic() - started)


if __name__ == "__main__":
    main()
