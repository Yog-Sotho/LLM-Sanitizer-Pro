"""Command Line Interface and main orchestration loop for LLM Dataset Sanitizer PRO."""
import argparse
import json
import logging
import multiprocessing
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

try:
    from tqdm import tqdm as _tqdm
    TQDM_AVAILABLE = True
except ImportError:
    _tqdm = None  # type: ignore[assignment]
    TQDM_AVAILABLE = False

try:
    import openpyxl as _openpyxl
    OPENPYXL_AVAILABLE = True
except ImportError:
    _openpyxl = None  # type: ignore[assignment]
    OPENPYXL_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None  # type: ignore[assignment]
    PANDAS_AVAILABLE = False

from sanitizer_pro.utils import (
    FilterReason, ConfigurationError, _STDIN, _STDOUT, _EXCEL_WARN_MB_DEFAULT, _MAX_DEPTH_DEFAULT,
    _ALLCAPS_MIN_LEN_DEFAULT, _ALLCAPS_MIN_ALPHA_DEFAULT, resolve_fmt
)
from sanitizer_pro.config import (
    load_config_file, collect_explicit_args, apply_config_to_args,
    load_custom_pii_patterns, load_field_config, build_field_ops, load_quality_script
)
from sanitizer_pro.core import sanitize_record, TokenTruncator, get_record_hash
from sanitizer_pro.dedup import make_deduper
from sanitizer_pro.pii import PseudoRegistry
from sanitizer_pro.io.readers import read_records
from sanitizer_pro.io.writers import StreamingWriter, ShardedWriter, SplitWriter, parse_split_spec
from sanitizer_pro.worker import _worker_init, _worker_fn

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
║        S A N I T I Z E R   P R O   v 3 . 0                  ║
║                                                              ║
║   ▸ Multi-format  ▸ PII Redaction  ▸ Quality Filtering       ║
║   ▸ Fuzzy Dedup   ▸ Parallel Jobs  ▸ LLM-Ready Output        ║
║                                                              ║
║          Production-Grade Cleaner for LLM Training           ║
╚══════════════════════════════════════════════════════════════╝
"""

# =============================================================================
# Statistics Tracker
# =============================================================================

_CHAR_BUCKETS = [0, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
_WORD_BUCKETS = [0, 5, 10, 25, 50, 100, 250, 500, 1000, 2500]

class RunStats:
    """Tracks processing metrics and histograms."""
    def __init__(self) -> None:
        self.total = 0
        self.kept = 0
        self.filtered_quality = 0
        self.filtered_lang = 0
        self.filtered_require = 0
        self.filtered_code = 0
        self.filtered_profanity = 0
        self.filtered_contaminated = 0
        self.deduplicated = 0
        self.malformed = 0
        self.sampled_out = 0
        self.char_hist: Dict[int, int] = {b: 0 for b in _CHAR_BUCKETS}
        self.word_hist: Dict[int, int] = {b: 0 for b in _WORD_BUCKETS}
        self.lang_dist: Dict[str, int] = {}

    def record_kept(self, text: str, lang: Optional[str] = None) -> None:
        self.kept += 1
        n = len(text)
        bucket = next((b for b in reversed(_CHAR_BUCKETS) if n >= b), _CHAR_BUCKETS[0])
        self.char_hist[bucket] += 1
        w = len(text.split())
        wbucket = next((b for b in reversed(_WORD_BUCKETS) if w >= b), _WORD_BUCKETS[0])
        self.word_hist[wbucket] += 1
        if lang:
            self.lang_dist[lang] = self.lang_dist.get(lang, 0) + 1

    def to_dict(self) -> Dict[str, Any]:
        total = self.total or 1
        return {
            'total': self.total,
            'kept': self.kept,
            'kept_pct': round(self.kept / total * 100, 4),
            'filtered_quality': self.filtered_quality,
            'filtered_language': self.filtered_lang,
            'filtered_require': self.filtered_require,
            'filtered_code': self.filtered_code,
            'filtered_profanity': self.filtered_profanity,
            'filtered_contaminated': self.filtered_contaminated,
            'deduplicated': self.deduplicated,
            'malformed': self.malformed,
            'sampled_out': self.sampled_out,
            'char_length_histogram': {f">={k}": v for k, v in sorted(self.char_hist.items())},
            'word_count_histogram': {f">={k}": v for k, v in sorted(self.word_hist.items())},
            'language_distribution': dict(sorted(self.lang_dist.items(), key=lambda x: -x[1])),
        }

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
        description="LLM Dataset Sanitizer PRO v3.0 — Modular Production Cleaner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  sanitize --input data.jsonl --output clean.jsonl --deduplicate --remove-pii
  sanitize --input data.jsonl --output chatml.jsonl --fuzzy-dedup --format-chatml
  sanitize --input huge.jsonl --output clean.jsonl --jobs 8 --dedup-backend sqlite
"""
    )

    # Core I/O
    parser.add_argument('--input', default=None, help="Input file or '-' for stdin.")
    parser.add_argument('--output', default=None, help="Output file or '-' for stdout.")
    parser.add_argument('--input-format', default=None, metavar='FMT', help="Override input format.")
    parser.add_argument('--output-format', default=None, metavar='FMT', help="Override output format.")
    parser.add_argument('--config', default=None, metavar='PATH', help="YAML or JSON config file.")
    parser.add_argument('--generate-config', default=None, const='yaml', nargs='?', choices=['yaml', 'json'], metavar='FMT', help="Print config template and exit.")

    # Quality Filters
    qg = parser.add_argument_group('Quality Filters')
    qg.add_argument('--min-chars', type=int, default=50)
    qg.add_argument('--max-chars', type=int, default=20000)
    qg.add_argument('--min-words', type=int, default=8)
    qg.add_argument('--min-ascii-ratio', type=float, default=0.85)
    qg.add_argument('--min-unique-ratio', type=float, default=0.25)
    qg.add_argument('--text-fields', default='', help='Comma-separated fields for quality scoring.')
    qg.add_argument('--text-fields-depth', type=int, default=20)
    qg.add_argument('--reject-allcaps', action='store_true')
    qg.add_argument('--allcaps-min-len', type=int, default=_ALLCAPS_MIN_LEN_DEFAULT)
    qg.add_argument('--allcaps-min-alpha', type=int, default=_ALLCAPS_MIN_ALPHA_DEFAULT)
    qg.add_argument('--require-fields', default='')
    qg.add_argument('--quality-script', default=None, metavar='PATH')
    qg.add_argument('--max-depth', type=int, default=_MAX_DEPTH_DEFAULT)
    qg.add_argument('--lang-filter', default='')
    qg.add_argument('--lang-confidence', type=float, default=0.0)
    qg.add_argument('--reject-code', action='store_true', help='Reject records detected as code snippets.')
    qg.add_argument('--reject-profanity', action='store_true', help='Reject records containing profanity.')

    # Features
    fg = parser.add_argument_group('Features')
    fg.add_argument('--deduplicate', action='store_true')
    fg.add_argument('--fuzzy-dedup', action='store_true', help='Use MinHash+LSH for near-duplicate detection.')
    fg.add_argument('--fuzzy-threshold', type=float, default=0.8, metavar='T',
                    help='Jaccard similarity threshold for --fuzzy-dedup (0-1, default 0.8).')
    fg.add_argument('--dedup-fields', default='')
    fg.add_argument('--dedup-normalize', action='store_true')
    fg.add_argument('--dedup-backend', default='memory', choices=['memory', 'sqlite'])
    fg.add_argument('--dedup-db-path', default=None, metavar='PATH')
    fg.add_argument('--remove-pii', action='store_true')
    fg.add_argument('--pii-mask', action='store_true')
    fg.add_argument('--pii-pseudonymize', action='store_true')
    fg.add_argument('--pseudo-map-file', default=None, metavar='PATH')
    fg.add_argument('--pii-patterns-file', default=None, metavar='PATH')
    fg.add_argument('--pii-ner', action='store_true',
                    help='Also detect PII with a named-entity model (person names by default). '
                         'Requires spacy (+en_core_web_sm) or transformers.')
    fg.add_argument('--pii-ner-backend', default='auto', choices=['auto', 'spacy', 'transformers'])
    fg.add_argument('--pii-ner-entities', default='person', metavar='KINDS',
                    help='Comma-separated entity kinds to redact: person,location,org (default: person).')
    fg.add_argument('--pii-ner-model', default=None, metavar='NAME',
                    help='Override the NER model (spaCy model name or HF model id).')
    fg.add_argument('--clean-html', action='store_true')
    fg.add_argument('--paragraph-mode', action='store_true')
    fg.add_argument('--txt-fallback-field', default=None, metavar='FIELD')
    fg.add_argument('--field-config', default=None, metavar='PATH')
    fg.add_argument('--max-tokens', type=int, default=None)
    fg.add_argument('--tokenizer', default='whitespace')
    fg.add_argument('--sample', type=float, default=None)
    fg.add_argument('--seed', type=int, default=None)
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
    dg.add_argument('--decontam-ngram', type=int, default=8, metavar='N',
                    help='Word n-gram size for overlap detection (default 8).')
    dg.add_argument('--decontam-min-hits', type=int, default=1, metavar='N',
                    help='Minimum colliding n-grams to flag a record (default 1).')
    dg.add_argument('--decontam-cache', default=None, metavar='DIR',
                    help='Benchmark download cache dir (default ~/.cache/llm-sanitizer-pro/benchmarks).')

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
    io_g.add_argument('--encoding', default='utf-8')
    io_g.add_argument('--shard-size', type=int, default=None, metavar='N')
    io_g.add_argument('--json-path', default='item', metavar='PATH')

    # Runtime
    rt = parser.add_argument_group('Runtime')
    rt.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    rt.add_argument('--quiet', action='store_true')
    rt.add_argument('--no-progress', action='store_true')
    rt.add_argument('--jobs', type=int, default=1)
    rt.add_argument('--chunk-size', type=int, default=64, metavar='N')
    rt.add_argument('--dry-run', action='store_true')
    rt.add_argument('--dry-run-size', type=int, default=10_000)
    rt.add_argument('--stats-only', action='store_true')
    rt.add_argument('--debug-records', action='store_true')
    rt.add_argument('--stats-file', default=None, metavar='PATH')

    return parser

# =============================================================================
# Main Orchestration
# =============================================================================

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.generate_config is not None:
        template = {a.dest: a.default for a in parser._actions
                    if a.dest not in {'help', 'generate_config', 'config'} and a.default is not None}
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
        from sanitizer_pro.decontam import KNOWN_BENCHMARKS
        for name, spec in sorted(KNOWN_BENCHMARKS.items()):
            print(f"{name:<12} {spec.repo:<40} {spec.note}")
        sys.exit(0)

    if args.input is None or args.output is None:
        parser.error("--input and --output are required.")

    explicit_args = collect_explicit_args(parser)
    if args.config:
        try:
            cfg = load_config_file(args.config)
            apply_config_to_args(args, cfg, explicit_args)
        except Exception as exc:
            print(f"ERROR loading config: {exc}", file=sys.stderr)
            sys.exit(1)

    # Derived lists (config files may supply these as real lists already)
    def _as_list(v: Any) -> Optional[List[str]]:
        if isinstance(v, (list, tuple)):
            return [str(x).strip() for x in v if str(x).strip()] or None
        return [f.strip() for f in str(v or '').split(',') if f.strip()] or None

    args.text_fields_list = _as_list(args.text_fields)
    args.dedup_fields_list = _as_list(args.dedup_fields)
    args.csv_columns_list = _as_list(args.csv_columns)
    args.require_fields_list = _as_list(args.require_fields)
    lang_filter_set = set(x.lower() for x in _as_list(args.lang_filter) or []) or None

    if args.debug_records:
        args.log_level = 'DEBUG'

    if getattr(args, 'quick', False):
        for dest, val in {'remove_pii': True, 'deduplicate': True, 'clean_html': True, 'dedup_normalize': True}.items():
            if dest not in explicit_args: setattr(args, dest, val)

    if args.seed is not None:
        random.seed(args.seed)

    eff_level = 'WARNING' if args.quiet else args.log_level
    logging.basicConfig(
        level=getattr(logging, eff_level),
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[logging.StreamHandler(sys.stderr)],
        force=True
    )

    if not args.quiet and eff_level in {'DEBUG', 'INFO'} and args.output != _STDOUT:
        print(BANNER, file=sys.stderr)

    # Validation
    if args.pii_mask and not args.remove_pii: logging.warning("--pii-mask has no effect without --remove-pii.")
    if args.pii_pseudonymize and not args.remove_pii: logging.warning("--pii-pseudonymize has no effect without --remove-pii.")
    if args.pii_ner and not args.remove_pii: logging.warning("--pii-ner has no effect without --remove-pii.")
    if args.sample is not None and not (0 < args.sample <= 1.0):
        logging.error("--sample must be in (0, 1]."); sys.exit(1)
    if args.jobs < 1:
        logging.error("--jobs must be >= 1."); sys.exit(1)
    if args.pii_pseudonymize and args.jobs > 1:
        logging.error("--pii-pseudonymize is not supported with --jobs > 1."); sys.exit(1)
    if args.shard_size is not None and args.shard_size < 1:
        logging.error("--shard-size must be >= 1."); sys.exit(1)
    if not (0 < args.fuzzy_threshold <= 1):
        logging.error("--fuzzy-threshold must be in (0, 1]."); sys.exit(1)
    if lang_filter_set:
        from sanitizer_pro.quality import LANGDETECT_AVAILABLE
        if not LANGDETECT_AVAILABLE:
            logging.error("--lang-filter requires langdetect (pip install langdetect); "
                          "without it every record would be filtered out.")
            sys.exit(1)

    split_spec: Optional[Dict[str, float]] = None
    if args.split:
        if args.shard_size:
            logging.error("--split and --shard-size are mutually exclusive."); sys.exit(1)
        try:
            split_spec = parse_split_spec(args.split)
        except ConfigurationError as exc:
            logging.error(str(exc)); sys.exit(1)

    input_fmt = resolve_fmt(args.input, args.input_format)
    if not input_fmt:
        logging.error("Cannot detect input format. Supply --input-format."); sys.exit(1)

    excel_sheet: Any = 0
    if input_fmt in {'.xlsx', '.xls'}:
        try:
            excel_sheet = resolve_excel_sheet(args.excel_sheet_name, args.excel_sheet_index, args.input if args.input != _STDIN else None)
        except ConfigurationError as exc:
            logging.error(str(exc)); sys.exit(1)

    if args.input != _STDIN and not os.path.exists(args.input):
        logging.error(f"Input file not found: {args.input}"); sys.exit(1)

    if args.output not in {_STDOUT, '/dev/null'}:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.', exist_ok=True)

    no_output_early = args.dry_run or args.stats_only
    output_fmt = resolve_fmt(args.output, args.output_format)
    if not output_fmt:
        if no_output_early or args.output == '/dev/null':
            output_fmt = input_fmt or '.jsonl'
        else:
            logging.error("Cannot detect output format. Supply --output-format."); sys.exit(1)

    # Load optional resources
    try:
        extra_pii = load_custom_pii_patterns(args.pii_patterns_file) if args.pii_patterns_file else None
        field_ops = build_field_ops(load_field_config(args.field_config)) if args.field_config else None
        quality_fn = load_quality_script(args.quality_script) if args.quality_script else None
    except Exception as exc:
        logging.error(f"Failed to load auxiliary config: {exc}"); sys.exit(1)
    truncator = TokenTruncator(args.max_tokens, args.tokenizer) if args.max_tokens else None
    pseudo_registry = PseudoRegistry() if args.pii_pseudonymize else None

    ner_redactor = None
    if args.pii_ner and args.remove_pii:
        from sanitizer_pro.ner import NERRedactor
        try:
            # Built here even for --jobs > 1 (workers reload their own copy) so
            # a missing backend fails fast with a clear message.
            ner_redactor = NERRedactor(backend=args.pii_ner_backend,
                                       entities=str(args.pii_ner_entities).split(','),
                                       model=args.pii_ner_model)
        except (ConfigurationError, ImportError) as exc:
            logging.error(str(exc)); sys.exit(1)

    contamination_index = None
    if args.decontaminate or args.decontam_refs:
        from sanitizer_pro.decontam import build_index, resolve_benchmark_names
        try:
            contamination_index = build_index(
                benchmarks=resolve_benchmark_names(args.decontaminate) if args.decontaminate else None,
                ref_files=[p.strip() for p in args.decontam_refs.split(',') if p.strip()] if args.decontam_refs else None,
                cache_dir=args.decontam_cache, ngram=args.decontam_ngram,
                min_hits=args.decontam_min_hits, encoding=args.encoding,
            )
        except (ConfigurationError, ImportError) as exc:
            logging.error(str(exc)); sys.exit(1)
        except Exception as exc:
            logging.error(f"Failed to build decontamination index: {exc}"); sys.exit(1)

    logging.info(f"Start: {args.input} ({input_fmt}) → {args.output} ({output_fmt}) | jobs={args.jobs}")

    no_output = args.dry_run or args.stats_only
    try:
        deduper = make_deduper(args.dedup_backend, args.dedup_db_path, fuzzy=args.fuzzy_dedup,
                               fuzzy_threshold=args.fuzzy_threshold) if (args.deduplicate or args.fuzzy_dedup) else None
    except ImportError as exc:
        logging.error(str(exc)); sys.exit(1)
    run_stats = RunStats()

    try:
        record_iter: Iterator[Dict[str, Any]] = read_records(
            args.input, encoding=args.encoding, paragraph_mode=args.paragraph_mode,
            csv_delimiter=args.csv_delimiter, csv_no_header=args.csv_no_header,
            csv_columns=args.csv_columns_list, excel_sheet=excel_sheet,
            excel_warn_mb=args.excel_warn_size, input_format=input_fmt, json_path=args.json_path
        )
    except Exception as exc:
        logging.critical(f"Failed to open input: {exc}"); sys.exit(1)

    use_progress = TQDM_AVAILABLE and not args.no_progress and not args.quiet and args.input != _STDIN

    def _handle(sanitized: Optional[Dict[str, Any]], reason: Optional[FilterReason], quality_text: str, lang: Optional[str], writer: Any) -> None:
        if sanitized is None:
            if reason == FilterReason.LANGUAGE: run_stats.filtered_lang += 1
            elif reason == FilterReason.REQUIRE: run_stats.filtered_require += 1
            elif reason == FilterReason.CODE: run_stats.filtered_code += 1
            elif reason == FilterReason.PROFANITY: run_stats.filtered_profanity += 1
            else: run_stats.filtered_quality += 1
            return

        if contamination_index is not None and contamination_index.is_contaminated(quality_text):
            run_stats.filtered_contaminated += 1
            return

        if args.sample is not None and random.random() >= args.sample:
            run_stats.sampled_out += 1
            return

        if deduper is not None:
            if args.fuzzy_dedup:
                if deduper.contains(quality_text):
                    run_stats.deduplicated += 1
                    return
                deduper.add(quality_text)
            else:
                h = get_record_hash(sanitized, args.dedup_fields_list, args.dedup_normalize)
                if deduper.contains(h):
                    run_stats.deduplicated += 1
                    return
                deduper.add(h)

        run_stats.record_kept(quality_text, lang=lang)
        if writer is not None:
            writer.write(sanitized)

    def _process(writer: Any) -> None:
        if args.jobs > 1:
            def _dispatchable() -> Iterator[Dict[str, Any]]:
                # Filter non-dict records in the parent so `malformed` stays accurate.
                for rec in record_iter:
                    if isinstance(rec, dict):
                        yield rec
                    else:
                        run_stats.total += 1
                        run_stats.malformed += 1

            pool = multiprocessing.Pool(
                processes=args.jobs, initializer=_worker_init,
                initargs=(args, extra_pii, lang_filter_set, field_ops, args.require_fields_list, args.text_fields_list)
            )
            stopped_early = False
            try:
                imap_iter = pool.imap(_worker_fn, _dispatchable(), chunksize=args.chunk_size)
                if use_progress:
                    imap_iter = _tqdm(imap_iter, desc="Processing", unit="rec", dynamic_ncols=True, smoothing=0.1)
                for sanitized, reason, quality_text, lang in imap_iter:
                    run_stats.total += 1
                    if args.dry_run and run_stats.total > args.dry_run_size:
                        stopped_early = True
                        break
                    _handle(sanitized, reason, quality_text, lang, writer)
            except BaseException:
                stopped_early = True
                raise
            finally:
                if stopped_early:
                    pool.terminate()
                else:
                    pool.close()
                pool.join()
            return

        # Single-threaded
        if use_progress:
            record_iter_wrapped = _tqdm(record_iter, desc="Sanitizing", unit="rec", dynamic_ncols=True, smoothing=0.1)
        else:
            record_iter_wrapped = record_iter

        for record in record_iter_wrapped:
            run_stats.total += 1
            if args.dry_run and run_stats.total > args.dry_run_size: break
            if not isinstance(record, dict):
                run_stats.malformed += 1
                continue
            
            sanitized, reason, quality_text, lang = sanitize_record(
                record, args, text_fields=args.text_fields_list, extra_pii_patterns=extra_pii,
                lang_filter=lang_filter_set, field_ops=field_ops, truncator=truncator,
                pseudo_registry=pseudo_registry, require_fields=args.require_fields_list, quality_fn=quality_fn,
                ner_redactor=ner_redactor
            )
            _handle(sanitized, reason, quality_text, lang, writer)

    writer_ctx: Any = None
    try:
        if no_output:
            _process(writer=None)
        elif split_spec:
            writer_ctx = SplitWriter(args.output, output_fmt, args.encoding, split_spec, txt_fallback_field=args.txt_fallback_field)
            with writer_ctx as writer: _process(writer=writer)
        elif args.shard_size:
            writer_ctx = ShardedWriter(args.output, output_fmt, args.encoding, args.shard_size, txt_fallback_field=args.txt_fallback_field)
            with writer_ctx as writer: _process(writer=writer)
        else:
            writer_ctx = StreamingWriter(args.output, output_fmt, args.encoding, txt_fallback_field=args.txt_fallback_field)
            with writer_ctx as writer: _process(writer=writer)
    except KeyboardInterrupt:
        logging.warning("Interrupted — flushing output …")
        if writer_ctx is not None and hasattr(writer_ctx, 'flush'): writer_ctx.flush()
        sys.exit(130)
    except Exception as exc:
        logging.critical(f"Fatal error: {exc}", exc_info=True)
        sys.exit(1)
    finally:
        if deduper is not None: deduper.close()
        if pseudo_registry is not None and args.pseudo_map_file:
            try:
                Path(args.pseudo_map_file).write_text(json.dumps(pseudo_registry.to_dict(), indent=2, ensure_ascii=False), encoding='utf-8')
            except Exception as exc:
                logging.warning(f"Could not write pseudonym map: {exc}")

    # Final Report
    total = run_stats.total
    kept = run_stats.kept
    kept_pct = (kept / total * 100) if total > 0 else 0.0
    sep = '=' * 62
    mode_tag = ' [DRY RUN]' if args.dry_run else (' [STATS ONLY]' if args.stats_only else '')
    
    lines = [
        f"\n{sep}", f"SANITIZATION COMPLETE — v3.0{mode_tag}", sep,
        f"Total records processed : {total:,}",
        f"Kept                    : {kept:,}  ({kept_pct:.2f}%)",
        f"Filtered (quality)      : {run_stats.filtered_quality:,}",
        f"Filtered (language)     : {run_stats.filtered_lang:,}",
        f"Filtered (require)      : {run_stats.filtered_require:,}",
        f"Filtered (code)         : {run_stats.filtered_code:,}",
        f"Filtered (profanity)    : {run_stats.filtered_profanity:,}",
        f"Filtered (contaminated) : {run_stats.filtered_contaminated:,}",
        f"Deduplicated            : {run_stats.deduplicated:,}",
        f"Malformed               : {run_stats.malformed:,}",
        f"Sampled out             : {run_stats.sampled_out:,}",
        sep
    ]
    print('\n'.join(lines), file=sys.stderr)

    if args.stats_file:
        try:
            Path(args.stats_file).write_text(json.dumps({'version': '3.0', **run_stats.to_dict()}, indent=2), encoding='utf-8')
        except Exception as exc:
            logging.warning(f"Could not write stats file: {exc}")

if __name__ == "__main__":
    main()
