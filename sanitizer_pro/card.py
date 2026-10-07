"""Hugging Face dataset card (README.md) rendered from a run manifest.

The processing section is generated and exact: steps and their settings,
the record funnel, redaction counts by kind (counts only, never values),
decontamination results, models, file digests and the command to reproduce
the run. The parts only the dataset owner can write (summary, source,
license, intended use) are left as marked placeholders.
"""
import json
from typing import Any, Dict, List, Tuple

_SIZE_CATEGORIES = [(1_000, 'n<1K'), (10_000, '1K<n<10K'), (100_000, '10K<n<100K'),
                    (1_000_000, '100K<n<1M'), (10_000_000, '1M<n<10M'),
                    (100_000_000, '10M<n<100M'), (1_000_000_000, '100M<n<1B')]
TODO = '<!-- TODO: dataset owner -->'


def size_category(n: int) -> str:
    for limit, label in _SIZE_CATEGORIES:
        if n < limit:
            return label
    return 'n>1B'


def _languages(counts: Dict[str, Any], top: int = 20) -> List[str]:
    """Hub language codes (ISO 639-1 where one exists) of the detected languages."""
    from sanitizer_pro.langid import language_aliases
    codes: List[str] = []
    for label in list(counts.get('language_distribution', {}))[:top]:
        aliases = language_aliases(label)
        two = sorted(a for a in aliases if len(a) == 2)
        code = two[0] if two else label.split('_')[0].split('-')[0].lower()
        if code not in codes:
            codes.append(code)
    return codes


def _steps(cfg: Dict[str, Any], models: Dict[str, Any]) -> List[Tuple[str, str]]:
    steps: List[Tuple[str, str]] = []

    def add(name: str, detail: str = '') -> None:
        steps.append((name, detail))

    if cfg.get('clean_html'):
        add('HTML stripping')
    add('Unicode NFKC normalization and whitespace cleanup')
    if cfg.get('redact_secrets'):
        add('Secrets redaction', 'API keys, tokens, private keys, connection strings')
    if cfg.get('remove_pii'):
        mode = ('pseudonymized' if cfg.get('pii_pseudonymize')
                else 'masked' if cfg.get('pii_mask') else 'replaced with [PII_*] tokens')
        detail = f"regex detectors with validators, {mode}"
        if 'pii_ner' in models:
            ner = models['pii_ner']
            detail += f"; NER: {ner['backend']} ({ner['model']})"
        add('PII redaction', detail)
    gates = [f"{k}={cfg[k]}" for k in ('min_chars', 'max_chars', 'min_words',
                                       'min_unique_ratio', 'min_ascii_ratio') if cfg.get(k)]
    if gates:
        add('Quality gates', ', '.join(gates))
    if cfg.get('quality_rules'):
        add('Pretraining rule filters', ', '.join(cfg['quality_rules']))
    if 'language_id' in models:
        lid = models['language_id']
        add('Language filter', f"keep {', '.join(cfg.get('lang_filter') or [])}; "
                               f"{lid['backend']} ({lid['model']}), "
                               f"confidence >= {cfg.get('lang_confidence')}")
    if cfg.get('validate_chat'):
        add('Chat validation', 'lenient' if cfg.get('chat_lenient') else 'strict')
    if 'decontamination' in models:
        d = models['decontamination']
        add('Benchmark decontamination',
            f"{d['ngram']}-gram overlap with {', '.join(d['benchmarks']) or 'reference files'}")
    if 'quality_scorer' in models:
        q = models['quality_scorer']
        bar = []
        if cfg.get('quality_min_score') is not None:
            bar.append(f"min score {cfg['quality_min_score']}")
        if cfg.get('keep_top_percent') is not None:
            bar.append(f"top {cfg['keep_top_percent']}%")
        add('Quality scoring', f"{q['backend']} ({q['model']})" +
            (f"; {', '.join(bar)}" if bar else ''))
    if cfg.get('sample') is not None:
        add('Sampling', f"{cfg['sample']} (content hash, seed {cfg.get('seed')})")
    if cfg.get('deduplicate') and not (cfg.get('fuzzy_dedup') or cfg.get('semantic_dedup')):
        add('Exact deduplication', 'SHA-256 of the record' +
            (f" fields {', '.join(cfg['dedup_fields'])}" if cfg.get('dedup_fields') else ''))
    if cfg.get('fuzzy_dedup'):
        add('Near-duplicate removal',
            f"MinHash LSH, Jaccard >= {cfg.get('fuzzy_threshold')} on word 3-shingles")
    if cfg.get('semantic_dedup'):
        s = models.get('semantic_dedup', {})
        add('Semantic deduplication',
            f"cosine >= {cfg.get('semantic_threshold')} ({s.get('model')}, {s.get('index')})")
    if cfg.get('format_chatml'):
        add('Formatting', 'ChatML messages')
    elif cfg.get('format_instruct'):
        add('Formatting', 'instruction/input/output')
    return steps


def _table(rows: List[Tuple[str, Any]], header: Tuple[str, str]) -> List[str]:
    lines = [f"| {header[0]} | {header[1]} |", "|---|---:|"]
    lines += [f"| {k} | {v:,} |" if isinstance(v, int) else f"| {k} | {v} |" for k, v in rows]
    return lines


def render_dataset_card(manifest: Dict[str, Any]) -> str:
    counts: Dict[str, Any] = manifest.get('counts', {})
    cfg: Dict[str, Any] = manifest.get('config', {})
    models: Dict[str, Any] = manifest.get('models', {})
    tool = manifest.get('tool', {})
    kept = int(counts.get('kept', 0))

    tags = ['llm-sanitizer-pro']
    if cfg.get('remove_pii'):
        tags.append('pii-redacted')
    if cfg.get('deduplicate') or cfg.get('fuzzy_dedup') or cfg.get('semantic_dedup'):
        tags.append('deduplicated')
    if 'decontamination' in models:
        tags.append('decontaminated')
    front = ['---', 'license: other  # TODO: dataset owner', 'size_categories:',
             f'- {size_category(kept)}']
    langs = _languages(counts)
    if langs:
        front += ['language:'] + [f'- {code}' for code in langs]
    front += ['tags:'] + [f'- {t}' for t in tags] + ['---']

    out = front + [
        '', '# Dataset Card', '', TODO,
        '', '## Dataset Summary', '', TODO,
        '', '## Source Data', '', TODO,
        '', '## Processing',
        '',
        f"Processed with [llm-sanitizer-pro](https://github.com/Yog-Sotho/LLM-Sanitizer-Pro) "
        f"{tool.get('version', '')} on {manifest.get('run', {}).get('finished_at', '')} "
        f"(config SHA-256 `{manifest.get('config_sha256', '')[:16]}…`).",
        '',
    ]
    out += [f"{i}. **{name}**" + (f": {detail}" if detail else '')
            for i, (name, detail) in enumerate(_steps(cfg, models), 1)]

    funnel = [('Input records', counts.get('total', 0))]
    for key, label in [('malformed', 'Malformed'), ('filtered_quality', 'Quality gates'),
                       ('filtered_rules', 'Rule filters'), ('filtered_language', 'Language'),
                       ('filtered_require', 'Missing required fields'),
                       ('filtered_code', 'Code'), ('filtered_profanity', 'Profanity'),
                       ('filtered_chat_invalid', 'Invalid chat'),
                       ('filtered_contaminated', 'Benchmark contamination'),
                       ('filtered_low_score', 'Low quality score'),
                       ('sampled_out', 'Sampled out'), ('deduplicated', 'Duplicates')]:
        if counts.get(key):
            funnel.append((f"− {label}", counts[key]))
    funnel.append(('**Kept**', f"**{kept:,}** ({counts.get('kept_pct', 0)}%)"))
    out += ['', '### Records', ''] + _table(funnel, ('Stage', 'Records'))

    if counts.get('pii_redactions'):
        out += ['', '### Redactions', '',
                'Values detected and replaced (counts only):', '']
        out += _table(sorted(counts['pii_redactions'].items(), key=lambda x: -x[1]),
                      ('Kind', 'Replaced'))
    if counts.get('contaminated_by'):
        out += ['', '### Decontamination', '']
        out += _table(list(counts['contaminated_by'].items()), ('Benchmark', 'Records removed'))
    if counts.get('language_distribution'):
        out += ['', '### Languages (kept records)', '']
        out += _table(list(counts['language_distribution'].items())[:20], ('Language', 'Records'))

    if models:
        out += ['', '### Models and reference data', '', '```json']
        out += json.dumps(models, indent=2).splitlines() + ['```']
    files = [('input', f) for f in manifest.get('inputs', [])] + \
            [('output', f) for f in manifest.get('outputs', [])]
    if files:
        out += ['', '### Files', '', '| Role | File | Bytes | SHA-256 |', '|---|---|---:|---|']
        for role, f in files:
            name = f.get('path') or f.get('uri', '')
            out.append(f"| {role} | `{name}` | {f.get('bytes', '')} | "
                       f"`{f.get('sha256', '')}` |")
    command = manifest.get('run', {}).get('command')
    if command:
        out += ['', '### Reproduce', '', '```bash', 'sanitize ' + ' '.join(command[1:]), '```']
    out += [
        '', '## Limitations', '',
        'PII and secret detection is automated. Regex detectors are high-precision but do not '
        'find every name, address or identifier written in free text; review the data before '
        'release if it may contain personal information. Rule filters and quality scorers '
        'were tuned on English web text.',
        '', TODO, '',
    ]
    return '\n'.join(out)
