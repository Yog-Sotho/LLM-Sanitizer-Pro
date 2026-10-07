"""Compare two runs: `sanitize diff A B`.

A and B are run manifests (--manifest) or stats files (--stats-file).
Prints what changed in the per-stage counts (with breakdowns such as PII
kinds, rule failures and contamination by benchmark) and, for manifests,
in the configuration, tool and dependency versions, models, and input and
output files (by SHA-256).

    sanitize diff old/run.json new/run.json
    sanitize diff a.stats.json b.stats.json --json
    sanitize diff baseline.json candidate.json --fail-on-change   # CI gate
"""
import argparse
import json
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sanitizer_pro.manifest import load_run

Change = Tuple[str, Any, Any]          # (key, A value, B value)


def _numeric(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def diff_counts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Scalar count changes and per-key changes inside breakdown dicts."""
    scalars: List[Change] = []
    breakdowns: Dict[str, List[Change]] = {}
    for key in sorted(set(a) | set(b)):
        if key == 'version':
            continue
        va, vb = a.get(key), b.get(key)
        if isinstance(va, dict) or isinstance(vb, dict):
            da, db = va or {}, vb or {}
            changes = [(k, da.get(k, 0), db.get(k, 0)) for k in sorted(set(da) | set(db))
                       if da.get(k, 0) != db.get(k, 0)]
            if changes:
                breakdowns[key] = changes
        elif va != vb:
            scalars.append((key, va, vb))
    return {'scalars': scalars, 'breakdowns': breakdowns}


def _diff_dicts(a: Dict[str, Any], b: Dict[str, Any]) -> List[Change]:
    return [(k, a.get(k), b.get(k)) for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]


def _files(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {e.get('path') or e.get('uri', '?'): e.get('sha256') or e.get('bytes')
            for e in entries}


def diff_runs(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {'counts': diff_counts(a.get('counts', {}), b.get('counts', {}))}
    if a.get('tool', {}).get('version') != b.get('tool', {}).get('version'):
        result['tool_version'] = (a.get('tool', {}).get('version'),
                                  b.get('tool', {}).get('version'))
    for section in ('config', 'dependencies', 'models'):
        if section in a and section in b:
            changes = _diff_dicts(a[section], b[section])
            if changes:
                result[section] = changes
    for section in ('inputs', 'outputs'):
        if section in a and section in b:
            fa, fb = _files(a[section]), _files(b[section])
            if fa != fb:
                if sorted(fa.values(), key=str) == sorted(fb.values(), key=str):
                    result[section] = 'same content, different paths'
                else:
                    result[section] = _diff_dicts(fa, fb)
    return result


def has_changes(result: Dict[str, Any]) -> bool:
    counts = result['counts']
    return bool(counts['scalars'] or counts['breakdowns'] or len(result) > 1)


def _fmt_delta(va: Any, vb: Any) -> str:
    if _numeric(va) and _numeric(vb):
        delta = vb - va
        pct = f" ({delta / va * 100:+.1f}%)" if va else ''
        return f"{va:,} → {vb:,}  {delta:+,}{pct}" if isinstance(delta, int) \
            else f"{va} → {vb}  {delta:+.4g}"
    return f"{va!r} → {vb!r}"


def format_text(result: Dict[str, Any], name_a: str, name_b: str) -> str:
    lines = [f"A: {name_a}", f"B: {name_b}", '']
    if not has_changes(result):
        return '\n'.join(lines + ['No differences.'])
    if 'tool_version' in result:
        lines.append(f"tool version: {result['tool_version'][0]} → {result['tool_version'][1]}")
    counts = result['counts']
    if counts['scalars']:
        lines += ['', 'Counts:']
        lines += [f"  {k:<24}{_fmt_delta(va, vb)}" for k, va, vb in counts['scalars']]
    for key, changes in counts['breakdowns'].items():
        lines += ['', f"{key}:"]
        lines += [f"  {k:<24}{_fmt_delta(va, vb)}" for k, va, vb in changes]
    for section in ('config', 'models', 'dependencies', 'inputs', 'outputs'):
        value = result.get(section)
        if not value:
            continue
        lines += ['', f"{section}:"]
        if isinstance(value, str):
            lines.append(f"  {value}")
        else:
            lines += [f"  {k}: {json.dumps(va)} → {json.dumps(vb)}" for k, va, vb in value]
    return '\n'.join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog='sanitize diff', description='Compare two run manifests or stats files.')
    ap.add_argument('a', help='Baseline run (manifest or stats JSON).')
    ap.add_argument('b', help='Run to compare against the baseline.')
    ap.add_argument('--json', action='store_true', help='Machine-readable output.')
    ap.add_argument('--fail-on-change', action='store_true',
                    help='Exit with status 1 when anything differs.')
    args = ap.parse_args(argv)
    try:
        result = diff_runs(load_run(args.a), load_run(args.b))
    except (OSError, ValueError) as exc:
        print(f"sanitize diff: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps({'a': args.a, 'b': args.b, 'changed': has_changes(result),
                          **result}, indent=2, default=str))
    else:
        print(format_text(result, args.a, args.b))
    return 1 if args.fail_on_change and has_changes(result) else 0
