"""Audit report generation: a self-contained HTML artifact per run.

Turns the run's statistics, PII redaction counts, and collected samples into a
single HTML file with no external assets — suitable for archiving next to the
output dataset, attaching to a compliance ticket, or rendering in a SaaS UI.
"""
import html
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

_MAX_SAMPLES_PER_REASON = 5
_SAMPLE_TRUNCATE = 400

_REASON_LABELS = {
    'malformed': 'Malformed (not a JSON object)',
    'quality': 'Quality gates (length / words / ratios)',
    'language': 'Language filter',
    'require_fields': 'Missing required fields',
    'code': 'Code detection',
    'profanity': 'Profanity filter',
    'chat': 'Chat validation',
    'contaminated': 'Benchmark contamination',
    'low_score': 'Quality score',
    'duplicate': 'Deduplication',
    'sampled_out': 'Random sampling',
}

_PII_LABELS = {
    'email': 'Email addresses', 'phone': 'Phone numbers', 'card': 'Card numbers',
    'ssn': 'SSNs', 'ip': 'IP addresses', 'url': 'URLs', 'custom': 'Custom patterns',
    'person': 'Person names (NER)', 'location': 'Locations (NER)', 'org': 'Organizations (NER)',
}


class AuditSampleCollector:
    """Bounded reservoirs of example records for the audit report."""

    def __init__(self, max_per_reason: int = _MAX_SAMPLES_PER_REASON) -> None:
        self.max_per_reason = max_per_reason
        self.dropped: Dict[str, List[str]] = {}
        self.pii_diffs: List[Tuple[str, str]] = []

    @staticmethod
    def _snippet(record: Any) -> str:
        try:
            s = json.dumps(record, ensure_ascii=False, default=str)
        except Exception:
            s = repr(record)
        return s[:_SAMPLE_TRUNCATE] + ('…' if len(s) > _SAMPLE_TRUNCATE else '')

    def add_dropped(self, reason: str, record: Any) -> None:
        bucket = self.dropped.setdefault(reason, [])
        if len(bucket) < self.max_per_reason:
            bucket.append(self._snippet(record))

    def add_pii_diff(self, before: Any, after: Any) -> None:
        if len(self.pii_diffs) >= self.max_per_reason:
            return
        pair = (self._snippet(before), self._snippet(after))
        if pair not in self.pii_diffs:  # duplicates are redacted before dedup drops them
            self.pii_diffs.append(pair)

    @property
    def wants_pii_diffs(self) -> bool:
        return len(self.pii_diffs) < self.max_per_reason


def _e(v: Any) -> str:
    return html.escape(str(v), quote=True)


def _bar_rows(items: List[Tuple[str, int]], total: int, color_var: str) -> str:
    if not items:
        return '<p class="empty">None.</p>'
    peak = max(n for _, n in items) or 1
    rows = []
    for label, n in items:
        width = max(0.6, n / peak * 100)
        pct = f"{n / total * 100:.1f}%" if total else '—'
        rows.append(
            f'<div class="row" title="{_e(label)}: {n:,} ({pct})">'
            f'<span class="rlabel">{_e(label)}</span>'
            f'<span class="track"><span class="bar" style="width:{width:.1f}%;'
            f'background:var({color_var})"></span></span>'
            f'<span class="rval">{n:,}</span><span class="rpct">{pct}</span></div>')
    return '<div class="chart">' + ''.join(rows) + '</div>'


def _drop_items(stats: Dict[str, Any]) -> List[Tuple[str, int]]:
    mapping = [
        ('malformed', stats.get('malformed', 0)),
        ('quality', stats.get('filtered_quality', 0)),
        ('language', stats.get('filtered_language', 0)),
        ('require_fields', stats.get('filtered_require', 0)),
        ('code', stats.get('filtered_code', 0)),
        ('profanity', stats.get('filtered_profanity', 0)),
        ('chat', stats.get('filtered_chat_invalid', 0)),
        ('contaminated', stats.get('filtered_contaminated', 0)),
        ('low_score', stats.get('filtered_low_score', 0)),
        ('duplicate', stats.get('deduplicated', 0)),
        ('sampled_out', stats.get('sampled_out', 0)),
    ]
    return [(_REASON_LABELS[k], v) for k, v in mapping if v > 0]


def generate_report_html(
    stats: Dict[str, Any],
    samples: Optional[AuditSampleCollector] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> str:
    meta = meta or {}
    total = stats.get('total', 0)
    kept = stats.get('kept', 0)
    dropped_total = total - kept
    pii = stats.get('pii_redactions', {}) or {}
    pii_total = sum(pii.values())
    ts = meta.get('timestamp') or datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')

    meta_rows = ''.join(
        f'<div class="meta-row"><span>{_e(k)}</span><span>{_e(v)}</span></div>'
        for k, v in meta.items() if k != 'timestamp' and v not in (None, ''))

    tiles = f"""
    <div class="tiles">
      <div class="tile"><div class="tval">{total:,}</div><div class="tlabel">Records processed</div></div>
      <div class="tile"><div class="tval">{kept:,}</div><div class="tlabel">Kept ({stats.get('kept_pct', 0):.1f}%)</div></div>
      <div class="tile"><div class="tval">{dropped_total:,}</div><div class="tlabel">Removed</div></div>
      <div class="tile"><div class="tval">{pii_total:,}</div><div class="tlabel">PII redactions</div></div>
    </div>"""

    sections = [f"""
    <section><h2>Why records were removed</h2>
    {_bar_rows(_drop_items(stats), total, '--series-1')}</section>"""]

    if pii:
        pii_items = [(_PII_LABELS.get(k, k), v) for k, v in pii.items()]
        sections.append(f"""
        <section><h2>PII redactions by type</h2>
        {_bar_rows(pii_items, pii_total, '--series-2')}
        <p class="note">Counts are individual redaction events across all record fields.</p></section>""")

    chat_reasons = stats.get('chat_invalid_reasons') or {}
    if chat_reasons:
        items = [(k.replace('_', ' '), v) for k, v in chat_reasons.items()]
        sections.append(f"""
        <section><h2>Chat validation failures</h2>
        {_bar_rows(items, stats.get('filtered_chat_invalid', 0), '--series-1')}</section>""")

    score_hist = stats.get('quality_score_histogram') or {}
    if score_hist:
        mean = stats.get('quality_score_mean')
        items = [(f"{k} – {float(k) + 0.1:.1f}", v) for k, v in score_hist.items()]
        sections.append(f"""
        <section><h2>Quality score distribution <span class="hmeta">mean {mean}</span></h2>
        {_bar_rows(items, sum(score_hist.values()), '--series-1')}</section>""")

    lang_dist = stats.get('language_distribution') or {}
    if lang_dist:
        items = list(lang_dist.items())[:10]
        sections.append(f"""
        <section><h2>Language distribution (kept records)</h2>
        {_bar_rows(items, sum(lang_dist.values()), '--series-1')}</section>""")

    if samples is not None and samples.pii_diffs:
        diffs = ''.join(
            f'<div class="diff"><div class="before"><span class="dtag">before</span>'
            f'<code>{_e(b)}</code></div><div class="after"><span class="dtag">after</span>'
            f'<code>{_e(a)}</code></div></div>'
            for b, a in samples.pii_diffs)
        sections.append(f"""
        <section><h2>PII redaction samples</h2>{diffs}</section>""")

    if samples is not None and samples.dropped:
        blocks = []
        for reason, examples in samples.dropped.items():
            items = ''.join(f'<li><code>{_e(s)}</code></li>' for s in examples)
            blocks.append(
                f'<details><summary>{_e(_REASON_LABELS.get(reason, reason))} '
                f'({len(examples)} sample{"s" if len(examples) != 1 else ""})</summary>'
                f'<ul>{items}</ul></details>')
        sections.append(f"""
        <section><h2>Samples of removed records</h2>{''.join(blocks)}</section>""")

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sanitization Audit Report</title>
<style>
:root {{
  --surface: #fcfcfb; --plane: #f9f9f7;
  --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --hairline: #e1e0d9; --border: rgba(11,11,11,0.10);
  --series-1: #2a78d6; --series-2: #1baf7a; --track: #f0efec;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --surface: #1a1a19; --plane: #0d0d0d;
    --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
    --hairline: #2c2c2a; --border: rgba(255,255,255,0.10);
    --series-1: #3987e5; --series-2: #199e70; --track: #232322;
  }}
}}
* {{ box-sizing: border-box; margin: 0; }}
body {{ font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  background: var(--plane); color: var(--ink); padding: 24px; line-height: 1.5; }}
main {{ max-width: 860px; margin: 0 auto; }}
header h1 {{ font-size: 20px; font-weight: 650; }}
header .sub {{ color: var(--muted); font-size: 13px; margin-top: 2px; }}
.meta {{ margin-top: 12px; border: 1px solid var(--border); border-radius: 8px;
  background: var(--surface); padding: 8px 14px; font-size: 13px; }}
.meta-row {{ display: flex; justify-content: space-between; gap: 16px;
  padding: 3px 0; border-bottom: 1px solid var(--hairline); }}
.meta-row:last-child {{ border-bottom: none; }}
.meta-row span:first-child {{ color: var(--muted); }}
.meta-row span:last-child {{ color: var(--ink-2); text-align: right; overflow-wrap: anywhere; }}
.tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 10px; margin: 18px 0; }}
.tile {{ background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 14px 16px; }}
.tval {{ font-size: 26px; font-weight: 650; }}
.tlabel {{ font-size: 12px; color: var(--muted); margin-top: 2px; }}
section {{ background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 16px 18px; margin-bottom: 14px; }}
h2 {{ font-size: 14px; font-weight: 650; margin-bottom: 12px; }}
.hmeta {{ font-weight: 400; color: var(--muted); font-size: 12px; margin-left: 6px; }}
.chart {{ display: grid; gap: 6px; }}
.row {{ display: grid; grid-template-columns: minmax(140px, 220px) 1fr 70px 52px;
  align-items: center; gap: 10px; font-size: 13px; }}
.rlabel {{ color: var(--ink-2); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.track {{ display: block; height: 12px; background: var(--track); border-radius: 4px; overflow: hidden; }}
.bar {{ display: block; height: 100%; border-radius: 0 4px 4px 0; }}
.rval {{ text-align: right; font-variant-numeric: tabular-nums; }}
.rpct {{ text-align: right; color: var(--muted); font-variant-numeric: tabular-nums; font-size: 12px; }}
.note, .empty {{ color: var(--muted); font-size: 12px; margin-top: 10px; }}
code {{ font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-size: 12px;
  overflow-wrap: anywhere; color: var(--ink-2); }}
.diff {{ border: 1px solid var(--hairline); border-radius: 6px; margin-bottom: 8px; overflow: hidden; }}
.diff > div {{ padding: 8px 12px; }}
.diff .before {{ border-bottom: 1px solid var(--hairline); }}
.dtag {{ display: inline-block; font-size: 10px; text-transform: uppercase; letter-spacing: 0.06em;
  color: var(--muted); margin-right: 8px; min-width: 44px; }}
details {{ border-bottom: 1px solid var(--hairline); padding: 8px 0; }}
details:last-child {{ border-bottom: none; }}
summary {{ cursor: pointer; font-size: 13px; color: var(--ink-2); }}
details ul {{ margin: 8px 0 4px 18px; display: grid; gap: 6px; }}
footer {{ color: var(--muted); font-size: 12px; text-align: center; margin: 20px 0 8px; }}
</style></head><body><main>
<header>
  <h1>Sanitization Audit Report</h1>
  <div class="sub">{_e(ts)}</div>
  <div class="meta">{meta_rows}</div>
</header>
{tiles}
{''.join(sections)}
<footer>Generated by LLM Dataset Sanitizer PRO v{_e(meta.get('version', '3.0'))}</footer>
</main></body></html>
"""
