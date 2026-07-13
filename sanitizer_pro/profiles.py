"""Named presets that bundle sensible flag combinations for common jobs.

A profile is a set of ``{arg_dest: value}`` defaults applied only to flags the
user did not set explicitly, so ``--profile fine-tune --min-chars 5`` keeps the
profile's other choices while overriding just ``min-chars``. Precedence is:
explicit CLI flags > --config file > --profile > built-in defaults.

All profile defaults use dependency-free features (regex PII, secrets, exact
dedup, heuristic gates) so a profile never fails for lack of an optional
package. Layer optional features (--pii-ner, --semantic-dedup, --decontaminate)
on top as needed.
"""
from typing import Any, Dict

# Each profile: human note + the arg defaults it sets.
PROFILES: Dict[str, Dict[str, Any]] = {
    'fine-tune': {
        '_note': 'Instruction / chat SFT data: redact PII + secrets, exact dedup, '
                 'HTML cleanup, modest length floor.',
        'remove_pii': True,
        'redact_secrets': True,
        'deduplicate': True,
        'dedup_normalize': True,
        'clean_html': True,
        'min_chars': 20,
        'min_words': 5,
    },
    'pretrain': {
        '_note': 'Large text corpora: aggressive dedup + quality gates for '
                 'document-length text, secrets redaction, HTML cleanup.',
        'redact_secrets': True,
        'deduplicate': True,
        'clean_html': True,
        'min_chars': 200,
        'min_words': 50,
        'min_unique_ratio': 0.35,
        'quality_min_score': 0.3,
    },
    'rag': {
        '_note': 'Retrieval chunks: redact PII + secrets, dedup near-identical '
                 'passages, keep chunks reasonably sized.',
        'remove_pii': True,
        'redact_secrets': True,
        'deduplicate': True,
        'dedup_normalize': True,
        'clean_html': True,
        'min_chars': 50,
        'min_words': 10,
    },
}

PROFILE_NAMES = tuple(PROFILES)


def profile_settings(name: str) -> Dict[str, Any]:
    """Return the {dest: value} defaults for a profile (without the note)."""
    return {k: v for k, v in PROFILES[name].items() if not k.startswith('_')}


def describe_profiles() -> str:
    lines = []
    for name, spec in PROFILES.items():
        note = spec.get('_note', '')
        settings = ', '.join(f"{k}={v}" for k, v in spec.items() if not k.startswith('_'))
        lines.append(f"{name}\n  {note}\n  sets: {settings}")
    return '\n\n'.join(lines)
