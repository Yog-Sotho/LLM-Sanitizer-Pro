"""Secret / credential detection and redaction.

Training data scraped from code, logs, or support tickets routinely carries
live credentials — API keys, tokens, private keys, connection strings. Leaking
those into a fine-tuning corpus is both a security incident and a way to teach
a model to emit real secrets. This module redacts them with high-precision
patterns keyed to each provider's documented token shape, so false positives on
ordinary prose stay near zero.

Redaction reuses the PII machinery: tokens (``[SECRET_AWS_KEY]``), masking
(last 4 chars kept), or stable pseudonymization via the shared registry.
"""
import re
from typing import Callable, Dict, List, Match, Optional, Tuple

from sanitizer_pro.pii import PseudoRegistry, apply_patterns

# Each entry: (compiled regex, replacement token, kind). Kinds are stable
# identifiers surfaced in stats/audit reports. Patterns are deliberately
# anchored to provider-documented shapes (fixed prefixes, exact lengths).
_SECRET_PATTERNS: List[Tuple[re.Pattern[str], str, str]] = [
    # Private keys (PEM blocks) — match the whole block, any key type.
    (re.compile(
        r'-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----'
        r'.*?-----END (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----',
        re.DOTALL), '[SECRET_PRIVATE_KEY]', 'private_key'),
    # AWS
    (re.compile(r'\b(?:AKIA|ASIA)[0-9A-Z]{16}\b'), '[SECRET_AWS_KEY]', 'aws_access_key'),
    # GitHub tokens (ghp_, gho_, ghu_, ghs_, ghr_ + 36, or fine-grained pat)
    (re.compile(r'\bgh[posru]_[A-Za-z0-9]{36,}\b'), '[SECRET_GITHUB_TOKEN]', 'github_token'),
    (re.compile(r'\bgithub_pat_[A-Za-z0-9_]{60,}\b'), '[SECRET_GITHUB_TOKEN]', 'github_token'),
    # Anthropic before OpenAI so sk-ant-… isn't swallowed by the generic sk- rule
    (re.compile(r'\bsk-ant-[A-Za-z0-9_-]{20,}\b'), '[SECRET_ANTHROPIC_KEY]', 'anthropic_key'),
    (re.compile(r'\bsk-(?!ant-)(?:proj-)?[A-Za-z0-9_-]{20,}\b'), '[SECRET_OPENAI_KEY]', 'openai_key'),
    # Google API keys
    (re.compile(r'\bAIza[0-9A-Za-z_-]{35}\b'), '[SECRET_GOOGLE_KEY]', 'google_api_key'),
    # Slack tokens
    (re.compile(r'\bxox[baprs]-[A-Za-z0-9-]{10,}\b'), '[SECRET_SLACK_TOKEN]', 'slack_token'),
    # Stripe
    (re.compile(r'\b[rs]k_(?:live|test)_[A-Za-z0-9]{20,}\b'), '[SECRET_STRIPE_KEY]', 'stripe_key'),
    # Twilio
    (re.compile(r'\bSK[0-9a-fA-F]{32}\b'), '[SECRET_TWILIO_KEY]', 'twilio_key'),
    # SendGrid
    (re.compile(r'\bSG\.[A-Za-z0-9_-]{22}\.[A-Za-z0-9_-]{43}\b'), '[SECRET_SENDGRID_KEY]', 'sendgrid_key'),
    # JSON Web Tokens
    (re.compile(r'\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b'),
     '[SECRET_JWT]', 'jwt'),
    # Slack incoming-webhook / generic bearer in Authorization headers
    (re.compile(r'(?i)\bBearer\s+[A-Za-z0-9._~+/-]{20,}={0,2}'), '[SECRET_BEARER_TOKEN]', 'bearer_token'),
    # Connection strings with inline credentials (postgres://user:pass@host/db)
    (re.compile(r'\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp)://'
                r'[^\s:/@]+:[^\s:/@]+@[^\s]+'), '[SECRET_CONNECTION_STRING]', 'connection_string'),
    # Generic assignment: api_key = "….", token: '….' (>=16 chars, has digit)
    (re.compile(
        r'(?i)\b(?:api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|password|passwd)'
        r'\s*[:=]\s*["\']([A-Za-z0-9_\-./+=]{16,})["\']'),
     '[SECRET_GENERIC]', 'generic_secret'),
]

SECRET_KINDS = tuple(kind for _, _, kind in _SECRET_PATTERNS)


def _mask_tail(m: Match[str], keep: int = 4) -> str:
    """Keep a stable prefix marker and the last `keep` chars for correlation."""
    s = m.group(0)
    tail = s[-keep:] if len(s) > keep else ''
    return f"[SECRET…{tail}]" if tail else '[SECRET]'


_SECRET_MASK_FNS: Dict[str, Callable[[Match[str]], str]] = {
    kind: _mask_tail for kind in SECRET_KINDS if kind != 'private_key'
}

_SECRET_PSEUDO_TEMPLATES: Dict[str, str] = {
    kind: kind.upper() + '_{n:04d}' for kind in SECRET_KINDS
}


def _install_pseudo_templates(registry: PseudoRegistry) -> None:
    """Ensure the shared registry knows how to name secret kinds."""
    for kind, tmpl in _SECRET_PSEUDO_TEMPLATES.items():
        registry._TEMPLATES.setdefault(kind, tmpl)


def redact_secrets(
    text: str,
    mask: bool = False,
    pseudo_registry: Optional[PseudoRegistry] = None,
    counters: Optional[Dict[str, int]] = None,
    extra_patterns: Optional[List[Tuple[re.Pattern[str], str, str]]] = None,
) -> str:
    """Redact secrets/credentials. `counters` tallies hits per secret kind."""
    if pseudo_registry is not None:
        _install_pseudo_templates(pseudo_registry)
    return apply_patterns(
        text, _SECRET_PATTERNS + (extra_patterns or []),
        mask=mask, pseudo_registry=pseudo_registry, counters=counters,
        mask_fns=_SECRET_MASK_FNS)


def contains_secret(text: str) -> bool:
    return any(p.search(text) for p, _, _ in _SECRET_PATTERNS)
