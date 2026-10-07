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
import math
import re
from collections import Counter
from typing import Callable, Dict, List, Match, Optional, Tuple

from sanitizer_pro.pii import (
    _GATES, _TRIMMERS, _VALIDATORS, Gate, PseudoRegistry, _trim_url, apply_patterns,
    combined_patterns,
)

# Each entry: (compiled regex, replacement token, kind). Kinds are stable
# identifiers surfaced in stats/audit reports. Patterns are deliberately
# anchored to provider-documented shapes (fixed prefixes, exact lengths). A
# named group 'secret' redacts only the value and keeps its variable name.
_PRIVATE_KEY_RE = re.compile(
    # Any PEM/armored private key: RSA, EC, DSA, OPENSSH, ENCRYPTED, PGP … BLOCK.
    # A truncated block (no END line) still leaks key material, so it is
    # redacted to the end of the text.
    r'-----BEGIN ((?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?)-----'
    r'.*?(?:-----END \1-----|\Z)', re.DOTALL)
_CONNECTION_RE = re.compile(
    # postgres://user:pass@host/db — stops at whitespace/quotes/brackets;
    # trailing punctuation is trimmed. Commas stay (multi-host MongoDB URIs).
    r'\b(?:postgres(?:ql)?|mysql|mariadb|mongodb(?:\+srv)?|redis|rediss|amqps?|mssql)://'
    r'[^\s:/@\'"`<>]+:[^\s:/@\'"`<>]+@[^\s\'"`<>]+')
_GENERIC_RE = re.compile(
    r'(?i)\b(?:api[_-]?key|api[_-]?secret|app[_-]?secret|secret[_-]?key|private[_-]?key'
    r'|access[_-]?token|auth[_-]?token|client[_-]?secret|password|passwd)'
    r'\s*[:=]\s*(?P<q>["\']?)(?P<secret>[A-Za-z0-9_\-./+=]{16,})(?P=q)')

_SECRET_PATTERNS: List[Tuple[re.Pattern[str], str, str]] = [
    (_PRIVATE_KEY_RE, '[SECRET_PRIVATE_KEY]', 'private_key'),
    # AWS access key ids, and secret access keys next to their variable name
    (re.compile(r'\b(?:AKIA|ASIA)[0-9A-Z]{16}\b'), '[SECRET_AWS_KEY]', 'aws_access_key'),
    (re.compile(r'(?i)\baws_?secret_?(?:access_?)?key\b["\']?\s*[:=]\s*["\']?'
                r'(?P<secret>[A-Za-z0-9/+=]{40})(?![A-Za-z0-9/+=])'),
     '[SECRET_AWS_SECRET_KEY]', 'aws_secret_key'),
    # GitHub tokens (ghp_, gho_, ghu_, ghs_, ghr_ + 36, or fine-grained pat)
    (re.compile(r'\bgh[posru]_[A-Za-z0-9]{36,}\b'), '[SECRET_GITHUB_TOKEN]', 'github_token'),
    (re.compile(r'\bgithub_pat_[A-Za-z0-9_]{60,}\b'), '[SECRET_GITHUB_TOKEN]', 'github_token'),
    # GitLab personal/project/deploy/runner/trigger tokens
    (re.compile(r'\bgl(?:pat|ptt|dt|rt|soat|cbt|ft)-[A-Za-z0-9_\-]{20,}'),
     '[SECRET_GITLAB_TOKEN]', 'gitlab_token'),
    # Hugging Face user and org tokens
    (re.compile(r'\b(?:hf|api_org)_[A-Za-z0-9]{30,}\b'), '[SECRET_HF_TOKEN]', 'huggingface_token'),
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
    # npm and PyPI publish tokens
    (re.compile(r'\bnpm_[A-Za-z0-9]{36}\b'), '[SECRET_NPM_TOKEN]', 'npm_token'),
    (re.compile(r'\bpypi-AgEIcHlwaS5vcmc[A-Za-z0-9_-]{50,}'), '[SECRET_PYPI_TOKEN]', 'pypi_token'),
    # Azure storage account keys inside connection strings
    (re.compile(r'(?i)\bAccountKey=(?P<secret>[A-Za-z0-9+/]{40,}={0,2})'),
     '[SECRET_AZURE_KEY]', 'azure_storage_key'),
    # JSON Web Tokens
    (re.compile(r'\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b'),
     '[SECRET_JWT]', 'jwt'),
    # Generic bearer token in Authorization headers
    (re.compile(r'(?i)\bBearer\s+[A-Za-z0-9._~+/-]{20,}={0,2}'), '[SECRET_BEARER_TOKEN]', 'bearer_token'),
    # Connection strings with inline credentials (postgres://user:pass@host/db)
    (_CONNECTION_RE, '[SECRET_CONNECTION_STRING]', 'connection_string'),
    # Generic assignment: api_key = "…", password: … (entropy-gated)
    (_GENERIC_RE, '[SECRET_GENERIC]', 'generic_secret'),
]


def _shannon_entropy(s: str) -> float:
    counts = Counter(s)
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in counts.values())


# Documentation/config-template values ("your-api-key-here", "changeme").
_PLACEHOLDER_WORDS = ('your', 'here', 'example', 'placeholder', 'changeme', 'xxxx',
                      'dummy', 'redacted', 'insert', 'replace')


def _is_generic_secret(text: str, m: Match[str]) -> bool:
    """Assigned values must look random and not be template placeholders:
    quoted ones need >= 3 bits/char of
    entropy, unquoted ones (identifiers such as `password = get_password_from_env`
    are common) >= 3.5 bits/char plus both letters and digits."""
    value = m.group('secret')
    if any(word in value.lower() for word in _PLACEHOLDER_WORDS):
        return False
    entropy = _shannon_entropy(value)
    if m.group('q'):
        return entropy >= 3.0
    return (entropy >= 3.5 and any(c.isdigit() for c in value)
            and any(c.isalpha() for c in value))


_VALIDATORS[_GENERIC_RE] = _is_generic_secret
_TRIMMERS[_CONNECTION_RE] = _trim_url

# Prefilters (see pii.Gate): the literal every pattern's match must contain.
_SECRET_GATES = [
    Gate(('-----BEGIN ',)),
    Gate(('AKIA', 'ASIA')),
    Gate.ignorecase('aws'),
    Gate(('ghp_', 'gho_', 'ghu_', 'ghs_', 'ghr_')),
    Gate(('github_pat_',)),
    Gate(('glpat-', 'glptt-', 'gldt-', 'glrt-', 'glsoat-', 'glcbt-', 'glft-')),
    Gate(('hf_', 'api_org_')),
    Gate(('sk-ant-',)),
    Gate(('sk-',)),
    Gate(('AIza',)),
    Gate(('xox',)),
    Gate(('k_live_', 'k_test_')),
    Gate(('SK',)),
    Gate(('SG.',)),
    Gate(('npm_',)),
    Gate(('pypi-AgEIcHlwaS5vcmc',)),
    Gate.ignorecase('accountkey='),
    Gate(('eyJ',)),
    Gate.ignorecase('bearer'),
    Gate(('://',)),
    # every keyword alternative contains one of these
    Gate.ignorecase('api', 'secret', 'private', 'access', 'auth', 'passw'),
]
assert len(_SECRET_GATES) == len(_SECRET_PATTERNS)
_GATES.update((p, g) for (p, _, _), g in zip(_SECRET_PATTERNS, _SECRET_GATES))

SECRET_KINDS = tuple(dict.fromkeys(kind for _, _, kind in _SECRET_PATTERNS))


def _mask_tail(m: Match[str], keep: int = 4) -> str:
    """Keep a stable prefix marker and the last `keep` chars for correlation."""
    s = m.group('secret') if 'secret' in m.re.groupindex and m.group('secret') else m.group(0)
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
    registry.add_templates(_SECRET_PSEUDO_TEMPLATES)


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
        text, combined_patterns(_SECRET_PATTERNS, extra_patterns),
        mask=mask, pseudo_registry=pseudo_registry, counters=counters,
        mask_fns=_SECRET_MASK_FNS)


def contains_secret(text: str) -> bool:
    return any(p.search(text) for p, _, _ in _SECRET_PATTERNS)
