"""PII detection, masking, pseudonymization, and safe HTML stripping."""
import html
import ipaddress
import re
import unicodedata
from typing import Any, Callable, Dict, List, Optional, Tuple

_BLOCK_TAGS = frozenset(
    'address article aside blockquote body br caption dd details dialog div dl dt '
    'fieldset figcaption figure footer form h1 h2 h3 h4 h5 h6 head header hr html li '
    'main nav ol p pre section summary table tbody tfoot thead title tr ul'.split())
_INLINE_TAGS = frozenset(
    'a abbr b bdi bdo big button cite code data del dfn em font i img input ins kbd '
    'label link mark meta meter q s samp small span strong sub sup td th time tt u var '
    'wbr center option select textarea'.split())
_TAG_NAMES = '|'.join(sorted(_BLOCK_TAGS | _INLINE_TAGS, key=len, reverse=True))
# Element bodies that are never prose. Lazy match up to the matching close tag.
_DROP_BLOCK_RE = re.compile(
    r'<(script|style|noscript|template)\b[^<>]*>.*?</\1\s*>', re.IGNORECASE | re.DOTALL)
_COMMENT_RE = re.compile(r'<!--.*?-->|<!doctype[^<>]*>|<!\[CDATA\[.*?\]\]>',
                         re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(rf'</?({_TAG_NAMES})\b[^<>]*>', re.IGNORECASE)
# Evidence that a string really is markup (and not code like `if a<b and c>d`):
# a closing tag, a void/self-closing tag, a comment, a doctype or a dropped block.
_LOOKS_LIKE_HTML_RE = re.compile(
    rf'</({_TAG_NAMES}|script|style|noscript|template)\s*>|<(br|hr|img|meta|link|input|wbr)\b[^<>]*>'
    rf'|<({_TAG_NAMES})\b[^<>]*/>|<!--|<!doctype', re.IGNORECASE)


def looks_like_html(text: str) -> bool:
    return bool(_LOOKS_LIKE_HTML_RE.search(text))


def _tag_replacement(m: re.Match[str]) -> str:
    return '\n' if m.group(1).lower() in _BLOCK_TAGS else ''


def strip_html(text: str) -> str:
    """Remove HTML markup, keeping the visible text.

    Only strings that look like HTML are touched, so code and math such as
    `x<y and y>z` or `a<b` pass through unchanged. Script/style bodies and
    comments are dropped, block-level tags become line breaks, inline tags
    vanish without inserting spaces, and character references are decoded
    (after tag removal, so escaped markup stays literal text).
    All patterns use negated character classes, so matching stays linear."""
    if not text:
        return text
    if '<' in text and looks_like_html(text):
        text = _COMMENT_RE.sub('', text)
        text = _DROP_BLOCK_RE.sub('', text)
        text = _TAG_RE.sub(_tag_replacement, text)
    # Decode entities last, so escaped markup (&lt;b&gt;) survives as text.
    return html.unescape(text) if '&' in text else text


# ---------------------------------------------------------------------------
# Validators: a regex finds candidates, a validator decides whether the match
# really is PII. They are keyed by the built-in compiled pattern, so custom
# user patterns (--pii-patterns-file) are applied exactly as written.
# ---------------------------------------------------------------------------

Validator = Callable[[str, 're.Match[str]'], bool]   # (text, match) -> is PII


def _digits(s: str) -> str:
    return re.sub(r'\D', '', s)


def luhn_valid(digits: str) -> bool:
    """True if `digits` (13-19 digits) passes the Luhn checksum."""
    if not digits.isdigit() or not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def iban_valid(iban: str) -> bool:
    """True if `iban` (spaces allowed) passes the ISO 13616 mod-97 check."""
    s = iban.replace(' ', '').upper()
    if not 15 <= len(s) <= 34 or not s[:2].isalpha() or not s[2:4].isdigit() or not s.isalnum():
        return False
    rearranged = s[4:] + s[:4]
    return int(''.join(str(int(ch, 36)) for ch in rearranged)) % 97 == 1


def _is_card(text: str, m: 're.Match[str]') -> bool:
    digits = _digits(m.group(0))
    # Issuer ranges in use start with 2-6 (Mastercard 2/5, Amex/Diners/JCB 3,
    # Visa 4, Discover/UnionPay/Maestro 6); Luhn rejects ~90% of random IDs.
    return digits[:1] in '23456' and luhn_valid(digits)


def _is_iban(text: str, m: 're.Match[str]') -> bool:
    return iban_valid(m.group(0))


def _is_ssn(text: str, m: 're.Match[str]') -> bool:
    area, group, serial = m.group(0).split('-')
    return (area not in ('000', '666') and not area.startswith('9')
            and group != '00' and serial != '0000')


def _is_intl_phone(text: str, m: 're.Match[str]') -> bool:
    return 8 <= len(_digits(m.group(0))) <= 15   # E.164 allows at most 15 digits


# "v1.2.3.4", "version 10.0.0.1", "build 4.3.2.1": software versions, not hosts.
_VERSION_CONTEXT_RE = re.compile(
    r'(?:\bv|\bversion|\bver\.?|\brelease|\bbuild|\bfirmware|\brev\.?)\s*$', re.IGNORECASE)


def _is_ip(text: str, m: 're.Match[str]') -> bool:
    try:
        ip = ipaddress.ip_address(m.group(0))
    except ValueError:
        return False
    if ip.is_loopback or ip.is_unspecified or ip.is_link_local:
        return False  # 127.0.0.1, 0.0.0.0, ::1, 169.254.x / fe80:: identify no one
    if ip.version == 6 and m.group(0).count(':') < 3:
        return False  # code such as `a[1::2]` parses as an address
    return not _VERSION_CONTEXT_RE.search(text[max(0, m.start() - 12):m.start()])


_URL_TRAILING = '.,;:!?\'"'
_URL_BRACKETS = {')': '(', ']': '[', '}': '{', '>': '<'}


def _trim_url(url: str) -> Tuple[str, str]:
    """Split trailing sentence punctuation (and unbalanced closing brackets)
    off a URL match: 'https://x.org/a).' -> ('https://x.org/a', ').')."""
    end = len(url)
    while end > 0:
        ch = url[end - 1]
        if ch in _URL_TRAILING:
            end -= 1
        elif ch in _URL_BRACKETS and url[:end].count(ch) > url[:end].count(_URL_BRACKETS[ch]):
            end -= 1
        else:
            break
    return url[:end], url[end:]


# Order matters: URLs/emails first (so their fragments aren't re-matched), then
# IBANs and cards (long digit runs) before SSNs and phones that could consume
# a part of them, then IP addresses.
_EMAIL_RE = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', re.IGNORECASE)
_URL_RE = re.compile(r'https?://\S+', re.IGNORECASE)
_WWW_RE = re.compile(r'\bwww\.\S+', re.IGNORECASE)
_IBAN_RE = re.compile(r'\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,3})?\b')
_CARD_RE = re.compile(r'(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d-])')
_SSN_RE = re.compile(r'(?<![\d-])\d{3}-\d{2}-\d{4}(?![\d-])')
# NANP needs separators: a bare 10-digit run is far more often an ID or a
# Unix timestamp than a phone number.
_PHONE_RE = re.compile(r'(?<![\d-])(?:\(\d{3}\)\s?|\d{3}[-.\s])\d{3}[-.\s]\d{4}(?![\d-])')
# '+' country code, then up to six digit groups (French pairs, '+81 3-1234-5678',
# '+1 (555) 123 4567'); the validator requires 8-15 digits in total (E.164).
_INTL_PHONE_RE = re.compile(r'\+\d{1,3}(?:[ .\-]?\(?\d{1,4}\)?){1,6}\b')
_IPV4_RE = re.compile(
    r'(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]\d|\d)\.){3}'
    r'(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]\d|\d)(?!\d|\.\d)')
_IPV6_RE = re.compile(r'(?<![:\w])(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}(?![:\w])')

_PII_PATTERNS: List[Tuple[re.Pattern[str], str, str]] = [
    (_EMAIL_RE, '[PII_EMAIL]', 'email'),
    (_URL_RE, '[PII_URL]', 'url'),
    (_WWW_RE, '[PII_URL]', 'url'),
    (_IBAN_RE, '[PII_IBAN]', 'iban'),
    (_CARD_RE, '[PII_CARD]', 'card'),
    (_SSN_RE, '[PII_SSN]', 'ssn'),
    (_INTL_PHONE_RE, '[PII_PHONE]', 'phone'),   # before NANP so '+1 (555) …' stays whole
    (_PHONE_RE, '[PII_PHONE]', 'phone'),
    (_IPV4_RE, '[PII_IP]', 'ip'),
    (_IPV6_RE, '[PII_IP]', 'ip'),
]

_VALIDATORS: Dict[re.Pattern[str], Validator] = {
    _IBAN_RE: _is_iban, _CARD_RE: _is_card, _SSN_RE: _is_ssn,
    _INTL_PHONE_RE: _is_intl_phone, _IPV4_RE: _is_ip, _IPV6_RE: _is_ip,
}
_TRIMMERS: Dict[re.Pattern[str], Callable[[str], Tuple[str, str]]] = {
    _URL_RE: _trim_url, _WWW_RE: _trim_url,
}

def _mask_email(m: re.Match[str]) -> str:
    full = m.group(0)
    try:
        local, domain = full.split('@', 1)
        masked_local = '***' if len(local) <= 1 else local[0] + '***' + local[-1]
        return f"{masked_local}@{domain}"
    except ValueError:
        return '[PII_EMAIL]'

def _mask_last_digits(digits: str, n: int = 4) -> str:
    return digits[-n:] if len(digits) >= n else '*' * n

def _mask_phone(m: re.Match[str]) -> str:
    digits = re.sub(r'\D', '', m.group(0))
    return f"***-***-{_mask_last_digits(digits)}"

def _mask_card(m: re.Match[str]) -> str:
    digits = re.sub(r'\D', '', m.group(0))
    return f"****-****-****-{_mask_last_digits(digits)}"

def _mask_iban(m: re.Match[str]) -> str:
    compact = m.group(0).replace(' ', '')
    return f"{compact[:2]}**-****-{_mask_last_digits(compact)}"

def _mask_ip(m: re.Match[str]) -> str:
    """/16 (IPv4) or /32 (IPv6) prefix kept, host part masked."""
    try:
        ip = ipaddress.ip_address(m.group(0))
        if isinstance(ip, ipaddress.IPv4Address):
            return f"{ip.packed[0]}.{ip.packed[1]}.***.***"
        groups = ip.exploded.split(':')
        return f"{int(groups[0], 16):x}:{int(groups[1], 16):x}:****::"
    except ValueError:
        return '[PII_IP]'

def _mask_ssn(m: re.Match[str]) -> str:
    digits = re.sub(r'\D', '', m.group(0))
    return f"***-**-{_mask_last_digits(digits)}"

_MASK_FN: Dict[str, Callable[[re.Match[str]], str]] = {
    'email': _mask_email, 'phone': _mask_phone, 'card': _mask_card, 'iban': _mask_iban,
    'ip': _mask_ip, 'ssn': _mask_ssn,
}

class PseudoRegistry:
    """Maps real PII values to stable pseudonyms within a run."""
    _TEMPLATES: Dict[str, str] = {
        'email': 'email_{n:04d}@redacted.local', 'phone': 'phone_{n:04d}',
        'card': 'card_{n:04d}', 'ssn': '000-00-{n:04d}', 'iban': 'IBAN_{n:04d}',
        'url': 'https://redacted-{n:04d}.local', 'custom': 'pii_{n:04d}',
        'person': 'Person_{n:04d}', 'location': 'Place_{n:04d}', 'org': 'Org_{n:04d}',
        'address': 'Address_{n:04d}', 'date_of_birth': 'DOB_{n:04d}', 'id_number': 'ID_{n:04d}',
        'financial': 'ACCT_{n:04d}', 'username': 'user_{n:04d}',
        'credential': 'CREDENTIAL_{n:04d}',
    }

    def __init__(self) -> None:
        self._map: Dict[str, str] = {}
        self._counts: Dict[str, int] = {}
        self._templates: Dict[str, str] = dict(self._TEMPLATES)

    def add_templates(self, templates: Dict[str, str]) -> None:
        """Name additional kinds (e.g. secret types) without overriding existing ones."""
        for kind, tmpl in templates.items():
            self._templates.setdefault(kind, tmpl)

    def get_or_create(self, value: str, kind: str) -> str:
        if value in self._map:
            return self._map[value]
        n = self._counts.get(kind, 0) + 1
        self._counts[kind] = n
        if kind == 'ip':
            # A valid address in 10/8 (the old '0.0.0.{n}' broke past n=255).
            pseudo = f"10.{(n >> 16) & 255}.{(n >> 8) & 255}.{n & 255}"
        else:
            pseudo = self._templates.get(kind, 'pii_{n:04d}').format(n=n)
        self._map[value] = pseudo
        return pseudo

    def to_dict(self) -> Dict[str, str]:
        return dict(self._map)

    def to_state(self) -> Dict[str, Dict[str, Any]]:
        return {'map': dict(self._map), 'counts': dict(self._counts)}

    @classmethod
    def from_state(cls, state: Dict[str, Dict[str, Any]]) -> 'PseudoRegistry':
        reg = cls()
        reg._map = dict(state.get('map', {}))
        reg._counts = dict(state.get('counts', {}))
        return reg

def apply_patterns(
    text: str,
    patterns: List[Tuple[re.Pattern[str], str, str]],
    mask: bool = False,
    pseudo_registry: Optional[PseudoRegistry] = None,
    counters: Optional[Dict[str, int]] = None,
    mask_fns: Optional[Dict[str, Callable[[re.Match[str]], str]]] = None,
) -> str:
    """Apply a list of (regex, token, kind) redaction patterns to text.

    Shared by PII and secrets redaction. Built-in patterns may have a
    validator (match rejected -> text kept) and a trimmer (trailing
    punctuation kept outside the redaction); a named group 'secret' limits
    the redaction to that span. Pseudonymization takes
    precedence, then masking (for kinds with a mask function), then plain
    token replacement. `counters` tallies substitutions per kind."""
    mask_fns = mask_fns if mask_fns is not None else _MASK_FN
    for pattern, token, kind in patterns:
        validate = _VALIDATORS.get(pattern)
        trim = _TRIMMERS.get(pattern)
        current = text
        hits = 0

        has_secret_group = 'secret' in pattern.groupindex

        def _sub(m: re.Match[str]) -> str:
            nonlocal hits
            if validate is not None and not validate(current, m):
                return m.group(0)
            # A named group 'secret' marks the sensitive span; the rest of the
            # match (e.g. `api_key = "`) is context and stays.
            if has_secret_group and m.group('secret') is not None:
                lo, hi = m.span('secret')
                prefix, value, suffix = (current[m.start():lo], current[lo:hi],
                                         current[hi:m.end()])
            else:
                prefix, value, suffix = '', m.group(0), ''
            if trim is not None:
                value, tail = trim(value)
                suffix = tail + suffix
            if not value:
                return m.group(0)
            hits += 1
            if pseudo_registry is not None:
                return prefix + pseudo_registry.get_or_create(value, kind) + suffix
            if mask and kind in mask_fns:
                return prefix + mask_fns[kind](m) + suffix
            return prefix + token + suffix

        text = pattern.sub(_sub, current)
        if hits and counters is not None:
            counters[kind] = counters.get(kind, 0) + hits
    return text


def find_spans(
    text: str, patterns: List[Tuple[re.Pattern[str], str, str]],
) -> List[Tuple[int, int, str]]:
    """(start, end, kind) of everything apply_patterns() would redact, in
    offsets of the original text. Patterns run in the same order; spans
    claimed by an earlier pattern are blanked so later ones cannot re-match
    them, mirroring sequential redaction."""
    work = text
    found: List[Tuple[int, int, str]] = []
    for pattern, _token, kind in patterns:
        validate = _VALIDATORS.get(pattern)
        trim = _TRIMMERS.get(pattern)
        claimed = []
        for m in pattern.finditer(work):
            if validate is not None and not validate(work, m):
                continue
            if 'secret' in pattern.groupindex and m.group('secret') is not None:
                start, end = m.span('secret')
            else:
                start, end = m.span()
            if trim is not None:
                end = start + len(trim(work[start:end])[0])
            if end > start:
                claimed.append((start, end, kind))
        for start, end, _ in claimed:
            work = work[:start] + '\x00' * (end - start) + work[end:]
        found.extend(claimed)
    return sorted(found)


def find_pii_spans(
    text: str, extra_patterns: Optional[List[Tuple[re.Pattern[str], str, str]]] = None,
) -> List[Tuple[int, int, str]]:
    """Spans the regex PII redactor would redact (see find_spans)."""
    return find_spans(text, _PII_PATTERNS + (extra_patterns or []))


def redact_pii(
    text: str,
    mask: bool = False,
    extra_patterns: Optional[List[Tuple[re.Pattern[str], str, str]]] = None,
    pseudo_registry: Optional[PseudoRegistry] = None,
    counters: Optional[Dict[str, int]] = None,
) -> str:
    """Redact, mask, or pseudonymize PII. When `counters` is given, tallies
    the number of substitutions per PII kind into it."""
    return apply_patterns(
        text, _PII_PATTERNS + (extra_patterns or []),
        mask=mask, pseudo_registry=pseudo_registry, counters=counters)

def clean_text(text: str, remove_html: bool = True) -> str:
    """Normalize unicode, strip HTML, and tidy whitespace.

    Newlines are preserved (they carry structure in code/markdown training
    data); only control characters, horizontal whitespace runs, and excessive
    blank lines are collapsed.
    """
    if not isinstance(text, str):
        return text
    text = unicodedata.normalize('NFKC', text)
    if remove_html:
        text = strip_html(text)
    text = re.sub(r'[\x00-\x08\x0B-\x1F\x7F-\x9F]', ' ', text)  # keep \t (0x09) and \n (0x0A)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r' ?\n ?', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()
