"""PII detection, masking, pseudonymization, and safe HTML stripping."""
import hashlib
import hmac
import html
import ipaddress
import logging
import re
import unicodedata
from typing import Any, Callable, Dict, FrozenSet, List, NamedTuple, Optional, Set, Tuple

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


class Gate(NamedTuple):
    """A cheap necessary condition for a pattern to match: one of `literals`
    (or, case-insensitively, `ci_literals`) occurs in the text, and/or the
    text has a digit. Patterns whose gate fails are skipped without running
    them — most text holds no PII, and all gate literals are found in one
    regex pass, far cheaper than one scan per pattern with lookarounds.
    Case-insensitive literals are matched by an IGNORECASE regex, which folds
    case exactly like the gated patterns do (Kelvin sign, dotless i, long s)."""
    literals: Tuple[str, ...] = ()
    digit: bool = False
    ci_literals: Tuple[str, ...] = ()

    @classmethod
    def ignorecase(cls, *literals: str, digit: bool = False) -> 'Gate':
        return cls((), digit, literals)


_DIGIT_RE = re.compile(r'\d')

# Keyed by built-in pattern, like the validators; every gate must be implied
# by its pattern (a match is impossible when the gate fails).
_GATES: Dict[re.Pattern[str], Gate] = {
    _EMAIL_RE: Gate(('@',)),
    _URL_RE: Gate(('://',)),
    _WWW_RE: Gate.ignorecase('www.'),
    _IBAN_RE: Gate(digit=True),
    _CARD_RE: Gate(digit=True),
    _SSN_RE: Gate(digit=True),
    _INTL_PHONE_RE: Gate(('+',), digit=True),
    _PHONE_RE: Gate(digit=True),
    _IPV4_RE: Gate(digit=True),
    _IPV6_RE: Gate((':',)),
}


class _Step(NamedTuple):
    pattern: re.Pattern[str]
    token: str
    kind: str
    validate: Optional[Validator]
    trim: Optional[Callable[[str], Tuple[str, str]]]
    gate: Optional[Gate]
    literal_ids: FrozenSet[int]      # trigger ids, any of which opens the gate


class _TriggerScan:
    """Finds which gate literals occur in a text. Substring tests (`in`)
    are exact and fast. Case-insensitive literals are tested against
    text.lower() when the text is ASCII; otherwise an IGNORECASE regex
    decides, so Unicode case folding matches the gated patterns exactly
    (Kelvin sign, dotless i, long s). Its zero-width lookahead reports every
    position where a literal starts (longest first), and literals that are
    prefixes of the one found there are implied (`sk-` inside `sk-ant-`)."""

    def __init__(self, literals: List[str], ignorecase: bool = False) -> None:
        order = sorted(set(literals), key=len, reverse=True)
        self.ids = {lit: i for i, lit in enumerate(order)}
        self.ignorecase = ignorecase
        self.items = [(lit.lower() if ignorecase else lit, i) for i, lit in enumerate(order)]
        self.implied: Dict[int, FrozenSet[int]] = {
            g: frozenset(self.ids[o] for o in order if lit.lower().startswith(o.lower()))
            for g, lit in enumerate(order, start=1)}
        self.regex = (re.compile('(?=' + '|'.join(f'({re.escape(x)})' for x in order) + ')',
                                 re.IGNORECASE) if order and ignorecase else None)

    def scan(self, text: str) -> Set[int]:
        if not self.ignorecase:
            return {i for lit, i in self.items if lit in text}
        if text.isascii():
            low = text.lower()
            return {i for lit, i in self.items if lit in low}
        found: Set[int] = set()
        if self.regex is not None:
            for m in self.regex.finditer(text):
                found |= self.implied[m.lastindex or 0]
        return found


class PatternPlan:
    """A pattern list prepared for repeated application: validators,
    trimmers and gates looked up once, gate literals compiled into two
    trigger scans (case-sensitive and case-insensitive)."""

    def __init__(self, patterns: List[Tuple[re.Pattern[str], str, str]]) -> None:
        gates = [_GATES.get(p) for p, _, _ in patterns]
        self.cs = _TriggerScan([lit for g in gates if g for lit in g.literals])
        self.ci = _TriggerScan([lit for g in gates if g for lit in g.ci_literals], True)
        n_cs = len(self.cs.ids)
        self.steps = [
            _Step(p, token, kind, _VALIDATORS.get(p), _TRIMMERS.get(p), g,
                  frozenset([self.cs.ids[x] for x in g.literals]
                            + [n_cs + self.ci.ids[x] for x in g.ci_literals]) if g else frozenset())
            for (p, token, kind), g in zip(patterns, gates)]
        self._n_cs = n_cs
        # Which steps each trigger can open; steps gated only on a digit; ungated steps.
        self._by_trigger: Dict[int, List[int]] = {}
        self._digit_only: List[int] = []
        self._always: List[int] = []
        for i, step in enumerate(self.steps):
            if step.gate is None:
                self._always.append(i)
            elif step.literal_ids:
                for t in step.literal_ids:
                    self._by_trigger.setdefault(t, []).append(i)
            elif step.gate.digit:
                self._digit_only.append(i)
            else:
                self._always.append(i)

    def candidates(self, text: str, after: int = -1) -> List[int]:
        """Indices (> after, ascending) of the steps whose gate is open."""
        found = set(self._always)
        if self._by_trigger:
            for t in self.triggers(text):
                found.update(self._by_trigger.get(t, ()))
        digit: Optional[bool] = None
        if self._digit_only:
            digit = _DIGIT_RE.search(text) is not None
            if digit:
                found.update(self._digit_only)
        out = []
        for i in sorted(found):
            if i <= after:
                continue
            gate = self.steps[i].gate
            if gate is not None and gate.digit and self.steps[i].literal_ids:
                if digit is None:
                    digit = _DIGIT_RE.search(text) is not None
                if not digit:
                    continue
            out.append(i)
        return out

    def triggers(self, text: str) -> Set[int]:
        found = self.cs.scan(text)
        if self.ci.items:
            found |= {self._n_cs + i for i in self.ci.scan(text)}
        return found


_PLANS: Dict[int, Tuple[List[Tuple[re.Pattern[str], str, str]], PatternPlan]] = {}


def pattern_plan(patterns: List[Tuple[re.Pattern[str], str, str]]) -> PatternPlan:
    """The cached plan for a pattern list (kept by identity; the cache holds
    the list, so its id cannot be reused while cached)."""
    hit = _PLANS.get(id(patterns))
    if hit is not None and hit[0] is patterns:
        return hit[1]
    if len(_PLANS) > 64:  # lists built per call by API users: bound the cache
        _PLANS.clear()
    plan = PatternPlan(patterns)
    _PLANS[id(patterns)] = (patterns, plan)
    return plan


_COMBINED: Dict[Tuple[int, int], List[Tuple[re.Pattern[str], str, str]]] = {}


def combined_patterns(base: List[Tuple[re.Pattern[str], str, str]],
                      extra: Optional[List[Tuple[re.Pattern[str], str, str]]]
                      ) -> List[Tuple[re.Pattern[str], str, str]]:
    """base + extra, as the same list object on every call with the same two
    lists, so pattern_plan() finds it in its cache."""
    if not extra:
        return base
    key = (id(base), id(extra))
    hit = _COMBINED.get(key)
    if hit is None or hit[len(base):] != extra:
        if len(_COMBINED) > 64:
            _COMBINED.clear()
        hit = _COMBINED[key] = base + extra
    return hit


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
    """Maps real PII values to stable pseudonyms.

    Without a key, pseudonyms are numbered in order of first appearance
    (Person_0001, Person_0002, ...): stable within one process. With a
    `key`, the id is derived from HMAC-SHA256(key, kind, value): every
    worker process and every run with the same key gives a value the same
    pseudonym, without sharing state. Keyed ids are 12 lowercase hex digits
    (48 bits; Person_3fa94c07e21b): hex, not decimal, so that no pseudonym
    holds the long digit runs that later patterns (cards) would redact
    again. IPv4 pseudonyms stay addresses in 10.0.0.0/8 (24 bits). With
    very many distinct values two can share a pseudonym; that is logged
    when it happens here."""
    _TEMPLATES: Dict[str, str] = {
        'email': 'email_{n:04d}@redacted.local', 'phone': 'phone_{n:04d}',
        'card': 'card_{n:04d}', 'ssn': '000-00-{n:04d}', 'iban': 'IBAN_{n:04d}',
        'url': 'https://redacted-{n:04d}.local', 'custom': 'pii_{n:04d}',
        'person': 'Person_{n:04d}', 'location': 'Place_{n:04d}', 'org': 'Org_{n:04d}',
        'address': 'Address_{n:04d}', 'date_of_birth': 'DOB_{n:04d}', 'id_number': 'ID_{n:04d}',
        'financial': 'ACCT_{n:04d}', 'username': 'user_{n:04d}',
        'credential': 'CREDENTIAL_{n:04d}',
    }

    def __init__(self, key: Optional[str] = None, track_new: bool = False) -> None:
        self._map: Dict[str, str] = {}
        self._counts: Dict[str, int] = {}
        self._templates: Dict[str, str] = dict(self._TEMPLATES)
        self._key = key.encode('utf-8') if key else None
        self._issued: Dict[str, str] = {}            # keyed mode: pseudonym -> value
        self._new: Optional[List[Tuple[str, str]]] = [] if track_new else None

    @property
    def keyed(self) -> bool:
        return self._key is not None

    def add_templates(self, templates: Dict[str, str]) -> None:
        """Name additional kinds (e.g. secret types) without overriding existing ones."""
        for kind, tmpl in templates.items():
            self._templates.setdefault(kind, tmpl)

    def get_or_create(self, value: str, kind: str) -> str:
        if value in self._map:
            return self._map[value]
        template = self._templates.get(kind, 'pii_{n:04d}')
        if self._key is not None:
            digest = hmac.new(self._key, f"{kind}\x00{value}".encode('utf-8'),
                              hashlib.sha256).digest()
            n = int.from_bytes(digest[:3], 'big')
            template = template.replace('{n:04d}', digest[:6].hex())
        else:
            n = self._counts.get(kind, 0) + 1
            self._counts[kind] = n
        if kind == 'ip':
            # A valid address in 10/8 (the old '0.0.0.{n}' broke past n=255).
            pseudo = f"10.{(n >> 16) & 255}.{(n >> 8) & 255}.{n & 255}"
        else:
            pseudo = template.format(n=n)
        if self._key is not None:
            other = self._issued.setdefault(pseudo, value)
            if other != value:
                logging.warning(f"Pseudonym collision: two distinct {kind} values map to "
                                f"{pseudo} (keyed ids are 48-bit; IPv4 24-bit).")
        self._map[value] = pseudo
        if self._new is not None:
            self._new.append((value, pseudo))
        return pseudo

    def drain_new(self) -> List[Tuple[str, str]]:
        """Mappings created since the last call (track_new=True): workers
        send these to the parent, which merges them for the map file."""
        new, self._new = (self._new or []), ([] if self._new is not None else None)
        return new

    def merge(self, pairs: List[Tuple[str, str]]) -> None:
        for value, pseudo in pairs:
            self._map.setdefault(value, pseudo)

    def to_dict(self) -> Dict[str, str]:
        return dict(self._map)

    def to_state(self) -> Dict[str, Dict[str, Any]]:
        return {'map': dict(self._map), 'counts': dict(self._counts)}

    @classmethod
    def from_state(cls, state: Dict[str, Dict[str, Any]],
                   key: Optional[str] = None) -> 'PseudoRegistry':
        reg = cls(key=key)
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
    plan = pattern_plan(patterns)
    todo = plan.candidates(text)
    pos = 0
    while pos < len(todo):
        index = todo[pos]
        pos += 1
        step = plan.steps[index]
        pattern, token, kind = step.pattern, step.token, step.kind
        validate, trim = step.validate, step.trim
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
        if hits:
            # Replacements (pseudonyms) may add triggers or digits for later patterns.
            todo, pos = plan.candidates(text, after=index), 0
            if counters is not None:
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
    plan = pattern_plan(patterns)
    todo = plan.candidates(work)
    pos = 0
    while pos < len(todo):
        index = todo[pos]
        pos += 1
        step = plan.steps[index]
        pattern, kind, validate, trim = step.pattern, step.kind, step.validate, step.trim
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
        if claimed:
            todo, pos = plan.candidates(work, after=index), 0
        found.extend(claimed)
    return sorted(found)


def find_pii_spans(
    text: str, extra_patterns: Optional[List[Tuple[re.Pattern[str], str, str]]] = None,
) -> List[Tuple[int, int, str]]:
    """Spans the regex PII redactor would redact (see find_spans)."""
    return find_spans(text, combined_patterns(_PII_PATTERNS, extra_patterns))


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
        text, combined_patterns(_PII_PATTERNS, extra_patterns),
        mask=mask, pseudo_registry=pseudo_registry, counters=counters)

_CONTROL_RE = re.compile(r'[\x00-\x08\x0B-\x1F\x7F-\x9F]')  # keeps \t (0x09) and \n (0x0A)
_HSPACE_RE = re.compile(r'[ \t]{2,}|\t')  # single spaces are already final
_NEWLINE_SPACE_RE = re.compile(r' ?\n ?')
_BLANK_LINES_RE = re.compile(r'\n{3,}')


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
    text = _CONTROL_RE.sub(' ', text)
    text = _HSPACE_RE.sub(' ', text)
    if '\n' in text:
        text = _NEWLINE_SPACE_RE.sub('\n', text)
        text = _BLANK_LINES_RE.sub('\n\n', text)
    return text.strip()
