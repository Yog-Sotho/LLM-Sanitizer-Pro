"""PII detection, masking, pseudonymization, and safe HTML stripping."""
import ipaddress
import re
import unicodedata
from html.parser import HTMLParser
from typing import Callable, Dict, List, Optional, Tuple

class MLStripper(HTMLParser):
    """Safe HTML tag stripper using standard library state machine."""
    def __init__(self) -> None:
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text: List[str] = []

    def handle_data(self, d: str) -> None:
        self.text.append(d)

    def get_data(self) -> str:
        return ' '.join(self.text)

def strip_html(html: str) -> str:
    """Safely strip HTML tags without regex vulnerabilities."""
    s = MLStripper()
    try:
        s.feed(html)
        return s.get_data()
    except Exception:
        return html

# Order matters: URLs/emails first (so their fragments aren't re-matched), then
# longer numeric patterns (card) before shorter ones (SSN, phone) that could
# otherwise consume a prefix of them.
_PII_PATTERNS: List[Tuple[re.Pattern[str], str, str]] = [
    (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', re.IGNORECASE), '[PII_EMAIL]', 'email'),
    (re.compile(r'https?://\S+', re.IGNORECASE), '[PII_URL]', 'url'),
    (re.compile(r'\bwww\.\S+', re.IGNORECASE), '[PII_URL]', 'url'),
    (re.compile(r'\b(?:\d{4}[ \-]?){3}\d{4}\b'), '[PII_CARD]', 'card'),
    (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), '[PII_SSN]', 'ssn'),
    (re.compile(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'), '[PII_PHONE]', 'phone'),
    (re.compile(r'\+\d{1,3}[\s\-]?\(?\d{1,4}\)?[\s\-]?\d{1,4}[\s\-]?\d{1,9}\b'), '[PII_PHONE]', 'phone'),
    (re.compile(r'\b(?:(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]\d|\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]\d|\d)\b'), '[PII_IP]', 'ip'),
]

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

def _mask_ip(m: re.Match[str]) -> str:
    """Mathematically correct /16 subnet masking using ipaddress."""
    try:
        ip = ipaddress.ip_address(m.group(0))
        if isinstance(ip, ipaddress.IPv4Address):
            return f"{ip.packed[0]}.{ip.packed[1]}.***.***"
        return "****:****:****::"
    except ValueError:
        return '[PII_IP]'

def _mask_ssn(m: re.Match[str]) -> str:
    digits = re.sub(r'\D', '', m.group(0))
    return f"***-**-{_mask_last_digits(digits)}"

_MASK_FN: Dict[str, Callable[[re.Match[str]], str]] = {
    'email': _mask_email, 'phone': _mask_phone, 'card': _mask_card,
    'ip': _mask_ip, 'ssn': _mask_ssn,
}

class PseudoRegistry:
    """Maps real PII values to stable pseudonyms within a run."""
    _TEMPLATES: Dict[str, str] = {
        'email': 'email_{n:04d}@redacted.local', 'phone': 'phone_{n:04d}',
        'card': 'card_{n:04d}', 'ip': '0.0.0.{n}', 'ssn': '000-00-{n:04d}',
        'url': 'https://redacted-{n:04d}.local', 'custom': 'pii_{n:04d}',
    }

    def __init__(self) -> None:
        self._map: Dict[str, str] = {}
        self._counts: Dict[str, int] = {}

    def get_or_create(self, value: str, kind: str) -> str:
        if value in self._map:
            return self._map[value]
        n = self._counts.get(kind, 0) + 1
        self._counts[kind] = n
        tmpl = self._TEMPLATES.get(kind, 'pii_{n:04d}')
        pseudo = tmpl.format(n=n)
        self._map[value] = pseudo
        return pseudo

    def to_dict(self) -> Dict[str, str]:
        return dict(self._map)

def redact_pii(
    text: str,
    mask: bool = False,
    extra_patterns: Optional[List[Tuple[re.Pattern[str], str, str]]] = None,
    pseudo_registry: Optional[PseudoRegistry] = None,
) -> str:
    """Redact, mask, or pseudonymize PII."""
    for pattern, token, kind in (_PII_PATTERNS + (extra_patterns or [])):
        if pseudo_registry is not None:
            def _sub_pseudo(m: re.Match[str], _k: str = kind, _r: PseudoRegistry = pseudo_registry) -> str:
                return _r.get_or_create(m.group(0), _k)
            text = pattern.sub(_sub_pseudo, text)
        elif mask and kind in _MASK_FN:
            text = pattern.sub(_MASK_FN[kind], text)
        else:
            text = pattern.sub(token, text)
    return text

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
