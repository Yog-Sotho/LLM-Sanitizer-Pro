"""Shared utilities, constants, and custom exceptions."""
import enum
import gzip
import sys
from pathlib import Path
from typing import Optional, TextIO

class FilterReason(enum.Enum):
    QUALITY = 'quality'
    LANGUAGE = 'language'
    REQUIRE = 'require_fields'
    PROFANITY = 'profanity'
    CODE = 'code'

class ConfigurationError(ValueError):
    """Raised for invalid CLI arguments or config files."""

class InputFormatError(ValueError):
    """Raised when input file cannot be read or parsed."""

_STDIN = '-'
_STDOUT = '-'
_EXCEL_WARN_MB_DEFAULT = 100
_MAX_DEPTH_DEFAULT = 100
_ALLCAPS_MIN_LEN_DEFAULT = 50
_ALLCAPS_MIN_ALPHA_DEFAULT = 10

def smart_open(path: str, mode: str = 'r', encoding: str = 'utf-8') -> TextIO:
    """Open plain or gzip files, or return stdin/stdout."""
    if path in {_STDIN, _STDOUT}:
        return sys.stdin if 'r' in mode else sys.stdout
    if path.lower().endswith('.gz'):
        return gzip.open(path, mode + 't', encoding=encoding)  # type: ignore[return-value]
    return open(path, mode, encoding=encoding)

def get_file_format(path: str) -> str:
    """Extract file extension, handling .gz."""
    if path in {_STDIN, _STDOUT}:
        return ''
    p = Path(path)
    if str(p).lower().endswith('.gz'):
        p = Path(p.stem)
    return p.suffix.lower()

def resolve_fmt(path: str, override: Optional[str]) -> str:
    """Resolve format from override or path extension."""
    if override:
        s = override if override.startswith('.') else '.' + override
        return s.lower()
    return get_file_format(path)
