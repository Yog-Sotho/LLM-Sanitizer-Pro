"""Deterministic, content-addressed randomness for sampling and splits.

A record's position in [0, 1) is derived from a salted SHA-256 of its
canonical JSON, so the same record always gets the same sampling decision and
the same train/val/test split: independent of input order, of upstream
filters, of resuming, and of how many workers ran. `--seed` changes the salt.
"""
import hashlib
import json
from typing import Any, Optional

_SCALE = float(1 << 64)


def content_fraction(record: Any, seed: Optional[int] = None, purpose: str = 'sample') -> float:
    """Map a record to a uniform value in [0, 1) (stable for equal content)."""
    payload = json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(',', ':'),
                         default=str)
    digest = hashlib.sha256(f"{purpose}:{seed or 0}:".encode() + payload.encode('utf-8')).digest()
    return int.from_bytes(digest[:8], 'big') / _SCALE


def keep_in_sample(record: Any, fraction: float, seed: Optional[int] = None) -> bool:
    return content_fraction(record, seed, 'sample') < fraction
