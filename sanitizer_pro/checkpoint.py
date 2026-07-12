"""Resumable runs: crash-safe checkpointing of pipeline progress.

With ``--resume``, the CLI writes ``<output>.checkpoint.json`` every
``--checkpoint-interval`` input records (atomically). If the run crashes or is
interrupted, rerunning the same command with ``--resume`` skips the
already-consumed input records, restores statistics and pseudonym state, and
appends to the existing output. The checkpoint is deleted on success.

Exact-dedup state is only durable with ``--dedup-backend sqlite`` and an
explicit ``--dedup-db-path``; with the in-memory backends a resumed run
restarts dedup from empty (a warning is emitted).
"""
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from sanitizer_pro.utils import ConfigurationError

CHECKPOINT_VERSION = 1


def checkpoint_path(output_path: str) -> str:
    return output_path + '.checkpoint.json'


def input_fingerprint(input_path: str) -> Dict[str, Any]:
    """Identity of the input so a checkpoint is never applied to different data."""
    if input_path.startswith('hf://'):
        return {'kind': 'hf', 'uri': input_path}
    st = os.stat(input_path)
    return {'kind': 'file', 'path': os.path.abspath(input_path),
            'size': st.st_size, 'mtime': round(st.st_mtime, 3)}


def save_checkpoint(output_path: str, *, input_path: str, records_read: int,
                    stats_state: Dict[str, Any],
                    pseudo_state: Optional[Dict[str, Any]] = None) -> None:
    payload = {
        'version': CHECKPOINT_VERSION,
        'input': input_fingerprint(input_path),
        'records_read': records_read,
        'stats': stats_state,
        'pseudo': pseudo_state,
    }
    path = checkpoint_path(output_path)
    tmp = path + '.tmp'
    Path(tmp).write_text(json.dumps(payload), encoding='utf-8')
    os.replace(tmp, path)


def load_checkpoint(output_path: str, input_path: str) -> Optional[Dict[str, Any]]:
    """Load and validate a checkpoint; None when there is nothing to resume."""
    path = checkpoint_path(output_path)
    if not os.path.exists(path):
        return None
    try:
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
    except Exception as exc:
        raise ConfigurationError(f"Corrupt checkpoint {path}: {exc}. "
                                 "Delete it to start fresh.") from None
    if payload.get('version') != CHECKPOINT_VERSION:
        raise ConfigurationError(
            f"Checkpoint {path} has unsupported version {payload.get('version')}. "
            "Delete it to start fresh.")
    current = input_fingerprint(input_path)
    if payload.get('input') != current:
        raise ConfigurationError(
            f"Checkpoint {path} was created for a different input "
            f"({payload.get('input')} vs {current}). Delete it to start fresh.")
    return payload


def clear_checkpoint(output_path: str) -> None:
    try:
        os.remove(checkpoint_path(output_path))
    except OSError:
        pass


def warn_about_volatile_state(args: Any) -> None:
    """Point out state that does not survive a resume."""
    if (getattr(args, 'deduplicate', False)
            and (args.dedup_backend != 'sqlite' or not args.dedup_db_path)):
        logging.warning(
            "--resume with in-memory dedup: duplicate detection restarts empty on "
            "resume. Use --dedup-backend sqlite --dedup-db-path PATH for exact "
            "cross-resume dedup.")
    if getattr(args, 'fuzzy_dedup', False):
        logging.warning("--resume with --fuzzy-dedup: the MinHash index restarts "
                        "empty on resume; near-duplicates across the boundary may "
                        "slip through.")
