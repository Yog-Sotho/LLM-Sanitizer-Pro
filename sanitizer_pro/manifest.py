"""Run manifest: a machine-readable record of how a dataset was produced.

    sanitize --input raw/ --output clean.jsonl --remove-pii --manifest run.json

The manifest names the tool and dependency versions, the models used, the
full configuration (and its hash), every input and output file with its
size and SHA-256, the per-stage counts, and timing. Two manifests are enough
to tell whether two datasets were produced the same way (`sanitize diff`),
and the dataset card (`--dataset-card`) is rendered from one.

Secrets never enter it: the pseudonymization key is recorded only as set or
unset, and its command-line value is masked.
"""
import dataclasses
import hashlib
import json
import os
import platform
import re
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from sanitizer_pro import __version__

SCHEMA = 'llm-sanitizer-pro/run-manifest'
SCHEMA_VERSION = 1
SECRET_FIELDS = frozenset({'pseudo_key'})
SECRET_FLAGS = ('--pseudo-key',)
_DEPENDENCIES = (
    'tqdm', 'pyarrow', 'pandas', 'openpyxl', 'ijson', 'pyyaml', 'langdetect', 'fasttext',
    'fasttext-numpy2-wheel', 'datasketch', 'rensa', 'usearch', 'model2vec', 'numpy',
    'transformers', 'torch', 'spacy', 'gliner2', 'huggingface-hub',
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, re.Pattern):
        return value.pattern
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable(v) for v in value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def config_dict(config: Any) -> Dict[str, Any]:
    """Every setting, JSON-safe, with secrets reduced to whether they are set."""
    out: Dict[str, Any] = {}
    for f in dataclasses.fields(config):
        value = getattr(config, f.name)
        if f.name in SECRET_FIELDS:
            out[f.name] = '<set>' if value else None
        else:
            out[f.name] = _jsonable(value)
    return out


def config_hash(config: Any) -> str:
    canonical = json.dumps(config_dict(config), sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def scrub_argv(argv: Sequence[str]) -> List[str]:
    """The command line with secret flag values masked."""
    out: List[str] = []
    mask_next = False
    for arg in argv:
        if mask_next:
            out.append('***')
            mask_next = False
        elif arg in SECRET_FLAGS:
            out.append(arg)
            mask_next = True
        elif any(arg.startswith(flag + '=') for flag in SECRET_FLAGS):
            out.append(arg.split('=', 1)[0] + '=***')
        else:
            out.append(arg)
    return out


def file_digest(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def describe_file(path: str, digest: bool = True) -> Dict[str, Any]:
    if path.startswith('hf://'):
        return {'uri': path}
    if path == '-':
        return {'path': '<stdin>'}
    st = os.stat(path)
    entry: Dict[str, Any] = {'path': path, 'bytes': st.st_size}
    if digest:
        entry['sha256'] = file_digest(path)
    return entry


def dependency_versions() -> Dict[str, str]:
    found = {}
    for name in _DEPENDENCIES:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return found


def models_used(config: Any, transformer: Any = None) -> Dict[str, Any]:
    """The models and reference data the configuration loads, by stage."""
    from sanitizer_pro.langid import FASTTEXT_MODELS
    from sanitizer_pro.scoring import FASTTEXT_SCORERS, FineWebEduScorer
    c = config
    models: Dict[str, Any] = {}
    if c.lang_filter:
        ident = getattr(transformer, 'lang_identifier', None)
        name = getattr(ident, 'name', c.lang_backend)
        default = FASTTEXT_MODELS.get(name)
        models['language_id'] = {'backend': name, 'model': c.lang_model or (
            '/'.join(default) if default else name)}
    scorer = getattr(transformer, 'scorer', None)
    if scorer is not None:
        backend = getattr(scorer, 'backend_name', c.quality_scorer)
        default_model = {'perplexity': 'distilgpt2',
                         'fineweb-edu': FineWebEduScorer.DEFAULT_MODEL}.get(backend)
        if backend in FASTTEXT_SCORERS:
            default_model = '/'.join(FASTTEXT_SCORERS[backend][:2])
        models['quality_scorer'] = {'backend': backend,
                                    'model': c.quality_model or default_model}
    if c.remove_pii and c.pii_ner:
        from sanitizer_pro import ner
        defaults = {'gliner': ner._GLINER_DEFAULT_MODEL, 'spacy': ner._SPACY_DEFAULT_MODEL,
                    'transformers': ner._HF_DEFAULT_MODEL}
        ner_obj = getattr(transformer, 'ner', None)
        backend = getattr(ner_obj, 'backend_name', c.pii_ner_backend)
        models['pii_ner'] = {'backend': backend,
                             'model': c.pii_ner_model or defaults.get(backend, backend),
                             'entities': _jsonable(getattr(ner_obj, 'entities',
                                                           c.pii_ner_entities))}
    if c.semantic_dedup:
        models['semantic_dedup'] = {'model': c.semantic_model, 'index': c.semantic_index}
    if c.tokenizer and c.tokenizer != 'whitespace' and (c.max_tokens or c.chat_max_tokens):
        models['tokenizer'] = c.tokenizer
    if c.decontaminate or c.decontam_refs:
        from sanitizer_pro.decontam import KNOWN_BENCHMARKS, resolve_benchmark_names
        names = []
        if c.decontaminate:
            raw = c.decontaminate if isinstance(c.decontaminate, str) else ','.join(c.decontaminate)
            names = resolve_benchmark_names(raw)
        models['decontamination'] = {
            'benchmarks': {n: KNOWN_BENCHMARKS[n].repo for n in names},
            'reference_files': [describe_file(p) for p in (c.decontam_refs or [])
                                if os.path.exists(p)],
            'ngram': c.decontam_ngram,
        }
    return models


def build_manifest(*, config: Any, stats: Dict[str, Any], inputs: Sequence[str],
                   outputs: Sequence[str], started_at: float, finished_at: float,
                   transformer: Any = None, argv: Optional[Sequence[str]] = None,
                   jobs: int = 1, digest: bool = True) -> Dict[str, Any]:
    def iso(ts: float) -> str:
        return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec='seconds')

    return {
        'schema': SCHEMA,
        'schema_version': SCHEMA_VERSION,
        'tool': {'name': 'llm-sanitizer-pro', 'version': __version__,
                 'python': platform.python_version(), 'platform': platform.platform()},
        'run': {'started_at': iso(started_at), 'finished_at': iso(finished_at),
                'duration_s': round(finished_at - started_at, 3), 'jobs': jobs,
                'command': scrub_argv(argv if argv is not None else sys.argv)},
        'config_sha256': config_hash(config),
        'config': config_dict(config),
        'models': models_used(config, transformer),
        'dependencies': dependency_versions(),
        'inputs': [describe_file(p, digest) for p in inputs],
        'outputs': [describe_file(p, digest) for p in outputs if os.path.exists(p)],
        'counts': stats,
    }


def write_manifest(path: str, manifest: Dict[str, Any]) -> None:
    tmp = path + '.tmp'
    Path(tmp).write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(tmp, path)


def load_run(path: str) -> Dict[str, Any]:
    """A manifest or a --stats-file, as {'counts': ..., 'config': ..., ...}."""
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: not a JSON object")
    if data.get('schema') == SCHEMA:
        return data
    if 'total' in data and 'kept' in data:      # a --stats-file
        return {'counts': data, 'tool': {'version': data.get('version')}}
    raise ValueError(f"{path}: neither a run manifest nor a stats file")
