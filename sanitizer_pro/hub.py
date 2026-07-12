"""Hugging Face Hub integration: read datasets directly as input.

Datasets on the Hub are addressable as input paths::

    sanitize --input hf://openai/gsm8k --output clean.jsonl ...
    sanitize --input hf://openai/gsm8k/main/train ...

URI grammar: ``hf://<owner>/<name>[/<config>[/<split>]]`` — config defaults to
the repo's first listed config, split defaults to ``train``.

Files are fetched through the Hub's parquet conversion API (no ``datasets``
library needed — only ``pyarrow``) and cached under
``~/.cache/llm-sanitizer-pro/datasets``. The same machinery backs benchmark
downloads for decontamination.
"""
import json
import logging
import os
import ssl
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from sanitizer_pro.utils import ConfigurationError, InputFormatError

HF_URI_PREFIX = 'hf://'
_HF_PARQUET_API = 'https://huggingface.co/api/datasets/{repo}/parquet'
_DATASET_CACHE = os.path.join('~', '.cache', 'llm-sanitizer-pro', 'datasets')


def _ssl_context() -> ssl.SSLContext:
    cafile = (os.environ.get('REQUESTS_CA_BUNDLE') or os.environ.get('SSL_CERT_FILE')
              or os.environ.get('CURL_CA_BUNDLE'))
    return ssl.create_default_context(cafile=cafile if cafile and os.path.exists(cafile) else None)


def http_get(url: str, timeout: float = 300.0) -> bytes:
    headers = {'User-Agent': 'llm-sanitizer-pro'}
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    if token and 'huggingface.co' in url:
        headers['Authorization'] = f'Bearer {token}'
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as resp:
        return resp.read()


def list_parquet(repo: str) -> Dict[str, Any]:
    """Return the Hub's parquet map for a dataset: {config: {split: [urls]}}."""
    try:
        return json.loads(http_get(_HF_PARQUET_API.format(repo=repo), timeout=60.0))
    except Exception as exc:
        raise ConfigurationError(
            f"Could not list parquet files for '{repo}' on the Hugging Face Hub: {exc}. "
            "For private/gated datasets set HF_TOKEN.") from None


@dataclass(frozen=True)
class HFDatasetRef:
    repo: str                      # owner/name
    config: Optional[str] = None   # None → first listed config
    split: str = 'train'

    def __str__(self) -> str:
        return f"hf://{self.repo}/{self.config or '<default>'}/{self.split}"


def parse_hf_uri(uri: str) -> HFDatasetRef:
    """Parse hf://owner/name[/config[/split]]."""
    if not uri.startswith(HF_URI_PREFIX):
        raise ConfigurationError(f"Not a Hugging Face URI: {uri}")
    parts = [p for p in uri[len(HF_URI_PREFIX):].split('/') if p]
    if len(parts) < 2 or len(parts) > 4:
        raise ConfigurationError(
            f"Invalid Hub URI '{uri}'. Expected hf://owner/name[/config[/split]].")
    repo = f"{parts[0]}/{parts[1]}"
    config = parts[2] if len(parts) >= 3 else None
    split = parts[3] if len(parts) == 4 else 'train'
    return HFDatasetRef(repo=repo, config=config, split=split)


def resolve_parquet_urls(ref: HFDatasetRef) -> List[str]:
    api_json = list_parquet(ref.repo)
    if not api_json:
        raise ConfigurationError(f"No parquet conversions available for '{ref.repo}'.")
    config = ref.config
    if config is None:
        config = 'default' if 'default' in api_json else sorted(api_json)[0]
        logging.info(f"hf://{ref.repo}: using config '{config}' "
                     f"(available: {sorted(api_json)})")
    if config not in api_json:
        raise ConfigurationError(
            f"Config '{config}' not found in '{ref.repo}'. Available: {sorted(api_json)}")
    splits = api_json[config]
    if ref.split not in splits:
        raise ConfigurationError(
            f"Split '{ref.split}' not found in '{ref.repo}' config '{config}'. "
            f"Available: {sorted(splits)}")
    urls = splits[ref.split]
    if not urls:
        raise ConfigurationError(f"No parquet files listed for {ref}.")
    return urls


def download_parquet(ref: HFDatasetRef, cache_dir: Optional[str] = None,
                     cache_ns: str = _DATASET_CACHE) -> List[Path]:
    """Download (or reuse cached) parquet shards for a dataset reference."""
    safe = f"{ref.repo.replace('/', '__')}__{ref.config or 'default'}__{ref.split}"
    base = Path(os.path.expanduser(cache_dir or cache_ns)) / safe
    base.mkdir(parents=True, exist_ok=True)

    manifest = base / 'manifest.json'
    if manifest.exists():
        try:
            cached = [base / f for f in json.loads(manifest.read_text())]
            if cached and all(p.exists() and p.stat().st_size > 0 for p in cached):
                return cached
        except Exception:
            pass  # stale/corrupt manifest — re-download

    urls = resolve_parquet_urls(ref)
    paths: List[Path] = []
    for i, url in enumerate(urls):
        dest = base / f"part-{i:03d}.parquet"
        logging.info(f"Downloading {ref.repo} [{ref.split}] shard {i + 1}/{len(urls)} …")
        tmp = dest.with_suffix('.tmp')
        tmp.write_bytes(http_get(url))
        os.replace(tmp, dest)
        paths.append(dest)
    manifest.write_text(json.dumps([p.name for p in paths]))
    return paths


def iter_hub_records(uri: str, cache_dir: Optional[str] = None) -> Iterator[Dict[str, Any]]:
    """Stream records from a hf:// dataset URI."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError(
            "Reading hf:// datasets requires pyarrow (pip install pyarrow).") from None
    ref = parse_hf_uri(uri)
    paths = download_parquet(ref, cache_dir=cache_dir)
    for path in paths:
        for batch in pq.ParquetFile(str(path)).iter_batches():
            yield from (r for r in batch.to_pylist() if isinstance(r, dict))


def iter_parquet_texts(repo: str, parts: Tuple[Tuple[str, str], ...],
                       fields: Tuple[str, ...], cache_dir: Optional[str] = None,
                       cache_ns: str = os.path.join('~', '.cache', 'llm-sanitizer-pro',
                                                    'benchmarks')) -> Iterator[str]:
    """Yield text fields from Hub parquet shards (used by decontamination)."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError(
            "--decontaminate needs pyarrow to read benchmark parquet files "
            "(pip install pyarrow). Alternatively supply local reference files "
            "via --decontam-refs.") from None
    for config, split in parts:
        ref = HFDatasetRef(repo=repo, config=config, split=split)
        for path in download_parquet(ref, cache_dir=cache_dir, cache_ns=cache_ns):
            for batch in pq.ParquetFile(str(path)).iter_batches():
                for record in batch.to_pylist():
                    for field in fields:
                        v = record.get(field)
                        if isinstance(v, str) and v.strip():
                            yield v


def is_hub_uri(path: Optional[str]) -> bool:
    return bool(path) and str(path).startswith(HF_URI_PREFIX)


__all__ = ['HFDatasetRef', 'HF_URI_PREFIX', 'download_parquet', 'http_get', 'is_hub_uri',
           'iter_hub_records', 'iter_parquet_texts', 'list_parquet', 'parse_hf_uri',
           'resolve_parquet_urls', 'InputFormatError']
