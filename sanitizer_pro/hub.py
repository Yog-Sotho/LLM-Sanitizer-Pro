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
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from sanitizer_pro.utils import ConfigurationError, InputFormatError

HF_URI_PREFIX = 'hf://'
_HF_PARQUET_API = 'https://huggingface.co/api/datasets/{repo}/parquet'
_DATASET_CACHE = os.path.join('~', '.cache', 'llm-sanitizer-pro', 'datasets')


_RETRY_STATUS = {429, 500, 502, 503, 504}
_ATTEMPTS = 4
_CHUNK = 1 << 20
_sleep = time.sleep  # patched in tests


def _ssl_context() -> ssl.SSLContext:
    cafile = (os.environ.get('REQUESTS_CA_BUNDLE') or os.environ.get('SSL_CERT_FILE')
              or os.environ.get('CURL_CA_BUNDLE'))
    return ssl.create_default_context(cafile=cafile if cafile and os.path.exists(cafile) else None)


def is_hf_host(url: str) -> bool:
    """True only for Hugging Face hosts (exact hostname match, not substring)."""
    host = (urllib.parse.urlparse(url).hostname or '').lower()
    return any(host == d or host.endswith('.' + d) for d in ('huggingface.co', 'hf.co'))


class _HFRedirectHandler(urllib.request.HTTPRedirectHandler):
    """urllib copies Authorization onto redirects; drop it when the target is
    not a Hugging Face host (parquet shards redirect to storage/CDN hosts)."""

    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any,
                         newurl: str) -> Any:
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and not is_hf_host(newurl):
            new.remove_header('Authorization')
        return new


def _open(url: str, timeout: float) -> Any:
    headers = {'User-Agent': 'llm-sanitizer-pro'}
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    if token and is_hf_host(url):
        headers['Authorization'] = f'Bearer {token}'
    opener = urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=_ssl_context()), _HFRedirectHandler())
    return opener.open(urllib.request.Request(url, headers=headers), timeout=timeout)


def _retry_delay(exc: Exception, attempt: int) -> Optional[float]:
    """Seconds to wait before retrying, or None when the error is final."""
    if attempt >= _ATTEMPTS - 1:
        return None
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code not in _RETRY_STATUS:
            return None
        retry_after = exc.headers.get('Retry-After') if exc.headers else None
        if retry_after and retry_after.isdigit():
            return min(float(retry_after), 60.0)
    elif not isinstance(exc, (urllib.error.URLError, TimeoutError, ConnectionError)):
        return None
    return float(2 ** (attempt + 1))  # 2, 4, 8 s


def http_get(url: str, timeout: float = 300.0) -> bytes:
    """GET a (small) resource with retries on 429/5xx and network errors."""
    for attempt in range(_ATTEMPTS):
        try:
            with _open(url, timeout) as resp:
                return bytes(resp.read())
        except Exception as exc:
            delay = _retry_delay(exc, attempt)
            if delay is None:
                raise
            logging.warning(f"GET {url} failed ({exc}); retrying in {delay:.0f}s")
            _sleep(delay)
    raise AssertionError('unreachable')


def http_download(url: str, dest: Path, timeout: float = 300.0) -> None:
    """Stream a file to `dest` (atomically) with retries; verifies the size
    against Content-Length so a cut-off transfer is never cached."""
    part = dest.with_name(dest.name + '.part')
    for attempt in range(_ATTEMPTS):
        try:
            with _open(url, timeout) as resp, open(part, 'wb') as out:
                expected = resp.headers.get('Content-Length')
                written = 0
                while True:
                    chunk = resp.read(_CHUNK)
                    if not chunk:
                        break
                    out.write(chunk)
                    written += len(chunk)
            if expected is not None and expected.isdigit() and written != int(expected):
                raise ConnectionError(f"incomplete download: {written} of {expected} bytes")
            os.replace(part, dest)
            return
        except Exception as exc:
            delay = _retry_delay(exc, attempt)
            if delay is None:
                try:
                    part.unlink()
                except OSError:
                    pass
                raise
            logging.warning(f"Download of {url} failed ({exc}); retrying in {delay:.0f}s")
            _sleep(delay)


def list_parquet(repo: str) -> Dict[str, Any]:
    """Return the Hub's parquet map for a dataset: {config: {split: [urls]}}."""
    try:
        data = json.loads(http_get(_HF_PARQUET_API.format(repo=repo), timeout=60.0))
        if not isinstance(data, dict):
            raise ValueError("unexpected response (not a JSON object)")
        return data
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
    return [str(u) for u in urls]


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
        http_download(url, dest)
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


def _field_values(record: Dict[str, Any], path: str) -> List[str]:
    """Strings at a dotted field path: 'question', 'options' (list of str),
    'choices.text' (list inside a struct), 'answers_spans.spans'."""
    value: Any = record
    for key in path.split('.'):
        if not isinstance(value, dict) or key not in value:
            return []
        value = value[key]
    items = value if isinstance(value, (list, tuple)) else [value]
    return [v for v in items if isinstance(v, str) and v.strip()]


def iter_parquet_texts(repo: str, parts: Tuple[Tuple[str, str], ...],
                       fields: Tuple[str, ...], cache_dir: Optional[str] = None,
                       cache_ns: str = os.path.join('~', '.cache', 'llm-sanitizer-pro',
                                                    'benchmarks')) -> Iterator[str]:
    """Yield text fields from Hub parquet shards (used by decontamination).

    A part's config may be '*' (every config of the dataset). Field paths may
    be dotted and may name list-valued columns. If none of `fields` exists in
    the data, raise instead of silently indexing nothing."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError(
            "--decontaminate needs pyarrow to read benchmark parquet files "
            "(pip install pyarrow). Alternatively supply local reference files "
            "via --decontam-refs.") from None
    expanded: List[Tuple[str, str]] = []
    for config, split in parts:
        if config == '*':
            expanded += [(c, split) for c in sorted(list_parquet(repo))]
        else:
            expanded.append((config, split))
    for config, split in expanded:
        ref = HFDatasetRef(repo=repo, config=config, split=split)
        checked = False
        for path in download_parquet(ref, cache_dir=cache_dir, cache_ns=cache_ns):
            pf = pq.ParquetFile(str(path))
            if not checked:
                columns = set(pf.schema_arrow.names)
                if not any(f.split('.')[0] in columns for f in fields):
                    raise ConfigurationError(
                        f"{ref}: none of the indexed fields {list(fields)} exist "
                        f"(columns: {sorted(columns)}).")
                checked = True
            for batch in pf.iter_batches():
                for record in batch.to_pylist():
                    for field in fields:
                        yield from _field_values(record, field)


def resolve_model_file(repo: str, filename: str, revision: str = 'main') -> Path:
    """Local path of a model file from the Hub: huggingface_hub's cache when
    that library is installed, else this package's own cache and downloader."""
    try:
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(repo, filename, revision=revision))
    except ImportError:
        pass
    cache = Path(os.path.expanduser(os.path.join(
        '~', '.cache', 'llm-sanitizer-pro', 'models', repo.replace('/', '__'), revision)))
    dest = cache / filename
    if not dest.exists():
        cache.mkdir(parents=True, exist_ok=True)
        logging.info(f"Downloading {repo}/{filename} (one-time) …")
        http_download(f"https://huggingface.co/{repo}/resolve/{revision}/{filename}", dest)
    return dest


def is_hub_uri(path: Optional[str]) -> bool:
    return bool(path) and str(path).startswith(HF_URI_PREFIX)


__all__ = ['HFDatasetRef', 'HF_URI_PREFIX', 'download_parquet', 'http_download', 'http_get',
           'resolve_model_file',
           'is_hf_host', 'is_hub_uri',
           'iter_hub_records', 'iter_parquet_texts', 'list_parquet', 'parse_hf_uri',
           'resolve_parquet_urls', 'InputFormatError']
