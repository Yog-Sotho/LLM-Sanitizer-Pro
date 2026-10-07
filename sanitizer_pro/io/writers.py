"""Streaming writers with crash safety, sharding, and dataset splitting."""
import csv
import json
import logging
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from sanitizer_pro.utils import ConfigurationError, smart_open, _STDOUT

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore[assignment]
try:
    import xlsxwriter
except ImportError:
    xlsxwriter = None  # type: ignore[assignment]
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = pq = None  # type: ignore[assignment]

_STREAM_FORMATS = {'.jsonl', '.txt', '.csv', '.json'}
_BUFFERED_FORMATS = {'.xlsx', '.xls', '.parquet'}
SUPPORTED_OUTPUT_FORMATS = _STREAM_FORMATS | _BUFFERED_FORMATS


def _csv_value(v: Any) -> Any:
    """CSV cell for a record value: nested structures as JSON, not Python repr."""
    if isinstance(v, (dict, list, tuple)):
        return json.dumps(v, ensure_ascii=False, default=str)
    return v


def _scalar_or_json(v: Any) -> Any:
    return v if v is None or isinstance(v, str) else json.dumps(v, ensure_ascii=False, default=str)


def _tmp_sibling(path: str, tag: str) -> str:
    """A fresh temp file next to `path` that keeps a trailing .gz, so
    smart_open applies the same compression as the final file."""
    directory = os.path.dirname(os.path.abspath(path))
    gz = '.gz' if path.lower().endswith('.gz') else ''
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=f".{os.path.basename(path)}.",
                               suffix=f".{tag}{gz}")
    os.close(fd)
    return tmp


_PARQUET_BATCH = 50_000


class StreamingWriter:
    """Write records one at a time. JSON/CSV/Parquet/Excel outputs are atomic
    (temp file + os.replace) and keep every column that appears in any
    record: CSV and Parquet stage rows on disk and render them at close with
    the union of all fields (Parquet with a unified, type-promoted schema,
    written in batches). JSONL/TXT, and CSV in append/durable (resume) mode,
    stream straight to the destination."""

    APPENDABLE_FORMATS = {'.jsonl', '.txt', '.csv'}

    def __init__(self, output_path: str, fmt: str, encoding: str = 'utf-8',
                 txt_fallback_field: Optional[str] = None, append: bool = False,
                 durable: bool = False) -> None:
        if fmt not in SUPPORTED_OUTPUT_FORMATS:
            raise ConfigurationError(
                f"Unsupported output format '{fmt}'. Supported: {sorted(SUPPORTED_OUTPUT_FORMATS)}")
        if append and fmt not in self.APPENDABLE_FORMATS:
            raise ConfigurationError(
                f"Append/resume is only supported for {sorted(self.APPENDABLE_FORMATS)} outputs.")
        self.append = append
        if fmt in {'.xlsx', '.xls'} and not (xlsxwriter or pd):
            raise ImportError("Excel output requires: pip install xlsxwriter (or pandas+openpyxl)")
        if fmt == '.parquet' and not (pa and pq):
            raise ImportError("Parquet output requires: pip install pyarrow")
        if fmt in _BUFFERED_FORMATS and output_path == _STDOUT:
            raise ConfigurationError(f"{fmt} output cannot be written to stdout.")
        self.output_path, self.fmt, self.encoding = output_path, fmt, encoding
        self.txt_fallback_field = txt_fallback_field
        # CSV must stream in place when appending (resume) or when the output
        # offset has to be durable; otherwise it is staged so late columns fit.
        self._staged = (fmt == '.parquet' or
                        (fmt == '.csv' and output_path != _STDOUT and not append and not durable))
        self._file: Any = None
        self._tmp_path: Optional[str] = None
        self._stage_path: Optional[str] = None
        self._fields: Dict[str, None] = {}  # ordered union of keys (staged formats)
        self._csv_writer: Optional[csv.DictWriter] = None
        self._csv_fields: Optional[List[str]] = None
        self._dropped_columns: Dict[str, int] = {}
        self._buffer: List[Dict[str, Any]] = []
        self._json_first = True
        self._count = 0

    def __enter__(self) -> 'StreamingWriter':
        if self._staged:
            self._stage_path = _tmp_sibling(self.output_path, 'staging.jsonl')
            self._file = open(self._stage_path, 'w', encoding='utf-8')
        elif self.fmt == '.json':
            if self.output_path == _STDOUT:
                self._file = sys.stdout
            else:
                self._tmp_path = _tmp_sibling(self.output_path, 'tmp')
                self._file = smart_open(self._tmp_path, 'w', encoding=self.encoding)
            self._file.write('[\n')
        elif self.fmt in {'.jsonl', '.txt', '.csv'}:
            if self.append and self.fmt == '.csv' and os.path.exists(self.output_path) \
                    and os.path.getsize(self.output_path) > 0:
                # Recover the original header so appended rows keep column order
                # and no second header row is emitted.
                with smart_open(self.output_path, 'r', encoding=self.encoding) as existing:
                    header = existing.readline()
                fields = next(csv.reader([header])) if header.strip() else None
                self._file = smart_open(self.output_path, 'a', encoding=self.encoding)
                if fields:
                    self._csv_fields = fields
                    self._csv_writer = csv.DictWriter(
                        self._file, fieldnames=fields, extrasaction='ignore')
            else:
                self._file = smart_open(self.output_path, 'a' if self.append else 'w',
                                        encoding=self.encoding)
        return self

    def write(self, record: Dict[str, Any]) -> None:
        self._count += 1
        if self._staged:
            for k in record:
                if k not in self._fields:
                    self._fields[str(k)] = None
            self._file.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
        elif self.fmt == '.jsonl':
            self._file.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
        elif self.fmt == '.txt':
            text = record.get('text')
            if text is None and self.txt_fallback_field:
                text = record.get(self.txt_fallback_field)
            if text is None:
                text = json.dumps(record, ensure_ascii=False, default=str)
            self._file.write(str(text).replace('\n', ' ') + '\n')
        elif self.fmt == '.json':
            if not self._json_first:
                self._file.write(',\n')
            self._file.write(json.dumps(record, ensure_ascii=False, default=str))
            self._json_first = False
        elif self.fmt == '.csv':
            if self._csv_writer is None:
                self._csv_fields = list(record.keys())
                self._csv_writer = csv.DictWriter(
                    self._file, fieldnames=self._csv_fields, extrasaction='ignore')
                self._csv_writer.writeheader()
            self._note_dropped_columns(record)
            self._csv_writer.writerow({k: _csv_value(v) for k, v in record.items()})
        else:
            self._buffer.append(record)

    def _note_dropped_columns(self, record: Dict[str, Any]) -> None:
        """Streaming CSV cannot grow its header: count values that do not fit."""
        known = self._csv_fields or []
        for k in record:
            if k not in known:
                if k not in self._dropped_columns:
                    logging.warning(
                        f"CSV output {self.output_path}: field '{k}' is not in the header "
                        f"({len(known)} columns, fixed by the first/existing row); its "
                        "values are dropped. Use .jsonl to keep every field.")
                self._dropped_columns[k] = self._dropped_columns.get(k, 0) + 1

    @property
    def dropped_columns(self) -> Dict[str, int]:
        """Values dropped per field by streaming CSV output (normally empty)."""
        return dict(self._dropped_columns)

    def flush(self) -> None:
        if self._file is not None and not self._file.closed:
            self._file.flush()

    def durable_size(self) -> int:
        """Make everything written so far durable and return the output size
        in bytes — a safe truncation point for resuming after a crash.

        gzip output is closed and reopened in append mode so the offset falls
        on a complete gzip member boundary (a flushed-but-open member is not
        decodable after truncation)."""
        if (self.output_path == _STDOUT or self.fmt not in self.APPENDABLE_FORMATS
                or self._staged):
            raise ConfigurationError(
                f"{self.fmt} output to {self.output_path} has no durable size "
                "(open it with durable=True).")
        if self.output_path.lower().endswith('.gz'):
            self._file.close()
            self._file = smart_open(self.output_path, 'a', encoding=self.encoding)
            if self._csv_writer is not None:
                self._csv_writer = csv.DictWriter(
                    self._file, fieldnames=self._csv_fields, extrasaction='ignore')
            fd = os.open(self.output_path, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        else:
            self._file.flush()
            os.fsync(self._file.fileno())
        return os.path.getsize(self.output_path)

    # -- staged rendering -----------------------------------------------------

    def _staged_rows(self, fields: List[str], stringify: Optional[set] = None
                     ) -> Iterator[Dict[str, Any]]:
        stringify = stringify or set()
        with open(self._stage_path, 'r', encoding='utf-8') as src:
            for line in src:
                rec = json.loads(line)
                yield {f: (_scalar_or_json(rec.get(f)) if f in stringify else rec.get(f))
                       for f in fields}

    def _staged_batches(self, fields: List[str], stringify: Optional[set] = None
                        ) -> Iterator[List[Dict[str, Any]]]:
        batch: List[Dict[str, Any]] = []
        for row in self._staged_rows(fields, stringify):
            batch.append(row)
            if len(batch) >= _PARQUET_BATCH:
                yield batch
                batch = []
        if batch:
            yield batch

    def _render_csv(self, dest: str) -> None:
        fields = list(self._fields)
        with smart_open(dest, 'w', encoding=self.encoding) as out:
            if not fields:
                return
            writer = csv.DictWriter(out, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            for row in self._staged_rows(fields):
                writer.writerow({k: _csv_value(v) for k, v in row.items()})

    def _parquet_schema(self, fields: List[str], stringify: set) -> Any:
        schemas = [pa.Table.from_pylist(b).schema for b in self._staged_batches(fields, stringify)]
        if not schemas:
            return pa.schema([(f, pa.null()) for f in fields])
        return pa.unify_schemas(schemas, promote_options='permissive')

    def _conflicting_fields(self, fields: List[str]) -> set:
        """Fields whose values cannot share one Arrow type (e.g. int and str)."""
        conflicts = set()
        per_field: Dict[str, List[Any]] = {f: [] for f in fields}
        for batch in self._staged_batches(fields):
            for f in fields:
                if f in conflicts:
                    continue
                try:
                    per_field[f].append(pa.array([r[f] for r in batch]).type)
                except (pa.ArrowInvalid, pa.ArrowTypeError):
                    conflicts.add(f)
        for f, types in per_field.items():
            if f in conflicts or len(set(types)) < 2:
                continue
            try:
                pa.unify_schemas([pa.schema([(f, t)]) for t in types],
                                 promote_options='permissive')
            except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError):
                conflicts.add(f)
        return conflicts

    def _render_parquet(self, dest: str) -> None:
        fields = list(self._fields)
        stringify: set = set()
        try:
            schema = self._parquet_schema(fields, stringify)
        except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError):
            stringify = self._conflicting_fields(fields)
            logging.warning(f"Parquet output {self.output_path}: fields with mixed types "
                            f"stored as strings: {sorted(stringify)}")
            schema = self._parquet_schema(fields, stringify)
        with pq.ParquetWriter(dest, schema) as writer:
            for batch in self._staged_batches(fields, stringify):
                writer.write_table(pa.Table.from_pylist(batch, schema=schema))

    def _write_buffered(self) -> None:
        if self.fmt in {'.xlsx', '.xls'}:
            if xlsxwriter:
                wb = xlsxwriter.Workbook(self.output_path, {'constant_memory': True})
                ws = wb.add_worksheet()
                if self._buffer:
                    headers = list(dict.fromkeys(k for row in self._buffer for k in row))
                    for c, h in enumerate(headers):
                        ws.write(0, c, h)
                    for r, row in enumerate(self._buffer, 1):
                        for c, h in enumerate(headers):
                            v = row.get(h)
                            if v is not None and not isinstance(v, (str, int, float, bool)):
                                v = json.dumps(v, ensure_ascii=False, default=str)
                            ws.write(r, c, v)
                wb.close()
            else:
                pd.DataFrame.from_records(self._buffer).to_excel(self.output_path, index=False)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        try:
            if exc_type is None:
                if self._staged:
                    self._file.close()
                    self._tmp_path = _tmp_sibling(self.output_path, 'tmp')
                    if self.fmt == '.csv':
                        self._render_csv(self._tmp_path)
                    else:
                        self._render_parquet(self._tmp_path)
                    os.replace(self._tmp_path, self.output_path)
                    self._tmp_path = None
                elif self.fmt == '.json' and self._file is not None:
                    self._file.write('\n]\n')
                    if self._tmp_path:
                        self._file.close()
                        os.replace(self._tmp_path, self.output_path)
                        self._tmp_path = None
                elif self.fmt in _BUFFERED_FORMATS:
                    self._write_buffered()
        finally:
            if self._file is not None and self._file not in (sys.stdout, sys.stderr):
                try:
                    self._file.close()
                except Exception:
                    pass
            for leftover in (self._stage_path, self._tmp_path):
                if leftover:
                    try:
                        os.remove(leftover)
                    except OSError:
                        pass
            self._stage_path = self._tmp_path = None


def _derive_path(base: str, tag: str) -> str:
    """Insert a tag before the extension: out.jsonl.gz + '00001' → out.00001.jsonl.gz"""
    p = Path(base)
    n_suffixes = 2 if p.name.lower().endswith('.gz') and len(p.suffixes) >= 2 else 1
    suffixes = ''.join(p.suffixes[-n_suffixes:]) if p.suffix else ''
    stem = p.name[:-len(suffixes)] if suffixes else p.name
    return str(p.with_name(f"{stem}.{tag}{suffixes}"))


class ShardedWriter:
    """Split output into fixed-size shards: out.jsonl → out.00000.jsonl, out.00001.jsonl, …"""

    def __init__(self, output_path: str, fmt: str, encoding: str = 'utf-8',
                 shard_size: int = 100_000, txt_fallback_field: Optional[str] = None) -> None:
        if output_path == _STDOUT:
            raise ConfigurationError("--shard-size cannot be used with stdout output.")
        if shard_size < 1:
            raise ConfigurationError("--shard-size must be >= 1.")
        self.output_path, self.fmt, self.encoding = output_path, fmt, encoding
        self.shard_size = shard_size
        self.txt_fallback_field = txt_fallback_field
        self._shard_index = 0
        self._in_shard = 0
        self._writer: Optional[StreamingWriter] = None

    def __enter__(self) -> 'ShardedWriter':
        self._open_next()
        return self

    def _open_next(self) -> None:
        path = _derive_path(self.output_path, f"{self._shard_index:05d}")
        self._writer = StreamingWriter(path, self.fmt, self.encoding,
                                       txt_fallback_field=self.txt_fallback_field)
        self._writer.__enter__()
        self._in_shard = 0

    def write(self, record: Dict[str, Any]) -> None:
        if self._in_shard >= self.shard_size:
            self._writer.__exit__(None, None, None)
            self._shard_index += 1
            self._open_next()
        self._writer.write(record)
        self._in_shard += 1

    def flush(self) -> None:
        if self._writer is not None:
            self._writer.flush()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._writer is not None:
            self._writer.__exit__(exc_type, exc_val, exc_tb)


def parse_split_spec(spec: str) -> Dict[str, float]:
    """Parse 'train=0.9,val=0.05,test=0.05' into a validated {name: ratio} dict."""
    result: Dict[str, float] = {}
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '=' not in part:
            raise ConfigurationError(f"Invalid split entry '{part}'. Expected name=ratio.")
        name, _, ratio_s = part.partition('=')
        name = name.strip()
        if not name or not name.replace('_', '').replace('-', '').isalnum():
            raise ConfigurationError(f"Invalid split name '{name}'.")
        if name in result:
            raise ConfigurationError(f"Duplicate split name '{name}'.")
        try:
            ratio = float(ratio_s)
        except ValueError:
            raise ConfigurationError(f"Invalid split ratio '{ratio_s}' for '{name}'.") from None
        if not 0 < ratio <= 1:
            raise ConfigurationError(f"Split ratio for '{name}' must be in (0, 1].")
        result[name] = ratio
    if len(result) < 2:
        raise ConfigurationError("--split needs at least two parts, e.g. train=0.9,val=0.1")
    total = sum(result.values())
    if abs(total - 1.0) > 1e-6:
        raise ConfigurationError(f"Split ratios must sum to 1.0 (got {total:.4f}).")
    return result


class SplitWriter:
    """Randomly route records into named splits: out.jsonl → out.train.jsonl, out.val.jsonl, …"""

    def __init__(self, output_path: str, fmt: str, encoding: str = 'utf-8',
                 split_spec: Optional[Dict[str, float]] = None,
                 txt_fallback_field: Optional[str] = None) -> None:
        if output_path == _STDOUT:
            raise ConfigurationError("--split cannot be used with stdout output.")
        if not split_spec:
            raise ConfigurationError("SplitWriter requires a split spec.")
        self.output_path, self.fmt, self.encoding = output_path, fmt, encoding
        self.txt_fallback_field = txt_fallback_field
        self._names: List[str] = list(split_spec.keys())
        self._cumulative: List[float] = []
        acc = 0.0
        for name in self._names:
            acc += split_spec[name]
            self._cumulative.append(acc)
        self._cumulative[-1] = 1.0
        self._writers: Dict[str, StreamingWriter] = {}

    def __enter__(self) -> 'SplitWriter':
        for name in self._names:
            w = StreamingWriter(_derive_path(self.output_path, name), self.fmt, self.encoding,
                                txt_fallback_field=self.txt_fallback_field)
            w.__enter__()
            self._writers[name] = w
        return self

    def write(self, record: Dict[str, Any]) -> None:
        r = random.random()
        for name, edge in zip(self._names, self._cumulative):
            if r <= edge:
                self._writers[name].write(record)
                return
        self._writers[self._names[-1]].write(record)

    def flush(self) -> None:
        for w in self._writers.values():
            w.flush()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        errors: List[BaseException] = []
        for w in self._writers.values():
            try:
                w.__exit__(exc_type, exc_val, exc_tb)
            except Exception as exc:
                errors.append(exc)
        if errors and exc_type is None:
            raise errors[0]
