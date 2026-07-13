"""Streaming writers with crash safety, sharding, and dataset splitting."""
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

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


class StreamingWriter:
    """Write records one at a time; buffered formats (Excel/Parquet) collect and
    materialize on close. JSON output is atomic (tmp file + os.replace)."""

    APPENDABLE_FORMATS = {'.jsonl', '.txt', '.csv'}

    def __init__(self, output_path: str, fmt: str, encoding: str = 'utf-8',
                 txt_fallback_field: Optional[str] = None, append: bool = False) -> None:
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
        self._file: Any = None
        self._tmp_path: Optional[str] = None
        self._csv_writer: Optional[csv.DictWriter] = None
        self._csv_fields: Optional[List[str]] = None
        self._buffer: List[Dict[str, Any]] = []
        self._json_first = True
        self._count = 0

    def __enter__(self) -> 'StreamingWriter':
        if self.fmt == '.json':
            if self.output_path == _STDOUT:
                self._file = sys.stdout
            else:
                self._tmp_path = self.output_path + '.tmp'
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
        if self.fmt == '.jsonl':
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
            self._csv_writer.writerow(record)
        else:
            self._buffer.append(record)

    def flush(self) -> None:
        if self._file is not None and not self._file.closed:
            self._file.flush()

    def _write_buffered(self) -> None:
        if self.fmt in {'.xlsx', '.xls'}:
            if xlsxwriter:
                wb = xlsxwriter.Workbook(self.output_path, {'constant_memory': True})
                ws = wb.add_worksheet()
                if self._buffer:
                    headers = list(self._buffer[0].keys())
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
        elif self.fmt == '.parquet':
            pq.write_table(pa.Table.from_pylist(self._buffer), self.output_path)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        try:
            if exc_type is None:
                if self.fmt == '.json' and self._file is not None:
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
            if self._tmp_path and exc_type is not None:
                try:
                    os.remove(self._tmp_path)
                except OSError:
                    pass


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
