"""Format-agnostic file streaming readers."""
import csv
import gzip
import json
import logging
import os
from pathlib import Path
from typing import Any, Iterator, List, Optional

from sanitizer_pro.utils import InputFormatError, smart_open, _STDIN

try:
    import ijson
except ImportError:
    ijson = None
try:
    import pandas as pd
except ImportError:
    pd = None
try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

# Large text fields (e.g. scraped documents) easily exceed csv's 128 KiB default.
csv.field_size_limit(min(2**31 - 1, 512 * 1024 * 1024))

SUPPORTED_INPUT_FORMATS = {'.jsonl', '.json', '.csv', '.tsv', '.txt', '.parquet', '.xlsx', '.xls'}


class MalformedRecord:
    """Placeholder yielded (with yield_malformed=True) for input that is not a
    JSON object — an unparseable JSONL line or a non-object JSON item — so the
    caller can count it instead of it vanishing."""

    __slots__ = ('location', 'error')

    def __init__(self, location: str, error: str) -> None:
        self.location, self.error = location, error

    def __repr__(self) -> str:
        return f"MalformedRecord({self.location}: {self.error})"


def read_records(
    input_path: str, encoding: str = 'utf-8', paragraph_mode: bool = False,
    csv_delimiter: Optional[str] = None, csv_no_header: bool = False,
    csv_columns: Optional[List[str]] = None, excel_sheet: Any = 0,
    excel_warn_mb: float = 100, input_format: Optional[str] = None, json_path: str = 'item',
    hf_cache: Optional[str] = None, yield_malformed: bool = False
) -> Iterator[Any]:
    """Stream records (dicts) from a file, stdin or hf:// URI.

    With yield_malformed=True, unparseable JSONL lines and non-object JSON
    items are yielded as MalformedRecord markers (counted by the caller);
    otherwise they are skipped with a warning."""
    if input_path.startswith('hf://'):
        from sanitizer_pro.hub import iter_hub_records
        yield from iter_hub_records(input_path, cache_dir=hf_cache)
        return

    fmt = input_format or Path(input_path).suffix.lower()
    if fmt not in SUPPORTED_INPUT_FORMATS:
        raise InputFormatError(
            f"Unsupported input format '{fmt}'. Supported: {sorted(SUPPORTED_INPUT_FORMATS)}")

    if fmt in {'.parquet', '.xlsx', '.xls'} and input_path == _STDIN:
        raise InputFormatError(f"{fmt} input cannot be read from stdin.")

    if fmt == '.parquet':
        if not pq:
            raise ImportError("Parquet requires: pip install pyarrow")
        for batch in pq.ParquetFile(input_path).iter_batches():
            yield from (r for r in batch.to_pylist() if isinstance(r, dict))
        return

    if fmt in {'.xlsx', '.xls'}:
        if not pd:
            raise ImportError("Excel requires: pip install pandas openpyxl")
        if os.path.getsize(input_path) / (1024 * 1024) > excel_warn_mb:
            logging.warning(f"Excel file > {excel_warn_mb}MB. High memory usage expected.")
        yield from pd.read_excel(input_path, sheet_name=excel_sheet).to_dict(orient='records')
        return

    if fmt == '.json':
        is_gz = input_path.lower().endswith('.gz')
        if ijson and input_path != _STDIN:
            streamed_any = False
            try:
                f_obj = gzip.open(input_path, 'rb') if is_gz else open(input_path, 'rb')
                with f_obj:
                    for i, item in enumerate(ijson.items(f_obj, json_path)):
                        streamed_any = True
                        if isinstance(item, dict):
                            yield item
                        elif yield_malformed:
                            yield MalformedRecord(f"item {i}", f"not an object ({type(item).__name__})")
                if streamed_any:
                    return
                logging.debug(f"ijson found no items at path '{json_path}', "
                              "falling back to json.load")
            except Exception as e:
                if streamed_any:
                    raise
                logging.warning(f"ijson streaming failed ({e}), falling back to json.load")
        with smart_open(input_path, 'r', encoding=encoding) as f:
            data = json.load(f)
            for i, item in enumerate(data if isinstance(data, list) else [data]):
                if isinstance(item, dict) or not yield_malformed:
                    yield item
                else:
                    yield MalformedRecord(f"item {i}", f"not an object ({type(item).__name__})")
        return

    with smart_open(input_path, 'r', encoding=encoding) as f:
        if fmt == '.jsonl':
            bad_lines = 0
            for lineno, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    bad_lines += 1
                    if bad_lines <= 5:
                        logging.warning(f"Skipping malformed JSONL line {lineno}: {exc}")
                    if yield_malformed:
                        yield MalformedRecord(f"line {lineno}", str(exc))
                    continue
                yield item
            if bad_lines > 5:
                logging.warning(f"Skipped {bad_lines} malformed JSONL lines in total.")
        elif fmt in {'.csv', '.tsv'}:
            delimiter = csv_delimiter or ('\t' if fmt == '.tsv' else ',')
            fieldnames: Optional[List[str]] = None
            if csv_columns:
                fieldnames = csv_columns
            elif csv_no_header:
                # Peek at the first row to know how many columns to synthesize.
                first = f.readline()
                if not first:
                    return
                width = len(next(csv.reader([first], delimiter=delimiter)))
                fieldnames = [f"col_{i}" for i in range(width)]
                yield from csv.DictReader([first], fieldnames=fieldnames, delimiter=delimiter,
                                          restkey='_extra', restval=None)
            reader = csv.DictReader(f, fieldnames=fieldnames, delimiter=delimiter,
                                    restkey='_extra', restval=None)
            if csv_columns and not csv_no_header:
                next(reader, None)  # user supplied names; skip the file's own header row
            yield from reader
        elif fmt == '.txt':
            if paragraph_mode:
                paras: List[str] = []
                for line in f:
                    if line.strip() == '':
                        if paras:
                            yield {"text": ' '.join(paras)}
                        paras = []
                    else:
                        paras.append(line.strip())
                if paras:
                    yield {"text": ' '.join(paras)}
            else:
                for line in f:
                    if line.strip():
                        yield {"text": line.strip()}
