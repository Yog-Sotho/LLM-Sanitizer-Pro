"""Format-agnostic file streaming readers."""
import csv
import gzip
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from sanitizer_pro.utils import InputFormatError, smart_open, _STDIN

try:
    import ijson
except ImportError:
    ijson = None  # type: ignore[assignment]
try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore[assignment]
try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None  # type: ignore[assignment]

# Large text fields (e.g. scraped documents) easily exceed csv's 128 KiB default.
csv.field_size_limit(min(2**31 - 1, 512 * 1024 * 1024))

SUPPORTED_INPUT_FORMATS = {'.jsonl', '.json', '.csv', '.tsv', '.txt', '.parquet', '.xlsx', '.xls'}


def read_records(
    input_path: str, encoding: str = 'utf-8', paragraph_mode: bool = False,
    csv_delimiter: Optional[str] = None, csv_no_header: bool = False,
    csv_columns: Optional[List[str]] = None, excel_sheet: Any = 0,
    excel_warn_mb: float = 100, input_format: Optional[str] = None, json_path: str = 'item'
) -> Iterator[Dict[str, Any]]:
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
                    for item in ijson.items(f_obj, json_path):
                        streamed_any = True
                        if isinstance(item, dict):
                            yield item
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
            yield from (data if isinstance(data, list) else [data])
        return

    with smart_open(input_path, 'r', encoding=encoding) as f:
        if fmt == '.jsonl':
            bad_lines = 0
            for lineno, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    bad_lines += 1
                    if bad_lines <= 5:
                        logging.warning(f"Skipping malformed JSONL line {lineno}: {exc}")
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
