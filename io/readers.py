"""Format-agnostic file streaming readers."""
import csv
import gzip
import itertools
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from sanitize_pro.utils import InputFormatError, smart_open, _STDIN

try: import ijson
except ImportError: ijson = None
try: import pandas as pd
except ImportError: pd = None
try: import pyarrow.parquet as pq
except ImportError: pq = None

def read_records(
    input_path: str, encoding: str = 'utf-8', paragraph_mode: bool = False,
    csv_delimiter: Optional[str] = None, csv_no_header: bool = False,
    csv_columns: Optional[List[str]] = None, excel_sheet: Any = 0,
    excel_warn_mb: float = 100, input_format: Optional[str] = None, json_path: str = 'item'
) -> Iterator[Dict[str, Any]]:
    fmt = input_format or Path(input_path).suffix.lower()
    if fmt == '.parquet':
        if not pq: raise ImportError("Parquet requires: pip install pyarrow")
        for batch in pq.ParquetFile(input_path).iter_batches():
            yield from (r for r in batch.to_pylist() if isinstance(r, dict))
        return
    if fmt in {'.xlsx', '.xls'}:
        if not pd: raise ImportError("Excel requires: pip install pandas openpyxl")
        if os.path.getsize(input_path) / (1024*1024) > excel_warn_mb:
            logging.warning(f"Excel file > {excel_warn_mb}MB. High memory usage expected.")
        yield from pd.read_excel(input_path, sheet_name=excel_sheet).to_dict(orient='records')
        return
    if fmt == '.json':
        is_gz = input_path.lower().endswith('.gz')
        if ijson:
            try:
                f_obj = gzip.open(input_path, 'rb') if is_gz else open(input_path, 'rb')
                with f_obj:
                    for item in ijson.items(f_obj, json_path):
                        if isinstance(item, dict): yield item
                return
            except Exception as e:
                logging.warning(f"ijson failed ({e}), falling back to json.load")
        with smart_open(input_path, 'r', encoding=encoding) as f:
            data = json.load(f)
            yield from (data if isinstance(data, list) else [data])
        return
        
    with smart_open(input_path, 'r', encoding=encoding) as f:
        if fmt == '.jsonl':
            for line in f:
                if line.strip():
                    try: yield json.loads(line)
                    except json.JSONDecodeError: pass
        elif fmt == '.csv':
            delimiter = csv_delimiter or ','
            reader = csv.DictReader(f, delimiter=delimiter)
            yield from (row for row in reader)
        elif fmt == '.txt':
            if paragraph_mode:
                paras: List[str] = []
                for line in f:
                    if line.strip() == '':
                        if paras: yield {"text": ' '.join(paras)}
                        paras = []
                    else: paras.append(line.strip())
                if paras: yield {"text": ' '.join(paras)}
            else:
                for line in f:
                    if line.strip(): yield {"text": line.strip()}
