"""Streaming writers with crash safety and sharding."""
import csv
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from sanitize_pro.utils import smart_open

try: import pandas as pd
except ImportError: pd = None
try: import xlsxwriter
except ImportError: xlsxwriter = None
try: import pyarrow as pa; import pyarrow.parquet as pq
except ImportError: pa = pq = None

class StreamingWriter:
    def __init__(self, output_path: str, fmt: str, encoding: str = 'utf-8', txt_fallback_field: Optional[str] = None) -> None:
        self.output_path, self.fmt, self.encoding = output_path, fmt, encoding
        self.txt_fallback_field = txt_fallback_field
        self._file = None
        self._tmp_path: Optional[str] = None
        self._csv_writer: Optional[csv.DictWriter] = None
        self._csv_fields: Optional[List[str]] = None
        self._excel_buf: List[Dict[str, Any]] = []
        self._json_first = True
        self._count = 0

    def __enter__(self) -> 'StreamingWriter':
        if self.fmt == '.json':
            self._tmp_path = self.output_path + '.tmp'
            self._file = smart_open(self._tmp_path, 'w', encoding=self.encoding)
            self._file.write('[\n')
        elif self.fmt in {'.jsonl', '.txt', '.csv'}:
            self._file = smart_open(self.output_path, 'w', encoding=self.encoding)
        return self

    def write(self, record: Dict[str, Any]) -> None:
        self._count += 1
        if self.fmt == '.jsonl':
            self._file.write(json.dumps(record, ensure_ascii=False) + '\n')
        elif self.fmt == '.txt':
            text = record.get('text') or record.get(self.txt_fallback_field) or json.dumps(record)
            self._file.write(str(text).replace('\n', ' ') + '\n')
        elif self.fmt == '.json':
            if not self._json_first: self._file.write(',\n')
            self._file.write(json.dumps(record, ensure_ascii=False))
            self._json_first = False
        elif self.fmt == '.csv':
            if not self._csv_writer:
                self._csv_fields = list(record.keys())
                self._csv_writer = csv.DictWriter(self._file, fieldnames=self._csv_fields, extrasaction='ignore')
                self._csv_writer.writeheader()
            self._csv_writer.writerow(record)
        else:
            self._excel_buf.append(record)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        try:
            if exc_type is None:
                if self.fmt == '.json' and self._file:
                    self._file.write('\n]')
                    self._file.close()
                    os.replace(self._tmp_path, self.output_path)
                elif self.fmt in {'.xlsx', '.xls'}:
                    if xlsxwriter:
                        wb = xlsxwriter.Workbook(self.output_path, {'constant_memory': True})
                        ws = wb.add_worksheet()
                        if self._excel_buf:
                            headers = list(self._excel_buf[0].keys())
                            for c, h in enumerate(headers): ws.write(0, c, h)
                            for r, row in enumerate(self._excel_buf, 1):
                                for c, h in enumerate(headers): ws.write(r, c, row.get(h))
                        wb.close()
                    elif pd:
                        pd.DataFrame.from_records(self._excel_buf).to_excel(self.output_path, index=False)
                elif self.fmt == '.parquet' and pa and pq:
                    pq.write_table(pa.Table.from_pylist(self._excel_buf), self.output_path)
        finally:
            if self._file and self._file not in (sys.stdout, sys.stderr):
                self._file.close()
