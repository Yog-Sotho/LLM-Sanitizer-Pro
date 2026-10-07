"""Tests for streaming readers and writers."""
import json

import pytest

from sanitizer_pro.io.readers import read_records
from sanitizer_pro.io.writers import (
    ShardedWriter, SplitWriter, StreamingWriter, parse_split_spec,
)
from sanitizer_pro.utils import ConfigurationError, InputFormatError


class TestReaders:
    def test_jsonl(self, tmp_path):
        p = tmp_path / "in.jsonl"
        p.write_text('{"a": 1}\n\n{"a": 2}\nnot json\n{"a": 3}\n')
        recs = list(read_records(str(p)))
        assert [r["a"] for r in recs] == [1, 2, 3]

    def test_json_array(self, tmp_path):
        p = tmp_path / "in.json"
        p.write_text('[{"a": 1}, {"a": 2}]')
        assert len(list(read_records(str(p)))) == 2

    def test_json_single_object_fallback(self, tmp_path):
        p = tmp_path / "in.json"
        p.write_text('{"a": 1}')
        recs = list(read_records(str(p)))
        assert recs == [{"a": 1}]

    def test_csv_with_header(self, tmp_path):
        p = tmp_path / "in.csv"
        p.write_text("name,age\nalice,30\nbob,25\n")
        recs = list(read_records(str(p)))
        assert recs[0]["name"] == "alice" and recs[1]["age"] == "25"

    def test_csv_no_header(self, tmp_path):
        p = tmp_path / "in.csv"
        p.write_text("alice,30\nbob,25\n")
        recs = list(read_records(str(p), csv_no_header=True))
        assert recs[0] == {"col_0": "alice", "col_1": "30"}
        assert len(recs) == 2

    def test_csv_custom_columns(self, tmp_path):
        p = tmp_path / "in.csv"
        p.write_text("name,age\nalice,30\n")
        recs = list(read_records(str(p), csv_columns=["n", "a"]))
        assert recs == [{"n": "alice", "a": "30"}]

    def test_tsv(self, tmp_path):
        p = tmp_path / "in.tsv"
        p.write_text("name\tage\nalice\t30\n")
        recs = list(read_records(str(p)))
        assert recs == [{"name": "alice", "age": "30"}]

    def test_txt_lines(self, tmp_path):
        p = tmp_path / "in.txt"
        p.write_text("line one\n\nline two\n")
        recs = list(read_records(str(p)))
        assert recs == [{"text": "line one"}, {"text": "line two"}]

    def test_txt_paragraph_mode(self, tmp_path):
        p = tmp_path / "in.txt"
        p.write_text("para one a\npara one b\n\npara two\n")
        recs = list(read_records(str(p), paragraph_mode=True))
        assert recs == [{"text": "para one a para one b"}, {"text": "para two"}]

    def test_gzip_jsonl(self, tmp_path):
        import gzip
        p = tmp_path / "in.jsonl.gz"
        with gzip.open(p, 'wt', encoding='utf-8') as f:
            f.write('{"a": 1}\n')
        recs = list(read_records(str(p), input_format='.jsonl'))
        assert recs == [{"a": 1}]

    def test_unsupported_format(self, tmp_path):
        with pytest.raises(InputFormatError):
            list(read_records(str(tmp_path / "x.pdf")))


class TestStreamingWriter:
    def test_jsonl(self, tmp_path):
        out = str(tmp_path / "out.jsonl")
        with StreamingWriter(out, '.jsonl') as w:
            w.write({"a": 1})
            w.write({"a": 2})
        lines = open(out).read().splitlines()
        assert [json.loads(line)["a"] for line in lines] == [1, 2]

    def test_json_atomic(self, tmp_path):
        out = str(tmp_path / "out.json")
        with StreamingWriter(out, '.json') as w:
            w.write({"a": 1})
            w.write({"a": 2})
        assert json.load(open(out)) == [{"a": 1}, {"a": 2}]
        assert not (tmp_path / "out.json.tmp").exists()

    def test_json_tmp_removed_on_error(self, tmp_path):
        out = str(tmp_path / "out.json")
        with pytest.raises(RuntimeError):
            with StreamingWriter(out, '.json') as w:
                w.write({"a": 1})
                raise RuntimeError("boom")
        assert not (tmp_path / "out.json.tmp").exists()
        assert not (tmp_path / "out.json").exists()

    def test_csv(self, tmp_path):
        out = str(tmp_path / "out.csv")
        with StreamingWriter(out, '.csv') as w:
            w.write({"a": "1", "b": "2"})
        content = open(out).read()
        assert content.startswith("a,b")

    def test_txt_fallback_field(self, tmp_path):
        out = str(tmp_path / "out.txt")
        with StreamingWriter(out, '.txt', txt_fallback_field='content') as w:
            w.write({"content": "hello"})
        assert open(out).read().strip() == "hello"

    def test_unsupported_format_rejected(self, tmp_path):
        with pytest.raises(ConfigurationError):
            StreamingWriter(str(tmp_path / "o.pdf"), '.pdf')


HETERO = [
    {"id": 1, "text": "first"},
    {"id": 2, "text": "second", "extra": "late column"},
    {"id": 3, "meta": {"src": "web", "n": 2}, "tags": ["a", "b"]},
]


def _no_temp_files(tmp_path):
    return [p.name for p in tmp_path.iterdir() if p.name.startswith('.')] == []


class TestColumnPreservation:
    """Records with differing keys must not lose fields in tabular outputs."""

    def test_json_gz_is_really_gzipped(self, tmp_path):
        import gzip
        out = tmp_path / "o.json.gz"
        with StreamingWriter(str(out), '.json') as w:
            w.write({"a": 1})
        with gzip.open(out, 'rt') as fh:
            assert json.load(fh) == [{"a": 1}]
        assert _no_temp_files(tmp_path)

    def test_csv_union_header_and_json_nested(self, tmp_path):
        import csv
        out = tmp_path / "o.csv"
        with StreamingWriter(str(out), '.csv') as w:
            for r in HETERO:
                w.write(r)
        rows = list(csv.DictReader(out.open()))
        assert list(rows[0]) == ["id", "text", "extra", "meta", "tags"]
        assert rows[1]["extra"] == "late column"
        assert json.loads(rows[2]["meta"]) == {"src": "web", "n": 2}
        assert json.loads(rows[2]["tags"]) == ["a", "b"]
        assert rows[0]["extra"] == ""
        assert _no_temp_files(tmp_path)

    def test_csv_gz(self, tmp_path):
        import csv
        import gzip
        out = tmp_path / "o.csv.gz"
        with StreamingWriter(str(out), '.csv') as w:
            for r in HETERO:
                w.write(r)
        with gzip.open(out, 'rt') as fh:
            assert len(list(csv.DictReader(fh))) == 3

    def test_csv_durable_mode_warns_about_dropped_columns(self, tmp_path, caplog):
        out = tmp_path / "o.csv"
        with StreamingWriter(str(out), '.csv', durable=True) as w:
            for r in HETERO:
                w.write(r)
            assert w.dropped_columns == {"extra": 1, "meta": 1, "tags": 1}
        assert "'extra' is not in the header" in caplog.text

    def test_staged_output_absent_on_error(self, tmp_path):
        out = tmp_path / "o.csv"
        with pytest.raises(RuntimeError):
            with StreamingWriter(str(out), '.csv') as w:
                w.write({"a": 1})
                raise RuntimeError("boom")
        assert not out.exists() and _no_temp_files(tmp_path)

    def test_empty_csv(self, tmp_path):
        out = tmp_path / "o.csv"
        with StreamingWriter(str(out), '.csv'):
            pass
        assert out.read_text() == ""

    @pytest.mark.parametrize("batch", [50_000, 1])  # 1 → unify across many batches
    def test_parquet_union_and_promotion(self, tmp_path, monkeypatch, batch):
        pq = pytest.importorskip("pyarrow.parquet")
        import sanitizer_pro.io.writers as writers
        monkeypatch.setattr(writers, '_PARQUET_BATCH', batch)
        out = tmp_path / "o.parquet"
        with StreamingWriter(str(out), '.parquet') as w:
            w.write({"id": 1, "score": 1})
            w.write({"id": 2, "score": 2.5, "extra": "x", "meta": {"a": 1}})
            w.write({"id": 3, "meta": {"b": "y"}})
        rows = pq.read_table(out).to_pylist()
        assert [r["extra"] for r in rows] == [None, "x", None]
        assert [r["score"] for r in rows] == [1.0, 2.5, None]
        assert rows[1]["meta"] == {"a": 1, "b": None} and rows[2]["meta"] == {"a": None, "b": "y"}
        assert _no_temp_files(tmp_path)

    @pytest.mark.parametrize("batch", [50_000, 1])
    def test_parquet_mixed_types_become_strings(self, tmp_path, monkeypatch, batch):
        pq = pytest.importorskip("pyarrow.parquet")
        import sanitizer_pro.io.writers as writers
        monkeypatch.setattr(writers, '_PARQUET_BATCH', batch)
        out = tmp_path / "o.parquet"
        with StreamingWriter(str(out), '.parquet') as w:
            w.write({"v": 1, "ok": 1})
            w.write({"v": "two", "ok": 2})
        rows = pq.read_table(out).to_pylist()
        assert [r["v"] for r in rows] == ["1", "two"]
        assert [r["ok"] for r in rows] == [1, 2]

    def test_parquet_empty(self, tmp_path):
        pq = pytest.importorskip("pyarrow.parquet")
        out = tmp_path / "o.parquet"
        with StreamingWriter(str(out), '.parquet'):
            pass
        assert pq.read_table(out).num_rows == 0

    def test_excel_union_header(self, tmp_path):
        pytest.importorskip("xlsxwriter")
        pd = pytest.importorskip("pandas")
        pytest.importorskip("openpyxl")
        out = tmp_path / "o.xlsx"
        with StreamingWriter(str(out), '.xlsx') as w:
            for r in HETERO:
                w.write(r)
        df = pd.read_excel(out)
        assert list(df.columns) == ["id", "text", "extra", "meta", "tags"]
        assert df["extra"][1] == "late column"


class TestShardedWriter:
    def test_shards(self, tmp_path):
        out = str(tmp_path / "out.jsonl")
        with ShardedWriter(out, '.jsonl', shard_size=2) as w:
            for i in range(5):
                w.write({"i": i})
        shards = sorted(tmp_path.glob("out.*.jsonl"))
        assert len(shards) == 3
        assert len(open(shards[0]).read().splitlines()) == 2
        assert len(open(shards[2]).read().splitlines()) == 1


class TestSplitSpec:
    def test_valid(self):
        spec = parse_split_spec("train=0.8,val=0.1,test=0.1")
        assert spec == {"train": 0.8, "val": 0.1, "test": 0.1}

    def test_must_sum_to_one(self):
        with pytest.raises(ConfigurationError):
            parse_split_spec("train=0.5,val=0.1")

    def test_needs_two_parts(self):
        with pytest.raises(ConfigurationError):
            parse_split_spec("train=1.0")

    def test_bad_ratio(self):
        with pytest.raises(ConfigurationError):
            parse_split_spec("train=abc,val=0.1")

    def test_duplicate_name(self):
        with pytest.raises(ConfigurationError):
            parse_split_spec("train=0.5,train=0.5")


class TestSplitWriter:
    def test_all_records_land_in_exactly_one_split(self, tmp_path):
        import random
        random.seed(0)
        out = str(tmp_path / "out.jsonl")
        spec = parse_split_spec("train=0.8,val=0.2")
        with SplitWriter(out, '.jsonl', split_spec=spec) as w:
            for i in range(200):
                w.write({"i": i})
        train = open(tmp_path / "out.train.jsonl").read().splitlines()
        val = open(tmp_path / "out.val.jsonl").read().splitlines()
        assert len(train) + len(val) == 200
        assert len(train) > len(val)
