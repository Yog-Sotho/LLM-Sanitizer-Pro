"""Multi-file input (directories, globs) and parallel chunked JSONL reading."""
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from sanitizer_pro.checkpoint import input_fingerprint
from sanitizer_pro.io.readers import MalformedRecord, read_records
from sanitizer_pro.io.sources import (
    Chunk, chunkable, expand_inputs, is_multi_input, jsonl_chunks, read_chunk,
)

REPO = Path(__file__).resolve().parent.parent


def run_cli(*argv):
    return subprocess.run([sys.executable, "-m", "sanitizer_pro", *argv],
                          capture_output=True, text=True, cwd=str(REPO))


def words(rng, n):
    return " ".join(rng.choice(["alpha", "beta", "gamma", "delta", "pi", "rho", "tau", "mu",
                                "contact", "me", "at", "x@example.com", "+44 20 7946 0958"])
                    for _ in range(n))


@pytest.fixture
def tree(tmp_path):
    (tmp_path / "b").mkdir()
    (tmp_path / ".hidden").mkdir()
    for name in ("a.jsonl", "b/c.jsonl", "b/d.csv", ".hidden/e.jsonl", "notes.md", ".f.jsonl"):
        (tmp_path / name).write_text('{"text": "x"}\n' if name.endswith("jsonl") else "text\nx\n")
    return tmp_path


class TestExpandInputs:
    def test_single_file_and_special_inputs(self, tree):
        assert expand_inputs(str(tree / "a.jsonl")) == [str(tree / "a.jsonl")]
        assert expand_inputs("-") == ["-"]
        assert expand_inputs("hf://datasets/x/y") == ["hf://datasets/x/y"]
        assert not is_multi_input(str(tree / "a.jsonl"))

    def test_directory_is_recursive_sorted_and_skips_hidden(self, tree):
        files = expand_inputs(str(tree))
        assert [Path(f).relative_to(tree).as_posix() for f in files] == \
            ["a.jsonl", "b/c.jsonl", "b/d.csv"]

    def test_glob(self, tree):
        files = expand_inputs(str(tree / "**" / "*.jsonl"))
        assert [Path(f).name for f in files] == ["a.jsonl", "c.jsonl"]

    def test_output_inside_input_dir_is_excluded(self, tree):
        files = expand_inputs(str(tree), exclude=[str(tree / "a.jsonl")])
        assert str(tree / "a.jsonl") not in files

    def test_no_match_is_an_error(self, tmp_path):
        with pytest.raises(ValueError, match="No input files"):
            expand_inputs(str(tmp_path / "*.jsonl"))

    def test_fingerprint_tracks_every_file(self, tree):
        before = input_fingerprint(str(tree))
        assert before["kind"] == "files" and len(before["files"]) == 3
        (tree / "b" / "new.jsonl").write_text('{"text": "y"}\n')
        assert input_fingerprint(str(tree)) != before


def _sequential(path):
    out = []
    for item in read_records(str(path), yield_malformed=True):
        out.append("MALFORMED" if isinstance(item, MalformedRecord) else item)
    return out


class TestChunks:
    @pytest.fixture
    def messy(self, tmp_path):
        rng = random.Random(3)
        lines = []
        for i in range(60):
            kind = rng.random()
            if kind < 0.1:
                lines.append("")
            elif kind < 0.2:
                lines.append("{not json")
            elif kind < 0.25:
                lines.append("[1, 2]")
            else:
                lines.append(json.dumps({"i": i, "text": words(rng, rng.randint(0, 30)),
                                         "u": "é✓日本"}, ensure_ascii=rng.random() < 0.5))
        endings = ["\n", "\r\n", "\r"]
        data = "".join(line + rng.choice(endings) for line in lines)
        path = tmp_path / "messy.jsonl"
        path.write_bytes(data.encode("utf-8"))
        return path

    @pytest.mark.parametrize("chunk_bytes", [1, 7, 64, 333, 1 << 20])
    def test_chunks_reproduce_sequential_reading(self, messy, chunk_bytes):
        got = []
        for chunk in jsonl_chunks([str(messy)], chunk_bytes):
            got.extend("MALFORMED" if isinstance(x, MalformedRecord) else x
                       for x in read_chunk(chunk))
        assert got == _sequential(messy)

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        assert [x for c in jsonl_chunks([str(p)]) for x in read_chunk(c)] == []

    def test_no_trailing_newline(self, tmp_path):
        p = tmp_path / "t.jsonl"
        p.write_text('{"a": 1}\n{"a": 2}')
        got = [x for c in jsonl_chunks([str(p)], 5) for x in read_chunk(c)]
        assert got == [{"a": 1}, {"a": 2}]

    def test_chunkable(self, tmp_path):
        assert chunkable(["a.jsonl", "b.jsonl"])
        assert not chunkable(["a.jsonl.gz"])
        assert not chunkable(["a.jsonl", "b.csv"])
        assert chunkable(["a.txt"], input_format=".jsonl")
        assert not chunkable(["-"])
        assert Chunk("p", 0, 1).path == "p"


class TestCli:
    def _make_inputs(self, root):
        rng = random.Random(9)
        (root / "in" / "nested").mkdir(parents=True)
        recs = [{"text": words(rng, 25)} for _ in range(400)]
        recs += recs[:40]                                  # duplicates across files
        for i, path in enumerate(["in/p1.jsonl", "in/nested/p2.jsonl", "in/p3.jsonl"]):
            part = recs[i::3]
            body = "\n".join(json.dumps(r) for r in part) + "\n\nnot json\n"
            (root / path).write_text(body)
        return root / "in"

    def test_jobs_output_identical_to_single_process(self, tmp_path):
        src = self._make_inputs(tmp_path)
        outs = {}
        for jobs in ("1", "3"):
            out, stats = tmp_path / f"o{jobs}.jsonl", tmp_path / f"s{jobs}.json"
            r = run_cli("--input", str(src), "--output", str(out), "--jobs", jobs,
                        "--remove-pii", "--deduplicate", "--min-words", "3",
                        "--stats-file", str(stats), "--quiet", "--no-progress")
            assert r.returncode == 0, r.stderr
            outs[jobs] = (out.read_text(), json.loads(stats.read_text()))
        assert outs["1"][0] == outs["3"][0]
        s1, s3 = outs["1"][1], outs["3"][1]
        for key in ("total", "kept", "malformed", "deduplicated", "pii_redactions",
                    "word_count_histogram", "char_length_histogram"):
            assert s1[key] == s3[key], key
        assert s1["malformed"] == 3 and s1["deduplicated"] > 0

    def test_glob_input_and_mixed_formats(self, tmp_path):
        (tmp_path / "a.jsonl").write_text(
            json.dumps({"text": "first record with enough words in it"}) + "\n")
        (tmp_path / "b.csv").write_text("text\nsecond record with enough words in it\n")
        out = tmp_path / "out.jsonl"
        r = run_cli("--input", str(tmp_path / "*.*"), "--output", str(out),
                    "--min-chars", "0", "--min-words", "3", "--quiet", "--no-progress")
        assert r.returncode == 0, r.stderr
        texts = [json.loads(line)["text"] for line in out.read_text().splitlines()]
        assert texts == ["first record with enough words in it",
                         "second record with enough words in it"]

    def test_resume_with_directory_input(self, tmp_path):
        src = self._make_inputs(tmp_path)
        out = tmp_path / "out.jsonl"
        args = ("--input", str(src), "--output", str(out), "--resume",
                "--checkpoint-interval", "50", "--min-words", "3", "--quiet", "--no-progress")
        r = run_cli(*args)
        assert r.returncode == 0, r.stderr
        full = out.read_text()
        # Simulate a crash after a checkpoint: re-create one mid-run.
        from sanitizer_pro.checkpoint import save_checkpoint
        from sanitizer_pro.stats import RunStats
        out.write_text("")
        save_checkpoint(str(out), input_path=str(src), records_read=0,
                        stats_state=RunStats().to_state(), output_bytes=0)
        r = run_cli(*args)
        assert r.returncode == 0, r.stderr
        assert out.read_text() == full
        # A changed input set invalidates the checkpoint.
        save_checkpoint(str(out), input_path=str(src), records_read=0,
                        stats_state=RunStats().to_state(), output_bytes=0)
        (src / "p4.jsonl").write_text('{"text": "new file"}\n')
        r = run_cli(*args)
        assert r.returncode != 0 and "different input" in r.stderr
