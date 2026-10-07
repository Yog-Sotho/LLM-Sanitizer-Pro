"""Tests for resumable runs (--resume): checkpoint state, append writers, e2e."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from sanitizer_pro.checkpoint import (
    clear_checkpoint, checkpoint_path, input_fingerprint, load_checkpoint, save_checkpoint,
)
from sanitizer_pro.io.writers import StreamingWriter
from sanitizer_pro.pii import PseudoRegistry
from sanitizer_pro.stats import RunStats
from sanitizer_pro.utils import ConfigurationError

REPO = Path(__file__).resolve().parent.parent


class TestStateRoundTrips:
    def test_runstats_roundtrip(self):
        rs = RunStats()
        rs.total = 100
        rs.record_kept("some text with words here", lang="en")
        rs.record_score(0.73)
        rs.merge_pii_counts({'email': 4})
        rs.chat_invalid_reasons['no_assistant_reply'] = 2

        restored = RunStats.from_state(json.loads(json.dumps(rs.to_state())))
        assert restored.total == 100 and restored.kept == 1
        assert restored.pii_counts == {'email': 4}
        assert restored.lang_dist == {'en': 1}
        assert restored.to_dict() == rs.to_dict()

    def test_pseudo_registry_roundtrip(self):
        reg = PseudoRegistry()
        reg.get_or_create("a@b.co", "email")
        restored = PseudoRegistry.from_state(json.loads(json.dumps(reg.to_state())))
        # existing value keeps its pseudonym; new values continue the sequence
        assert restored.get_or_create("a@b.co", "email") == "email_0001@redacted.local"
        assert restored.get_or_create("c@d.co", "email") == "email_0002@redacted.local"


class TestCheckpointFile:
    def test_save_load_clear(self, tmp_path):
        inp = tmp_path / "in.jsonl"
        inp.write_text('{"a": 1}\n')
        out = str(tmp_path / "out.jsonl")
        save_checkpoint(out, input_path=str(inp), records_read=42,
                        stats_state=RunStats().to_state())
        payload = load_checkpoint(out, str(inp))
        assert payload['records_read'] == 42
        clear_checkpoint(out)
        assert load_checkpoint(out, str(inp)) is None

    def test_input_change_rejected(self, tmp_path):
        inp = tmp_path / "in.jsonl"
        inp.write_text('{"a": 1}\n')
        out = str(tmp_path / "out.jsonl")
        save_checkpoint(out, input_path=str(inp), records_read=1,
                        stats_state=RunStats().to_state())
        inp.write_text('{"a": 1}\n{"a": 2}\n')  # input grew — fingerprint mismatch
        with pytest.raises(ConfigurationError, match="different input"):
            load_checkpoint(out, str(inp))

    def test_hf_fingerprint(self):
        fp = input_fingerprint("hf://openai/gsm8k/main/train")
        assert fp == {'kind': 'hf', 'uri': "hf://openai/gsm8k/main/train"}

    def test_corrupt_checkpoint_rejected(self, tmp_path):
        inp = tmp_path / "in.jsonl"
        inp.write_text('{"a": 1}\n')
        out = str(tmp_path / "out.jsonl")
        Path(checkpoint_path(out)).write_text("{not json")
        with pytest.raises(ConfigurationError, match="Corrupt"):
            load_checkpoint(out, str(inp))


class TestAppendWriters:
    def test_jsonl_append(self, tmp_path):
        out = str(tmp_path / "o.jsonl")
        with StreamingWriter(out, '.jsonl') as w:
            w.write({"i": 1})
        with StreamingWriter(out, '.jsonl', append=True) as w:
            w.write({"i": 2})
        assert [json.loads(x)["i"] for x in open(out)] == [1, 2]

    def test_csv_append_recovers_header(self, tmp_path):
        out = str(tmp_path / "o.csv")
        with StreamingWriter(out, '.csv') as w:
            w.write({"a": "1", "b": "2"})
        with StreamingWriter(out, '.csv', append=True) as w:
            w.write({"b": "4", "a": "3"})  # different key order must not matter
        lines = open(out).read().splitlines()
        assert lines == ["a,b", "1,2", "3,4"]

    def test_append_rejected_for_json(self, tmp_path):
        with pytest.raises(ConfigurationError, match="Append"):
            StreamingWriter(str(tmp_path / "o.json"), '.json', append=True)


def run_cli(*argv: str):
    return subprocess.run([sys.executable, '-m', 'sanitizer_pro', *argv],
                          capture_output=True, text=True, cwd=str(REPO))


CRASH_SCRIPT = """
def quality_check(record):
    if record.get("id") == 3:
        raise RuntimeError("simulated crash")
    return True
"""

OK_SCRIPT = """
def quality_check(record):
    return True
"""


class TestResumeEndToEnd:
    def _write_input(self, tmp_path):
        inp = tmp_path / "in.jsonl"
        recs = [{"id": i, "text": f"A sufficiently long unique training record number {i} "
                                  "with plenty of distinct words to pass the filters."}
                for i in range(1, 7)]
        inp.write_text('\n'.join(json.dumps(r) for r in recs) + '\n')
        return inp

    def test_crash_then_resume_produces_complete_output(self, tmp_path):
        inp = self._write_input(tmp_path)
        out = tmp_path / "out.jsonl"
        crash = tmp_path / "crash.py"
        crash.write_text(CRASH_SCRIPT)
        ok = tmp_path / "ok.py"
        ok.write_text(OK_SCRIPT)

        r1 = run_cli('--input', str(inp), '--output', str(out), '--resume',
                     '--checkpoint-interval', '2', '--quality-script', str(crash),
                     '--min-chars', '20', '--min-words', '5', '--no-progress', '--quiet')
        assert r1.returncode == 1
        ckpt = json.loads((tmp_path / "out.jsonl.checkpoint.json").read_text())
        assert ckpt['records_read'] == 2
        assert len(out.read_text().splitlines()) == 2

        r2 = run_cli('--input', str(inp), '--output', str(out), '--resume',
                     '--checkpoint-interval', '2', '--quality-script', str(ok),
                     '--min-chars', '20', '--min-words', '5', '--no-progress', '--quiet')
        assert r2.returncode == 0, r2.stderr
        ids = [json.loads(line)['id'] for line in out.read_text().splitlines()]
        assert ids == [1, 2, 3, 4, 5, 6]  # complete, no duplicates, in order
        assert not (tmp_path / "out.jsonl.checkpoint.json").exists()
        assert 'Total records processed : 6' in r2.stderr

    def test_fresh_run_with_resume_flag_leaves_no_checkpoint(self, tmp_path):
        inp = self._write_input(tmp_path)
        out = tmp_path / "out.jsonl"
        r = run_cli('--input', str(inp), '--output', str(out), '--resume',
                    '--min-chars', '20', '--min-words', '5', '--no-progress', '--quiet')
        assert r.returncode == 0, r.stderr
        assert len(out.read_text().splitlines()) == 6
        assert not (tmp_path / "out.jsonl.checkpoint.json").exists()

    def test_resume_incompatible_with_jobs(self, tmp_path):
        inp = self._write_input(tmp_path)
        r = run_cli('--input', str(inp), '--output', str(tmp_path / "o.jsonl"),
                    '--resume', '--jobs', '2')
        assert r.returncode == 1
        assert 'not compatible' in r.stderr

    def test_resume_incompatible_with_json_output(self, tmp_path):
        inp = self._write_input(tmp_path)
        r = run_cli('--input', str(inp), '--output', str(tmp_path / "o.json"), '--resume')
        assert r.returncode == 1
        assert 'appendable' in r.stderr


HARD_CRASH_SCRIPT = """
import os
def quality_check(record):
    if record.get("id") == {crash_at}:
        os._exit(9)  # like SIGKILL / OOM-kill: no cleanup, no final flush
    return True
"""


class TestHardCrashResume:
    """A hard crash leaves rows written after the last checkpoint on disk
    (buffer flushes). Resume must discard them instead of duplicating."""

    N = 12
    INTERVAL = 4
    CRASH_AT = 11  # checkpoint at 8; records 9-10 (~10 KB) already hit the disk

    def _input(self, tmp_path):
        inp = tmp_path / "in.jsonl"
        recs = [{"id": i, "text": ' '.join(f"w{i}x{j}" for j in range(900))}
                for i in range(1, self.N + 1)]
        inp.write_text('\n'.join(json.dumps(r) for r in recs) + '\n')
        return inp

    def _run(self, tmp_path, inp, out, script, *extra):
        return run_cli('--input', str(inp), '--output', str(out), '--resume',
                       '--checkpoint-interval', str(self.INTERVAL), '--quality-script',
                       str(script), '--min-chars', '20', '--min-words', '5',
                       '--no-progress', '--quiet', *extra)

    def _scripts(self, tmp_path):
        crash = tmp_path / "crash.py"
        crash.write_text(HARD_CRASH_SCRIPT.replace('{crash_at}', str(self.CRASH_AT)))
        ok = tmp_path / "ok.py"
        ok.write_text(OK_SCRIPT)
        return crash, ok

    @staticmethod
    def _ids(text):
        return [json.loads(line)['id'] for line in text.splitlines()]

    def test_no_duplicates_after_hard_crash(self, tmp_path):
        inp, out = self._input(tmp_path), tmp_path / "out.jsonl"
        crash, ok = self._scripts(tmp_path)
        r1 = self._run(tmp_path, inp, out, crash)
        assert r1.returncode == 9
        ckpt = json.loads((tmp_path / "out.jsonl.checkpoint.json").read_text())
        assert ckpt['records_read'] == 8
        # the crash left rows past the checkpoint on disk — the bug's precondition
        assert out.stat().st_size > ckpt['output_bytes']

        r2 = self._run(tmp_path, inp, out, ok)
        assert r2.returncode == 0, r2.stderr
        assert self._ids(out.read_text()) == list(range(1, self.N + 1))
        assert f'Total records processed : {self.N}' in r2.stderr
        assert f'Kept                    : {self.N}' in r2.stderr

    def test_gzip_output_survives_hard_crash(self, tmp_path):
        import gzip
        inp, out = self._input(tmp_path), tmp_path / "out.jsonl.gz"
        crash, ok = self._scripts(tmp_path)
        assert self._run(tmp_path, inp, out, crash).returncode == 9
        r2 = self._run(tmp_path, inp, out, ok)
        assert r2.returncode == 0, r2.stderr
        with gzip.open(out, 'rt') as fh:
            assert self._ids(fh.read()) == list(range(1, self.N + 1))

    def test_sqlite_dedup_hashes_after_checkpoint_are_forgotten(self, tmp_path):
        import sqlite3
        from sanitizer_pro.core import get_record_hash
        inp, out = self._input(tmp_path), tmp_path / "out.jsonl"
        db = tmp_path / "dedup.db"
        crash, ok = self._scripts(tmp_path)
        dedup = ('--deduplicate', '--dedup-backend', 'sqlite', '--dedup-db-path', str(db))
        assert self._run(tmp_path, inp, out, crash, *dedup).returncode == 9
        # Simulate a batch commit that landed after the checkpoint: the hashes
        # of records 9 and 10 are in the DB, but the checkpoint is at 8.
        recs = [json.loads(line) for line in inp.read_text().splitlines()]
        with sqlite3.connect(db) as conn:
            conn.executemany('INSERT OR IGNORE INTO hashes VALUES (?)',
                             [(get_record_hash(recs[i]),) for i in (8, 9)])
        r2 = self._run(tmp_path, inp, out, ok, *dedup)
        assert r2.returncode == 0, r2.stderr
        assert self._ids(out.read_text()) == list(range(1, self.N + 1))
        assert 'Deduplicated            : 0' in r2.stderr

    def test_csv_output_survives_hard_crash(self, tmp_path):
        import csv
        inp, out = self._input(tmp_path), tmp_path / "out.csv"
        crash, ok = self._scripts(tmp_path)
        assert self._run(tmp_path, inp, out, crash).returncode == 9
        r2 = self._run(tmp_path, inp, out, ok)
        assert r2.returncode == 0, r2.stderr
        ids = [int(r['id']) for r in csv.DictReader(out.open())]
        assert ids == list(range(1, self.N + 1))

    def test_sampling_is_identical_to_an_uninterrupted_run(self, tmp_path):
        inp = self._input(tmp_path)
        crash, ok = self._scripts(tmp_path)
        sample = ('--sample', '0.5', '--seed', '7')
        ref = tmp_path / "ref.jsonl"
        assert self._run(tmp_path, inp, ref, ok, *sample).returncode == 0
        out = tmp_path / "out.jsonl"
        assert self._run(tmp_path, inp, out, crash, *sample).returncode == 9
        assert self._run(tmp_path, inp, out, ok, *sample).returncode == 0
        assert self._ids(out.read_text()) == self._ids(ref.read_text())

    def test_output_shorter_than_checkpoint_is_rejected(self, tmp_path):
        inp, out = self._input(tmp_path), tmp_path / "out.jsonl"
        crash, ok = self._scripts(tmp_path)
        assert self._run(tmp_path, inp, out, crash).returncode == 9
        out.write_text('')  # output replaced behind the checkpoint's back
        r2 = self._run(tmp_path, inp, out, ok)
        assert r2.returncode == 1
        assert 'shorter' in r2.stderr


def test_v1_checkpoint_still_loads(tmp_path):
    inp = tmp_path / "in.jsonl"
    inp.write_text('{"a": 1}\n')
    out = str(tmp_path / "out.jsonl")
    Path(checkpoint_path(out)).write_text(json.dumps({
        'version': 1, 'input': input_fingerprint(str(inp)), 'records_read': 1,
        'stats': RunStats().to_state(), 'pseudo': None}))
    assert load_checkpoint(out, str(inp))['records_read'] == 1


class TestSQLiteRollback:
    def test_mark_and_rollback(self, tmp_path):
        from sanitizer_pro.dedup import SQLiteDeduper
        d = SQLiteDeduper(str(tmp_path / "d.db"), batch_size=2)
        for h in ('a', 'b', 'c'):
            d.add(h)
        mark = d.high_water_mark()
        for h in ('d', 'e', 'b'):  # 'b' already known: no new row
            d.add(h)
        d.flush()
        assert d.rollback_to(mark) == 2
        assert [d.contains(h) for h in 'abcde'] == [True, True, True, False, False]
        d.close()
