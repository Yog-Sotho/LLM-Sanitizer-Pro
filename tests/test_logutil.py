"""Tests for structured JSON logging."""
import json
import logging
import subprocess
import sys
from pathlib import Path

from sanitizer_pro.logutil import JsonLogFormatter

REPO = Path(__file__).resolve().parent.parent


def _record(msg, level=logging.INFO, exc_info=None):
    return logging.LogRecord('test', level, __file__, 1, msg, None, exc_info)


class TestJsonFormatter:
    def test_basic_fields(self):
        out = json.loads(JsonLogFormatter().format(_record("hello")))
        assert out['level'] == 'INFO' and out['msg'] == 'hello'
        assert out['logger'] == 'test' and 'ts' in out

    def test_exception_included(self):
        try:
            raise ValueError("boom")
        except ValueError:
            rec = _record("failed", level=logging.ERROR, exc_info=sys.exc_info())
        out = json.loads(JsonLogFormatter().format(rec))
        assert 'ValueError: boom' in out['exc']

    def test_non_ascii(self):
        out = json.loads(JsonLogFormatter().format(_record("café ☕")))
        assert out['msg'] == 'café ☕'


def test_cli_json_logs_are_valid_jsonl(tmp_path):
    inp, out = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    inp.write_text(json.dumps(
        {"text": "a reasonably long sample record with enough words to pass filters here"}) + '\n')
    r = subprocess.run(
        [sys.executable, '-m', 'sanitizer_pro', '--input', str(inp), '--output', str(out),
         '--log-format', 'json', '--min-chars', '10', '--min-words', '5', '--no-progress'],
        capture_output=True, text=True, cwd=str(REPO))
    assert r.returncode == 0, r.stderr
    events = [json.loads(line) for line in r.stderr.splitlines() if line.strip()]
    assert events, "expected JSON log lines on stderr"
    # every stderr line parses as JSON, and the final one is the completion summary
    assert events[-1]['event'] == 'complete'
    assert events[-1]['total'] == 1 and events[-1]['kept'] == 1
