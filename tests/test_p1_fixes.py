"""Regression tests for the Phase 1 P1 fixes: malformed-input accounting,
full-text decontamination, structure-preserving truncation, Hub client
hardening, API benchmark names, and config-file validation."""
import argparse
import email.message
import io
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from sanitizer_pro import Sanitizer, SanitizerConfig, hub
from sanitizer_pro.config import apply_config_to_args
from sanitizer_pro.core import TokenTruncator
from sanitizer_pro.io.readers import MalformedRecord, read_records
from sanitizer_pro.utils import ConfigurationError

REPO = Path(__file__).resolve().parent.parent
GOOD = ("The committee reviewed the proposal in detail and concluded that the "
        "plan was feasible for the next fiscal year despite budget concerns.")
BENCH_Q = ("Natalia sold clips to 48 of her friends in April, and then she sold "
           "half as many clips in May. How many clips did Natalia sell altogether "
           "in April and May?")


def run_cli(*argv: str):
    return subprocess.run([sys.executable, '-m', 'sanitizer_pro', *argv],
                          capture_output=True, text=True, cwd=str(REPO))


# -- malformed input ----------------------------------------------------------

class TestMalformedAccounting:
    def test_jsonl_markers_only_on_request(self, tmp_path):
        p = tmp_path / "in.jsonl"
        p.write_text('{"a": 1}\n{bad json\n{"a": 2}\n')
        assert list(read_records(str(p))) == [{"a": 1}, {"a": 2}]
        items = list(read_records(str(p), yield_malformed=True))
        assert isinstance(items[1], MalformedRecord) and items[1].location == "line 2"

    def test_json_array_non_objects_marked(self, tmp_path):
        p = tmp_path / "in.json"
        p.write_text('[{"a": 1}, 5, "x"]')
        items = list(read_records(str(p), yield_malformed=True))
        assert items[0] == {"a": 1}
        assert sum(isinstance(i, MalformedRecord) for i in items) == 2

    def test_cli_counts_bad_lines(self, tmp_path):
        inp, stats = tmp_path / "in.jsonl", tmp_path / "stats.json"
        inp.write_text(json.dumps({"text": GOOD}) + '\n{bad json\n[1, 2]\n')
        r = run_cli('--input', str(inp), '--output', str(tmp_path / 'o.jsonl'),
                    '--stats-file', str(stats), '--no-progress', '--quiet')
        assert r.returncode == 0, r.stderr
        s = json.loads(stats.read_text())
        assert s['total'] == 3 and s['malformed'] == 2 and s['kept'] == 1


# -- decontamination ----------------------------------------------------------

class TestFullTextDecontamination:
    def _sanitizer(self, tmp_path, **kw):
        ref = tmp_path / "bench.jsonl"
        ref.write_text(json.dumps({"question": BENCH_Q}) + '\n')
        cfg = SanitizerConfig(min_chars=1, min_words=1, min_unique_ratio=0.0,
                              decontam_refs=[str(ref)], **kw)
        return Sanitizer(cfg)

    def test_contamination_past_the_8k_quality_slice(self, tmp_path):
        filler = ' '.join(f"word{i}" for i in range(2500))  # > 8192 chars
        with self._sanitizer(tmp_path, max_chars=10 ** 6) as s:
            r = s.process_record({"text": f"{filler} {BENCH_Q}"})
        assert not r.kept and r.reason == 'contaminated'

    def test_contamination_outside_text_fields(self, tmp_path):
        with self._sanitizer(tmp_path, text_fields=['instruction']) as s:
            r = s.process_record({"instruction": "Solve the following problem.",
                                  "output": BENCH_Q})
        assert not r.kept and r.reason == 'contaminated'

    def test_api_resolves_benchmark_names(self, monkeypatch):
        seen = {}
        from sanitizer_pro import decontam

        def fake_build_index(benchmarks=None, **kw):
            seen['names'] = benchmarks
            return decontam.NGramIndex()
        monkeypatch.setattr(decontam, 'build_index', fake_build_index)
        Sanitizer(SanitizerConfig(decontaminate=['all'])).close()
        assert seen['names'] == list(decontam.KNOWN_BENCHMARKS)
        with pytest.raises(ConfigurationError, match="Unknown benchmark"):
            Sanitizer(SanitizerConfig(decontaminate=['mmlu', 'nope']))


# -- truncation ---------------------------------------------------------------

def test_truncation_preserves_structure():
    t = TokenTruncator(4)
    assert t.truncate("def f(x):\n    return x + 1\n\nprint(f(2))") == "def f(x):\n    return x"
    assert t.truncate("a  b\nc") == "a  b\nc"  # under the limit: unchanged


# -- Hub client ---------------------------------------------------------------

class TestHubClient:
    @pytest.mark.parametrize("url,ok", [
        ("https://huggingface.co/api/x", True),
        ("https://cdn-lfs.huggingface.co/x", True),
        ("https://cas-bridge.xethub.hf.co/x", True),
        ("https://evil.example/?huggingface.co", False),
        ("https://huggingface.co.evil.example/x", False),
        ("https://s3.amazonaws.com/bucket/x", False),
    ])
    def test_is_hf_host(self, url, ok):
        assert hub.is_hf_host(url) is ok

    def test_token_only_sent_to_hf(self, monkeypatch):
        monkeypatch.setenv('HF_TOKEN', 'tok')
        sent = {}

        class FakeOpener:
            def open(self, req, timeout):
                sent[req.full_url] = req.get_header('Authorization')
                return io.BytesIO(b'{}')
        monkeypatch.setattr(urllib.request, 'build_opener', lambda *h: FakeOpener())
        hub.http_get("https://huggingface.co/api/a")
        hub.http_get("https://evil.example/?huggingface.co")
        assert sent == {"https://huggingface.co/api/a": "Bearer tok",
                        "https://evil.example/?huggingface.co": None}

    def test_redirect_to_foreign_host_drops_token(self):
        req = urllib.request.Request("https://huggingface.co/a",
                                     headers={'Authorization': 'Bearer tok'})
        h = hub._HFRedirectHandler()
        foreign = h.redirect_request(req, None, 302, 'Found', {}, "https://s3.example/x")
        same = h.redirect_request(req, None, 302, 'Found', {}, "https://cdn-lfs.huggingface.co/x")
        assert foreign.get_header('Authorization') is None
        assert same.get_header('Authorization') == 'Bearer tok'

    def _http_error(self, code):
        return urllib.error.HTTPError("u", code, "err", email.message.Message(), None)

    def test_retries_then_succeeds(self, monkeypatch):
        calls, sleeps = [], []
        responses = [self._http_error(503), urllib.error.URLError("reset"), b"ok"]

        def fake_open(url, timeout):
            calls.append(url)
            r = responses.pop(0)
            if isinstance(r, Exception):
                raise r
            return io.BytesIO(r)
        monkeypatch.setattr(hub, '_open', fake_open)
        monkeypatch.setattr(hub, '_sleep', sleeps.append)
        assert hub.http_get("https://huggingface.co/x") == b"ok"
        assert len(calls) == 3 and sleeps == [2.0, 4.0]

    def test_client_errors_are_not_retried(self, monkeypatch):
        calls = []

        def fake_open(url, timeout):
            calls.append(url)
            raise self._http_error(404)
        monkeypatch.setattr(hub, '_open', fake_open)
        monkeypatch.setattr(hub, '_sleep', lambda s: None)
        with pytest.raises(urllib.error.HTTPError):
            hub.http_get("https://huggingface.co/x")
        assert len(calls) == 1

    def test_truncated_download_never_cached(self, tmp_path, monkeypatch):
        class Resp(io.BytesIO):
            headers = {'Content-Length': '10'}
        monkeypatch.setattr(hub, '_open', lambda url, timeout: Resp(b'12345'))
        monkeypatch.setattr(hub, '_sleep', lambda s: None)
        dest = tmp_path / "part-000.parquet"
        with pytest.raises(ConnectionError, match="incomplete"):
            hub.http_download("https://huggingface.co/x", dest)
        assert not dest.exists() and not list(tmp_path.iterdir())


# -- config files -------------------------------------------------------------

class TestConfigValidation:
    def _parser(self):
        from sanitizer_pro.cli import build_parser
        return build_parser()

    def _apply(self, cfg):
        parser = self._parser()
        args = parser.parse_args(['--input', 'x', '--output', 'y'])
        apply_config_to_args(args, cfg, set(), parser)
        return args

    def test_unknown_key_is_an_error_with_hint(self):
        with pytest.raises(ConfigurationError, match="did you mean 'remove_pii'"):
            self._apply({'remove_pi': True})

    def test_dashed_keys_accepted(self):
        assert self._apply({'remove-pii': True}).remove_pii is True

    def test_numbers_and_bools_coerced(self):
        args = self._apply({'min_chars': "30", 'min_unique_ratio': 0, 'deduplicate': "yes"})
        assert args.min_chars == 30 and args.min_unique_ratio == 0.0
        assert args.deduplicate is True

    @pytest.mark.parametrize("cfg,msg", [
        ({'min_chars': "lots"}, "number"),
        ({'min_chars': True}, "number"),
        ({'remove_pii': "maybe"}, "true/false"),
        ({'dedup_backend': "redis"}, "one of"),
    ])
    def test_bad_values_rejected(self, cfg, msg):
        with pytest.raises(ConfigurationError, match=msg):
            self._apply(cfg)

    def test_explicit_flags_still_win(self):
        parser = self._parser()
        args = parser.parse_args(['--input', 'x', '--output', 'y', '--min-chars', '5'])
        apply_config_to_args(args, {'min_chars': 99}, {'min_chars'}, parser)
        assert args.min_chars == 5

    def test_generated_template_round_trips(self, tmp_path):
        r = run_cli('--generate-config', 'json')
        cfg = tmp_path / "cfg.json"
        cfg.write_text(r.stdout)
        inp = tmp_path / "in.jsonl"
        inp.write_text(json.dumps({"text": GOOD}) + '\n')
        r2 = run_cli('--input', str(inp), '--output', str(tmp_path / 'o.jsonl'),
                     '--config', str(cfg), '--no-progress', '--quiet')
        assert r2.returncode == 0, r2.stderr

    def test_cli_reports_typo(self, tmp_path):
        cfg = tmp_path / "cfg.json"
        cfg.write_text(json.dumps({'remove_pi': True}))
        r = run_cli('--input', 'x.jsonl', '--output', str(tmp_path / 'o.jsonl'),
                    '--config', str(cfg))
        assert r.returncode == 1 and "Unknown config key" in r.stderr


def test_argparse_namespace_unaffected_without_parser():
    args = argparse.Namespace(a=1)
    apply_config_to_args(args, {'a': "2"}, set())
    assert args.a == "2"  # no parser → no coercion (library callers keep old behavior)
