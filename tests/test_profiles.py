"""Tests for named profiles/presets."""
import json
import subprocess
import sys
from pathlib import Path

from sanitizer_pro.profiles import (
    PROFILE_NAMES, PROFILES, describe_profiles, profile_settings,
)

REPO = Path(__file__).resolve().parent.parent


def run_cli(*argv: str):
    return subprocess.run([sys.executable, '-m', 'sanitizer_pro', *argv],
                          capture_output=True, text=True, cwd=str(REPO))


class TestProfileDefs:
    def test_known_profiles(self):
        assert set(PROFILE_NAMES) == {'fine-tune', 'pretrain', 'rag'}

    def test_settings_exclude_note(self):
        for name in PROFILE_NAMES:
            s = profile_settings(name)
            assert '_note' not in s and s

    def test_every_profile_has_a_note(self):
        assert all(PROFILES[n].get('_note') for n in PROFILE_NAMES)

    def test_describe_lists_all(self):
        text = describe_profiles()
        assert all(n in text for n in PROFILE_NAMES)


class TestProfileCLI:
    def _write(self, tmp_path, records):
        p = tmp_path / "in.jsonl"
        p.write_text('\n'.join(json.dumps(r) for r in records) + '\n')
        return p

    def test_list(self):
        r = run_cli('--profile', 'list')
        assert r.returncode == 0 and 'fine-tune' in r.stdout and 'pretrain' in r.stdout

    def test_unknown_profile_errors(self, tmp_path):
        inp = self._write(tmp_path, [{"text": "x"}])
        r = run_cli('--input', str(inp), '--output', str(tmp_path / "o.jsonl"),
                    '--profile', 'nope')
        assert r.returncode == 2
        assert 'profile' in r.stderr.lower()

    def test_fine_tune_applies_pii_and_secrets(self, tmp_path):
        inp = self._write(tmp_path, [
            {"text": "deploy key AKIAIOSFODNN7EXAMPLE and mail a@b.co in this longer sample record"}])
        out = tmp_path / "o.jsonl"
        r = run_cli('--input', str(inp), '--output', str(out), '--profile', 'fine-tune',
                    '--no-progress', '--quiet')
        assert r.returncode == 0, r.stderr
        rec = json.loads(out.read_text().splitlines()[0])
        assert '[SECRET_AWS_KEY]' in rec['text'] and '[PII_EMAIL]' in rec['text']

    def test_explicit_flag_overrides_profile(self, tmp_path):
        # fine-tune sets min_chars=20; overriding to 500 must drop this short record
        inp = self._write(tmp_path, [{"text": "deploy key here in a shortish record now"}])
        out = tmp_path / "o.jsonl"
        stats = tmp_path / "s.json"
        r = run_cli('--input', str(inp), '--output', str(out), '--profile', 'fine-tune',
                    '--min-chars', '500', '--stats-file', str(stats),
                    '--no-progress', '--quiet')
        assert r.returncode == 0, r.stderr
        data = json.loads(stats.read_text())
        assert data['kept'] == 0 and data['filtered_quality'] == 1

    def test_pretrain_length_floor(self, tmp_path):
        # pretrain sets min_chars=200: a short record is dropped by the profile alone
        inp = self._write(tmp_path, [{"text": "too short for pretraining corpus standards here"}])
        out = tmp_path / "o.jsonl"
        r = run_cli('--input', str(inp), '--output', str(out), '--profile', 'pretrain',
                    '--no-progress', '--quiet')
        assert r.returncode == 0, r.stderr
        assert out.read_text().strip() == ''
