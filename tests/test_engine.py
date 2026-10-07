"""Tests for the shared pipeline engine: one config type, one Sanitizer used
by the CLI and the API, worker parity, and content-addressed sampling/splits."""
import dataclasses
import json
import pickle

import pytest

from sanitizer_pro import Sanitizer, SanitizerConfig
from sanitizer_pro.cli import build_parser
from sanitizer_pro.core import RecordTransformer
from sanitizer_pro.io.writers import SplitWriter
from sanitizer_pro.sampling import content_fraction, keep_in_sample
from sanitizer_pro.settings import DEFAULTS
from sanitizer_pro.utils import ConfigurationError

GOOD = ("The committee reviewed the proposal in detail and concluded that the "
        "plan was feasible for the next fiscal year despite budget concerns.")


def records(n=200):
    return [{"id": i, "text": f"{GOOD} Item {i} mail u{i}@example.com."} for i in range(n)]


def relaxed(**kw):
    base = dict(min_chars=10, min_words=3, min_unique_ratio=0.0)
    base.update(kw)
    return SanitizerConfig(**base)


class TestOneConfig:
    def _cli(self, *argv):
        return build_parser().parse_args(['--input', 'x', '--output', 'y', *argv])

    def test_cli_defaults_match_config_defaults(self):
        ns = vars(self._cli())
        for f in dataclasses.fields(SanitizerConfig):
            if f.name in ns and not isinstance(getattr(DEFAULTS, f.name), (list, tuple)) \
                    and ns[f.name] not in ('', None):
                assert ns[f.name] == getattr(DEFAULTS, f.name), f.name

    def test_from_namespace_converts_cli_strings(self):
        cfg = SanitizerConfig.from_namespace(self._cli(
            '--text-fields', 'a, b', '--lang-filter', 'EN,fr', '--chat-roles', 'user,assistant',
            '--decontaminate', 'gsm8k', '--pii-ner-entities', 'person,org', '--sample', '0.5'))
        assert cfg.text_fields == ['a', 'b'] and cfg.lang_filter == ['en', 'fr']
        assert cfg.chat_roles == ('user', 'assistant') and cfg.decontaminate == ['gsm8k']
        assert cfg.pii_ner_entities == ('person', 'org') and cfg.sample == 0.5

    def test_from_namespace_loads_pattern_and_field_files(self, tmp_path):
        pats = tmp_path / "p.json"
        pats.write_text(json.dumps([{"pattern": r"EMP-\d+", "token": "[EMP]"}]))
        fields_ = tmp_path / "f.json"
        fields_.write_text(json.dumps([{"field": "secret", "action": "drop"}]))
        cfg = SanitizerConfig.from_namespace(self._cli(
            '--pii-patterns-file', str(pats), '--field-config', str(fields_)))
        assert cfg.extra_pii_patterns[0][1] == '[EMP]'
        assert cfg.field_ops[1] == {'secret'}

    @pytest.mark.parametrize("kw,msg", [
        (dict(sample=0), "sample"), (dict(sample=1.5), "sample"),
        (dict(dedup_backend='redis'), "dedup_backend"),
        (dict(max_tokens=0), "max_tokens"),
        (dict(validate_chat=True, chat_roles=()), "chat_roles"),
        (dict(decontam_ngram=1), "decontam_ngram"),
    ])
    def test_validate(self, kw, msg):
        with pytest.raises(ConfigurationError, match=msg):
            SanitizerConfig(**kw).validate()

    def test_config_is_picklable_for_workers(self):
        import re
        cfg = relaxed(extra_pii_patterns=[(re.compile('x'), '[X]', 'custom')],
                      field_ops=({}, {'a'}, set(), set()))
        assert pickle.loads(pickle.dumps(cfg)) == cfg


class TestEngine:
    def test_worker_path_matches_in_process_path(self):
        cfg = relaxed(remove_pii=True, redact_secrets=True, deduplicate=True)
        recs = records() + records(20)  # with duplicates
        with Sanitizer(cfg) as a:
            inline = list(a.process(recs))
        worker = RecordTransformer(pickle.loads(pickle.dumps(cfg)))
        with Sanitizer(cfg) as b:
            parallel = []
            for r in recs:
                counts = {}
                parallel += b.feed_transformed(worker.transform(r, None, counts), counts)
            parallel += b.finish()
        assert parallel == inline
        assert a.stats.to_dict() == b.stats.to_dict()

    def test_feed_and_finish_apply_top_percent(self):
        cfg = relaxed(keep_top_percent=10)
        with Sanitizer(cfg) as s:
            streamed = [out for r in records() for out in s.feed(r)]
            assert streamed == []          # buffered until the end
            final = s.finish()
        assert len(final) == 20 and s.stats.kept == 20

    def test_process_record_unchanged_by_top_percent(self):
        with Sanitizer(relaxed(keep_top_percent=10)) as s:
            assert s.process_record(records(1)[0]).kept

    def test_sampling_reason_and_counts(self):
        with Sanitizer(relaxed(sample=0.5, seed=1)) as s:
            results = [s.process_record(r) for r in records()]
        dropped = [r for r in results if not r.kept]
        assert all(r.reason == 'sampled_out' for r in dropped)
        assert s.stats.sampled_out == len(dropped)
        assert 60 < len(dropped) < 140


class TestContentAddressedSampling:
    def test_stable_and_order_independent(self):
        recs = records()
        with Sanitizer(relaxed(sample=0.3, seed=7)) as a:
            fwd = {r['id'] for r in a.process(recs)}
        with Sanitizer(relaxed(sample=0.3, seed=7)) as b:
            rev = {r['id'] for r in b.process(list(reversed(recs)))}
        assert fwd == rev and 30 < len(fwd) < 90

    def test_seed_changes_selection(self):
        rec = records(500)
        a = {r['id'] for r in rec if keep_in_sample(r, 0.5, seed=1)}
        b = {r['id'] for r in rec if keep_in_sample(r, 0.5, seed=2)}
        assert a != b

    def test_fraction_is_uniform_enough(self):
        vals = [content_fraction({"i": i}) for i in range(4000)]
        assert all(0 <= v < 1 for v in vals)
        assert 0.45 < sum(v < 0.5 for v in vals) / len(vals) < 0.55

    def test_split_assignment_independent_of_order_and_filters(self, tmp_path):
        def split_of(recs, name):
            out = tmp_path / name / "o.jsonl"
            out.parent.mkdir()
            spec = {"train": 0.8, "test": 0.2}
            with SplitWriter(str(out), '.jsonl', split_spec=spec, seed=3) as w:
                for r in recs:
                    w.write(r)
            return {part: {json.loads(line)['id'] for line in
                           (out.parent / f"o.{part}.jsonl").read_text().splitlines()}
                    for part in spec}
        recs = records()
        full = split_of(recs, "a")
        subset = split_of(list(reversed(recs[::2])), "b")   # different order, half filtered out
        assert subset["test"] == {i for i in full["test"] if i % 2 == 0}
        assert subset["train"] == {i for i in full["train"] if i % 2 == 0}


def test_version_is_single_sourced():
    import subprocess
    import sys
    from pathlib import Path

    from sanitizer_pro import __version__
    from sanitizer_pro.report import generate_report_html
    repo = Path(__file__).resolve().parent.parent
    r = subprocess.run([sys.executable, '-m', 'sanitizer_pro', '--version'],
                       capture_output=True, text=True, cwd=str(repo))
    assert r.stdout.strip() == f"sanitize {__version__}"
    assert f"v{__version__}</footer>" in generate_report_html({'total': 0, 'kept': 0})
    pyproject = (repo / 'pyproject.toml').read_text()
    assert 'dynamic = ["version"]' in pyproject and 'sanitizer_pro.__version__' in pyproject
