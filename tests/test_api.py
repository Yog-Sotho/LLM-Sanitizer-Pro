"""Tests for the Python API (Sanitizer / SanitizerConfig / ProcessResult)."""
import json

import pytest

from sanitizer_pro import ProcessResult, Sanitizer, SanitizerConfig
from sanitizer_pro.utils import ConfigurationError

GOOD = ("The committee reviewed the proposal in detail and concluded that the "
        "plan was feasible for the next fiscal year despite budget concerns.")
SPAM = "buy now click here buy now click here buy now click here buy now click here"


def relaxed(**overrides) -> SanitizerConfig:
    base = dict(min_chars=10, min_words=3, min_unique_ratio=0.0, min_ascii_ratio=0.0)
    base.update(overrides)
    return SanitizerConfig(**base)


class TestConfigValidation:
    def test_defaults_valid(self):
        SanitizerConfig().validate()

    def test_bad_min_score(self):
        with pytest.raises(ConfigurationError):
            Sanitizer(SanitizerConfig(quality_min_score=2.0))

    def test_bad_top_percent(self):
        with pytest.raises(ConfigurationError):
            Sanitizer(SanitizerConfig(keep_top_percent=0))


class TestProcessRecord:
    def test_kept_record(self):
        with Sanitizer(relaxed()) as s:
            result = s.process_record({"text": GOOD})
        assert result.kept and result.record == {"text": GOOD}
        assert result.reason is None

    def test_quality_rejection(self):
        with Sanitizer(SanitizerConfig(min_chars=500)) as s:
            result = s.process_record({"text": "too short"})
        assert not result.kept and result.reason == 'quality'
        assert s.stats.filtered_quality == 1

    def test_malformed(self):
        with Sanitizer(relaxed()) as s:
            result = s.process_record("not a dict")
        assert not result.kept and result.reason == 'malformed'
        assert s.stats.malformed == 1

    def test_pii_redaction(self):
        with Sanitizer(relaxed(remove_pii=True)) as s:
            result = s.process_record(
                {"text": "Contact john@example.com about the quarterly numbers soon."})
        assert '[PII_EMAIL]' in result.record['text']

    def test_pseudonymize_and_export(self, tmp_path):
        cfg = relaxed(remove_pii=True, pii_pseudonymize=True)
        with Sanitizer(cfg) as s:
            s.process_record({"text": "Reach me at a@b.co for details about the offer."})
            out = tmp_path / "map.json"
            s.export_pseudonym_map(str(out))
        assert json.loads(out.read_text()) == {"a@b.co": "email_0001@redacted.local"}

    def test_dedup_state_shared_across_calls(self):
        with Sanitizer(relaxed(deduplicate=True)) as s:
            first = s.process_record({"text": GOOD})
            second = s.process_record({"text": GOOD})
        assert first.kept and not second.kept
        assert second.reason == 'duplicate'
        assert s.stats.deduplicated == 1

    def test_chat_validation_reason(self):
        cfg = relaxed(validate_chat=True)
        with Sanitizer(cfg) as s:
            result = s.process_record(
                {"messages": [{"role": "user", "content": "a question with no answer here"}]})
        assert not result.kept and result.reason == 'chat:no_assistant_reply'

    def test_scoring_reason_and_value(self):
        with Sanitizer(relaxed(quality_min_score=0.6)) as s:
            result = s.process_record({"text": SPAM})
        assert not result.kept and result.reason == 'low_score'
        assert result.score is not None and result.score < 0.6

    def test_decontamination(self, tmp_path):
        refs = tmp_path / "refs.txt"
        refs.write_text(GOOD + "\n")
        with Sanitizer(relaxed(decontam_refs=[str(refs)])) as s:
            result = s.process_record({"text": f"prefix {GOOD} suffix"})
        assert not result.kept and result.reason == 'contaminated'


class TestProcessStream:
    def test_basic_stream(self):
        records = [{"text": GOOD}, {"text": "no"}, {"text": GOOD + " Again, differently."}]
        with Sanitizer(relaxed(min_chars=30)) as s:
            kept = list(s.process(records))
        assert len(kept) == 2
        assert s.stats.total == 3 and s.stats.kept == 2

    def test_keep_top_percent_stream(self):
        records = []
        for i in range(4):
            records.append({"id": i * 2, "text": GOOD + f" Extra sentence number {i}."})
            records.append({"id": i * 2 + 1, "text": SPAM + f" spam {i}"})
        with Sanitizer(relaxed(keep_top_percent=50)) as s:
            kept = list(s.process(records))
        assert len(kept) == 4
        assert [r['id'] for r in kept] == [0, 2, 4, 6]  # prose only, input order

    def test_score_annotation(self):
        with Sanitizer(relaxed(quality_score_field='_q')) as s:
            kept = list(s.process([{"text": GOOD}]))
        assert 0.0 <= kept[0]['_q'] <= 1.0

    def test_format_chatml_stream(self):
        cfg = relaxed(format_chatml=True, validate_chat=True)
        records = [
            {"instruction": "Explain tides to a child in simple words.",
             "output": "The moon pulls the ocean, making the water rise and fall."},
            {"instruction": "A prompt with no output cannot train anything."},
        ]
        with Sanitizer(cfg) as s:
            kept = list(s.process(records))
        assert len(kept) == 1
        assert [m['role'] for m in kept[0]['messages']] == ['user', 'assistant']

    def test_stats_schema_matches_cli(self):
        with Sanitizer(relaxed()) as s:
            list(s.process([{"text": GOOD}]))
            report = s.stats.to_dict()
        for key in ('total', 'kept', 'kept_pct', 'deduplicated', 'filtered_quality',
                    'filtered_contaminated', 'chat_invalid_reasons', 'quality_score_mean'):
            assert key in report


def test_process_result_is_dataclass():
    r = ProcessResult(record=None, kept=False, reason='quality')
    assert r.reason == 'quality' and r.score is None


def test_export_pseudonym_map_requires_enablement(tmp_path):
    with Sanitizer(relaxed()) as s:
        with pytest.raises(ConfigurationError):
            s.export_pseudonym_map(str(tmp_path / "m.json"))
