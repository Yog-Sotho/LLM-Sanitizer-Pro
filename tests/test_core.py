"""Tests for record sanitization, hashing, and LLM formatting."""
import argparse
import datetime

from sanitizer_pro.core import (
    format_chatml, format_instruct, get_record_hash, sanitize_record, TokenTruncator,
)
from sanitizer_pro.utils import FilterReason


def make_args(**overrides) -> argparse.Namespace:
    base = dict(
        clean_html=False, remove_pii=False, pii_mask=False,
        min_chars=1, max_chars=100000, min_words=1,
        min_ascii_ratio=0.0, min_unique_ratio=0.0,
        reject_allcaps=False, reject_code=False, reject_profanity=False,
        max_depth=100, text_fields_depth=20, lang_confidence=0.0,
        format_chatml=False, format_instruct=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


class TestGetRecordHash:
    def test_identical_records_same_hash(self):
        assert get_record_hash({"a": 1}) == get_record_hash({"a": 1})

    def test_key_order_irrelevant(self):
        assert get_record_hash({"a": 1, "b": 2}) == get_record_hash({"b": 2, "a": 1})

    def test_normalization(self):
        h1 = get_record_hash({"t": "Hello   World"}, normalize=True)
        h2 = get_record_hash({"t": "hello world"}, normalize=True)
        assert h1 == h2

    def test_dedup_fields_subset(self):
        h1 = get_record_hash({"t": "x", "id": 1}, dedup_fields=["t"])
        h2 = get_record_hash({"t": "x", "id": 2}, dedup_fields=["t"])
        assert h1 == h2

    def test_non_json_serializable_values(self):
        """Regression: datetimes from Parquet/Excel must not crash hashing."""
        h = get_record_hash({"ts": datetime.datetime(2026, 1, 1)})
        assert isinstance(h, str) and len(h) == 64


class TestSanitizeRecord:
    def test_basic_pass(self):
        rec = {"text": "a perfectly fine record"}
        out, reason, qtext, lang = sanitize_record(rec, make_args())
        assert reason is None and out == rec and "perfectly" in qtext

    def test_non_dict_rejected(self):
        out, reason, _, _ = sanitize_record("not a dict", make_args())
        assert out is None and reason == FilterReason.QUALITY

    def test_quality_min_chars(self):
        out, reason, _, _ = sanitize_record({"text": "hi"}, make_args(min_chars=50))
        assert out is None and reason == FilterReason.QUALITY

    def test_require_fields(self):
        out, reason, _, _ = sanitize_record(
            {"text": "long enough text", "label": ""},
            make_args(), require_fields=["label"])
        assert out is None and reason == FilterReason.REQUIRE

    def test_pii_removed_recursively(self):
        rec = {"nested": {"deep": ["email me: a@b.co plus other words"]}}
        args = make_args(remove_pii=True)
        out, reason, _, _ = sanitize_record(rec, args)
        assert reason is None
        assert out["nested"]["deep"][0] == "email me: [PII_EMAIL] plus other words"

    def test_field_ops_drop_and_rename(self):
        rec = {"secret": "x", "old": "some sufficiently long text value"}
        field_ops = ({"old": "new"}, {"secret"}, set(), set())
        out, reason, _, _ = sanitize_record(rec, make_args(), field_ops=field_ops)
        assert reason is None
        assert "secret" not in out and out["new"] == "some sufficiently long text value"

    def test_profanity_filter(self):
        out, reason, _, _ = sanitize_record(
            {"text": "this is fucking unacceptable content right here"},
            make_args(reject_profanity=True))
        assert out is None and reason == FilterReason.PROFANITY

    def test_code_filter(self):
        out, reason, _, _ = sanitize_record(
            {"text": "def main():\n    import os\n    print(os.name)"},
            make_args(reject_code=True))
        assert out is None and reason == FilterReason.CODE


class TestFormatting:
    def test_chatml_full(self):
        rec = {"system": "be kind", "instruction": "add", "input": "2+2", "output": "4"}
        out = format_chatml(rec)
        roles = [m["role"] for m in out["messages"]]
        assert roles == ["system", "user", "assistant"]
        assert out["messages"][1]["content"] == "add\n2+2"

    def test_chatml_passthrough(self):
        rec = {"messages": [{"role": "user", "content": "hi"}]}
        assert format_chatml(rec) == {"messages": rec["messages"]}

    def test_chatml_alt_keys(self):
        out = format_chatml({"prompt": "q", "response": "a"})
        roles = [m["role"] for m in out["messages"]]
        assert roles == ["user", "assistant"]

    def test_chatml_text_fallback(self):
        out = format_chatml({"text": "just text"})
        assert out["messages"] == [{"role": "user", "content": "just text"}]

    def test_instruct(self):
        out = format_instruct({"prompt": "q", "completion": "a"})
        assert out == {"instruction": "q", "input": "", "output": "a"}


class TestTokenTruncator:
    def test_whitespace_truncation(self):
        t = TokenTruncator(3, 'whitespace')
        assert t.truncate("one two three four five") == "one two three"

    def test_no_truncation_needed(self):
        t = TokenTruncator(10, 'whitespace')
        assert t.truncate("short text") == "short text"
