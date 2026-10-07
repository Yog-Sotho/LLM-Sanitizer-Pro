"""Tests for the audit report (PII counters, sample collection, HTML generation)."""

from sanitizer_pro import Sanitizer, SanitizerConfig
from sanitizer_pro.pii import redact_pii
from sanitizer_pro.report import AuditSampleCollector, generate_report_html

GOOD = ("The committee reviewed the proposal in detail and concluded that the "
        "plan was feasible for the next fiscal year despite budget concerns.")


def relaxed(**overrides) -> SanitizerConfig:
    base = dict(min_chars=10, min_words=3, min_unique_ratio=0.0, min_ascii_ratio=0.0)
    base.update(overrides)
    return SanitizerConfig(**base)


class TestPiiCounters:
    def test_redact_pii_counts_by_kind(self):
        counters = {}
        redact_pii("mail a@b.co and c@d.co, call 555-123-4567", counters=counters)
        assert counters == {'email': 2, 'phone': 1}

    def test_counts_flow_into_stats(self):
        with Sanitizer(relaxed(remove_pii=True)) as s:
            s.process_record({"text": "write to a@b.co or visit https://example.com today please"})
            s.process_record({"text": "my server is at 10.0.0.1 and answers on port eighty"})
        assert s.stats.pii_counts == {'email': 1, 'url': 1, 'ip': 1}
        assert s.stats.to_dict()['pii_redactions'] == {'email': 1, 'url': 1, 'ip': 1}


class TestSampleCollector:
    def test_dropped_capped_per_reason(self):
        c = AuditSampleCollector(max_per_reason=2)
        for i in range(5):
            c.add_dropped('quality', {"i": i})
        assert len(c.dropped['quality']) == 2

    def test_snippet_truncation(self):
        c = AuditSampleCollector()
        c.add_dropped('quality', {"text": "x" * 2000})
        assert len(c.dropped['quality'][0]) < 500

    def test_pii_diff_capped(self):
        c = AuditSampleCollector(max_per_reason=1)
        c.add_pii_diff({"a": 1}, {"a": 2})
        c.add_pii_diff({"b": 1}, {"b": 2})
        assert len(c.pii_diffs) == 1 and not c.wants_pii_diffs


class TestSanitizerAudit:
    def test_samples_collected_via_api(self):
        with Sanitizer(relaxed(remove_pii=True, quality_min_score=0.9)) as s:
            s.process_record({"text": "hi"})              # quality drop
            s.process_record({"text": "email a@b.co " + GOOD})  # pii diff (kept or scored)
        assert s.audit_samples.dropped.get('quality')
        assert s.audit_samples.pii_diffs

    def test_report_html_renders(self):
        with Sanitizer(relaxed(remove_pii=True, deduplicate=True)) as s:
            s.process_record({"text": "contact a@b.co about " + GOOD})
            s.process_record({"text": "contact a@b.co about " + GOOD})
            s.process_record({"text": "hi"})
            html = s.report_html(meta={"Input": "x.jsonl", "version": "3.0"})
        assert html.startswith('<!doctype html>')
        assert 'Records processed' in html and 'PII redactions by type' in html
        assert 'Email addresses' in html
        assert 'Deduplication' in html

    def test_report_escapes_content(self):
        with Sanitizer(relaxed()) as s:
            s.process_record({"text": "<script>alert(1)</script>"})
            html = s.report_html()
        assert '<script>alert(1)</script>' not in html

    def test_write_report(self, tmp_path):
        out = tmp_path / "audit.html"
        with Sanitizer(relaxed()) as s:
            s.process_record({"text": GOOD})
            s.write_report(str(out))
        assert out.exists() and 'Sanitization Audit Report' in out.read_text()


class TestReportPrivacy:
    LEAKY = {"text": "mail jane@example.org key AKIAABCDEFGHIJKLMNOP"}

    def test_dropped_samples_are_redacted(self):
        cfg = relaxed(remove_pii=True, redact_secrets=True, min_words=50)  # everything drops
        with Sanitizer(cfg) as s:
            s.process_record(self.LEAKY)
            html = s.report_html()
        assert s.audit_samples.dropped['quality']
        assert 'jane@example.org' not in html and 'AKIAABCDEFGHIJKLMNOP' not in html
        assert '[PII_EMAIL]' in html and '[SECRET_AWS_KEY]' in html

    def test_pii_diff_masks_originals(self):
        with Sanitizer(relaxed(remove_pii=True, redact_secrets=True)) as s:
            s.process_record({"text": "write to jane@example.org about " + GOOD})
            html = s.report_html()
        (before, after), = s.audit_samples.pii_diffs
        assert 'jane@example.org' not in before and 'jane@example.org' not in html
        assert '\u2022' in before and '[PII_EMAIL]' in after
        assert 'write to' in before            # unchanged context is kept

    def test_masked_mode_does_not_leak_through_dropped_samples(self):
        # --pii-mask keeps partial values in the *output*; dropped samples use
        # full tokens so nothing partial leaks into the report.
        cfg = relaxed(remove_pii=True, pii_mask=True, min_words=50)
        with Sanitizer(cfg) as s:
            s.process_record({"text": "call 555-123-4567 now"})
        assert '4567' not in s.audit_samples.dropped['quality'][0]

    def test_pseudonym_map_untouched_by_report_redaction(self):
        cfg = relaxed(remove_pii=True, pii_pseudonymize=True, min_words=50)
        with Sanitizer(cfg) as s:
            s.process_record({"text": "mail jane@example.org"})
            assert list(s.pseudo_registry.to_dict()) == ['jane@example.org']

    def test_raw_opt_in(self):
        cfg = relaxed(remove_pii=True, redact_secrets=True, min_words=50,
                      report_raw_samples=True)
        with Sanitizer(cfg) as s:
            s.process_record(self.LEAKY)
            assert 'jane@example.org' in s.report_html()

    def test_no_redaction_configured_keeps_data_as_is(self):
        with Sanitizer(relaxed(min_words=50)) as s:
            s.process_record({"text": "plain sample"})
        assert 'plain sample' in s.audit_samples.dropped['quality'][0]


def test_mask_removed_only_keeps_text_present_after():
    from sanitizer_pro.report import _mask_removed
    before = '{"t": "id 4111111111111111 for bob@x.io"}'
    after = '{"t": "id [PII_CARD] for [PII_EMAIL]"}'
    masked = _mask_removed(before, after)
    assert '4111' not in masked and 'bob@x.io' not in masked
    assert masked.startswith('{"t": "id ') and ' for ' in masked


def test_generate_report_empty_stats():
    html = generate_report_html({'total': 0, 'kept': 0, 'kept_pct': 0})
    assert 'Records processed' in html
