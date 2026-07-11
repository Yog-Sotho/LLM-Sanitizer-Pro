"""Tests for PII redaction, masking, pseudonymization, and text cleaning."""
from sanitizer_pro.pii import clean_text, redact_pii, strip_html, PseudoRegistry


class TestRedaction:
    def test_email(self):
        assert redact_pii("mail me at john.doe@example.com now") == "mail me at [PII_EMAIL] now"

    def test_url(self):
        assert redact_pii("see https://example.com/x?y=1 ok") == "see [PII_URL] ok"
        assert redact_pii("visit www.example.com today") == "visit [PII_URL] today"

    def test_phone(self):
        assert redact_pii("call 555-123-4567 now") == "call [PII_PHONE] now"
        assert redact_pii("intl +1 (555) 123 4567") == "intl [PII_PHONE]"

    def test_card_not_eaten_by_phone(self):
        out = redact_pii("card 4111 1111 1111 1111 end")
        assert out == "card [PII_CARD] end"
        out = redact_pii("card 4111-1111-1111-1111 end")
        assert out == "card [PII_CARD] end"

    def test_ssn(self):
        assert redact_pii("ssn 123-45-6789 end") == "ssn [PII_SSN] end"

    def test_ip(self):
        assert redact_pii("host 192.168.10.20 end") == "host [PII_IP] end"


class TestMasking:
    def test_email_mask(self):
        out = redact_pii("john.doe@example.com", mask=True)
        assert out == "j***e@example.com"

    def test_phone_mask(self):
        assert redact_pii("555-123-4567", mask=True) == "***-***-4567"

    def test_card_mask(self):
        assert redact_pii("4111 1111 1111 1111", mask=True) == "****-****-****-1111"

    def test_ip_mask_keeps_slash16(self):
        assert redact_pii("192.168.10.20", mask=True) == "192.168.***.***"

    def test_ssn_mask(self):
        assert redact_pii("123-45-6789", mask=True) == "***-**-6789"


class TestPseudonymization:
    def test_stable_within_run(self):
        reg = PseudoRegistry()
        a = redact_pii("a@x.com wrote to a@x.com", pseudo_registry=reg)
        assert a.count("email_0001@redacted.local") == 2

    def test_distinct_values_get_distinct_pseudonyms(self):
        reg = PseudoRegistry()
        out = redact_pii("a@x.com and b@y.com", pseudo_registry=reg)
        assert "email_0001@redacted.local" in out
        assert "email_0002@redacted.local" in out

    def test_map_export(self):
        reg = PseudoRegistry()
        redact_pii("a@x.com", pseudo_registry=reg)
        assert reg.to_dict() == {"a@x.com": "email_0001@redacted.local"}


class TestCleanText:
    def test_html_stripped(self):
        assert "bold" in clean_text("<b>bold</b> text")
        assert "<b>" not in clean_text("<b>bold</b> text")

    def test_newlines_preserved(self):
        out = clean_text("line one\nline two", remove_html=False)
        assert out == "line one\nline two"

    def test_excess_blank_lines_collapsed(self):
        out = clean_text("a\n\n\n\n\nb", remove_html=False)
        assert out == "a\n\nb"

    def test_control_chars_removed(self):
        assert clean_text("a\x00b\x07c", remove_html=False) == "a b c"

    def test_nfkc_normalization(self):
        assert clean_text("ﬁle", remove_html=False) == "file"

    def test_horizontal_whitespace_collapsed(self):
        assert clean_text("a    b\t\tc", remove_html=False) == "a b c"


def test_strip_html_nested():
    out = ' '.join(strip_html("<div><p>hello</p> <span>world</span></div>").split())
    assert out == "hello world"
