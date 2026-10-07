"""Tests for PII redaction, masking, pseudonymization, and text cleaning."""
import pytest

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


class TestStripHtml:
    """--clean-html must remove markup without corrupting non-HTML text."""

    @pytest.mark.parametrize("text", [
        "if x<y and y>z: pass",
        "a<b",
        "x <y",
        "price < 5 dollars and > 3",
        "a<b and c>d",
        "List<String> list = new ArrayList<>();",
        "for (i = 0; i<n; i++) { if (a[i]>max) max = a[i]; }",
    ])
    def test_non_html_unchanged(self, text):
        assert strip_html(text) == text

    def test_inline_tags_do_not_split_words(self):
        assert strip_html("foo<b>bar</b>baz") == "foobarbaz"
        assert clean_text("un<em>believ</em>able") == "unbelievable"

    def test_block_tags_become_line_breaks(self):
        assert clean_text("<p>Line one</p><p>Line two<br>three</p>") == \
            "Line one\n\nLine two\nthree"

    def test_script_style_comments_dropped(self):
        out = clean_text("<!-- x --><style>p{color:red}</style><script>alert(1)</script>"
                         "<p>shown</p>")
        assert out == "shown"

    def test_trailing_text_kept(self):
        assert clean_text("<b>bold</b> then a < b at the end") == "bold then a < b at the end"

    def test_entities_decoded_after_tags(self):
        assert strip_html("AT&amp;T &copy;") == "AT&T ©"
        # escaped markup is content, not markup
        assert strip_html("<p>use &lt;b&gt; for bold</p>").strip() == "use <b> for bold"

    def test_no_catastrophic_backtracking(self):
        import time
        evil = "<div" + " a" * 20000 + "<p>" * 2000 + "</p" * 2000
        t = time.perf_counter()
        strip_html(evil + "</div>")
        assert time.perf_counter() - t < 1.0


class TestPrecision:
    """Things that look numeric but are not PII must survive redaction."""

    @pytest.mark.parametrize("text", [
        "Order 1234567812345678 shipped",          # 16 digits, fails Luhn
        "tracking 9400111899223197428490 issued",  # long ID, not a card
        "timestamp 1696512345 seconds",            # bare 10 digits
        "Upgrade to version 1.2.3.4 now",          # version string
        "build 10.0.19041.1 is out",               # Windows build number
        "OID 1.3.6.1.4.1 registered",              # dotted identifier
        "ssn 000-00-0000 and 666-12-3456 and 900-12-3456 are invalid",
        "localhost 127.0.0.1 and 0.0.0.0",
        "slice a[1::2] and b[::3]",
        "years 2019-2020-2021",
        "score +1 2 3",
    ])
    def test_not_redacted(self, text):
        counters = {}
        assert redact_pii(text, counters=counters) == text
        assert counters == {}

    def test_url_keeps_trailing_punctuation(self):
        assert redact_pii("See https://example.com/page). Next") == "See [PII_URL]). Next"
        assert redact_pii("(see https://en.wikipedia.org/wiki/Foo_(bar))") == "(see [PII_URL])"
        assert redact_pii("go to www.example.com, then") == "go to [PII_URL], then"


class TestRecall:
    def test_ipv6(self):
        assert redact_pii("client 2001:db8:85a3::8a2e:370:7334 connected") == \
            "client [PII_IP] connected"

    def test_iban(self):
        assert redact_pii("IBAN DE89370400440532013000 pay") == "IBAN [PII_IBAN] pay"
        assert redact_pii("acct GB82 WEST 1234 5698 7654 32 ok") == "acct [PII_IBAN] ok"

    def test_invalid_iban_ignored(self):
        assert redact_pii("ref DE00370400440532013000 x") == "ref DE00370400440532013000 x"

    def test_amex_15_digits(self):
        assert redact_pii("amex 3782 822463 10005 ok") == "amex [PII_CARD] ok"

    def test_uk_phone(self):
        assert redact_pii("Call +44 20 7946 0958 today") == "Call [PII_PHONE] today"

    def test_parenthesized_nanp(self):
        assert redact_pii("office (555) 123-4567") == "office [PII_PHONE]"


class TestMaskingNewKinds:
    def test_iban_mask(self):
        assert redact_pii("DE89370400440532013000", mask=True) == "DE**-****-3000"

    def test_ipv6_mask_keeps_prefix(self):
        assert redact_pii("2001:db8:85a3::8a2e:370:7334", mask=True) == "2001:db8:****::"


def test_pseudonym_ips_stay_valid_past_255():
    import ipaddress
    reg = PseudoRegistry()
    ips = [reg.get_or_create(f"192.168.{i // 250}.{i % 250 + 1}", 'ip') for i in range(300)]
    assert len(set(ips)) == 300
    for ip in ips:
        ipaddress.ip_address(ip)  # raises if invalid


def test_registry_templates_are_per_instance():
    from sanitizer_pro.secrets import redact_secrets
    redact_secrets("AKIAABCDEFGHIJKLMNOP", pseudo_registry=PseudoRegistry())
    assert 'aws_access_key' not in PseudoRegistry._TEMPLATES


def test_custom_patterns_bypass_builtin_validators():
    import re
    custom = [(re.compile(r'\bORD-\d{16}\b'), '[ORDER]', 'card')]
    assert redact_pii("ORD-1234567812345678", extra_patterns=custom) == "[ORDER]"
