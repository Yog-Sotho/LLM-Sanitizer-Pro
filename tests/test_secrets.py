"""Tests for secret / credential detection and redaction."""
from sanitizer_pro import Sanitizer, SanitizerConfig
from sanitizer_pro.pii import PseudoRegistry
from sanitizer_pro.secrets import contains_secret, redact_secrets

PRIVATE_KEY = (
    "-----BEGIN RSA PRIVATE KEY-----\n"
    "MIIBOgIBAAJBAKj34GkxFhD90vcNLYLInFEX6Ppy1tPf9Cnzj4p4WGeKLs1Pt8Q\n"
    "uKUpRKfFLfRYC9AIKjbJTWit+CqvjfR?\n"
    "-----END RSA PRIVATE KEY-----")


class TestDetection:
    def test_aws_key(self):
        assert redact_secrets("id AKIAIOSFODNN7EXAMPLE go") == "id [SECRET_AWS_KEY] go"

    def test_github_token(self):
        t = "ghp_" + "a" * 36
        assert redact_secrets(f"tok {t} end") == "tok [SECRET_GITHUB_TOKEN] end"

    def test_openai_key(self):
        assert "[SECRET_OPENAI_KEY]" in redact_secrets("sk-proj-" + "A" * 30)

    def test_anthropic_key(self):
        assert "[SECRET_ANTHROPIC_KEY]" in redact_secrets("sk-ant-" + "B" * 30)

    def test_google_key(self):
        assert "[SECRET_GOOGLE_KEY]" in redact_secrets("AIza" + "0" * 35)

    def test_slack_token(self):
        assert "[SECRET_SLACK_TOKEN]" in redact_secrets("xoxb-" + "1234567890-abcdef")

    def test_jwt(self):
        jwt = "eyJhbGciOiJI.eyJzdWIiOiIxMjM0.SflKxwRJSMeKKF2QT4"
        assert "[SECRET_JWT]" in redact_secrets(f"auth {jwt}")

    def test_private_key_block(self):
        out = redact_secrets(f"key:\n{PRIVATE_KEY}\ndone")
        assert out == "key:\n[SECRET_PRIVATE_KEY]\ndone"

    def test_connection_string_consumes_full_uri(self):
        out = redact_secrets("db postgres://admin:s3cr3t@host:5432/prod end")
        assert out == "db [SECRET_CONNECTION_STRING] end"

    def test_bearer_token(self):
        out = redact_secrets("Authorization: Bearer " + "x" * 30)
        assert "[SECRET_BEARER_TOKEN]" in out

    def test_generic_assignment(self):
        assert "[SECRET_GENERIC]" in redact_secrets('api_key = "aB3xY9zK1mN4pQ7rS2tU5v"')

    def test_clean_prose_untouched(self):
        text = "The committee approved the budget after a long and detailed discussion today."
        assert redact_secrets(text) == text

    def test_no_false_positive_on_uuid(self):
        # a plain UUID should not trip the generic/key patterns
        text = "request id 550e8400-e29b-41d4-a716-446655440000 processed"
        assert redact_secrets(text) == text


class TestModes:
    def test_masking_keeps_tail(self):
        out = redact_secrets("AKIAIOSFODNN7EXAMPLE", mask=True)
        assert out == "[SECRET…MPLE]"

    def test_pseudonymization_stable(self):
        reg = PseudoRegistry()
        out = redact_secrets("k AKIAIOSFODNN7EXAMPLE and AKIAIOSFODNN7EXAMPLE", pseudo_registry=reg)
        assert out.count("AWS_ACCESS_KEY_0001") == 2

    def test_counters(self):
        ctr = {}
        redact_secrets("AKIAIOSFODNN7EXAMPLE ghp_" + "a" * 36, counters=ctr)
        assert ctr == {'aws_access_key': 1, 'github_token': 1}


class TestContains:
    def test_hit_and_miss(self):
        assert contains_secret("here is AKIAIOSFODNN7EXAMPLE")
        assert not contains_secret("here is nothing sensitive at all")


class TestPipelineIntegration:
    def _cfg(self, **kw):
        base = dict(redact_secrets=True, min_chars=10, min_words=3,
                    min_unique_ratio=0.0, min_ascii_ratio=0.0)
        base.update(kw)
        return SanitizerConfig(**base)

    def test_secrets_redacted_without_remove_pii(self):
        with Sanitizer(self._cfg()) as s:
            res = s.process_record(
                {"text": "deploy key AKIAIOSFODNN7EXAMPLE used in the pipeline script"})
        assert "[SECRET_AWS_KEY]" in res.record["text"]
        assert s.stats.pii_counts.get("aws_access_key") == 1

    def test_secrets_and_pii_together(self):
        cfg = self._cfg(remove_pii=True)
        with Sanitizer(cfg) as s:
            res = s.process_record(
                {"text": "mail me at a@b.co with token ghp_" + "a" * 36 + " for access"})
        assert "[SECRET_GITHUB_TOKEN]" in res.record["text"]
        assert "[PII_EMAIL]" in res.record["text"]

    def test_secrets_pseudonymized_via_config(self):
        cfg = self._cfg(pii_pseudonymize=True)
        with Sanitizer(cfg) as s:
            res = s.process_record(
                {"text": "primary key AKIAIOSFODNN7EXAMPLE for the backup service account"})
        assert "AWS_ACCESS_KEY_0001" in res.record["text"]
