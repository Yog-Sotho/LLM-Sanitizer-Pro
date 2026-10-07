"""Tests for secret / credential detection and redaction."""
import pytest
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


class TestCoverage:
    """Credential formats the original rule set missed."""

    def test_huggingface_tokens(self):
        # Fixtures are assembled at runtime so no token-shaped literal is committed.
        fake_hf = "hf" + "_" + "AbCdEfGhIj" * 3 + "KlMn"
        assert redact_secrets(f"token {fake_hf}") == "token [SECRET_HF_TOKEN]"
        assert "[SECRET_HF_TOKEN]" in redact_secrets("api_org_" + "Q" * 34)
        # library identifiers with underscores are not tokens
        assert redact_secrets("call hf_hub_download(repo)") == "call hf_hub_download(repo)"

    def test_gitlab_npm_pypi(self):
        assert redact_secrets("glpat" + "-" + "AbCdEfGhIj" * 2) == "[SECRET_GITLAB_TOKEN]"
        assert redact_secrets("npm_" + "a1" * 18) == "[SECRET_NPM_TOKEN]"
        assert redact_secrets("pypi-AgEIcHlwaS5vcmc" + "x" * 60) == "[SECRET_PYPI_TOKEN]"

    def test_aws_secret_key_keeps_variable_name(self):
        out = redact_secrets("aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")
        assert out == "aws_secret_access_key = [SECRET_AWS_SECRET_KEY]"

    def test_azure_account_key(self):
        conn = "DefaultEndpointsProtocol=https;AccountName=acct;AccountKey=" + "A" * 86 + "==;"
        assert redact_secrets(conn) == \
            "DefaultEndpointsProtocol=https;AccountName=acct;AccountKey=[SECRET_AZURE_KEY];"

    @pytest.mark.parametrize("label", ["ENCRYPTED PRIVATE KEY", "PRIVATE KEY",
                                       "OPENSSH PRIVATE KEY", "PGP PRIVATE KEY BLOCK"])
    def test_all_private_key_armors(self, label):
        block = f"-----BEGIN {label}-----\nMIIabc\n-----END {label}-----"
        assert redact_secrets(f"k:\n{block}\nok") == "k:\n[SECRET_PRIVATE_KEY]\nok"

    def test_truncated_private_key_still_redacted(self):
        assert redact_secrets("dump -----BEGIN RSA PRIVATE KEY-----\nMIIBOgIBAAJBAKj34") == \
            "dump [SECRET_PRIVATE_KEY]"

    def test_public_key_untouched(self):
        text = "-----BEGIN PUBLIC KEY-----\nMIIBIjAN\n-----END PUBLIC KEY-----"
        assert redact_secrets(text) == text


class TestPrecisionAndContext:
    def test_connection_string_stops_at_quote(self):
        assert redact_secrets('url = "postgres://u:p@host:5432/db", next') == \
            'url = "[SECRET_CONNECTION_STRING]", next'
        assert redact_secrets("see mysql://root:pw@db/app.") == "see [SECRET_CONNECTION_STRING]."

    def test_mongodb_multi_host_kept_whole(self):
        uri = "mongodb://u:p@h1:27017,h2:27017/db?replicaSet=rs0"
        assert redact_secrets(f"{uri} end") == "[SECRET_CONNECTION_STRING] end"

    def test_generic_keeps_variable_name_and_quotes(self):
        assert redact_secrets('api_key = "aB3xY9zK1mN4pQ7rS2tU5v"') == \
            'api_key = "[SECRET_GENERIC]"'

    def test_generic_unquoted_env_style(self):
        assert redact_secrets("PASSWORD=x7Kq92LmZp4Rt8Wv") == "PASSWORD=[SECRET_GENERIC]"

    @pytest.mark.parametrize("text", [
        'password = get_password_from_env',      # identifier, not a secret
        'api_key = "aaaaaaaaaaaaaaaaaaaa"',       # zero entropy placeholder
        'secret_key = "your-secret-key-here"',    # documentation placeholder
    ])
    def test_generic_low_entropy_ignored(self, text):
        assert redact_secrets(text) == text

    def test_masked_generic_tail_comes_from_value(self):
        out = redact_secrets('api_key = "aB3xY9zK1mN4pQ7rS2tU5v"', mask=True)
        assert out == 'api_key = "[SECRET…tU5v]"'
