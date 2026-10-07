"""Pattern prefilters (pii.Gate) must never change what gets redacted."""
import random
import subprocess
import sys

import pytest

from sanitizer_pro import pii, secrets

FRAGMENTS = [
    "maria@example.com", "https://x.example.org/a).", "www.Example.com/p", "+44 20 7946 0958",
    "(415) 555-0123", "4111 1111 1111 1111", "DE89 3704 0044 0532 0130 00", "123-45-6789",
    "192.0.2.7", "2001:db8::8a2e:370:7334", "-----BEGIN RSA PRIVATE KEY-----\nabc\n",
    "AKIA" + "ABCDEFGHIJKLMNOP", "sk-ant-" + "a" * 24, "sk-" + "b" * 24, "AIza" + "c" * 35,
    "xoxb-" + "1234567890ab", "Bearer " + "d" * 30, "postgres://u:p4ss@db.example:5432/x",
    'api_key = "' + "Zq8xLm2Pw9Rt4Yv6" + '"', "AccountKey=" + "e" * 44, "eyJ" + "f" * 10 + ".g" * 2,
    "AWS_SECRET_ACCESS_KEY=" + "h1/" * 13 + "x", "ghp_" + "i" * 36, "hf_" + "j" * 34,
    "PASSWORD: " + "Xk3!" * 5, "Key", "paſsword=" + "Qw7Er8Ty9Ui0Op1A",
    "ﬁle", "naïve", "日本語", "@", "://", "+", "12", "sk-", "Bear", "aws", ":",
]
FILLER = ["the", "data", "is", "fine", ".", ",", "\n", "42", "x", "-", "AKI"]


def _random_text(rng):
    parts = []
    for _ in range(rng.randint(0, 12)):
        parts.append(rng.choice(FRAGMENTS) if rng.random() < 0.4 else rng.choice(FILLER))
    return " ".join(parts)


@pytest.fixture
def ungated(monkeypatch):
    """apply_patterns with every gate disabled: the reference behavior."""
    def run(fn, *args, **kwargs):
        monkeypatch.setattr(pii, "_GATES", {})
        monkeypatch.setattr(pii, "_PLANS", {})
        try:
            return fn(*args, **kwargs)
        finally:
            monkeypatch.undo()
            pii._PLANS.clear()
    return run


@pytest.mark.parametrize("mode", ["token", "mask", "pseudo"])
def test_gates_never_change_redaction(ungated, mode):
    rng = random.Random(7)
    for _ in range(1500):
        text = _random_text(rng)
        kwargs = {"mask": mode == "mask"}
        regs = [pii.PseudoRegistry(), pii.PseudoRegistry()] if mode == "pseudo" else [None, None]
        c1, c2 = {}, {}
        gated = secrets.redact_secrets(
            pii.redact_pii(text, pseudo_registry=regs[0], counters=c1, **kwargs),
            pseudo_registry=regs[0], counters=c1, **kwargs)
        reference = ungated(lambda: secrets.redact_secrets(
            pii.redact_pii(text, pseudo_registry=regs[1], counters=c2, **kwargs),
            pseudo_registry=regs[1], counters=c2, **kwargs))
        assert gated == reference, text
        assert c1 == c2, text


def test_find_spans_unchanged_by_gates(ungated):
    rng = random.Random(11)
    for _ in range(800):
        text = _random_text(rng)
        assert pii.find_pii_spans(text) == ungated(pii.find_pii_spans, text), text


def test_unicode_case_folding_reaches_ignorecase_patterns():
    # The Kelvin sign and long s match 'k'/'s' under re.IGNORECASE; the
    # gate must not skip them on non-ASCII text.
    value = "Qw7Er8Ty9Ui0Op1AsD2"
    for text in (f"paſsword={value}", f"api_Key: {value}"):
        assert value not in secrets.redact_secrets(text), text


def test_custom_patterns_always_run():
    import re
    extra = [(re.compile(r"EMP-\d{4}"), "[EMP]", "custom")]
    assert pii.redact_pii("id EMP-1234 here", extra_patterns=extra) == "id [EMP] here"
    # A different list object with the same content is honored too.
    extra2 = [(re.compile(r"EMP-\d{4}"), "[ID]", "custom")]
    assert pii.redact_pii("id EMP-1234 here", extra_patterns=extra2) == "id [ID] here"


def test_bench_harness_runs(tmp_path):
    out = tmp_path / "bench.json"
    proc = subprocess.run(
        [sys.executable, "-m", "benchmarks.bench", "--records", "300",
         "--scenarios", "passthrough,regex", "--json", str(out)],
        capture_output=True, text=True, check=True)
    assert "regex" in proc.stdout
    import json
    results = json.loads(out.read_text())["results"]
    assert results["regex"]["records"] == 300
