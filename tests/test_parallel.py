"""--jobs N: keyed pseudonyms, report samples and bounded dispatch."""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from sanitizer_pro.pii import PseudoRegistry, redact_pii
from sanitizer_pro.settings import SanitizerConfig

REPO = Path(__file__).resolve().parent.parent
TEXT = "Mail maria@example.com or call +44 20 7946 0958 from 192.0.2.7."


def run_cli(*argv, env=None):
    return subprocess.run([sys.executable, "-m", "sanitizer_pro", *argv], capture_output=True,
                          text=True, cwd=str(REPO), env={**os.environ, **(env or {})})


class TestKeyedPseudonyms:
    def test_same_key_same_pseudonyms_without_shared_state(self):
        a, b = PseudoRegistry(key="k"), PseudoRegistry(key="k")
        redact_pii("first: bob@example.org", pseudo_registry=a)   # different history
        assert redact_pii(TEXT, pseudo_registry=a) == redact_pii(TEXT, pseudo_registry=b)

    def test_different_key_different_pseudonyms(self):
        assert (redact_pii(TEXT, pseudo_registry=PseudoRegistry(key="k1"))
                != redact_pii(TEXT, pseudo_registry=PseudoRegistry(key="k2")))

    def test_shapes_are_kept(self):
        out = redact_pii(TEXT, pseudo_registry=PseudoRegistry(key="k"))
        assert "@redacted.local" in out and " 10." in out and "phone_" in out

    def test_unkeyed_numbering_unchanged(self):
        out = redact_pii(TEXT, pseudo_registry=PseudoRegistry())
        assert "email_0001@redacted.local" in out and "10.0.0.1" in out

    def test_drain_and_merge(self):
        worker = PseudoRegistry(key="k", track_new=True)
        redact_pii(TEXT, pseudo_registry=worker)
        new = worker.drain_new()
        assert len(new) == 3 and worker.drain_new() == []
        parent = PseudoRegistry(key="k")
        parent.merge(new)
        assert parent.to_dict() == worker.to_dict()

    def test_collision_is_logged(self, caplog, monkeypatch):
        reg = PseudoRegistry(key="k")
        monkeypatch.setattr("sanitizer_pro.pii.hmac.new",
                            lambda *a, **k: type("H", (), {"digest": lambda self: b"\0" * 32})())
        reg.get_or_create("a@x.org", "email")
        reg.get_or_create("b@x.org", "email")
        assert "collision" in caplog.text

    def test_key_survives_checkpoint_state(self):
        key = "s3cr3t-key-xyz"
        reg = PseudoRegistry(key=key)
        reg.get_or_create("a@x.org", "email")
        restored = PseudoRegistry.from_state(reg.to_state(), key=key)
        assert restored.get_or_create("b@x.org", "email") == \
            PseudoRegistry(key=key).get_or_create("b@x.org", "email")
        assert key not in json.dumps(reg.to_state())      # never written to checkpoints

    def test_key_not_in_config_repr(self):
        assert "hunter2" not in repr(SanitizerConfig(pii_pseudonymize=True, pseudo_key="hunter2"))


class TestWindow:
    def test_bounds_items_in_flight_and_close_unblocks(self):
        from sanitizer_pro.cli import _Window
        w = _Window(3)
        taken = []
        t = threading.Thread(target=lambda: taken.extend(w.feed(range(100))))
        t.start()
        time.sleep(0.5)
        assert len(taken) == 3          # blocked: nothing marked done
        w.done()
        time.sleep(0.5)
        assert len(taken) == 4
        w.close()
        t.join(timeout=2)
        assert not t.is_alive()


def _records(n=300):
    people = ["ana@example.com", "bo@example.org", "cy@example.net"]
    return [{"text": f"Record {i}: write to {people[i % 3]} about the quarterly energy "
                     f"report and the renewable grid plan."} for i in range(n)]


@pytest.mark.parametrize("suffix", [".jsonl", ".csv"])
def test_pseudonymize_with_jobs_matches_single_process(tmp_path, suffix):
    inp = tmp_path / f"in{suffix}"
    recs = _records()
    if suffix == ".jsonl":
        inp.write_text("".join(json.dumps(r) + "\n" for r in recs))
    else:
        inp.write_text("text\n" + "".join(f'"{r["text"]}"\n' for r in recs))
    outs = {}
    for jobs in ("1", "3"):
        out, mp = tmp_path / f"o{jobs}.jsonl", tmp_path / f"m{jobs}.json"
        r = run_cli("--input", str(inp), "--output", str(out), "--jobs", jobs,
                    "--remove-pii", "--pii-pseudonymize", "--pseudo-map-file", str(mp),
                    "--chunk-size", "7", "--quiet", "--no-progress",
                    env={"SANITIZE_PSEUDO_KEY": "test-key"})
        assert r.returncode == 0, r.stderr
        outs[jobs] = (out.read_text(), json.loads(mp.read_text()))
    assert outs["1"] == outs["3"]
    assert len(outs["1"][1]) == 3


def test_jobs_without_key_is_consistent_within_the_run(tmp_path):
    inp = tmp_path / "in.jsonl"
    inp.write_text("".join(json.dumps(r) + "\n" for r in _records(600)))
    out = tmp_path / "out.jsonl"
    r = run_cli("--input", str(inp), "--output", str(out), "--jobs", "3", "--remove-pii",
                "--pii-pseudonymize", "--quiet", "--no-progress",
                env={"SANITIZE_PSEUDO_KEY": ""})
    assert r.returncode == 0, r.stderr
    emails = {line.split("write to ")[1].split(" about")[0]
              for line in out.read_text().splitlines()}
    assert len(emails) == 3 and all(e.startswith("email_") for e in emails)


def test_report_has_samples_with_jobs(tmp_path):
    inp = tmp_path / "in.jsonl"
    recs = _records(200) + [{"text": "too short"}] * 20
    inp.write_text("".join(json.dumps(r) + "\n" for r in recs))
    report = tmp_path / "r.html"
    r = run_cli("--input", str(inp), "--output", str(tmp_path / "o.jsonl"), "--jobs", "2",
                "--remove-pii", "--report", str(report), "--quiet", "--no-progress")
    assert r.returncode == 0, r.stderr
    html = report.read_text()
    assert "too short" in html                 # a dropped-record sample
    assert "[PII_EMAIL]" in html               # a PII diff, redacted
    assert "ana@example.com" not in html
