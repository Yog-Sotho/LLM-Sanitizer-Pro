"""Run manifest, dataset card and `sanitize diff`."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from sanitizer_pro.card import render_dataset_card, size_category
from sanitizer_pro.diff import diff_runs, has_changes
from sanitizer_pro.manifest import (
    SCHEMA, build_manifest, config_dict, config_hash, load_run, scrub_argv,
)
from sanitizer_pro.settings import SanitizerConfig

REPO = Path(__file__).resolve().parent.parent


def run_cli(*argv):
    return subprocess.run([sys.executable, "-m", "sanitizer_pro", *argv],
                          capture_output=True, text=True, cwd=str(REPO))


@pytest.fixture
def dataset(tmp_path):
    src = tmp_path / "in.jsonl"
    rows = [{"text": f"Record {i} mentions ana{i}@example.com and the renewable energy "
                     f"report for the quarterly grid plan."} for i in range(60)]
    rows += rows[:10]
    src.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return src


def _run(dataset, tmp_path, name, *extra):
    out, manifest = tmp_path / f"{name}.jsonl", tmp_path / f"{name}.manifest.json"
    card = tmp_path / f"{name}.md"
    r = run_cli("--input", str(dataset), "--output", str(out), "--remove-pii", "--deduplicate",
                "--manifest", str(manifest), "--dataset-card", str(card),
                "--quiet", "--no-progress", *extra)
    assert r.returncode == 0, r.stderr
    return json.loads(manifest.read_text()), card.read_text(), manifest


class TestManifest:
    def test_contents(self, dataset, tmp_path):
        m, _, _ = _run(dataset, tmp_path, "a")
        assert m["schema"] == SCHEMA
        assert m["counts"]["total"] == 70 and m["counts"]["deduplicated"] == 10
        assert m["inputs"][0]["sha256"] and m["inputs"][0]["bytes"] == dataset.stat().st_size
        assert m["outputs"][0]["path"].endswith("a.jsonl") and len(m["outputs"][0]["sha256"]) == 64
        assert m["config"]["remove_pii"] is True and len(m["config_sha256"]) == 64
        assert "--remove-pii" in m["run"]["command"]
        assert m["dependencies"].get("tqdm")

    def test_secret_key_never_recorded(self, dataset, tmp_path):
        m, card, path = _run(dataset, tmp_path, "s", "--pii-pseudonymize",
                             "--pseudo-key", "correct-horse-battery")
        raw = path.read_text() + card
        assert "correct-horse-battery" not in raw
        assert m["config"]["pseudo_key"] == "<set>"
        assert m["run"]["command"][m["run"]["command"].index("--pseudo-key") + 1] == "***"

    def test_split_outputs_listed(self, dataset, tmp_path):
        m, _, _ = _run(dataset, tmp_path, "sp", "--split", "train=0.8,test=0.2")
        assert sorted(Path(o["path"]).name for o in m["outputs"]) == \
            ["sp.test.jsonl", "sp.train.jsonl"]

    def test_config_hash_is_stable_and_sensitive(self):
        a = SanitizerConfig(remove_pii=True)
        assert config_hash(a) == config_hash(SanitizerConfig(remove_pii=True))
        assert config_hash(a) != config_hash(SanitizerConfig(remove_pii=True, min_chars=1))
        # The key changes nothing but "set or not": no verifier for it leaks.
        k1 = config_hash(SanitizerConfig(pii_pseudonymize=True, pseudo_key="one"))
        assert k1 == config_hash(SanitizerConfig(pii_pseudonymize=True, pseudo_key="two"))

    def test_scrub_argv(self):
        assert scrub_argv(["x", "--pseudo-key", "s", "--pseudo-key=t", "--other", "v"]) == \
            ["x", "--pseudo-key", "***", "--pseudo-key=***", "--other", "v"]

    def test_config_dict_is_json_safe(self):
        import re
        cfg = SanitizerConfig(extra_pii_patterns=[(re.compile(r"EMP-\d+"), "[E]", "custom")])
        json.dumps(config_dict(cfg))

    def test_build_manifest_without_files(self):
        m = build_manifest(config=SanitizerConfig(), stats={"total": 0, "kept": 0},
                           inputs=["hf://datasets/x/y"], outputs=[], started_at=0,
                           finished_at=1, argv=["sanitize"])
        assert m["inputs"] == [{"uri": "hf://datasets/x/y"}] and m["run"]["duration_s"] == 1


class TestDatasetCard:
    def test_card_has_front_matter_steps_and_counts(self, dataset, tmp_path):
        _, card, _ = _run(dataset, tmp_path, "c")
        assert card.startswith("---\nlicense: other")
        assert "- n<1K" in card and "- pii-redacted" in card and "- deduplicated" in card
        assert "**PII redaction**" in card and "**Exact deduplication**" in card
        assert "| − Duplicates | 10 |" in card
        assert "| email | 70 |" in card              # redaction runs before dedup
        assert "ana1@example.com" not in card
        assert "sanitize --input" in card

    def test_size_categories(self):
        assert size_category(999) == "n<1K" and size_category(1000) == "1K<n<10K"
        assert size_category(5 * 10 ** 9) == "n>1B"

    def test_languages_from_distribution(self):
        card = render_dataset_card({"counts": {"kept": 5, "language_distribution":
                                               {"eng_Latn": 3, "cmn_Hani": 2}}})
        assert "language:\n- en\n- zh" in card


class TestDiff:
    def test_identical_runs(self, dataset, tmp_path):
        _, _, a = _run(dataset, tmp_path, "d1")
        r = run_cli("diff", str(a), str(a), "--fail-on-change")
        assert r.returncode == 0 and "No differences." in r.stdout

    def test_changed_config_and_counts(self, dataset, tmp_path):
        _, _, a = _run(dataset, tmp_path, "e1")
        _, _, b = _run(dataset, tmp_path, "e2", "--min-words", "200")
        r = run_cli("diff", str(a), str(b), "--fail-on-change")
        assert r.returncode == 1
        assert "min_words: 8 → 200" in r.stdout and "kept" in r.stdout
        data = json.loads(run_cli("diff", str(a), str(b), "--json").stdout)
        assert data["changed"] and ["min_words", 8, 200] in data["config"]

    def test_stats_files_and_bad_input(self, tmp_path):
        a, b = tmp_path / "a.json", tmp_path / "b.json"
        a.write_text(json.dumps({"total": 10, "kept": 8, "pii_redactions": {"email": 2}}))
        b.write_text(json.dumps({"total": 10, "kept": 7, "pii_redactions": {"email": 3}}))
        result = diff_runs(load_run(str(a)), load_run(str(b)))
        assert has_changes(result)
        assert ("kept", 8, 7) in result["counts"]["scalars"]
        assert result["counts"]["breakdowns"]["pii_redactions"] == [("email", 2, 3)]
        (tmp_path / "x.json").write_text("[1]")
        assert run_cli("diff", str(a), str(tmp_path / "x.json")).returncode == 2


class TestReleaseTooling:
    def test_cli_reference_is_current(self):
        r = subprocess.run([sys.executable, "scripts/gen_cli_reference.py", "--check"],
                           capture_output=True, text=True, cwd=str(REPO))
        assert r.returncode == 0, r.stderr

    def test_every_option_is_documented(self):
        from sanitizer_pro.cli import build_parser
        parser = build_parser()
        undocumented = [a.option_strings[0] for g in parser._action_groups
                        for a in g._group_actions if a.option_strings and not a.help]
        assert undocumented == []

    def test_release_notes_extracts_one_section(self):
        sys.path.insert(0, str(REPO / "scripts"))
        from release_notes import section
        text = "# Changelog\n\n## 2.0.0 (x)\n\nnew\n\n## 1.0.0\n\nold\n"
        assert section("2.0.0", text) == "new\n"
        assert section("1.0.0", text) == "old\n"
        with pytest.raises(SystemExit):
            section("3.0.0", text)
