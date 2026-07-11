"""End-to-end CLI tests via subprocess."""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def run_cli(*argv: str):
    return subprocess.run(
        [sys.executable, '-m', 'sanitizer_pro', *argv],
        capture_output=True, text=True, cwd=str(REPO),
    )


def write_jsonl(path: Path, records) -> None:
    path.write_text('\n'.join(json.dumps(r) for r in records) + '\n')


SAMPLE = [
    {"text": "Contact john@example.com about the renewable energy report published this quarter."},
    {"text": "Contact john@example.com about the renewable energy report published this quarter."},
    {"text": "A completely different record with enough words to satisfy default filters easily."},
    {"text": "short"},
]


def test_end_to_end_pii_and_dedup(tmp_path):
    inp, out = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    write_jsonl(inp, SAMPLE)
    r = run_cli('--input', str(inp), '--output', str(out),
                '--remove-pii', '--deduplicate', '--min-chars', '20',
                '--min-words', '5', '--no-progress', '--quiet')
    assert r.returncode == 0, r.stderr
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(lines) == 2
    assert '[PII_EMAIL]' in lines[0]['text']


def test_missing_input_errors():
    r = run_cli('--input', '/nonexistent/x.jsonl', '--output', '/tmp/never.jsonl')
    assert r.returncode == 1
    assert 'not found' in r.stderr


def test_requires_input_and_output():
    r = run_cli()
    assert r.returncode == 2  # argparse error


def test_split_and_shard_mutually_exclusive(tmp_path):
    inp = tmp_path / "in.jsonl"
    write_jsonl(inp, SAMPLE)
    r = run_cli('--input', str(inp), '--output', str(tmp_path / 'o.jsonl'),
                '--split', 'train=0.9,val=0.1', '--shard-size', '10')
    assert r.returncode == 1
    assert 'mutually exclusive' in r.stderr


def test_invalid_split_spec(tmp_path):
    inp = tmp_path / "in.jsonl"
    write_jsonl(inp, SAMPLE)
    r = run_cli('--input', str(inp), '--output', str(tmp_path / 'o.jsonl'),
                '--split', 'train=0.5,val=0.1')
    assert r.returncode == 1
    assert 'sum to 1.0' in r.stderr


def test_stats_file(tmp_path):
    inp, out = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    stats = tmp_path / "stats.json"
    write_jsonl(inp, SAMPLE)
    r = run_cli('--input', str(inp), '--output', str(out),
                '--min-chars', '20', '--min-words', '5',
                '--stats-file', str(stats), '--no-progress', '--quiet')
    assert r.returncode == 0, r.stderr
    data = json.loads(stats.read_text())
    assert data['total'] == 4 and data['kept'] == 3


def test_generate_config_json():
    r = run_cli('--generate-config', 'json')
    assert r.returncode == 0
    cfg = json.loads(r.stdout)
    assert 'min_chars' in cfg


def test_parallel_jobs(tmp_path):
    inp, out = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    write_jsonl(inp, SAMPLE * 25)
    r = run_cli('--input', str(inp), '--output', str(out),
                '--jobs', '2', '--deduplicate', '--min-chars', '20',
                '--min-words', '5', '--no-progress', '--quiet')
    assert r.returncode == 0, r.stderr
    lines = out.read_text().splitlines()
    assert len(lines) == 2  # dedup collapses the 100 records to 2 unique


def test_chatml_output(tmp_path):
    inp, out = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    write_jsonl(inp, [{"instruction": "Explain gravity in one clear paragraph for students.",
                       "output": "Gravity is the force that attracts objects toward each other."}])
    r = run_cli('--input', str(inp), '--output', str(out),
                '--format-chatml', '--min-chars', '10', '--min-words', '3',
                '--no-progress', '--quiet')
    assert r.returncode == 0, r.stderr
    rec = json.loads(out.read_text().splitlines()[0])
    assert [m['role'] for m in rec['messages']] == ['user', 'assistant']


def test_decontaminate_list():
    r = run_cli('--decontaminate', 'list')
    assert r.returncode == 0
    assert 'mmlu' in r.stdout and 'gsm8k' in r.stdout


def test_decontaminate_with_local_refs(tmp_path):
    bench_q = ("Natalia sold clips to 48 of her friends in April, and then she sold "
               "half as many clips in May. How many clips did Natalia sell altogether?")
    refs = tmp_path / "refs.txt"
    refs.write_text(bench_q + "\n")
    inp, out = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    stats = tmp_path / "stats.json"
    write_jsonl(inp, [
        {"text": f"Question: {bench_q} Answer: 72"},  # contaminated
        {"text": "A clean training example about photosynthesis with plenty of words to pass filters."},
    ])
    r = run_cli('--input', str(inp), '--output', str(out),
                '--decontam-refs', str(refs), '--min-chars', '20', '--min-words', '5',
                '--stats-file', str(stats), '--no-progress', '--quiet')
    assert r.returncode == 0, r.stderr
    lines = out.read_text().splitlines()
    assert len(lines) == 1 and 'photosynthesis' in lines[0]
    data = json.loads(stats.read_text())
    assert data['filtered_contaminated'] == 1


def test_decontaminate_unknown_benchmark(tmp_path):
    inp = tmp_path / "in.jsonl"
    write_jsonl(inp, SAMPLE)
    r = run_cli('--input', str(inp), '--output', str(tmp_path / 'o.jsonl'),
                '--decontaminate', 'nosuchbench')
    assert r.returncode == 1
    assert 'Unknown benchmark' in r.stderr
