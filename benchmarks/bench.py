"""Throughput benchmarks for the sanitize CLI.

Each scenario runs ``python -m sanitizer_pro`` on a synthetic corpus
(benchmarks/corpus.py) in a fresh process, and reports records/s, MB/s and
peak RSS. ``--check`` compares against the targets below and exits non-zero
when one is missed — the nightly CI job runs it.

    python -m benchmarks.bench                       # all scenarios, 100k records
    python -m benchmarks.bench --records 20000 --scenarios regex,jobs
    python -m benchmarks.bench --check --json results.json

Targets are regression guards set from measurements (100k records, one
x86 core, Python 3.13): passthrough ~12.6k rec/s, regex PII + secrets ~7.1k,
with exact dedup ~6.1k; --jobs 4 gave 3.3x (20.6k). They leave ~30% headroom for
slower CI runners. Throughput beyond one core comes from --jobs, up to the
parent process's ceiling (~20k rec/s for regex + dedup).
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, NamedTuple, Optional, Sequence

from benchmarks.corpus import write as write_corpus


class Scenario(NamedTuple):
    name: str
    flags: Sequence[str]
    target_rps: Optional[float] = None   # minimum records/s
    jobs: int = 1
    needs: Sequence[str] = ()            # importable modules required


SCENARIOS: List[Scenario] = [
    Scenario("passthrough", [], target_rps=8_000),
    Scenario("regex", ["--remove-pii", "--redact-secrets"], target_rps=5_000),
    Scenario("regex+dedup", ["--remove-pii", "--redact-secrets", "--deduplicate"],
             target_rps=4_500),
    Scenario("dedup-sqlite", ["--deduplicate", "--dedup-backend", "sqlite"]),
    Scenario("fuzzy", ["--fuzzy-dedup"], target_rps=3_000, needs=("rensa",)),
    Scenario("fuzzy-sqlite", ["--fuzzy-dedup", "--dedup-backend", "sqlite"], needs=("rensa",)),
    Scenario("fuzzy-datasketch", ["--fuzzy-dedup", "--fuzzy-backend", "datasketch"],
             needs=("datasketch",)),
    Scenario("rules", ["--quality-rules", "all"]),
]
JOBS_FLAGS = ["--remove-pii", "--redact-secrets", "--deduplicate"]
# --jobs N must beat one process by this factor. Not proportional to N: the
# parent's stateful stages (dedup, stats, writing) cap parallel throughput
# at ~20k rec/s, so the faster a machine's single core, the smaller the
# ratio (GitHub runner, 50k records: 1.63x at 2 jobs, 1.93x at 4).
SCALING_FLOOR = 1.3


def _available(modules: Sequence[str]) -> bool:
    for m in modules:
        try:
            __import__(m)
        except ImportError:
            return False
    return True


def run(corpus: str, flags: Sequence[str], jobs: int = 1) -> Dict[str, float]:
    out_dir = tempfile.mkdtemp(prefix="bench-")
    cmd = [sys.executable, "-m", "sanitizer_pro", "--input", corpus,
           "--output", os.path.join(out_dir, "out.jsonl"), "--quiet", "--no-progress",
           "--jobs", str(jobs), "--stats-file", os.path.join(out_dir, "stats.json"), *flags]
    err_path = os.path.join(out_dir, "stderr.txt")
    t0 = time.perf_counter()
    with open(err_path, "w") as err:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=err)
        _, status, usage = os.wait4(proc.pid, 0)   # this child's own peak RSS
    elapsed = time.perf_counter() - t0
    proc.returncode = os.waitstatus_to_exitcode(status)
    if proc.returncode:
        with open(err_path) as f:
            raise RuntimeError(f"{' '.join(cmd)} failed:\n{f.read()[-2000:]}")
    with open(os.path.join(out_dir, "stats.json"), encoding="utf-8") as f:
        stats = json.load(f)
    total = stats.get("total", 0)
    return {"seconds": round(elapsed, 2), "records": total,
            "rps": round(total / elapsed), "mb_s": round(os.path.getsize(corpus) / elapsed / 1e6, 1),
            "kept": stats.get("kept", 0),
            "peak_rss_mb": round(usage.ru_maxrss / 1024)}   # KiB on Linux


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m benchmarks.bench")
    ap.add_argument("--records", type=int, default=100_000)
    ap.add_argument("--corpus", default=None, help="Reuse an existing JSONL corpus.")
    ap.add_argument("--scenarios", default="all",
                    help="Comma-separated scenario names, 'jobs', or 'all'.")
    ap.add_argument("--max-jobs", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--check", action="store_true", help="Fail when a target is missed.")
    ap.add_argument("--json", default=None, help="Write results to this file.")
    a = ap.parse_args(argv)

    corpus = a.corpus
    if corpus is None:
        corpus = os.path.join(tempfile.mkdtemp(prefix="bench-corpus-"), "corpus.jsonl")
        write_corpus(corpus, a.records)
    wanted = None if a.scenarios == "all" else set(a.scenarios.split(","))

    results: Dict[str, Dict[str, float]] = {}
    failures: List[str] = []
    print(f"{'scenario':<16}{'rec/s':>10}{'MB/s':>8}{'peak MB':>9}{'seconds':>9}  target")
    for sc in SCENARIOS:
        if wanted is not None and sc.name not in wanted:
            continue
        if not _available(sc.needs):
            print(f"{sc.name:<16}  skipped (needs {', '.join(sc.needs)})")
            continue
        r = run(corpus, sc.flags)
        results[sc.name] = r
        verdict = ""
        if sc.target_rps:
            ok = r["rps"] >= sc.target_rps
            verdict = f">= {sc.target_rps:,.0f} {'ok' if ok else 'MISSED'}"
            if not ok:
                failures.append(f"{sc.name}: {r['rps']:,.0f} rec/s < {sc.target_rps:,.0f}")
        print(f"{sc.name:<16}{r['rps']:>10,.0f}{r['mb_s']:>8}{r['peak_rss_mb']:>9}"
              f"{r['seconds']:>9}  {verdict}")

    if wanted is None or "jobs" in wanted:
        base = results.get("regex+dedup") or run(corpus, JOBS_FLAGS)
        jobs = 2
        while jobs <= a.max_jobs:
            r = run(corpus, JOBS_FLAGS, jobs=jobs)
            results[f"jobs={jobs}"] = r
            speedup = r["rps"] / base["rps"]
            ok = speedup >= SCALING_FLOOR
            if not ok:
                failures.append(f"jobs={jobs}: speedup {speedup:.2f}x < "
                                f"{SCALING_FLOOR:.2f}x")
            print(f"{'jobs=' + str(jobs):<16}{r['rps']:>10,.0f}{r['mb_s']:>8}"
                  f"{r['peak_rss_mb']:>9}{r['seconds']:>9}  speedup {speedup:.2f}x "
                  f"(>= {SCALING_FLOOR:.1f}x {'ok' if ok else 'MISSED'})")
            jobs *= 2

    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"records": a.records, "cpus": os.cpu_count(), "results": results,
                       "failures": failures}, f, indent=2)
    if a.check and failures:
        print("\nTargets missed:\n  " + "\n  ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
