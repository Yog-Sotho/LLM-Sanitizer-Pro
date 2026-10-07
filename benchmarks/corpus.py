"""Deterministic synthetic corpus for throughput benchmarks.

Records look like instruction-tuning data: an ``instruction`` and an
``output`` of a few sentences (~600 characters in total by default), with
realistic rates of the things the pipeline acts on:

  * ~8% contain PII (emails, phones, cards, IPs), ~2% a credential
  * ~10% are exact duplicates and ~5% near-duplicates of an earlier record
  * ~3% are junk (too short, or symbol soup)

    python -m benchmarks.corpus --records 100000 --out /tmp/bench.jsonl
"""
import argparse
import json
import random
from typing import Any, Dict, Iterator, List

_WORDS = (
    "the of and to in is that for it as with was on be by this are from or have an they "
    "which one you were all we can her has there been if more when will would who so no "
    "model data training dataset language system result method value process function "
    "example table section figure analysis approach performance memory network output "
    "input sample record source quality filter token sentence document paragraph number "
    "first second large small different important several common general specific simple"
).split()
_TOPICS = ["Explain", "Summarize", "Describe", "Compare", "List the steps to", "Outline"]


def _sentence(rng: random.Random, lo: int = 8, hi: int = 18) -> str:
    words = [rng.choice(_WORDS) for _ in range(rng.randint(lo, hi))]
    return words[0].capitalize() + " " + " ".join(words[1:]) + "."


def _pii(rng: random.Random) -> str:
    kind = rng.randrange(4)
    if kind == 0:
        return f"Contact maria.jensen{rng.randint(1, 999)}@example.com for details."
    if kind == 1:
        return f"Call (415) 555-01{rng.randint(10, 99)} after noon."
    if kind == 2:
        return "The card 4111 1111 1111 1111 was declined."
    return f"Requests came from 192.0.2.{rng.randint(1, 254)} overnight."


def _secret(rng: random.Random) -> str:
    # Built at runtime: no token-shaped literal lives in the source.
    body = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(36))
    return f"export GITHUB_TOKEN=ghp_{body}"


def generate(n: int, seed: int = 1234, sentences: int = 6) -> Iterator[Dict[str, Any]]:
    rng = random.Random(seed)
    history: List[Dict[str, Any]] = []
    for i in range(n):
        roll = rng.random()
        if history and roll < 0.10:
            rec = dict(rng.choice(history))
        elif history and roll < 0.15:
            base = rng.choice(history)
            words = base["output"].split()
            j = rng.randrange(len(words))
            words[j] = rng.choice(_WORDS)
            rec = {"instruction": base["instruction"], "output": " ".join(words)}
        elif roll < 0.18:
            rec = {"instruction": "?", "output": rng.choice(["ok", "#### $$$ ### !!!", "..."])}
        else:
            body = [_sentence(rng) for _ in range(sentences)]
            if rng.random() < 0.08:
                body.insert(rng.randrange(len(body)), _pii(rng))
            if rng.random() < 0.02:
                body.append(_secret(rng))
            rec = {"instruction": f"{rng.choice(_TOPICS)} {rng.choice(_WORDS)} "
                                  f"{rng.choice(_WORDS)} in record {i}.",
                   "output": " ".join(body)}
            if len(history) < 5000:
                history.append(rec)
            else:
                history[rng.randrange(5000)] = rec
        yield rec


def write(path: str, n: int, seed: int = 1234) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in generate(n, seed):
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    write(a.out, a.records, a.seed)
