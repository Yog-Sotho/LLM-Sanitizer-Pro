"""Print the CHANGELOG.md section for a version (release notes).

    python scripts/release_notes.py 4.0.0 > notes.md
"""
import re
import sys
from pathlib import Path

CHANGELOG = Path(__file__).resolve().parent.parent / "CHANGELOG.md"


def section(version: str, text: str) -> str:
    heading = re.compile(rf"^## {re.escape(version)}\b.*$", re.MULTILINE)
    m = heading.search(text)
    if not m:
        raise SystemExit(f"CHANGELOG.md has no '## {version}' section")
    nxt = re.compile(r"^## ", re.MULTILINE).search(text, m.end())
    return text[m.end():nxt.start() if nxt else len(text)].strip() + "\n"


if __name__ == "__main__":
    print(section(sys.argv[1].lstrip("v"), CHANGELOG.read_text(encoding="utf-8")), end="")
