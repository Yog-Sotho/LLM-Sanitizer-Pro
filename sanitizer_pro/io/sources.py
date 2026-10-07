"""Input sources: one file, a directory, or a glob pattern; and byte-range
chunks of JSONL files, so worker processes can read input in parallel.

Multiple files are read in sorted path order, which makes runs (and
--resume) deterministic. Directories are searched recursively for files
with a supported extension; hidden files and checkpoint files are skipped.
"""
import glob
import io
import json
import os
from pathlib import Path
from typing import Any, Iterator, List, NamedTuple, Optional, Sequence, Union

from sanitizer_pro.io.readers import SUPPORTED_INPUT_FORMATS, MalformedRecord, read_records
from sanitizer_pro.utils import _STDIN, InputFormatError, get_file_format

_GLOB_CHARS = frozenset('*?[')


def is_multi_input(spec: str) -> bool:
    """True when `spec` names a directory or a glob pattern."""
    if spec == _STDIN or spec.startswith('hf://'):
        return False
    return os.path.isdir(spec) or (any(c in spec for c in _GLOB_CHARS)
                                   and not os.path.exists(spec))


def expand_inputs(spec: str, exclude: Sequence[str] = ()) -> List[str]:
    """The files `spec` names, in sorted order. `exclude` drops paths (e.g.
    the output file when it lives inside an input directory)."""
    if not is_multi_input(spec):
        return [spec]
    skip = {os.path.abspath(p) for p in exclude}
    if os.path.isdir(spec):
        candidates = [str(p) for p in Path(spec).rglob('*')
                      if p.is_file() and not any(part.startswith('.') for part in
                                                 p.relative_to(spec).parts)
                      and get_file_format(str(p)) in SUPPORTED_INPUT_FORMATS]
    else:
        candidates = [p for p in glob.glob(spec, recursive=True) if os.path.isfile(p)]
    files = sorted(p for p in candidates
                   if os.path.abspath(p) not in skip and not p.endswith('.checkpoint.json'))
    if not files:
        raise InputFormatError(f"No input files found for '{spec}'.")
    return files


def iter_files(files: Sequence[str], input_format: Optional[str] = None,
               **reader_kwargs: Any) -> Iterator[Any]:
    """Records of each file in turn (format from each file's extension
    unless `input_format` is given)."""
    for path in files:
        fmt = input_format or get_file_format(path)
        yield from read_records(path, input_format=fmt, **reader_kwargs)


# -- parallel JSONL reading ------------------------------------------------------

class Chunk(NamedTuple):
    """Lines of `path` that start in the byte range [start, end)."""
    path: str
    start: int
    end: int


def chunkable(files: Sequence[str], input_format: Optional[str] = None) -> bool:
    """Plain (uncompressed) JSONL files on disk can be split into byte ranges."""
    return bool(files) and all(
        f != _STDIN and not f.startswith('hf://') and not f.lower().endswith('.gz')
        and (input_format or get_file_format(f)) == '.jsonl' for f in files)


def jsonl_chunks(files: Sequence[str], chunk_bytes: int = 4 << 20) -> Iterator[Chunk]:
    for path in files:
        size = os.path.getsize(path)
        for start in range(0, max(size, 1), chunk_bytes):
            yield Chunk(path, start, min(start + chunk_bytes, size))


def read_chunk(chunk: Chunk, encoding: str = 'utf-8') -> List[Union[Any, MalformedRecord]]:
    """Parse the records of one chunk exactly as read_records() parses the
    whole file: blank lines skipped, unparseable lines as MalformedRecord,
    universal newlines (\\n, \\r\\n and a lone \\r all end a line)."""
    with open(chunk.path, 'rb') as f:
        if chunk.start > 0:
            f.seek(chunk.start - 1)
            if f.read(1) != b'\n':
                f.readline()            # the partial line belongs to the previous chunk
        data = bytearray()
        while f.tell() < chunk.end or (chunk.start == 0 and chunk.end == 0):
            raw = f.readline()
            if not raw:
                break
            data += raw
    out: List[Union[Any, MalformedRecord]] = []
    # Chunks end after a '\n', so splitting here matches text-mode reading.
    for line in io.StringIO(data.decode(encoding), newline=None):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            out.append(MalformedRecord(f"{chunk.path} (bytes {chunk.start}-{chunk.end})",
                                       str(exc)))
    return out
