"""Deduplication backends: exact (memory / SQLite), MinHash LSH (rensa or
datasketch signatures; memory or SQLite band store), and semantic."""
import atexit
import hashlib
import multiprocessing
import os
import sqlite3
import struct
import tempfile
from typing import Any, Callable, Dict, Iterator, List, Optional, Protocol, Set, Tuple

from sanitizer_pro.settings import FUZZY_BACKENDS


class Deduper(Protocol):
    """What the pipeline needs from a dedup backend; keys are hashes or texts."""

    def contains(self, key: str) -> bool: ...

    def add(self, key: str) -> None: ...

    def close(self) -> None: ...


class MemoryDeduper:
    def __init__(self) -> None:
        self._seen: Set[str] = set()

    def contains(self, h: str) -> bool:
        return h in self._seen

    def add(self, h: str) -> None:
        self._seen.add(h)

    def close(self) -> None:
        self._seen.clear()


class _SQLiteFile:
    """A SQLite connection tuned for bulk writes; a temporary file (deleted
    at exit) unless `db_path` names one to keep."""

    def __init__(self, db_path: Optional[str] = None, batch_size: int = 5000) -> None:
        self.batch_size = batch_size
        self._tmp_path: Optional[str] = None
        if db_path:
            self._db_path = db_path
        else:
            fd, self._db_path = tempfile.mkstemp(suffix='.dedup.db')
            os.close(fd)
            self._tmp_path = self._db_path
        self._conn = sqlite3.connect(self._db_path)
        self._conn.execute('PRAGMA journal_mode=WAL')
        self._conn.execute('PRAGMA synchronous=NORMAL')
        if self._tmp_path:
            atexit.register(self._atexit_cleanup)

    def _unlink_db_files(self) -> None:
        if not self._tmp_path:
            return
        for suffix in ('', '-wal', '-shm'):
            try:
                os.unlink(self._tmp_path + suffix)
            except OSError:
                pass
        self._tmp_path = None

    def _atexit_cleanup(self) -> None:
        if multiprocessing.parent_process() is not None:
            return
        try:
            self._conn.close()
        except Exception:
            pass
        self._unlink_db_files()

    def flush(self) -> None:
        self._conn.commit()

    def close(self) -> None:
        self.flush()
        try:
            self._conn.close()
        except Exception:
            pass
        self._unlink_db_files()


class SQLiteDeduper(_SQLiteFile):
    """Disk-backed hash dedup with batched commits.

    Writes are buffered (batch_size rows per commit) for throughput; a shadow
    in-memory set of the unflushed buffer keeps `contains` exact within the
    batch window.
    """

    def __init__(self, db_path: Optional[str] = None, batch_size: int = 5000) -> None:
        super().__init__(db_path, batch_size)
        self.buffer: List[Tuple[str]] = []
        self._pending: Set[str] = set()
        self._conn.execute('CREATE TABLE IF NOT EXISTS hashes (h TEXT PRIMARY KEY)')
        self._conn.commit()

    def contains(self, h: str) -> bool:
        if h in self._pending:
            return True
        return self._conn.execute('SELECT 1 FROM hashes WHERE h=?', (h,)).fetchone() is not None

    def add(self, h: str) -> None:
        if h in self._pending:
            return
        self.buffer.append((h,))
        self._pending.add(h)
        if len(self.buffer) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if self.buffer:
            self._conn.executemany('INSERT OR IGNORE INTO hashes VALUES (?)', self.buffer)
            self._conn.commit()
            self.buffer.clear()
            self._pending.clear()

    def high_water_mark(self) -> int:
        """Commit pending hashes and return the largest rowid. Rows inserted
        later get larger rowids, so the mark identifies exactly the hashes
        known at a checkpoint."""
        self.flush()
        row = self._conn.execute('SELECT COALESCE(MAX(rowid), 0) FROM hashes').fetchone()
        return int(row[0])

    def rollback_to(self, mark: int) -> int:
        """Forget hashes inserted after `mark` (they belong to records whose
        output was discarded on resume). Returns the number removed."""
        self.flush()
        cur = self._conn.execute('DELETE FROM hashes WHERE rowid > ?', (mark,))
        self._conn.commit()
        return cur.rowcount


_SEED = 1


def _optimal_bands(threshold: float, num_perm: int, fp_weight: float = 0.5,
                   fn_weight: float = 0.5) -> Tuple[int, int]:
    """(bands, rows) minimizing the weighted area of false positives (pairs
    below `threshold` that collide in some band) and false negatives (pairs
    above it that collide in none). The same criterion as datasketch, with a
    plain midpoint rule instead of scipy."""

    def area(lo: float, hi: float, f: Callable[[float], float], steps: int = 200) -> float:
        width = (hi - lo) / steps
        return sum(f(lo + (i + 0.5) * width) for i in range(steps)) * width

    best, best_err = (1, num_perm), float('inf')
    for b in range(1, num_perm + 1):
        for r in range(1, num_perm // b + 1):
            fp = area(0.0, threshold, lambda s: 1 - (1 - s ** r) ** b)
            fn = area(threshold, 1.0, lambda s: (1 - s ** r) ** b)
            err = fp_weight * fp + fn_weight * fn
            if err < best_err:
                best, best_err = (b, r), err
    return best


def _shingles(text: str, size: int) -> List[str]:
    words = text.lower().split()
    if len(words) < size:
        return [' '.join(words)] if words else []
    return [' '.join(words[i:i + size]) for i in range(len(words) - size + 1)]


def _signer(backend: str, num_perm: int) -> Tuple[str, Callable[[List[str]], List[int]]]:
    """(backend name, shingles -> MinHash signature)."""
    if backend in ('auto', 'rensa'):
        try:
            from rensa import RMinHash

            def sign_rensa(shingles: List[str]) -> List[int]:
                m = RMinHash(num_perm=num_perm, seed=_SEED)
                m.update(shingles)
                return list(m.digest())

            return 'rensa', sign_rensa
        except ImportError:
            if backend == 'rensa':
                raise ImportError("--fuzzy-backend rensa requires: pip install rensa") from None
    try:
        from datasketch import MinHash
    except ImportError:
        raise ImportError("Fuzzy dedup requires rensa (fast) or datasketch: "
                          "pip install 'llm-sanitizer-pro[fuzzy]'") from None

    def sign_datasketch(shingles: List[str]) -> List[int]:
        m = MinHash(num_perm=num_perm, seed=_SEED)
        m.update_batch([s.encode('utf-8') for s in shingles])
        return [int(v) for v in m.hashvalues]

    return 'datasketch', sign_datasketch


Signature = bytes   # num_perm little-endian uint32 values


def _similarity(a: Signature, b: Signature) -> float:
    """Estimated Jaccard similarity: the fraction of equal MinHash values."""
    n = len(a) // 4
    va, vb = struct.unpack(f'<{n}I', a), struct.unpack(f'<{n}I', b)
    return int(sum(x == y for x, y in zip(va, vb))) / n


class _MemoryIndex:
    """Signatures and band buckets in memory: ~2.4 KB per kept record at the
    default 128 permutations (measured: 1.1 GB peak at 500k records, while
    the SQLite index stays at ~140 MB)."""

    def __init__(self) -> None:
        self._sigs: List[Signature] = []
        self._buckets: Dict[int, Any] = {}   # band key -> doc id, or list of ids

    def candidates(self, keys: List[int]) -> Iterator[Signature]:
        seen: Set[int] = set()
        for k in keys:
            hit = self._buckets.get(k)
            for doc in (hit if isinstance(hit, list) else () if hit is None else (hit,)):
                if doc not in seen:
                    seen.add(doc)
                    yield self._sigs[doc]

    def add(self, keys: List[int], sig: Signature) -> None:
        doc = len(self._sigs)
        self._sigs.append(sig)
        for k in keys:
            hit = self._buckets.get(k)
            if hit is None:
                self._buckets[k] = doc            # most buckets hold one record
            elif isinstance(hit, list):
                hit.append(doc)
            else:
                self._buckets[k] = [hit, doc]

    def close(self) -> None:
        self._sigs.clear()
        self._buckets.clear()


class _SQLiteIndex(_SQLiteFile):
    """Signatures and band buckets on disk: memory stays constant however
    large the input. Record ids grow with insertion, so checkpoints roll
    back exactly like the exact-dedup store (high_water_mark/rollback_to)."""

    def __init__(self, db_path: Optional[str], meta: str, batch_size: int = 5000) -> None:
        super().__init__(db_path, batch_size)
        self._pending_sigs: Dict[int, Signature] = {}
        self._pending_keys: Dict[int, List[int]] = {}
        c = self._conn
        c.execute('CREATE TABLE IF NOT EXISTS minhash (id INTEGER PRIMARY KEY, sig BLOB)')
        c.execute('CREATE TABLE IF NOT EXISTS lsh (k INTEGER, id INTEGER)')
        c.execute('CREATE INDEX IF NOT EXISTS lsh_k ON lsh (k)')
        c.execute('CREATE TABLE IF NOT EXISTS minhash_meta (v TEXT)')
        row = c.execute('SELECT v FROM minhash_meta').fetchone()
        if row is None:
            c.execute('INSERT INTO minhash_meta VALUES (?)', (meta,))
        elif row[0] != meta:
            self.close()
            raise ValueError(f"Fuzzy dedup DB {db_path} was built with different settings "
                             f"({row[0]}; now {meta}). Use a new --dedup-db-path.")
        c.commit()
        last = c.execute('SELECT COALESCE(MAX(id), 0) FROM minhash').fetchone()[0]
        self._next_id = int(last) + 1

    def candidates(self, keys: List[int]) -> Iterator[Signature]:
        seen: Set[int] = set()
        for k in keys:
            for doc in self._pending_keys.get(k, ()):
                if doc not in seen:
                    seen.add(doc)
                    yield self._pending_sigs[doc]
        marks = ','.join('?' * len(keys))
        rows = self._conn.execute(
            f'SELECT DISTINCT m.id, m.sig FROM lsh JOIN minhash m ON m.id = lsh.id '
            f'WHERE lsh.k IN ({marks})', keys).fetchall()
        for doc, sig in rows:
            if doc not in seen:
                seen.add(doc)
                yield bytes(sig)

    def add(self, keys: List[int], sig: Signature) -> None:
        doc = self._next_id
        self._next_id += 1
        self._pending_sigs[doc] = sig
        for k in keys:
            self._pending_keys.setdefault(k, []).append(doc)
        if len(self._pending_sigs) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if self._pending_sigs:
            self._conn.executemany('INSERT INTO minhash VALUES (?, ?)',
                                   list(self._pending_sigs.items()))
            self._conn.executemany('INSERT INTO lsh VALUES (?, ?)',
                                   [(k, d) for k, docs in self._pending_keys.items()
                                    for d in docs])
            self._pending_sigs.clear()
            self._pending_keys.clear()
        self._conn.commit()

    def high_water_mark(self) -> int:
        self.flush()
        return self._next_id - 1

    def rollback_to(self, mark: int) -> int:
        self.flush()
        self._conn.execute('DELETE FROM lsh WHERE id > ?', (mark,))
        cur = self._conn.execute('DELETE FROM minhash WHERE id > ?', (mark,))
        self._conn.commit()
        self._next_id = mark + 1
        return cur.rowcount


class MinHashDeduper:
    """Near-duplicate detection over word 3-shingles.

    LSH finds candidates: band parameters favor recall (few misses above
    `threshold`). Each candidate is then verified by comparing MinHash
    signatures: the record is a duplicate when the estimated Jaccard
    similarity to an earlier record is >= threshold - `tolerance` (the
    tolerance absorbs the estimator's noise, ~0.03 at 128 permutations).
    Measured on synthetic pairs (benchmarks/fuzzy_recall.py) at t = 0.7,
    0.8 and 0.85: 97-98% of pairs at or above the threshold are caught, none
    0.15 or more below it, about half of those within 0.05 below it.

    Signatures come from rensa (Rust, ~10x faster) when installed, else
    datasketch. The index lives in memory, or with store='sqlite' on disk
    with constant memory."""

    RECALL_WEIGHT = 0.9
    TOLERANCE = 0.03

    def __init__(self, threshold: float = 0.8, num_perm: int = 128, shingle_size: int = 3,
                 backend: str = 'auto', store: str = 'memory',
                 db_path: Optional[str] = None, tolerance: Optional[float] = None) -> None:
        if backend not in FUZZY_BACKENDS:
            raise ValueError(f"Unknown fuzzy backend '{backend}' {FUZZY_BACKENDS}.")
        self.backend, self._sign = _signer(backend, num_perm)
        self.threshold = threshold
        self.tolerance = self.TOLERANCE if tolerance is None else tolerance
        self.num_perm = num_perm
        self.shingle_size = shingle_size
        self.bands, self.rows = _optimal_bands(threshold, num_perm,
                                               1 - self.RECALL_WEIGHT, self.RECALL_WEIGHT)
        meta = (f"{self.backend}:perm={num_perm}:seed={_SEED}:shingle={shingle_size}"
                f":bands={self.bands}x{self.rows}")
        self.index: Any = (_SQLiteIndex(db_path, meta) if store == 'sqlite'
                           else _MemoryIndex())
        self._last: Optional[Tuple[str, List[int], Signature]] = None  # contains() -> add()

    def _sketch(self, text: str) -> Tuple[List[int], Signature]:
        if self._last is not None and self._last[0] is text:
            return self._last[1], self._last[2]
        values = self._sign(_shingles(text, self.shingle_size))
        sig = struct.pack(f'<{self.num_perm}I', *values)
        r = self.rows
        keys = [int.from_bytes(hashlib.blake2b(sig[b * r * 4:(b + 1) * r * 4],
                                               digest_size=8, salt=b.to_bytes(2, 'little')
                                               ).digest(), 'little', signed=True)
                for b in range(self.bands)]
        self._last = (text, keys, sig)
        return keys, sig

    def similarity_to_index(self, text: str) -> float:
        """Highest estimated similarity to an indexed candidate (0 if none)."""
        keys, sig = self._sketch(text)
        return max((_similarity(sig, other) for other in self.index.candidates(keys)),
                   default=0.0)

    def contains(self, text: str) -> bool:
        keys, sig = self._sketch(text)
        bar = self.threshold - self.tolerance
        return any(_similarity(sig, other) >= bar for other in self.index.candidates(keys))

    def add(self, text: str) -> None:
        keys, sig = self._sketch(text)
        self.index.add(keys, sig)

    def flush(self) -> None:
        if hasattr(self.index, 'flush'):
            self.index.flush()

    def high_water_mark(self) -> int:
        return int(self.index.high_water_mark()) if store_is_durable(self.index) else 0

    def rollback_to(self, mark: int) -> int:
        return int(self.index.rollback_to(mark)) if store_is_durable(self.index) else 0

    def close(self) -> None:
        self.index.close()


def store_is_durable(store: Any) -> bool:
    return isinstance(store, _SQLiteIndex)


def is_durable(deduper: Any) -> bool:
    """True for dedup state that lives in a SQLite file (exact or fuzzy),
    which checkpoints can mark and roll back."""
    return isinstance(deduper, SQLiteDeduper) or (
        isinstance(deduper, MinHashDeduper) and store_is_durable(deduper.index))


class SemanticDeduper:
    """Embedding-based near-duplicate detection: catches paraphrases that share
    no n-grams. Static embeddings (model2vec, no torch) + random-hyperplane LSH
    for candidate lookup, verified with exact cosine similarity — so there are
    no false positives beyond the threshold itself."""

    _NUM_BITS = 64
    _BAND_BITS = 8

    def __init__(self, threshold: float = 0.9, model: str = 'minishlab/potion-base-8M',
                 _embed_fn: Optional[Callable[[str], Any]] = None) -> None:
        try:
            import numpy as np
        except ImportError:
            raise ImportError("Semantic dedup requires: pip install model2vec") from None
        self._np = np
        self.threshold = threshold
        if _embed_fn is not None:
            self._embed_raw = _embed_fn
        else:
            try:
                from model2vec import StaticModel
            except ImportError:
                raise ImportError("Semantic dedup requires: pip install model2vec") from None
            m = StaticModel.from_pretrained(model)
            self._embed_raw = lambda text: m.encode([text])[0]
        self._planes: Any = None  # lazily sized to the embedding dim
        self._vectors: List[Any] = []
        self._buckets: Dict[Tuple[int, int], List[int]] = {}
        self._last: Optional[Tuple[str, Any, int]] = None  # (text, vector, signature) cache

    def _embed(self, text: str) -> Tuple[Any, int]:
        if self._last is not None and self._last[0] == text:
            return self._last[1], self._last[2]
        np = self._np
        v = np.asarray(self._embed_raw(text), dtype=np.float32)
        norm = float(np.linalg.norm(v))
        if norm > 0:
            v = v / norm
        if self._planes is None:
            self._planes = np.random.RandomState(0).randn(v.shape[0], self._NUM_BITS)
        bits = (v @ self._planes) > 0
        sig = int(np.packbits(bits).tobytes().hex(), 16)
        self._last = (text, v, sig)
        return v, sig

    def _bands(self, sig: int) -> Iterator[Tuple[int, int]]:
        for band in range(self._NUM_BITS // self._BAND_BITS):
            yield band, (sig >> (band * self._BAND_BITS)) & ((1 << self._BAND_BITS) - 1)

    def contains(self, text: str) -> bool:
        if not self._vectors:
            self._embed(text)  # warm the cache for the add() that may follow
            return False
        v, sig = self._embed(text)
        candidates: Set[int] = set()
        for key in self._bands(sig):
            candidates.update(self._buckets.get(key, ()))
        for idx in candidates:
            if float(v @ self._vectors[idx]) >= self.threshold:
                return True
        return False

    def add(self, text: str) -> None:
        v, sig = self._embed(text)
        idx = len(self._vectors)
        self._vectors.append(v)
        for key in self._bands(sig):
            self._buckets.setdefault(key, []).append(idx)

    def close(self) -> None:
        self._vectors.clear()
        self._buckets.clear()


def make_deduper(backend: str, db_path: Optional[str] = None, fuzzy: bool = False,
                 fuzzy_threshold: float = 0.8, semantic: bool = False,
                 semantic_threshold: float = 0.9,
                 semantic_model: str = 'minishlab/potion-base-8M',
                 fuzzy_backend: str = 'auto') -> Deduper:
    if semantic:
        return SemanticDeduper(threshold=semantic_threshold, model=semantic_model)
    if fuzzy:
        return MinHashDeduper(threshold=fuzzy_threshold, backend=fuzzy_backend,
                              store=backend, db_path=db_path)
    return SQLiteDeduper(db_path=db_path) if backend == 'sqlite' else MemoryDeduper()
