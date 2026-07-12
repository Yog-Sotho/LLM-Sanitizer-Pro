"""Deduplication backends: Memory, SQLite (Batched), and MinHash (Optional)."""
import atexit
import multiprocessing
import os
import sqlite3
import tempfile
from typing import List, Optional, Set


class MemoryDeduper:
    def __init__(self) -> None:
        self._seen: Set[str] = set()

    def contains(self, h: str) -> bool:
        return h in self._seen

    def add(self, h: str) -> None:
        self._seen.add(h)

    def close(self) -> None:
        self._seen.clear()


class SQLiteDeduper:
    """Disk-backed hash dedup with batched commits.

    Writes are buffered (batch_size rows per commit) for throughput; a shadow
    in-memory set of the unflushed buffer keeps `contains` exact within the
    batch window.
    """

    def __init__(self, db_path: Optional[str] = None, batch_size: int = 5000) -> None:
        self.batch_size = batch_size
        self.buffer: List[tuple] = []
        self._pending: Set[str] = set()
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
        self._conn.execute('CREATE TABLE IF NOT EXISTS hashes (h TEXT PRIMARY KEY)')
        self._conn.commit()

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

    def close(self) -> None:
        self.flush()
        try:
            self._conn.close()
        except Exception:
            pass
        self._unlink_db_files()


try:
    from datasketch import MinHash, MinHashLSH
    DATASKETCH_AVAILABLE = True
except ImportError:
    DATASKETCH_AVAILABLE = False


class MinHashDeduper:
    """Fuzzy deduplication using MinHash + LSH over word 3-shingles."""

    def __init__(self, threshold: float = 0.8, num_perm: int = 128, shingle_size: int = 3) -> None:
        if not DATASKETCH_AVAILABLE:
            raise ImportError("Fuzzy dedup requires: pip install datasketch")
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.num_perm = num_perm
        self.shingle_size = shingle_size
        self._idx = 0

    def _minhash(self, text: str) -> 'MinHash':
        m = MinHash(num_perm=self.num_perm)
        words = text.lower().split()
        if len(words) < self.shingle_size:
            shingles = [' '.join(words)] if words else []
        else:
            shingles = [' '.join(words[i:i + self.shingle_size])
                        for i in range(len(words) - self.shingle_size + 1)]
        for s in shingles:
            m.update(s.encode('utf-8'))
        return m

    def contains(self, text: str) -> bool:
        return len(self.lsh.query(self._minhash(text))) > 0

    def add(self, text: str) -> None:
        self.lsh.insert(f"doc_{self._idx}", self._minhash(text))
        self._idx += 1

    def close(self) -> None:
        pass


class SemanticDeduper:
    """Embedding-based near-duplicate detection: catches paraphrases that share
    no n-grams. Static embeddings (model2vec, no torch) + random-hyperplane LSH
    for candidate lookup, verified with exact cosine similarity — so there are
    no false positives beyond the threshold itself."""

    _NUM_BITS = 64
    _BAND_BITS = 8

    def __init__(self, threshold: float = 0.9, model: str = 'minishlab/potion-base-8M',
                 _embed_fn=None) -> None:
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
        self._planes = None  # lazily sized to the embedding dim
        self._vectors: List = []
        self._buckets: dict = {}
        self._last: Optional[tuple] = None  # (text, vector, signature) cache

    def _embed(self, text: str):
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

    def _bands(self, sig: int):
        for band in range(self._NUM_BITS // self._BAND_BITS):
            yield band, (sig >> (band * self._BAND_BITS)) & ((1 << self._BAND_BITS) - 1)

    def contains(self, text: str) -> bool:
        if not self._vectors:
            self._embed(text)  # warm the cache for the add() that may follow
            return False
        v, sig = self._embed(text)
        candidates = set()
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
                 semantic_model: str = 'minishlab/potion-base-8M'):
    if semantic:
        return SemanticDeduper(threshold=semantic_threshold, model=semantic_model)
    if fuzzy:
        return MinHashDeduper(threshold=fuzzy_threshold)
    return SQLiteDeduper(db_path=db_path) if backend == 'sqlite' else MemoryDeduper()
