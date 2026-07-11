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


def make_deduper(backend: str, db_path: Optional[str] = None, fuzzy: bool = False,
                 fuzzy_threshold: float = 0.8):
    if fuzzy:
        return MinHashDeduper(threshold=fuzzy_threshold)
    return SQLiteDeduper(db_path=db_path) if backend == 'sqlite' else MemoryDeduper()
