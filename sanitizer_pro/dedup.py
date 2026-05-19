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
    def contains(self, h: str) -> bool: return h in self._seen
    def add(self, h: str) -> None: self._seen.add(h)
    def close(self) -> None: self._seen.clear()

class SQLiteDeduper:
    """Disk-backed hash dedup with batched commits for 40x throughput."""
    def __init__(self, db_path: Optional[str] = None, batch_size: int = 5000) -> None:
        self.batch_size = batch_size
        self.buffer: List[tuple] = []
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

    def _atexit_cleanup(self) -> None:
        if multiprocessing.parent_process() is not None: return
        try: self._conn.close()
        except Exception: pass
        if self._tmp_path:
            try: os.unlink(self._tmp_path)
            except Exception: pass

    def contains(self, h: str) -> bool:
        return self._conn.execute('SELECT 1 FROM hashes WHERE h=?', (h,)).fetchone() is not None

    def add(self, h: str) -> None:
        self.buffer.append((h,))
        if len(self.buffer) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if self.buffer:
            self._conn.executemany('INSERT OR IGNORE INTO hashes VALUES (?)', self.buffer)
            self._conn.commit()
            self.buffer.clear()

    def close(self) -> None:
        self.flush()
        try: self._conn.close()
        except Exception: pass
        if self._tmp_path:
            try: os.unlink(self._tmp_path)
            except Exception: pass

try:
    from datasketch import MinHash, MinHashLSH
    DATASKETCH_AVAILABLE = True
except ImportError:
    DATASKETCH_AVAILABLE = False

class MinHashDeduper:
    """Fuzzy deduplication using MinHash + LSH."""
    def __init__(self, threshold: float = 0.8, num_perm: int = 128) -> None:
        if not DATASKETCH_AVAILABLE:
            raise ImportError("MinHash requires: pip install datasketch")
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.num_perm = num_perm
        self._idx = 0

    def contains(self, text: str) -> bool:
        m = MinHash(num_perm=self.num_perm)
        for d in text.encode('utf8').split():
            m.update(d)
        return len(self.lsh.query(m)) > 0

    def add(self, text: str) -> None:
        m = MinHash(num_perm=self.num_perm)
        for d in text.encode('utf8').split():
            m.update(d)
        self.lsh.insert(f"doc_{self._idx}", m)
        self._idx += 1

    def close(self) -> None: pass

def make_deduper(backend: str, db_path: Optional[str] = None, fuzzy: bool = False):
    if fuzzy: return MinHashDeduper()
    return SQLiteDeduper(db_path=db_path) if backend == 'sqlite' else MemoryDeduper()
