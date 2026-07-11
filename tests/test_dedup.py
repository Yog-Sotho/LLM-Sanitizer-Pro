"""Tests for deduplication backends."""
import os

from sanitizer_pro.dedup import MemoryDeduper, SQLiteDeduper


class TestMemoryDeduper:
    def test_roundtrip(self):
        d = MemoryDeduper()
        assert not d.contains("h1")
        d.add("h1")
        assert d.contains("h1")
        d.close()


class TestSQLiteDeduper:
    def test_roundtrip(self, tmp_path):
        db = str(tmp_path / "d.db")
        d = SQLiteDeduper(db_path=db)
        d.add("h1")
        assert d.contains("h1")
        d.close()

    def test_contains_sees_unflushed_buffer(self, tmp_path):
        """Regression: duplicates within the batch window must be detected."""
        d = SQLiteDeduper(db_path=str(tmp_path / "d.db"), batch_size=5000)
        d.add("h1")  # stays in buffer, not yet committed
        assert d.contains("h1"), "unflushed hash must still be visible to contains()"
        d.close()

    def test_flush_at_batch_size(self, tmp_path):
        d = SQLiteDeduper(db_path=str(tmp_path / "d.db"), batch_size=3)
        for i in range(3):
            d.add(f"h{i}")
        assert not d.buffer, "buffer should auto-flush at batch_size"
        assert d.contains("h0")
        d.close()

    def test_persistence_across_instances(self, tmp_path):
        db = str(tmp_path / "d.db")
        d1 = SQLiteDeduper(db_path=db)
        d1.add("h1")
        d1.close()
        d2 = SQLiteDeduper(db_path=db)
        assert d2.contains("h1")
        d2.close()
        assert os.path.exists(db), "user-supplied db must not be deleted on close"

    def test_tmp_db_cleaned_up(self):
        d = SQLiteDeduper()
        path = d._db_path
        d.add("h1")
        d.close()
        assert not os.path.exists(path)
