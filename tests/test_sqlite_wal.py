from __future__ import annotations

from c3ae.storage.sqlite_store import SQLiteStore


def test_sqlite_store_enables_wal(tmp_path) -> None:
    store = SQLiteStore(tmp_path / "c3ae.db")
    mode = store._conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert str(mode).lower() == "wal"
    store.close()
