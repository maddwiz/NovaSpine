from __future__ import annotations

from pathlib import Path

from c3ae.storage.sqlite_store import SQLiteStore
from c3ae.types import Chunk
from c3ae.utils import utcnow


class TrackingLock:
    def __init__(self) -> None:
        self.depth = 0

    def __enter__(self):
        self.depth += 1
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.depth -= 1

    @property
    def held(self) -> bool:
        return self.depth > 0


class TrackingConnection:
    def __init__(self, conn, lock: TrackingLock, events: list[tuple[str, bool]]) -> None:
        self._conn = conn
        self._lock = lock
        self._events = events

    def execute(self, sql: str, params=()):
        verb = sql.lstrip().split(None, 1)[0].upper()
        self._events.append((verb, self._lock.held))
        return self._conn.execute(sql, params)

    def commit(self):
        self._events.append(("COMMIT", self._lock.held))
        return self._conn.commit()

    def __getattr__(self, name: str):
        return getattr(self._conn, name)


def test_single_statement_writes_execute_under_lock(tmp_path: Path):
    store = SQLiteStore(tmp_path / "c3ae.db")
    events: list[tuple[str, bool]] = []
    try:
        lock = TrackingLock()
        store._write_lock = lock
        store._conn = TrackingConnection(store._conn, lock, events)

        store.insert_chunk(
            Chunk(
                id="chunk-1",
                source_id="unit:test",
                content="hello",
                metadata={},
                created_at=utcnow(),
            )
        )

        mutating = [held for verb, held in events if verb in {"INSERT", "UPDATE", "DELETE", "COMMIT"}]
        assert mutating
        assert all(mutating)
    finally:
        store.close()


def test_upsert_paths_hold_lock_for_select_and_write(tmp_path: Path):
    store = SQLiteStore(tmp_path / "c3ae.db")
    try:
        store.insert_structured_fact(
            source_chunk_id="chunk-1",
            entity="Melanie",
            relation="painted",
            value="sunset",
            fact_date="2023-07-12",
            confidence=0.9,
            metadata={"session_id": "s1"},
        )

        events: list[tuple[str, bool]] = []
        lock = TrackingLock()
        store._write_lock = lock
        store._conn = TrackingConnection(store._conn, lock, events)

        store.insert_structured_fact(
            source_chunk_id="chunk-1",
            entity="Melanie",
            relation="painted",
            value="sunset",
            fact_date="2023-07-12",
            confidence=0.95,
            metadata={"session_id": "s2"},
        )

        assert events
        assert all(held for _, held in events)
        assert any(verb == "SELECT" for verb, _ in events)
        assert any(verb == "UPDATE" for verb, _ in events)
        assert any(verb == "COMMIT" for verb, _ in events)
    finally:
        store.close()
