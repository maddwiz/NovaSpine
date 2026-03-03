from __future__ import annotations

from pathlib import Path

from c3ae.storage.sqlite_store import SQLiteStore


def test_sqlite_store_structured_facts_roundtrip(tmp_path: Path):
    db_path = tmp_path / "c3ae.db"
    store = SQLiteStore(db_path)
    try:
        fact_id = store.insert_structured_fact(
            source_chunk_id="chunk-1",
            entity="Melanie",
            relation="painted",
            value="sunset",
            fact_date="2023-07-12",
            confidence=0.9,
            metadata={"session_id": "s1"},
        )
        assert fact_id
        assert store.count_structured_facts() == 1

        listed = store.list_structured_facts(entity="Melanie", relation="painted", limit=10)
        assert len(listed) == 1
        assert listed[0]["value"] == "sunset"

        hits = store.search_structured_facts_fts("Melanie painted sunset", limit=5)
        assert hits
        assert hits[0].source == "structured_fact"
        assert hits[0].metadata.get("entity") == "Melanie"
    finally:
        store.close()

