from __future__ import annotations

from c3ae.config import MemoryManagerConfig
from c3ae.memory_manager import MemoryWriteAdmissionManager
from c3ae.storage.sqlite_store import SQLiteStore
from c3ae.types import Chunk


def test_write_admission_noops_duplicate_current_fact(tmp_path):
    store = SQLiteStore(tmp_path / "test.db")
    try:
        store.insert_chunk(Chunk(id="chunk-a", content="Desmond prefers coffee.", source_id="test"))
        fact_id = store.insert_structured_fact(
            source_chunk_id="chunk-a",
            entity="Desmond",
            relation="prefers",
            value="coffee",
            confidence=0.9,
        )

        manager = MemoryWriteAdmissionManager(MemoryManagerConfig())
        decision = manager.decide_structured_fact(
            store,
            source_chunk_id="chunk-a",
            entity="Desmond",
            relation="prefers",
            value="coffee",
            confidence=0.9,
        )

        assert decision.action == "NOOP"
        assert decision.target_id == fact_id
    finally:
        store.close()


def test_write_admission_supersedes_single_valued_fact(tmp_path):
    store = SQLiteStore(tmp_path / "test.db")
    try:
        store.insert_chunk(Chunk(id="chunk-a", content="Desmond prefers coffee.", source_id="test"))
        fact_id = store.insert_structured_fact(
            source_chunk_id="chunk-a",
            entity="Desmond",
            relation="prefers",
            value="coffee",
            confidence=0.9,
        )

        manager = MemoryWriteAdmissionManager(MemoryManagerConfig())
        decision = manager.decide_structured_fact(
            store,
            source_chunk_id="chunk-b",
            entity="Desmond",
            relation="prefers",
            value="matcha",
            confidence=0.9,
        )

        assert decision.action == "SUPERSEDE"
        assert decision.metadata["supersedes_fact_ids"] == [fact_id]
    finally:
        store.close()


def test_graph_edges_keep_bitemporal_metadata(tmp_path):
    store = SQLiteStore(tmp_path / "test.db")
    try:
        src = store.upsert_entity("Desmond")
        dst = store.upsert_entity("Denver")
        store.add_edge(
            src,
            relation="location",
            dst_entity_id=dst,
            source_chunk_id="chunk-a",
            confidence=0.8,
            valid_from="2025-05-01",
            provenance={"source": "unit-test"},
        )

        graph = store.query_graph("Desmond")
        edge = graph["edges"][0]
        assert edge["valid_from"] == "2025-05-01"
        assert edge["metadata"]["valid_from"] == "2025-05-01"
        assert edge["metadata"]["provenance"] == {"source": "unit-test"}
    finally:
        store.close()
