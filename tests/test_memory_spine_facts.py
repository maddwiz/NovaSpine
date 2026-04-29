from __future__ import annotations

import asyncio

from c3ae.config import Config
from c3ae.ingestion.fact_extractor import StructuredFact
from c3ae.memory_spine.spine import MemorySpine
from c3ae.types import Chunk


def test_memory_spine_indexes_structured_facts_when_enabled(tmp_path):
    async def _run() -> None:
        cfg = Config()
        cfg.data_dir = tmp_path
        cfg.ingestion.enable_fact_extraction = True
        cfg.ingestion.fact_extraction_mode = "heuristic"
        cfg.ingestion.fact_max_per_chunk = 8
        cfg.ingestion.fact_min_confidence = 0.5
        cfg.ensure_dirs()

        spine = MemorySpine(cfg)
        try:
            await spine.ingest_text(
                "Caroline moved from Sweden to Denver in 2019.",
                source_id="test-facts",
                skip_chunking=True,
            )
            assert spine.sqlite.count_structured_facts() >= 1

            facts = spine.query_structured_facts(entity="Caroline", relation="moved_from", limit=10)
            assert facts
            assert any("Sweden" in str(f.get("value", "")) for f in facts)
        finally:
            await spine.close()

    asyncio.run(_run())


def test_structured_facts_keep_bitemporal_metadata_and_supersede(tmp_path):
    cfg = Config()
    cfg.data_dir = tmp_path
    cfg.ensure_dirs()
    spine = MemorySpine(cfg)
    try:
        spine.sqlite.insert_chunk(
            Chunk(
                id="chunk-1",
                content="Desmond prefers coffee.",
                source_id="test",
                metadata={},
            )
        )
        spine._persist_structured_facts(
            chunk_id="chunk-1",
            facts=[
                StructuredFact(
                    entity="Desmond",
                    relation="prefers",
                    value="coffee",
                    date="2025-01-01",
                    confidence=0.9,
                    valid_from="2025-01-01",
                    observed_at="2025-01-02",
                )
            ],
            metadata={"session_id": "s1"},
        )
        first = spine.sqlite.list_structured_facts(entity="Desmond", relation="prefers")[0]
        assert first["valid_from"] == "2025-01-01"
        assert first["observed_at"] == "2025-01-02"

        spine.sqlite.insert_chunk(
            Chunk(
                id="chunk-2",
                content="Desmond now prefers matcha.",
                source_id="test",
                metadata={},
            )
        )
        spine._persist_structured_facts(
            chunk_id="chunk-2",
            facts=[
                StructuredFact(
                    entity="Desmond",
                    relation="prefers",
                    value="matcha",
                    date="2026-01-01",
                    confidence=0.9,
                    valid_from="2026-01-01",
                )
            ],
            metadata={"session_id": "s2"},
        )
        current = spine.sqlite.list_current_structured_facts(entity="Desmond", relation="prefers")
        assert [fact["value"] for fact in current] == ["matcha"]
        old = spine.sqlite.get_structured_fact(first["id"])
        assert old is not None
        assert old["metadata"]["fact_status"] == "historical"
        assert old["metadata"]["superseded_by"] != ""
    finally:
        spine.sqlite.close()
