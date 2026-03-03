from __future__ import annotations

import asyncio

from c3ae.config import Config
from c3ae.memory_spine.spine import MemorySpine


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

