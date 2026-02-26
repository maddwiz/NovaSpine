from __future__ import annotations

import asyncio

from c3ae.config import Config
from c3ae.memory_spine.spine import MemorySpine
from c3ae.rlm.reader import RLMReader


def test_rlm_reader_uses_heuristic_when_no_llm_key(tmp_path):
    async def _run() -> None:
        config = Config()
        config.data_dir = tmp_path
        config.venice.api_key = ""
        config.ensure_dirs()

        spine = MemorySpine(config)
        reader = RLMReader(spine, use_llm_extraction=True, max_claims_per_chunk=3)
        try:
            text = (
                "System uptime increased by 12 percent after the patch. "
                "Error rate dropped from 4.1 to 1.3 per minute. "
                "Engineers concluded the cache invalidation bug was fixed."
            )
            result = await reader.read_text(text, topic="ops")
            assert result.evidence_packs
            assert len(result.evidence_packs) <= 3
            assert any("Heuristic extraction" in p.reasoning for p in result.evidence_packs)
        finally:
            await reader.close()
            await spine.close()

    asyncio.run(_run())
