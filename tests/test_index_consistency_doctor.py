from __future__ import annotations

import asyncio

from c3ae.config import Config
from c3ae.memory_spine.spine import MemorySpine


def test_index_consistency_flags_missing_vector(tmp_path):
    async def _run() -> None:
        cfg = Config()
        cfg.data_dir = tmp_path
        cfg.venice.embedding_provider = "hash"
        cfg.venice.embedding_model = "local-hash-v1"
        cfg.venice.embedding_dims = 64
        cfg.governance.require_evidence = False
        cfg.governance.contradiction_check = False
        cfg.memory_manager.enabled = False

        spine = MemorySpine(cfg)
        try:
            entry = await spine.add_knowledge(
                "Preference",
                "The user prefers concise summaries.",
                bypass_governance=True,
            )
            chunk_id = next(
                link["chunk_id"]
                for link in spine.sqlite.list_reasoning_chunk_links()
                if link["reasoning_entry_id"] == entry.id
            )

            assert spine.faiss.remove(chunk_id)
            report = spine.check_index_consistency()

            assert chunk_id in report["chunks_missing_vectors"]
            assert entry.id in report["active_reasoning_entries_missing_vectors"]
        finally:
            await spine.close()

    asyncio.run(_run())
