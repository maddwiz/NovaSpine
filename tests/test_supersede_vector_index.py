from __future__ import annotations

import asyncio

from c3ae.config import Config
from c3ae.memory_spine.spine import MemorySpine
from c3ae.reasoning_bank.manager import WriteDecision


def _config(tmp_path):
    cfg = Config()
    cfg.data_dir = tmp_path
    cfg.venice.embedding_provider = "hash"
    cfg.venice.embedding_model = "local-hash-v1"
    cfg.venice.embedding_dims = 64
    cfg.governance.require_evidence = False
    cfg.governance.contradiction_check = False
    cfg.memory_manager.enabled = False
    return cfg


def _reasoning_chunk_id(spine: MemorySpine, entry_id: str) -> str:
    for link in spine.sqlite.list_reasoning_chunk_links():
        if link["reasoning_entry_id"] == entry_id:
            return link["chunk_id"]
    raise AssertionError(f"missing reasoning chunk for {entry_id}")


def test_sync_supersede_indexes_replacement_vector(tmp_path):
    spine = MemorySpine(_config(tmp_path))
    try:
        first = asyncio.run(
            spine.add_knowledge(
                "Favorite drink",
                "The user prefers coffee.",
                bypass_governance=True,
            )
        )

        replacement = spine.supersede_knowledge(
            first.id,
            "Favorite drink",
            "The user now prefers matcha.",
        )
        old_chunk_id = _reasoning_chunk_id(spine, first.id)
        chunk_id = _reasoning_chunk_id(spine, replacement.id)

        assert spine.sqlite.get_chunk(old_chunk_id).metadata["entry_status"] == "superseded"  # type: ignore[union-attr]
        assert spine.faiss.get_vector_by_external_id(chunk_id) is not None
        rows = asyncio.run(spine.recall("what beverage does the user currently prefer matcha?", top_k=5))
        assert any("matcha" in row["content"].lower() for row in rows)
    finally:
        asyncio.run(spine.close())


def test_add_knowledge_update_path_uses_async_supersede(tmp_path):
    async def _run() -> None:
        spine = MemorySpine(_config(tmp_path))
        try:
            first = await spine.add_knowledge(
                "Favorite drink",
                "The user prefers coffee.",
                bypass_governance=True,
            )

            async def decide_update(_entry):
                return WriteDecision(action="UPDATE", target_id=first.id, reason="test_update")

            spine.write_manager.decide_async = decide_update  # type: ignore[method-assign]
            replacement = await spine.add_knowledge(
                "Favorite drink",
                "The user now prefers matcha.",
                bypass_governance=True,
            )
            chunk_id = _reasoning_chunk_id(spine, replacement.id)

            assert replacement.id != first.id
            assert spine.bank.get(first.id).status.value == "superseded"  # type: ignore[union-attr]
            assert spine.faiss.get_vector_by_external_id(chunk_id) is not None
        finally:
            await spine.close()

    asyncio.run(_run())
