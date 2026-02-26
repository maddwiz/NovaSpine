from __future__ import annotations

import asyncio

from c3ae.config import Config
from c3ae.llm import create_chat_backend
from c3ae.memory_spine.spine import MemorySpine


def test_graph_and_consolidation_pipeline(tmp_path):
    async def _run() -> None:
        cfg = Config()
        cfg.data_dir = tmp_path
        cfg.ensure_dirs()
        spine = MemorySpine(cfg)
        try:
            await spine.ingest_text(
                "Desmond prefers NovaSpine. NovaSpine uses FAISS for vector retrieval.",
                source_id="test:graph",
            )
            graph = await spine.graph_query("Desmond", depth=2)
            assert graph["nodes"]
            assert graph["edges"]

            # Add repeated content to trigger consolidation clusters.
            for _ in range(3):
                spine.ingest_text_sync(
                    "The user prefers compact reports and short summaries for daily updates.",
                    source_id="test:consolidation",
                )
            result = spine.consolidate(max_chunks=100)
            assert result["chunks_processed"] >= 1
            assert spine.sqlite.count_consolidated_memories() >= 1

            preview = spine.forget_stale(dry_run=True, max_age_days=1, limit=50)
            assert preview["candidate_count"] >= 0
        finally:
            await spine.close()

    asyncio.run(_run())


def test_memory_write_manager_noop_on_duplicate(tmp_path):
    async def _run() -> None:
        cfg = Config()
        cfg.data_dir = tmp_path
        cfg.ensure_dirs()
        spine = MemorySpine(cfg)
        try:
            ev = spine.add_evidence(
                claim="Desmond likes concise summaries.",
                sources=["test"],
                confidence=0.9,
                reasoning="seed evidence",
            )
            first = await spine.add_knowledge(
                title="Preference",
                content="Desmond likes concise summaries.",
                evidence_ids=[ev.id],
            )
            second = await spine.add_knowledge(
                title="Preference",
                content="Desmond likes concise summaries.",
                evidence_ids=[ev.id],
            )
            assert second.id == first.id
        finally:
            await spine.close()

    asyncio.run(_run())


def test_backend_factory_returns_expected_types():
    venice = create_chat_backend("venice", api_key="")
    assert venice.__class__.__name__ == "VeniceBackend"

    openai = create_chat_backend("openai", api_key="test-key")
    assert openai.__class__.__name__ == "OpenAIBackend"

    anthropic = create_chat_backend("anthropic", api_key="test-key")
    assert anthropic.__class__.__name__ == "AnthropicBackend"

    ollama = create_chat_backend("ollama")
    assert ollama.__class__.__name__ == "OllamaBackend"
