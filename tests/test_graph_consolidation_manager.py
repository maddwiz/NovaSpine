from __future__ import annotations

import asyncio

import numpy as np

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
            assert result["clusters_created"] >= 1
            assert spine.sqlite.count_consolidated_memories() >= 1

            preview = spine.forget_stale(dry_run=True, max_age_days=1, limit=50)
            assert preview["candidate_count"] >= 0
        finally:
            await spine.close()

    asyncio.run(_run())


def test_graph_llm_mode_falls_back_to_heuristic(tmp_path):
    class _BadChat:
        async def chat(self, *args, **kwargs):  # noqa: ANN002, ANN003
            class _R:
                content = "{broken json"

            return _R()

        async def close(self) -> None:
            return None

    async def _run() -> None:
        cfg = Config()
        cfg.data_dir = tmp_path
        cfg.graph.extraction_mode = "llm"
        cfg.ensure_dirs()
        spine = MemorySpine(cfg)
        spine._graph_chat = _BadChat()
        try:
            await spine.ingest_text(
                "Desmond prefers NovaSpine for memory retrieval.",
                source_id="test:graph-llm-fallback",
            )
            graph = await spine.graph_query("Desmond", depth=1)
            assert graph["nodes"]
            assert graph["edges"]
        finally:
            await spine.close()

    asyncio.run(_run())


def test_semantic_vector_clustering_groups_related_chunks(tmp_path):
    async def _run() -> None:
        cfg = Config()
        cfg.data_dir = tmp_path
        cfg.consolidation.vector_similarity_threshold = 0.90
        cfg.consolidation.entity_overlap_threshold = 0.99
        cfg.consolidation.lexical_overlap_threshold = 0.99
        cfg.ensure_dirs()
        spine = MemorySpine(cfg)
        try:
            c1 = spine.ingest_text_sync(
                "Latency dropped by 37 percent after cache warmup optimization.",
                source_id="test:cluster",
            )[0]
            c2 = spine.ingest_text_sync(
                "Cache warmup optimization reduced response latency by roughly one third.",
                source_id="test:cluster",
            )[0]
            spine.ingest_text_sync(
                "The weather forecast predicts rain this weekend.",
                source_id="test:cluster",
            )

            # Seed semantic vectors directly so clustering can use FAISS similarity.
            v = np.zeros((cfg.venice.embedding_dims,), dtype=np.float32)
            v[0] = 0.9
            v[1] = 0.1
            v[2] = 0.1
            spine.faiss.add(v, c1)
            spine.faiss.add(v, c2)

            result = spine.consolidate(max_chunks=100)
            assert result["clusters_created"] >= 1
            rows = spine.sqlite.list_consolidated_memories(limit=10)
            assert any({c1, c2}.issubset(set(r["source_chunk_ids"])) for r in rows)
        finally:
            await spine.close()

    asyncio.run(_run())


def test_consolidate_async_uses_llm_enrichment_when_available(tmp_path):
    class _LLMStub:
        async def chat(self, messages, **kwargs):  # noqa: ANN001, ANN003
            user_msg = messages[-1].content.lower()
            class _R:
                content = ""

            r = _R()
            if "key stable facts" in user_msg:
                r.content = '{"facts":["Latency improved after cache warmup","Error rate decreased after retry tuning"]}'
            else:
                r.content = '{"summary":"Performance tuning pattern consolidated across sessions."}'
            return r

        async def close(self) -> None:
            return None

    async def _run() -> None:
        cfg = Config()
        cfg.data_dir = tmp_path
        cfg.consolidation.use_llm_enrichment = True
        cfg.ensure_dirs()
        spine = MemorySpine(cfg)
        spine._consolidation_chat = _LLMStub()
        try:
            for _ in range(3):
                spine.ingest_text_sync(
                    "Cache warmup improved latency and reduced errors in production.",
                    source_id="test:llm-consolidation",
                )
            result = await spine.consolidate_async(max_chunks=100)
            assert result["clusters_created"] >= 1
            assert result["llm_enriched_clusters"] >= 1
            rows = spine.sqlite.list_consolidated_memories(limit=5)
            assert any("Performance tuning pattern consolidated" in r["summary"] for r in rows)
        finally:
            await spine.close()

    asyncio.run(_run())


def test_dream_consolidate_reports_contradictions_and_skills(tmp_path):
    async def _run() -> None:
        cfg = Config()
        cfg.data_dir = tmp_path
        cfg.ensure_dirs()
        spine = MemorySpine(cfg)
        try:
            spine.ingest_text_sync("Desmond prefers tea.", source_id="test:dream")
            spine.ingest_text_sync("Desmond prefers coffee.", source_id="test:dream")
            for _ in range(4):
                spine.ingest_text_sync(
                    "Always run regression checks before deployment to production.",
                    source_id="test:dream",
                )

            report = spine.dream_consolidate()
            assert "contradictions" in report
            assert "skill_candidates" in report
            assert "recompression_preview" in report
            assert report["consolidation"]["clusters_created"] >= 1
            assert len(report["contradictions"]) >= 1
            assert len(report["skill_candidates"]) >= 1
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
