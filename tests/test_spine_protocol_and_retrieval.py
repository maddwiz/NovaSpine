from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from c3ae.config import Config, RetrievalConfig
from c3ae.embeddings.backends import HashEmbedder, create_embedder
from c3ae.memory_spine.spine import MemorySpine
from c3ae.retrieval.hybrid import HybridSearch
from c3ae.types import SearchResult


class _KeywordStub:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.last_query = ""

    def search_all(self, query: str, limit: int = 20) -> list[SearchResult]:
        self.last_query = query
        return list(self.results[:limit])


class _VectorStub:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results

    def search(self, query_vector, top_k: int = 20) -> list[SearchResult]:
        return list(self.results[:top_k])


def _ts(hours_old: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours_old)).isoformat()


def _mk_result(
    rid: str,
    score: float,
    hours_old: float,
    source_kind: str = "chunk",
) -> SearchResult:
    return SearchResult(
        id=rid,
        content=f"content-{rid}",
        score=score,
        source="test",
        metadata={"_created_at": _ts(hours_old), "_source_kind": source_kind},
    )


def test_hybrid_entity_query_shifts_weight_to_keyword():
    cfg = RetrievalConfig(vector_weight=0.7, keyword_weight=0.3, adaptive_weights=True, enable_decay=False)
    kw = _KeywordStub([_mk_result("kw-hit", 0.2, 0.1)])
    vec = _VectorStub([_mk_result("vec-hit", 0.2, 0.1)])
    hybrid = HybridSearch(kw, vec, config=cfg)

    results = hybrid.search("find user_id abc-123", query_vector=None, top_k=5)
    # keyword-only path (no vector) still works and returns deterministic ordering
    assert results and results[0].id == "kw-hit"

    merged = hybrid.search("find user_id abc-123", query_vector=object(), top_k=5)
    assert merged[0].id == "kw-hit"


def test_hybrid_query_expansion_applies_to_open_domain_queries():
    cfg = RetrievalConfig(
        vector_weight=0.7,
        keyword_weight=0.3,
        adaptive_weights=False,
        enable_decay=False,
        enable_query_expansion=True,
        query_expansion_max_terms=4,
    )
    kw = _KeywordStub([_mk_result("kw-hit", 0.2, 0.1)])
    vec = _VectorStub([])
    hybrid = HybridSearch(kw, vec, config=cfg)
    query = "who is the owner of reading football club"
    out = hybrid.search(query, query_vector=None, top_k=5)
    assert out and out[0].id == "kw-hit"
    assert kw.last_query != query
    assert "owns" in kw.last_query or "person" in kw.last_query


def test_hybrid_query_expansion_skips_benchmark_case_tokens():
    cfg = RetrievalConfig(
        vector_weight=0.7,
        keyword_weight=0.3,
        adaptive_weights=False,
        enable_decay=False,
        enable_query_expansion=True,
        query_expansion_max_terms=4,
    )
    kw = _KeywordStub([_mk_result("kw-hit", 0.2, 0.1)])
    vec = _VectorStub([])
    hybrid = HybridSearch(kw, vec, config=cfg)
    query = "__DMR_CASE_00012__ who is the owner of reading football club"
    out = hybrid.search(query, query_vector=None, top_k=5)
    assert out and out[0].id == "kw-hit"
    assert kw.last_query == query


def test_hybrid_decay_and_access_boost_are_applied():
    cfg = RetrievalConfig(
        vector_weight=0.7,
        keyword_weight=0.3,
        adaptive_weights=False,
        enable_decay=True,
        decay_half_life_hours=1.0,
        decay_min_factor=0.5,
        access_boost_per_hit=0.5,
        access_boost_cap=2.0,
    )
    old = _mk_result("old", score=1.0, hours_old=24.0)
    fresh = _mk_result("fresh", score=0.8, hours_old=0.0)

    kw = _KeywordStub([old, fresh])
    vec = _VectorStub([])

    # No access history: fresher entry should win.
    hybrid_a = HybridSearch(kw, vec, config=cfg, access_counts_getter=lambda ids: {})
    results_a = hybrid_a.search("any query", query_vector=None, top_k=2)
    assert [r.id for r in results_a] == ["fresh", "old"]

    # High access count on old entry can reinforce it.
    hybrid_b = HybridSearch(kw, vec, config=cfg, access_counts_getter=lambda ids: {"old": 10})
    results_b = hybrid_b.search("any query", query_vector=None, top_k=2)
    assert [r.id for r in results_b] == ["old", "fresh"]


def test_hybrid_rrf_overlap_bonus_promotes_dual_source_hit():
    def _r(rid: str) -> SearchResult:
        return _mk_result(rid, score=0.1, hours_old=0.1)

    kw_results = [_r(f"kw-{i:02d}") for i in range(40)] + [_r("both")]
    vec_results = [_r("vec-top")] + [_r(f"vec-{i:02d}") for i in range(39)] + [_r("both")]

    cfg_no_boost = RetrievalConfig(
        vector_weight=0.7,
        keyword_weight=0.3,
        adaptive_weights=False,
        enable_decay=False,
        rrf_k=30,
        rrf_overlap_boost=1.0,
    )
    h_no = HybridSearch(_KeywordStub(kw_results), _VectorStub(vec_results), config=cfg_no_boost)
    out_no = h_no.search("generic query", query_vector=object(), top_k=3)
    assert out_no[0].id != "both"

    cfg_boost = RetrievalConfig(
        vector_weight=0.7,
        keyword_weight=0.3,
        adaptive_weights=False,
        enable_decay=False,
        rrf_k=30,
        rrf_overlap_boost=2.0,
    )
    h_yes = HybridSearch(_KeywordStub(kw_results), _VectorStub(vec_results), config=cfg_boost)
    out_yes = h_yes.search("generic query", query_vector=object(), top_k=3)
    assert out_yes[0].id == "both"


def test_hash_embedder_is_deterministic_and_normalized():
    async def _run() -> None:
        emb = HashEmbedder(dims=64)
        v1 = await emb.embed_single("alpha beta gamma")
        v2 = await emb.embed_single("alpha beta gamma")
        v3 = await emb.embed_single("delta epsilon zeta")
        assert v1.shape == (64,)
        assert abs(float((v1 * v1).sum()) - 1.0) < 1e-5
        assert abs(float((v2 * v2).sum()) - 1.0) < 1e-5
        assert (v1 == v2).all()
        assert not (v1 == v3).all()

        cfg = Config().venice.model_copy(update={"embedding_provider": "hash", "embedding_dims": 64})
        backend = create_embedder(cfg)
        arr = await backend.embed(["a b c", "d e f"])
        assert arr.shape == (2, 64)
        await backend.close()

    asyncio.run(_run())


def test_protocol_clients_and_cos_pruning(tmp_path):
    async def _run() -> None:
        config = Config()
        config.data_dir = tmp_path
        config.cos.max_key_facts = 3
        config.cos.max_open_questions = 2
        config.ensure_dirs()

        spine = MemorySpine(config)
        try:
            # COS retention bounds.
            session_id = spine.start_session("s-1")
            spine.cos.create(session_id, "start", key_facts=["f1"], open_questions=["q1"])
            spine.cos.update(session_id, "s2", new_facts=["f2", "f3"], new_questions=["q2"])
            spine.cos.update(session_id, "s3", new_facts=["f4"], new_questions=["q3"])
            latest = spine.cos.get_latest(session_id)
            assert latest is not None
            assert latest.key_facts == ["f2", "f3", "f4"]
            assert latest.open_questions == ["q2", "q3"]

            # Stable protocol client.
            client_v1 = spine.protocol_client("v1")
            chunk_ids = await client_v1.ingest(
                "Desmond prefers direct status updates.", source_id="test"
            )
            assert chunk_ids

            recalled = await client_v1.recall("direct status", top_k=3)
            assert recalled
            assert "direct" in recalled[0]["content"].lower()

            context = await client_v1.augment(
                "direct status",
                top_k=2,
                format="xml",
                min_score=0.0,
                roles=["unknown"],
            )
            assert context.startswith("<relevant-memories>")

            status = client_v1.status()
            assert status["protocol_version"] == "v1"

            client_v2 = spine.protocol_client("v2")
            client_v2.set_decay_config(12.0)
            assert spine.config.retrieval.decay_half_life_hours == 12.0
            graph = await client_v2.graph_query("Desmond", depth=2)
            assert graph["mode"] == "graph"

            # Case token is propagated to all chunks from the same source text.
            case_chunk_ids = await client_v1.ingest(
                "__DMR_CASE_00001__ " + ("Alpha beta gamma delta " * 120),
                source_id="bench:case",
            )
            assert len(case_chunk_ids) >= 2
            for chunk_id in case_chunk_ids:
                row = spine.sqlite.get_chunk(chunk_id)
                assert row is not None
                assert row.metadata.get("benchmark_case_token") == "__DMR_CASE_00001__"
        finally:
            await spine.close()

    asyncio.run(_run())


def test_recall_dedupes_by_benchmark_doc_id(tmp_path):
    async def _run() -> None:
        cfg = Config()
        cfg.data_dir = tmp_path
        cfg.ensure_dirs()

        spine = MemorySpine(cfg)
        try:
            async def _fake_search(query: str, top_k: int | None = None) -> list[SearchResult]:
                return [
                    SearchResult(
                        id="a",
                        content="Alpha one",
                        score=0.9,
                        source="test",
                        metadata={"benchmark_doc_id": "doc-1", "benchmark_source": "bench:doc-1"},
                    ),
                    SearchResult(
                        id="b",
                        content="Alpha two",
                        score=0.8,
                        source="test",
                        metadata={"benchmark_doc_id": "doc-1", "benchmark_source": "bench:doc-1"},
                    ),
                    SearchResult(
                        id="c",
                        content="Bravo one",
                        score=0.7,
                        source="test",
                        metadata={"benchmark_doc_id": "doc-2", "benchmark_source": "bench:doc-2"},
                    ),
                ]

            spine.search = _fake_search  # type: ignore[method-assign]
            rows = await spine.recall("alpha", top_k=3)
            assert len(rows) == 2
            assert [r["metadata"]["benchmark_doc_id"] for r in rows] == ["doc-1", "doc-2"]
        finally:
            await spine.close()

    asyncio.run(_run())


def test_case_token_query_skips_graph_lookup(tmp_path):
    async def _run() -> None:
        cfg = Config()
        cfg.data_dir = tmp_path
        cfg.ensure_dirs()
        spine = MemorySpine(cfg)
        try:
            spine.ingest_text_sync(
                "__DMR_CASE_00007__ Alpha memory that should be found quickly.",
                source_id="bench:case",
            )
            called = {"graph": False}

            def _graph_probe(query: str, limit: int = 0):  # noqa: ANN202
                called["graph"] = True
                return []

            spine.sqlite.search_graph_context = _graph_probe  # type: ignore[method-assign]
            rows = await spine.recall("__DMR_CASE_00007__ alpha memory", top_k=3)
            assert rows
            assert called["graph"] is False
        finally:
            await spine.close()

    asyncio.run(_run())


def test_ingest_sync_skip_chunking_uses_single_chunk(tmp_path):
    cfg = Config()
    cfg.data_dir = tmp_path
    cfg.ensure_dirs()
    spine = MemorySpine(cfg)
    try:
        long_text = "__DMR_CASE_00099__ " + ("alpha beta gamma delta " * 250)
        chunked_ids = spine.ingest_text_sync(long_text, source_id="bench:chunked")
        single_ids = spine.ingest_text_sync(
            long_text,
            source_id="bench:single",
            skip_chunking=True,
        )
        assert len(chunked_ids) > 1
        assert len(single_ids) == 1
    finally:
        asyncio.run(spine.close())
