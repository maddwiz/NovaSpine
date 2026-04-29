from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from c3ae.config import Config, RetrievalConfig
from c3ae.embeddings.backends import HashEmbedder, create_embedder
from c3ae.memory_spine.spine import MemorySpine, _RECALL_OVERFETCH_CAP
from c3ae.retrieval.hybrid import HybridSearch
from c3ae.types import Chunk, SearchResult


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
    metadata: dict[str, object] | None = None,
) -> SearchResult:
    return SearchResult(
        id=rid,
        content=f"content-{rid}",
        score=score,
        source="test",
        metadata={
            "_created_at": _ts(hours_old),
            "_source_kind": source_kind,
            **(metadata or {}),
        },
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


def test_hybrid_access_boost_can_be_disabled():
    cfg = RetrievalConfig(
        vector_weight=0.7,
        keyword_weight=0.3,
        adaptive_weights=False,
        enable_decay=True,
        decay_half_life_hours=1.0,
        decay_min_factor=0.5,
        enable_access_boost=False,
        access_boost_per_hit=0.5,
        access_boost_cap=2.0,
    )
    old = _mk_result("old", score=1.0, hours_old=24.0)
    fresh = _mk_result("fresh", score=0.8, hours_old=0.0)

    hybrid = HybridSearch(
        _KeywordStub([old, fresh]),
        _VectorStub([]),
        config=cfg,
        access_counts_getter=lambda ids: {"old": 10},
    )
    results = hybrid.search("any query", query_vector=None, top_k=2)

    assert [r.id for r in results] == ["fresh", "old"]


def test_hybrid_apply_decay_flag_disables_decay_reranking():
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
    hybrid = HybridSearch(kw, vec, config=cfg, access_counts_getter=lambda ids: {})

    without_decay = hybrid.search("any query", query_vector=None, top_k=2, apply_decay=False)
    with_decay = hybrid.search("any query", query_vector=None, top_k=2, apply_decay=True)

    assert [r.id for r in without_decay] == ["old", "fresh"]
    assert [r.id for r in with_decay] == ["fresh", "old"]


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


def test_hybrid_first_person_query_boosts_personal_scope():
    cfg = RetrievalConfig(
        vector_weight=0.7,
        keyword_weight=0.3,
        adaptive_weights=False,
        enable_decay=False,
        personal_memory_boost=1.8,
        shared_memory_penalty=0.8,
    )
    team_hit = _mk_result(
        "team-hit",
        score=0.5,
        hours_old=0.1,
        metadata={"path": "memory/sessions/team/2026-01-01-team.md"},
    )
    personal_hit = _mk_result(
        "personal-hit",
        score=0.5,
        hours_old=0.1,
        metadata={"path": "memory/sessions/personal/2026-01-01-me.md", "session_id": "personal-01"},
    )
    kw = _KeywordStub([team_hit, personal_hit])
    vec = _VectorStub([team_hit, personal_hit])
    hybrid = HybridSearch(kw, vec, config=cfg)

    merged = hybrid.search("What coffee do I usually order?", query_vector=object(), top_k=2)
    assert [r.id for r in merged] == ["personal-hit", "team-hit"]


def test_query_profile_infers_relation_hints_for_current_state_questions():
    profile = HybridSearch.analyze_query("What espresso order do I default to now?")
    assert profile.is_first_person is True
    assert profile.wants_current_state is True
    assert profile.fact_relation_hints == ("coffee_order",)

    changed = HybridSearch.analyze_query("What espresso drink did I move on from, and what do I order now?")
    assert changed.wants_current_state is True
    assert changed.wants_history is True
    assert changed.fact_relation_hints == ("coffee_order",)

    multi = HybridSearch.analyze_query(
        "What's my current travel profile in one short list: base city, coffee, seat, bag, charger, and notebook?"
    )
    assert set(multi.fact_relation_hints) >= {"location", "coffee_order", "flight_seat", "bag", "charger", "notebook"}


def test_query_profile_detects_literal_assistant_recall_queries():
    profile = HybridSearch.analyze_query(
        "I think we discussed work from home jobs for seniors earlier. "
        "Can you remind me what was the 7th job in the list you provided?"
    )
    assert profile.wants_literal_recall is True
    assert profile.prefers_assistant_response is True


def test_literal_recall_queries_penalize_structured_results_and_boost_assistant_turns():
    cfg = RetrievalConfig(
        vector_weight=0.7,
        keyword_weight=0.3,
        adaptive_weights=False,
        enable_decay=False,
    )
    hybrid = HybridSearch(_KeywordStub([]), _VectorStub([]), config=cfg)
    profile = HybridSearch.analyze_query(
        "I'm going back to our previous conversation about DIY home decor. "
        "Can you remind me what sealant you recommended last time?"
    )

    assistant_multiplier = hybrid._scope_multiplier({"role": "assistant"}, profile)
    structured_multiplier = hybrid._scope_multiplier(
        {"_source_kind": "structured_truth", "relation": "location"},
        profile,
    )

    assert assistant_multiplier > 1.0
    assert structured_multiplier < 1.0
    assert assistant_multiplier > structured_multiplier


def test_route_structured_query_fans_out_multi_relation_current_queries(tmp_path, monkeypatch):
    config = Config()
    config.data_dir = tmp_path
    config.ingestion.enable_fact_extraction = True
    config.ensure_dirs()

    spine = MemorySpine(config)
    profile = spine.hybrid_search.analyze_query(
        "What's my current travel profile in one short list: base city, coffee, seat, bag, charger, and notebook?"
    )

    seen: list[str] = []

    def _fake_current(*, entity: str = "", relation: str = "", limit: int = 0):
        seen.append(relation)
        return []

    monkeypatch.setattr(spine.sqlite, "list_current_structured_facts", _fake_current)
    assert spine._route_structured_query(
        "What's my current travel profile in one short list: base city, coffee, seat, bag, charger, and notebook?",
        profile,
        top_k=5,
    ) == []
    assert set(seen) >= {"location", "coffee_order", "flight_seat", "bag", "charger", "notebook"}


def test_route_structured_query_uses_relation_hint_for_specific_queries(tmp_path, monkeypatch):
    config = Config()
    config.data_dir = tmp_path
    config.ingestion.enable_fact_extraction = True
    config.ensure_dirs()

    spine = MemorySpine(config)
    profile = spine.hybrid_search.analyze_query("What espresso order do I default to now?")
    seen: dict[str, object] = {}

    def _fake_current(*, entity: str = "", relation: str = "", limit: int = 0):
        seen["entity"] = entity
        seen["relation"] = relation
        seen["limit"] = limit
        return []

    monkeypatch.setattr(spine.sqlite, "list_current_structured_facts", _fake_current)
    assert spine._route_structured_query("What espresso order do I default to now?", profile, top_k=4) == []
    assert seen == {"entity": "User", "relation": "coffee_order", "limit": 16}


def test_fact_source_metadata_hoists_nested_source_provenance(tmp_path):
    config = Config()
    config.data_dir = tmp_path
    config.ensure_dirs()
    spine = MemorySpine(config)
    merged = spine._fact_source_metadata(
        {
            "source_chunk_id": "chunk-1",
            "metadata": {
                "source": {
                    "path": "memory/sessions/personal/2026-01-01.md",
                    "session_id": "personal-01",
                    "source_id": "session:personal-01",
                    "memory_scope": "personal",
                }
            },
        }
    )
    assert merged["path"] == "memory/sessions/personal/2026-01-01.md"
    assert merged["session_id"] == "personal-01"
    assert merged["source_id"] == "session:personal-01"
    assert merged["memory_scope"] == "personal"
    assert merged["source_chunk_id"] == "chunk-1"


def test_build_chunk_metadata_infers_role_from_transcript_content(tmp_path):
    cfg = Config()
    cfg.data_dir = tmp_path
    cfg.ensure_dirs()
    spine = MemorySpine(cfg)
    try:
        metadata = spine._build_chunk_metadata(
            "Benchmark: longmemeval_m\n\nassistant: The seventh job was virtual bookkeeping.",
            "bench:doc",
            {},
        )
        assert metadata["role"] == "assistant"
    finally:
        asyncio.run(spine.close())


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


def test_augment_overfetch_is_capped(tmp_path):
    async def _run() -> None:
        cfg = Config()
        cfg.data_dir = tmp_path
        cfg.ensure_dirs()

        spine = MemorySpine(cfg)
        try:
            observed_top_k: list[int | None] = []

            async def _fake_search(query: str, top_k: int | None = None) -> list[SearchResult]:
                observed_top_k.append(top_k)
                return [
                    SearchResult(
                        id="alpha",
                        content="Alpha memory",
                        score=0.9,
                        source="test",
                        metadata={"role": "assistant"},
                    )
                ]

            spine.search = _fake_search  # type: ignore[method-assign]
            context = await spine.augment("test", top_k=50, min_score=0.0)
            assert context.startswith("<relevant-memories>")
            assert observed_top_k
            assert observed_top_k[0] == _RECALL_OVERFETCH_CAP
            assert observed_top_k[0] <= _RECALL_OVERFETCH_CAP
        finally:
            await spine.close()

    asyncio.run(_run())


def test_augment_keeps_structured_facts_without_conversation_role(tmp_path):
    async def _run() -> None:
        cfg = Config()
        cfg.data_dir = tmp_path
        cfg.ensure_dirs()

        spine = MemorySpine(cfg)
        try:
            async def _fake_search(query: str, top_k: int | None = None) -> list[SearchResult]:
                return [
                    SearchResult(
                        id="fact-1",
                        content="User notebook Field Notes",
                        score=4.0,
                        source="structured_fact_current",
                        metadata={"_source_kind": "structured_fact_current"},
                    ),
                    SearchResult(
                        id="unknown-1",
                        content="Unknown role scratch",
                        score=4.0,
                        source="test",
                        metadata={},
                    ),
                ]

            spine.search = _fake_search  # type: ignore[method-assign]
            context = await spine.augment("notebook", top_k=5, min_score=0.0)
            assert "User notebook Field Notes" in context
            assert "Unknown role scratch" not in context
        finally:
            await spine.close()

    asyncio.run(_run())


def test_structured_truth_history_excludes_duplicate_current_values(tmp_path):
    cfg = Config()
    cfg.data_dir = tmp_path
    cfg.ensure_dirs()
    spine = MemorySpine(cfg)
    try:
        rows = spine._rank_structured_truth(
            "What bag did I move on from, and what bag replaced it?",
            [
                {
                    "entity": "User",
                    "relation": "bag",
                    "current_facts": [{"value": "Peak Design 20L"}],
                    "historical_facts": [
                        {"value": "olive Evergoods CPL24"},
                        {"value": "Peak Design 20L"},
                        {"value": "Peak Design 20L"},
                    ],
                }
            ],
            top_k=1,
        )
        assert rows
        content = rows[0].content
        assert "current: Peak Design 20L" in content
        assert "previous/historical: olive Evergoods CPL24" in content
        assert content.index("previous/historical") < content.index("current:")
        assert content.count("Peak Design 20L") == 1
    finally:
        asyncio.run(spine.close())


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


def test_literal_recall_query_skips_structured_shortcuts(tmp_path):
    async def _run() -> None:
        cfg = Config()
        cfg.data_dir = tmp_path
        cfg.ingestion.enable_fact_extraction = True
        cfg.graph.enabled = True
        cfg.ensure_dirs()
        spine = MemorySpine(cfg)
        try:
            called = {"route": False, "graph": False, "facts": False}

            async def _fake_embed_text(query: str):  # noqa: ANN202
                return None

            def _fake_route(query: str, profile, *, top_k: int):  # noqa: ANN001, ANN202
                called["route"] = True
                return [
                    SearchResult(
                        id="fact-1",
                        content="User location previous/historical: New Orleans",
                        score=9.0,
                        source="structured_truth",
                        metadata={"_source_kind": "structured_truth"},
                    )
                ]

            def _fake_hybrid_search(query: str, query_vector=None, top_k: int | None = None, apply_decay: bool = True):  # noqa: ANN001, ANN202
                return [
                    SearchResult(
                        id="assistant-1",
                        content="assistant: Use Mod Podge acrylic sealer for the newspaper flower vase.",
                        score=1.0,
                        source="hybrid",
                        metadata={"role": "assistant", "benchmark_doc_id": "answer_ultrachat_563222"},
                    )
                ]

            def _fake_graph(query: str, limit: int = 0):  # noqa: ANN202
                called["graph"] = True
                return []

            def _fake_facts(query: str, limit: int = 0):  # noqa: ANN202
                called["facts"] = True
                return [
                    SearchResult(
                        id="fact-1",
                        content="User location previous/historical: New Orleans",
                        score=9.0,
                        source="structured_truth",
                        metadata={"_source_kind": "structured_truth"},
                    )
                ]

            spine._embed_text = _fake_embed_text  # type: ignore[method-assign]
            spine._route_structured_query = _fake_route  # type: ignore[method-assign]
            spine.hybrid_search.search = _fake_hybrid_search  # type: ignore[method-assign]
            spine.sqlite.search_graph_context = _fake_graph  # type: ignore[method-assign]
            spine.sqlite.search_structured_facts_fts = _fake_facts  # type: ignore[method-assign]
            spine._record_access = lambda results: None  # type: ignore[method-assign]

            rows = await spine.search(
                "I'm going back to our previous conversation about DIY home decor projects. "
                "Can you remind me what sealant you recommended for the newspaper flower vase?",
                top_k=3,
            )
            assert rows
            assert rows[0].id == "assistant-1"
            assert called == {"route": False, "graph": False, "facts": False}
        finally:
            await spine.close()

    asyncio.run(_run())


def test_case_token_query_does_not_increment_access_counts(tmp_path):
    async def _run() -> None:
        cfg = Config()
        cfg.data_dir = tmp_path
        cfg.ensure_dirs()
        spine = MemorySpine(cfg)
        try:
            ids = spine.ingest_text_sync(
                "__DMR_CASE_00088__ Alpha benchmark memory for access tracking.",
                source_id="bench:case",
            )
            target_id = ids[0]
            before = spine.sqlite.get_memory_access_count(target_id)
            assert before == 0

            rows = await spine.recall("__DMR_CASE_00088__ alpha benchmark memory", top_k=3)
            assert rows

            after = spine.sqlite.get_memory_access_count(target_id)
            assert after == 0

            rows_non_bench = await spine.recall("alpha benchmark memory", top_k=3)
            assert rows_non_bench
            after_non_bench = spine.sqlite.get_memory_access_count(target_id)
            assert after_non_bench >= 1
        finally:
            await spine.close()

    asyncio.run(_run())


def test_access_tracking_can_be_disabled(tmp_path):
    cfg = Config()
    cfg.data_dir = tmp_path
    cfg.retrieval.enable_access_tracking = False
    cfg.ensure_dirs()
    spine = MemorySpine(cfg)
    try:
        chunk = Chunk(content="Alpha benchmark memory for access tracking.", source_id="bench:access")
        spine.sqlite.insert_chunk(chunk)

        rows = spine.search_keyword("alpha benchmark memory", top_k=3)

        assert rows
        assert spine.sqlite.get_memory_access_count(chunk.id) == 0
    finally:
        asyncio.run(spine.close())


def test_case_token_query_falls_back_when_tail_terms_miss(tmp_path):
    async def _run() -> None:
        cfg = Config()
        cfg.data_dir = tmp_path
        cfg.ensure_dirs()
        spine = MemorySpine(cfg)
        try:
            spine.ingest_text_sync(
                "__DMR_CASE_00123__ This passage intentionally uses rare wording.",
                source_id="bench:case",
                skip_chunking=True,
            )
            # The tail terms are absent from content; fallback should still return case-scoped chunk.
            rows = await spine.recall(
                "__DMR_CASE_00123__ who owns reading football club",
                top_k=3,
            )
            assert rows
            assert rows[0]["metadata"].get("benchmark_case_token") == "__DMR_CASE_00123__"
        finally:
            await spine.close()

    asyncio.run(_run())


def test_ingest_sync_skip_chunking_still_applies_embedding_safety_splits(tmp_path):
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
        assert len(single_ids) >= 1
        assert len(single_ids) < len(chunked_ids)
    finally:
        asyncio.run(spine.close())


def test_search_routes_first_person_current_state_queries_to_structured_truth(tmp_path):
    async def _run() -> None:
        cfg = Config()
        cfg.data_dir = tmp_path
        cfg.ensure_dirs()
        cfg.ingestion.enable_fact_extraction = True
        cfg.ingestion.fact_extraction_mode = "heuristic"
        cfg.venice = cfg.venice.model_copy(
            update={
                "embedding_provider": "hash",
                "embedding_model": "hash-test",
                "embedding_dims": 64,
            }
        )

        spine = MemorySpine(cfg)
        try:
            await spine.ingest(
                "Earlier in the year I was still based in Denver.",
                source_id="memory/sessions/personal/2026-02-02-session-04.md",
                metadata={"path": "memory/sessions/personal/2026-02-02-session-04.md", "session_id": "personal-04"},
            )
            await spine.ingest(
                "Since March, my home base has been Santa Fe.",
                source_id="memory/sessions/personal/2026-03-15-session-05.md",
                metadata={"path": "memory/sessions/personal/2026-03-15-session-05.md", "session_id": "personal-05"},
            )
            await spine.ingest(
                "Rowan usually works out of Phoenix.",
                source_id="memory/sessions/team/2026-03-15-team-001.md",
                metadata={"path": "memory/sessions/team/2026-03-15-team-001.md", "session_id": "team-001"},
            )

            rows = await spine.search("Where am I based these days?", top_k=3)
            assert rows
            assert rows[0].source == "structured_fact_current"
            assert "santa fe" in rows[0].content.lower()

            truth = await spine.search("Which city did I leave, and where am I now?", top_k=3)
            assert truth
            assert truth[0].source == "structured_truth"
            assert "santa fe" in truth[0].content.lower()
            assert "denver" in truth[0].content.lower()
        finally:
            await spine.close()

    asyncio.run(_run())
