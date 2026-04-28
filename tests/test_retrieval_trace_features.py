from c3ae.retrieval.features import extract_candidate_features
from c3ae.retrieval.hybrid import HybridSearch
from c3ae.types import SearchResult


class _Keyword:
    def search_all(self, query, limit=20):  # noqa: ANN001
        return [
            SearchResult(
                id="a",
                content="Caroline researched adoption agencies.",
                score=1.0,
                source="fts5",
                metadata={"role": "user"},
            )
        ]


class _Vector:
    def search(self, query_vector, top_k=20):  # noqa: ANN001
        return [
            SearchResult(
                id="b",
                content="Caroline wrote notes about adoption agencies.",
                score=0.8,
                source="vector",
                metadata={"_source_kind": "chunk"},
            )
        ]


def test_hybrid_search_records_trace_when_enabled():
    search = HybridSearch(_Keyword(), _Vector())
    search.config.enable_tracing = True

    rows = search.search("What did Caroline research?", query_vector=None, top_k=1)

    assert rows
    assert "keyword_search_ms" in search.last_trace.timings_ms
    assert "total_ms" in search.last_trace.timings_ms


def test_candidate_feature_extractor_keeps_ranking_data_separate():
    rows = [
        SearchResult(
            id="a",
            content="Caroline researched adoption agencies.",
            score=0.42,
            source="hybrid",
            metadata={"role": "user", "_source_kind": "chunk"},
        )
    ]

    features = extract_candidate_features("What did Caroline research?", rows)

    assert features[0].chunk_id == "a"
    assert features[0].rrf_score == 0.42
    assert features[0].token_overlap > 0
    assert features[0].role == "user"
