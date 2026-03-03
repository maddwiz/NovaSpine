from __future__ import annotations

from c3ae.retrieval.cross_encoder import CrossEncoderReranker
from c3ae.types import SearchResult


class _FakeModel:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores

    def predict(self, pairs):
        return self.scores[: len(pairs)]


def test_cross_encoder_reranks_rows_and_sets_metadata_score():
    reranker = CrossEncoderReranker(model=_FakeModel([0.1, 0.9, 0.2]))
    rows = [
        {"id": "a", "content": "alpha item", "score": 0.2, "metadata": {}},
        {"id": "b", "content": "beta item", "score": 0.2, "metadata": {}},
        {"id": "c", "content": "gamma item", "score": 0.2, "metadata": {}},
    ]
    out = reranker.rerank_rows("beta query", rows, top_n=2)
    assert [r["id"] for r in out] == ["b", "c"]
    assert float(out[0]["score"]) == 0.9
    assert float((out[0]["metadata"] or {}).get("cross_encoder_score", 0.0)) == 0.9


def test_cross_encoder_reranks_search_results():
    reranker = CrossEncoderReranker(model=_FakeModel([0.6, 0.2]))
    results = [
        SearchResult(id="x", content="first", score=0.01, source="hybrid", metadata={}),
        SearchResult(id="y", content="second", score=0.01, source="hybrid", metadata={}),
    ]
    out = reranker.rerank_results("query", results, top_n=2)
    assert [r.id for r in out] == ["x", "y"]
    assert out[0].source == "cross_encoder"
    assert float(out[0].metadata.get("cross_encoder_score", 0.0)) == 0.6

