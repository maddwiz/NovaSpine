from __future__ import annotations

import pytest

from c3ae.retrieval.chained import (
    build_expanded_query,
    chained_recall,
    extract_entities,
    merge_rrf_rows,
)


def test_extract_entities_finds_capitalized_names():
    ents = extract_entities("Caroline met Melanie in New York and called NovaTech support.", max_entities=6)
    assert "Caroline" in ents
    assert "Melanie" in ents
    assert "New York" in ents


def test_build_expanded_query_adds_discovered_entities():
    rows = [
        {"id": "1", "content": "Caroline moved from Sweden to Denver.", "metadata": {}},
        {"id": "2", "content": "Melanie painted a sunset in July.", "metadata": {}},
    ]
    q = build_expanded_query("How long has Caroline had her friends?", rows, entity_limit=3)
    assert q != "How long has Caroline had her friends?"
    assert "Sweden" in q or "Melanie" in q


def test_merge_rrf_rows_prefers_overlap_candidates():
    a = [
        {"id": "x", "content": "alpha", "score": 1.0, "metadata": {}},
        {"id": "y", "content": "beta", "score": 0.9, "metadata": {}},
    ]
    b = [
        {"id": "x", "content": "alpha", "score": 1.0, "metadata": {}},
        {"id": "z", "content": "gamma", "score": 0.8, "metadata": {}},
    ]
    out = merge_rrf_rows(a, b, top_k=3, rrf_k=30)
    assert out
    assert out[0]["id"] == "x"


class _SpineStub:
    async def recall(self, query: str, top_k: int = 20, session_filter: str | None = None):
        if "Sweden" in query:
            return [
                {"id": "2", "content": "Caroline moved from Sweden in 2019.", "score": 0.9, "metadata": {}},
                {"id": "1", "content": "Caroline has a close friend group.", "score": 0.8, "metadata": {}},
            ][:top_k]
        return [
            {"id": "1", "content": "Caroline has a close friend group and mentioned Sweden before.", "score": 1.0, "metadata": {}},
            {"id": "3", "content": "Random other note.", "score": 0.4, "metadata": {}},
        ][:top_k]


@pytest.mark.asyncio
async def test_chained_recall_runs_second_hop_and_merges():
    out = await chained_recall(
        spine=_SpineStub(),
        query="How long has Caroline had her current group of friends?",
        top_k=3,
        max_hops=2,
        entity_limit=3,
        fetch_k=3,
    )
    assert out
    ids = [r["id"] for r in out]
    assert "2" in ids
