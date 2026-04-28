"""Candidate feature extraction for future learned rerankers."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel

from c3ae.types import SearchResult

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "to", "of", "in", "on",
    "for", "and", "or", "with", "what", "when", "who", "where", "which",
    "did", "do", "does", "at", "by", "from", "as", "it", "be", "this",
    "that", "how", "many", "much",
}


class CandidateFeatures(BaseModel):
    chunk_id: str
    bm25_score: float | None = None
    vector_score: float | None = None
    rrf_score: float | None = None
    graph_score: float | None = None
    fact_score: float | None = None
    entity_overlap: float = 0.0
    token_overlap: float = 0.0
    recency_score: float | None = None
    importance_score: float | None = None
    access_count: int = 0
    source_kind: str = ""
    role: str = ""


def _tokens(text: str) -> set[str]:
    return {
        tok
        for tok in re.findall(r"[a-z0-9_]+", (text or "").lower())
        if len(tok) > 1 and tok not in _STOPWORDS
    }


def _entities(text: str) -> set[str]:
    return {m.group(0).lower() for m in re.finditer(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})?\b", text or "")}


def _score_for_source(source: str, score: float) -> dict[str, float | None]:
    source_l = (source or "").lower()
    out: dict[str, float | None] = {
        "bm25_score": None,
        "vector_score": None,
        "rrf_score": None,
        "graph_score": None,
        "fact_score": None,
    }
    if source_l in {"fts5", "keyword"}:
        out["bm25_score"] = score
    elif source_l == "vector":
        out["vector_score"] = score
    elif source_l in {"graph", "graph_context"}:
        out["graph_score"] = score
    elif source_l.startswith("structured_") or source_l == "structured_fact":
        out["fact_score"] = score
    else:
        out["rrf_score"] = score
    return out


def extract_candidate_features(
    query: str,
    candidates: list[SearchResult],
    *,
    access_counts: dict[str, int] | None = None,
) -> list[CandidateFeatures]:
    q_tokens = _tokens(query)
    q_entities = _entities(query)
    access_counts = access_counts or {}
    features: list[CandidateFeatures] = []
    for row in candidates:
        metadata: dict[str, Any] = row.metadata or {}
        c_tokens = _tokens(row.content)
        c_entities = _entities(row.content)
        token_overlap = len(q_tokens & c_tokens) / max(1, len(q_tokens)) if q_tokens else 0.0
        entity_overlap = len(q_entities & c_entities) / max(1, len(q_entities)) if q_entities else 0.0
        scores = _score_for_source(row.source, float(row.score))
        features.append(
            CandidateFeatures(
                chunk_id=row.id,
                token_overlap=round(token_overlap, 4),
                entity_overlap=round(entity_overlap, 4),
                access_count=int(access_counts.get(row.id, 0)),
                source_kind=str(metadata.get("_source_kind", "")),
                role=str(metadata.get("role", "")),
                **scores,
            )
        )
    return features
