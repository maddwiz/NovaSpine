"""Hybrid retrieval: weighted merge of keyword + vector search."""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Callable

import numpy as np

from c3ae.config import RetrievalConfig
from c3ae.retrieval.keyword import KeywordSearch
from c3ae.retrieval.vector import VectorSearch
from c3ae.types import SearchResult


class HybridSearch:
    """Combines FTS5 keyword and FAISS vector search with weighted scoring."""

    def __init__(
        self,
        keyword: KeywordSearch,
        vector: VectorSearch,
        config: RetrievalConfig | None = None,
        access_count_getter: Callable[[str], int] | None = None,
        access_counts_getter: Callable[[list[str]], dict[str, int]] | None = None,
    ) -> None:
        self.keyword = keyword
        self.vector = vector
        self.config = config or RetrievalConfig()
        self._access_count_getter = access_count_getter
        self._access_counts_getter = access_counts_getter

    def search(
        self,
        query: str,
        query_vector: np.ndarray | None = None,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Run hybrid search.

        If query_vector is None, falls back to keyword-only search.
        """
        top_k = top_k or self.config.default_top_k
        # Over-fetch aggressively to reduce false negatives before fusion.
        fetch_k = max(top_k * 5, 100)

        # Keyword results
        kw_results = self.keyword.search_all(query, limit=fetch_k)

        # Vector results (if we have an embedding)
        vec_results: list[SearchResult] = []
        if query_vector is not None:
            vec_results = self.vector.search(query_vector, top_k=fetch_k)

        if not vec_results:
            return self._rerank_with_decay(kw_results, top_k=top_k)
        if not kw_results:
            return self._rerank_with_decay(vec_results, top_k=top_k)

        merged = self._merge(kw_results, vec_results, query=query, top_k=fetch_k)
        return self._rerank_with_decay(merged, top_k=top_k)

    def _merge(
        self,
        kw_results: list[SearchResult],
        vec_results: list[SearchResult],
        query: str,
        top_k: int,
    ) -> list[SearchResult]:
        """Reciprocal rank fusion with configurable weights."""
        k = 60  # RRF constant
        keyword_weight, vector_weight = self._weights_for_query(query)

        scores: dict[str, float] = {}
        best_result: dict[str, SearchResult] = {}

        # Keyword contributions
        for rank, r in enumerate(kw_results):
            rrf = keyword_weight / (k + rank + 1)
            scores[r.id] = scores.get(r.id, 0.0) + rrf
            if r.id not in best_result:
                best_result[r.id] = r

        # Vector contributions
        for rank, r in enumerate(vec_results):
            rrf = vector_weight / (k + rank + 1)
            scores[r.id] = scores.get(r.id, 0.0) + rrf
            if r.id not in best_result:
                best_result[r.id] = r

        # Sort by combined score
        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
        results = []
        for rid in sorted_ids[:top_k]:
            r = best_result[rid]
            results.append(SearchResult(
                id=r.id,
                content=r.content,
                score=scores[rid],
                source="hybrid",
                metadata=r.metadata,
            ))
        return results

    def _weights_for_query(self, query: str) -> tuple[float, float]:
        kw = max(0.0001, float(self.config.keyword_weight))
        vec = max(0.0001, float(self.config.vector_weight))
        if not self.config.adaptive_weights:
            total = kw + vec
            return kw / total, vec / total

        intent = self._classify_intent(query)
        if intent == "entity_lookup":
            kw *= 2.0
            vec *= 0.5
        elif intent == "temporal":
            kw *= 1.3
            vec *= 0.9
        else:  # conceptual
            kw *= 0.6
            vec *= 1.5

        total = kw + vec
        return kw / total, vec / total

    @staticmethod
    def _classify_intent(query: str) -> str:
        q = query.strip().lower()
        temporal_terms = {
            "when", "before", "after", "timeline", "history", "previous",
            "earlier", "later", "recent", "latest", "last",
        }
        entity_terms = {
            "who", "id", "symbol", "ticker", "account", "email",
            "name", "exact", "lookup", "find",
        }
        tokens = set(re.findall(r"[a-z0-9_\-]+", q))

        if '"' in q or "'" in q:
            return "entity_lookup"
        if tokens & temporal_terms:
            return "temporal"
        if tokens & entity_terms:
            return "entity_lookup"
        if re.search(r"\b[a-z]+[-_][a-z0-9]+\b", q):
            return "entity_lookup"
        if re.search(r"\b[A-Z]{2,}\b", query):
            return "entity_lookup"
        return "conceptual"

    def _rerank_with_decay(self, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        if not results:
            return []

        access_counts: dict[str, int] = {}
        if self._access_counts_getter is not None:
            try:
                access_counts = self._access_counts_getter([r.id for r in results])
            except Exception:
                access_counts = {}

        rescored: list[SearchResult] = []
        for r in results:
            multiplier = self._score_multiplier(r, access_counts=access_counts)
            rescored.append(
                SearchResult(
                    id=r.id,
                    content=r.content,
                    score=float(r.score) * multiplier,
                    source=r.source,
                    metadata=r.metadata,
                )
            )
        rescored.sort(key=lambda x: x.score, reverse=True)
        return rescored[:top_k]

    def _score_multiplier(self, result: SearchResult, access_counts: dict[str, int]) -> float:
        if not self.config.enable_decay:
            return 1.0

        metadata = result.metadata or {}
        created_at_raw = metadata.get("_created_at") or metadata.get("created_at")
        created_at = self._parse_datetime(created_at_raw)

        decay = 1.0
        if created_at is not None:
            age_hours = max(
                0.0,
                (datetime.now(timezone.utc) - created_at).total_seconds() / 3600.0,
            )
            half_life = max(1.0, float(self.config.decay_half_life_hours))
            decay = math.exp(-math.log(2.0) * (age_hours / half_life))
            decay = max(float(self.config.decay_min_factor), decay)

        access_count = access_counts.get(result.id, 0)
        if access_count == 0 and self._access_count_getter is not None:
            try:
                access_count = int(self._access_count_getter(result.id))
            except Exception:
                access_count = 0
        access_boost = min(
            1.0 + float(self.config.access_boost_per_hit) * max(0, access_count),
            float(self.config.access_boost_cap),
        )

        source_kind = str(metadata.get("_source_kind", "")).lower()
        entry_type = str(metadata.get("type", "")).lower()
        importance = (
            float(self.config.evidence_importance_boost)
            if source_kind == "reasoning_entry" or entry_type == "reasoning_entry"
            else 1.0
        )

        return decay * access_boost * importance

    @staticmethod
    def _parse_datetime(raw: object) -> datetime | None:
        if not isinstance(raw, str) or not raw.strip():
            return None
        s = raw.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            return None
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
