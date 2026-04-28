"""Hybrid retrieval: weighted merge of keyword + vector search."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
import time
from datetime import datetime, timezone
from typing import Callable

import numpy as np

from c3ae.config import RetrievalConfig
from c3ae.retrieval.features import CandidateFeatures, extract_candidate_features
from c3ae.retrieval.keyword import KeywordSearch
from c3ae.retrieval.query_expansion import maybe_expand_query
from c3ae.retrieval.trace import RetrievalTrace
from c3ae.retrieval.vector import VectorSearch
from c3ae.types import SearchResult


_FIRST_PERSON_TERMS = {"i", "me", "my", "mine", "myself"}
_CURRENT_STATE_TERMS = {"current", "currently", "today", "now", "present", "latest"}
_HISTORY_TERMS = {
    "before",
    "previous",
    "previously",
    "earlier",
    "formerly",
    "history",
    "historical",
    "changed",
    "change",
    "changes",
    "leave",
    "left",
    "old",
    "older",
    "replaced",
    "replace",
    "from",
}
_PERSONAL_SCOPE_TERMS = ("personal", "private", "dm", "user", "direct")
_SHARED_SCOPE_TERMS = ("team", "shared", "channel", "group")
_LOCATION_HINT_TERMS = {"where", "based", "base", "live", "living", "city", "location", "home"}
_COFFEE_HINT_TERMS = {"coffee", "espresso", "latte", "cappuccino", "americano", "flat", "white", "milk"}
_BAG_HINT_TERMS = {"bag", "bags", "pack", "backpack", "backpacks", "carry"}
_NOTEBOOK_HINT_TERMS = {"notebook", "journal", "field", "notes", "moleskine"}
_CHARGER_HINT_TERMS = {"charger", "charging", "battery", "power"}
_SEAT_HINT_TERMS = {"seat", "aisle", "window", "middle", "plane", "planes", "flight", "boarding"}
_SNACK_HINT_TERMS = {"movie", "movies", "theater", "theatre", "film", "films", "concession", "snack", "popcorn", "kettle", "corn"}
_DRINK_HINT_TERMS = {"drink", "drinks", "water", "sparkling"}
_LITERAL_RECALL_PHRASES = (
    "remind me",
    "previous chat",
    "previous conversation",
    "our previous conversation",
    "our previous chat",
    "last time",
    "earlier",
    "we discussed",
    "we talked about",
    "going back to our previous conversation",
    "going back to our previous chat",
)
_ASSISTANT_REFERENCE_PHRASES = (
    "you recommended",
    "you provided",
    "you suggested",
    "you mentioned",
    "you told me",
    "you said",
)
_LIST_RECALL_TERMS = {
    "first",
    "second",
    "third",
    "fourth",
    "fifth",
    "sixth",
    "seventh",
    "eighth",
    "ninth",
    "tenth",
    "list",
}


@dataclass(frozen=True)
class QueryProfile:
    intent: str
    tokens: tuple[str, ...]
    is_first_person: bool = False
    wants_current_state: bool = False
    wants_history: bool = False
    has_multiple_facets: bool = False
    fact_relation_hints: tuple[str, ...] = ()
    wants_literal_recall: bool = False
    prefers_assistant_response: bool = False


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
        self.last_trace = RetrievalTrace()
        self.last_candidate_features: list[CandidateFeatures] = []

    def search(
        self,
        query: str,
        query_vector: np.ndarray | None = None,
        top_k: int | None = None,
        apply_decay: bool = True,
    ) -> list[SearchResult]:
        """Run hybrid search.

        If query_vector is None, falls back to keyword-only search.
        """
        total_start = time.perf_counter()
        trace = RetrievalTrace()
        self.last_candidate_features = []
        top_k = top_k or self.config.default_top_k
        profile = self.analyze_query(query)
        # Over-fetch aggressively to reduce false negatives before fusion.
        fetch_k = max(top_k * 5, 100)

        kw_query = query
        if self.config.enable_query_expansion and not profile.wants_literal_recall:
            kw_query = maybe_expand_query(
                query,
                max_extra_terms=int(self.config.query_expansion_max_terms),
            )

        # Keyword results
        stage_start = time.perf_counter()
        kw_results = self.keyword.search_all(kw_query, limit=fetch_k)
        trace.add_timing("keyword_search_ms", (time.perf_counter() - stage_start) * 1000.0)
        trace.add_counter("keyword_results", len(kw_results))

        # Vector results (if we have an embedding)
        vec_results: list[SearchResult] = []
        if query_vector is not None:
            stage_start = time.perf_counter()
            vec_results = self.vector.search(query_vector, top_k=fetch_k)
            trace.add_timing("vector_search_ms", (time.perf_counter() - stage_start) * 1000.0)
        else:
            trace.add_timing("vector_search_ms", 0.0)
        trace.add_counter("vector_results", len(vec_results))

        if not vec_results:
            stage_start = time.perf_counter()
            if not apply_decay:
                out = kw_results[:top_k]
            else:
                out = self._rerank_with_decay(kw_results, top_k=top_k, profile=profile)
            trace.add_timing("rerank_ms", (time.perf_counter() - stage_start) * 1000.0)
            trace.add_timing("fusion_ms", 0.0)
            self._finalize_trace(trace, total_start, out, query)
            return out
        if not kw_results:
            stage_start = time.perf_counter()
            if not apply_decay:
                out = vec_results[:top_k]
            else:
                out = self._rerank_with_decay(vec_results, top_k=top_k, profile=profile)
            trace.add_timing("rerank_ms", (time.perf_counter() - stage_start) * 1000.0)
            trace.add_timing("fusion_ms", 0.0)
            self._finalize_trace(trace, total_start, out, query)
            return out

        stage_start = time.perf_counter()
        merged = self._merge(kw_results, vec_results, query=query, top_k=fetch_k)
        trace.add_timing("fusion_ms", (time.perf_counter() - stage_start) * 1000.0)
        if not apply_decay:
            out = merged[:top_k]
            trace.add_timing("rerank_ms", 0.0)
        else:
            stage_start = time.perf_counter()
            out = self._rerank_with_decay(merged, top_k=top_k, profile=profile)
            trace.add_timing("rerank_ms", (time.perf_counter() - stage_start) * 1000.0)
        self._finalize_trace(trace, total_start, out, query)
        return out

    def _finalize_trace(
        self,
        trace: RetrievalTrace,
        total_start: float,
        results: list[SearchResult],
        query: str,
    ) -> None:
        trace.add_counter("final_results", len(results))
        trace.add_timing("total_ms", (time.perf_counter() - total_start) * 1000.0)
        self.last_trace = trace
        if bool(self.config.capture_candidate_features):
            access_counts: dict[str, int] = {}
            if self._access_counts_getter is not None:
                try:
                    access_counts = self._access_counts_getter([r.id for r in results])
                except Exception:
                    access_counts = {}
            self.last_candidate_features = extract_candidate_features(
                query,
                results,
                access_counts=access_counts,
            )

    def _merge(
        self,
        kw_results: list[SearchResult],
        vec_results: list[SearchResult],
        query: str,
        top_k: int,
    ) -> list[SearchResult]:
        """Reciprocal rank fusion with configurable weights."""
        k = max(1, int(self.config.rrf_k))
        keyword_weight, vector_weight = self._weights_for_query(query)

        scores: dict[str, float] = {}
        best_result: dict[str, SearchResult] = {}
        kw_ids: set[str] = set()
        vec_ids: set[str] = set()

        # Keyword contributions
        for rank, r in enumerate(kw_results):
            rrf = keyword_weight / (k + rank + 1)
            scores[r.id] = scores.get(r.id, 0.0) + rrf
            if r.id not in best_result:
                best_result[r.id] = r
            kw_ids.add(r.id)

        # Vector contributions
        for rank, r in enumerate(vec_results):
            rrf = vector_weight / (k + rank + 1)
            scores[r.id] = scores.get(r.id, 0.0) + rrf
            if r.id not in best_result:
                best_result[r.id] = r
            vec_ids.add(r.id)

        overlap_boost = float(self.config.rrf_overlap_boost)
        if overlap_boost > 1.0:
            for rid in (kw_ids & vec_ids):
                scores[rid] = scores.get(rid, 0.0) * overlap_boost

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
        return self._weights_for_profile(self.analyze_query(query))

    def _weights_for_profile(self, profile: QueryProfile) -> tuple[float, float]:
        kw = max(0.0001, float(self.config.keyword_weight))
        vec = max(0.0001, float(self.config.vector_weight))
        if not self.config.adaptive_weights:
            total = kw + vec
            return kw / total, vec / total

        intent = profile.intent
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

    @classmethod
    def analyze_query(cls, query: str) -> QueryProfile:
        q = query.strip().lower()
        tokens = tuple(re.findall(r"[a-z0-9_\-]+", q))
        token_set = set(tokens)
        wants_literal_recall = cls._wants_literal_recall(q, token_set)
        prefers_assistant_response = wants_literal_recall and cls._prefers_assistant_response(q, token_set)
        is_first_person = bool(token_set & _FIRST_PERSON_TERMS) or bool(
            re.search(r"\b(am|was|were)\s+i\b", q)
        )
        wants_current_state = is_first_person and (
            bool(token_set & _CURRENT_STATE_TERMS)
            or any(phrase in q for phrase in ("these days", "right now", "at the moment", "home base"))
        )
        wants_history = is_first_person and (
            bool(token_set & _HISTORY_TERMS)
            or any(
                phrase in q
                for phrase in (
                    "used to",
                    "left behind",
                    "moved from",
                    "move on from",
                    "moved on from",
                    "move from",
                    "old city",
                    "old bag",
                    "older city",
                    "from winter to spring",
                    "from spring to",
                    "what changed",
                    "what did i move",
                )
            )
        )
        has_multiple_facets = len(tokens) >= 5 and (
            " and " in q or "," in q or " plus " in q
        )
        fact_relation_hints = cls.infer_fact_relation_hints(query)
        return QueryProfile(
            intent=cls.classify_intent(query),
            tokens=tokens,
            is_first_person=is_first_person,
            wants_current_state=wants_current_state,
            wants_history=wants_history,
            has_multiple_facets=has_multiple_facets,
            fact_relation_hints=fact_relation_hints,
            wants_literal_recall=wants_literal_recall,
            prefers_assistant_response=prefers_assistant_response,
        )

    @classmethod
    def _wants_literal_recall(cls, q: str, token_set: set[str]) -> bool:
        has_recall_marker = any(phrase in q for phrase in _LITERAL_RECALL_PHRASES) or bool(
            re.search(r"\b(what was|what were|what did|which one|the one)\b", q)
        )
        if not has_recall_marker:
            return False
        assistant_or_chat_reference = (
            "you" in token_set
            or any(phrase in q for phrase in _ASSISTANT_REFERENCE_PHRASES)
            or "chat" in token_set
            or "conversation" in token_set
            or "discussed" in token_set
            or "talked" in token_set
        )
        has_list_or_specificity_marker = bool(
            re.search(r"\b\d+(?:st|nd|rd|th)\b", q)
        ) or bool(token_set & _LIST_RECALL_TERMS)
        return assistant_or_chat_reference or has_list_or_specificity_marker

    @staticmethod
    def _prefers_assistant_response(q: str, token_set: set[str]) -> bool:
        return (
            "you" in token_set
            or any(phrase in q for phrase in _ASSISTANT_REFERENCE_PHRASES)
            or "recommended" in token_set
            or "provided" in token_set
            or "suggested" in token_set
            or "said" in token_set
            or "told" in token_set
        )

    @classmethod
    def infer_fact_relation_hints(cls, query: str) -> tuple[str, ...]:
        q = query.strip().lower()
        tokens = set(re.findall(r"[a-z0-9_\-]+", q))
        hints: list[str] = []

        def add(relation: str) -> None:
            if relation not in hints:
                hints.append(relation)

        if (tokens & _LOCATION_HINT_TERMS) or any(phrase in q for phrase in ("home base", "based these days", "based now", "where am i")):
            add("location")
        if (tokens & _COFFEE_HINT_TERMS) or any(phrase in q for phrase in ("coffee order", "espresso order", "espresso drink", "drink order")):
            add("coffee_order")
        if (tokens & _BAG_HINT_TERMS) or any(phrase in q for phrase in ("everyday carry", "workday bag")):
            add("bag")
        if (tokens & _NOTEBOOK_HINT_TERMS) or any(phrase in q for phrase in ("field notes", "paper notebook")):
            add("notebook")
        if (tokens & _CHARGER_HINT_TERMS) or "power bank" in q:
            add("charger")
        if (tokens & _SEAT_HINT_TERMS) or any(phrase in q for phrase in ("aisle seat", "window seat")):
            add("flight_seat")
        if (tokens & _SNACK_HINT_TERMS) or any(phrase in q for phrase in ("movie night", "theater snack", "theatre snack")):
            add("movie_snack")
        if ("movie_snack" in hints or (tokens & _SNACK_HINT_TERMS)) and ((tokens & _DRINK_HINT_TERMS) or "sparkling water" in q):
            add("movie_drink")
        return tuple(hints)

    @staticmethod
    def classify_intent(query: str) -> str:
        q = query.strip().lower()
        temporal_terms = {
            "when", "before", "after", "timeline", "history", "previous",
            "earlier", "later", "recent", "latest", "last", "current", "currently", "now", "today",
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

    def _rerank_with_decay(
        self,
        results: list[SearchResult],
        top_k: int,
        profile: QueryProfile,
    ) -> list[SearchResult]:
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
            multiplier = self._score_multiplier(r, access_counts=access_counts, profile=profile)
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

    def _score_multiplier(
        self,
        result: SearchResult,
        access_counts: dict[str, int],
        profile: QueryProfile,
    ) -> float:
        metadata = result.metadata or {}
        multiplier = self._scope_multiplier(metadata, profile)
        created_at_raw = metadata.get("_created_at") or metadata.get("created_at")
        created_at = self._parse_datetime(created_at_raw)

        decay = 1.0
        if self.config.enable_decay and created_at is not None:
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

        return multiplier * decay * access_boost * importance

    def _scope_multiplier(self, metadata: dict[str, object], profile: QueryProfile) -> float:
        multiplier = 1.0
        source_kind = self._metadata_value(metadata, "_source_kind").strip().lower()
        role = self._metadata_value(metadata, "role").strip().lower()

        if profile.wants_literal_recall:
            if source_kind.startswith("structured_") or source_kind in {"graph", "graph_context"}:
                multiplier *= 0.18
            elif role == "assistant" and profile.prefers_assistant_response:
                multiplier *= 1.45
            elif role in {"assistant", "user"}:
                multiplier *= 1.12

        if profile.is_first_person:
            scope = self._infer_scope(metadata)
            if scope == "personal":
                multiplier *= float(self.config.personal_memory_boost)
            elif scope == "shared":
                multiplier *= float(self.config.shared_memory_penalty)
            entity = self._metadata_value(metadata, "entity").strip().lower()
            if entity == "user":
                multiplier *= 1.10
        relation = self._metadata_value(metadata, "relation").strip().lower()
        structured_kinds = {"structured_fact", "structured_truth", "structured_fact_current"}
        if profile.wants_current_state and source_kind in structured_kinds:
            if profile.fact_relation_hints:
                if relation in profile.fact_relation_hints:
                    multiplier *= float(self.config.current_fact_boost)
                else:
                    multiplier *= 0.78
            else:
                multiplier *= float(self.config.current_fact_boost)
        elif profile.wants_history and source_kind in structured_kinds and profile.fact_relation_hints:
            if relation in profile.fact_relation_hints:
                multiplier *= 1.10
            else:
                multiplier *= 0.82
        return multiplier

    @classmethod
    def _infer_scope(cls, metadata: dict[str, object]) -> str:
        direct_scope = cls._metadata_value(metadata, "memory_scope").strip().lower()
        if direct_scope in {"personal", "shared"}:
            return direct_scope
        haystack = " ".join(
            filter(
                None,
                [
                    cls._metadata_value(metadata, "path"),
                    cls._metadata_value(metadata, "source_id"),
                    cls._metadata_value(metadata, "session_id"),
                    cls._metadata_value(metadata, "entity"),
                    cls._metadata_value(metadata, "relation"),
                    cls._metadata_value(metadata, "role"),
                    cls._metadata_value(metadata.get("source"), "path"),
                    cls._metadata_value(metadata.get("source"), "source_id"),
                    cls._metadata_value(metadata.get("source"), "session_id"),
                    cls._metadata_value(metadata.get("source"), "memory_scope"),
                ],
            )
        ).lower()
        if any(term in haystack for term in _PERSONAL_SCOPE_TERMS):
            return "personal"
        if any(term in haystack for term in _SHARED_SCOPE_TERMS):
            return "shared"
        return ""

    @staticmethod
    def _metadata_value(raw: object, key: str) -> str:
        if not isinstance(raw, dict):
            return ""
        value = raw.get(key)
        return value if isinstance(value, str) else ""

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
