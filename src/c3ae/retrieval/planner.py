"""Lightweight query planning for memory retrieval.

The planner is intentionally deterministic and side-effect free.  It does not
change ranking by itself; callers can attach the plan to benchmark rows or use
it later to route retrieval through table, list, graph, or temporal readers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re

from c3ae.qa.normalizer import infer_answer_type, normalize_for_match


PLAN_ROUTES = {
    "direct_recall",
    "table_lookup",
    "list_lookup",
    "current_state",
    "historical_state",
    "temporal_math",
    "multi_session",
}


@dataclass(frozen=True)
class QueryPlan:
    route: str
    answer_type: str
    intents: list[str] = field(default_factory=list)
    strategies: list[str] = field(default_factory=list)
    confidence: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "route": self.route,
            "answer_type": self.answer_type,
            "intents": list(self.intents),
            "strategies": list(self.strategies),
            "confidence": round(float(self.confidence), 3),
            "notes": list(self.notes),
        }


def _has_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def plan_memory_query(question: str) -> QueryPlan:
    """Classify a memory question into a first-pass retrieval route.

    This is not a learned policy.  It captures the stable routing distinctions
    surfaced by LongMemEval/LoCoMo style failures: table/list extraction,
    temporal arithmetic, current-vs-historical state, and cross-session order.
    """

    q = normalize_for_match(question)
    answer_type = infer_answer_type(question)
    intents: list[str] = []
    strategies: list[str] = ["hybrid_recall", "exact_keyword"]
    notes: list[str] = []
    route = "direct_recall"
    confidence = 0.35

    table_terms = (
        "table",
        "column",
        "row",
        "schedule",
        "shift",
        "spreadsheet",
        "csv",
        "markdown",
    )
    if _has_any(q, table_terms):
        route = "table_lookup"
        intents.append("structured_table")
        strategies.extend(["markdown_table_reader", "row_column_match"])
        confidence = 0.78

    list_terms = ("list", "all", "which", "items", "options", "tasks", "projects")
    if route == "direct_recall" and _has_any(f" {q} ", tuple(f" {t} " for t in list_terms)):
        route = "list_lookup"
        intents.append("enumeration")
        strategies.extend(["list_reader", "dedupe_entities"])
        confidence = 0.62

    temporal_terms = (
        "when",
        "date",
        "year",
        "before",
        "after",
        "between",
        "earlier",
        "later",
        "last",
        "first",
        "how many days",
        "how long",
    )
    if _has_any(q, temporal_terms):
        intents.append("temporal")
        strategies.extend(["temporal_fact_lookup", "date_normalization"])
        if route == "direct_recall" or _has_any(q, ("between", "how many days", "how long")):
            route = "temporal_math"
            confidence = max(confidence, 0.70)

    if _has_any(q, ("current", "currently", "now", "latest", "present")):
        intents.append("current_state")
        strategies.extend(["active_memory_preference", "supersession_filter"])
        if route == "direct_recall":
            route = "current_state"
            confidence = max(confidence, 0.72)

    if _has_any(q, ("used to", "previous", "previously", "former", "formerly", "old", "past")):
        intents.append("historical_state")
        strategies.extend(["superseded_memory_allowed", "timeline_lookup"])
        if route == "direct_recall" or route == "current_state":
            route = "historical_state"
            confidence = max(confidence, 0.72)

    session_patterns = (
        r"\bsession\b",
        r"\bconversation\b",
        r"\bchat\b",
        r"\bround\b",
        r"\bturn\b",
        r"\bsaid earlier\b",
        r"\bmentioned before\b",
    )
    if any(re.search(pattern, q) for pattern in session_patterns):
        intents.append("cross_session")
        strategies.extend(["episode_retrieval", "neighbor_context_expansion"])
        if route == "direct_recall":
            route = "multi_session"
            confidence = max(confidence, 0.66)

    if answer_type in {"count", "number"}:
        strategies.append("numeric_solver")
    elif answer_type in {"date", "year"}:
        strategies.append("temporal_span_reader")
    elif answer_type == "list":
        strategies.append("list_reader")

    if route not in PLAN_ROUTES:
        notes.append(f"unknown_route={route}")
        route = "direct_recall"
        confidence = min(confidence, 0.35)

    return QueryPlan(
        route=route,
        answer_type=answer_type,
        intents=_dedupe(intents),
        strategies=_dedupe(strategies),
        confidence=confidence,
        notes=notes,
    )

