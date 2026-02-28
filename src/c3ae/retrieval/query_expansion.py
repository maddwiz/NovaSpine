"""Lightweight keyword query expansion for QA-style prompts."""

from __future__ import annotations

import re

from c3ae.utils import extract_benchmark_case_token

_PHRASE_EXPANSIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("how many", ("number", "count", "total")),
    ("how much", ("amount", "total", "quantity")),
    ("what year", ("year", "date", "time")),
    ("what date", ("date", "year", "time")),
)

_TOKEN_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "who": ("person", "name"),
    "when": ("date", "year", "time"),
    "where": ("location", "place"),
    "owner": ("owns", "owned", "ownership"),
    "wrote": ("author", "written", "writer"),
    "founded": ("founder", "established"),
    "capital": ("capital_city", "city"),
}


def maybe_expand_query(query: str, max_extra_terms: int = 4) -> str:
    """Expand open-domain QA queries with compact lexical hints."""
    raw = (query or "").strip()
    if not raw:
        return raw
    if extract_benchmark_case_token(raw):
        return raw

    lowered = raw.lower()
    base_terms = set(re.findall(r"[a-z0-9_]+", lowered))
    extras: list[str] = []
    seen = set(base_terms)

    def _add(term: str) -> None:
        t = term.strip().lower()
        if not t or t in seen:
            return
        seen.add(t)
        extras.append(t)

    for phrase, terms in _PHRASE_EXPANSIONS:
        if phrase in lowered:
            for term in terms:
                _add(term)

    for tok in base_terms:
        for term in _TOKEN_EXPANSIONS.get(tok, ()):
            _add(term)

    if not extras:
        return raw
    return f"{raw} {' '.join(extras[: max(1, int(max_extra_terms))])}"
