"""Multi-hop chained retrieval helpers."""

from __future__ import annotations

import hashlib
import re
from typing import Any


_ENTITY_STOPWORDS = {
    "What",
    "When",
    "Where",
    "Who",
    "Which",
    "How",
    "Why",
    "Did",
    "Does",
    "Do",
    "Is",
    "Are",
    "Was",
    "Were",
    "The",
    "A",
    "An",
    "And",
    "Or",
    "In",
    "On",
    "At",
    "For",
    "To",
    "From",
    "By",
    "Of",
}


def _row_key(row: dict[str, Any]) -> str:
    md = row.get("metadata") or {}
    benchmark_doc_id = str(md.get("benchmark_doc_id", "")).strip()
    if benchmark_doc_id:
        return f"doc:{benchmark_doc_id}"
    benchmark_source = str(md.get("benchmark_source", "")).strip()
    if benchmark_source:
        return f"source:{benchmark_source}"
    rid = str(row.get("id", "")).strip()
    if rid:
        return f"id:{rid}"
    content = re.sub(r"\s+", " ", str(row.get("content", "")).strip().lower())
    if content:
        digest = hashlib.sha1(content.encode("utf-8", errors="ignore")).hexdigest()
        return f"content:{digest}"
    return "empty"


def extract_entities(text: str, *, max_entities: int = 8) -> list[str]:
    if not text:
        return []
    found: list[str] = []
    seen: set[str] = set()
    for m in re.finditer(r"\b([A-Z][\w-]+(?:\s+[A-Z][\w-]+){0,3})\b", text):
        ent = m.group(1).strip()
        if not ent or ent in _ENTITY_STOPWORDS:
            continue
        if ent in seen:
            continue
        seen.add(ent)
        found.append(ent)
        if len(found) >= max(1, int(max_entities)):
            break
    return found


def build_expanded_query(
    query: str,
    rows: list[dict[str, Any]],
    *,
    entity_limit: int = 5,
    seed_rows: int = 3,
) -> str:
    if not rows:
        return query
    query_lower = query.lower()
    entities: list[str] = []
    seen: set[str] = set()
    for row in rows[: max(1, int(seed_rows))]:
        for ent in extract_entities(str(row.get("content", "")), max_entities=8):
            if ent.lower() in query_lower:
                continue
            if ent in seen:
                continue
            seen.add(ent)
            entities.append(ent)
            if len(entities) >= max(1, int(entity_limit)):
                break
        if len(entities) >= max(1, int(entity_limit)):
            break
    if not entities:
        return query
    return f"{query} {' '.join(entities)}"


def merge_rrf_rows(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
    *,
    top_k: int = 10,
    rrf_k: int = 30,
) -> list[dict[str, Any]]:
    if not rows_a and not rows_b:
        return []
    k = max(1, int(rrf_k))
    scores: dict[str, float] = {}
    best: dict[str, dict[str, Any]] = {}
    for rank, row in enumerate(rows_a):
        key = _row_key(row)
        scores[key] = scores.get(key, 0.0) + (1.0 / (k + rank + 1))
        if key not in best:
            best[key] = row
    for rank, row in enumerate(rows_b):
        key = _row_key(row)
        scores[key] = scores.get(key, 0.0) + (1.0 / (k + rank + 1))
        if key not in best:
            best[key] = row
    ranked = sorted(scores, key=lambda key: scores[key], reverse=True)
    out: list[dict[str, Any]] = []
    for key in ranked[: max(1, int(top_k))]:
        row = dict(best[key])
        row["score"] = float(scores[key])
        out.append(row)
    return out


async def chained_recall(
    *,
    spine: Any,
    query: str,
    top_k: int = 10,
    max_hops: int = 2,
    entity_limit: int = 5,
    fetch_k: int | None = None,
    session_filter: str | None = None,
) -> list[dict[str, Any]]:
    if max_hops <= 1:
        return await spine.recall(query, top_k=max(1, int(top_k)), session_filter=session_filter)
    fetch = max(1, int(fetch_k or max(top_k * 2, top_k)))
    hop_1 = await spine.recall(query, top_k=fetch, session_filter=session_filter)
    if not hop_1:
        return []
    expanded = build_expanded_query(query, hop_1, entity_limit=max(1, int(entity_limit)))
    if expanded.strip() == query.strip():
        return hop_1[: max(1, int(top_k))]
    hop_2 = await spine.recall(expanded, top_k=fetch, session_filter=session_filter)
    if not hop_2:
        return hop_1[: max(1, int(top_k))]
    return merge_rrf_rows(hop_1, hop_2, top_k=max(1, int(top_k)))

