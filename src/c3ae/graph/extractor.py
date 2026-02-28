"""Entity/relation extraction for memory graph indexing."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from c3ae.utils import parse_json_object

_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "for", "of", "in", "on", "at", "by",
    "with", "from", "into", "that", "this", "is", "are", "was", "were", "be",
    "it", "as", "we", "you", "they", "he", "she", "i",
}

_REL_PATTERNS: list[tuple[str, str, float]] = [
    (r"\b(prefers|prefer)\b", "prefers", 0.70),
    (r"\b(likes|like)\b", "likes", 0.68),
    (r"\b(uses|use|used)\b", "uses", 0.72),
    (r"\b(depends on|depends)\b", "depends_on", 0.73),
    (r"\b(works on|working on|builds|built)\b", "works_on", 0.69),
    (r"\b(is|are|was|were)\b", "is", 0.58),
    (r"\b(owns|owned)\b", "owns", 0.76),
]


@dataclass
class ExtractedGraph:
    entities: list[str] = field(default_factory=list)
    relations: list[tuple[str, str, str]] = field(default_factory=list)  # src, relation, dst
    entity_confidence: dict[str, float] = field(default_factory=dict)
    relation_confidence: dict[tuple[str, str, str], float] = field(default_factory=dict)
    mode: str = "heuristic"


def extract_graph_facts(
    text: str,
    max_entities: int = 16,
    max_relations: int = 8,
) -> ExtractedGraph:
    """Extract entities and relations with deterministic heuristics."""
    sentences = _split_sentences(text)
    entities: list[str] = []
    relations: list[tuple[str, str, str]] = []
    entity_conf: dict[str, float] = {}
    relation_conf: dict[tuple[str, str, str], float] = {}

    for sentence in sentences:
        ents = _extract_entities(sentence)
        for ent, conf in ents:
            if ent not in entities:
                entities.append(ent)
            entity_conf[ent] = max(entity_conf.get(ent, 0.0), conf)
            if len(entities) >= max_entities:
                break

        for pattern, rel, base_conf in _REL_PATTERNS:
            m = re.search(pattern, sentence, flags=re.IGNORECASE)
            if not m:
                continue
            left = sentence[: m.start()].strip()
            right = sentence[m.end() :].strip()
            src = _extract_anchor(left, from_left=True)
            dst = _extract_anchor(right, from_left=False)
            if not src or not dst or src.lower() == dst.lower():
                continue
            tup = (src, rel, dst)
            if tup not in relations:
                relations.append(tup)
            calibrated = base_conf
            calibrated += 0.08 if src in entity_conf else 0.0
            calibrated += 0.08 if dst in entity_conf else 0.0
            relation_conf[tup] = max(
                relation_conf.get(tup, 0.0),
                _clamp_confidence(calibrated),
            )
            if len(relations) >= max_relations:
                break
        if len(relations) >= max_relations and len(entities) >= max_entities:
            break

    return ExtractedGraph(
        entities=entities[:max_entities],
        relations=relations[:max_relations],
        entity_confidence=entity_conf,
        relation_confidence=relation_conf,
        mode="heuristic",
    )


async def extract_graph_facts_async(
    text: str,
    *,
    extraction_mode: str = "heuristic",
    chat_backend: Any | None = None,
    max_entities: int = 16,
    max_relations: int = 8,
    temperature: float = 0.0,
    max_tokens: int = 800,
) -> ExtractedGraph:
    """Extract graph facts using LLM mode with safe heuristic fallback."""
    mode = (extraction_mode or "heuristic").strip().lower()
    if mode != "llm" or chat_backend is None:
        return extract_graph_facts(text, max_entities=max_entities, max_relations=max_relations)

    try:
        extracted = await _extract_graph_facts_llm(
            text,
            chat_backend=chat_backend,
            max_entities=max_entities,
            max_relations=max_relations,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if extracted.entities or extracted.relations:
            return extracted
    except Exception:
        pass
    return extract_graph_facts(text, max_entities=max_entities, max_relations=max_relations)


async def _extract_graph_facts_llm(
    text: str,
    *,
    chat_backend: Any,
    max_entities: int,
    max_relations: int,
    temperature: float,
    max_tokens: int,
) -> ExtractedGraph:
    from c3ae.llm.venice_chat import Message

    prompt = (
        "Extract a temporal memory graph from the text.\n"
        "Return strict JSON only with keys: entities, relations.\n"
        "Schema:\n"
        "{"
        "\"entities\":[{\"name\":\"...\",\"confidence\":0.0-1.0}],"
        "\"relations\":[{\"src\":\"...\",\"relation\":\"...\",\"dst\":\"...\",\"confidence\":0.0-1.0}]"
        "}\n"
        "Rules:\n"
        "- Keep entity names concise.\n"
        "- relation must be snake_case and semantic (e.g. prefers, uses, depends_on, works_on, owns).\n"
        f"- Maximum {max_entities} entities and {max_relations} relations.\n"
        f"Text:\n{text}"
    )
    response = await chat_backend.chat(
        [
            Message(role="system", content="You extract high-precision entities and relations."),
            Message(role="user", content=prompt),
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        json_mode=True,
    )
    payload = parse_json_object(response.content)
    entity_rows = payload.get("entities", [])
    relation_rows = payload.get("relations", [])

    entities: list[str] = []
    relations: list[tuple[str, str, str]] = []
    entity_conf: dict[str, float] = {}
    relation_conf: dict[tuple[str, str, str], float] = {}

    if isinstance(entity_rows, list):
        for row in entity_rows:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name", "")).strip()
            if not name:
                continue
            if name not in entities:
                entities.append(name)
            conf = _clamp_confidence(_coerce_float(row.get("confidence"), 0.78))
            entity_conf[name] = max(entity_conf.get(name, 0.0), conf)
            if len(entities) >= max_entities:
                break

    if isinstance(relation_rows, list):
        for row in relation_rows:
            if not isinstance(row, dict):
                continue
            src = str(row.get("src", "")).strip()
            rel = _normalize_relation(str(row.get("relation", "")).strip())
            dst = str(row.get("dst", "")).strip()
            if not src or not rel or not dst:
                continue
            if src.lower() == dst.lower():
                continue
            tup = (src, rel, dst)
            if tup not in relations:
                relations.append(tup)
            conf = _clamp_confidence(_coerce_float(row.get("confidence"), 0.74))
            relation_conf[tup] = max(relation_conf.get(tup, 0.0), conf)
            if len(relations) >= max_relations:
                break

    return ExtractedGraph(
        entities=entities[:max_entities],
        relations=relations[:max_relations],
        entity_confidence=entity_conf,
        relation_confidence=relation_conf,
        mode="llm",
    )


def _split_sentences(text: str) -> list[str]:
    lines = [ln.strip() for ln in re.split(r"[\n\r]+", text) if ln.strip()]
    sentences: list[str] = []
    for ln in lines:
        sentences.extend([s.strip() for s in re.split(r"(?<=[\.\!\?])\s+", ln) if s.strip()])
    if not sentences:
        sentences = [text.strip()] if text.strip() else []
    return sentences


def _extract_entities(sentence: str) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    seen: set[str] = set()

    for m in re.finditer(r"\b(?:[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+){0,3}|[A-Z]{2,6}\d*)\b", sentence):
        val = m.group(0).strip()
        if val.lower() in _STOPWORDS:
            continue
        if val not in seen:
            seen.add(val)
            conf = 0.82 if val.isupper() else 0.74
            out.append((val, conf))

    tokens = re.findall(r"[a-z0-9_\-]{4,}", sentence.lower())
    for tok in tokens:
        if tok in _STOPWORDS or tok.isdigit():
            continue
        if tok not in seen:
            seen.add(tok)
            out.append((tok, 0.42))
    return out


def _extract_anchor(text: str, from_left: bool) -> str:
    tokens = re.findall(r"[A-Za-z0-9_\-]{2,}", text)
    if not tokens:
        return ""
    tokens = [t for t in tokens if t.lower() not in _STOPWORDS]
    if not tokens:
        return ""
    if from_left:
        window = tokens[-3:]
    else:
        window = tokens[:3]
    if len(window) >= 2 and window[0][0].isupper() and window[1][0].isupper():
        return " ".join(window[:2])
    return window[-1] if from_left else window[0]


def _normalize_relation(relation: str) -> str:
    if not relation:
        return ""
    rel = relation.strip().lower()
    rel = re.sub(r"[^a-z0-9_ ]+", "", rel)
    rel = re.sub(r"\s+", "_", rel)
    return rel


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _clamp_confidence(value: float) -> float:
    return max(0.05, min(0.99, float(value)))

