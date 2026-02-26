"""Lightweight entity/relation extraction for memory graph indexing."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "for", "of", "in", "on", "at", "by",
    "with", "from", "into", "that", "this", "is", "are", "was", "were", "be",
    "it", "as", "we", "you", "they", "he", "she", "i",
}

_REL_PATTERNS: list[tuple[str, str]] = [
    (r"\b(prefers|prefer)\b", "prefers"),
    (r"\b(likes|like)\b", "likes"),
    (r"\b(uses|use|used)\b", "uses"),
    (r"\b(depends on|depends)\b", "depends_on"),
    (r"\b(works on|working on|builds|built)\b", "works_on"),
    (r"\b(is|are|was|were)\b", "is"),
    (r"\b(owns|owned)\b", "owns"),
]


@dataclass
class ExtractedGraph:
    entities: list[str] = field(default_factory=list)
    relations: list[tuple[str, str, str]] = field(default_factory=list)  # src, relation, dst


def extract_graph_facts(
    text: str,
    max_entities: int = 16,
    max_relations: int = 8,
) -> ExtractedGraph:
    """Extract entities and simple relations from text using deterministic heuristics."""
    lines = [ln.strip() for ln in re.split(r"[\n\r]+", text) if ln.strip()]
    sentences: list[str] = []
    for ln in lines:
        sentences.extend([s.strip() for s in re.split(r"(?<=[\.\!\?])\s+", ln) if s.strip()])
    if not sentences:
        sentences = [text.strip()] if text.strip() else []

    entities: list[str] = []
    relations: list[tuple[str, str, str]] = []

    for sentence in sentences:
        ents = _extract_entities(sentence)
        for ent in ents:
            if ent not in entities:
                entities.append(ent)
            if len(entities) >= max_entities:
                break

        for pattern, rel in _REL_PATTERNS:
            m = re.search(pattern, sentence, flags=re.IGNORECASE)
            if not m:
                continue
            left = sentence[: m.start()].strip()
            right = sentence[m.end() :].strip()
            src = _extract_anchor(left, from_left=True)
            dst = _extract_anchor(right, from_left=False)
            if src and dst and src.lower() != dst.lower():
                tup = (src, rel, dst)
                if tup not in relations:
                    relations.append(tup)
            if len(relations) >= max_relations:
                break
        if len(relations) >= max_relations and len(entities) >= max_entities:
            break

    return ExtractedGraph(
        entities=entities[:max_entities],
        relations=relations[:max_relations],
    )


def _extract_entities(sentence: str) -> list[str]:
    out: list[str] = []

    # Title-case or uppercase terms, including multi-word entities.
    for m in re.finditer(r"\b(?:[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+){0,3}|[A-Z]{2,6}\d*)\b", sentence):
        val = m.group(0).strip()
        if val.lower() in _STOPWORDS:
            continue
        if val not in out:
            out.append(val)

    # Add key noun-like lowercase tokens for personalization memories.
    tokens = re.findall(r"[a-z0-9_\-]{3,}", sentence.lower())
    for tok in tokens:
        if tok in _STOPWORDS:
            continue
        if tok.isdigit():
            continue
        if tok not in out:
            out.append(tok)
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
    # Prefer the most specific token unless this looks like a name phrase.
    if len(window) >= 2 and window[0][0].isupper() and window[1][0].isupper():
        return " ".join(window[:2])
    return window[-1] if from_left else window[0]
