"""Structured fact extraction from raw memory text."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from c3ae.utils import parse_json_object


_MONTH_PATTERN = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
_DATE_PATTERNS = (
    rf"\b{_MONTH_PATTERN}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,\s*\d{{4}})?\b",
    r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b",
    r"\b\d{4}-\d{2}-\d{2}\b",
    r"\b(19|20)\d{2}\b",
)
_RELATION_PATTERNS: list[tuple[str, str, float]] = [
    (r"(?P<entity>[A-Z][\w\-]+(?:\s+[A-Z][\w\-]+){0,2})\s+moved from\s+(?P<value>[^.,;\n]{2,80})", "moved_from", 0.86),
    (r"(?P<entity>[A-Z][\w\-]+(?:\s+[A-Z][\w\-]+){0,2})\s+moved to\s+(?P<value>[^.,;\n]{2,80})", "moved_to", 0.86),
    (r"(?P<entity>[A-Z][\w\-]+(?:\s+[A-Z][\w\-]+){0,2})\s+painted\s+(?P<value>[^.,;\n]{2,80})", "painted", 0.84),
    (r"(?P<entity>[A-Z][\w\-]+(?:\s+[A-Z][\w\-]+){0,2})\s+bought\s+(?P<value>[^.,;\n]{2,80})", "bought", 0.82),
    (r"(?P<entity>[A-Z][\w\-]+(?:\s+[A-Z][\w\-]+){0,2})\s+works at\s+(?P<value>[^.,;\n]{2,80})", "works_at", 0.82),
    (r"(?P<entity>[A-Z][\w\-]+(?:\s+[A-Z][\w\-]+){0,2})\s+has\s+(?P<value>\d+\s+[A-Za-z][^.,;\n]{0,40})", "has", 0.78),
]


@dataclass(frozen=True)
class StructuredFact:
    entity: str
    relation: str
    value: str
    date: str = ""
    confidence: float = 0.0
    source_span: str = ""


def _split_sentences(text: str) -> list[str]:
    lines = [ln.strip() for ln in re.split(r"[\n\r]+", text or "") if ln.strip()]
    out: list[str] = []
    for ln in lines:
        out.extend([s.strip() for s in re.split(r"(?<=[\.\!\?])\s+", ln) if s.strip()])
    return out


def _extract_date_hint(sentence: str) -> str:
    for pat in _DATE_PATTERNS:
        m = re.search(pat, sentence, flags=re.IGNORECASE)
        if m:
            return m.group(0).strip()
    return ""


def _normalize_relation(relation: str) -> str:
    rel = (relation or "").strip().lower()
    rel = re.sub(r"[^a-z0-9_ ]+", "", rel)
    rel = re.sub(r"\s+", "_", rel)
    return rel


def _clean_value(value: str) -> str:
    out = re.sub(r"\s+", " ", value or "").strip(" \t\r\n,;:.")
    # Remove leading determiners to keep facts canonical.
    out = re.sub(r"^(?:the|a|an)\s+", "", out, flags=re.IGNORECASE).strip()
    return out


def extract_facts(text: str, *, max_facts: int = 10) -> list[StructuredFact]:
    """High-precision heuristic structured fact extraction."""
    if not text.strip():
        return []
    out: list[StructuredFact] = []
    seen: set[tuple[str, str, str, str]] = set()
    for sentence in _split_sentences(text):
        sentence_date = _extract_date_hint(sentence)
        for pattern, relation, base_conf in _RELATION_PATTERNS:
            m = re.search(pattern, sentence, flags=re.IGNORECASE)
            if not m:
                continue
            entity = re.sub(r"\s+", " ", m.group("entity")).strip()
            value = _clean_value(m.group("value"))
            if not entity or not value:
                continue
            key = (entity.lower(), relation, value.lower(), sentence_date.lower())
            if key in seen:
                continue
            seen.add(key)
            out.append(
                StructuredFact(
                    entity=entity,
                    relation=_normalize_relation(relation),
                    value=value,
                    date=sentence_date,
                    confidence=float(base_conf),
                    source_span=sentence.strip()[:200],
                )
            )
            if len(out) >= max(1, int(max_facts)):
                return out
    return out


async def extract_facts_async(
    text: str,
    *,
    mode: str = "heuristic",
    chat_backend: Any | None = None,
    max_facts: int = 10,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> list[StructuredFact]:
    """Extract structured facts with optional LLM mode and heuristic fallback."""
    selected_mode = (mode or "heuristic").strip().lower()
    if selected_mode != "llm" or chat_backend is None:
        return extract_facts(text, max_facts=max_facts)
    try:
        from c3ae.llm import Message

        prompt = (
            "Extract high-precision structured memory facts from text.\n"
            "Return strict JSON with key 'facts' as a list of objects:\n"
            "{\"facts\":[{\"entity\":\"...\",\"relation\":\"...\",\"value\":\"...\",\"date\":\"...\",\"confidence\":0.0}]}\n"
            f"- Keep relation in snake_case.\n"
            f"- At most {max(1, int(max_facts))} facts.\n"
            "Text:\n"
            f"{text}"
        )
        resp = await chat_backend.chat(
            [
                Message(role="system", content="You extract concise structured facts."),
                Message(role="user", content=prompt),
            ],
            temperature=float(temperature),
            max_tokens=max(64, int(max_tokens)),
            json_mode=True,
        )
        payload = parse_json_object(resp.content)
        rows = payload.get("facts", []) if isinstance(payload, dict) else []
        out: list[StructuredFact] = []
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                entity = re.sub(r"\s+", " ", str(row.get("entity", "")).strip())
                relation = _normalize_relation(str(row.get("relation", "")))
                value = _clean_value(str(row.get("value", "")))
                fact_date = str(row.get("date", "")).strip()
                conf_raw = row.get("confidence", 0.75)
                try:
                    conf = float(conf_raw)
                except Exception:
                    conf = 0.75
                if not entity or not relation or not value:
                    continue
                out.append(
                    StructuredFact(
                        entity=entity,
                        relation=relation,
                        value=value,
                        date=fact_date,
                        confidence=max(0.05, min(0.99, conf)),
                        source_span="",
                    )
                )
                if len(out) >= max(1, int(max_facts)):
                    break
        if out:
            return out
    except Exception:
        pass
    return extract_facts(text, max_facts=max_facts)

