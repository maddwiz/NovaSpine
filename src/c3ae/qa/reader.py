"""Deterministic extractive reader over retrieved memory rows."""

from __future__ import annotations

import re
from typing import Any

from c3ae.qa.normalizer import clean_answer, infer_answer_type, normalize_answer, normalize_for_match
from c3ae.qa.types import MemoryAnswer
from c3ae.qa.verifier import verify_answer_support

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "to", "of", "in", "on",
    "for", "and", "or", "with", "what", "when", "who", "where", "which",
    "did", "do", "does", "at", "by", "from", "as", "it", "be", "this",
    "that", "how", "many", "much", "my", "me", "i",
}
_MONTH_RE = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)


def _row_id(row: Any) -> str:
    if isinstance(row, dict):
        return str(row.get("id", "") or row.get("chunk_id", "") or row.get("source_id", ""))
    return str(getattr(row, "id", ""))


def _row_content(row: Any) -> str:
    if isinstance(row, dict):
        return str(row.get("content", ""))
    return str(getattr(row, "content", ""))


def _row_metadata(row: Any) -> dict[str, Any]:
    if isinstance(row, dict):
        md = row.get("metadata") or {}
    else:
        md = getattr(row, "metadata", {}) or {}
    return md if isinstance(md, dict) else {}


def _tokens(text: str) -> set[str]:
    out: set[str] = set()
    for tok in re.findall(r"[a-z0-9_]+", normalize_for_match(text)):
        if len(tok) <= 1 or tok in _STOPWORDS:
            continue
        out.add(tok)
        if tok.endswith("s") and len(tok) > 3:
            out.add(tok[:-1])
    return out


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text or "") if s and s.strip()]


def _subject_terms(question: str) -> set[str]:
    q = normalize_for_match(question)
    important = _tokens(q)
    family_terms = {"dad", "father", "mother", "mom", "sister", "brother", "wife", "husband"}
    return important & family_terms


def _rank_sentences(question: str, rows: list[Any]) -> list[tuple[float, str, str, dict[str, Any]]]:
    q_tokens = _tokens(question)
    subjects = _subject_terms(question)
    q_norm = normalize_for_match(question)
    wants_current = any(term in q_norm.split() for term in ("current", "currently", "now", "present"))
    wants_history = "used to" in q_norm or any(term in q_norm.split() for term in ("previous", "previously", "former", "formerly", "old"))
    ranked: list[tuple[float, str, str, dict[str, Any]]] = []
    for index, row in enumerate(rows):
        content = _row_content(row)
        citation = _row_id(row)
        metadata = _row_metadata(row)
        for sentence in _split_sentences(content) or [content]:
            s_tokens = _tokens(sentence)
            if subjects and not (subjects & s_tokens):
                continue
            overlap = len(q_tokens & s_tokens) / max(1, len(q_tokens))
            score = overlap + max(0.0, 1.0 - (index * 0.02))
            status = str(metadata.get("entry_status", metadata.get("status", ""))).lower()
            if status == "active":
                score += 0.05
            s_norm = normalize_for_match(sentence)
            if wants_current:
                if "used to" in s_norm or status == "superseded":
                    score -= 0.45
                if "currently" in s_norm.split() or "now" in s_norm.split():
                    score += 0.30
            if wants_history and ("used to" in s_norm or status == "superseded"):
                score += 0.30
            ranked.append((score, citation, sentence, metadata))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked


def _extract_year_or_date(sentence: str, answer_type: str) -> str:
    if re.search(r"\b(last|this|next) year\b", sentence, re.IGNORECASE):
        return re.search(r"\b(?:last|this|next) year\b", sentence, re.IGNORECASE).group(0)  # type: ignore[union-attr]
    if answer_type == "year":
        m = re.search(r"\b(19|20)\d{2}\b", sentence)
        return m.group(0) if m else ""
    for pattern in (
        rf"\b{_MONTH_RE}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,\s*(?:19|20)\d{{2}})?\b",
        rf"\b{_MONTH_RE}\s+(?:19|20)\d{{2}}\b",
        r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b",
        r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b",
        r"\b(19|20)\d{2}\b",
    ):
        m = re.search(pattern, sentence, re.IGNORECASE)
        if m:
            return m.group(0)
    return ""


def _extract_count(question: str, ranked: list[tuple[float, str, str, dict[str, Any]]]) -> tuple[str, list[str], list[str], dict[str, Any]]:
    q = normalize_for_match(question)
    if "project" in q or "projects" in q:
        hits: list[tuple[str, str, dict[str, Any]]] = []
        seen: set[str] = set()
        for _, citation, sentence, metadata in ranked:
            if not re.search(r"\b(project|led|leading)\b", sentence, re.IGNORECASE):
                continue
            name = ""
            m = re.search(r"\bproject\s+([A-Z][A-Za-z0-9_-]{2,})\b", sentence, re.IGNORECASE)
            if not m:
                m = re.search(r"\b([A-Z][A-Za-z0-9_-]{2,})\b", sentence)
            if m:
                name = m.group(1).lower()
            key = name or normalize_for_match(sentence)
            if key in seen:
                continue
            seen.add(key)
            hits.append((citation, sentence, metadata))
        if hits:
            return str(len(hits)), [h[0] for h in hits], [h[1] for h in hits], hits[0][2]
    for _, citation, sentence, metadata in ranked:
        normalized = normalize_answer(sentence, "count", metadata)
        if re.fullmatch(r"\d+(?:\.\d+)?", normalized.answer):
            return normalized.answer, [citation], [sentence], metadata
    return "", [], [], {}


def _extract_preference(sentence: str) -> str:
    patterns = [
        r"\b(?:prefers?|likes?|favo(?:u)?rite\s+(?:\w+\s+)?(?:is|was)|order\s+(?:is|was)|uses?|carries?)\s+(?P<ans>[^.;!?]+)",
        r"\b(?:is|was)\s+(?:currently\s+)?using\s+(?P<ans>[^.;!?]+)",
        r"\b(?:coffee|drink|bag|notebook)\s+(?:order\s+)?(?:is|was)\s+(?P<ans>[^.;!?]+)",
    ]
    for pattern in patterns:
        m = re.search(pattern, sentence, re.IGNORECASE)
        if m:
            ans = clean_answer(m.group("ans"))
            ans = re.sub(r"\b(?:now|currently|these days)\b", "", ans, flags=re.IGNORECASE).strip()
            return clean_answer(ans)
    return ""


def _extract_free_text(question: str, sentence: str) -> str:
    q = normalize_for_match(question)
    if "research" in q:
        m = re.search(r"\bresearched\s+(?P<ans>[^.;!?]+?)(?:\s+before\b|$)", sentence, re.IGNORECASE)
        if m:
            return clean_answer(m.group("ans"))
    if "gift" in q:
        m = re.search(r"\b(?:gave|gifted|bought)\s+(?:me\s+)?(?P<ans>[^.;!?]+)", sentence, re.IGNORECASE)
        if m:
            return clean_answer(m.group("ans"))
    for pattern in (
        r"\b(?:is|was|were)\s+(?P<ans>[^.;!?]+)",
        r"\b(?:did|does|do)\s+(?P<ans>[^.;!?]+)",
    ):
        m = re.search(pattern, sentence, re.IGNORECASE)
        if m:
            return clean_answer(m.group("ans"))
    return ""


def _extract_candidate(
    question: str,
    answer_type: str,
    ranked: list[tuple[float, str, str, dict[str, Any]]],
) -> tuple[str, list[str], list[str], dict[str, Any]]:
    if answer_type in {"count", "number"}:
        return _extract_count(question, ranked)
    for _, citation, sentence, metadata in ranked[:24]:
        if answer_type in {"date", "year"}:
            raw = _extract_year_or_date(sentence, answer_type)
        elif answer_type == "relationship_status":
            m = re.search(r"\b(single|married|divorced|engaged|dating|separated|widowed)\b", sentence, re.IGNORECASE)
            raw = m.group(1) if m else ""
        elif answer_type == "location":
            m = re.search(r"\b(?:live|lives|living|based|moved)\s+(?:in|to|near|at)\s+(?P<ans>[A-Z][A-Za-z .'-]+)", sentence)
            raw = clean_answer(m.group("ans")) if m else ""
        elif answer_type == "person":
            m = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b", sentence)
            raw = m.group(1) if m else ""
        elif answer_type == "preference":
            raw = _extract_preference(sentence)
        elif answer_type == "yes_no":
            raw = "yes"
        else:
            raw = _extract_free_text(question, sentence)
        if raw:
            return raw, [citation], [sentence], metadata
    return "", [], [], {}


def answer_from_memory(question: str, rows: list[Any]) -> MemoryAnswer:
    answer_type = infer_answer_type(question)
    ranked = _rank_sentences(question, rows)
    if not ranked:
        return MemoryAnswer(
            answer="not enough information",
            answer_type="not_enough_information",
            confidence=0.0,
            abstain=True,
            verifier_status="abstained",
            metadata={"reason": "no_candidate_sentences"},
        )

    raw, citations, spans, metadata = _extract_candidate(question, answer_type, ranked)
    if not raw:
        return MemoryAnswer(
            answer="not enough information",
            answer_type="not_enough_information",
            confidence=0.0,
            abstain=True,
            verifier_status="abstained",
            metadata={"reason": "no_supported_candidate"},
        )

    normalized = normalize_answer(raw, answer_type, metadata)
    verification = verify_answer_support(
        normalized.answer,
        spans,
        normalized.answer_type,
        metadata=metadata,
    )
    if verification.status == "unsupported":
        return MemoryAnswer(
            answer="not enough information",
            answer_type="not_enough_information",
            confidence=0.0,
            citations=citations,
            supporting_spans=spans,
            abstain=True,
            normalization_steps=normalized.steps,
            verifier_status=verification.status,
            metadata={"reason": verification.reason},
        )

    confidence = verification.confidence
    if verification.status == "partial":
        confidence = min(confidence, 0.5)
    return MemoryAnswer(
        answer=normalized.answer,
        answer_type=normalized.answer_type,
        confidence=confidence,
        citations=citations,
        supporting_spans=spans,
        abstain=False,
        normalization_steps=normalized.steps,
        verifier_status=verification.status,
        metadata={"reason": verification.reason},
    )
