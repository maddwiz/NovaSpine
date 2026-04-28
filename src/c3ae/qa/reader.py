"""Deterministic extractive reader over retrieved memory rows."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from c3ae.qa.normalizer import clean_answer, clean_cell, infer_answer_type, normalize_answer, normalize_for_match
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


def _question_answer_type(question: str) -> str:
    answer_type = infer_answer_type(question)
    q = normalize_for_match(question)
    q_tokens = _tokens(q)
    asks_plural_which = q.startswith("which ") and any(tok.endswith("s") and len(tok) > 3 for tok in q_tokens)
    if (
        answer_type == "free_text"
        and (
            q.startswith("what are ")
            or asks_plural_which
            or " list " in f" {q} "
            or " all " in f" {q} "
        )
    ):
        return "list"
    return answer_type


def _temporal_intent(question: str) -> tuple[bool, bool]:
    q_norm = normalize_for_match(question)
    q_parts = q_norm.split()
    wants_current = any(term in q_parts for term in ("current", "currently", "now", "present", "latest"))
    wants_history = "used to" in q_norm or any(
        term in q_parts for term in ("previous", "previously", "former", "formerly", "old", "past")
    )
    return wants_current, wants_history


def _temporal_score(question: str, text: str, metadata: dict[str, Any]) -> float:
    wants_current, wants_history = _temporal_intent(question)
    if not wants_current and not wants_history:
        return 0.0

    status = str(metadata.get("entry_status", metadata.get("status", ""))).lower()
    text_norm = normalize_for_match(text)
    score = 0.0
    historical = (
        "used to" in text_norm
        or any(term in text_norm.split() for term in ("previous", "previously", "former", "formerly", "old", "past"))
        or status == "superseded"
    )
    current = (
        any(term in text_norm.split() for term in ("current", "currently", "now", "present", "latest"))
        or status == "active"
    )
    if wants_current:
        if historical:
            score -= 0.65
        if current:
            score += 0.45
    if wants_history:
        if historical:
            score += 0.45
        if current and not historical:
            score -= 0.25
    return score


def _subject_terms(question: str) -> set[str]:
    q = normalize_for_match(question)
    important = _tokens(q)
    family_terms = {"dad", "father", "mother", "mom", "sister", "brother", "wife", "husband"}
    return important & family_terms


def _rank_sentences(question: str, rows: list[Any]) -> list[tuple[float, str, str, dict[str, Any]]]:
    q_tokens = _tokens(question)
    subjects = _subject_terms(question)
    q_norm = normalize_for_match(question)
    wants_current, wants_history = _temporal_intent(question)
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
            score += _temporal_score(question, sentence, metadata)
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
    m = re.search(r"\b(?:yesterday|today|tomorrow|last week|next week)\b", sentence, re.IGNORECASE)
    if m:
        return m.group(0)
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


def _split_table_cells(line: str) -> list[str]:
    return [clean_cell(cell) for cell in line.strip().strip("|").split("|")]


def _is_table_separator(line: str) -> bool:
    return bool(re.fullmatch(r"\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?", line.strip()))


def _table_candidates(question: str, row: Any) -> list[tuple[float, str, str, str, dict[str, Any]]]:
    content = _row_content(row)
    citation = _row_id(row)
    metadata = _row_metadata(row)
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    q_tokens = _tokens(question)
    q_norm = normalize_for_match(question)
    candidates: list[tuple[float, str, str, str, dict[str, Any]]] = []
    identity_headers = {"name", "person", "people", "user", "subject", "who"}

    for index in range(len(lines) - 1):
        if "|" not in lines[index] or not _is_table_separator(lines[index + 1]):
            continue
        headers = _split_table_cells(lines[index])
        if len(headers) < 2:
            continue
        for data_line in lines[index + 2 :]:
            if "|" not in data_line or _is_table_separator(data_line):
                break
            cells = _split_table_cells(data_line)
            if len(cells) != len(headers):
                continue
            row_span = "; ".join(
                f"{header}: {cell}" for header, cell in zip(headers, cells, strict=True) if header and cell
            )
            row_tokens = _tokens(row_span)
            if q_tokens and not (q_tokens & row_tokens):
                continue
            for header, cell in zip(headers, cells, strict=True):
                if not cell:
                    continue
                header_tokens = _tokens(header)
                if header_tokens & identity_headers:
                    continue
                header_norm = normalize_for_match(header)
                score = len(q_tokens & header_tokens) + (0.15 * len(q_tokens & row_tokens))
                if "prefer" in q_norm and any(tok in header_norm for tok in ("prefer", "favorite", "favourite")):
                    score += 1.0
                if any(tok in q_norm for tok in ("drink", "order", "coffee")) and any(
                    tok in header_norm for tok in ("drink", "order", "coffee")
                ):
                    score += 1.0
                score += _temporal_score(question, f"{header}: {cell}", metadata)
                if score > 0:
                    candidates.append((score, cell, citation, row_span, metadata))
    return candidates


def _labeled_span_candidates(question: str, row: Any) -> list[tuple[float, str, str, str, dict[str, Any]]]:
    content = _row_content(row)
    citation = _row_id(row)
    metadata = _row_metadata(row)
    q_tokens = _tokens(question)
    candidates: list[tuple[float, str, str, str, dict[str, Any]]] = []
    for span in _split_sentences(content) or [content]:
        line = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s+", "", span).strip()
        if "|" in line and _is_table_separator(line):
            continue
        m = re.match(r"(?P<label>[A-Za-z][A-Za-z0-9 _/'().-]{1,80})\s*(?::|=|--?|->)\s*(?P<value>.+)$", line)
        if not m:
            continue
        label = clean_answer(m.group("label"))
        value = clean_answer(m.group("value"))
        if not value:
            continue
        label_tokens = _tokens(label)
        score = len(q_tokens & label_tokens) + _temporal_score(question, label, metadata)
        if score > 0:
            candidates.append((score, value, citation, f"{label}: {value}", metadata))
    return candidates


def _extract_list_from_span(question: str, answer_type: str, row: Any) -> tuple[str, str, str, dict[str, Any]] | None:
    if answer_type != "list":
        return None
    candidates = _labeled_span_candidates(question, row)
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, value, citation, span, metadata = candidates[0]
    return value, citation, span, metadata


def _extract_structured_candidate(
    question: str,
    answer_type: str,
    rows: list[Any],
) -> tuple[str, list[str], list[str], dict[str, Any]]:
    candidates: list[tuple[float, str, str, str, dict[str, Any]]] = []
    for index, row in enumerate(rows):
        row_bias = max(0.0, 1.0 - (index * 0.02))
        for candidate in _table_candidates(question, row):
            score, raw, citation, span, metadata = candidate
            candidates.append((score + row_bias, raw, citation, span, metadata))
        listed = _extract_list_from_span(question, answer_type, row)
        if listed is not None:
            raw, citation, span, metadata = listed
            candidates.append((row_bias + 1.0 + _temporal_score(question, span, metadata), raw, citation, span, metadata))
        for score, raw, citation, span, metadata in _labeled_span_candidates(question, row):
            candidates.append((score + row_bias, raw, citation, span, metadata))
    if not candidates:
        return "", [], [], {}
    candidates.sort(key=lambda item: item[0], reverse=True)
    if answer_type == "list":
        values: list[str] = []
        citations: list[str] = []
        spans: list[str] = []
        seen_values: set[str] = set()
        for score, raw, citation, span, metadata in candidates[:12]:
            if score <= 0:
                continue
            normalized = normalize_answer(raw, "list", metadata).answer
            parts = [clean_answer(part) for part in normalized.split(",") if clean_answer(part)]
            for part in parts or [clean_answer(raw)]:
                key = normalize_for_match(part)
                if not key or key in seen_values:
                    continue
                seen_values.add(key)
                values.append(part)
            if citation not in citations:
                citations.append(citation)
            if span not in spans:
                spans.append(span)
            if len(values) >= 10:
                break
        if values:
            return ", ".join(values), citations, spans, candidates[0][4]
    _, raw, citation, span, metadata = candidates[0]
    return raw, [citation], [span], metadata


def _parse_normalized_date(value: str) -> datetime | None:
    value = clean_answer(value)
    for fmt in ("%d %B %Y", "%B %d %Y", "%B %d, %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _date_values_from_text(text: str, metadata: dict[str, Any]) -> list[datetime]:
    values: list[datetime] = []
    patterns = [
        r"\b(?:yesterday|today|tomorrow|last week|next week)\b",
        rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH_RE}\s*(?:19|20)\d{{2}}\b",
        rf"\b{_MONTH_RE}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,\s*(?:19|20)\d{{2}})?\b",
        r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b",
    ]
    seen: set[str] = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            raw = match.group(0)
            normalized = normalize_answer(raw, "date", metadata)
            parsed = _parse_normalized_date(normalized.answer)
            if parsed is None:
                continue
            key = parsed.strftime("%Y-%m-%d")
            if key in seen:
                continue
            seen.add(key)
            values.append(parsed)
    return values


def _extract_temporal_delta(
    question: str,
    ranked: list[tuple[float, str, str, dict[str, Any]]],
) -> tuple[str, list[str], list[str], dict[str, Any]]:
    q_norm = normalize_for_match(question)
    if "how many days" not in q_norm and "how long" not in q_norm:
        return "", [], [], {}

    candidates: list[tuple[datetime, str, str, dict[str, Any]]] = []
    for _, citation, sentence, metadata in ranked[:24]:
        for value in _date_values_from_text(sentence, metadata):
            candidates.append((value, citation, sentence, metadata))
    question_dates = _date_values_from_text(question, {})
    for value in question_dates:
        candidates.append((value, "", question, {}))
    if len(candidates) < 2:
        return "", [], [], {}

    candidates.sort(key=lambda item: item[0])
    days = abs((candidates[-1][0] - candidates[0][0]).days)
    citations = [citation for _, citation, _, _ in candidates if citation]
    spans = [span for _, _, span, _ in candidates if span != question]
    metadata = dict(candidates[0][3] or {})
    metadata["computed_temporal_delta_days"] = days
    return str(days), citations, spans, metadata


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
        temporal_delta = _extract_temporal_delta(question, ranked)
        if temporal_delta[0]:
            return temporal_delta
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
        elif answer_type == "list":
            raw = _extract_free_text(question, sentence)
        elif answer_type == "yes_no":
            raw = "yes"
        else:
            raw = _extract_free_text(question, sentence)
        if raw:
            return raw, [citation], [sentence], metadata
    return "", [], [], {}


def answer_from_memory(question: str, rows: list[Any]) -> MemoryAnswer:
    from c3ae.retrieval.planner import plan_memory_query

    query_plan = plan_memory_query(question)
    answer_type = _question_answer_type(question)
    if query_plan.route == "list_lookup" and answer_type == "free_text":
        answer_type = "list"
    ranked = _rank_sentences(question, rows)
    if not ranked:
        return MemoryAnswer(
            answer="not enough information",
            answer_type="not_enough_information",
            confidence=0.0,
            abstain=True,
            verifier_status="abstained",
            metadata={"reason": "no_candidate_sentences", "query_plan": query_plan.to_dict()},
        )

    raw, citations, spans, metadata = _extract_structured_candidate(question, answer_type, rows)
    if not raw:
        raw, citations, spans, metadata = _extract_candidate(question, answer_type, ranked)
    if not raw:
        return MemoryAnswer(
            answer="not enough information",
            answer_type="not_enough_information",
            confidence=0.0,
            abstain=True,
            verifier_status="abstained",
            metadata={"reason": "no_supported_candidate", "query_plan": query_plan.to_dict()},
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
            metadata={"reason": verification.reason, "query_plan": query_plan.to_dict()},
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
        metadata={"reason": verification.reason, "query_plan": query_plan.to_dict()},
    )
