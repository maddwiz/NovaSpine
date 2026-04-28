"""Cheap support checks for extractive memory answers."""

from __future__ import annotations

import re
from typing import Any

from c3ae.qa.normalizer import normalize_answer, normalize_for_match
from c3ae.qa.types import VerificationResult


def verify_answer_support(
    answer: str,
    supporting_spans: list[str],
    answer_type: str,
    metadata: dict[str, Any] | None = None,
) -> VerificationResult:
    if not answer or normalize_for_match(answer) == "not enough information":
        return VerificationResult(status="abstained", confidence=1.0, reason="abstained")
    if not supporting_spans:
        return VerificationResult(status="unsupported", confidence=0.0, reason="no_supporting_spans")

    norm_answer = normalize_answer(answer, answer_type, metadata).answer
    answer_norm = normalize_for_match(norm_answer)
    joined = " ".join(supporting_spans)
    joined_norm = normalize_for_match(joined)
    if not answer_norm or not joined_norm:
        return VerificationResult(status="unsupported", confidence=0.0, reason="empty_answer_or_context")

    if answer_norm in joined_norm:
        return VerificationResult(status="supported", confidence=0.95, reason="answer_text_present")

    if answer_type in {"date", "year"}:
        normalized_from_span = normalize_answer(joined, answer_type, metadata)
        if normalize_for_match(normalized_from_span.answer) == answer_norm and normalized_from_span.steps:
            return VerificationResult(status="supported", confidence=0.86, reason="relative_date_supported_by_metadata")

    if answer_type in {"count", "number"}:
        try:
            count = int(float(answer_norm))
        except ValueError:
            count = -1
        if count > 0 and len([s for s in supporting_spans if s.strip()]) >= count:
            return VerificationResult(status="supported", confidence=0.72, reason="count_supported_by_span_count")

    answer_tokens = set(answer_norm.split())
    span_tokens = set(joined_norm.split())
    if answer_tokens:
        overlap = len(answer_tokens & span_tokens) / len(answer_tokens)
        if overlap >= 0.75:
            return VerificationResult(status="supported", confidence=0.78, reason="token_overlap")
        if overlap >= 0.45:
            return VerificationResult(status="partial", confidence=0.48, reason="partial_token_overlap")

    if answer_type == "yes_no":
        if answer_norm == "yes" and re.search(r"\b(is|are|was|were|does|did|has|have)\b", joined_norm):
            return VerificationResult(status="partial", confidence=0.45, reason="affirmative_relation_present")

    return VerificationResult(status="unsupported", confidence=0.0, reason="answer_not_supported")
