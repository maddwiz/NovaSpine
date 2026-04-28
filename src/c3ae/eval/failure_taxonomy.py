"""Row-level memory QA failure classification."""

from __future__ import annotations

from dataclasses import dataclass

from c3ae.eval.qa_metrics import best_exact_match
from c3ae.qa.normalizer import infer_answer_type, is_abstention, normalize_answer, normalize_for_match

FAILURE_KINDS = {
    "retrieval_miss",
    "rerank_miss",
    "evidence_present_answer_wrong",
    "evidence_present_answer_unverified",
    "abstain_false_positive",
    "abstain_false_negative",
    "unsupported_answer",
    "success",
    "unknown",
}


@dataclass(frozen=True)
class FailureClassification:
    failure_kind: str
    normalized_gold: str
    normalized_pred: str
    answer_type: str

    def to_dict(self) -> dict[str, str]:
        return {
            "failure_kind": self.failure_kind,
            "normalized_gold": self.normalized_gold,
            "normalized_pred": self.normalized_pred,
            "answer_type": self.answer_type,
        }


def classify_qa_failure(
    *,
    question: str,
    gold_answers: list[str],
    pred_answer: str,
    expected_evidence_ids: list[str] | set[str],
    candidate_evidence_ids: list[str] | set[str],
    final_context_evidence_ids: list[str] | set[str],
    verifier_status: str = "unchecked",
    answer_type: str | None = None,
) -> FailureClassification:
    inferred_type = answer_type or infer_answer_type(question)
    gold = gold_answers[0] if gold_answers else ""
    normalized_gold = normalize_answer(gold, inferred_type).answer if gold else ""
    normalized_pred = normalize_answer(pred_answer, inferred_type).answer if pred_answer else ""

    gold_is_abstain = bool(gold_answers) and all(is_abstention(ans) for ans in gold_answers)
    pred_is_abstain = is_abstention(pred_answer)

    if gold_answers and best_exact_match(pred_answer, gold_answers) >= 1.0:
        if verifier_status in {"unsupported", "partial"}:
            return FailureClassification(
                "evidence_present_answer_unverified",
                normalize_for_match(normalized_gold),
                normalize_for_match(normalized_pred),
                inferred_type,
            )
        return FailureClassification(
            "success",
            normalize_for_match(normalized_gold),
            normalize_for_match(normalized_pred),
            inferred_type,
        )

    if gold_is_abstain and not pred_is_abstain:
        return FailureClassification(
            "abstain_false_negative",
            normalize_for_match(normalized_gold),
            normalize_for_match(normalized_pred),
            inferred_type,
        )

    expected = {str(x) for x in expected_evidence_ids if str(x)}
    candidates = {str(x) for x in candidate_evidence_ids if str(x)}
    final = {str(x) for x in final_context_evidence_ids if str(x)}
    if expected:
        if expected.isdisjoint(candidates):
            kind = "retrieval_miss"
        elif expected.isdisjoint(final):
            kind = "rerank_miss"
        elif pred_is_abstain and not gold_is_abstain:
            kind = "abstain_false_positive"
        elif verifier_status == "unsupported":
            kind = "unsupported_answer"
        elif verifier_status in {"partial", "unchecked"}:
            kind = "evidence_present_answer_unverified"
        else:
            kind = "evidence_present_answer_wrong"
    elif pred_is_abstain and not gold_is_abstain:
        kind = "abstain_false_positive"
    elif verifier_status == "unsupported":
        kind = "unsupported_answer"
    else:
        kind = "unknown"
    return FailureClassification(
        kind,
        normalize_for_match(normalized_gold),
        normalize_for_match(normalized_pred),
        inferred_type,
    )
