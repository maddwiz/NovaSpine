from c3ae.eval.failure_taxonomy import classify_qa_failure


def test_failure_taxonomy_separates_retrieval_and_rerank_misses():
    retrieval = classify_qa_failure(
        question="What did Caroline research?",
        gold_answers=["adoption agencies"],
        pred_answer="not enough information",
        expected_evidence_ids={"doc-1"},
        candidate_evidence_ids={"doc-2"},
        final_context_evidence_ids={"doc-2"},
    )
    rerank = classify_qa_failure(
        question="What did Caroline research?",
        gold_answers=["adoption agencies"],
        pred_answer="not enough information",
        expected_evidence_ids={"doc-1"},
        candidate_evidence_ids={"doc-1", "doc-2"},
        final_context_evidence_ids={"doc-2"},
    )

    assert retrieval.failure_kind == "retrieval_miss"
    assert rerank.failure_kind == "rerank_miss"


def test_failure_taxonomy_marks_answer_stage_when_evidence_present():
    result = classify_qa_failure(
        question="What did Caroline research?",
        gold_answers=["adoption agencies"],
        pred_answer="foster homes",
        expected_evidence_ids={"doc-1"},
        candidate_evidence_ids={"doc-1"},
        final_context_evidence_ids={"doc-1"},
        verifier_status="supported",
    )

    assert result.failure_kind == "evidence_present_answer_wrong"


def test_failure_taxonomy_marks_success():
    result = classify_qa_failure(
        question="What did Caroline research?",
        gold_answers=["Adoption agencies"],
        pred_answer="adoption agencies",
        expected_evidence_ids={"doc-1"},
        candidate_evidence_ids={"doc-1"},
        final_context_evidence_ids={"doc-1"},
        verifier_status="supported",
    )

    assert result.failure_kind == "success"
