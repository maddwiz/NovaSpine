from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_run_memory_qa_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "run_memory_qa.py"
    spec = importlib.util.spec_from_file_location("run_memory_qa_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_deterministic_solver_extracts_explicit_count_phrase():
    mod = _load_run_memory_qa_module()
    recalled = [
        {
            "content": "I had been preparing for 7 days before the team meeting.",
            "score": 1.0,
            "metadata": {"session_id": "s1"},
        }
    ]
    ans = mod._deterministic_structured_answer(
        "How many days before the team meeting had I been preparing?",
        recalled,
        "NUMBER",
    )
    assert ans == "7 days"


def test_deterministic_solver_computes_date_difference_days():
    mod = _load_run_memory_qa_module()
    recalled = [
        {
            "content": (
                "I attended a workshop on March 5, 2024. "
                "The team meeting happened on March 12, 2024."
            ),
            "score": 1.0,
            "metadata": {"session_id": "s1"},
        }
    ]
    ans = mod._deterministic_structured_answer(
        "How many days between the workshop and the team meeting?",
        recalled,
        "NUMBER",
    )
    assert ans == "7 days"


def test_answer_context_diversifies_sessions():
    mod = _load_run_memory_qa_module()
    recalled = [
        {
            "id": "a",
            "content": "Session A detail",
            "score": 1.0,
            "metadata": {"session_id": "sA", "benchmark_doc_id": "dA"},
        },
        {
            "id": "b",
            "content": "Session A extra detail",
            "score": 0.95,
            "metadata": {"session_id": "sA", "benchmark_doc_id": "dB"},
        },
        {
            "id": "c",
            "content": "Session B key detail",
            "score": 0.92,
            "metadata": {"session_id": "sB", "benchmark_doc_id": "dC"},
        },
        {
            "id": "d",
            "content": "Session C key detail",
            "score": 0.90,
            "metadata": {"session_id": "sC", "benchmark_doc_id": "dD"},
        },
    ]
    ctx = mod._build_answer_context(
        "What happened across sessions?",
        recalled,
        k=3,
        per_chunk_chars=400,
        total_chars=4000,
        min_score_ratio=0.0,
        min_score_abs=0.0,
        context_rerank="none",
        context_pool_multiplier=3,
        context_overlap_weight=0.35,
        context_session_diversity_min=3,
    )
    assert "doc_id=dA" in ctx
    assert "doc_id=dC" in ctx
    assert "doc_id=dD" in ctx


def test_typed_normalization_shortens_verbose_person_answer():
    mod = _load_run_memory_qa_module()
    ans = mod._normalize_answer_by_type(
        "The answer is Xiu Li Dai, according to the context provided.",
        "PERSON_NAME",
    )
    assert ans == "Xiu Li Dai"


def test_extract_query_entities_prefers_named_subjects():
    mod = _load_run_memory_qa_module()
    ents = mod._extract_query_entities("__LOCOMO_CASE_0040__ How many times has Melanie gone to the beach in 2023?")
    assert "Melanie" in ents


def test_sentence_context_mode_prioritizes_named_entity_sentence():
    mod = _load_run_memory_qa_module()
    recalled = [
        {
            "id": "r1",
            "score": 1.0,
            "content": (
                "Melanie painted horses last week. "
                "Caroline painted a sunset this month."
            ),
            "metadata": {"benchmark_doc_id": "doc1", "session_id": "s1"},
        },
        {
            "id": "r2",
            "score": 0.95,
            "content": "Melanie visited the beach two times in 2023.",
            "metadata": {"benchmark_doc_id": "doc2", "session_id": "s2"},
        },
    ]
    ctx = mod._build_answer_context(
        "What did Caroline paint recently?",
        recalled,
        k=2,
        per_chunk_chars=220,
        total_chars=1000,
        min_score_ratio=0.0,
        min_score_abs=0.0,
        context_rerank="lexical",
        context_pool_multiplier=3,
        context_overlap_weight=0.45,
        context_session_diversity_min=1,
        context_mode="sentence",
        context_sentences_per_doc=2,
        entity_focus=["Caroline"],
    )
    assert "Caroline painted a sunset" in ctx


def test_should_span_refine_flags_long_non_contextual_answers():
    mod = _load_run_memory_qa_module()
    ctx = "[1] doc_id=x score=1.0\nCaroline moved from Sweden four years ago."
    assert mod._should_span_refine(
        "Caroline mentioned many unrelated details about various topics",
        "SHORT_PHRASE",
        ctx,
    )


def test_fallback_answer_from_context_returns_typed_short_answer():
    mod = _load_run_memory_qa_module()
    recalled = [
        {"content": "Melanie has 3 children and Caroline has 1 child."},
    ]
    pred = mod._fallback_answer_from_context("How many children does Melanie have?", recalled)
    assert pred == "3"
