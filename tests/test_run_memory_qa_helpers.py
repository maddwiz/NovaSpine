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


def test_benchmark_reader_choice_resolves_before_relation():
    mod = _load_run_memory_qa_module()
    recalled = [
        {
            "content": (
                "I attended the Data Analysis using Python webinar before the "
                "Effective Time Management workshop."
            ),
            "score": 1.0,
            "metadata": {"session_id": "s1"},
        }
    ]
    ans = mod._benchmark_reader_choice(
        "Which event did I attend first, the 'Effective Time Management' workshop or the 'Data Analysis using Python' webinar?",
        recalled,
    )
    assert ans == "Data Analysis using Python"


def test_benchmark_reader_numeric_extracts_duration():
    mod = _load_run_memory_qa_module()
    recalled = [
        {
            "content": "I had been working for 4 years and 9 months before starting at NovaTech.",
            "score": 1.0,
            "metadata": {"session_id": "s1"},
        }
    ]
    ans = mod._benchmark_reader_numeric(
        "How long have I been working before I started my current job at NovaTech?",
        recalled,
    )
    assert ans == "4 years and 9 months"


def test_fallback_answer_from_context_benchmark_mode_uses_reader():
    mod = _load_run_memory_qa_module()
    recalled = [
        {
            "content": (
                "I attended the Data Analysis using Python webinar before the "
                "Effective Time Management workshop."
            ),
            "score": 1.0,
            "metadata": {"session_id": "s1"},
        }
    ]
    pred = mod._fallback_answer_from_context(
        "Which event did I attend first, the 'Effective Time Management' workshop or the 'Data Analysis using Python' webinar?",
        recalled,
        mode="benchmark",
    )
    assert pred == "Data Analysis using Python"


def test_strict_post_validate_rejects_answer_not_tied_to_entity():
    mod = _load_run_memory_qa_module()
    ctx = (
        "[1] doc_id=a score=1.0\nMelanie has 3 children.\n\n"
        "[2] doc_id=b score=0.9\nCaroline has 1 child."
    )
    reject = mod._should_reject_answer(
        "How many children does Melanie have?",
        "1",
        "NUMBER",
        ctx,
        strict_post_validate=True,
    )
    assert reject is True


def test_result_key_prefers_benchmark_source_then_content_hash():
    mod = _load_run_memory_qa_module()
    row = {
        "id": "chunk-1",
        "content": "Caroline enjoys hiking on weekends.",
        "metadata": {"benchmark_source": "locomo:case:12"},
    }
    key = mod._result_key(row)
    assert key == "source:locomo:case:12"

    row_no_source = {
        "id": "chunk-2",
        "content": "Caroline enjoys hiking on weekends.",
        "metadata": {},
    }
    key2 = mod._result_key(row_no_source)
    assert key2.startswith("content:")
    assert len(key2) > len("content:")


def test_diversify_by_session_uses_benchmark_source_when_session_missing():
    mod = _load_run_memory_qa_module()
    recalled = [
        {
            "id": "a",
            "content": "From source A",
            "score": 1.0,
            "metadata": {"benchmark_source": "srcA"},
        },
        {
            "id": "b",
            "content": "Another from source A",
            "score": 0.9,
            "metadata": {"benchmark_source": "srcA"},
        },
        {
            "id": "c",
            "content": "From source B",
            "score": 0.85,
            "metadata": {"benchmark_source": "srcB"},
        },
    ]
    out = mod._diversify_by_session(recalled, top_k=2, min_sessions=2)
    assert len(out) == 2
    sources = [str((r.get("metadata") or {}).get("benchmark_source", "")) for r in out]
    assert "srcA" in sources and "srcB" in sources


def test_build_recall_variants_adds_case_plus_entity_variant():
    mod = _load_run_memory_qa_module()
    q = "__LOCOMO_CASE_0042__ How many times did Melanie visit the beach?"
    variants = mod._build_recall_variants(q)
    texts = [v[0] for v in variants]
    assert "__LOCOMO_CASE_0042__ Melanie" in texts
