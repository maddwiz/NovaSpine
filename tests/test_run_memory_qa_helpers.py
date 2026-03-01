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

