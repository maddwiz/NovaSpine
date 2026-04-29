from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_train_memory_reranker_from_candidate_features(tmp_path):
    rows = [
        {
            "question_id": "q1",
            "expected_evidence_ids": ["doc-positive"],
            "candidate_features": [
                {
                    "chunk_id": "chunk-a",
                    "benchmark_doc_id": "doc-positive",
                    "rrf_score": 0.9,
                    "token_overlap": 0.8,
                    "entity_overlap": 1.0,
                    "role": "assistant",
                    "source_kind": "chunk",
                },
                {
                    "chunk_id": "chunk-b",
                    "benchmark_doc_id": "doc-negative",
                    "rrf_score": 0.1,
                    "token_overlap": 0.1,
                    "entity_overlap": 0.0,
                    "role": "user",
                    "source_kind": "chunk",
                },
            ],
        },
        {
            "question_id": "q2",
            "expected_evidence_ids": ["doc-two"],
            "candidate_features": [
                {
                    "chunk_id": "chunk-c",
                    "benchmark_doc_id": "doc-two",
                    "vector_score": 0.75,
                    "token_overlap": 0.7,
                    "role": "user",
                    "source_kind": "chunk",
                },
                {
                    "chunk_id": "chunk-d",
                    "benchmark_doc_id": "doc-other",
                    "vector_score": 0.2,
                    "token_overlap": 0.05,
                    "role": "assistant",
                    "source_kind": "chunk",
                },
            ],
        },
    ]
    data = tmp_path / "rows.jsonl"
    data.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    out = tmp_path / "reranker.json"

    subprocess.run(
        [
            sys.executable,
            str(Path("scripts/train_memory_reranker.py")),
            "--data",
            str(data),
            "--out",
            str(out),
            "--epochs",
            "20",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    policy = payload["retrieval_reranker"]
    assert policy["policy_type"] == "linear_logistic_v1"
    assert policy["training_samples"] == 4
    assert policy["positive_samples"] == 2
    assert "token_overlap" in policy["feature_order"]


def test_train_memory_reranker_keeps_explicit_all_negative_rows(tmp_path):
    rows = [
        {
            "question_id": "q-negative",
            "candidate_features": [
                {"chunk_id": "a", "label": False, "token_overlap": 0.3},
                {"chunk_id": "b", "label": 0, "token_overlap": 0.1},
            ],
        }
    ]
    data = tmp_path / "negative.jsonl"
    data.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    out = tmp_path / "reranker.json"

    subprocess.run(
        [
            sys.executable,
            str(Path("scripts/train_memory_reranker.py")),
            "--data",
            str(data),
            "--out",
            str(out),
            "--epochs",
            "5",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        capture_output=True,
    )

    policy = json.loads(out.read_text(encoding="utf-8"))["retrieval_reranker"]
    assert policy["training_samples"] == 2
    assert policy["positive_samples"] == 0
    assert policy["negative_samples"] == 2
