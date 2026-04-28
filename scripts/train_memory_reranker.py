#!/usr/bin/env python3
"""Train an offline linear reranker from NovaSpine benchmark JSONL rows.

This does not change runtime ranking. It turns rows emitted with
`--capture-candidate-features` into a small JSON policy artifact that can be
reviewed before any future runtime integration.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


BASE_FEATURES = [
    "bm25_score",
    "vector_score",
    "rrf_score",
    "graph_score",
    "fact_score",
    "entity_overlap",
    "token_overlap",
    "recency_score",
    "importance_score",
    "access_count",
    "role_user",
    "role_assistant",
    "source_chunk",
    "source_structured",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an offline NovaSpine candidate reranker.")
    parser.add_argument("--data", required=True, help="Benchmark row JSONL with candidate_features")
    parser.add_argument("--out", required=True, help="Output reranker JSON path")
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.08, help="Learning rate")
    parser.add_argument("--l2", type=float, default=0.0005, help="L2 regularization")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_samples(Path(args.data))
    if not samples:
        raise SystemExit("No labeled candidate samples found")

    weights = [0.0 for _ in BASE_FEATURES]
    bias = 0.0
    for _ in range(max(1, int(args.epochs))):
        for features, label in samples:
            x = [features.get(name, 0.0) for name in BASE_FEATURES]
            pred = _sigmoid(sum(w * v for w, v in zip(weights, x, strict=True)) + bias)
            err = float(label) - pred
            for i, value in enumerate(x):
                weights[i] += float(args.lr) * (err * value - float(args.l2) * weights[i])
            bias += float(args.lr) * err

    predictions = [(_predict(features, weights, bias), label) for features, label in samples]
    accuracy = sum(int(pred >= 0.5) == label for pred, label in predictions) / max(1, len(samples))
    positives = sum(label for _, label in samples)
    payload = {
        "retrieval_reranker": {
            "policy_type": "linear_logistic_v1",
            "feature_order": BASE_FEATURES,
            "weights": [round(v, 8) for v in weights],
            "bias": round(bias, 8),
            "threshold": 0.5,
            "training_samples": len(samples),
            "positive_samples": positives,
            "negative_samples": len(samples) - positives,
            "training_accuracy": round(accuracy, 6),
            "notes": [
                "Offline artifact only; runtime ranking is unchanged unless a future config explicitly loads it.",
                "Labels are derived from expected evidence IDs, candidate doc IDs, or explicit candidate labels.",
            ],
        }
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"saved": str(out), **payload["retrieval_reranker"]}, indent=2))


def load_samples(path: Path) -> list[tuple[dict[str, float], int]]:
    samples: list[tuple[dict[str, float], int]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            expected = _id_set(row.get("expected_evidence_ids"))
            candidates = row.get("candidate_features")
            if not isinstance(candidates, list):
                continue

            row_samples: list[tuple[dict[str, float], int]] = []
            has_explicit_labels = False
            has_positive = False
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                has_explicit_labels = has_explicit_labels or any(
                    key in candidate for key in ("label", "relevant", "target")
                )
                label = _candidate_label(candidate, expected)
                if label is None:
                    continue
                has_positive = has_positive or bool(label)
                row_samples.append((_feature_vector(candidate), label))
            if has_positive or has_explicit_labels:
                samples.extend(row_samples)
    return samples


def _candidate_label(candidate: dict[str, Any], expected_ids: set[str]) -> int | None:
    for key in ("label", "relevant", "target"):
        if key in candidate:
            value = candidate.get(key)
            if isinstance(value, bool):
                return int(value)
            text = str(value).strip().lower()
            if text in {"1", "true", "yes", "positive", "relevant"}:
                return 1
            if text in {"0", "false", "no", "negative", "irrelevant"}:
                return 0
    if not expected_ids:
        return None
    candidate_ids = {
        str(candidate.get("chunk_id", "")),
        str(candidate.get("benchmark_doc_id", "")),
        str(candidate.get("benchmark_source", "")),
        str(candidate.get("doc_id", "")),
        str(candidate.get("source_id", "")),
    }
    return int(bool(expected_ids & {item for item in candidate_ids if item}))


def _feature_vector(candidate: dict[str, Any]) -> dict[str, float]:
    role = str(candidate.get("role", "")).lower()
    source = str(candidate.get("source_kind", "")).lower()
    values = {
        "role_user": 1.0 if role == "user" else 0.0,
        "role_assistant": 1.0 if role == "assistant" else 0.0,
        "source_chunk": 1.0 if source in {"", "chunk"} else 0.0,
        "source_structured": 1.0 if source.startswith("structured") else 0.0,
    }
    for key in BASE_FEATURES:
        if key in values:
            continue
        values[key] = _coerce_float(candidate.get(key), 0.0)
    values["access_count"] = min(values["access_count"], 100.0) / 100.0
    return values


def _id_set(value: Any) -> set[str]:
    if not isinstance(value, list | tuple | set):
        return set()
    return {str(item).strip() for item in value if str(item).strip()}


def _coerce_float(value: Any, default: float) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _sigmoid(value: float) -> float:
    value = max(-60.0, min(60.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def _predict(features: dict[str, float], weights: list[float], bias: float) -> float:
    x = [features.get(name, 0.0) for name in BASE_FEATURES]
    return _sigmoid(sum(w * v for w, v in zip(weights, x, strict=True)) + bias)


if __name__ == "__main__":
    main()
