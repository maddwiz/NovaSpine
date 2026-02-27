#!/usr/bin/env python3
"""Train a reward-weighted softmax memory policy (offline policy gradient).

Input JSONL row formats (either style):
{"action":"NOOP","reward":1.0,"features":{"similarity":0.93,"evidence_ratio":0.2}}
{"label":"UPDATE","similarity":0.81,"title_similarity":0.72,"content_similarity":0.79}

Outputs a policy JSON compatible with MemoryManagerConfig.learned_policy_path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ACTIONS = ["ADD", "UPDATE", "DELETE", "NOOP"]
DEFAULT_FEATURES = [
    "similarity",
    "title_similarity",
    "content_similarity",
    "evidence_ratio",
    "length_norm",
    "negation",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train reward-weighted memory policy.")
    p.add_argument("--data", required=True, help="JSONL training data")
    p.add_argument("--out", required=True, help="Output JSON file")
    p.add_argument("--epochs", type=int, default=500, help="Training epochs")
    p.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    p.add_argument("--l2", type=float, default=0.0005, help="L2 regularization")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    samples = _load_samples(Path(args.data))
    if not samples:
        raise SystemExit("No valid samples found")

    feature_order = _feature_order(samples)
    n_actions = len(ACTIONS)
    n_features = len(feature_order)
    weights = [[0.0 for _ in range(n_features)] for _ in range(n_actions)]
    bias = [0.0 for _ in range(n_actions)]

    for _ in range(max(1, int(args.epochs))):
        for s in samples:
            x = [float(s["features"].get(f, 0.0)) for f in feature_order]
            action_idx = ACTIONS.index(str(s["action"]))
            reward = float(s["reward"])
            logits = []
            for i in range(n_actions):
                dot = sum(weights[i][j] * x[j] for j in range(n_features)) + bias[i]
                logits.append(dot)
            probs = _softmax(logits)

            for i in range(n_actions):
                grad = ((1.0 if i == action_idx else 0.0) - probs[i]) * reward
                for j in range(n_features):
                    weights[i][j] += float(args.lr) * (grad * x[j] - float(args.l2) * weights[i][j])
                bias[i] += float(args.lr) * grad

    correct = 0
    for s in samples:
        pred, conf = _predict(s["features"], feature_order, weights, bias)
        _ = conf
        if pred == s["action"]:
            correct += 1
    accuracy = correct / max(1, len(samples))

    payload = {
        "memory_manager": {
            "policy_type": "reward_weighted_softmax_v1",
            "actions": ACTIONS,
            "feature_order": feature_order,
            "weights": [[round(v, 8) for v in row] for row in weights],
            "bias": [round(v, 8) for v in bias],
            "training_samples": len(samples),
            "training_accuracy": round(accuracy, 6),
        }
    }
    out = Path(args.out)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "saved": str(out),
                "samples": len(samples),
                "features": feature_order,
                "training_accuracy": round(accuracy, 6),
            },
            indent=2,
        )
    )


def _load_samples(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            action = str(obj.get("action") or obj.get("label") or "ADD").upper()
            if action not in set(ACTIONS):
                continue
            reward = _coerce_float(obj.get("reward"), 1.0)
            features: dict[str, float] = {}
            raw_features = obj.get("features")
            if isinstance(raw_features, dict):
                for k, v in raw_features.items():
                    features[str(k)] = _coerce_float(v, 0.0)
            for key in DEFAULT_FEATURES:
                if key in obj and key not in features:
                    features[key] = _coerce_float(obj.get(key), 0.0)
            if "similarity" not in features:
                features["similarity"] = _coerce_float(obj.get("similarity"), 0.0)
            rows.append(
                {
                    "action": action,
                    "reward": max(0.01, reward),
                    "features": features,
                }
            )
    return rows


def _feature_order(samples: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set(DEFAULT_FEATURES)
    for s in samples:
        seen.update(s.get("features", {}).keys())
    ordered = [f for f in DEFAULT_FEATURES if f in seen]
    rest = sorted(f for f in seen if f not in DEFAULT_FEATURES)
    return ordered + rest


def _predict(
    features: dict[str, float],
    feature_order: list[str],
    weights: list[list[float]],
    bias: list[float],
) -> tuple[str, float]:
    x = [float(features.get(f, 0.0)) for f in feature_order]
    logits = [sum(weights[i][j] * x[j] for j in range(len(feature_order))) + bias[i] for i in range(len(ACTIONS))]
    probs = _softmax(logits)
    idx = max(range(len(probs)), key=lambda i: probs[i])
    return ACTIONS[idx], float(probs[idx])


def _softmax(vals: list[float]) -> list[float]:
    if not vals:
        return []
    m = max(vals)
    exps = [pow(2.718281828, v - m) for v in vals]
    z = sum(exps) or 1.0
    return [v / z for v in exps]


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


if __name__ == "__main__":
    main()
