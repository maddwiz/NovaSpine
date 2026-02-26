#!/usr/bin/env python3
"""Train/tune memory manager policy thresholds from labeled decisions.

Input JSONL rows:
{"similarity":0.95,"label":"NOOP"}
{"similarity":0.83,"label":"UPDATE"}
{"similarity":0.40,"label":"ADD"}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tune NovaSpine memory policy thresholds.")
    p.add_argument("--data", required=True, help="Path to labeled JSONL data")
    p.add_argument("--out", required=True, help="Output JSON config path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    samples = _load(Path(args.data))
    if not samples:
        raise SystemExit("No samples found")

    add = [s for s in samples if s["label"] == "ADD"]
    upd = [s for s in samples if s["label"] == "UPDATE"]
    nop = [s for s in samples if s["label"] == "NOOP"]

    update_thr = _mean([s["similarity"] for s in upd]) if upd else 0.80
    noop_thr = _mean([s["similarity"] for s in nop]) if nop else 0.92
    if noop_thr < update_thr:
        noop_thr = min(0.99, update_thr + 0.05)

    policy = {
        "memory_manager": {
            "enabled": True,
            "use_llm_policy": False,
            "similarity_update_threshold": round(update_thr, 4),
            "similarity_noop_threshold": round(noop_thr, 4),
            "training_samples": len(samples),
            "label_distribution": {
                "ADD": len(add),
                "UPDATE": len(upd),
                "NOOP": len(nop),
            },
        }
    }
    out = Path(args.out)
    out.write_text(json.dumps(policy, indent=2), encoding="utf-8")
    print(f"saved policy to {out}")


def _load(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                continue
            try:
                sim = float(obj.get("similarity", 0.0))
                label = str(obj.get("label", "ADD")).upper()
            except Exception:
                continue
            if label not in {"ADD", "UPDATE", "NOOP"}:
                continue
            rows.append({"similarity": max(0.0, min(1.0, sim)), "label": label})
    return rows


def _mean(vals: list[float]) -> float:
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


if __name__ == "__main__":
    main()
