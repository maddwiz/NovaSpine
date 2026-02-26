#!/usr/bin/env python3
"""Run memory retrieval benchmarks (LoCoMo/LongMemEval/DMR-style adapters).

Input format (JSONL):
{"query":"...", "expected_ids":["id1","id2"]}
or
{"query":"...", "expected_text_contains":["tokenA","tokenB"]}
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from c3ae.config import Config
from c3ae.memory_spine.spine import MemorySpine


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run NovaSpine retrieval benchmark dataset.")
    p.add_argument("--dataset", required=True, help="Path to JSONL benchmark data")
    p.add_argument("--name", default="custom", help="Benchmark name (locomo|longmemeval|dmr|custom)")
    p.add_argument("--top-k", type=int, default=10, help="Recall depth")
    p.add_argument("--data-dir", default="", help="Optional C3AE data directory")
    p.add_argument("--out", default="", help="Optional output JSON path")
    return p.parse_args()


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    cfg = Config()
    if args.data_dir:
        cfg.data_dir = Path(args.data_dir)
    cfg.ensure_dirs()
    spine = MemorySpine(cfg)
    try:
        rows = _load_jsonl(Path(args.dataset))
        if not rows:
            raise ValueError("dataset has no rows")

        hits = 0
        mrr_total = 0.0
        for row in rows:
            query = str(row.get("query", "")).strip()
            if not query:
                continue
            expected_ids = {str(x) for x in row.get("expected_ids", []) if str(x)}
            expected_tokens = [str(x).lower() for x in row.get("expected_text_contains", []) if str(x)]

            recalled = await spine.recall(query, top_k=args.top_k)
            rank_hit = None
            for i, item in enumerate(recalled, start=1):
                if expected_ids and str(item.get("id", "")) in expected_ids:
                    rank_hit = i
                    break
                if expected_tokens:
                    txt = str(item.get("content", "")).lower()
                    if all(tok in txt for tok in expected_tokens):
                        rank_hit = i
                        break
            if rank_hit is not None:
                hits += 1
                mrr_total += 1.0 / rank_hit

        n = max(1, len(rows))
        result = {
            "benchmark": args.name,
            "dataset": str(args.dataset),
            "rows": len(rows),
            "top_k": args.top_k,
            "recall_at_k": hits / n,
            "mrr": mrr_total / n,
        }
        return result
    finally:
        await spine.close()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def main() -> None:
    args = _parse_args()
    result = asyncio.run(_run(args))
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
