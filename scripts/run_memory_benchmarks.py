#!/usr/bin/env python3
"""Run memory retrieval benchmarks (LoCoMo/LongMemEval/DMR-style adapters).

Eval JSONL row formats:
{"query":"...", "expected_ids":["id1","id2"]}
{"query":"...", "expected_doc_ids":["docA"]}  # matches metadata.benchmark_doc_id
{"query":"...", "expected_text_contains":["tokenA","tokenB"]}

Optional corpus JSONL row formats:
{"doc_id":"docA", "text":"..."}
{"doc_id":"docB", "content":"...", "source_id":"bench:docB", "metadata":{"session":"s1"}}
"""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any

from c3ae.config import Config
from c3ae.memory_spine.spine import MemorySpine


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run NovaSpine retrieval benchmark dataset.")
    p.add_argument("--dataset", required=True, help="Path to JSONL benchmark data")
    p.add_argument("--corpus", default="", help="Optional JSONL corpus to ingest before eval")
    p.add_argument("--name", default="custom", help="Benchmark name (locomo|longmemeval|dmr|custom)")
    p.add_argument("--top-k", type=int, default=10, help="Recall depth")
    p.add_argument("--data-dir", default="", help="Optional C3AE data directory")
    p.add_argument(
        "--ingest-sync",
        action="store_true",
        help="Ingest benchmark corpus via keyword-only sync path (no embedding API required)",
    )
    p.add_argument(
        "--skip-chunking",
        action="store_true",
        help="Ingest each corpus document as a single chunk (benchmark mode).",
    )
    p.add_argument(
        "--min-publish-rows",
        type=int,
        default=100,
        help="Minimum eval rows before results are considered publishable",
    )
    p.add_argument(
        "--min-publish-docs",
        type=int,
        default=500,
        help="Minimum ingested docs before results are considered publishable",
    )
    p.add_argument("--out", default="", help="Optional output JSON path")
    return p.parse_args()


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    cfg = Config()
    tmp_dir: str | None = None
    if args.data_dir:
        cfg.data_dir = Path(args.data_dir)
    else:
        tmp_dir = tempfile.mkdtemp(prefix="novaspine-bench-")
        cfg.data_dir = Path(tmp_dir)
    cfg.ensure_dirs()
    spine = MemorySpine(cfg)
    try:
        rows = _load_jsonl(Path(args.dataset))
        if not rows:
            raise ValueError("dataset has no rows")

        corpus_docs = _build_corpus_docs(rows, _load_jsonl(Path(args.corpus)) if args.corpus else [])
        ingested_docs = 0
        ingested_chunks = 0
        for doc in corpus_docs:
            text = doc["text"]
            if args.ingest_sync:
                chunk_ids = spine.ingest_text_sync(
                    text,
                    source_id=doc["source_id"],
                    metadata=doc["metadata"],
                    skip_chunking=args.skip_chunking,
                )
            else:
                chunk_ids = await spine.ingest_text(
                    text,
                    source_id=doc["source_id"],
                    metadata=doc["metadata"],
                    skip_chunking=args.skip_chunking,
                )
            ingested_docs += 1
            ingested_chunks += len(chunk_ids)

        hits = 0
        mrr_total = 0.0
        query_rows = 0
        for row in rows:
            query = str(row.get("query", "")).strip()
            if not query:
                continue
            query_rows += 1
            expected_ids = {str(x) for x in row.get("expected_ids", []) if str(x)}
            expected_doc_ids = {str(x) for x in row.get("expected_doc_ids", []) if str(x)}
            expected_tokens = [str(x).lower() for x in row.get("expected_text_contains", []) if str(x)]

            recalled = await spine.recall(query, top_k=args.top_k)
            rank_hit = None
            for i, item in enumerate(recalled, start=1):
                if expected_ids and str(item.get("id", "")) in expected_ids:
                    rank_hit = i
                    break
                if expected_doc_ids:
                    meta = item.get("metadata") or {}
                    if str(meta.get("benchmark_doc_id", "")) in expected_doc_ids:
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

        n = max(1, query_rows)
        warnings: list[str] = []
        if query_rows < int(args.min_publish_rows):
            warnings.append(
                f"eval rows ({query_rows}) below publish threshold ({int(args.min_publish_rows)})"
            )
        if ingested_docs < int(args.min_publish_docs):
            warnings.append(
                f"ingested docs ({ingested_docs}) below publish threshold ({int(args.min_publish_docs)})"
            )
        dataset_l = str(args.dataset).lower()
        corpus_l = str(args.corpus).lower()
        if "bench/fixtures" in dataset_l or "bench/fixtures" in corpus_l:
            warnings.append("fixture dataset detected; use official converted corpora for publishable claims")

        result = {
            "benchmark": args.name,
            "dataset": str(args.dataset),
            "rows": query_rows,
            "top_k": args.top_k,
            "recall_at_k": hits / n,
            "mrr": mrr_total / n,
            "ingested_documents": ingested_docs,
            "ingested_chunks": ingested_chunks,
            "data_dir": str(cfg.data_dir),
            "ephemeral_data_dir": bool(tmp_dir),
            "skip_chunking": bool(args.skip_chunking),
            "publishable": not warnings,
            "quality_warnings": warnings,
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


def _build_corpus_docs(
    eval_rows: list[dict[str, Any]],
    corpus_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []

    def _append_doc(
        *,
        text: str,
        doc_id: str,
        source_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        t = text.strip()
        if not t:
            return
        meta = dict(metadata or {})
        meta.setdefault("benchmark_doc_id", doc_id)
        meta.setdefault("benchmark_source", source_id)
        docs.append(
            {
                "text": t,
                "doc_id": doc_id,
                "source_id": source_id,
                "metadata": meta,
            }
        )

    for idx, row in enumerate(corpus_rows):
        text = str(row.get("text") or row.get("content") or row.get("memory") or "").strip()
        if not text:
            continue
        doc_id = str(row.get("doc_id") or row.get("id") or f"corpus_{idx:04d}")
        source_id = str(row.get("source_id") or f"benchmark:{doc_id}")
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        _append_doc(text=text, doc_id=doc_id, source_id=source_id, metadata=metadata)

    for idx, row in enumerate(eval_rows):
        base_doc_id = str(row.get("doc_id") or f"eval_{idx:04d}")
        source_id = str(row.get("source_id") or f"benchmark:{base_doc_id}")
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}

        memory = row.get("memory")
        if isinstance(memory, str) and memory.strip():
            _append_doc(
                text=memory,
                doc_id=base_doc_id,
                source_id=source_id,
                metadata=metadata,
            )

        memories = row.get("memories")
        if isinstance(memories, list):
            for j, mem in enumerate(memories):
                if not isinstance(mem, str) or not mem.strip():
                    continue
                _append_doc(
                    text=mem,
                    doc_id=f"{base_doc_id}_m{j}",
                    source_id=f"{source_id}:m{j}",
                    metadata=metadata,
                )

    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for doc in docs:
        key = (doc["doc_id"], doc["text"])
        if key not in unique:
            unique[key] = doc
    return list(unique.values())


def main() -> None:
    args = _parse_args()
    result = asyncio.run(_run(args))
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
