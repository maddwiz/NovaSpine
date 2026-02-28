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
from c3ae.eval.corpus import build_corpus_docs, load_jsonl
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
        "--ingest-batch-size",
        type=int,
        default=64,
        help="Batch size for async bulk ingestion (ignored with --ingest-sync).",
    )
    p.add_argument(
        "--skip-chunking",
        action="store_true",
        help="Ingest each corpus document as a single chunk (benchmark mode).",
    )
    p.add_argument(
        "--embed-local",
        action="store_true",
        help="Use local hash embeddings and vector retrieval (no external API keys).",
    )
    p.add_argument(
        "--embed-provider",
        default="",
        choices=["", "venice", "openai", "ollama", "hash", "localhash", "sbert"],
        help="Override embedding provider.",
    )
    p.add_argument("--embed-model", default="", help="Optional embedding model override")
    p.add_argument("--embed-dims", type=int, default=0, help="Optional embedding dims override")
    p.add_argument(
        "--query-expansion",
        action="store_true",
        help="Enable keyword query expansion in hybrid retrieval.",
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
    mode_notes: list[str] = []
    if args.embed_local:
        cfg.venice.embedding_provider = "hash"
        cfg.venice.embedding_model = "local-hash-v1"
        if cfg.venice.embedding_dims > 384:
            cfg.venice.embedding_dims = 384
            mode_notes.append("embed_local: reduced embedding_dims to 384 for local hash vectors")
        # Hash embeddings are weaker semantically than model-based vectors:
        # keep hybrid mostly lexical and use vectors as a secondary signal.
        cfg.retrieval.adaptive_weights = False
        cfg.retrieval.keyword_weight = 0.85
        cfg.retrieval.vector_weight = 0.15
        mode_notes.append("embed_local: retrieval profile set to keyword-heavy hybrid (0.85/0.15)")
    if args.embed_provider:
        cfg.venice.embedding_provider = args.embed_provider
        mode_notes.append(f"embed_provider override: {args.embed_provider}")
        if args.embed_provider == "sbert":
            if not args.embed_model:
                cfg.venice.embedding_model = "all-MiniLM-L6-v2"
                mode_notes.append("sbert default model: all-MiniLM-L6-v2")
            if args.embed_dims <= 0:
                cfg.venice.embedding_dims = 384
                mode_notes.append("sbert default dims: 384")
    if args.embed_model:
        cfg.venice.embedding_model = args.embed_model
        mode_notes.append(f"embed_model override: {args.embed_model}")
    if args.embed_dims > 0:
        cfg.venice.embedding_dims = int(args.embed_dims)
        mode_notes.append(f"embed_dims override: {cfg.venice.embedding_dims}")
    if args.query_expansion:
        cfg.retrieval.enable_query_expansion = True
        mode_notes.append("query_expansion=on")
    tmp_dir: str | None = None
    if args.data_dir:
        cfg.data_dir = Path(args.data_dir)
    else:
        tmp_dir = tempfile.mkdtemp(prefix="novaspine-bench-")
        cfg.data_dir = Path(tmp_dir)
    cfg.ensure_dirs()
    spine = MemorySpine(cfg)
    try:
        rows = load_jsonl(Path(args.dataset))
        if not rows:
            raise ValueError("dataset has no rows")

        corpus_docs = build_corpus_docs(rows, load_jsonl(Path(args.corpus)) if args.corpus else [])
        ingested_docs = 0
        ingested_chunks = 0
        use_ingest_sync = bool(args.ingest_sync) and not bool(args.embed_local)
        if args.embed_local and args.ingest_sync:
            mode_notes.append("embed_local enabled: ignoring --ingest-sync to build vector index")
        if use_ingest_sync:
            for doc in corpus_docs:
                text = doc["text"]
                chunk_ids = spine.ingest_text_sync(
                    text,
                    source_id=doc["source_id"],
                    metadata=doc["metadata"],
                    skip_chunking=args.skip_chunking,
                )
                ingested_docs += 1
                ingested_chunks += len(chunk_ids)
        else:
            batch_size = max(1, int(args.ingest_batch_size))
            mode_notes.append(f"ingest_batch_size={batch_size}")
            if batch_size > 1:
                ingested_docs, ingested_chunks = await spine.ingest_documents(
                    corpus_docs,
                    skip_chunking=args.skip_chunking,
                    batch_size=batch_size,
                )
            else:
                for doc in corpus_docs:
                    text = doc["text"]
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
            "embed_local": bool(args.embed_local),
            "publishable": not warnings,
            "quality_warnings": warnings,
            "mode_notes": mode_notes,
        }
        return result
    finally:
        await spine.close()


def main() -> None:
    args = _parse_args()
    result = asyncio.run(_run(args))
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
