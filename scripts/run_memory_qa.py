#!/usr/bin/env python3
"""Run end-to-end QA over NovaSpine retrieval context.

Expected QA eval JSONL row format:
{"query":"...", "expected_answers":["..."], "expected_doc_ids":["doc1","doc2"]}

Corpus JSONL rows are the same as run_memory_benchmarks.py.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import tempfile
from pathlib import Path
from typing import Any

from c3ae.config import Config
from c3ae.eval import best_exact_match, best_f1, extractive_answer
from c3ae.memory_spine.spine import MemorySpine


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run NovaSpine end-to-end QA benchmark.")
    p.add_argument("--dataset", required=True, help="QA eval JSONL path")
    p.add_argument("--corpus", default="", help="Optional corpus JSONL to ingest before eval")
    p.add_argument("--name", default="qa_custom", help="Benchmark run name")
    p.add_argument("--top-k", type=int, default=10, help="Recall depth")
    p.add_argument("--data-dir", default="", help="Optional persistent data dir")
    p.add_argument("--ingest-sync", action="store_true", help="Use sync ingest path")
    p.add_argument("--skip-chunking", action="store_true", help="Ingest docs as single chunks")
    p.add_argument("--embed-local", action="store_true", help="Use local hash embeddings")
    p.add_argument(
        "--answer-mode",
        default="extractive",
        choices=["extractive", "oracle_doc"],
        help="Answer strategy. oracle_doc uses expected doc hit as an answer gate.",
    )
    p.add_argument("--out", default="", help="Optional output JSON path")
    return p.parse_args()


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
                if isinstance(mem, str) and mem.strip():
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


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    cfg = Config()
    mode_notes: list[str] = []
    if args.embed_local:
        cfg.venice.embedding_provider = "hash"
        cfg.venice.embedding_model = "local-hash-v1"
        if cfg.venice.embedding_dims > 384:
            cfg.venice.embedding_dims = 384
            mode_notes.append("embed_local: reduced embedding_dims to 384 for local hash vectors")
        cfg.retrieval.adaptive_weights = False
        cfg.retrieval.keyword_weight = 0.85
        cfg.retrieval.vector_weight = 0.15
        mode_notes.append("embed_local: retrieval profile set to keyword-heavy hybrid (0.85/0.15)")

    tmp_dir: str | None = None
    if args.data_dir:
        cfg.data_dir = Path(args.data_dir)
    else:
        tmp_dir = tempfile.mkdtemp(prefix="novaspine-qa-")
        cfg.data_dir = Path(tmp_dir)
    cfg.ensure_dirs()
    spine = MemorySpine(cfg)
    try:
        rows = _load_jsonl(Path(args.dataset))
        if not rows:
            raise ValueError("dataset has no rows")
        corpus_docs = _build_corpus_docs(rows, _load_jsonl(Path(args.corpus)) if args.corpus else [])
        use_ingest_sync = bool(args.ingest_sync) and not bool(args.embed_local)
        if args.embed_local and args.ingest_sync:
            mode_notes.append("embed_local enabled: ignoring --ingest-sync to build vector index")

        ingested_docs = 0
        ingested_chunks = 0
        for doc in corpus_docs:
            if use_ingest_sync:
                cids = spine.ingest_text_sync(
                    doc["text"],
                    source_id=doc["source_id"],
                    metadata=doc["metadata"],
                    skip_chunking=args.skip_chunking,
                )
            else:
                cids = await spine.ingest_text(
                    doc["text"],
                    source_id=doc["source_id"],
                    metadata=doc["metadata"],
                    skip_chunking=args.skip_chunking,
                )
            ingested_docs += 1
            ingested_chunks += len(cids)

        n = 0
        doc_hits = 0
        em_sum = 0.0
        f1_sum = 0.0
        skipped = 0
        sample_logs: list[dict[str, Any]] = []

        for row in rows:
            query = str(row.get("query", "")).strip()
            answers = row.get("expected_answers", [])
            if isinstance(answers, str):
                answers = [answers]
            answers = [str(a).strip() for a in answers if str(a).strip()]
            if not query or not answers:
                skipped += 1
                continue

            expected_doc_ids = {str(x) for x in row.get("expected_doc_ids", []) if str(x)}
            recalled = await spine.recall(query, top_k=args.top_k)
            docs = [str((r.get("metadata") or {}).get("benchmark_doc_id", "")) for r in recalled]
            hit = any(d in expected_doc_ids for d in docs) if expected_doc_ids else False

            if args.answer_mode == "oracle_doc" and expected_doc_ids:
                pred = answers[0] if hit else extractive_answer(query, recalled)
            else:
                pred = extractive_answer(query, recalled)

            em = best_exact_match(pred, answers)
            f1 = best_f1(pred, answers)

            n += 1
            doc_hits += int(hit)
            em_sum += em
            f1_sum += f1

            if len(sample_logs) < 20:
                sample_logs.append(
                    {
                        "query": query[:240],
                        "prediction": pred[:240],
                        "gold": answers[0][:240],
                        "doc_hit": bool(hit),
                        "em": em,
                        "f1": f1,
                    }
                )

        denom = max(1, n)
        return {
            "benchmark": args.name,
            "dataset": str(args.dataset),
            "rows_total": len(rows),
            "rows_scored": n,
            "rows_skipped": skipped,
            "top_k": args.top_k,
            "answer_mode": args.answer_mode,
            "doc_hit_rate": doc_hits / denom,
            "exact_match": em_sum / denom,
            "token_f1": f1_sum / denom,
            "ingested_documents": ingested_docs,
            "ingested_chunks": ingested_chunks,
            "data_dir": str(cfg.data_dir),
            "ephemeral_data_dir": bool(tmp_dir),
            "skip_chunking": bool(args.skip_chunking),
            "embed_local": bool(args.embed_local),
            "mode_notes": mode_notes,
            "samples": sample_logs,
        }
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
