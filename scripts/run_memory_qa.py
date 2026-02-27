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
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from c3ae.config import Config
from c3ae.eval import best_exact_match, best_f1, extractive_answer
from c3ae.llm import Message, create_chat_backend
from c3ae.memory_spine.spine import MemorySpine

_CASE_TOKEN_RE = re.compile(r"__\w+_CASE_\d+__", re.IGNORECASE)


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
        "--embed-provider",
        default="",
        choices=["", "venice", "openai", "ollama", "hash", "localhash"],
        help="Override embedding provider (defaults to config/env).",
    )
    p.add_argument("--embed-model", default="", help="Optional embedding model override")
    p.add_argument("--embed-dims", type=int, default=0, help="Optional embedding dims override")
    p.add_argument(
        "--tune-preset",
        default="none",
        choices=["none", "keyword_plus", "hybrid_qa"],
        help="Apply retrieval tuning preset.",
    )
    p.add_argument("--keyword-weight", type=float, default=-1.0, help="Explicit keyword weight override")
    p.add_argument("--vector-weight", type=float, default=-1.0, help="Explicit vector weight override")
    p.add_argument("--rrf-k", type=int, default=0, help="Explicit RRF k override")
    p.add_argument(
        "--rrf-overlap-boost",
        type=float,
        default=0.0,
        help="Explicit overlap boost override",
    )
    p.add_argument("--disable-graph", action="store_true", help="Disable graph indexing/merge")
    p.add_argument(
        "--answer-mode",
        default="extractive",
        choices=["extractive", "oracle_doc", "llm"],
        help="Answer strategy. oracle_doc uses expected doc hit as an answer gate.",
    )
    p.add_argument(
        "--answer-provider",
        default=os.environ.get("C3AE_QA_LLM_PROVIDER", "openai"),
        choices=["venice", "openai", "anthropic", "ollama"],
        help="LLM provider for --answer-mode llm.",
    )
    p.add_argument(
        "--answer-model",
        default=os.environ.get("C3AE_QA_LLM_MODEL", "gpt-4.1-mini"),
        help="LLM model for --answer-mode llm.",
    )
    p.add_argument("--answer-temperature", type=float, default=0.0, help="LLM answer temperature")
    p.add_argument("--answer-max-tokens", type=int, default=160, help="LLM answer max tokens")
    p.add_argument(
        "--answer-context-k",
        type=int,
        default=8,
        help="Number of recalled chunks to provide to answer LLM",
    )
    p.add_argument(
        "--answer-max-context-chars",
        type=int,
        default=12000,
        help="Global max context chars provided to answer LLM",
    )
    p.add_argument(
        "--answer-chunk-chars",
        type=int,
        default=1400,
        help="Per-chunk truncation length for answer context",
    )
    p.add_argument(
        "--no-answer-fallback",
        action="store_true",
        help="Disable extractive fallback when LLM answer is empty/invalid.",
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


def _clean_question(query: str) -> str:
    return re.sub(r"\s+", " ", _CASE_TOKEN_RE.sub(" ", query)).strip()


def _clean_answer(ans: str) -> str:
    a = re.sub(r"\s+", " ", ans.strip())
    a = a.strip(" \"'")
    if a.lower().startswith("answer:"):
        a = a.split(":", 1)[1].strip()
    return a


def _extract_json_dict(s: str) -> dict[str, Any]:
    raw = s.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            return {}
    return {}


def _build_answer_context(recalled: list[dict[str, Any]], k: int, per_chunk_chars: int, total_chars: int) -> str:
    chunks: list[str] = []
    used = 0
    for i, row in enumerate(recalled[: max(1, k)]):
        content = str(row.get("content", "")).strip()
        if not content:
            continue
        if len(content) > per_chunk_chars:
            content = content[:per_chunk_chars].rstrip() + " ..."
        md = row.get("metadata") or {}
        doc_id = str(md.get("benchmark_doc_id", "")).strip() or str(row.get("id", ""))
        score = float(row.get("score", 0.0))
        block = f"[{i+1}] doc_id={doc_id} score={score:.4f}\n{content}"
        if used + len(block) > total_chars:
            break
        chunks.append(block)
        used += len(block)
    return "\n\n".join(chunks)


async def _answer_with_llm(
    *,
    query: str,
    recalled: list[dict[str, Any]],
    llm_backend: Any,
    context_k: int,
    per_chunk_chars: int,
    total_chars: int,
    temperature: float,
    max_tokens: int,
) -> str:
    clean_q = _clean_question(query)
    context = _build_answer_context(recalled, context_k, per_chunk_chars, total_chars)
    if not context:
        return ""
    system = (
        "You are a strict QA answerer. Use only the provided context.\n"
        "Return JSON only: {\"answer\":\"...\"}.\n"
        "Rules:\n"
        "- answer with the shortest exact phrase possible\n"
        "- no explanation, no extra keys, no markdown\n"
        "- if uncertain, return {\"answer\":\"\"}"
    )
    user = (
        f"Question:\n{clean_q}\n\n"
        f"Context:\n{context}\n\n"
        "Return JSON now."
    )
    resp = await llm_backend.chat(
        [Message(role="system", content=system), Message(role="user", content=user)],
        temperature=temperature,
        max_tokens=max_tokens,
        json_mode=True,
    )
    obj = _extract_json_dict(resp.content)
    ans = _clean_answer(str(obj.get("answer", ""))) if obj else ""
    if ans:
        return ans
    return _clean_answer(resp.content)


def _validate_llm_provider(provider: str, cfg: Config) -> None:
    p = provider.strip().lower()
    if p == "openai" and not os.environ.get("OPENAI_API_KEY", ""):
        raise RuntimeError("OPENAI_API_KEY is required for --answer-provider openai")
    if p == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY", ""):
        raise RuntimeError("ANTHROPIC_API_KEY is required for --answer-provider anthropic")
    if p == "venice":
        venice_key = cfg.venice.api_key or os.environ.get("VENICE_API_KEY", "")
        if not venice_key:
            raise RuntimeError("VENICE_API_KEY is required for --answer-provider venice")


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
    if args.embed_provider:
        cfg.venice.embedding_provider = args.embed_provider
        mode_notes.append(f"embed_provider override: {args.embed_provider}")
    if args.embed_model:
        cfg.venice.embedding_model = args.embed_model
        mode_notes.append(f"embed_model override: {args.embed_model}")
    if args.embed_dims > 0:
        cfg.venice.embedding_dims = int(args.embed_dims)
        mode_notes.append(f"embed_dims override: {cfg.venice.embedding_dims}")
    if args.disable_graph:
        cfg.graph.enabled = False
        mode_notes.append("graph disabled")
    if args.tune_preset == "keyword_plus":
        cfg.retrieval.adaptive_weights = False
        cfg.retrieval.keyword_weight = 0.88
        cfg.retrieval.vector_weight = 0.12
        cfg.retrieval.rrf_overlap_boost = 1.35
        mode_notes.append("tune_preset=keyword_plus")
    elif args.tune_preset == "hybrid_qa":
        cfg.retrieval.adaptive_weights = True
        cfg.retrieval.keyword_weight = 0.45
        cfg.retrieval.vector_weight = 0.55
        cfg.retrieval.rrf_overlap_boost = 1.40
        mode_notes.append("tune_preset=hybrid_qa")
    if args.keyword_weight >= 0.0:
        cfg.retrieval.keyword_weight = float(args.keyword_weight)
        mode_notes.append(f"keyword_weight override: {cfg.retrieval.keyword_weight}")
    if args.vector_weight >= 0.0:
        cfg.retrieval.vector_weight = float(args.vector_weight)
        mode_notes.append(f"vector_weight override: {cfg.retrieval.vector_weight}")
    if args.rrf_k > 0:
        cfg.retrieval.rrf_k = int(args.rrf_k)
        mode_notes.append(f"rrf_k override: {cfg.retrieval.rrf_k}")
    if args.rrf_overlap_boost > 0:
        cfg.retrieval.rrf_overlap_boost = float(args.rrf_overlap_boost)
        mode_notes.append(f"rrf_overlap_boost override: {cfg.retrieval.rrf_overlap_boost}")

    tmp_dir: str | None = None
    if args.data_dir:
        cfg.data_dir = Path(args.data_dir)
    else:
        tmp_dir = tempfile.mkdtemp(prefix="novaspine-qa-")
        cfg.data_dir = Path(tmp_dir)
    cfg.ensure_dirs()
    spine = MemorySpine(cfg)
    llm_backend: Any | None = None
    try:
        if args.answer_mode == "llm":
            _validate_llm_provider(args.answer_provider, cfg)
            llm_kwargs: dict[str, Any] = {
                "model": args.answer_model,
                "temperature": args.answer_temperature,
                "max_tokens": args.answer_max_tokens,
            }
            if args.answer_provider == "venice":
                llm_kwargs["api_key"] = cfg.venice.api_key or os.environ.get("VENICE_API_KEY", "")
            llm_backend = create_chat_backend(provider=args.answer_provider, **llm_kwargs)
            mode_notes.append(
                f"answer_llm: provider={args.answer_provider}, model={args.answer_model}, temp={args.answer_temperature}"
            )
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
            elif args.answer_mode == "llm":
                pred = await _answer_with_llm(
                    query=query,
                    recalled=recalled,
                    llm_backend=llm_backend,
                    context_k=args.answer_context_k,
                    per_chunk_chars=args.answer_chunk_chars,
                    total_chars=args.answer_max_context_chars,
                    temperature=args.answer_temperature,
                    max_tokens=args.answer_max_tokens,
                )
                if not pred and not args.no_answer_fallback:
                    pred = extractive_answer(query, recalled)
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
            "answer_provider": args.answer_provider if args.answer_mode == "llm" else "",
            "answer_model": args.answer_model if args.answer_mode == "llm" else "",
            "samples": sample_logs,
        }
    finally:
        if llm_backend is not None:
            await llm_backend.close()
        await spine.close()


def main() -> None:
    args = _parse_args()
    try:
        result = asyncio.run(_run(args))
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)}, indent=2))
        raise SystemExit(2)
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
