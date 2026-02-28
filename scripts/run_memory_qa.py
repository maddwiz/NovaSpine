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
import random
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from c3ae.config import Config
from c3ae.eval.corpus import build_corpus_docs, load_jsonl
from c3ae.eval import best_exact_match, best_f1, extractive_answer
from c3ae.llm import Message, create_chat_backend
from c3ae.memory_spine.spine import MemorySpine
from c3ae.utils import extract_benchmark_case_token, parse_json_object, strip_benchmark_case_tokens

_QUERY_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "of", "in", "on", "for",
    "and", "or", "with", "what", "when", "who", "where", "which", "did", "do",
    "does", "at", "by", "as", "it", "be", "this", "that", "how", "many",
    "much", "kind", "type", "into", "about",
}
_KEYWORD_KEEP = {
    "to",
    "from",
    "before",
    "after",
    "between",
    "during",
    "until",
    "till",
    "since",
    "times",
    "time",
    "days",
    "months",
    "years",
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run NovaSpine end-to-end QA benchmark.")
    p.add_argument("--dataset", required=True, help="QA eval JSONL path")
    p.add_argument("--row-offset", type=int, default=0, help="Start evaluating at this row index.")
    p.add_argument("--row-limit", type=int, default=0, help="Evaluate at most this many rows (0 = all).")
    p.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Emit progress to stderr every N scored rows (0 disables).",
    )
    p.add_argument("--corpus", default="", help="Optional corpus JSONL to ingest before eval")
    p.add_argument("--name", default="qa_custom", help="Benchmark run name")
    p.add_argument("--top-k", type=int, default=10, help="Recall depth")
    p.add_argument("--data-dir", default="", help="Optional persistent data dir")
    p.add_argument("--ingest-sync", action="store_true", help="Use sync ingest path")
    p.add_argument(
        "--ingest-batch-size",
        type=int,
        default=64,
        help="Batch size for async bulk ingestion (ignored with --ingest-sync).",
    )
    p.add_argument("--skip-chunking", action="store_true", help="Ingest docs as single chunks")
    p.add_argument(
        "--reuse-index",
        action="store_true",
        help="Reuse existing index in --data-dir when chunks already exist (skip ingestion).",
    )
    p.add_argument("--embed-local", action="store_true", help="Use local hash embeddings")
    p.add_argument(
        "--embed-provider",
        default="",
        choices=["", "venice", "openai", "ollama", "hash", "localhash", "sbert"],
        help="Override embedding provider (defaults to config/env).",
    )
    p.add_argument("--embed-model", default="", help="Optional embedding model override")
    p.add_argument("--embed-dims", type=int, default=0, help="Optional embedding dims override")
    p.add_argument(
        "--dmr-openai-large",
        action="store_true",
        help="For DMR datasets, default to text-embedding-3-large (3072 dims) when using OpenAI.",
    )
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
        "--query-expansion",
        action="store_true",
        help="Enable keyword query expansion in hybrid retrieval.",
    )
    p.add_argument(
        "--recall-variants",
        action="store_true",
        help="Enable multi-variant recall ensemble (off by default).",
    )
    p.add_argument(
        "--no-recall-variants",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--variant-fetch-multiplier",
        type=int,
        default=3,
        help="Per-variant fetch multiplier for ensemble recall.",
    )
    p.add_argument(
        "--variant-rrf-k",
        type=int,
        default=30,
        help="RRF k constant for variant merge.",
    )
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
    p.add_argument(
        "--answer-model-hard",
        default=os.environ.get("C3AE_QA_LLM_MODEL_HARD", ""),
        help="Optional stronger model for DATE/NUMBER questions.",
    )
    p.add_argument(
        "--answer-route-hard",
        default="auto",
        choices=["off", "auto", "on"],
        help="Route DATE/NUMBER questions to --answer-model-hard (or derived non-mini model).",
    )
    p.add_argument("--answer-temperature", type=float, default=0.0, help="LLM answer temperature")
    p.add_argument(
        "--answer-timeout-seconds",
        type=float,
        default=35.0,
        help="HTTP timeout for answer LLM calls.",
    )
    p.add_argument("--answer-max-tokens", type=int, default=192, help="LLM answer max tokens")
    p.add_argument(
        "--answer-context-k",
        type=int,
        default=5,
        help="Number of recalled chunks to provide to answer LLM",
    )
    p.add_argument(
        "--answer-max-context-chars",
        type=int,
        default=9000,
        help="Global max context chars provided to answer LLM",
    )
    p.add_argument(
        "--answer-chunk-chars",
        type=int,
        default=1200,
        help="Per-chunk truncation length for answer context",
    )
    p.add_argument(
        "--answer-min-score-ratio",
        type=float,
        default=0.3,
        help="Minimum score ratio vs top score for context inclusion (except top-1).",
    )
    p.add_argument(
        "--answer-min-score-abs",
        type=float,
        default=0.0,
        help="Absolute minimum score for context inclusion (except top-1).",
    )
    p.add_argument(
        "--answer-normalize-mode",
        default="typed",
        choices=["typed", "legacy"],
        help="Answer normalization mode (typed improves EM on short factual answers).",
    )
    p.add_argument(
        "--answer-context-rerank",
        default="none",
        choices=["none", "lexical"],
        help="Re-rank answer-context candidates before truncation.",
    )
    p.add_argument(
        "--answer-context-pool-multiplier",
        type=int,
        default=3,
        help="Candidate pool multiplier for answer-context reranking.",
    )
    p.add_argument(
        "--answer-context-overlap-weight",
        type=float,
        default=0.35,
        help="Weight of query-content lexical overlap in answer-context reranking.",
    )
    p.add_argument("--answer-retries", type=int, default=3, help="Retry attempts for answer LLM calls.")
    p.add_argument(
        "--answer-retry-base-seconds",
        type=float,
        default=0.8,
        help="Base delay for exponential retry backoff.",
    )
    p.add_argument(
        "--answer-min-interval-seconds",
        type=float,
        default=0.0,
        help="Minimum wall-clock delay between LLM answer calls (rate-limit control).",
    )
    p.add_argument(
        "--answer-reasoning",
        default="auto",
        choices=["off", "on", "auto"],
        help="Reasoning mode for answer generation.",
    )
    p.add_argument(
        "--rerank-with-llm",
        action="store_true",
        help="Use LLM to re-rank retrieved candidates before answering.",
    )
    p.add_argument(
        "--rerank-candidates",
        type=int,
        default=10,
        help="Number of candidates passed to LLM reranker.",
    )
    p.add_argument(
        "--rerank-top-n",
        type=int,
        default=0,
        help="Keep top-N after reranking (0 keeps all candidates).",
    )
    p.add_argument(
        "--diversify-sessions",
        action="store_true",
        help="Promote session diversity in top-k recall (helpful for LoCoMo).",
    )
    p.add_argument(
        "--no-diversify-sessions",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--min-session-diversity",
        type=int,
        default=3,
        help="Minimum unique sessions to prioritize when diversity is enabled.",
    )
    p.add_argument(
        "--no-answer-fallback",
        action="store_true",
        help="Disable extractive fallback when LLM answer is empty/invalid.",
    )
    p.add_argument("--out", default="", help="Optional output JSON path")
    return p.parse_args()


def _clean_question(query: str) -> str:
    return strip_benchmark_case_tokens(query)


def _extract_case_token(query: str) -> str:
    return extract_benchmark_case_token(query)


def _strip_leading_wh(clean_query: str) -> str:
    tokens = clean_query.split()
    if not tokens:
        return clean_query
    first = tokens[0].lower()
    if first in {"what", "when", "where", "who", "which", "how"} and len(tokens) > 2:
        return " ".join(tokens[1:])
    return clean_query


def _query_keywords(clean_query: str, limit: int = 10) -> str:
    toks = [t.lower() for t in re.findall(r"[a-z0-9_]+", clean_query)]
    keep = [t for t in toks if len(t) > 1 and (t in _KEYWORD_KEEP or t not in _QUERY_STOPWORDS)]
    if not keep:
        return ""
    return " ".join(keep[:limit])


def _build_recall_variants(query: str) -> list[tuple[str, float]]:
    clean = _clean_question(query)
    case = _extract_case_token(query)
    variants: list[tuple[str, float]] = []

    def _add(q: str, w: float) -> None:
        qn = re.sub(r"\s+", " ", q).strip()
        if qn and all(v[0] != qn for v in variants):
            variants.append((qn, w))

    _add(query, 1.0)
    if clean and clean != query:
        _add(clean, 0.75)
    stripped = _strip_leading_wh(clean)
    if stripped and stripped != clean:
        _add(stripped, 0.65)
    kws = _query_keywords(clean)
    if kws:
        _add(kws, 0.60)
    if case:
        _add(case, 0.50)
        if kws:
            _add(f"{case} {kws}", 0.85)
        elif clean:
            _add(f"{case} {clean}", 0.85)
    return variants


def _result_key(row: dict[str, Any]) -> str:
    rid = str(row.get("id", "")).strip()
    if rid:
        return rid
    md = row.get("metadata") or {}
    doc = str(md.get("benchmark_doc_id", "")).strip()
    if doc:
        return f"doc:{doc}"
    content = str(row.get("content", "")).strip().lower()
    return f"content:{content[:120]}"


def _dedupe_by_doc_id(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen_doc: set[str] = set()
    seen_key: set[str] = set()
    for r in rows:
        md = r.get("metadata") or {}
        doc_id = str(md.get("benchmark_doc_id", "")).strip()
        key = _result_key(r)
        if doc_id:
            if doc_id in seen_doc:
                continue
            seen_doc.add(doc_id)
        if key in seen_key:
            continue
        seen_key.add(key)
        out.append(r)
    return out


def _merge_variant_recall(
    variant_results: list[tuple[float, list[dict[str, Any]]]],
    *,
    top_k: int,
    rrf_k: int,
) -> list[dict[str, Any]]:
    if not variant_results:
        return []
    scores: dict[str, float] = {}
    best_row: dict[str, dict[str, Any]] = {}
    for weight, rows in variant_results:
        for rank, row in enumerate(rows):
            key = _result_key(row)
            score = float(weight) / float(max(1, rrf_k) + rank + 1)
            scores[key] = scores.get(key, 0.0) + score
            current = best_row.get(key)
            if current is None or float(row.get("score", 0.0)) > float(current.get("score", 0.0)):
                best_row[key] = row
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    merged: list[dict[str, Any]] = []
    for key, merged_score in ranked:
        row = dict(best_row[key])
        row["score"] = float(merged_score)
        merged.append(row)
    merged = _dedupe_by_doc_id(merged)
    return merged[: max(1, top_k)]


def _clean_answer(ans: str) -> str:
    a = re.sub(r"\s+", " ", ans.strip())
    a = a.strip(" \"'")
    if a.lower().startswith("answer:"):
        a = a.split(":", 1)[1].strip()
    # Prefer canonical terse forms for benchmark scoring.
    lower = a.lower()
    if lower in {"once", "twice", "thrice"}:
        return {"once": "1", "twice": "2", "thrice": "3"}[lower]
    word_to_num = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
        "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11",
        "twelve": "12", "thirteen": "13", "fourteen": "14", "fifteen": "15", "sixteen": "16",
        "seventeen": "17", "eighteen": "18", "nineteen": "19", "twenty": "20",
    }
    m = re.match(r"^(?P<num>[a-z]+)\s+(?P<unit>day|days|week|weeks|month|months|year|years|episode|episodes)$", lower)
    if m and m.group("num") in word_to_num:
        return f"{word_to_num[m.group('num')]} {m.group('unit')}"
    if lower in word_to_num:
        return word_to_num[lower]
    return a


def _normalize_answer_by_type(answer: str, answer_type: str) -> str:
    a = (answer or "").strip()
    if not a:
        return ""
    # Normalize obvious wrappers before type-specific extraction.
    a = re.sub(r"^\s*(?:answer|final answer)\s*:\s*", "", a, flags=re.IGNORECASE).strip()

    if answer_type == "YES_NO":
        m = re.search(r"\b(yes|no)\b", a, re.IGNORECASE)
        return m.group(1).lower() if m else _clean_answer(a)

    if answer_type == "NUMBER":
        for pat in (
            r"\b\d+(?:\.\d+)?\s*(?:days?|weeks?|months?|years?|times?|episodes?|children|kids|hours?|minutes?)\b",
            r"\b\d+(?:\.\d+)?\b",
        ):
            m = re.search(pat, a, re.IGNORECASE)
            if m:
                return _clean_answer(m.group(0))
        word_num = re.search(
            r"\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
            r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|once|twice|thrice)\b",
            a,
            re.IGNORECASE,
        )
        if word_num:
            return _clean_answer(word_num.group(0))
        return _clean_answer(a)

    if answer_type == "DATE":
        month = (
            r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
            r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
        )
        for pat in (
            rf"\b{month}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,\s*\d{{4}})?\b",
            r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b",
            r"\b\d{4}-\d{2}-\d{2}\b",
            r"\b\d{4}\b",
        ):
            m = re.search(pat, a, re.IGNORECASE)
            if m:
                return _clean_answer(m.group(0))
        return _clean_answer(a)

    # Most benchmark answers are short phrases. Keep first clause and cap length.
    clause = re.split(r"[\n\r;.!?]", a, maxsplit=1)[0].strip(" \"'")
    toks = clause.split()
    if len(toks) > 8:
        clause = " ".join(toks[:8])
    if answer_type == "PLACE":
        clause = re.sub(r"^\s*(?:in|at|on|from)\s+", "", clause, flags=re.IGNORECASE)
    return _clean_answer(clause)


def _detect_answer_type(query: str) -> str:
    q = _clean_question(query).lower().strip()
    if q.startswith(("who ", "whose ")):
        return "PERSON_NAME"
    if q.startswith("when ") or "what date" in q or "what year" in q:
        return "DATE"
    if q.startswith("where "):
        return "PLACE"
    if "how many" in q or "how much" in q:
        return "NUMBER"
    if _is_binary_choice_question(q):
        return "CHOICE"
    if q.startswith(("is ", "are ", "was ", "were ", "did ", "does ", "do ", "has ", "can ")):
        return "YES_NO"
    return "SHORT_PHRASE"


_TYPE_HINTS = {
    "PERSON_NAME": "Answer with just the person's name.",
    "DATE": "Answer with just the date, year, or time span.",
    "PLACE": "Answer with just the place/location name.",
    "NUMBER": "Answer with just the number and unit.",
    "CHOICE": "Answer with one of the provided options only.",
    "YES_NO": "Answer yes or no only.",
    "SHORT_PHRASE": "Answer with the shortest phrase possible (1-6 words).",
}


def _is_binary_choice_question(q: str) -> bool:
    if " or " not in q:
        return False
    if re.search(r"[\"']([^\"']+)[\"']\s+or\s+[\"']([^\"']+)[\"']", q):
        return True
    if re.match(r"^(is|are|was|were|did|does|do|has|have|can|could|should|would)\b.*\bor\b", q):
        return True
    if re.match(r"^which\b.*\bor\b", q):
        # Avoid broad non-binary "first or second step" style phrasing.
        if re.search(r"\b(first|second|third|next|previous|earlier|later)\b", q):
            return False
        return True
    return False


def _few_shot_examples() -> str:
    return (
        "Examples:\n"
        "Q: How many days between the workshop on March 5 and the meeting on March 12?\n"
        "Context: [1] ... workshop on March 5 ... [2] ... team meeting on March 12 ...\n"
        "A: {\"answer\":\"7 days\"}\n\n"
        "Q: Which device did I set up first, the thermostat or the router?\n"
        "Context: [1] ... Monday installed the smart thermostat ... [2] ... Wednesday set up the router ...\n"
        "A: {\"answer\":\"smart thermostat\"}\n\n"
        "Q: What color is Sarah's car?\n"
        "Context: [1] ... Sarah just bought a red Honda Civic ...\n"
        "A: {\"answer\":\"red\"}"
    )


def _derive_hard_model(base_model: str, explicit_hard_model: str) -> str:
    if explicit_hard_model.strip():
        return explicit_hard_model.strip()
    model = base_model.strip()
    if not model:
        return ""
    if "-mini" in model:
        return model.replace("-mini", "")
    if model.endswith("mini"):
        return model[:-4].rstrip("-_")
    return ""


def _build_answer_context(
    query: str,
    recalled: list[dict[str, Any]],
    k: int,
    per_chunk_chars: int,
    total_chars: int,
    min_score_ratio: float,
    min_score_abs: float,
    context_rerank: str = "none",
    context_pool_multiplier: int = 3,
    context_overlap_weight: float = 0.35,
) -> str:
    chunks: list[str] = []
    used = 0
    if not recalled:
        return ""
    rows = recalled[: max(1, k)]
    if context_rerank == "lexical":
        pool = max(1, int(context_pool_multiplier))
        pool_rows = recalled[: max(1, k * pool)]
        q_toks = set(re.findall(r"[a-z0-9_]+", _clean_question(query).lower()))
        q_toks = {t for t in q_toks if len(t) > 1 and t not in _QUERY_STOPWORDS}
        if q_toks and pool_rows:
            top_raw = max(float(r.get("score", 0.0)) for r in pool_rows)
            top_raw = top_raw if top_raw > 0.0 else 1.0
            w = min(1.0, max(0.0, float(context_overlap_weight)))
            scored_rows: list[tuple[float, int, dict[str, Any]]] = []
            for idx, row in enumerate(pool_rows):
                content = str(row.get("content", "")).lower()
                s_toks = set(re.findall(r"[a-z0-9_]+", content))
                overlap = (len(q_toks & s_toks) / max(1, len(q_toks))) if s_toks else 0.0
                base = float(row.get("score", 0.0)) / top_raw
                fused = (1.0 - w) * base + w * overlap
                scored_rows.append((fused, -idx, row))
            scored_rows.sort(reverse=True)
            rows = [row for _, _, row in scored_rows[: max(1, k)]]

    top_score = max(float(r.get("score", 0.0)) for r in rows) if rows else 0.0
    min_score = max(float(min_score_abs), top_score * max(0.0, float(min_score_ratio)))
    for i, row in enumerate(rows):
        score = float(row.get("score", 0.0))
        if i > 0 and score < min_score:
            continue
        content = str(row.get("content", "")).strip()
        if not content:
            continue
        if len(content) > per_chunk_chars:
            content = content[:per_chunk_chars].rstrip() + " ..."
        md = row.get("metadata") or {}
        doc_id = str(md.get("benchmark_doc_id", "")).strip() or str(row.get("id", ""))
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
    min_score_ratio: float,
    min_score_abs: float,
    context_rerank: str,
    context_pool_multiplier: int,
    context_overlap_weight: float,
    normalize_mode: str,
    temperature: float,
    max_tokens: int,
    retries: int,
    retry_base_seconds: float,
    reasoning_mode: str,
    llm_backend_hard: Any | None = None,
    answer_route_hard: str = "auto",
) -> str:
    clean_q = _clean_question(query)
    context = _build_answer_context(
        clean_q,
        recalled,
        context_k,
        per_chunk_chars,
        total_chars,
        min_score_ratio=min_score_ratio,
        min_score_abs=min_score_abs,
        context_rerank=context_rerank,
        context_pool_multiplier=context_pool_multiplier,
        context_overlap_weight=context_overlap_weight,
    )
    if not context:
        return ""
    answer_type = _detect_answer_type(clean_q)
    use_reasoning = reasoning_mode == "on" or (
        reasoning_mode == "auto" and answer_type in {"NUMBER", "DATE", "CHOICE"}
    )
    use_hard_backend = (
        llm_backend_hard is not None
        and answer_type in {"NUMBER", "DATE"}
        and answer_route_hard != "off"
    )
    backend = llm_backend_hard if use_hard_backend else llm_backend
    response_schema = "{\"reasoning\":\"...\", \"answer\":\"...\"}" if use_reasoning else "{\"answer\":\"...\"}"
    system = (
        "You are a strict QA answerer. Use only the provided context.\n"
        f"Return JSON only: {response_schema}.\n"
        "Rules:\n"
        "- answer with the shortest exact phrase possible\n"
        "- prefer canonical forms: numerals (2 not twice), concise entities, no leading articles\n"
        "- keep units when present in context (for example: '7 days', '5 months')\n"
        f"- expected answer type: {answer_type} ({_TYPE_HINTS[answer_type]})\n"
        "- no explanation, no extra keys, no markdown\n"
        "- if uncertain, return an empty answer"
    )
    system += "\n\n" + _few_shot_examples()
    if use_reasoning:
        system += "\n- include concise reasoning (<= 40 words) before final answer"
    user = (
        f"Question:\n{clean_q}\n\n"
        f"Context:\n{context}\n\n"
        "Return JSON now."
    )
    last_exc: Exception | None = None
    max_tries = max(1, int(retries))
    for attempt in range(max_tries):
        try:
            resp = await backend.chat(
                [Message(role="system", content=system), Message(role="user", content=user)],
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=True,
            )
            obj = parse_json_object(resp.content)
            if obj:
                raw = str(obj.get("answer", ""))
                ans = (
                    _normalize_answer_by_type(raw, answer_type)
                    if normalize_mode == "typed"
                    else _clean_answer(raw)
                )
                if ans:
                    return ans
                return ""
            if normalize_mode == "typed":
                return _normalize_answer_by_type(resp.content, answer_type)
            return _clean_answer(resp.content)
        except Exception as e:
            last_exc = e
            if attempt >= max_tries - 1:
                break
            delay = float(retry_base_seconds) * (2**attempt) + random.uniform(0.0, 0.25)
            await asyncio.sleep(delay)
    if last_exc is not None:
        # Avoid aborting an entire benchmark run on transient provider rate-limits.
        return ""
    return ""


async def _rerank_with_llm(
    *,
    query: str,
    recalled: list[dict[str, Any]],
    llm_backend: Any,
    candidates: int,
    top_n: int,
    temperature: float,
    max_tokens: int,
) -> list[dict[str, Any]]:
    if not recalled:
        return []
    pool = recalled[: max(1, int(candidates))]
    prompt_lines = []
    for i, row in enumerate(pool, start=1):
        text = str(row.get("content", "")).strip().replace("\n", " ")
        if len(text) > 220:
            text = text[:220].rstrip() + " ..."
        prompt_lines.append(f"[{i}] {text}")
    system = (
        "Rank passages by relevance to the question. "
        "Return strict JSON: {\"ranking\": [indices in best-to-worst order]}."
    )
    user = (
        f"Question:\n{_clean_question(query)}\n\n"
        "Passages:\n"
        f"{chr(10).join(prompt_lines)}\n\n"
        "Return JSON now."
    )
    resp = await llm_backend.chat(
        [Message(role="system", content=system), Message(role="user", content=user)],
        temperature=temperature,
        max_tokens=max_tokens,
        json_mode=True,
    )
    parsed = parse_json_object(resp.content)
    ranking = parsed.get("ranking", []) if isinstance(parsed, dict) else []
    order: list[int] = []
    if isinstance(ranking, list):
        for x in ranking:
            try:
                idx = int(x)
            except Exception:
                continue
            if 1 <= idx <= len(pool) and idx not in order:
                order.append(idx)
    reordered = [pool[i - 1] for i in order]
    seen_ids = {_result_key(r) for r in reordered}
    for row in pool:
        key = _result_key(row)
        if key not in seen_ids:
            reordered.append(row)
            seen_ids.add(key)
    tail = [r for r in recalled if _result_key(r) not in seen_ids]
    out = reordered + tail
    keep = int(top_n) if int(top_n) > 0 else len(out)
    return out[:keep]


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


def _diversify_by_session(
    recalled: list[dict[str, Any]],
    *,
    top_k: int,
    min_sessions: int,
) -> list[dict[str, Any]]:
    if not recalled or min_sessions <= 1:
        return recalled[: max(1, top_k)]
    session_picks: list[dict[str, Any]] = []
    rest: list[dict[str, Any]] = []
    seen_sessions: set[str] = set()
    seen_rows: set[str] = set()
    for row in recalled:
        row_key = _result_key(row)
        if row_key in seen_rows:
            continue
        seen_rows.add(row_key)
        meta = row.get("metadata") or {}
        sid = str(meta.get("session_id", "")).strip()
        if sid and sid not in seen_sessions and len(seen_sessions) < min_sessions:
            seen_sessions.add(sid)
            session_picks.append(row)
        else:
            rest.append(row)
    merged = session_picks + rest
    return merged[: max(1, top_k)]


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    cfg = Config()
    mode_notes: list[str] = []
    dataset_name = str(args.dataset).lower()
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
    if args.dmr_openai_large and "dmr" in dataset_name and cfg.venice.embedding_provider == "openai":
        if not args.embed_model:
            cfg.venice.embedding_model = "text-embedding-3-large"
            mode_notes.append("dmr_openai_large: embed_model=text-embedding-3-large")
        if args.embed_dims <= 0:
            cfg.venice.embedding_dims = 3072
            mode_notes.append("dmr_openai_large: embed_dims=3072")
    if args.disable_graph:
        cfg.graph.enabled = False
        mode_notes.append("graph disabled")
    if args.query_expansion:
        cfg.retrieval.enable_query_expansion = True
        mode_notes.append("query_expansion=on")
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
    use_recall_variants = bool(args.recall_variants) and not bool(args.no_recall_variants)
    mode_notes.append(f"recall_variants={'on' if use_recall_variants else 'off'}")
    auto_session_diversity = (
        bool(args.diversify_sessions)
        or ("locomo" in dataset_name and not bool(args.no_diversify_sessions))
    )
    if auto_session_diversity:
        mode_notes.append(
            "session_diversity=on"
            f" min_sessions={max(1, int(args.min_session_diversity))}"
        )

    tmp_dir: str | None = None
    if args.data_dir:
        cfg.data_dir = Path(args.data_dir)
    else:
        tmp_dir = tempfile.mkdtemp(prefix="novaspine-qa-")
        cfg.data_dir = Path(tmp_dir)
    cfg.ensure_dirs()
    spine = MemorySpine(cfg)
    llm_backend: Any | None = None
    llm_backend_hard: Any | None = None
    try:
        if args.answer_mode == "llm":
            _validate_llm_provider(args.answer_provider, cfg)
            llm_kwargs: dict[str, Any] = {
                "model": args.answer_model,
                "temperature": args.answer_temperature,
                "timeout": args.answer_timeout_seconds,
                "max_tokens": args.answer_max_tokens,
            }
            if args.answer_provider == "venice":
                llm_kwargs["api_key"] = cfg.venice.api_key or os.environ.get("VENICE_API_KEY", "")
            llm_backend = create_chat_backend(provider=args.answer_provider, **llm_kwargs)
            mode_notes.append(
                f"answer_llm: provider={args.answer_provider}, model={args.answer_model}, temp={args.answer_temperature}"
            )
            hard_model = _derive_hard_model(args.answer_model, args.answer_model_hard)
            if args.answer_route_hard != "off" and hard_model and hard_model != args.answer_model:
                hard_kwargs = dict(llm_kwargs)
                hard_kwargs["model"] = hard_model
                llm_backend_hard = create_chat_backend(provider=args.answer_provider, **hard_kwargs)
                mode_notes.append(
                    "answer_hard_route:on"
                    f" mode={args.answer_route_hard}"
                    f" hard_model={hard_model}"
                )
            else:
                mode_notes.append(f"answer_hard_route:{args.answer_route_hard}")
            if args.rerank_with_llm:
                mode_notes.append(
                    "llm_rerank:on"
                    f" candidates={max(1, int(args.rerank_candidates))}"
                    f" keep_top_n={max(0, int(args.rerank_top_n))}"
                )
        rows = load_jsonl(Path(args.dataset))
        if not rows:
            raise ValueError("dataset has no rows")
        row_offset = max(0, int(args.row_offset))
        row_limit = max(0, int(args.row_limit))
        if row_offset > 0 or row_limit > 0:
            start = row_offset
            end = row_offset + row_limit if row_limit > 0 else None
            rows = rows[start:end]
            mode_notes.append(
                f"row_slice=offset:{row_offset},limit:{row_limit if row_limit > 0 else 'all'}"
            )
        corpus_docs = build_corpus_docs(rows, load_jsonl(Path(args.corpus)) if args.corpus else [])
        use_ingest_sync = bool(args.ingest_sync) and not bool(args.embed_local)
        if args.embed_local and args.ingest_sync:
            mode_notes.append("embed_local enabled: ignoring --ingest-sync to build vector index")

        ingested_docs = 0
        ingested_chunks = 0
        existing_chunks = int(spine.sqlite.count_chunks())
        if args.reuse_index and existing_chunks > 0:
            mode_notes.append(f"reuse_index=on (existing_chunks={existing_chunks}, ingestion_skipped)")
        else:
            if args.reuse_index:
                mode_notes.append("reuse_index=on but index empty; ingesting corpus")
            if use_ingest_sync:
                for doc in corpus_docs:
                    cids = spine.ingest_text_sync(
                        doc["text"],
                        source_id=doc["source_id"],
                        metadata=doc["metadata"],
                        skip_chunking=args.skip_chunking,
                    )
                    ingested_docs += 1
                    ingested_chunks += len(cids)
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
        last_answer_ts = 0.0

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
            if use_recall_variants:
                fetch_k = max(args.top_k * max(1, int(args.variant_fetch_multiplier)), args.top_k)
                fetch_k = max(fetch_k, max(1, int(args.rerank_candidates)))
                if auto_session_diversity:
                    fetch_k = max(fetch_k, max(1, int(args.top_k)) * 4)
                variants = _build_recall_variants(query)
                variant_rows: list[tuple[float, list[dict[str, Any]]]] = []
                for variant_q, weight in variants:
                    vr = await spine.recall(variant_q, top_k=fetch_k)
                    variant_rows.append((weight, vr))
                recalled = _merge_variant_recall(
                    variant_rows,
                    top_k=args.top_k,
                    rrf_k=max(1, int(args.variant_rrf_k)),
                )
            else:
                fetch_k = max(args.top_k, max(1, int(args.rerank_candidates)))
                if auto_session_diversity:
                    fetch_k = max(fetch_k, max(1, int(args.top_k)) * 4)
                recalled = await spine.recall(query, top_k=fetch_k)
            if (
                args.answer_mode == "llm"
                and bool(args.rerank_with_llm)
                and llm_backend is not None
                and recalled
            ):
                try:
                    recalled = await _rerank_with_llm(
                        query=query,
                        recalled=recalled,
                        llm_backend=llm_backend,
                        candidates=max(1, int(args.rerank_candidates)),
                        top_n=max(0, int(args.rerank_top_n)),
                        temperature=args.answer_temperature,
                        max_tokens=max(128, min(512, int(args.answer_max_tokens))),
                    )
                except Exception:
                    pass
            if auto_session_diversity:
                recalled = _diversify_by_session(
                    recalled,
                    top_k=max(1, int(args.top_k)),
                    min_sessions=max(1, int(args.min_session_diversity)),
                )
            recalled = recalled[: max(1, int(args.top_k))]
            docs = [str((r.get("metadata") or {}).get("benchmark_doc_id", "")) for r in recalled]
            hit = any(d in expected_doc_ids for d in docs) if expected_doc_ids else False

            if args.answer_mode == "oracle_doc" and expected_doc_ids:
                pred = answers[0] if hit else extractive_answer(query, recalled)
            elif args.answer_mode == "llm":
                min_interval = max(0.0, float(args.answer_min_interval_seconds))
                if min_interval > 0.0:
                    now = time.monotonic()
                    wait = min_interval - (now - last_answer_ts)
                    if wait > 0.0:
                        await asyncio.sleep(wait)
                pred = await _answer_with_llm(
                    query=query,
                    recalled=recalled,
                    llm_backend=llm_backend,
                    context_k=args.answer_context_k,
                    per_chunk_chars=args.answer_chunk_chars,
                    total_chars=args.answer_max_context_chars,
                    min_score_ratio=args.answer_min_score_ratio,
                    min_score_abs=args.answer_min_score_abs,
                    context_rerank=args.answer_context_rerank,
                    context_pool_multiplier=args.answer_context_pool_multiplier,
                    context_overlap_weight=args.answer_context_overlap_weight,
                    normalize_mode=args.answer_normalize_mode,
                    temperature=args.answer_temperature,
                    max_tokens=args.answer_max_tokens,
                    retries=args.answer_retries,
                    retry_base_seconds=args.answer_retry_base_seconds,
                    reasoning_mode=args.answer_reasoning,
                    llm_backend_hard=llm_backend_hard,
                    answer_route_hard=args.answer_route_hard,
                )
                last_answer_ts = time.monotonic()
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
            progress_every = max(0, int(args.progress_every))
            if progress_every > 0 and n % progress_every == 0:
                print(
                    (
                        f"[qa-progress] scored={n} "
                        f"doc_hit={doc_hits / max(1, n):.4f} "
                        f"em={em_sum / max(1, n):.4f} "
                        f"f1={f1_sum / max(1, n):.4f}"
                    ),
                    file=sys.stderr,
                    flush=True,
                )

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
            "row_offset": row_offset,
            "row_limit": row_limit,
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
            "reuse_index": bool(args.reuse_index),
            "embed_local": bool(args.embed_local),
            "mode_notes": mode_notes,
            "answer_provider": args.answer_provider if args.answer_mode == "llm" else "",
            "answer_model": args.answer_model if args.answer_mode == "llm" else "",
            "samples": sample_logs,
        }
    finally:
        if llm_backend is not None:
            await llm_backend.close()
        if llm_backend_hard is not None:
            await llm_backend_hard.close()
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
