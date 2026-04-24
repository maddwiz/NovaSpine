"""Shared utilities."""

from __future__ import annotations

import json
import hashlib
import re
from datetime import datetime, timezone
from typing import Any

import orjson

BENCH_CASE_TOKEN_RE = re.compile(r"__\w+_CASE_\d+__", re.IGNORECASE)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def json_dumps(obj: Any) -> str:
    return orjson.dumps(obj).decode()


def json_loads(data: str | bytes) -> Any:
    return orjson.loads(data)


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def chunk_text(text: str, max_chars: int = 2000, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks."""
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        # Try to break at a paragraph or sentence boundary
        if end < len(text):
            for sep in ["\n\n", "\n", ". ", " "]:
                idx = text.rfind(sep, start + max_chars // 2, end)
                if idx != -1:
                    end = idx + len(sep)
                    break
        chunks.append(text[start:end].strip())
        start = end - overlap
    return [c for c in chunks if c]


def estimate_text_tokens(text: str) -> int:
    """Best-effort token estimate without a model-specific tokenizer."""
    if not text:
        return 0
    stripped = text.strip()
    if not stripped:
        return 0
    non_ws_chars = len(re.sub(r"\s+", "", stripped))
    lexical_tokens = len(re.findall(r"\w+|[^\w\s]", stripped, flags=re.UNICODE))
    return max((non_ws_chars + 3) // 4, lexical_tokens)


def chunk_text_for_embedding(
    text: str,
    *,
    max_tokens: int = 384,
    overlap_tokens: int = 48,
    min_chars: int = 256,
) -> list[str]:
    """Recursively split text until chunks are small enough for embedding APIs."""
    stripped = (text or "").strip()
    if not stripped:
        return []
    if max_tokens <= 0:
        return [stripped]

    out: list[str] = []
    queue: list[str] = [stripped]
    while queue:
        current = queue.pop(0).strip()
        if not current:
            continue
        est_tokens = estimate_text_tokens(current)
        if est_tokens <= max_tokens or len(current) <= min_chars:
            out.append(current)
            continue

        chars_per_token = max(1.0, len(current) / max(1, est_tokens))
        max_chars = max(min_chars, int(max_tokens * chars_per_token))
        overlap_chars = max(32, int(overlap_tokens * chars_per_token))
        overlap_chars = min(overlap_chars, max(32, max_chars // 4))
        parts = chunk_text(current, max_chars=max_chars, overlap=overlap_chars)
        if len(parts) <= 1:
            midpoint = max(1, len(current) // 2)
            parts = [current[:midpoint].strip(), current[midpoint:].strip()]
        if len(parts) <= 1 or parts == [current]:
            out.append(current)
            continue
        queue = parts + queue
    return out


def iso_str(dt: datetime) -> str:
    return dt.isoformat()


def parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s)


def strip_benchmark_case_tokens(text: str) -> str:
    return re.sub(r"\s+", " ", BENCH_CASE_TOKEN_RE.sub(" ", text or "")).strip()


def extract_benchmark_case_token(raw: str) -> str:
    if not raw:
        return ""
    m = BENCH_CASE_TOKEN_RE.search(raw)
    return m.group(0).upper() if m else ""


def parse_json_object(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        return {}
    if "```json" in text:
        m = re.search(r"```json\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            text = m.group(1).strip()
    elif text.startswith("```"):
        m = re.search(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)
        if m:
            text = m.group(1).strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        pass
    # Best-effort extraction of a JSON object payload embedded in text.
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return {}
    try:
        data = json.loads(m.group(0))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}
