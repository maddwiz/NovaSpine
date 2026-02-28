"""Shared benchmark/QA corpus helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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


def build_corpus_docs(
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

