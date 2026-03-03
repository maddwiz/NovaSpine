"""Cross-encoder reranking helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from c3ae.types import SearchResult


@dataclass(frozen=True)
class CrossEncoderConfig:
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    max_content_chars: int = 512


class CrossEncoderReranker:
    """Rerank retrieval candidates with a cross-encoder model."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        *,
        max_content_chars: int = 512,
        model: Any | None = None,
    ) -> None:
        self.config = CrossEncoderConfig(model_name=model_name, max_content_chars=max(64, int(max_content_chars)))
        self._model = model

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except Exception as exc:  # pragma: no cover - dependency availability varies by env
            raise RuntimeError(
                "sentence-transformers is required for cross-encoder reranking. "
                "Install with: pip install -e '.[semantic]'"
            ) from exc
        self._model = CrossEncoder(self.config.model_name)
        return self._model

    def _score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        model = self._ensure_model()
        raw = model.predict(pairs)
        if hasattr(raw, "tolist"):
            raw = raw.tolist()
        return [float(x) for x in raw]

    def rerank_rows(
        self,
        query: str,
        rows: list[dict[str, Any]],
        *,
        top_n: int = 10,
    ) -> list[dict[str, Any]]:
        if not rows:
            return []
        limit = max(1, int(top_n))
        pairs = [(query, str((row.get("content") or ""))[: self.config.max_content_chars]) for row in rows]
        scores = self._score_pairs(pairs)
        ranked = sorted(
            zip(scores, rows),
            key=lambda pair: pair[0],
            reverse=True,
        )
        out: list[dict[str, Any]] = []
        for score, row in ranked[:limit]:
            copy_row = dict(row)
            copy_row["score"] = float(score)
            md = dict(copy_row.get("metadata") or {})
            md["cross_encoder_score"] = float(score)
            copy_row["metadata"] = md
            out.append(copy_row)
        return out

    def rerank_results(
        self,
        query: str,
        results: list[SearchResult],
        *,
        top_n: int = 10,
    ) -> list[SearchResult]:
        if not results:
            return []
        limit = max(1, int(top_n))
        pairs = [(query, str(r.content)[: self.config.max_content_chars]) for r in results]
        scores = self._score_pairs(pairs)
        ranked = sorted(
            zip(scores, results),
            key=lambda pair: pair[0],
            reverse=True,
        )
        out: list[SearchResult] = []
        for score, result in ranked[:limit]:
            md = dict(result.metadata or {})
            md["cross_encoder_score"] = float(score)
            out.append(
                SearchResult(
                    id=result.id,
                    content=result.content,
                    score=float(score),
                    source="cross_encoder",
                    metadata=md,
                )
            )
        return out

