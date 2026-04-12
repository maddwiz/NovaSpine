"""SQLite-backed embedding cache."""

from __future__ import annotations

import hashlib

import numpy as np

from c3ae.storage.sqlite_store import SQLiteStore


class EmbeddingCache:
    """Cache embeddings to avoid redundant API calls."""

    def __init__(self, store: SQLiteStore, model: str = "text-embedding-bge-m3") -> None:
        self.store = store
        self.model = model

    def _hash(self, text: str) -> str:
        payload = f"{self.model}\0{text}"
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, text: str) -> np.ndarray | None:
        h = self._hash(text)
        blob = self.store.get_cached_embedding(h)
        if blob is None:
            return None
        return np.frombuffer(blob, dtype=np.float32).copy()

    def put(self, text: str, embedding: np.ndarray) -> None:
        h = self._hash(text)
        self.store.cache_embedding(h, embedding.astype(np.float32).tobytes(), self.model)

    def get_batch(self, texts: list[str]) -> tuple[list[np.ndarray | None], list[int]]:
        """Returns (results, miss_indices) where results[i] is None for cache misses."""
        results: list[np.ndarray | None] = []
        miss_indices: list[int] = []
        for i, text in enumerate(texts):
            vec = self.get(text)
            results.append(vec)
            if vec is None:
                miss_indices.append(i)
        return results, miss_indices

    def put_batch(self, texts: list[str], embeddings: np.ndarray) -> None:
        for text, vec in zip(texts, embeddings):
            self.put(text, vec)
