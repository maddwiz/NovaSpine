"""FAISS vector search."""

from __future__ import annotations

import numpy as np

from c3ae.storage.faiss_store import FAISSStore
from c3ae.storage.sqlite_store import SQLiteStore
from c3ae.types import SearchResult


class VectorSearch:
    """Vector similarity search using FAISS."""

    def __init__(self, faiss_store: FAISSStore, sqlite_store: SQLiteStore) -> None:
        self.faiss = faiss_store
        self.sqlite = sqlite_store

    def search(self, query_vector: np.ndarray, top_k: int = 20) -> list[SearchResult]:
        """Search FAISS index, resolve chunk content from SQLite."""
        hits = self.faiss.search(query_vector, top_k=top_k)
        results = []
        for ext_id, score in hits:
            chunk = self.sqlite.get_chunk(ext_id)
            if chunk:
                metadata = {
                    **(chunk.metadata or {}),
                    "_created_at": chunk.created_at.isoformat(),
                    "_source_kind": "chunk",
                }
                results.append(SearchResult(
                    id=chunk.id,
                    content=chunk.content,
                    score=float(score),
                    source="vector",
                    metadata=metadata,
                ))
        return results

    def index_chunk(self, chunk_id: str, vector: np.ndarray) -> int:
        """Add a chunk vector to the FAISS index."""
        return self.faiss.add(vector, chunk_id)

    def index_batch(self, chunk_ids: list[str], vectors: np.ndarray) -> list[int]:
        return self.faiss.add_batch(vectors, chunk_ids)
