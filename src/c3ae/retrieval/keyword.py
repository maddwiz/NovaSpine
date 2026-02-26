"""FTS5/BM25 keyword search."""

from __future__ import annotations

from c3ae.storage.sqlite_store import SQLiteStore
from c3ae.types import SearchResult


class KeywordSearch:
    """Keyword search using SQLite FTS5."""

    def __init__(self, store: SQLiteStore) -> None:
        self.store = store

    def search_chunks(self, query: str, limit: int = 20) -> list[SearchResult]:
        return self.store.search_chunks_fts(query, limit=limit)

    def search_reasoning(self, query: str, limit: int = 20) -> list[SearchResult]:
        return self.store.search_reasoning_fts(query, limit=limit)

    def search_skills(self, query: str, limit: int = 10) -> list[SearchResult]:
        return self.store.search_skills_fts(query, limit=limit)

    def search_consolidated(self, query: str, limit: int = 20) -> list[SearchResult]:
        return self.store.search_consolidated_fts(query, limit=limit)

    def search_all(self, query: str, limit: int = 20) -> list[SearchResult]:
        """Search across chunks + reasoning bank."""
        chunks = self.search_chunks(query, limit=limit)
        reasoning = self.search_reasoning(query, limit=limit)
        consolidated = self.search_consolidated(query, limit=limit)
        combined = chunks + reasoning + consolidated
        combined.sort(key=lambda r: r.score, reverse=True)
        return combined[:limit]
