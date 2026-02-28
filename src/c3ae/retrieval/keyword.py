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
        """Search across chunks + reasoning bank with per-source rank normalization."""
        fetch = max(limit * 3, limit)
        groups = [
            self.search_chunks(query, limit=fetch),
            self.search_reasoning(query, limit=fetch),
            self.search_consolidated(query, limit=fetch),
        ]
        combined: list[SearchResult] = []
        for rows in groups:
            combined.extend(self._rank_normalize(rows))
        combined.sort(key=lambda r: r.score, reverse=True)
        out: list[SearchResult] = []
        seen: set[str] = set()
        for row in combined:
            if row.id in seen:
                continue
            seen.add(row.id)
            out.append(row)
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _rank_normalize(rows: list[SearchResult]) -> list[SearchResult]:
        if not rows:
            return []
        ordered = sorted(rows, key=lambda r: float(r.score), reverse=True)
        out: list[SearchResult] = []
        for rank, r in enumerate(ordered, start=1):
            out.append(
                SearchResult(
                    id=r.id,
                    content=r.content,
                    # Rank-based normalization avoids inflating weak singleton matches.
                    score=1.0 / float(rank),
                    source=r.source,
                    metadata=r.metadata,
                )
            )
        return out
