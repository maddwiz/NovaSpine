"""Memory Spine — orchestrator for Hot/Warm/Cold tiers + retrieval + governance."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from c3ae.config import Config
from c3ae.cos.cos import COSManager
from c3ae.embeddings.cache import EmbeddingCache
from c3ae.embeddings.factory import build_embedder
from c3ae.exceptions import GovernanceError
from c3ae.governance.audit import AuditLog
from c3ae.governance.guardian import Guardian
from c3ae.reasoning_bank.bank import ReasoningBank
from c3ae.reasoning_bank.evidence import EvidenceManager
from c3ae.retrieval.hybrid import HybridSearch
from c3ae.retrieval.keyword import KeywordSearch
from c3ae.retrieval.vector import VectorSearch
from c3ae.skill_capsules.registry import SkillRegistry
from c3ae.storage.faiss_store import FAISSStore
from c3ae.storage.sqlite_store import SQLiteStore
from c3ae.storage.vault import Vault
from c3ae.types import (
    Chunk,
    EvidencePack,
    ReasoningEntry,
    SearchResult,
    SkillCapsule,
)
from c3ae.utils import chunk_text


class MemorySpine:
    """Central orchestrator wiring all memory subsystems."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self.config.ensure_dirs()

        # Storage backends
        self.sqlite = SQLiteStore(self.config.db_path)
        self.embedder = build_embedder(self.config)
        self.faiss = FAISSStore(
            dims=self.embedder.dimensions,
            faiss_dir=self.config.faiss_dir,
            ivf_threshold=self.config.retrieval.faiss_ivf_threshold,
        )
        self.vault = Vault(self.config.vault_dir)

        # Subsystems
        self.embed_cache = EmbeddingCache(self.sqlite, self.embedder.model_name)
        self.keyword_search = KeywordSearch(self.sqlite)
        self.vector_search = VectorSearch(self.faiss, self.sqlite)
        self.hybrid_search = HybridSearch(
            self.keyword_search, self.vector_search, self.config.retrieval
        )
        self.cos = COSManager(self.sqlite)
        self.bank = ReasoningBank(self.sqlite)
        self.evidence = EvidenceManager(self.sqlite)
        self.skills = SkillRegistry(self.sqlite)
        self.guardian = Guardian(
            self.sqlite, self.config.governance,
            faiss_store=self.faiss,
            embedder=self.embedder,
            embed_cache=self.embed_cache,
        )
        self.audit = AuditLog(self.sqlite)

        # Hot tier: in-memory cache for recent items
        self._hot_cache: dict[str, Any] = {}

    # --- Ingest ---

    async def ingest_text(self, text: str, source_id: str = "",
                          metadata: dict[str, Any] | None = None) -> list[str]:
        """Chunk text, embed, and index. Returns chunk IDs."""
        chunks_text = chunk_text(text)
        chunk_ids = []
        for ct in chunks_text:
            chunk = Chunk(content=ct, source_id=source_id, metadata=metadata or {})
            self.sqlite.insert_chunk(chunk)
            chunk_ids.append(chunk.id)

        # Embed and index
        await self._embed_and_index(chunk_ids, chunks_text)
        self.audit.log_write("chunks", source_id or "inline", f"ingested {len(chunk_ids)} chunks")
        return chunk_ids

    def ingest_text_sync(self, text: str, source_id: str = "",
                         metadata: dict[str, Any] | None = None) -> list[str]:
        """Chunk text and index for FTS5 keyword search (no embeddings needed).

        This is the synchronous counterpart to ingest_text — it stores chunks
        in SQLite with full-text search but skips FAISS vector indexing.
        Use this for bulk ingestion where embedding API calls aren't practical.
        """
        chunks_text = chunk_text(text)
        chunk_ids = []
        for ct in chunks_text:
            chunk = Chunk(content=ct, source_id=source_id, metadata=metadata or {})
            self.sqlite.insert_chunk(chunk)
            chunk_ids.append(chunk.id)
        self.audit.log_write("chunks", source_id or "inline", f"ingested {len(chunk_ids)} chunks (sync)")
        return chunk_ids

    def ingest_session(self, session_path: Path) -> dict:
        """Parse and ingest an agent session file into searchable memory.

        Parses both Claude Code and OpenClaw JSONL formats, extracts
        meaningful content (messages, tool calls), chunks it, and stores
        in SQLite with FTS5 for keyword search.

        Returns dict with session_id, chunks_ingested, roles breakdown.
        """
        from c3ae.ingestion.session_parser import SessionParser

        parser = SessionParser()
        session_chunks = parser.parse_file(session_path)

        if not session_chunks:
            return {"session_id": session_path.stem, "chunks_ingested": 0, "roles": {}}

        # Only ingest roles that contain meaningful conversational content.
        # tool_call, tool_result, system, and unknown are noise that pollutes search.
        _INGEST_ROLES = {"user", "assistant"}

        session_id = session_chunks[0].session_id
        roles: dict[str, int] = {}
        skipped: dict[str, int] = {}
        total_ingested = 0

        for sc in session_chunks:
            if sc.role not in _INGEST_ROLES:
                skipped[sc.role] = skipped.get(sc.role, 0) + 1
                continue

            meta = {"role": sc.role, "session_id": sc.session_id,
                    "source_file": sc.source_file, "index": sc.index}
            meta.update(sc.metadata)

            chunk = Chunk(
                content=sc.content,
                source_id=f"session:{sc.session_id}",
                metadata=meta,
            )
            self.sqlite.insert_chunk(chunk)
            total_ingested += 1
            roles[sc.role] = roles.get(sc.role, 0) + 1

        self.audit.log_write("session_ingest", session_id,
                             f"ingested {total_ingested} chunks from {session_path.name}"
                             f" (skipped {sum(skipped.values())})")

        return {
            "session_id": session_id,
            "chunks_ingested": total_ingested,
            "roles": roles,
            "skipped": skipped,
        }

    async def ingest_file(self, file_path: Path, metadata: dict[str, Any] | None = None) -> list[str]:
        """Ingest a file from disk."""
        data = file_path.read_bytes()
        text = data.decode("utf-8", errors="replace")
        # Store in vault
        content_hash = self.vault.store_document(data, file_path.name, metadata)
        self.sqlite.insert_file(
            content_hash, str(file_path), content_hash, len(data),
            "", metadata,
        )
        return await self.ingest_text(text, source_id=content_hash, metadata=metadata)

    # --- Search ---

    async def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """Hybrid search across all memory tiers."""
        top_k = top_k or self.config.retrieval.default_top_k

        # Try to get query embedding (may fail if no API key)
        query_vec = None
        try:
            query_vec = await self._embed_text(query)
        except Exception:
            pass  # Fall back to keyword-only

        results = self.hybrid_search.search(query, query_vector=query_vec, top_k=top_k)
        self.audit.log_search(query, len(results))
        return results

    def search_keyword(self, query: str, top_k: int = 20) -> list[SearchResult]:
        """Keyword-only search (synchronous, no embedding needed)."""
        results = self.keyword_search.search_all(query, limit=top_k)
        self.audit.log_search(query, len(results))
        return results

    # --- Reasoning Bank (governed) ---

    async def add_knowledge(self, title: str, content: str,
                            tags: list[str] | None = None,
                            evidence_ids: list[str] | None = None,
                            session_id: str | None = None,
                            bypass_governance: bool = False) -> ReasoningEntry:
        """Add a reasoning entry through the governance gate."""
        entry = ReasoningEntry(
            title=title, content=content,
            tags=tags or [], evidence_ids=evidence_ids or [],
            session_id=session_id,
        )
        if not bypass_governance:
            ok, issues, warnings = await self.guardian.validate_and_report_async(entry)
            if not ok:
                self.audit.log_blocked("reasoning_entry", entry.id, "; ".join(issues))
                raise GovernanceError(f"Write blocked: {'; '.join(issues)}")
            if warnings:
                self.audit.log("warning", "reasoning_entry", entry.id,
                               "; ".join(warnings))

        self.sqlite.insert_reasoning_entry(entry)

        # Also index as a chunk so it's discoverable via vector search
        # and semantic contradiction detection can find it in FAISS
        combined = f"{title}. {content}"
        chunk = Chunk(content=combined, source_id=entry.id, metadata={"type": "reasoning_entry"})
        self.sqlite.insert_chunk(chunk)
        await self._embed_and_index([chunk.id], [combined])

        self.audit.log_write("reasoning_entry", entry.id, title)
        return entry

    def supersede_knowledge(self, old_id: str, new_title: str, new_content: str,
                            tags: list[str] | None = None,
                            evidence_ids: list[str] | None = None) -> ReasoningEntry:
        entry = self.bank.supersede(old_id, new_title, new_content, tags, evidence_ids)
        self.audit.log_write("reasoning_entry", entry.id, f"supersedes {old_id}")
        return entry

    # --- Evidence ---

    def add_evidence(self, claim: str, sources: list[str],
                     confidence: float = 0.0, reasoning: str = "") -> EvidencePack:
        pack = self.evidence.create(claim, sources, confidence, reasoning)
        self.audit.log_write("evidence_pack", pack.id, claim)
        return pack

    # --- Skills ---

    def register_skill(self, name: str, description: str, procedure: str,
                       tags: list[str] | None = None) -> SkillCapsule:
        capsule = self.skills.register(name, description, procedure, tags)
        self.audit.log_write("skill_capsule", capsule.id, name)
        return capsule

    # --- Sessions ---

    def start_session(self, session_id: str, metadata: dict[str, Any] | None = None) -> str:
        self.sqlite.create_session(session_id, metadata)
        self.audit.log("session_start", "session", session_id)
        return session_id

    def end_session(self, session_id: str) -> None:
        self.sqlite.end_session(session_id)
        self.audit.log("session_end", "session", session_id)

    # --- Status ---

    def status(self) -> dict[str, Any]:
        status = {
            "chunks": self.sqlite.count_chunks(),
            "vectors": self.faiss.size,
            "reasoning_entries": len(self.bank.list_active()),
            "skills": len(self.skills.list_all()),
            "vault_documents": len(self.vault.list_documents()),
        }
        # Add compression stats if vault is CompressedVault
        if hasattr(self.vault, 'compression_stats'):
            status["compression"] = self.vault.compression_stats()
        return status

    # --- Internals ---

    async def _embed_text(self, text: str) -> np.ndarray:
        cached = self.embed_cache.get(text)
        if cached is not None:
            return cached
        vec = await self.embedder.embed_single(text)
        self.embed_cache.put(text, vec)
        return vec

    async def _embed_and_index(self, chunk_ids: list[str], texts: list[str]) -> None:
        """Embed texts with caching and add to FAISS index."""
        results, miss_indices = self.embed_cache.get_batch(texts)

        if miss_indices:
            miss_texts = [texts[i] for i in miss_indices]
            try:
                new_vecs = await self.embedder.embed(miss_texts)
                self.embed_cache.put_batch(miss_texts, new_vecs)
                for j, mi in enumerate(miss_indices):
                    results[mi] = new_vecs[j]
            except Exception:
                # If embedding fails, skip vector indexing
                return

        # Index all successfully embedded chunks
        for cid, vec in zip(chunk_ids, results):
            if vec is not None:
                self.faiss.add(vec, cid)

        # Save FAISS index
        if self.config.faiss_dir:
            self.faiss.save()

    async def close(self) -> None:
        await self.embedder.close()
        self.sqlite.close()
        if self.config.faiss_dir:
            self.faiss.save()
