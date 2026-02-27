"""Memory Spine — orchestrator for Hot/Warm/Cold tiers + retrieval + governance."""

from __future__ import annotations

import asyncio
import inspect
import hashlib
import re
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np

from c3ae.config import Config
from c3ae.cos.cos import COSManager
from c3ae.embeddings.backends import create_embedder
from c3ae.embeddings.cache import EmbeddingCache
from c3ae.exceptions import GovernanceError
from c3ae.graph.extractor import ExtractedGraph, extract_graph_facts, extract_graph_facts_async
from c3ae.governance.audit import AuditLog
from c3ae.governance.guardian import Guardian
from c3ae.llm import create_chat_backend
from c3ae.reasoning_bank.bank import ReasoningBank
from c3ae.reasoning_bank.evidence import EvidenceManager
from c3ae.reasoning_bank.manager import MemoryWriteManager
from c3ae.retrieval.hybrid import HybridSearch
from c3ae.retrieval.keyword import KeywordSearch
from c3ae.retrieval.vector import VectorSearch
from c3ae.skill_capsules.registry import SkillRegistry
from c3ae.storage.faiss_store import FAISSStore
from c3ae.storage.sqlite_store import SQLiteStore
from c3ae.usc_bridge.compressed_vault import CompressedVault
from c3ae.types import (
    Chunk,
    EvidencePack,
    ReasoningEntry,
    SearchResult,
    SkillCapsule,
)
from c3ae.utils import chunk_text, iso_str, utcnow

_BENCH_CASE_TOKEN_RE = re.compile(r"__[A-Z]+_CASE_\d+__", re.IGNORECASE)


class MemorySpine:
    """Central orchestrator wiring all memory subsystems."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self.config.ensure_dirs()

        # Storage backends
        self.sqlite = SQLiteStore(self.config.db_path)
        self.faiss = FAISSStore(
            dims=self.config.venice.embedding_dims,
            faiss_dir=self.config.faiss_dir,
            ivf_threshold=self.config.retrieval.faiss_ivf_threshold,
        )
        self.vault = CompressedVault(self.config.vault_dir)

        # Subsystems
        self.embedder = create_embedder(self.config.venice)
        self.embed_cache = EmbeddingCache(self.sqlite, self.config.venice.embedding_model)
        self.keyword_search = KeywordSearch(self.sqlite)
        self.vector_search = VectorSearch(self.faiss, self.sqlite)
        self.hybrid_search = HybridSearch(
            self.keyword_search,
            self.vector_search,
            self.config.retrieval,
            access_count_getter=self.sqlite.get_memory_access_count,
            access_counts_getter=self.sqlite.get_memory_access_counts,
        )
        self.cos = COSManager(self.sqlite, self.config.cos)
        self.bank = ReasoningBank(self.sqlite)
        self.write_manager = MemoryWriteManager(self.bank, self.config.memory_manager)
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

        # USC cognitive deduplication store (lazy init)
        self._cogstore = None
        self._graph_chat = None
        self._consolidation_chat = None

    # --- Ingest ---

    async def ingest_text(self, text: str, source_id: str = "",
                          metadata: dict[str, Any] | None = None) -> list[str]:
        """Chunk text, embed, and index. Returns chunk IDs."""
        chunks_text = chunk_text(text)
        base_meta = self._build_chunk_metadata(text, source_id, metadata)
        skip_graph_index = bool(base_meta.get("benchmark_case_token"))
        chunk_ids = []
        for ct in chunks_text:
            chunk = Chunk(content=ct, source_id=source_id, metadata=dict(base_meta))
            self.sqlite.insert_chunk(chunk)
            if not skip_graph_index:
                await self._index_graph_chunk_async(chunk.id, chunk.content, chunk.metadata)
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
        base_meta = self._build_chunk_metadata(text, source_id, metadata)
        skip_graph_index = bool(base_meta.get("benchmark_case_token"))
        chunk_ids = []
        for ct in chunks_text:
            chunk = Chunk(content=ct, source_id=source_id, metadata=dict(base_meta))
            self.sqlite.insert_chunk(chunk)
            if not skip_graph_index:
                self._index_graph_chunk(chunk.id, chunk.content, chunk.metadata)
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
            self._index_graph_chunk(chunk.id, chunk.content, chunk.metadata)
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
        benchmark_case_query = bool(self._extract_benchmark_case_token(query))
        if self.config.graph.enabled and not benchmark_case_query:
            graph_results = self.sqlite.search_graph_context(query, limit=max(top_k, 5))
            if graph_results:
                results = self._merge_with_graph(query, results, graph_results, top_k=top_k)
        self._record_access(results)
        self.audit.log_search(query, len(results))
        return results

    def search_keyword(self, query: str, top_k: int = 20) -> list[SearchResult]:
        """Keyword-only search (synchronous, no embedding needed)."""
        results = self.keyword_search.search_all(query, limit=top_k)
        self._record_access(results)
        self.audit.log_search(query, len(results))
        return results

    # --- Stable Protocol Surface ---

    async def ingest(
        self,
        text: str,
        source_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        """Stable protocol alias for ingest_text()."""
        return await self.ingest_text(text, source_id=source_id, metadata=metadata)

    async def recall(
        self,
        query: str,
        top_k: int = 20,
        session_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Stable protocol method returning normalized recall rows."""
        results = await self.search(query, top_k=max(top_k * 2, top_k))
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for r in results:
            meta = r.metadata or {}
            session_id = str(meta.get("session_id", ""))
            if session_filter and session_filter not in session_id:
                continue
            benchmark_doc_id = str(meta.get("benchmark_doc_id", "")).strip()
            benchmark_source = str(meta.get("benchmark_source", "")).strip()
            if benchmark_doc_id:
                dedup_key = f"benchmark_doc:{benchmark_doc_id}"
            elif benchmark_source:
                dedup_key = f"benchmark_source:{benchmark_source}"
            else:
                dedup_key = r.content.strip().lower()[:200]
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            rows.append(
                {
                    "id": r.id,
                    "content": r.content,
                    "score": float(r.score),
                    "source": r.source,
                    "metadata": meta,
                }
            )
            if len(rows) >= top_k:
                break
        return rows

    async def augment(
        self,
        query: str,
        top_k: int = 5,
        format: str = "xml",
        min_score: float = 0.005,
        roles: list[str] | None = None,
    ) -> str:
        """Stable protocol method producing pre-formatted memory context."""
        allowed_roles = set(roles or ["user", "assistant"])
        recalled = await self.recall(query, top_k=max(top_k * 10, top_k))
        selected: list[dict[str, Any]] = []
        for row in recalled:
            meta = row.get("metadata") or {}
            role = str(meta.get("role", "unknown"))
            if role not in allowed_roles:
                continue
            if float(row.get("score", 0.0)) < min_score:
                continue
            selected.append(
                {
                    "content": str(row.get("content", "")),
                    "role": role,
                }
            )
            if len(selected) >= top_k:
                break
        return self._render_augmented_context(selected, format=format)

    def protocol_client(self, version: str = "v1") -> Any:
        """Create a versioned local Spine protocol client."""
        if version == "v1":
            from c3ae.protocol.client import SpineClientV1

            return SpineClientV1(self)
        if version == "v2":
            from c3ae.protocol.client import SpineClientV2

            return SpineClientV2(self)
        raise ValueError(f"Unsupported protocol version: {version}")

    def set_decay_config(self, half_life_hours: float) -> None:
        """V2 protocol hook for decay tuning without changing the interface."""
        self.config.retrieval.decay_half_life_hours = max(1.0, float(half_life_hours))

    async def graph_query(self, entity: str, depth: int = 2) -> dict[str, Any]:
        """Query temporal memory graph around an entity."""
        return self.sqlite.query_graph(entity, depth=depth)

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
        decision = await self.write_manager.decide_async(entry)
        if decision.action == "NOOP" and decision.target_id:
            existing = self.bank.get(decision.target_id)
            if existing:
                self.audit.log(
                    "reasoning_noop",
                    "reasoning_entry",
                    existing.id,
                    decision.reason,
                )
                return existing
        if decision.action == "DELETE" and decision.target_id:
            existing = self.bank.get(decision.target_id)
            if existing:
                self.bank.retract(decision.target_id)
                self.audit.log(
                    "reasoning_delete",
                    "reasoning_entry",
                    decision.target_id,
                    decision.reason,
                )
        if decision.action == "UPDATE" and decision.target_id:
            if not bypass_governance:
                ok, issues, warnings = await self.guardian.validate_and_report_async(entry)
                if not ok:
                    self.audit.log_blocked("reasoning_entry", entry.id, "; ".join(issues))
                    raise GovernanceError(f"Write blocked: {'; '.join(issues)}")
                if warnings:
                    self.audit.log("warning", "reasoning_entry", entry.id, "; ".join(warnings))
            superseded = self.supersede_knowledge(
                decision.target_id,
                new_title=title,
                new_content=content,
                tags=tags,
                evidence_ids=evidence_ids,
            )
            self.audit.log(
                "reasoning_update",
                "reasoning_entry",
                superseded.id,
                decision.reason,
            )
            return superseded

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
        chunk = Chunk(
            content=combined,
            source_id=entry.id,
            metadata={
                "type": "reasoning_entry",
                "reasoning_entry_id": entry.id,
            },
        )
        self.sqlite.insert_chunk(chunk)
        await self._index_graph_chunk_async(chunk.id, chunk.content, chunk.metadata)
        await self._embed_and_index([chunk.id], [combined])

        self.audit.log_write("reasoning_entry", entry.id, title)
        return entry

    def supersede_knowledge(self, old_id: str, new_title: str, new_content: str,
                            tags: list[str] | None = None,
                            evidence_ids: list[str] | None = None) -> ReasoningEntry:
        entry = self.bank.supersede(old_id, new_title, new_content, tags, evidence_ids)
        combined = f"{new_title}. {new_content}"
        chunk = Chunk(
            content=combined,
            source_id=entry.id,
            metadata={"type": "reasoning_entry", "reasoning_entry_id": entry.id},
        )
        self.sqlite.insert_chunk(chunk)
        self._index_graph_chunk(chunk.id, chunk.content, chunk.metadata)
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

    # --- USC Cognitive Dedup ---

    @property
    def cogstore(self):
        """Lazy-init C3-backed cognitive dedup store."""
        if self._cogstore is None:
            from c3ae.usc_bridge.c3_cogstore import C3CogStore
            self._cogstore = C3CogStore(self.config.db_path)
        return self._cogstore

    @property
    def _predictor(self):
        """Lazy-init predictive compressor."""
        if not hasattr(self, '_lazy_predictor'):
            from usc.cogdedup.predictor import PredictiveCompressor
            self._lazy_predictor = PredictiveCompressor(self.cogstore)
        return self._lazy_predictor

    @property
    def _integrity_verifier(self):
        """Lazy-init integrity verifier for delta/hash checks."""
        if not hasattr(self, '_lazy_integrity_verifier'):
            from usc.cogdedup.integrity import IntegrityVerifier, SecurityPolicy
            self._lazy_integrity_verifier = IntegrityVerifier(SecurityPolicy())
        return self._lazy_integrity_verifier

    @property
    def _anomaly_detector(self):
        """Lazy-init anomaly detector for compression ratio monitoring."""
        if not hasattr(self, '_lazy_anomaly_detector'):
            from usc.cogdedup.anomaly import AnomalyDetector
            self._lazy_anomaly_detector = AnomalyDetector()
        return self._lazy_anomaly_detector

    @property
    def _context_compactor(self):
        """Lazy-init context compactor for LLM prompt compression."""
        if not hasattr(self, '_lazy_context_compactor'):
            from usc.cogdedup.context_compactor import ContextCompactor
            self._lazy_context_compactor = ContextCompactor(self.cogstore)
        return self._lazy_context_compactor

    @property
    def _temporal_tracker(self):
        """Lazy-init temporal motif tracker for event sequence detection."""
        if not hasattr(self, '_lazy_temporal_tracker'):
            from usc.cogdedup.temporal import TemporalMotifTracker
            self._lazy_temporal_tracker = TemporalMotifTracker()
        return self._lazy_temporal_tracker

    @property
    def _recursive_compressor(self):
        """Lazy-init recursive compressor for C3's own state."""
        if not hasattr(self, '_lazy_recursive_compressor'):
            from usc.cogdedup.recursive import RecursiveCompressor
            self._lazy_recursive_compressor = RecursiveCompressor(self.cogstore)
        return self._lazy_recursive_compressor

    def compress_with_dedup(self, data: bytes, data_id: str = "") -> tuple[bytes, dict]:
        """Compress data using cognitive deduplication with integrity + anomaly detection.

        Returns (compressed_blob, stats).
        Stats includes integrity_hash for verification and anomaly_alert if detected.
        The more data processed, the better compression becomes.

        Args:
            data: Raw bytes to compress
            data_id: Optional ID for compression-aware retrieval mapping
        """
        from usc.cogdedup.codec import cogdedup_encode
        blob, stats = cogdedup_encode(data, self.cogstore,
                                      data_id=data_id, predictor=self._predictor)

        # Integrity: store hash of original data for later verification
        stats["integrity_hash"] = self._integrity_verifier.compute_hash(data).hex()

        # Anomaly detection: observe compression ratio
        ratio = len(data) / max(1, len(blob))
        alert = self._anomaly_detector.observe(ratio, label=data_id)
        if alert:
            stats["anomaly_alert"] = {"type": alert.severity, "z_score": alert.z_score}
            self.audit.log("anomaly_detected", "cogdedup", data_id,
                           f"{alert.severity}: z={alert.z_score:.2f}")

        return blob, stats

    def decompress_with_dedup(self, blob: bytes, expected_hash: str = "") -> bytes:
        """Decompress a cogdedup blob with optional integrity verification.

        Args:
            blob: UCOG compressed blob
            expected_hash: Hex hash from compress stats for verification
        """
        from usc.cogdedup.codec import cogdedup_decode
        data = cogdedup_decode(blob, self.cogstore, predictor=self._predictor)

        if expected_hash:
            actual = self._integrity_verifier.compute_hash(data).hex()
            if actual != expected_hash:
                raise ValueError(
                    f"Integrity check failed: expected {expected_hash[:16]}..., "
                    f"got {actual[:16]}..."
                )

        return data

    def stream_compressor(self, data_id: str = ""):
        """Create a streaming cogdedup encoder for real-time session compression.

        Returns a CogdedupStream instance. Feed data as it arrives,
        call finish() to get the UCOG blob.

        Usage:
            stream = spine.stream_compressor(data_id="session-123")
            stream.feed(b"[TOOL_CALL] web_search ...\\n")
            stream.feed(b"[TOOL_RESULT] ...\\n")
            blob, stats = stream.finish()
        """
        from usc.cogdedup.streaming import CogdedupStream
        return CogdedupStream(self.cogstore, data_id=data_id,
                              predictor=self._predictor)

    # --- Context Compaction (LLM Prompt Compression) ---

    def compress_prompt(self, prompt: str) -> dict:
        """Compress an LLM prompt by replacing known chunks with REF placeholders.

        Returns dict with compressed text, token savings, refs used.
        """
        result = self._context_compactor.compress_prompt(prompt)
        return {
            "compressed": result.text,
            "token_savings": result.savings_pct,
            "refs_used": result.refs_inserted,
            "original_tokens": result.original_tokens,
        }

    def expand_response(self, text: str) -> str:
        """Expand REF placeholders in a response back to full content."""
        return self._context_compactor.expand_response(text)

    # --- Temporal Event Tracking ---

    def track_event(self, event_type: str) -> dict | None:
        """Track a temporal event and return motif if a recurring pattern is detected."""
        motif = self._temporal_tracker.observe(event_type)
        if motif:
            return {"pattern": motif.pattern, "count": motif.occurrences,
                    "length": len(motif.pattern)}
        return None

    def track_events_batch(self, events: list[str]) -> list[dict]:
        """Track a batch of events, return all detected motifs."""
        self._temporal_tracker.observe_batch(events)
        return [{"pattern": m.pattern, "count": m.occurrences}
                for m in self._temporal_tracker.detected_motifs()]

    # --- Recursive Self-Compression ---

    def compress_memories(self, memories: list[dict]) -> tuple[bytes, dict]:
        """Compress a batch of C3 memories using cognitive dedup.

        Returns (blob, stats_dict).
        """
        result = self._recursive_compressor.compress_memories(memories)
        return result.blob, {
            "ratio": result.ratio,
            "original_size": result.original_size,
            "compressed_size": result.compressed_size,
        }

    def compress_reasoning_bank(self) -> tuple[bytes, dict]:
        """Compress the entire reasoning bank for archival."""
        entries = self.bank.list_active()
        items = [e.content for e in entries]
        result = self._recursive_compressor.compress_reasoning_bank(items)
        return result.blob, {
            "ratio": result.ratio,
            "entries": len(entries),
            "original_size": result.original_size,
            "compressed_size": result.compressed_size,
        }

    def compress_audit_log(self, limit: int = 10000) -> tuple[bytes, dict]:
        """Compress recent audit log for archival."""
        events = self.audit.recent(limit=limit)
        items = [{"action": e.action, "target_type": e.target_type,
                  "target_id": e.target_id, "detail": e.detail,
                  "created_at": str(e.created_at)} for e in events]
        result = self._recursive_compressor.compress_audit_log(items)
        return result.blob, {
            "ratio": result.ratio,
            "events": len(items),
            "original_size": result.original_size,
            "compressed_size": result.compressed_size,
        }

    # --- Session Orchestrator ---

    def compress_session(self, session_data: bytes, session_id: str = "") -> dict:
        """Compress an agent session with all cogdedup upgrades active.

        Combines: cognitive dedup, integrity hashing, anomaly detection,
        and temporal motif tracking in a single call.

        Returns dict with blob, stats, session_id, compressed_size, original_size.
        """
        # 1. Compress with cognitive dedup (includes integrity + anomaly)
        blob, stats = self.compress_with_dedup(session_data, data_id=session_id)

        # 2. Track temporal patterns from session event types
        lines = session_data.decode("utf-8", errors="replace").split("\n")
        event_types = []
        for line in lines:
            if line.startswith("[TOOL_CALL]"):
                event_types.append("tool_call")
            elif line.startswith("[TOOL_RESULT]"):
                event_types.append("tool_result")
            elif line.startswith("[SEARCH]"):
                event_types.append("search")
            elif line.startswith("[ERROR]"):
                event_types.append("error")
        if event_types:
            self._temporal_tracker.observe_batch(event_types)
            motifs = self._temporal_tracker.detected_motifs()
            stats["temporal_motifs"] = len(motifs)

        # 3. Audit
        ratio = len(session_data) / max(1, len(blob))
        self.audit.log("session_compressed", "session", session_id,
                       f"ratio={ratio:.1f}x, size={len(blob)}")

        return {
            "blob": blob,
            "stats": stats,
            "session_id": session_id,
            "compressed_size": len(blob),
            "original_size": len(session_data),
        }

    # --- Compression-Aware Retrieval ---

    def structural_similarity(self, data_id_a: str, data_id_b: str) -> float:
        """Compute structural similarity between two memories using shared chunks.

        Returns Jaccard similarity (0.0 to 1.0) over shared chunk IDs.
        This is a free signal from the compression layer — no embeddings needed.
        """
        return self.cogstore.structural_similarity(data_id_a, data_id_b)

    def find_structurally_similar(self, data_id: str, threshold: float = 0.3) -> list[tuple]:
        """Find memories structurally similar to the given one.

        Uses chunk overlap from cognitive deduplication.
        Returns list of (memory_id, jaccard_score) sorted by similarity.
        """
        return self.cogstore.find_structurally_similar(data_id, threshold)

    def deduplicate_results(self, results: list[SearchResult],
                            threshold: float = 0.8) -> list[SearchResult]:
        """Remove structurally duplicate search results.

        Uses chunk-level Jaccard similarity to identify results that are
        minor variants of each other. Keeps the highest-scored result
        from each cluster.
        """
        if not results or self._cogstore is None:
            return results

        keep = []
        seen_groups: list[set[int]] = []

        for r in results:
            r_chunks = self.cogstore.get_chunk_ids_for_data(r.id)
            if not r_chunks:
                keep.append(r)
                continue

            is_dup = False
            for group_chunks in seen_groups:
                if not group_chunks:
                    continue
                intersection = r_chunks & group_chunks
                union = r_chunks | group_chunks
                jaccard = len(intersection) / len(union) if union else 0.0
                if jaccard >= threshold:
                    is_dup = True
                    break

            if not is_dup:
                keep.append(r)
                seen_groups.append(r_chunks)

        return keep

    # --- Consolidation & Forgetting ---

    async def consolidate_async(
        self,
        session_id: str | None = None,
        max_chunks: int = 1000,
    ) -> dict[str, Any]:
        """Consolidate episodic chunks into higher-level semantic memories."""
        if not self.config.consolidation.enabled:
            return {"enabled": False, "clusters_created": 0, "chunks_processed": 0}

        recent_chunks = self._list_recent_chunks_for_consolidation(
            session_id=session_id,
            max_chunks=max_chunks,
        )
        if not recent_chunks:
            return {"enabled": True, "clusters_created": 0, "chunks_processed": 0}

        clusters = self._cluster_chunks_semantic(recent_chunks)
        min_size = max(1, int(self.config.consolidation.min_cluster_size))
        max_clusters = max(1, int(self.config.consolidation.max_clusters_per_run))

        created = 0
        llm_used = 0
        for cluster_chunks in clusters[:max_clusters]:
            if len(cluster_chunks) < min_size:
                continue
            facts, used_llm_facts = await self._extract_cluster_facts_async(cluster_chunks)
            summary, used_llm_summary = await self._summarize_cluster_async(cluster_chunks, facts=facts)
            llm_used += int(used_llm_facts or used_llm_summary)
            cluster_key = self._cluster_key(cluster_chunks, session_id=session_id)
            source_chunk_ids = [c.id for c in cluster_chunks]
            self.sqlite.upsert_consolidated_memory(
                cluster_key=cluster_key,
                summary=summary,
                facts=facts,
                source_chunk_ids=source_chunk_ids,
                metadata={
                    "session_id": session_id or "",
                    "cluster_size": len(cluster_chunks),
                    "llm_enriched": bool(used_llm_facts or used_llm_summary),
                    "cluster_mode": "semantic",
                },
            )
            created += 1

        self.audit.log(
            "consolidation",
            "memory",
            session_id or "global",
            f"clusters={created}, chunks={len(recent_chunks)}, llm_clusters={llm_used}",
        )
        return {
            "enabled": True,
            "clusters_created": created,
            "chunks_processed": len(recent_chunks),
            "llm_enriched_clusters": llm_used,
        }

    def consolidate(
        self,
        session_id: str | None = None,
        max_chunks: int = 1000,
    ) -> dict[str, Any]:
        """Synchronous consolidation path (semantic clustering + heuristic summaries)."""
        if not self.config.consolidation.enabled:
            return {"enabled": False, "clusters_created": 0, "chunks_processed": 0}

        recent_chunks = self._list_recent_chunks_for_consolidation(
            session_id=session_id,
            max_chunks=max_chunks,
        )
        if not recent_chunks:
            return {"enabled": True, "clusters_created": 0, "chunks_processed": 0}

        clusters = self._cluster_chunks_semantic(recent_chunks)
        min_size = max(1, int(self.config.consolidation.min_cluster_size))
        max_clusters = max(1, int(self.config.consolidation.max_clusters_per_run))

        created = 0
        for cluster_chunks in clusters[:max_clusters]:
            if len(cluster_chunks) < min_size:
                continue
            facts = self._extract_cluster_facts(cluster_chunks)
            summary = self._summarize_cluster(cluster_chunks, facts=facts)
            cluster_key = self._cluster_key(cluster_chunks, session_id=session_id)
            self.sqlite.upsert_consolidated_memory(
                cluster_key=cluster_key,
                summary=summary,
                facts=facts,
                source_chunk_ids=[c.id for c in cluster_chunks],
                metadata={
                    "session_id": session_id or "",
                    "cluster_size": len(cluster_chunks),
                    "llm_enriched": False,
                    "cluster_mode": "semantic",
                },
            )
            created += 1

        self.audit.log(
            "consolidation",
            "memory",
            session_id or "global",
            f"clusters={created}, chunks={len(recent_chunks)}",
        )
        return {
            "enabled": True,
            "clusters_created": created,
            "chunks_processed": len(recent_chunks),
            "llm_enriched_clusters": 0,
        }

    def forget_stale(
        self,
        max_age_days: int = 120,
        max_access_count: int = 0,
        limit: int = 1000,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        """Delete cold memories under age/access thresholds."""
        cutoff = iso_str(utcnow() - timedelta(days=max(1, max_age_days)))
        candidates = self.sqlite.list_chunks_with_access(limit=limit, older_than=cutoff)
        stale = [
            c for c in candidates
            if int(c.get("access_count", 0)) <= max_access_count
        ]
        if dry_run:
            return {
                "dry_run": True,
                "candidate_count": len(stale),
                "deleted": 0,
            }
        deleted = 0
        for row in stale:
            if self.sqlite.delete_chunk(str(row["id"])):
                deleted += 1
        self.audit.log("forget", "memory", "stale", f"deleted={deleted}")
        return {
            "dry_run": False,
            "candidate_count": len(stale),
            "deleted": deleted,
        }

    async def dream_consolidate_async(self) -> dict[str, Any]:
        """Run an expanded offline-style dream pass for idle periods."""
        consolidation = await self.consolidate_async()
        forgetting = self.forget_stale(dry_run=True)
        contradictions = self.sqlite.list_graph_contradictions(
            lookback_hours=max(24, int(self.config.consolidation.lookback_hours)),
            limit=30,
        )
        skill_candidates = self._propose_skill_candidates(limit=10)
        recompression_preview = self._dream_recompression_preview(limit=200)
        novelty = self._dream_novelty_estimate(limit=200)
        report = {
            "consolidation": consolidation,
            "forgetting_preview": forgetting,
            "contradictions": contradictions,
            "skill_candidates": skill_candidates,
            "recompression_preview": recompression_preview,
            "novelty": novelty,
        }
        self.audit.log(
            "dream_consolidation",
            "memory",
            "global",
            (
                "clusters="
                f"{consolidation.get('clusters_created', 0)}, "
                f"contradictions={len(contradictions)}, "
                f"skills={len(skill_candidates)}"
            ),
        )
        return report

    def dream_consolidate(self) -> dict[str, Any]:
        """Run an expanded dream pass; sync-safe wrapper for CLI/tests."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.dream_consolidate_async())

        consolidation = self.consolidate()
        forgetting = self.forget_stale(dry_run=True)
        contradictions = self.sqlite.list_graph_contradictions(
            lookback_hours=max(24, int(self.config.consolidation.lookback_hours)),
            limit=30,
        )
        return {
            "consolidation": consolidation,
            "forgetting_preview": forgetting,
            "contradictions": contradictions,
            "skill_candidates": self._propose_skill_candidates(limit=10),
            "recompression_preview": self._dream_recompression_preview(limit=200),
            "novelty": self._dream_novelty_estimate(limit=200),
        }

    def _list_recent_chunks_for_consolidation(
        self,
        session_id: str | None,
        max_chunks: int,
    ) -> list[Chunk]:
        lookback = utcnow() - timedelta(hours=max(1, int(self.config.consolidation.lookback_hours)))
        chunks = self.sqlite.list_chunks(
            limit=max_chunks,
            older_than=iso_str(utcnow()),
            session_id=session_id,
        )
        return [c for c in chunks if c.created_at >= lookback]

    def _cluster_chunks_semantic(self, chunks: list[Chunk]) -> list[list[Chunk]]:
        if not chunks:
            return []
        if len(chunks) == 1:
            return [[chunks[0]]]

        class _UnionFind:
            def __init__(self, n: int) -> None:
                self.parent = list(range(n))
                self.rank = [0] * n

            def find(self, x: int) -> int:
                while self.parent[x] != x:
                    self.parent[x] = self.parent[self.parent[x]]
                    x = self.parent[x]
                return x

            def union(self, a: int, b: int) -> None:
                ra = self.find(a)
                rb = self.find(b)
                if ra == rb:
                    return
                if self.rank[ra] < self.rank[rb]:
                    self.parent[ra] = rb
                elif self.rank[ra] > self.rank[rb]:
                    self.parent[rb] = ra
                else:
                    self.parent[rb] = ra
                    self.rank[ra] += 1

        n = len(chunks)
        uf = _UnionFind(n)
        chunk_ids = [c.id for c in chunks]

        # Pass 1: exact/near-exact lexical signatures.
        sig_to_idx: dict[str, list[int]] = {}
        token_sets: list[set[str]] = []
        for i, c in enumerate(chunks):
            tokens = self._token_set(c.content)
            token_sets.append(tokens)
            if tokens:
                sig_base = " ".join(sorted(tokens)[:80])
            else:
                sig_base = c.content.strip().lower()[:200]
            sig = hashlib.sha1(sig_base.encode("utf-8", errors="ignore")).hexdigest()[:20]
            sig_to_idx.setdefault(sig, []).append(i)
        for idxs in sig_to_idx.values():
            if len(idxs) < 2:
                continue
            base = idxs[0]
            for j in idxs[1:]:
                uf.union(base, j)

        # Pass 2: vector similarity from existing FAISS vectors.
        vector_map = self.faiss.get_vectors_by_external_ids(chunk_ids)
        vec_pairs: list[tuple[int, np.ndarray]] = []
        for i, cid in enumerate(chunk_ids):
            vec = vector_map.get(cid)
            if vec is None:
                continue
            vec_pairs.append((i, vec))
        if len(vec_pairs) >= 2:
            indices = [x[0] for x in vec_pairs]
            matrix = np.vstack([x[1] for x in vec_pairs]).astype(np.float32)
            matrix = np.ascontiguousarray(matrix)
            sims = matrix @ matrix.T
            sim_thr = float(self.config.consolidation.vector_similarity_threshold)
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    if float(sims[i, j]) >= sim_thr:
                        uf.union(indices[i], indices[j])

        # Pass 3: graph entity overlap signal.
        entity_map = self.sqlite.get_entities_for_chunk_ids(chunk_ids)
        pair_overlap: dict[tuple[int, int], int] = {}
        entity_to_idx: dict[str, list[int]] = {}
        for i, cid in enumerate(chunk_ids):
            entities = entity_map.get(cid, set())
            for name in entities:
                entity_to_idx.setdefault(name, []).append(i)
        for idxs in entity_to_idx.values():
            if len(idxs) < 2:
                continue
            for a_pos in range(len(idxs)):
                for b_pos in range(a_pos + 1, len(idxs)):
                    a, b = idxs[a_pos], idxs[b_pos]
                    key = (a, b) if a < b else (b, a)
                    pair_overlap[key] = pair_overlap.get(key, 0) + 1

        entity_thr = float(self.config.consolidation.entity_overlap_threshold)
        if pair_overlap and entity_thr > 0.0:
            for (a, b), inter_count in pair_overlap.items():
                ea = entity_map.get(chunk_ids[a], set())
                eb = entity_map.get(chunk_ids[b], set())
                if not ea and not eb:
                    continue
                union_n = len(ea | eb)
                if union_n == 0:
                    continue
                jaccard = inter_count / union_n
                if jaccard >= entity_thr:
                    uf.union(a, b)

        # Pass 4: lexical overlap for chunks without vectors/entities.
        lexical_thr = float(self.config.consolidation.lexical_overlap_threshold)
        token_to_idx: dict[str, list[int]] = {}
        for i, tokens in enumerate(token_sets):
            for tok in list(tokens)[:24]:
                token_to_idx.setdefault(tok, []).append(i)
        lexical_pairs: dict[tuple[int, int], int] = {}
        for idxs in token_to_idx.values():
            if len(idxs) < 2:
                continue
            for a_pos in range(len(idxs)):
                for b_pos in range(a_pos + 1, len(idxs)):
                    a, b = idxs[a_pos], idxs[b_pos]
                    key = (a, b) if a < b else (b, a)
                    lexical_pairs[key] = lexical_pairs.get(key, 0) + 1
        if lexical_pairs and lexical_thr > 0.0:
            for (a, b), inter_count in lexical_pairs.items():
                ta = token_sets[a]
                tb = token_sets[b]
                if not ta and not tb:
                    continue
                union_n = len(ta | tb)
                if union_n == 0:
                    continue
                jaccard = inter_count / union_n
                if jaccard >= lexical_thr:
                    uf.union(a, b)

        groups: dict[int, list[Chunk]] = {}
        for i, chunk in enumerate(chunks):
            root = uf.find(i)
            groups.setdefault(root, []).append(chunk)

        ordered = list(groups.values())
        ordered.sort(
            key=lambda g: (
                len(g),
                max(c.created_at for c in g),
            ),
            reverse=True,
        )
        return ordered

    # --- Status ---

    def status(self) -> dict[str, Any]:
        status = {
            "chunks": self.sqlite.count_chunks(),
            "vectors": self.faiss.size,
            "reasoning_entries": len(self.bank.list_active()),
            "skills": len(self.skills.list_all()),
            "vault_documents": len(self.vault.list_documents()),
        }
        try:
            status["graph"] = {
                "entities": self.sqlite.count_entities(),
                "edges": self.sqlite.count_edges(active_only=True),
            }
        except Exception:
            status["graph"] = {"entities": 0, "edges": 0}
        try:
            status["consolidated_memories"] = self.sqlite.count_consolidated_memories()
        except Exception:
            status["consolidated_memories"] = 0
        # Add compression stats if vault is CompressedVault
        if hasattr(self.vault, 'compression_stats'):
            status["compression"] = self.vault.compression_stats()
        if self._cogstore is not None:
            status["cogdedup"] = self._cogstore.stats()
        if hasattr(self, '_lazy_anomaly_detector'):
            report = self._anomaly_detector.drift_report()
            status["anomaly"] = {
                "total_observations": self._anomaly_detector._observation_count,
                "alerts": report.alerts_count,
                "mean_ratio": round(report.current_mean, 2),
            }
        if hasattr(self, '_lazy_temporal_tracker'):
            motifs = self._temporal_tracker.detected_motifs()
            status["temporal"] = {"motifs_detected": len(motifs)}
        return status

    @staticmethod
    def _render_augmented_context(memories: list[dict[str, str]], format: str = "xml") -> str:
        if not memories:
            return ""
        if format == "xml":
            lines = []
            for m in memories:
                role = m.get("role", "memory")
                content = m.get("content", "")
                tag = role if role != "unknown" else "memory"
                lines.append(f"  <{tag}>{content}</{tag}>")
            return "<relevant-memories>\n" + "\n".join(lines) + "\n</relevant-memories>"
        lines = [f"- [{m.get('role', 'memory')}] {m.get('content', '')}" for m in memories]
        return "Relevant memories:\n" + "\n".join(lines)

    def _record_access(self, results: list[SearchResult]) -> None:
        if not results:
            return
        ids: list[str] = []
        seen: set[str] = set()
        for r in results:
            if r.id not in seen:
                ids.append(r.id)
                seen.add(r.id)
        try:
            self.sqlite.increment_memory_access(ids)
        except Exception:
            # Access tracking must never break retrieval paths.
            return

    def _build_chunk_metadata(
        self,
        text: str,
        source_id: str,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Attach stable metadata hints that help downstream retrieval/reranking."""
        merged = dict(metadata or {})
        if "benchmark_case_token" not in merged:
            case_token = self._extract_benchmark_case_token(text) or self._extract_benchmark_case_token(source_id)
            if case_token:
                merged["benchmark_case_token"] = case_token
        return merged

    @staticmethod
    def _extract_benchmark_case_token(raw: str) -> str:
        if not raw:
            return ""
        m = _BENCH_CASE_TOKEN_RE.search(raw)
        return m.group(0).upper() if m else ""

    def _merge_with_graph(
        self,
        query: str,
        memory_results: list[SearchResult],
        graph_results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        if not graph_results:
            return memory_results[:top_k]
        intent = self.hybrid_search._classify_intent(query)
        mem_weight = 0.85
        graph_weight = float(self.config.retrieval.graph_weight)
        if intent in {"temporal"}:
            mem_weight = 0.60
            graph_weight = max(graph_weight, 0.40)
        elif intent in {"entity_lookup"}:
            mem_weight = 0.70
            graph_weight = max(graph_weight, 0.30)

        k = 60
        scores: dict[str, float] = {}
        best: dict[str, SearchResult] = {}
        for rank, r in enumerate(memory_results):
            scores[r.id] = scores.get(r.id, 0.0) + (mem_weight / (k + rank + 1))
            best.setdefault(r.id, r)
        for rank, r in enumerate(graph_results):
            key = f"graph:{r.id}"
            scores[key] = scores.get(key, 0.0) + (graph_weight / (k + rank + 1))
            best.setdefault(key, r)
        ranked = sorted(scores, key=lambda rid: scores[rid], reverse=True)
        out: list[SearchResult] = []
        for rid in ranked[:top_k]:
            r = best[rid]
            out.append(
                SearchResult(
                    id=r.id,
                    content=r.content,
                    score=scores[rid],
                    source=r.source,
                    metadata=r.metadata,
                )
            )
        return out

    def _index_graph_chunk(
        self,
        chunk_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Synchronous graph indexing path (heuristic extraction only)."""
        if not self.config.graph.enabled:
            return
        try:
            extracted = extract_graph_facts(
                text,
                max_entities=self.config.graph.max_entities_per_chunk,
                max_relations=self.config.graph.max_relations_per_chunk,
            )
            self._apply_graph_extraction(
                chunk_id=chunk_id,
                extracted=extracted,
                metadata=metadata,
            )
        except Exception:
            # Graph indexing must never block memory ingest.
            return

    async def _index_graph_chunk_async(
        self,
        chunk_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self.config.graph.enabled:
            return
        try:
            chat_backend = None
            if (self.config.graph.extraction_mode or "").strip().lower() == "llm":
                chat_backend = self._get_graph_chat_backend()
            extracted = await extract_graph_facts_async(
                text,
                extraction_mode=self.config.graph.extraction_mode,
                chat_backend=chat_backend,
                max_entities=self.config.graph.max_entities_per_chunk,
                max_relations=self.config.graph.max_relations_per_chunk,
                temperature=float(self.config.graph.llm_temperature),
                max_tokens=int(self.config.graph.llm_max_tokens),
            )
            self._apply_graph_extraction(
                chunk_id=chunk_id,
                extracted=extracted,
                metadata=metadata,
            )
        except Exception:
            return

    def _apply_graph_extraction(
        self,
        *,
        chunk_id: str,
        extracted: ExtractedGraph,
        metadata: dict[str, Any] | None,
    ) -> None:
        entity_ids: dict[str, str] = {}
        for name in extracted.entities:
            eid = self.sqlite.upsert_entity(name, metadata={"source_chunk_id": chunk_id})
            base_conf = extracted.entity_confidence.get(name, float(self.config.graph.mention_base_confidence))
            mention_conf = self._calibrate_graph_confidence(
                base=base_conf,
                metadata=metadata,
                kind="mention",
            )
            self.sqlite.add_entity_mention(eid, chunk_id, confidence=mention_conf)
            entity_ids[name] = eid
        for src_name, relation, dst_name in extracted.relations:
            src_id = entity_ids.get(src_name) or self.sqlite.upsert_entity(src_name)
            dst_id = entity_ids.get(dst_name) or self.sqlite.upsert_entity(dst_name)
            invalidate = relation in {"is", "prefers", "owns"}
            rel_key = (src_name, relation, dst_name)
            base_edge_conf = extracted.relation_confidence.get(
                rel_key,
                float(self.config.graph.edge_base_confidence),
            )
            edge_conf = self._calibrate_graph_confidence(
                base=base_edge_conf,
                metadata=metadata,
                kind="edge",
            )
            self.sqlite.add_edge(
                src_id,
                relation=relation,
                dst_entity_id=dst_id,
                source_chunk_id=chunk_id,
                confidence=edge_conf,
                metadata={
                    "source": metadata or {},
                    "extract_mode": extracted.mode,
                },
                invalidate_existing_relation=invalidate,
            )

    def _calibrate_graph_confidence(
        self,
        *,
        base: float,
        metadata: dict[str, Any] | None,
        kind: str,
    ) -> float:
        score = float(base)
        meta = metadata or {}
        kind_l = kind.lower()

        entry_type = str(meta.get("type", "")).lower()
        if entry_type == "reasoning_entry":
            score += float(self.config.graph.reasoning_confidence_boost)
        if "evidence" in entry_type:
            score += float(self.config.graph.evidence_confidence_boost)
        if meta.get("reasoning_entry_id"):
            score += float(self.config.graph.reasoning_confidence_boost) * 0.5
        if meta.get("session_id"):
            score += 0.02
        if kind_l == "mention":
            score *= 0.95
        elif kind_l == "edge":
            score *= 1.0

        min_conf = max(0.05, float(self.config.graph.min_confidence))
        return max(min_conf, min(0.99, score))

    def _get_graph_chat_backend(self):
        if self._graph_chat is not None:
            return self._graph_chat
        provider = (self.config.graph.llm_provider or "venice").strip().lower()
        kwargs: dict[str, Any] = {
            "temperature": float(self.config.graph.llm_temperature),
            "max_tokens": int(self.config.graph.llm_max_tokens),
        }
        if self.config.graph.llm_model:
            kwargs["model"] = self.config.graph.llm_model
        if provider in {"venice", "default"}:
            kwargs.setdefault("api_key", self.config.venice.api_key)
            kwargs.setdefault("base_url", self.config.venice.base_url)
            kwargs.setdefault("timeout", self.config.venice.chat_timeout)
        elif provider == "openai":
            import os

            kwargs.setdefault("api_key", os.environ.get("OPENAI_API_KEY", ""))
        elif provider == "anthropic":
            import os

            kwargs.setdefault("api_key", os.environ.get("ANTHROPIC_API_KEY", ""))
        self._graph_chat = create_chat_backend(provider=provider, **kwargs)
        return self._graph_chat

    def _get_consolidation_chat_backend(self):
        if self._consolidation_chat is not None:
            return self._consolidation_chat
        provider = (self.config.consolidation.llm_provider or "venice").strip().lower()
        kwargs: dict[str, Any] = {
            "temperature": float(self.config.consolidation.llm_temperature),
            "max_tokens": int(self.config.consolidation.llm_max_tokens),
        }
        if self.config.consolidation.llm_model:
            kwargs["model"] = self.config.consolidation.llm_model
        if provider in {"venice", "default"}:
            kwargs.setdefault("api_key", self.config.venice.api_key)
            kwargs.setdefault("base_url", self.config.venice.base_url)
            kwargs.setdefault("timeout", self.config.venice.chat_timeout)
        elif provider == "openai":
            import os

            kwargs.setdefault("api_key", os.environ.get("OPENAI_API_KEY", ""))
        elif provider == "anthropic":
            import os

            kwargs.setdefault("api_key", os.environ.get("ANTHROPIC_API_KEY", ""))
        self._consolidation_chat = create_chat_backend(provider=provider, **kwargs)
        return self._consolidation_chat

    def _cluster_key(self, chunks: list[Chunk], session_id: str | None = None) -> str:
        sid = str(session_id or "global")
        ids = sorted(c.id for c in chunks)
        sig = hashlib.sha1("|".join(ids).encode("utf-8", errors="ignore")).hexdigest()[:20]
        return f"{sid}:{sig}"

    def _extract_cluster_facts(self, chunks: list[Chunk]) -> list[str]:
        scored: list[tuple[float, str]] = []
        seen: set[str] = set()
        for chunk in chunks:
            sentences = re.split(r"(?<=[\.\!\?])\s+", chunk.content.replace("\n", " "))
            for sentence in sentences:
                s = sentence.strip().strip(".")
                if len(s) < 24:
                    continue
                norm = s.lower()
                if norm in seen:
                    continue
                seen.add(norm)
                score = 0.0
                if re.search(r"\b\d+(\.\d+)?(%|x|ms|s|m|h|d)?\b", s):
                    score += 1.2
                if re.search(r"\b(is|are|was|were|has|have|uses|depends|causes|prefers|owns)\b", norm):
                    score += 1.0
                if re.search(r"\b(should|must|always|never|before|after)\b", norm):
                    score += 0.7
                score += min(len(s) / 240.0, 0.6)
                scored.append((score, s))

        scored.sort(key=lambda x: x[0], reverse=True)
        facts: list[str] = []
        for _, sentence in scored:
            facts.append(sentence)
            if len(facts) >= 12:
                break
        return facts

    async def _extract_cluster_facts_async(self, chunks: list[Chunk]) -> tuple[list[str], bool]:
        heuristic = self._extract_cluster_facts(chunks)
        if not self.config.consolidation.use_llm_enrichment:
            return heuristic, False
        try:
            chat = self._get_consolidation_chat_backend()
            merged = "\n\n".join(c.content.strip() for c in chunks[:10])
            payload = (
                "Extract key stable facts from these episodic memory chunks.\n"
                "Return strict JSON with key 'facts' as a list of concise factual statements.\n"
                "Do not include speculation.\n"
                f"Chunks:\n{merged[:8000]}"
            )
            from c3ae.llm.venice_chat import Message

            resp = await chat.chat(
                [
                    Message(role="system", content="You extract concise factual memory notes."),
                    Message(role="user", content=payload),
                ],
                temperature=float(self.config.consolidation.llm_temperature),
                max_tokens=int(self.config.consolidation.llm_max_tokens),
                json_mode=True,
            )
            data = self._parse_json_object(resp.content)
            rows = data.get("facts", [])
            out: list[str] = []
            if isinstance(rows, list):
                for row in rows:
                    fact = str(row).strip().strip(".")
                    if len(fact) < 20:
                        continue
                    if fact not in out:
                        out.append(fact)
                    if len(out) >= 12:
                        break
            if out:
                return out, True
        except Exception:
            pass
        return heuristic, False

    def _summarize_cluster(self, chunks: list[Chunk], facts: list[str]) -> str:
        if not chunks:
            return ""
        head = chunks[0].content.strip().replace("\n", " ")
        if len(head) > 180:
            head = head[:177] + "..."
        entities = self.sqlite.get_entities_for_chunk_ids([c.id for c in chunks])
        flat_entities: dict[str, int] = {}
        for names in entities.values():
            for n in names:
                flat_entities[n] = flat_entities.get(n, 0) + 1
        top_entities = [k for k, _ in sorted(flat_entities.items(), key=lambda x: x[1], reverse=True)[:3]]
        if facts:
            if top_entities:
                return (
                    f"{head} (consolidated {len(chunks)} episodes; "
                    f"{len(facts)} key facts; entities: {', '.join(top_entities)})"
                )
            return f"{head} (consolidated {len(chunks)} episodes; {len(facts)} key facts)"
        return f"{head} (consolidated {len(chunks)} episodes)"

    async def _summarize_cluster_async(self, chunks: list[Chunk], facts: list[str]) -> tuple[str, bool]:
        heuristic = self._summarize_cluster(chunks, facts)
        if not self.config.consolidation.use_llm_enrichment:
            return heuristic, False
        try:
            chat = self._get_consolidation_chat_backend()
            payload = {
                "facts": facts[:12],
                "chunk_count": len(chunks),
                "sample_chunks": [c.content[:260] for c in chunks[:5]],
            }
            from c3ae.llm.venice_chat import Message

            resp = await chat.chat(
                [
                    Message(
                        role="system",
                        content="You produce one concise semantic memory summary from clustered episodes.",
                    ),
                    Message(
                        role="user",
                        content=(
                            "Return strict JSON: {\"summary\":\"...\"}\n"
                            f"Input: {payload}"
                        ),
                    ),
                ],
                temperature=float(self.config.consolidation.llm_temperature),
                max_tokens=int(self.config.consolidation.llm_max_tokens),
                json_mode=True,
            )
            data = self._parse_json_object(resp.content)
            summary = str(data.get("summary", "")).strip()
            if summary:
                return summary, True
        except Exception:
            pass
        return heuristic, False

    @staticmethod
    def _token_set(text: str) -> set[str]:
        tokens = re.findall(r"[a-z0-9_\-]{3,}", text.lower())
        return {t for t in tokens if not t.isdigit()}

    def _propose_skill_candidates(self, limit: int = 10) -> list[dict[str, Any]]:
        rows = self.sqlite.list_consolidated_memories(limit=200)
        verbs = {
            "run", "check", "verify", "build", "deploy", "monitor", "test",
            "summarize", "analyze", "track", "compare", "audit",
        }
        min_cluster = max(2, int(self.config.consolidation.skill_promotion_min_cluster_size))
        candidates: list[dict[str, Any]] = []
        for row in rows:
            metadata = row.get("metadata", {})
            cluster_size = int(metadata.get("cluster_size", 0))
            if cluster_size < min_cluster:
                continue
            facts = [str(x).strip() for x in row.get("facts", []) if str(x).strip()]
            text = f"{row.get('summary', '')} {' '.join(facts)}".lower()
            present_verbs = sorted(v for v in verbs if re.search(rf"\b{re.escape(v)}\b", text))
            if not present_verbs:
                continue
            keywords = [t for t in sorted(self._token_set(text)) if len(t) >= 4][:3]
            if not keywords:
                keywords = ["workflow"]
            name = " ".join(k.title() for k in keywords)
            procedure_lines = facts[:5] if facts else [str(row.get("summary", "")).strip()]
            procedure = "\n".join(f"- {line}" for line in procedure_lines if line)
            score = min(0.99, 0.18 * cluster_size + 0.06 * len(present_verbs) + 0.03 * len(facts))
            candidates.append(
                {
                    "name": name,
                    "score": round(score, 4),
                    "cluster_key": row.get("cluster_key", ""),
                    "verbs": present_verbs,
                    "procedure": procedure,
                }
            )
        candidates.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return candidates[:limit]

    def _dream_recompression_preview(self, limit: int = 200) -> dict[str, Any]:
        rows = self.sqlite.list_consolidated_memories(limit=limit)
        if not rows:
            return {"eligible_memories": 0, "compression_ratio": 0.0}
        payload = [
            {
                "id": r.get("id", ""),
                "summary": r.get("summary", ""),
                "facts": r.get("facts", []),
                "source_chunk_ids": r.get("source_chunk_ids", []),
            }
            for r in rows
        ]
        try:
            _, stats = self.compress_memories(payload)
            return {
                "eligible_memories": len(rows),
                "compression_ratio": round(float(stats.get("ratio", 0.0)), 4),
                "original_size": int(stats.get("original_size", 0)),
                "compressed_size": int(stats.get("compressed_size", 0)),
            }
        except Exception:
            return {"eligible_memories": len(rows), "compression_ratio": 0.0}

    def _dream_novelty_estimate(self, limit: int = 200) -> dict[str, Any]:
        rows = self.sqlite.list_consolidated_memories(limit=limit)
        total_refs = 0
        unique_refs: set[str] = set()
        for row in rows:
            ids = [str(x) for x in row.get("source_chunk_ids", []) if str(x)]
            total_refs += len(ids)
            unique_refs.update(ids)
        if total_refs == 0:
            return {"novelty_ratio": 0.0, "unique_source_chunks": 0, "total_references": 0}
        ratio = len(unique_refs) / total_refs
        return {
            "novelty_ratio": round(ratio, 4),
            "unique_source_chunks": len(unique_refs),
            "total_references": total_refs,
        }

    @staticmethod
    def _parse_json_object(raw: str) -> dict[str, Any]:
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
            import json

            data = json.loads(text)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

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
        await self.write_manager.close()
        for chat_obj in (self._graph_chat, self._consolidation_chat):
            if chat_obj is None:
                continue
            close_fn = getattr(chat_obj, "close", None)
            if close_fn is None:
                continue
            maybe = close_fn()
            if inspect.isawaitable(maybe):
                await maybe
        self._graph_chat = None
        self._consolidation_chat = None
        await self.embedder.close()
        self.sqlite.close()
        if self.config.faiss_dir:
            self.faiss.save()
