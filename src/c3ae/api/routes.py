"""FastAPI HTTP API for C3/Ae."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field

from c3ae.config import Config
from c3ae.memory_spine.spine import MemorySpine


# --- Request/Response Models ---

class SearchRequest(BaseModel):
    query: str
    top_k: int = 20
    keyword_only: bool = False


class SearchResultItem(BaseModel):
    id: str
    content: str
    score: float
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    results: list[SearchResultItem]
    count: int


class IngestRequest(BaseModel):
    text: str
    source_id: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    chunk_ids: list[str]
    count: int


class KnowledgeRequest(BaseModel):
    title: str
    content: str
    tags: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    session_id: str | None = None
    bypass_governance: bool = False


class KnowledgeResponse(BaseModel):
    id: str
    title: str
    status: str


class EvidenceRequest(BaseModel):
    claim: str
    sources: list[str]
    confidence: float = 0.0
    reasoning: str = ""


class EvidenceResponse(BaseModel):
    id: str
    claim: str


class SkillRequest(BaseModel):
    name: str
    description: str
    procedure: str
    tags: list[str] = Field(default_factory=list)


class SkillResponse(BaseModel):
    id: str
    name: str


class StatusResponse(BaseModel):
    chunks: int
    vectors: int
    reasoning_entries: int
    skills: int
    vault_documents: int


class AuditEventItem(BaseModel):
    id: str
    action: str
    target_type: str
    target_id: str
    detail: str
    outcome: str
    created_at: str


class AuditResponse(BaseModel):
    events: list[AuditEventItem]
    count: int


class RecallRequest(BaseModel):
    """Request to recall relevant memories for a given context."""
    query: str
    top_k: int = 10
    session_filter: str | None = None


class RecallItem(BaseModel):
    content: str
    role: str
    session_id: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class RecallResponse(BaseModel):
    memories: list[RecallItem]
    count: int
    query: str


class AugmentRequest(BaseModel):
    """Request pre-formatted memory context for LLM injection."""
    query: str
    top_k: int = 5
    min_score: float = 0.005
    format: str = "xml"  # "xml" or "plain"
    roles: list[str] | None = None  # Filter by role; default ["user", "assistant"]


class AugmentResponse(BaseModel):
    context: str
    count: int
    memories: list[RecallItem]


class SessionIngestRequest(BaseModel):
    path: str


class SessionIngestResponse(BaseModel):
    session_id: str
    chunks_ingested: int
    roles: dict[str, int]


class SessionListItem(BaseModel):
    session_id: str
    source: str
    chunks: int


class ReasonRequest(BaseModel):
    task: str
    steps: list[dict[str, Any]] | None = None
    max_steps: int = 10


class ReasonResponse(BaseModel):
    session_id: str
    steps_executed: int
    entries_written: int
    entries_blocked: int
    final_answer: str


# --- Supersede Knowledge ---

class SupersedeRequest(BaseModel):
    old_id: str
    new_title: str
    new_content: str
    tags: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


# --- Session Management ---

class SessionStartRequest(BaseModel):
    session_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)


# --- App factory ---

_spine: MemorySpine | None = None


def get_spine() -> MemorySpine:
    if _spine is None:
        raise HTTPException(status_code=500, detail="Memory spine not initialized")
    return _spine


def create_app(data_dir: str | None = None) -> FastAPI:
    global _spine
    config = Config()
    if data_dir:
        config.data_dir = Path(data_dir)
    _spine = MemorySpine(config)

    app = FastAPI(
        title="C3/Ae Memory API",
        version="0.1.0",
        default_response_class=ORJSONResponse,
    )

    # Bearer token auth middleware
    token = config.api.bearer_token

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        if request.url.path in ("/api/v1/health", "/docs", "/redoc", "/openapi.json"):
            return await call_next(request)
        if token:
            auth = request.headers.get("Authorization", "")
            if not auth.startswith("Bearer ") or auth[7:] != token:
                return ORJSONResponse({"detail": "Unauthorized"}, status_code=401)
        return await call_next(request)

    # --- Routes ---

    @app.get("/api/v1/health")
    async def health():
        return {"status": "ok", "service": "c3ae"}

    @app.get("/api/v1/status", response_model=StatusResponse)
    async def get_status(spine: MemorySpine = Depends(get_spine)):
        return spine.status()

    @app.post("/api/v1/memory/search", response_model=SearchResponse)
    async def memory_search(req: SearchRequest, spine: MemorySpine = Depends(get_spine)):
        if req.keyword_only:
            results = spine.search_keyword(req.query, top_k=req.top_k)
        else:
            results = await spine.search(req.query, top_k=req.top_k)
        return SearchResponse(
            results=[
                SearchResultItem(
                    id=r.id, content=r.content, score=r.score,
                    source=r.source, metadata=r.metadata,
                )
                for r in results
            ],
            count=len(results),
        )

    @app.post("/api/v1/memory/ingest", response_model=IngestResponse)
    async def memory_ingest(req: IngestRequest, spine: MemorySpine = Depends(get_spine)):
        import time as _time
        last_err = None
        for attempt in range(3):
            try:
                chunk_ids = await spine.ingest_text(req.text, source_id=req.source_id, metadata=req.metadata)
                return IngestResponse(chunk_ids=chunk_ids, count=len(chunk_ids))
            except sqlite3.OperationalError as e:
                last_err = e
                if "locked" in str(e) and attempt < 2:
                    _time.sleep(1.0 * (attempt + 1))
                    continue
                raise
            except Exception:
                raise
        raise last_err  # type: ignore[misc]

    @app.post("/api/v1/reasoning/add", response_model=KnowledgeResponse)
    async def reasoning_add(req: KnowledgeRequest, spine: MemorySpine = Depends(get_spine)):
        try:
            entry = await spine.add_knowledge(
                title=req.title, content=req.content,
                tags=req.tags, evidence_ids=req.evidence_ids,
                session_id=req.session_id,
                bypass_governance=req.bypass_governance,
            )
            return KnowledgeResponse(id=entry.id, title=entry.title, status=entry.status.value)
        except Exception as e:
            raise HTTPException(status_code=422, detail=str(e))

    @app.get("/api/v1/reasoning/list")
    async def reasoning_list(limit: int = 100, spine: MemorySpine = Depends(get_spine)):
        entries = spine.bank.list_active(limit=limit)
        return {
            "entries": [
                {"id": e.id, "title": e.title, "tags": e.tags, "created_at": str(e.created_at)}
                for e in entries
            ],
            "count": len(entries),
        }

    @app.post("/api/v1/evidence/add", response_model=EvidenceResponse)
    async def evidence_add(req: EvidenceRequest, spine: MemorySpine = Depends(get_spine)):
        pack = spine.add_evidence(req.claim, req.sources, req.confidence, req.reasoning)
        return EvidenceResponse(id=pack.id, claim=pack.claim)

    @app.post("/api/v1/skills/add", response_model=SkillResponse)
    async def skill_add(req: SkillRequest, spine: MemorySpine = Depends(get_spine)):
        capsule = spine.register_skill(req.name, req.description, req.procedure, req.tags)
        return SkillResponse(id=capsule.id, name=capsule.name)

    @app.get("/api/v1/skills/list")
    async def skill_list(limit: int = 100, spine: MemorySpine = Depends(get_spine)):
        capsules = spine.skills.list_all(limit=limit)
        return {
            "skills": [
                {"id": c.id, "name": c.name, "description": c.description, "tags": c.tags}
                for c in capsules
            ],
            "count": len(capsules),
        }

    @app.get("/api/v1/audit", response_model=AuditResponse)
    async def audit_log(limit: int = 100, target_type: str | None = None,
                        spine: MemorySpine = Depends(get_spine)):
        events = spine.audit.recent(limit=limit, target_type=target_type)
        return AuditResponse(
            events=[
                AuditEventItem(
                    id=e.id, action=e.action, target_type=e.target_type,
                    target_id=e.target_id, detail=e.detail, outcome=e.outcome,
                    created_at=str(e.created_at),
                )
                for e in events
            ],
            count=len(events),
        )

    # --- Memory Recall (the key integration point for agents) ---

    @app.post("/api/v1/memory/recall", response_model=RecallResponse)
    async def memory_recall(req: RecallRequest, spine: MemorySpine = Depends(get_spine)):
        """Search memory for content relevant to the given query.

        This is the primary endpoint for agents to recall past sessions,
        conversations, and knowledge. Uses hybrid search (Venice embeddings
        + FTS5 keyword) when available, falls back to keyword-only.
        """
        try:
            results = await spine.search(req.query, top_k=req.top_k * 2)
        except Exception:
            # Fall back to keyword-only if embedding fails
            results = spine.search_keyword(req.query, top_k=req.top_k)

        memories = []
        seen_content: set[str] = set()
        for r in results:
            meta = r.metadata or {}
            role = meta.get("role", "unknown")
            session_id = meta.get("session_id", "")

            # Filter by session if requested
            if req.session_filter and session_id and req.session_filter not in session_id:
                continue

            # Content dedup
            content_key = r.content.strip().lower()[:200]
            if content_key in seen_content:
                continue
            seen_content.add(content_key)

            memories.append(RecallItem(
                content=r.content,
                role=role,
                session_id=session_id,
                score=r.score,
                metadata=meta,
            ))
            if len(memories) >= req.top_k:
                break

        return RecallResponse(
            memories=memories,
            count=len(memories),
            query=req.query,
        )

    # --- Memory Augment (pre-formatted context for LLM injection) ---

    @app.post("/api/v1/memory/augment", response_model=AugmentResponse)
    async def memory_augment(req: AugmentRequest, spine: MemorySpine = Depends(get_spine)):
        """Return pre-formatted memory context ready for LLM injection.

        This is the product API for automatic context augmentation.
        Any integration (OpenClaw, LangChain, custom agents) can call this
        to get relevant memories formatted for injection into LLM prompts.
        """
        allowed_roles = set(req.roles or ["user", "assistant"])

        try:
            results = await spine.search(req.query, top_k=req.top_k * 10)
        except Exception:
            results = spine.search_keyword(req.query, top_k=req.top_k * 5)

        memories = []
        seen_content: set[str] = set()
        for r in results:
            meta = r.metadata or {}
            role = meta.get("role", "unknown")
            session_id = meta.get("session_id", "")
            if r.score < req.min_score:
                continue
            if role not in allowed_roles:
                continue
            # Content dedup — skip near-identical memories
            content_key = r.content.strip().lower()[:200]
            if content_key in seen_content:
                continue
            seen_content.add(content_key)
            memories.append(RecallItem(
                content=r.content,
                role=role,
                session_id=session_id,
                score=r.score,
                metadata=meta,
            ))
            if len(memories) >= req.top_k:
                break

        if not memories:
            return AugmentResponse(context="", count=0, memories=[])

        if req.format == "xml":
            lines = []
            for m in memories:
                tag = m.role if m.role != "unknown" else "memory"
                lines.append(f"  <{tag}>{m.content}</{tag}>")
            context = "<relevant-memories>\n" + "\n".join(lines) + "\n</relevant-memories>"
        else:
            lines = [f"- [{m.role}] {m.content}" for m in memories]
            context = "Relevant memories:\n" + "\n".join(lines)

        return AugmentResponse(context=context, count=len(memories), memories=memories)

    # --- Session Ingestion ---

    @app.post("/api/v1/sessions/ingest", response_model=SessionIngestResponse)
    async def session_ingest(req: SessionIngestRequest,
                             spine: MemorySpine = Depends(get_spine)):
        """Ingest a session file into searchable memory."""
        from pathlib import Path
        path = Path(req.path)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {req.path}")
        result = spine.ingest_session(path)
        return SessionIngestResponse(**result)

    @app.get("/api/v1/sessions/list")
    async def session_list(limit: int = 50, spine: MemorySpine = Depends(get_spine)):
        """List all ingested sessions with chunk counts."""
        rows = spine.sqlite._conn.execute(
            """SELECT json_extract(metadata, '$.session_id') as session_id,
                      json_extract(metadata, '$.source_file') as source_file,
                      COUNT(*) as chunk_count
               FROM chunks
               WHERE source_id LIKE 'session:%'
               GROUP BY json_extract(metadata, '$.session_id')
               ORDER BY chunk_count DESC
               LIMIT ?""",
            (limit,)
        ).fetchall()

        sessions = []
        for row in rows:
            sid = row["session_id"]
            if sid:
                sessions.append({
                    "session_id": sid,
                    "source": row["source_file"] or "",
                    "chunks": row["chunk_count"],
                })

        return {"sessions": sessions, "count": len(sessions)}

    # --- Extended Status ---

    @app.get("/api/v1/status/full")
    async def full_status(spine: MemorySpine = Depends(get_spine)):
        """Full system status with direct chunk count from SQLite."""
        status = spine.status()
        row = spine.sqlite._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        status["chunks"] = int(row[0]) if row else 0
        status["service"] = "novaspine"
        return status

    # --- Reasoning Bank: Supersede ---

    @app.post("/api/v1/reasoning/supersede", response_model=KnowledgeResponse)
    async def reasoning_supersede(req: SupersedeRequest,
                                  spine: MemorySpine = Depends(get_spine)):
        """Supersede an old reasoning entry with a new version."""
        try:
            entry = spine.supersede_knowledge(
                req.old_id, req.new_title, req.new_content,
                tags=req.tags, evidence_ids=req.evidence_ids,
            )
            return KnowledgeResponse(id=entry.id, title=entry.title, status=entry.status.value)
        except Exception as e:
            raise HTTPException(status_code=422, detail=str(e))

    # --- Session Management ---

    @app.post("/api/v1/sessions/start")
    async def session_start(req: SessionStartRequest,
                            spine: MemorySpine = Depends(get_spine)):
        """Start a new tracked session."""
        sid = spine.start_session(req.session_id, metadata=req.metadata)
        return {"session_id": sid, "status": "started"}

    @app.post("/api/v1/sessions/end")
    async def session_end(session_id: str, spine: MemorySpine = Depends(get_spine)):
        """End a tracked session."""
        spine.end_session(session_id)
        return {"session_id": session_id, "status": "ended"}


    return app
