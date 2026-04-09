"""FastAPI HTTP API for NovaSpine."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from c3ae.config import Config
from c3ae.memory_spine.spine import MemorySpine
from c3ae.wiki_layer import NovaSpineWiki


AUTH_EXEMPT_PATHS = ("/api/v1/health", "/docs", "/redoc", "/openapi.json")


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
    service: str = "novaspine"
    chunks: int
    vectors: int
    reasoning_entries: int
    skills: int
    vault_documents: int
    consolidated_memories: int | None = None
    graph: dict[str, Any] | None = None


def _status_payload(spine: MemorySpine) -> dict[str, Any]:
    payload = dict(spine.status())
    payload["service"] = "novaspine"
    return payload


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
    id: str = ""
    content: str
    role: str
    session_id: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class RecallResponse(BaseModel):
    memories: list[RecallItem]
    count: int
    query: str


class ExplainRequest(BaseModel):
    query: str
    top_k: int = 5
    session_filter: str | None = None


class RecallExplainItem(RecallItem):
    why_recalled: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)


class ExplainResponse(BaseModel):
    memories: list[RecallExplainItem]
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


class FactItem(BaseModel):
    id: str
    source_chunk_id: str
    entity: str
    relation: str
    value: str
    date: str = ""
    confidence: float = 0.0
    status: str = "current"
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = ""


class CurrentFactsResponse(BaseModel):
    facts: list[FactItem]
    count: int


class FactTruthGroup(BaseModel):
    entity: str
    relation: str
    current_facts: list[FactItem] = Field(default_factory=list)
    historical_facts: list[FactItem] = Field(default_factory=list)


class FactTruthResponse(BaseModel):
    fact_groups: list[FactTruthGroup]
    count: int


class FactConflictItem(BaseModel):
    entity: str
    relation: str
    value_count: int
    current_facts: list[FactItem] = Field(default_factory=list)
    historical_facts: list[FactItem] = Field(default_factory=list)


class FactConflictsResponse(BaseModel):
    conflicts: list[FactConflictItem]
    count: int


class FactResolveRequest(BaseModel):
    winner_fact_id: str
    loser_fact_ids: list[str] = Field(default_factory=list)
    reason: str = ""
    user_confirmation: str = ""
    resolution_ticket_id: str = ""


class FactResolveResponse(BaseModel):
    ok: bool
    winner_fact: FactItem
    superseded_facts: list[FactItem]
    resolved_at: str
    resolution_id: str


class WikiStatusResponse(BaseModel):
    service: str
    generated_at: str
    vault_root: str
    entity_pages: int
    current_claims: int
    historical_claims: int
    conflicts: int
    low_confidence: int
    open_questions: int
    reports: dict[str, str] = Field(default_factory=dict)
    cache: dict[str, str] = Field(default_factory=dict)


class WikiSearchRequest(BaseModel):
    query: str
    limit: int = 10


class WikiSearchItem(BaseModel):
    kind: str
    id: str
    title: str
    path: str = ""
    score: float = 0.0
    preview: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class WikiSearchResponse(BaseModel):
    ok: bool = True
    query: str
    count: int
    results: list[WikiSearchItem] = Field(default_factory=list)
    status: dict[str, Any] = Field(default_factory=dict)


class WikiGetResponse(BaseModel):
    ok: bool = True
    id: str
    entity: str = ""
    title: str = ""
    path: str = ""
    absolute_path: str = ""
    content: str = ""
    summary: str = ""
    claims: list[dict[str, Any]] = Field(default_factory=list)
    current_claims: list[dict[str, Any]] = Field(default_factory=list)
    historical_claims: list[dict[str, Any]] = Field(default_factory=list)
    conflict_relations: list[str] = Field(default_factory=list)
    manual: dict[str, Any] = Field(default_factory=dict)


class WikiApplyRequest(BaseModel):
    entity: str
    summary: str | None = None
    note: str | None = None
    open_questions: list[str] | None = None
    tags: list[str] | None = None


class WikiLintResponse(BaseModel):
    ok: bool = True
    status: dict[str, Any] = Field(default_factory=dict)
    counts: dict[str, int] = Field(default_factory=dict)
    conflicts: list[dict[str, Any]] = Field(default_factory=list)
    low_confidence: list[dict[str, Any]] = Field(default_factory=list)
    missing_evidence: list[dict[str, Any]] = Field(default_factory=list)
    reports: dict[str, str] = Field(default_factory=dict)


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


# --- Context Compaction Models ---

class CompressPromptRequest(BaseModel):
    prompt: str


class CompressPromptResponse(BaseModel):
    compressed: str
    token_savings: float
    refs_used: int
    original_tokens: int


class ExpandRequest(BaseModel):
    text: str


class ExpandResponse(BaseModel):
    expanded: str


# --- Temporal Tracking Models ---

class TrackEventRequest(BaseModel):
    event_type: str


class TrackEventResponse(BaseModel):
    motif: dict[str, Any] | None = None


class TrackEventsBatchRequest(BaseModel):
    events: list[str]


class TrackEventsBatchResponse(BaseModel):
    motifs: list[dict[str, Any]]
    count: int


def _fact_status(fact: dict[str, Any]) -> str:
    metadata = dict(fact.get("metadata") or {})
    status = str(metadata.get("fact_status", "")).strip().lower()
    return status if status in {"current", "historical"} else "current"


def _to_fact_item(fact: dict[str, Any]) -> FactItem:
    return FactItem(
        id=str(fact.get("id", "")),
        source_chunk_id=str(fact.get("source_chunk_id", "")),
        entity=str(fact.get("entity", "")),
        relation=str(fact.get("relation", "")),
        value=str(fact.get("value", "")),
        date=str(fact.get("date", "")),
        confidence=float(fact.get("confidence", 0.0)),
        status=_fact_status(fact),
        metadata=dict(fact.get("metadata") or {}),
        created_at=str(fact.get("created_at", "")),
    )


def _wiki(spine: MemorySpine) -> NovaSpineWiki:
    return NovaSpineWiki(spine)


def _build_recall_explain_item(row: dict[str, Any]) -> RecallExplainItem:
    metadata = dict(row.get("metadata") or {})
    source_kind = str(metadata.get("_source_kind") or metadata.get("type") or row.get("source") or "memory")
    reasons: list[str] = []
    score = float(row.get("score", 0.0))
    if score >= 0.8:
        reasons.append("very high recall score")
    elif score >= 0.5:
        reasons.append("strong recall score")
    elif score >= 0.2:
        reasons.append("moderate recall score")
    if metadata.get("session_id"):
        reasons.append("linked to a prior session")
    if metadata.get("source_file"):
        reasons.append("has a source transcript or file reference")
    if source_kind not in {"", "memory"}:
        reasons.append(f"memory source is {source_kind}")
    return RecallExplainItem(
        id=str(row.get("id", "")),
        content=str(row.get("content", "")),
        role=str(metadata.get("role", "unknown")),
        session_id=str(metadata.get("session_id", "")),
        score=score,
        metadata=metadata,
        why_recalled={"reasons": reasons, "score": score},
        provenance={
            "source_kind": source_kind,
            "session_id": str(metadata.get("session_id", "")),
            "source_file": str(metadata.get("source_file", "")),
            "source_id": str(metadata.get("source_id", "")),
            "created_at": str(metadata.get("_created_at", "")),
        },
    )


# --- Structural Similarity Models ---

class SimilarityRequest(BaseModel):
    data_id_a: str
    data_id_b: str


class SimilarityResponse(BaseModel):
    data_id_a: str
    data_id_b: str
    similarity: float


class FindSimilarResponse(BaseModel):
    data_id: str
    matches: list[dict[str, Any]]
    count: int


# --- Compression Models ---

class CompressMemoriesRequest(BaseModel):
    memories: list[dict[str, Any]]


class CompressResponse(BaseModel):
    compressed_size: int
    original_size: int
    ratio: float
    extra: dict[str, Any] = Field(default_factory=dict)


# --- Supersede Knowledge ---

class SupersedeRequest(BaseModel):
    old_id: str
    new_title: str
    new_content: str
    tags: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class GraphQueryRequest(BaseModel):
    entity: str
    depth: int = 2


class DecayConfigRequest(BaseModel):
    half_life_hours: float


class ConsolidateRequest(BaseModel):
    session_id: str | None = None
    max_chunks: int = 1000


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
        title="NovaSpine API",
        version="0.3.0",
    )

    # Bearer token auth middleware
    token = config.api.bearer_token
    auth_disabled = config.api.auth_disabled

    if not token and not auth_disabled:
        import warnings

        warnings.warn(
            "C3AE_API_TOKEN is not set and C3AE_AUTH_DISABLED is not set. "
            "Non-health and non-docs API routes will return 503 until authentication "
            "is configured. Set C3AE_API_TOKEN, or set C3AE_AUTH_DISABLED=1 for "
            "explicit local unauthenticated access.",
            stacklevel=2,
        )

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        if request.url.path in AUTH_EXEMPT_PATHS:
            return await call_next(request)
        if token:
            auth = request.headers.get("Authorization", "")
            if not auth.startswith("Bearer ") or auth[7:] != token:
                return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        elif not auth_disabled:
            return JSONResponse(
                {"detail": "API authentication not configured. Set C3AE_API_TOKEN."},
                status_code=503,
            )
        return await call_next(request)

    # --- Routes ---

    @app.get("/api/v1/health")
    async def health():
        return {"status": "ok", "service": "novaspine"}

    @app.get("/api/v1/status", response_model=StatusResponse)
    async def get_status(spine: MemorySpine = Depends(get_spine)):
        return _status_payload(spine)

    @app.post("/api/v1/memory/search", response_model=SearchResponse)
    async def memory_search(req: SearchRequest, spine: MemorySpine = Depends(get_spine)):
        """Lower-level search endpoint for tools and debugging.

        Agent integrations should generally prefer `/api/v1/memory/recall`
        or `/api/v1/memory/augment`.
        """
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
        last_err = None
        for attempt in range(3):
            try:
                chunk_ids = await spine.ingest_text(req.text, source_id=req.source_id, metadata=req.metadata)
                return IngestResponse(chunk_ids=chunk_ids, count=len(chunk_ids))
            except Exception as e:
                last_err = e
                if "locked" in str(e) and attempt < 2:
                    await asyncio.sleep(1.0 * (attempt + 1))
                    continue
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

    @app.post("/api/v1/reasoning/run", response_model=ReasonResponse)
    async def reasoning_run(req: ReasonRequest, spine: MemorySpine = Depends(get_spine)):
        from c3ae.pipeline.loop import PipelineLoop
        pipeline = PipelineLoop(spine, max_steps=req.max_steps)
        result = await pipeline.run(req.task, steps=req.steps)
        return ReasonResponse(
            session_id=result.session.session_id,
            steps_executed=len(result.session.steps),
            entries_written=len(result.entries_written),
            entries_blocked=len(result.entries_blocked),
            final_answer=result.session.final_answer,
        )

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
        rows = await spine.recall(
            req.query,
            top_k=req.top_k,
            session_filter=req.session_filter,
        )
        memories = [
            RecallItem(
                id=str(r.get("id", "")),
                content=str(r.get("content", "")),
                role=str((r.get("metadata") or {}).get("role", "unknown")),
                session_id=str((r.get("metadata") or {}).get("session_id", "")),
                score=float(r.get("score", 0.0)),
                metadata=dict(r.get("metadata") or {}),
            )
            for r in rows
        ]

        return RecallResponse(
            memories=memories,
            count=len(memories),
            query=req.query,
        )

    @app.post("/api/v1/memory/explain", response_model=ExplainResponse)
    async def memory_explain(req: ExplainRequest, spine: MemorySpine = Depends(get_spine)):
        rows = await spine.recall(
            req.query,
            top_k=req.top_k,
            session_filter=req.session_filter,
        )
        memories = [_build_recall_explain_item(row) for row in rows]
        return ExplainResponse(
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
        rows = await spine.recall(req.query, top_k=max(req.top_k * 10, req.top_k))
        allowed_roles = set(req.roles or ["user", "assistant"])
        filtered = []
        for r in rows:
            meta = dict(r.get("metadata") or {})
            role = str(meta.get("role", "unknown"))
            if role not in allowed_roles:
                continue
            if float(r.get("score", 0.0)) < req.min_score:
                continue
            filtered.append(r)
            if len(filtered) >= req.top_k:
                break

        memories = [
            RecallItem(
                id=str(r.get("id", "")),
                content=str(r.get("content", "")),
                role=str((r.get("metadata") or {}).get("role", "unknown")),
                session_id=str((r.get("metadata") or {}).get("session_id", "")),
                score=float(r.get("score", 0.0)),
                metadata=dict(r.get("metadata") or {}),
            )
            for r in filtered
        ]
        context = spine._render_augmented_context(
            [{"content": m.content, "role": m.role} for m in memories],
            format=req.format,
        )

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

    # --- Structured Facts / Current Truth ---

    @app.get("/api/v1/facts/current", response_model=CurrentFactsResponse)
    async def facts_current(
        entity: str = "",
        relation: str = "",
        limit: int = 20,
        spine: MemorySpine = Depends(get_spine),
    ):
        facts = spine.sqlite.list_current_structured_facts(
            entity=entity,
            relation=relation,
            limit=max(1, min(limit, 100)),
        )
        items = [_to_fact_item(fact) for fact in facts]
        return CurrentFactsResponse(facts=items, count=len(items))

    @app.get("/api/v1/facts/truth", response_model=FactTruthResponse)
    async def facts_truth(
        entity: str = "",
        relation: str = "",
        limit: int = 20,
        spine: MemorySpine = Depends(get_spine),
    ):
        groups = spine.sqlite.list_structured_truth(
            entity=entity,
            relation=relation,
            limit=max(1, min(limit, 100)),
        )
        items = [
            FactTruthGroup(
                entity=str(group["entity"]),
                relation=str(group["relation"]),
                current_facts=[_to_fact_item(fact) for fact in group["current_facts"]],
                historical_facts=[_to_fact_item(fact) for fact in group["historical_facts"]],
            )
            for group in groups
        ]
        return FactTruthResponse(fact_groups=items, count=len(items))

    @app.get("/api/v1/facts/conflicts", response_model=FactConflictsResponse)
    async def facts_conflicts(limit: int = 20, spine: MemorySpine = Depends(get_spine)):
        conflicts = spine.sqlite.list_structured_fact_conflicts(limit=max(1, min(limit, 100)))
        items = [
            FactConflictItem(
                entity=str(group["entity"]),
                relation=str(group["relation"]),
                value_count=int(group["value_count"]),
                current_facts=[_to_fact_item(fact) for fact in group["current_facts"]],
                historical_facts=[_to_fact_item(fact) for fact in group["historical_facts"]],
            )
            for group in conflicts
        ]
        return FactConflictsResponse(conflicts=items, count=len(items))

    @app.post("/api/v1/facts/resolve", response_model=FactResolveResponse)
    async def facts_resolve(req: FactResolveRequest, spine: MemorySpine = Depends(get_spine)):
        winner = spine.sqlite.get_structured_fact(req.winner_fact_id)
        if not winner:
            raise HTTPException(status_code=404, detail=f"Unknown winner fact id: {req.winner_fact_id}")

        loser_ids = [fact_id for fact_id in req.loser_fact_ids if fact_id and fact_id != req.winner_fact_id]
        if not loser_ids:
            truth_groups = spine.sqlite.list_structured_truth(
                entity=str(winner["entity"]),
                relation=str(winner["relation"]),
                limit=5,
            )
            for group in truth_groups:
                if (
                    str(group["entity"]).lower() == str(winner["entity"]).lower()
                    and str(group["relation"]).lower() == str(winner["relation"]).lower()
                ):
                    loser_ids = [
                        str(fact["id"])
                        for fact in group["current_facts"]
                        if str(fact["id"]) != req.winner_fact_id
                    ]
                    break

        try:
            resolved = spine.sqlite.resolve_structured_fact_conflict(
                winner_fact_id=req.winner_fact_id,
                loser_fact_ids=loser_ids,
                reason=req.reason,
                user_confirmation=req.user_confirmation,
                resolution_ticket_id=req.resolution_ticket_id,
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        return FactResolveResponse(
            ok=True,
            winner_fact=_to_fact_item(dict(resolved["winner_fact"] or {})),
            superseded_facts=[_to_fact_item(fact) for fact in resolved["superseded_facts"]],
            resolved_at=str(resolved["resolved_at"]),
            resolution_id=str(resolved["resolution_id"]),
        )

    # --- Wiki Layer / Durable Knowledge Views ---

    @app.get("/api/v1/wiki/status", response_model=WikiStatusResponse)
    async def wiki_status(spine: MemorySpine = Depends(get_spine)):
        return WikiStatusResponse(**_wiki(spine).compile())

    @app.post("/api/v1/wiki/search", response_model=WikiSearchResponse)
    async def wiki_search(req: WikiSearchRequest, spine: MemorySpine = Depends(get_spine)):
        payload = _wiki(spine).search(req.query, limit=max(1, min(int(req.limit or 10), 50)))
        return WikiSearchResponse(
            ok=bool(payload.get("ok", True)),
            query=str(payload.get("query", "")),
            count=int(payload.get("count", 0)),
            results=[
                WikiSearchItem(
                    kind=str(item.get("kind", "")),
                    id=str(item.get("id", "")),
                    title=str(item.get("title", "")),
                    path=str(item.get("path", "")),
                    score=float(item.get("score", 0.0)),
                    preview=str(item.get("preview", "")),
                    metadata=dict(item.get("metadata") or {}),
                )
                for item in payload.get("results", [])
            ],
            status=dict(payload.get("status") or {}),
        )

    @app.get("/api/v1/wiki/get", response_model=WikiGetResponse)
    async def wiki_get(
        entity: str = "",
        path: str = "",
        claim_id: str = "",
        spine: MemorySpine = Depends(get_spine),
    ):
        page = _wiki(spine).get_page(entity=entity, path=path, claim_id=claim_id)
        if not page:
            raise HTTPException(status_code=404, detail="Wiki page or claim not found")
        return WikiGetResponse(**page)

    @app.post("/api/v1/wiki/apply", response_model=WikiGetResponse)
    async def wiki_apply(req: WikiApplyRequest, spine: MemorySpine = Depends(get_spine)):
        try:
            page = _wiki(spine).apply_page_update(
                entity=req.entity,
                summary=req.summary,
                note=req.note,
                open_questions=req.open_questions,
                tags=req.tags,
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return WikiGetResponse(**page)

    @app.get("/api/v1/wiki/lint", response_model=WikiLintResponse)
    async def wiki_lint(limit: int = 20, spine: MemorySpine = Depends(get_spine)):
        payload = _wiki(spine).lint(limit=max(1, min(limit, 100)))
        return WikiLintResponse(**payload)

    # --- Extended Status ---

    @app.get("/api/v1/status/full")
    async def full_status(spine: MemorySpine = Depends(get_spine)):
        """Full system status including all cogdedup modules."""
        return _status_payload(spine)

    @app.post("/api/v1/memory/consolidate")
    async def memory_consolidate(req: ConsolidateRequest, spine: MemorySpine = Depends(get_spine)):
        """Run episodic -> semantic consolidation."""
        result = await spine.consolidate_async(session_id=req.session_id, max_chunks=req.max_chunks)
        return result

    @app.post("/api/v1/memory/dream")
    async def memory_dream(spine: MemorySpine = Depends(get_spine)):
        """Run an offline-style dream consolidation pass."""
        return await spine.dream_consolidate_async()

    @app.post("/api/v1/memory/forget-preview")
    async def memory_forget_preview(
        max_age_days: int = 120,
        max_access_count: int = 0,
        limit: int = 1000,
        spine: MemorySpine = Depends(get_spine),
    ):
        """Preview stale-memory forgetting candidates."""
        return spine.forget_stale(
            max_age_days=max_age_days,
            max_access_count=max_access_count,
            limit=limit,
            dry_run=True,
        )

    @app.post("/api/v2/graph/query")
    async def graph_query(req: GraphQueryRequest, spine: MemorySpine = Depends(get_spine)):
        """V2 graph retrieval endpoint."""
        graph = await spine.graph_query(req.entity, depth=req.depth)
        graph["protocol_version"] = "v2"
        return graph

    @app.post("/api/v2/decay/config")
    async def set_decay_config(req: DecayConfigRequest, spine: MemorySpine = Depends(get_spine)):
        """V2 decay tuning endpoint."""
        spine.set_decay_config(req.half_life_hours)
        return {"protocol_version": "v2", "half_life_hours": spine.config.retrieval.decay_half_life_hours}

    # --- Context Compaction (LLM Prompt Compression) ---

    @app.post("/api/v1/compression/prompt", response_model=CompressPromptResponse)
    async def compress_prompt(req: CompressPromptRequest,
                              spine: MemorySpine = Depends(get_spine)):
        """Compress an LLM prompt by replacing known chunks with REF placeholders."""
        result = spine.compress_prompt(req.prompt)
        return CompressPromptResponse(**result)

    @app.post("/api/v1/compression/expand", response_model=ExpandResponse)
    async def expand_response(req: ExpandRequest,
                              spine: MemorySpine = Depends(get_spine)):
        """Expand REF placeholders back to full content."""
        expanded = spine.expand_response(req.text)
        return ExpandResponse(expanded=expanded)

    @app.post("/api/v1/compression/memories", response_model=CompressResponse)
    async def compress_memories(req: CompressMemoriesRequest,
                                spine: MemorySpine = Depends(get_spine)):
        """Compress a batch of memories using cognitive dedup."""
        blob, stats = spine.compress_memories(req.memories)
        return CompressResponse(
            compressed_size=stats.get("compressed_size", len(blob)),
            original_size=stats.get("original_size", 0),
            ratio=stats.get("ratio", 0.0),
            extra=stats,
        )

    @app.post("/api/v1/compression/reasoning-bank", response_model=CompressResponse)
    async def compress_reasoning_bank(spine: MemorySpine = Depends(get_spine)):
        """Compress the entire reasoning bank for archival."""
        blob, stats = spine.compress_reasoning_bank()
        return CompressResponse(
            compressed_size=stats.get("compressed_size", len(blob)),
            original_size=stats.get("original_size", 0),
            ratio=stats.get("ratio", 0.0),
            extra=stats,
        )

    @app.post("/api/v1/compression/audit-log", response_model=CompressResponse)
    async def compress_audit_log(limit: int = 10000,
                                 spine: MemorySpine = Depends(get_spine)):
        """Compress recent audit log entries for archival."""
        blob, stats = spine.compress_audit_log(limit=limit)
        return CompressResponse(
            compressed_size=stats.get("compressed_size", len(blob)),
            original_size=stats.get("original_size", 0),
            ratio=stats.get("ratio", 0.0),
            extra=stats,
        )

    # --- Temporal Event Tracking ---

    @app.post("/api/v1/events/track", response_model=TrackEventResponse)
    async def track_event(req: TrackEventRequest,
                          spine: MemorySpine = Depends(get_spine)):
        """Track a temporal event. Returns motif if a pattern is detected."""
        motif = spine.track_event(req.event_type)
        return TrackEventResponse(motif=motif)

    @app.post("/api/v1/events/track-batch", response_model=TrackEventsBatchResponse)
    async def track_events_batch(req: TrackEventsBatchRequest,
                                 spine: MemorySpine = Depends(get_spine)):
        """Track a batch of events. Returns all detected motifs."""
        motifs = spine.track_events_batch(req.events)
        return TrackEventsBatchResponse(motifs=motifs, count=len(motifs))

    # --- Structural Similarity ---

    @app.post("/api/v1/similarity/compare", response_model=SimilarityResponse)
    async def structural_similarity(req: SimilarityRequest,
                                    spine: MemorySpine = Depends(get_spine)):
        """Compute structural similarity between two memories using shared chunks."""
        score = spine.structural_similarity(req.data_id_a, req.data_id_b)
        return SimilarityResponse(
            data_id_a=req.data_id_a, data_id_b=req.data_id_b, similarity=score,
        )

    @app.get("/api/v1/similarity/find/{data_id}", response_model=FindSimilarResponse)
    async def find_similar(data_id: str, threshold: float = 0.3,
                           spine: MemorySpine = Depends(get_spine)):
        """Find memories structurally similar to the given one."""
        matches = spine.find_structurally_similar(data_id, threshold=threshold)
        return FindSimilarResponse(
            data_id=data_id,
            matches=[{"id": m[0], "similarity": m[1]} for m in matches],
            count=len(matches),
        )

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

    # --- Cogdedup Stats ---

    @app.get("/api/v1/cogdedup/stats")
    async def cogdedup_stats(spine: MemorySpine = Depends(get_spine)):
        """Cognitive dedup statistics — compression store, anomaly, temporal."""
        result: dict[str, Any] = {}
        result["cogstore"] = spine.cogstore.stats()
        if hasattr(spine, '_lazy_anomaly_detector'):
            report = spine._anomaly_detector.drift_report()
            result["anomaly"] = {
                "total_observations": spine._anomaly_detector._observation_count,
                "alerts": report.alerts_count,
                "mean_ratio": round(report.current_mean, 2),
                "std_ratio": round(report.current_std, 2),
            }
        if hasattr(spine, '_lazy_temporal_tracker'):
            motifs = spine._temporal_tracker.detected_motifs()
            result["temporal"] = {
                "motifs_detected": len(motifs),
                "motifs": [{"pattern": m.pattern, "count": m.occurrences} for m in motifs],
            }
        return result

    @app.get("/api/v1/cogdedup/anomaly-report")
    async def anomaly_report(spine: MemorySpine = Depends(get_spine)):
        """Anomaly detection drift report from compression monitoring."""
        report = spine._anomaly_detector.drift_report()
        return {
            "total_observations": spine._anomaly_detector._observation_count,
            "alerts_count": report.alerts_count,
            "current_mean": round(report.current_mean, 2),
            "current_std": round(report.current_std, 2),
            "recent_alerts": [
                {"severity": a.severity, "z_score": round(a.z_score, 2)}
                for a in (report.recent_alerts if hasattr(report, 'recent_alerts') else [])
            ],
        }

    return app
