# C3/Ae Memory System — MVP Handoff Document

## For: Codex or any AI coding agent
## Repo: https://github.com/maddwiz/Nova-v1
## Date: 2026-02-10
## Author: Claude Opus 4.6 (on behalf of Desmond)

---

## 1. WHAT THIS IS

C3/Ae is a **long-term memory system for LLM agents**. It gives any AI agent the ability to remember past conversations, recall relevant context automatically, and build persistent knowledge over time.

**Core value proposition:** "Plug-in long-term memory for any LLM agent framework."

### What works today (production-tested):
- Hybrid search: FAISS vector + SQLite FTS5 keyword, merged via Reciprocal Rank Fusion
- Automatic session ingestion: watches session files, chunks, embeds, indexes
- REST API: 28 endpoints for search, ingest, recall, augment, reasoning, compression
- `/augment` endpoint: returns pre-formatted context ready for LLM injection
- Role filtering: only indexes user/assistant messages (tool calls, system msgs are noise)
- Content deduplication: prevents duplicate results across sessions
- OpenClaw integration: auto-recall before every LLM call, auto-capture after conversations

### What's broken or missing (the MVP gap):
- SQLite single-writer locking causes crashes under concurrent load
- No multi-tenancy (one user, one database)
- Embedding provider hardcoded to Venice AI
- cogdedup subsystem generates unbounded data (107M rows in production)
- Status endpoint reports wrong chunk counts
- No authentication on API
- No tests
- No documentation
- No packaging (pip install, Docker image)

---

## 2. ARCHITECTURE

```
┌─────────────────────────────────────────────────────┐
│                    REST API (FastAPI)                │
│                  src/c3ae/api/routes.py              │
│                                                     │
│  POST /api/v1/memory/augment    ← Main product API  │
│  POST /api/v1/memory/recall     ← Search memories   │
│  POST /api/v1/memory/ingest     ← Store text        │
│  POST /api/v1/memory/search     ← Full hybrid search│
│  GET  /api/v1/health            ← Health check      │
│  GET  /api/v1/status/full       ← System stats      │
│  ... 22 more endpoints                              │
└──────────────┬──────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────┐
│              MemorySpine (Core Engine)               │
│            src/c3ae/memory_spine/spine.py            │
│                                                     │
│  - ingest_text()      → chunk + embed + store       │
│  - ingest_session()   → parse session → chunk all   │
│  - search()           → hybrid search               │
│  - search_keyword()   → FTS5 only (fallback)        │
└──────────┬────────────┬─────────────────────────────┘
           │            │
┌──────────▼──┐  ┌──────▼─────────────────────────────┐
│ HybridSearch│  │       Storage Layer                 │
│  retrieval/ │  │                                     │
│             │  │  SQLiteStore  → chunks, FTS5, audit │
│  - RRF merge│  │  FAISSStore   → vector index        │
│  - 70/30 wt │  │  EmbeddingCache → avoid re-embeds   │
│  - k=60     │  │  Vault        → file storage        │
└─────────────┘  └─────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│           Ingestion Pipeline (Background)            │
│                                                     │
│  compress-watcher.py                                │
│    → Watches session .jsonl files                   │
│    → Compresses (100-400x ratio)                    │
│    → Parses into SessionChunks                      │
│    → Filters: only user + assistant roles           │
│    → Embeds via Venice AI                           │
│    → Stores in SQLite + FAISS                       │
│    → Runs every 5 minutes                           │
│                                                     │
│  session_parser.py                                  │
│    → Parses OpenClaw .jsonl session format           │
│    → Extracts role, content, metadata               │
│    → Handles content-block arrays (text, tool_use)  │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│           cogdedup (Compression Engine)              │
│          src/c3ae/usc_bridge/c3_cogstore.py          │
│                                                     │
│  WARNING: This subsystem is problematic.            │
│  - Builds co-occurrence matrices for deduplication  │
│  - cogdedup_cooccurrence table grew to 107M rows    │
│  - 4.5GB of derived data for 4K source chunks       │
│  - Needs to be either fixed with bounded storage    │
│    or removed entirely for MVP                      │
└─────────────────────────────────────────────────────┘
```

---

## 3. FILE-BY-FILE GUIDE

### Core (must understand):

| File | Lines | Purpose | MVP Priority |
|------|-------|---------|-------------|
| `src/c3ae/api/routes.py` | 733 | All REST endpoints. The `/augment` endpoint is the main product API. | **CRITICAL** |
| `src/c3ae/memory_spine/spine.py` | 632 | Core engine. Orchestrates ingest, embed, search. Contains `_INGEST_ROLES` filter. | **CRITICAL** |
| `src/c3ae/storage/sqlite_store.py` | 606 | SQLite storage. Chunks table, FTS5 index, embedding cache, audit log. Has `_commit()` retry for locking. | **CRITICAL** |
| `src/c3ae/retrieval/hybrid.py` | 93 | Reciprocal Rank Fusion merge of keyword + vector results. Clean, well-designed. | **CRITICAL** |
| `src/c3ae/types.py` | 111 | Pydantic data models: Chunk, SearchResult, ReasoningEntry, etc. | **CRITICAL** |
| `src/c3ae/config.py` | 77 | Configuration: Venice API, retrieval weights, API settings. | **CRITICAL** |

### Supporting (need cleanup):

| File | Lines | Purpose | MVP Priority |
|------|-------|---------|-------------|
| `src/c3ae/retrieval/keyword.py` | 30 | FTS5 keyword search wrapper. | HIGH |
| `src/c3ae/retrieval/vector.py` | 40 | FAISS vector search wrapper. | HIGH |
| `src/c3ae/storage/faiss_store.py` | 127 | FAISS index management. Supports Flat and IVF modes. | HIGH |
| `src/c3ae/embeddings/venice.py` | 69 | Venice AI embedding client. **Must be made pluggable.** | HIGH |
| `src/c3ae/embeddings/cache.py` | 47 | Caches embeddings by content hash. Prevents redundant API calls. | MEDIUM |
| `src/c3ae/ingestion/session_parser.py` | 278 | Parses .jsonl session files into SessionChunk objects. OpenClaw-specific format. | HIGH |
| `src/c3ae/governance/audit.py` | 40 | Best-effort audit logging to SQLite. | LOW |
| `src/c3ae/governance/guardian.py` | 200 | Content validation, size limits, contradiction checking. | LOW |

### Can remove for MVP:

| File | Lines | Purpose | Action |
|------|-------|---------|--------|
| `src/c3ae/usc_bridge/c3_cogstore.py` | 503 | Co-occurrence deduplication. Generates unbounded data. | **REMOVE** |
| `src/c3ae/usc_bridge/compressed_vault.py` | 248 | Compressed document storage. Not used in production. | **REMOVE** |
| `src/c3ae/cos/cos.py` | 83 | Carry-over summary generation. Not used. | **REMOVE** |
| `src/c3ae/mre/` | 404 | "Memory Reasoning Engine" — experimental agent loop. Not used. | **REMOVE** |
| `src/c3ae/rlm/reader.py` | 107 | "Reasoning Language Model" reader. Not used. | **REMOVE** |
| `src/c3ae/skill_capsules/` | 47 | Skill storage. Not used in production. | **REMOVE** |
| `src/c3ae/llm/venice_chat.py` | 116 | Chat completion client. Only used by MRE. | **REMOVE** |
| `src/c3ae/pipeline/loop.py` | 104 | Pipeline orchestration. Not used. | **REMOVE** |

### Scripts (operational, not part of library):

| File | Purpose | Action |
|------|---------|--------|
| `scripts/nova-memory-server.py` | Launches FastAPI server | Rename to `c3ae-server.py` |
| `scripts/nova-compress-watcher.py` | Background ingestion daemon | Rename to `c3ae-watcher.py` |
| `scripts/openclaw_session_hook.py` | OpenClaw integration hook | Move to examples/ |

---

## 4. DATABASE SCHEMA

### SQLite (`c3ae.db`):

```sql
-- Core chunks table (the main data)
CREATE TABLE chunks (
    id TEXT PRIMARY KEY,
    source_id TEXT,
    content TEXT NOT NULL,
    metadata TEXT,  -- JSON string
    created_at TEXT
);

-- Full-text search index (FTS5)
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    content,
    content_rowid='rowid'  -- linked to chunks table
);
-- Kept in sync via triggers on chunks table

-- Embedding cache (avoid re-embedding same content)
CREATE TABLE embedding_cache (
    content_hash TEXT PRIMARY KEY,
    embedding BLOB,  -- numpy array as bytes
    model TEXT,
    created_at TEXT
);

-- Audit log
CREATE TABLE audit_log (
    id TEXT PRIMARY KEY,
    action TEXT,
    target_type TEXT,
    target_id TEXT,
    detail TEXT,
    outcome TEXT,
    created_at TEXT
);

-- Reasoning bank
CREATE TABLE reasoning_entries (
    id TEXT PRIMARY KEY,
    title TEXT,
    content TEXT,
    tags TEXT,  -- JSON array
    evidence_ids TEXT,  -- JSON array
    status TEXT DEFAULT 'active',
    superseded_by TEXT,
    session_id TEXT,
    metadata TEXT,
    created_at TEXT
);

-- cogdedup tables (REMOVE THESE for MVP — they grow unbounded):
-- cogdedup_chunks, cogdedup_cooccurrence, cogdedup_lsh_bands, cogdedup_memory_chunks
```

### FAISS Index:

```
data/faiss/memory.index  — FAISS index file (Flat or IVF depending on size)
data/faiss/memory.idmap  — Maps FAISS integer IDs to chunk string IDs
```

- Dimensions: 1024 (BGE-M3 embeddings)
- Switches from Flat to IVF at 50,000 vectors (configurable)
- IVF uses nprobe=16 for search

---

## 5. KEY ALGORITHMS

### Hybrid Search (Reciprocal Rank Fusion)

The core differentiator. Most memory systems use ONLY vector search. C3/Ae combines both:

```python
# 1. Run keyword search (FTS5) → ranked list
# 2. Run vector search (FAISS) → ranked list
# 3. Merge using RRF:
#    score(doc) = Σ weight / (k + rank + 1)
#    where k=60 (constant), weight=0.7 for vector, 0.3 for keyword
# 4. Sort by combined score, return top_k

# This means:
# - Exact keyword matches boost results (user names, technical terms)
# - Semantic similarity catches conceptually related content
# - A result that appears in BOTH lists gets a big boost
```

**Why this matters:** If a user asks "do you remember Desmond?", pure vector search might miss it because "remember" is semantically vague. But keyword search finds "Desmond" instantly. The hybrid approach gets both.

### Role Filtering

Only `user` and `assistant` messages are indexed. In production, 80%+ of session content is noise:
- `tool_call`: JSON blobs of function calls
- `tool_result`: Raw API responses, HTML dumps, error traces
- `system`: System prompts, instructions (huge, repetitive)
- `unknown`: Metadata, formatting

Filtering these out took 497K chunks down to 4K and transformed search quality from unusable to excellent.

### Content Deduplication

Same conversations appear in multiple session files (archives, active sessions). The augment/recall endpoints use a content key (first 200 chars, lowercased) to skip near-identical results.

### Identity Search (in OpenClaw extension, not in core)

When a user sends a message, the system runs TWO parallel searches:
1. Message content → finds relevant memories
2. User identity (phone number, name) → finds "who is this person" memories

This solves the problem where "do you remember?" doesn't contain the user's name.

---

## 6. MVP TASK LIST (ordered by priority)

### Task 1: Fix SQLite Concurrency (CRITICAL)

**Problem:** SQLite is single-writer. The background watcher holds write locks during batch ingestion. When the API tries to write (auto-capture), it gets `sqlite3.OperationalError: database is locked` even with 30-second timeouts.

**Solution options (pick one):**

**Option A: WAL Mode (simplest, recommended for MVP)**
```python
# In sqlite_store.py, after connection:
self._conn.execute("PRAGMA journal_mode=WAL")
self._conn.execute("PRAGMA wal_autocheckpoint=1000")
```
WAL (Write-Ahead Logging) allows concurrent readers and one writer without blocking. This alone would fix 90% of the locking issues.

**Option B: Migrate to PostgreSQL (better for product)**
- Replace `sqlite_store.py` with a Postgres implementation
- Use connection pooling (asyncpg or psycopg3 with pool)
- Supports true concurrent writes
- Better for multi-tenancy later

**Option C: Separate read/write connections**
```python
self._read_conn = sqlite3.connect(db_path, check_same_thread=False, timeout=5)
self._write_conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30)
# Use read_conn for all SELECT queries
# Use write_conn (with retry) for INSERT/UPDATE
```

**Recommendation:** Do Option A now (one line fix), plan Option B for v2.

### Task 2: Make Embedding Provider Pluggable (HIGH)

**Problem:** `src/c3ae/embeddings/venice.py` hardcodes Venice AI's API. Users need OpenAI, Cohere, local models (sentence-transformers), etc.

**Solution:**

Create an abstract base class:
```python
# src/c3ae/embeddings/base.py
from abc import ABC, abstractmethod
import numpy as np

class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed(self, texts: list[str]) -> list[np.ndarray]:
        """Embed a batch of texts. Returns list of vectors."""
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return embedding dimensions."""
        ...
```

Implement providers:
```python
# src/c3ae/embeddings/openai.py — OpenAI text-embedding-3-small/large
# src/c3ae/embeddings/venice.py — Venice AI (existing, refactor to use base)
# src/c3ae/embeddings/local.py  — sentence-transformers (no API needed)
# src/c3ae/embeddings/ollama.py — Ollama local embeddings
```

Update config:
```python
class EmbeddingConfig(BaseModel):
    provider: str = "openai"  # "openai", "venice", "local", "ollama"
    model: str = "text-embedding-3-small"
    dimensions: int = 1536
    api_key: str = ""
    base_url: str = ""
```

**Key constraint:** When changing providers, the FAISS index must be rebuilt (different dimensions). Add a CLI command: `c3ae rebuild-index --provider openai`.

### Task 3: Remove cogdedup Subsystem (HIGH)

**Problem:** `c3ae/usc_bridge/c3_cogstore.py` builds co-occurrence matrices that grow quadratically. In production, 4K source chunks generated 107M co-occurrence rows (4.5GB). This is unsustainable.

**Solution:**
1. Delete `src/c3ae/usc_bridge/` entirely
2. Remove cogdedup table creation from `sqlite_store.py`
3. Remove cogdedup API endpoints from `routes.py` (lines referencing `/cogdedup/`)
4. Remove cogdedup imports from `spine.py`
5. The content dedup in the augment endpoint (first-200-chars key) is sufficient for MVP

### Task 4: Fix Status Endpoint (MEDIUM)

**Problem:** `GET /api/v1/status/full` reports wrong chunk count (543 vs actual 8,500+). The endpoint queries the `chunks` table but the actual data is spread across the ingestion pipeline.

**Solution:** Fix the query in `routes.py` to count correctly:
```python
# Current (broken):
chunks = spine.sqlite.count_chunks()  # May be using wrong method

# Fix: direct SQL count
cursor = spine.sqlite._conn.execute("SELECT COUNT(*) FROM chunks")
chunk_count = cursor.fetchone()[0]
```

### Task 5: Add Authentication (MEDIUM)

**Problem:** API is wide open. Anyone on localhost can read/write memories.

**Solution:** Bearer token auth (already partially configured in `config.py`):
```python
# config.py already has:
class APIConfig(BaseModel):
    bearer_token: str = Field(default_factory=lambda: os.environ.get("C3AE_API_TOKEN", ""))

# Add middleware to routes.py:
from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer(auto_error=False)

async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if not config.api.bearer_token:
        return  # No token configured = no auth required
    if not credentials or credentials.credentials != config.api.bearer_token:
        raise HTTPException(status_code=401, detail="Invalid token")
```

### Task 6: Make Session Parser Generic (MEDIUM)

**Problem:** `session_parser.py` is hardcoded for OpenClaw's `.jsonl` format. Other agent frameworks (LangChain, CrewAI, AutoGen) use different formats.

**Solution:** Create parser plugins:
```python
# src/c3ae/ingestion/parsers/base.py
class SessionParser(ABC):
    @abstractmethod
    def parse(self, file_path: Path) -> list[SessionChunk]: ...

# src/c3ae/ingestion/parsers/openclaw.py — current implementation
# src/c3ae/ingestion/parsers/langchain.py — LangChain chat history format
# src/c3ae/ingestion/parsers/jsonl.py — generic JSONL (role + content)
# src/c3ae/ingestion/parsers/plain.py — plain text files
```

Minimum for MVP: support a simple universal format:
```jsonl
{"role": "user", "content": "Hello", "timestamp": "2026-01-01T00:00:00Z"}
{"role": "assistant", "content": "Hi there!", "timestamp": "2026-01-01T00:00:01Z"}
```

### Task 7: Add Tests (MEDIUM)

**Current state:** Zero tests.

**Minimum test coverage for MVP:**

```
tests/
  test_sqlite_store.py    — CRUD operations, FTS5 search, concurrent access
  test_faiss_store.py     — Index creation, search, persistence
  test_hybrid_search.py   — RRF merge, weight tuning, edge cases
  test_spine.py           — ingest_text, ingest_session, search
  test_api.py             — All REST endpoints (use FastAPI TestClient)
  test_session_parser.py  — Parse various session formats
  test_embeddings.py      — Mock embedding provider, cache behavior
```

Use pytest + pytest-asyncio. Mock the embedding provider for fast tests.

### Task 8: Package for Distribution (MEDIUM)

**Current state:** No `pyproject.toml`, no Docker image, no pip package.

**Solution:**

```toml
# pyproject.toml
[project]
name = "c3ae"
version = "0.1.0"
description = "Long-term memory for LLM agents — hybrid vector + keyword search"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.100",
    "uvicorn>=0.20",
    "pydantic>=2.0",
    "numpy>=1.24",
    "faiss-cpu>=1.7",  # or faiss-gpu
    "httpx>=0.24",
]

[project.optional-dependencies]
openai = ["openai>=1.0"]
local = ["sentence-transformers>=2.2"]
venice = []  # uses httpx

[project.scripts]
c3ae = "c3ae.cli:main"
```

```dockerfile
# Dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install -e ".[openai]"
EXPOSE 8420
CMD ["c3ae", "serve"]
```

### Task 9: Write README (LOW)

Minimum viable README:
```markdown
# C3/Ae — Long-Term Memory for LLM Agents

## Quick Start
pip install c3ae
c3ae serve  # starts API on :8420

## Store a memory
curl -X POST localhost:8420/api/v1/memory/ingest \
  -H "Content-Type: application/json" \
  -d '{"text": "The user prefers dark mode"}'

## Search memories
curl -X POST localhost:8420/api/v1/memory/recall \
  -H "Content-Type: application/json" \
  -d '{"query": "user preferences", "top_k": 5}'

## Auto-inject into LLM context
curl -X POST localhost:8420/api/v1/memory/augment \
  -H "Content-Type: application/json" \
  -d '{"query": "what does the user like?", "top_k": 5, "format": "xml"}'
# Returns pre-formatted <relevant-memories> block ready for LLM injection
```

### Task 10: Multi-Tenancy (FUTURE — not MVP)

For v2: namespace chunks by `tenant_id`, separate FAISS indices per tenant, add API key management.

---

## 7. KNOWN BUGS

| Bug | File | Line(s) | Severity | Fix |
|-----|------|---------|----------|-----|
| `database is locked` on concurrent writes | `sqlite_store.py` | 266 | HIGH | Add WAL mode (Task 1) |
| Status endpoint reports wrong chunk count | `routes.py` | ~50-70 | MEDIUM | Fix SQL query |
| cogdedup tables grow unbounded | `c3_cogstore.py` | all | HIGH | Remove entirely (Task 3) |
| Ingest retry catches all exceptions, not just locked | `routes.py` | 330-345 | LOW | Narrow exception type |
| FTS5 triggers may not exist if DB created without them | `sqlite_store.py` | table creation | LOW | Add IF NOT EXISTS to trigger creation |
| FAISS index not rebuilt when embedding dimensions change | `faiss_store.py` | all | MEDIUM | Add dimension check on load |

---

## 8. CONFIGURATION REFERENCE

### Environment Variables:
```bash
C3AE_DATA_DIR=/path/to/data          # Where DB, FAISS, vault files live
C3AE_API_TOKEN=your-secret-token     # Bearer token for API auth
VENICE_API_KEY=your-venice-key       # Venice AI API key (for embeddings)
```

### Retrieval Tuning:
```python
RetrievalConfig(
    vector_weight=0.7,        # Weight for vector search in RRF (0.0-1.0)
    keyword_weight=0.3,       # Weight for keyword search in RRF (0.0-1.0)
    default_top_k=20,         # Default number of results
    faiss_ivf_threshold=50000,# Switch from Flat to IVF at this many vectors
    faiss_nprobe=16,          # IVF search breadth (higher = slower but more accurate)
)
```

### Augment Endpoint Parameters:
```json
{
    "query": "search text",
    "top_k": 5,
    "min_score": 0.005,
    "format": "xml",
    "roles": ["user", "assistant"]
}
```
- `format`: `"xml"` wraps results in `<relevant-memories>` tags; `"plain"` returns raw text
- `roles`: filter results by message role (default: user + assistant only)
- `min_score`: minimum RRF score threshold to include a result

---

## 9. DEPENDENCIES

### Current (Python 3.12+):
```
fastapi==0.115.*
uvicorn==0.34.*
pydantic==2.10.*
numpy==1.26.*    # or 2.x with compat fixes
faiss-cpu==1.9.*
httpx==0.28.*
```

### For MVP (add):
```
pytest==8.*
pytest-asyncio==0.24.*
httpx[test]  # for FastAPI TestClient
```

### Optional (by embedding provider):
```
openai>=1.0       # OpenAI embeddings
sentence-transformers>=2.2  # Local embeddings
```

---

## 10. QUICK START FOR DEVELOPMENT

```bash
# Clone
git clone https://github.com/maddwiz/Nova-v1.git
cd Nova-v1

# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Set embedding provider API key
export VENICE_API_KEY=your-key  # or OPENAI_API_KEY once provider is pluggable

# Run server
python scripts/nova-memory-server.py
# → API available at http://localhost:8420

# Test
curl localhost:8420/api/v1/health
# → {"status": "ok", "service": "c3ae"}

# Ingest some text
curl -X POST localhost:8420/api/v1/memory/ingest \
  -H "Content-Type: application/json" \
  -d '{"text": "My name is Desmond and I like dark mode"}'

# Search for it
curl -X POST localhost:8420/api/v1/memory/recall \
  -H "Content-Type: application/json" \
  -d '{"query": "what is the user name", "top_k": 3}'
```

---

## 11. SUCCESS CRITERIA FOR MVP

The MVP is shippable when:

1. [ ] `pip install c3ae && c3ae serve` works out of the box
2. [ ] Supports at least 2 embedding providers (OpenAI + one local option)
3. [ ] No SQLite locking errors under concurrent load
4. [ ] cogdedup removed, database stays bounded
5. [ ] Status endpoint reports correct numbers
6. [ ] Bearer token auth works
7. [ ] Session parser handles generic JSONL format
8. [ ] Core test suite passes (>80% coverage on core modules)
9. [ ] README with quick start, API docs, configuration guide
10. [ ] Docker image builds and runs
11. [ ] Can ingest 100K chunks without performance degradation
12. [ ] Search latency < 100ms for 100K chunks (hybrid mode)

---

## 12. COMPETITIVE LANDSCAPE

| Product | Search Type | Hosting | Weakness |
|---------|-------------|---------|----------|
| **Mem0** | Vector only | Cloud/self-hosted | No keyword search, misses exact matches |
| **Zep** | Vector + metadata | Self-hosted | Complex setup, requires Postgres + Redis |
| **LangChain Memory** | Various | In-process | Not persistent, no server mode |
| **Motorhead** | Vector | Self-hosted | Abandoned/minimal maintenance |
| **C3/Ae (this)** | Hybrid (vector + keyword RRF) | Self-hosted | Needs the fixes in this doc |

**C3/Ae's edge:** Hybrid search with RRF fusion. Most competitors only do vector search. C3/Ae catches both semantic similarity AND exact keyword matches. The `/augment` endpoint that returns pre-formatted LLM context is also unique — most competitors return raw results and leave formatting to the user.

---

## 13. ESTIMATED EFFORT

| Task | Effort | Dependency |
|------|--------|-----------|
| 1. Fix SQLite (WAL mode) | 1 hour | None |
| 2. Pluggable embeddings | 4-6 hours | None |
| 3. Remove cogdedup | 2 hours | None |
| 4. Fix status endpoint | 30 min | None |
| 5. Add auth | 2 hours | None |
| 6. Generic session parser | 4 hours | None |
| 7. Add tests | 8-12 hours | Tasks 1-4 |
| 8. Package (pyproject + Docker) | 3 hours | Task 2 |
| 9. Write README | 2 hours | Tasks 1-8 |
| **Total** | **~27-32 hours** | |

All tasks are independent except tests (which should run after fixes). Tasks 1-4 can be done in parallel.
