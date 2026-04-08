# NovaSpine — Long-Term Memory and Cognition for LLM Agents

Give any AI agent persistent memory with hybrid search (vector + keyword fusion). NovaSpine automatically stores, indexes, and retrieves relevant context from past conversations.

NovaSpine now also ships the reusable OpenClaw-facing integration layer that we
have been running live:

- the core memory engine in `src/`
- the OpenClaw memory and context plugins in `packages/openclaw-memory-plugin/` and `packages/openclaw-context-engine/`
- the OpenClaw consciousness plugin in `packages/openclaw-consciousness/`
- generic OpenClaw maintenance scripts in `integrations/openclaw/scripts/`

Intentionally not bundled as first-class repo content:

- machine-specific systemd/launchd units
- personal Discord and The Lab routing glue
- profile-local runtime state, secrets, or sidecar copies
- the old Nemo-specific sidecar wrapper layer

## Why NovaSpine?

Most LLM memory solutions use only vector search. NovaSpine combines **FAISS vector search** with **SQLite FTS5 keyword search** via **Reciprocal Rank Fusion** — catching both semantic similarity AND exact keyword matches.

| Feature | NovaSpine | Vector-only (Mem0, etc.) |
|---------|-----------|--------------------------|
| "Find messages about Desmond" | Keyword match + semantic | May miss if embedding is weak |
| "What did we discuss about auth?" | Semantic match + keyword boost | Works, but no keyword boost |
| Pre-formatted LLM injection | `/augment` endpoint | DIY formatting |
| Role filtering | Built-in (user/assistant only) | Manual |
| Content deduplication | Built-in | Manual |

## Quick Start

```bash
pip install novaspine
novaspine serve  # starts API on :8420
```

### Store a memory
```bash
curl -X POST localhost:8420/api/v1/memory/ingest \
  -H "Content-Type: application/json" \
  -d '{"text": "The user prefers dark mode and uses vim keybindings"}'
```

### Search memories
```bash
curl -X POST localhost:8420/api/v1/memory/recall \
  -H "Content-Type: application/json" \
  -d '{"query": "user preferences", "top_k": 5}'
```

### Auto-inject into LLM context
```bash
curl -X POST localhost:8420/api/v1/memory/augment \
  -H "Content-Type: application/json" \
  -d '{"query": "what does the user like?", "top_k": 5, "format": "xml"}'
```

Returns pre-formatted `<relevant-memories>` block ready for LLM context injection.

## Authentication

If `C3AE_API_TOKEN` is set, all endpoints except `/api/v1/health`, `/docs`, `/redoc`, and `/openapi.json` require Bearer auth:

```bash
export C3AE_API_TOKEN=your-secret-token
curl -X POST localhost:8420/api/v1/memory/recall \
  -H "Authorization: Bearer your-secret-token" \
  -H "Content-Type: application/json" \
  -d '{"query":"dark mode","top_k":5}'
```

## Architecture

```
┌─────────────────────────────────────────────┐
│            REST API (FastAPI)                │
│                                             │
│  /api/v1/memory/augment  ← Main product API │
│  /api/v1/memory/recall   ← Search memories  │
│  /api/v1/memory/ingest   ← Store text       │
│  /api/v1/health          ← Health check     │
└──────────────┬──────────────────────────────┘
               │
┌──────────────▼──────────────────────────────┐
│           MemorySpine (Core Engine)          │
│                                             │
│  ingest_text()   → chunk + embed + store    │
│  search()        → hybrid RRF search        │
│  search_keyword()→ FTS5 fallback            │
└──────────┬──────────┬───────────────────────┘
           │          │
    ┌──────▼───┐  ┌───▼──────────────────────┐
    │ Hybrid   │  │ Storage                  │
    │ Search   │  │                          │
    │          │  │ SQLiteStore → chunks,FTS5 │
    │ RRF merge│  │ FAISSStore → vectors     │
    │ 70/30    │  │ EmbeddingCache           │
    └──────────┘  └──────────────────────────┘
```

## How Hybrid Search Works

1. Run **keyword search** (FTS5) → ranked list
2. Run **vector search** (FAISS) → ranked list
3. Merge via **Reciprocal Rank Fusion**:
   - `score(doc) = Σ weight / (k + rank + 1)`
   - Vector weight: 0.7, Keyword weight: 0.3
4. Return top-k by combined score

Results that appear in **both** lists get a significant boost.

## Python API

```python
from c3ae.memory_spine.spine import MemorySpine
from c3ae.config import Config

spine = MemorySpine(Config())

# Store
await spine.ingest_text("User prefers dark mode")

# Search (hybrid)
results = await spine.search("user preferences", top_k=5)
for r in results:
    print(f"[{r.score:.2f}] {r.content}")

# Keyword-only fallback (no embedding API needed)
results = spine.search_keyword("dark mode", top_k=5)
```

## Configuration

```bash
# Environment variables
C3AE_DATA_DIR=/path/to/data          # Database + index location
C3AE_API_TOKEN=your-secret           # Bearer token for API auth
C3AE_EMBEDDING_PROVIDER=venice       # venice|openai|ollama|local
C3AE_EMBEDDING_MODEL=text-embedding-bge-m3
C3AE_EMBEDDING_DIMENSIONS=1024
C3AE_EMBEDDING_API_KEY=...           # for openai/venice (or use VENICE_API_KEY)
VENICE_API_KEY=your-key              # Venice fallback credential
```

If you switch provider/model dimensions, rebuild vectors:

```bash
novaspine rebuild-index --provider openai
```

## Docker

```bash
docker build -t novaspine:latest .
docker run --rm -p 8420:8420 \
  -e C3AE_EMBEDDING_PROVIDER=venice \
  -e VENICE_API_KEY=your-key \
  novaspine:latest
```

## Development

See [HANDOFF-MVP.md](HANDOFF-MVP.md) for detailed technical documentation, architecture guide, and MVP roadmap.

```bash
git clone https://github.com/maddwiz/NovaSpine.git
cd NovaSpine
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v
```

## Repo Layout

```text
src/                            Core NovaSpine engine
packages/openclaw-memory-plugin OpenClaw memory slot plugin
packages/openclaw-context-engine OpenClaw contextEngine plugin
packages/openclaw-consciousness OpenClaw plugin for continuity/cognition
integrations/openclaw/scripts/  Generic maintenance and ingestion scripts
scripts/                        Core CLI/server helpers
```

## OpenClaw Install

For a turnkey OpenClaw install from the repo:

```bash
./integrations/openclaw/install.sh
```

That copies the NovaSpine OpenClaw assets into a stable local install root,
installs the three plugins, and patches `~/.openclaw/openclaw.json` safely when
it exists.

For repo boundaries and what is intentionally excluded, see
`integrations/openclaw/README.md`.

## License

MIT
