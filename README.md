# NovaSpine — Durable Memory for AI Agents

NovaSpine gives an AI agent memory that survives across sessions, comes back in useful form, and can grow beyond raw chat history into facts, summaries, and working context.

You can use it as a standalone memory API for any agent stack, or install it as a turnkey memory/context/consciousness layer for OpenClaw.

## 30-Second Quickstart

If you just want to try NovaSpine from a terminal, start here:

```bash
pip install novaspine
printf '%s\n' 'Retrieval notes: keep keyword fallback for exact facts.' > notes.txt
novaspine ingest ./notes.txt
novaspine recall "what did I save about retrieval?"
novaspine status
novaspine doctor
```

That is the shortest path to:

- put memory in with `novaspine ingest`
- get useful memory back with `novaspine recall`
- inspect local state with `novaspine status`
- verify the install with `novaspine doctor`

If you want HTTP or OpenClaw integration after that, keep reading.

## Using NovaSpine from the Terminal

NovaSpine is usable directly from the terminal. The main commands are:

- `novaspine ingest`: store memory from a file
- `novaspine recall`: preferred high-level memory retrieval command
- `novaspine search`: lower-level/manual search for debugging and inspection
- `novaspine status`: inspect local memory state
- `novaspine doctor`: diagnose installation and OpenClaw wiring
- `novaspine serve`: run the local HTTP API for apps and agents

## Why People Use It

- **Remember across sessions** instead of starting cold every time
- **Recall the right context fast** with hybrid semantic + keyword retrieval
- **Inject memory into prompts cleanly** with `/augment`
- **Promote raw history into better memory** with facts, consolidation, and dream passes
- **Install quickly for OpenClaw** without building your own glue first

## What You Get

- **Hybrid retrieval**: FAISS vector search + SQLite FTS5 keyword search + RRF fusion
- **Prompt-ready memory**: `/augment` returns memory blocks formatted for LLM injection
- **Structured memory**: entities, facts, graph edges, reasoning entries, and skill capsules
- **Consolidation + dreams**: episodic memories can be compressed into more durable semantic memory
- **OpenClaw integration**: memory plugin, context engine, consciousness plugin, maintenance scripts, and config patcher
- **Benchmark tooling**: official-source prep, retrieval benchmarks, and end-to-end QA harnesses

## Choose Your Setup

### Option 1: any agent framework

Use NovaSpine as a Python package or local HTTP API.

1. Install:

```bash
pip install novaspine
```

2. Start the server:

```bash
novaspine serve
```

If you only want the CLI, you can skip `novaspine serve` until you need HTTP access.

3. Point your app or agent at:

- default local API: `http://127.0.0.1:8420`
- health check: `GET /api/v1/health`
- prompt-ready recall: `POST /api/v1/memory/augment`
- choose an auth mode before calling protected routes

### Option 2: OpenClaw

Use the bundled OpenClaw install kit for memory, context, and consciousness.

1. Clone the repo:

```bash
git clone https://github.com/maddwiz/NovaSpine.git
cd NovaSpine
```

2. Run the OpenClaw installer:

```bash
./scripts/install-openclaw.sh
```

3. Verify the install:

```bash
novaspine doctor
openclaw config validate
```

4. Start or restart OpenClaw normally.

The installer:

- copies the reusable integration layer
- installs the NovaSpine memory, context, and consciousness plugins when `openclaw` is available
- patches `openclaw.json`
- leaves you with `novaspine doctor` as the repair/verification check

If you update OpenClaw later, the supported repair flow is:

```bash
./scripts/install-openclaw.sh
novaspine doctor
openclaw config validate
```

### OpenClaw compatibility

NovaSpine's standalone Python/API layer is relatively stable. The OpenClaw integration layer depends on OpenClaw's plugin and config surfaces, so major or fast-moving OpenClaw updates can require a repair pass.

A compatibility matrix helps answer three user-facing questions:

- which OpenClaw versions are actually tested
- what still benefits from upstream OpenClaw updates
- which new upstream memory features need a NovaSpine port or adapter first

Current tested OpenClaw versions:

- `2026.4.5`
- `2026.4.7`
- `2026.4.9`
- `2026.4.10`
- `2026.4.11`
- `2026.4.12`

If an OpenClaw update changes plugin wiring, config shape, or slot bindings, re-running the installer is the supported repair path.

Quick rule:

- OpenClaw core/runtime improvements still matter
- slot-aware memory improvements can still benefit NovaSpine
- OpenClaw `2026.4.10` Active Memory can be carried through NovaSpine's memory plugin without enabling stock `active-memory`
- OpenClaw `2026.4.11` Dreaming import/UI additions remain stock `memory-core` functionality and are not auto-ported just because NovaSpine owns the memory slot
- OpenClaw `2026.4.12` memory-side fixes that map directly to NovaSpine are carried locally: unicode-safe wiki slugs and nested daily-note pickup for workspace memory recall
- NovaSpine-native dream diary/status and wiki surfaces can be carried forward in the integration layer
- `memory-core`-specific memory features are not guaranteed automatically when NovaSpine is the active memory slot

For the fuller upgrade/feature matrix, see [OPENCLAW_COMPATIBILITY.md](/home/nova/NovaSpine/OPENCLAW_COMPATIBILITY.md).

## Auth Modes

NovaSpine has two explicit API modes plus one protective default.

### Mode A: token-protected API

Set `C3AE_API_TOKEN` for secured local or remote access. All routes except the exempt health/docs routes require Bearer auth.

Exempt routes:

- `/api/v1/health`
- `/docs`
- `/redoc`
- `/openapi.json`

```bash
export C3AE_API_TOKEN=your-secret-token
novaspine serve

curl -X POST http://127.0.0.1:8420/api/v1/memory/recall \
  -H "Authorization: Bearer $C3AE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query":"dark mode","top_k":5}'
```

### Mode B: explicit local unauthenticated mode

Set `C3AE_AUTH_DISABLED=1` only when you intentionally want local unauthenticated access.

```bash
export C3AE_AUTH_DISABLED=1
novaspine serve

curl -X POST http://127.0.0.1:8420/api/v1/memory/augment \
  -H "Content-Type: application/json" \
  -d '{"query":"what does the user like?","top_k":3,"format":"xml"}'
```

### Default if neither is set

If neither `C3AE_API_TOKEN` nor `C3AE_AUTH_DISABLED=1` is set:

- `/api/v1/health`, `/docs`, `/redoc`, and `/openapi.json` still work
- all other routes return `503` until you explicitly choose one of the two modes above

## Benchmark Highlights

These are the strongest benchmark results already checked into `bench/results/` and summarized from official-source runs.

### Retrieval

| Benchmark | Best verified profile | Top-line result |
|---|---|---:|
| LongMemEval | `official_longmemeval_retrieval_hash_keywordplus_20260302_r29` | `doc_hit 1.000` |
| LoCoMo-MC10 | `official_locomo_retrieval_hash_keywordplus_k15_20260302_r29` | `doc_hit 0.937` |
| DMR-500 | `official_dmr_retrieval_sbert_variants_20260302_r29` | `doc_hit 0.848` |

### End-to-End QA

| Benchmark | Best verified profile | doc_hit | EM | F1 |
|---|---|---:|---:|---:|
| LongMemEval | `official_longmemeval_qa_openai_20260228_r9` | `1.000` | `0.304` | `0.367` |
| LoCoMo-MC10 quality | `official_locomo_qa_openai_20260228_r6_mini` | `0.860` | `0.455` | `0.484` |
| LoCoMo-MC10 high recall | `official_locomo_qa_openai_20260228_r9_k15_lexctx` | `0.944` | `0.448` | `0.458` |
| DMR-500 | `official_dmr_qa_openai_20260301_r16_large_legacy` | `0.950` | `0.628` | `0.632` |

Artifact summaries:

- `bench/results/official_benchmark_summary.md`
- `bench/results/official_qa_summary_20260228_openai_tuned.md`

## NovaSpine Stack

NovaSpine now ships as one reusable stack instead of “just the old memory engine”:

- `src/`: core memory engine, retrieval, graph, facts, consolidation, and dream paths
- `packages/openclaw-memory-plugin/`: OpenClaw memory capture and recall
- `packages/openclaw-context-engine/`: OpenClaw prompt-context injection
- `packages/openclaw-consciousness/`: continuity, low-noise cognition, goals, decisions, and thread resume hooks
- `integrations/openclaw/scripts/`: maintenance, sync, pruning, writeback, and dream runners

## Why NovaSpine?

Most memory layers stop at “vector DB + semantic search.” NovaSpine is designed as a fuller cognitive stack:

| Capability | NovaSpine | Typical vector-only memory |
|---|---|---|
| Exact keyword + semantic recall | Yes | Usually semantic only |
| Structured facts + graph | Yes | Often absent |
| Consolidation + dream passes | Yes | Rare |
| Prompt-ready augmentation | Built in | Usually manual |
| OpenClaw turnkey integration | Built in | Usually custom glue |
| Consciousness/continuity layer | Available | Rare |

## API Quick Start

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

## Search vs Recall

NovaSpine intentionally exposes both a lower-level search route and the agent-facing memory contract.

- `novaspine recall`: start here in the CLI for useful memory retrieval
- `novaspine search`: use this when you want lower-level/manual search inspection
- `/api/v1/memory/recall`: use this first for agent memory retrieval across sessions and memories
- `/api/v1/memory/augment`: use this first when you want prompt-ready context injection
- `/api/v1/memory/search`: lower-level search endpoint for tools, debugging, and raw hybrid-search access

If you are using the CLI, start with `recall`, not `search`. If you are building an agent integration, start with `recall` or `augment`, not `search`.

## First Run Sanity Check

### Standalone NovaSpine

```bash
export C3AE_AUTH_DISABLED=1
novaspine serve
```

In another shell:

```bash
curl http://127.0.0.1:8420/api/v1/health

curl -X POST http://127.0.0.1:8420/api/v1/memory/augment \
  -H "Content-Type: application/json" \
  -d '{"query":"sanity check","top_k":3,"format":"xml"}'
```

### OpenClaw install

```bash
./scripts/install-openclaw.sh
novaspine doctor
openclaw config validate
```

### Local plugin type checks

```bash
cd packages/openclaw-memory-plugin && npm install && npm run typecheck
cd ../openclaw-context-engine && npm install && npm run typecheck
```

### Query the memory graph
```bash
curl -X POST localhost:8420/api/v2/graph/query \
  -H "Content-Type: application/json" \
  -d '{"entity":"Desmond","depth":2}'
```

### Run memory consolidation
```bash
curl -X POST localhost:8420/api/v1/memory/consolidate \
  -H "Content-Type: application/json" \
  -d '{"max_chunks":1000}'
```

## Versioned Integration Contract (No Per-Project Rewrite)

Use the protocol client as the stable interface and keep `MemorySpine` internals free to evolve.

```python
from c3ae.config import Config
from c3ae.memory_spine import MemorySpine

spine = MemorySpine(Config())
client = spine.protocol_client("v1")

await client.ingest("User prefers short status updates", source_id="session:1")
rows = await client.recall("user preference", top_k=5)
context = await client.augment("user preference", top_k=3, format="xml")
status = client.status()
```

`/api/v1/*` remains the stable remote contract. Future additive methods are exposed under protocol `v2` without breaking `v1`.
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
3. Merge via **adaptive Reciprocal Rank Fusion** (query-intent weighted):
   - `score(doc) = Σ weight / (k + rank + 1)`
   - Base default: Vector 0.7, Keyword 0.3
4. Apply time/access/importance scoring:
   - Older memories decay (configurable half-life)
   - Frequently retrieved memories get reinforcement boost
   - Reasoning/evidence-linked entries get importance boost
5. Return top-k by final score

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
C3AE_AUTH_DISABLED=0                 # set to 1 only for explicit local unauthenticated use
C3AE_EMBEDDING_PROVIDER=venice       # preferred name; legacy: C3AE_EMBED_PROVIDER
C3AE_EMBEDDING_MODEL=text-embedding-bge-m3      # legacy: C3AE_EMBED_MODEL
C3AE_EMBEDDING_DIMENSIONS=1024                 # legacy: C3AE_EMBED_DIMS
C3AE_EMBEDDING_API_KEY=...           # shared embedding API key alias for venice/openai
VENICE_API_KEY=your-key              # legacy/provider-specific Venice key

# Memory write manager (runtime-tunable)
C3AE_MEMORY_MANAGER_ENABLED=1
C3AE_MEMORY_MANAGER_USE_LLM_POLICY=1
C3AE_MEMORY_MANAGER_UPDATE_THRESHOLD=0.84
C3AE_MEMORY_MANAGER_NOOP_THRESHOLD=0.945
C3AE_MEMORY_MANAGER_PROVIDER=openai   # venice|openai|anthropic|ollama
C3AE_MEMORY_MANAGER_MODEL=gpt-4.1-mini
C3AE_MEMORY_MANAGER_TEMPERATURE=0
C3AE_MEMORY_MANAGER_MAX_TOKENS=256
```

See [`examples/memory_manager.env.example`](examples/memory_manager.env.example) for a ready-to-copy profile.

If you switch provider/model dimensions, rebuild vectors:

```bash
novaspine rebuild-index --provider openai
```

See [`examples/memory_manager.env.example`](examples/memory_manager.env.example) for a ready-to-copy profile.

## Also Includes: USC Compression Engine

## Docker

```bash
docker build -t novaspine:latest .
docker run --rm -p 8420:8420 \
  -e C3AE_AUTH_DISABLED=1 \
  -e C3AE_EMBEDDING_PROVIDER=venice \
  -e VENICE_API_KEY=your-key \
  novaspine:latest
```

For a first local Docker run, `C3AE_AUTH_DISABLED=1` is the simplest path. For any shared or remote deployment, replace that with `C3AE_API_TOKEN=...` and use Bearer auth.

## Benchmarks

Use the benchmark harness to evaluate recall@k and MRR on LoCoMo/LongMemEval/DMR-style JSONL datasets.
You can ingest a benchmark corpus first and run fully offline (`--ingest-sync`):

```bash
python scripts/run_memory_benchmarks.py \
  --name locomo \
  --corpus ./bench/fixtures/locomo_corpus.jsonl \
  --dataset ./bench/fixtures/locomo_eval.jsonl \
  --top-k 10 \
  --ingest-sync \
  --out ./bench/results/locomo.json

python scripts/run_memory_benchmarks.py \
  --name longmemeval \
  --corpus ./bench/fixtures/longmemeval_corpus.jsonl \
  --dataset ./bench/fixtures/longmemeval_eval.jsonl \
  --top-k 10 \
  --ingest-sync \
  --out ./bench/results/longmemeval.json

python scripts/run_memory_benchmarks.py \
  --name dmr \
  --corpus ./bench/fixtures/dmr_corpus.jsonl \
  --dataset ./bench/fixtures/dmr_eval.jsonl \
  --top-k 10 \
  --ingest-sync \
  --out ./bench/results/dmr.json
```

Tune heuristic policy thresholds from labeled similarity decisions:

```bash
python scripts/train_memory_policy.py \
  --data ./bench/policy/memory_policy_train.jsonl \
  --out ./bench/policy/memory_policy_tuned.json
```

### Official-Source Benchmark Prep

Prepare official-source benchmark conversions (LongMemEval cleaned oracle, LoCoMo-MC10, and MemGPT QA DMR-style):

```bash
python scripts/prepare_official_benchmarks.py
```

Run converted official benchmarks:

```bash
python scripts/run_memory_benchmarks.py \
  --name longmemeval_official \
  --corpus ./bench/official/converted/longmemeval_oracle_corpus.jsonl \
  --dataset ./bench/official/converted/longmemeval_oracle_eval.jsonl \
  --top-k 10 \
  --ingest-sync \
  --out ./bench/results/official_longmemeval.json

python scripts/run_memory_benchmarks.py \
  --name locomo_mc10_official_source \
  --corpus ./bench/official/converted/locomo_mc10_corpus.jsonl \
  --dataset ./bench/official/converted/locomo_mc10_eval.jsonl \
  --top-k 10 \
  --ingest-sync \
  --out ./bench/results/official_locomo_mc10.json

python scripts/run_memory_benchmarks.py \
  --name dmr_memgpt_official_source_sample500 \
  --corpus ./bench/official/converted/dmr_memgpt_corpus.jsonl \
  --dataset ./bench/official/converted/dmr_memgpt_eval.jsonl \
  --top-k 10 \
  --ingest-sync \
  --out ./bench/results/official_dmr_memgpt_sample500.json
```

Run full QA scoring (doc hit + EM + token-F1) on converted official datasets:

```bash
python scripts/run_memory_qa.py \
  --name longmemeval_official_qa \
  --dataset ./bench/official/converted/longmemeval_oracle_qa_eval.jsonl \
  --corpus ./bench/official/converted/longmemeval_oracle_corpus.jsonl \
  --top-k 10 \
  --ingest-sync \
  --skip-chunking \
  --answer-mode extractive \
  --out ./bench/results/official_longmemeval_qa.json
```

Use an LLM answerer (competitor-style end-to-end QA):

```bash
export OPENAI_API_KEY=...

python scripts/run_memory_qa.py \
  --name longmemeval_official_qa_openai \
  --dataset ./bench/official/converted/longmemeval_oracle_qa_eval.jsonl \
  --corpus ./bench/official/converted/longmemeval_oracle_corpus.jsonl \
  --top-k 10 \
  --ingest-sync \
  --skip-chunking \
  --answer-mode llm \
  --answer-provider openai \
  --answer-model gpt-4.1-mini \
  --answer-context-k 8 \
  --answer-max-context-chars 12000 \
  --out ./bench/results/official_longmemeval_qa_openai.json
```

Optional retrieval tuning presets:

```bash
# Strong keyword profile (best for case-token benchmark runs)
--tune-preset keyword_plus

# Hybrid QA profile (semantic-heavy; use with async ingest + embeddings)
--tune-preset hybrid_qa --embed-provider openai
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
