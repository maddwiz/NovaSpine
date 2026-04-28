# NovaSpine Memory Roadmap, 2026

This note captures the memory architecture direction after the typed reader,
diagnostics, and first "god-tier memory" hardening pass. It is deliberately
implementation-oriented: each idea maps to repo work that improves answer
quality, memory consistency, latency, or multi-agent safety.

## Current Signal

The active LongMemEval-M run shows the main gap is no longer just retrieval. The
hard failures cluster around end-to-end answer construction: row routing,
current-vs-stale truth, multi-session reconstruction, exact span selection, and
typed normalization. The current branch adds deterministic infrastructure around
those failure points so future benchmark runs can say whether a miss came from
retrieval, rerank, reading, verification, or memory drift.

## Research-Aligned Upgrades

1. Retrieval agent / query planner

   Recent systems such as MemMachine route queries among direct retrieval,
   parallel decomposition, and iterative chain-of-query retrieval instead of
   sending every question through one vector/keyword path. NovaSpine now uses
   `QueryPlan` in retrieval traces, benchmark rows, keyword-variant execution,
   multi-session benchmark recall, and typed reader metadata. Next: add a true
   multi-tool route executor for table lookup, temporal math, graph traversal,
   and episodic expansion.

2. Bitemporal property graph

   APEX-MEM and MAGMA-style systems separate semantic, temporal, causal, and
   entity structure. NovaSpine now stores `valid_from`, `valid_to`,
   `observed_at`, transaction fields, provenance, and supersession metadata on
   structured facts and graph edges. Next: add first-class `contradicts`,
   `same_as`, and causal edge families plus query-time temporal filters.

3. Ground-truth-preserving episodic memory

   MemMachine's useful idea is not compressing everything into facts; it keeps
   full episodes available and expands nucleus matches with neighboring turns.
   NovaSpine keeps compact facts for fast lookup and now adds conservative
   same-session neighbor-turn expansion for table, list, temporal, and
   multi-session routes. Next: make the typed reader explicitly reason over the
   expanded episode block and record whether the answer came from the nucleus
   chunk or neighboring context.

4. Self-repair loop

   MemMA-style systems synthesize probe QA pairs during memory construction,
   test whether the memory can answer them, and repair failed memory before
   finalizing. NovaSpine now has a dry-run self-repair probe report in dream
   consolidation, with repair writes disabled by default. Next: generate richer
   probes from benchmark failure taxonomy and only write repairs when verifier
   support is strong.

5. Memory manager policy

   MemFactory/Memory-R1-style work points toward learned or policy-driven memory
   actions. NovaSpine now has a deterministic write-admission manager for
   facts/edges with `ALLOW`, `DENY`, `NOOP`, and `SUPERSEDE`, plus an offline
   reranker trainer that consumes benchmark `candidate_features` rows. Next:
   collect enough verified rows to train/evaluate learned reranking before any
   runtime config is allowed to load a learned artifact.

6. Hierarchical shared memory

   MemoryOS/Memori-style tiers and multi-agent memory work both argue against a
   single flat store. NovaSpine should maintain hot session memory, warm episode
   pages, durable user/project facts, and compressed archives, with ACLs and
   consistency rules before it becomes shared Claw3D agent memory.

7. Read-time verifier and citation contract

   LongMemEval frames long-term memory as indexing, retrieval, and reading. The
   live results show NovaSpine needs to win the reading stage, not just recall.
   The typed answer object, verifier status, citations, and normalized answer
   fields are now in place. The reader now handles table-cell extraction,
   cross-row list aggregation, reliable relative-day dates, and simple day-delta
   answers. Next: add multi-hop synthesis that can compose several cited spans
   into one concise answer without relying on unsupported free text.

8. Benchmark-honest optimization

   Recent benchmark commentary shows some memory results are inflated by short
   contexts, answer leakage, or judge quirks. NovaSpine should keep official
   rows machine-readable, preserve evidence IDs, log timeouts separately from
   wrong answers, and avoid benchmark-specific hidden shortcuts.

## Implementation Order

1. Reader v2 and embedding no-drop fallback: done in this branch.
2. QueryPlan row output and planner tests: done in this branch.
3. Conservative QueryPlan route execution for keyword variants and multi-session benchmark recall: done in this branch.
4. Bitemporal fields for facts/graph edges and current-vs-historical tests: done in this branch.
5. Dream/self-repair probe report: done in this branch.
6. Deterministic write admission policy for facts/edges: done in this branch.
7. Neighbor-turn episodic expansion for answer context: done in this branch.
8. Offline learned reranker training from benchmark candidate rows: done in this branch.
9. Next: richer contradiction/same-as graph semantics.
10. Next: benchmark report grouping by failure kind, route, latency stage, and self-repair outcome.

## Sources Reviewed

- MemMachine: https://arxiv.org/abs/2604.04853
- APEX-MEM: https://arxiv.org/abs/2604.14362
- MemFactory: https://arxiv.org/abs/2603.29493
- Memory-R1: https://arxiv.org/abs/2508.19828
- MAGMA: https://arxiv.org/abs/2601.03236
- MemMA: https://arxiv.org/abs/2603.18718
- A-MEM: https://arxiv.org/abs/2502.12110
- MemoryOS: https://arxiv.org/abs/2506.06326
- LongMemEval: https://proceedings.iclr.cc/paper_files/paper/2025/file/d813d324dbf0598bbdc9c8e79740ed01-Paper-Conference.pdf
- LLM Agent Memory Survey: https://openreview.net/forum?id=KPs1EgGKcT
