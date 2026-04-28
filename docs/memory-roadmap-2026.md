# NovaSpine Memory Roadmap, 2026

This note captures the next architecture moves after the typed reader and
diagnostics pass. It is deliberately implementation-oriented: each idea maps to
repo work that improves either answer quality, memory consistency, latency, or
multi-agent safety.

## Current Signal

The active LongMemEval-M run shows NovaSpine already has stronger evidence
retrieval than stock OpenClaw, but answer F1 lags because the system often
retrieves the right session and then returns the wrong span. The largest
remaining buckets are evidence-present answer selection, multi-session
reasoning, location/person/count extraction, and embedding-provider 400s that
leave chunks without vectors.

## Research-Aligned Upgrades

1. Retrieval agent / query planner

   Recent systems such as MemMachine route queries among direct retrieval,
   parallel decomposition, and iterative chain-of-query retrieval instead of
   sending every question through one vector/keyword path. NovaSpine should
   promote the new deterministic `QueryPlan` into a route executor that chooses
   table lookup, list lookup, current-state, historical, temporal math, or
   multi-session reconstruction.

2. Bitemporal property graph

   APEX-MEM and MAGMA-style systems separate semantic, temporal, causal, and
   entity structure. NovaSpine already has graph and fact extraction; the next
   step is adding `valid_from`, `valid_to`, `observed_at`, `supersedes`,
   `contradicts`, and `same_as` semantics to facts and edges so current-state
   questions stop competing with stale facts.

3. Ground-truth-preserving episodic memory

   MemMachine's useful idea is not compressing everything into facts; it keeps
   full episodes available and expands nucleus matches with neighboring turns.
   NovaSpine should keep compact facts for fast lookup, but answer readers need
   an episodic expansion path when a question depends on adjacent turns.

4. Self-repair loop

   MemMA-style systems synthesize probe QA pairs during memory construction,
   test whether the memory can answer them, and repair failed memory before
   finalizing. NovaSpine can implement this locally: dream/consolidation emits
   probes, runs the typed reader against fresh memory, and creates missing
   links/facts when verification fails.

5. Memory manager policy

   MemFactory/Memory-R1-style work points toward learned or policy-driven memory
   actions. Start deterministic: every proposed fact gets provenance,
   confidence, timestamp, utility, scope, and action (`ADD`, `UPDATE`, `NOOP`,
   `DELETE`). Later, train the manager from NovaSpine's failure taxonomy and
   traces.

6. Hierarchical shared memory

   MemoryOS/Memori-style tiers and multi-agent memory work both argue against a
   single flat store. NovaSpine should maintain hot session memory, warm episode
   pages, durable user/project facts, and compressed archives, with ACLs and
   consistency rules before it becomes shared Claw3D agent memory.

## Implementation Order

1. Reader v2 and embedding no-drop fallback: done in this branch.
2. QueryPlan row output and planner tests: done in this branch.
3. Promote QueryPlan into route execution for benchmark mode.
4. Add bitemporal fields to facts/graph edges and current-vs-historical tests.
5. Add dream/self-repair probes backed by the failure taxonomy.
6. Add shared-memory scopes and write admission policy.

## Sources Reviewed

- MemMachine: https://arxiv.org/abs/2604.04853
- APEX-MEM: https://arxiv.org/abs/2604.14362
- MemFactory: https://arxiv.org/abs/2603.29493
- Memori: https://arxiv.org/abs/2603.19935
- MAGMA: https://arxiv.org/abs/2601.03236
- MemMA: https://arxiv.org/abs/2603.18718
- A-MEM: https://arxiv.org/abs/2502.12110
- LLM Agent Memory Survey: https://openreview.net/forum?id=KPs1EgGKcT

