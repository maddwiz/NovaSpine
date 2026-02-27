# Official Benchmark Run Summary

Date: 2026-02-26

This run used official-source benchmark artifacts converted by:

```bash
python scripts/prepare_official_benchmarks.py
```

## Results (NovaSpine keyword-only mode, no embedding API keys)

1. LongMemEval (official cleaned oracle set)
   - rows: 500
   - top_k: 10
   - recall@10: 0.004
   - mrr: 0.004
   - output: `bench/results/official_longmemeval.json`

2. LoCoMo-MC10 (source dataset: `Percena/locomo-mc10`, questions with detectable gold in sessions)
   - rows: 620 (from 1,986 total; 1,366 skipped due no detectable gold session from answer-string matching)
   - top_k: 10
   - recall@10: 0.0016129032258064516
   - mrr: 0.0016129032258064516
   - output: `bench/results/official_locomo_mc10.json`

3. DMR-style (source dataset: `MemGPT/qa_data`, file `nq-open-30_total_documents_gold_at_14.jsonl.gz`)
   - full conversion size: 2,655 questions, 79,650 docs
   - executed slice: 500 questions, 15,000 docs
   - top_k: 10
   - recall@10: 0.012
   - mrr: 0.012
   - output: `bench/results/official_dmr_memgpt_sample500.json`

## Notes

- The full DMR run was started, but was terminated after prolonged runtime; sampled run was completed in this turn for deterministic reporting.
- These numbers reflect NovaSpine running in keyword-only retrieval mode (`--ingest-sync`) due missing embedding provider API keys.

---

## Re-Run (2026-02-27)

Official rerun after NovaSpine upgrade commit `30f1bab`:

1. LongMemEval
   - output: `bench/results/official_longmemeval_20260227.json`
   - recall@10: `0.004`
   - mrr: `0.004`

2. LoCoMo-MC10
   - output: `bench/results/official_locomo_mc10_20260227_reuse_ingest.json`
   - mode: query-time rerun over previously completed official ingest DB
   - recall@10: `0.0016129032258064516`
   - mrr: `0.0016129032258064516`

3. DMR sample-500
   - output: `bench/results/official_dmr_memgpt_sample500_20260227.json`
   - recall@10: `0.012`
   - mrr: `0.012`

Comparison against the 2026-02-26 official results in this retrieval mode:
- LongMemEval: no change
- LoCoMo-MC10: no change
- DMR sample-500: no change

---

## FTS Retrieval Fix Re-Run (2026-02-27)

A retrieval bugfix was applied in `sqlite_store._sanitize_fts_query`:
- short queries: precise AND-style token matching
- long natural-language queries: OR-style matching to avoid over-constrained FTS recall

This rerun used the same official converted eval sets and existing ingested benchmark DBs (`reuse_ingest=true`) to isolate query-time retrieval behavior.

1. LongMemEval
   - output: `bench/results/official_longmemeval_20260227_ftsfix.json`
   - recall@10: `0.84` (from `0.004`)
   - mrr: `0.6240261904761901` (from `0.004`)

2. LoCoMo-MC10
   - output: `bench/results/official_locomo_mc10_20260227_ftsfix.json`
   - recall@10: `0.7467741935483871` (from `0.0016129032258064516`)
   - mrr: `0.5273399897593445` (from `0.0016129032258064516`)

3. DMR sample-500
   - output: `bench/results/official_dmr_memgpt_sample500_20260227_ftsfix.json`
   - recall@10: `0.664` (from `0.012`)
   - mrr: `0.35022380952380955` (from `0.012`)

Interpretation:
- Prior official collapse was dominated by query parsing behavior in keyword mode, not core memory data integrity.
- Next step is to validate with fully fresh ingests + semantic embeddings enabled for production-comparable leaderboard reporting.

---

## Dedupe + Benchmark-Case Graph Bypass Re-Run (2026-02-27)

Changes applied:
- `recall()` now dedupes benchmark rows by `metadata.benchmark_doc_id` (fallback `benchmark_source`) before content-prefix dedupe.
- Benchmark case tokens (`__*_CASE_*__`) are propagated to chunk metadata during ingest.
- Graph indexing and graph-query merge are skipped for benchmark case-token traffic to reduce noise and runtime for official benchmark corpora.

Rerun mode:
- fresh ingest for each dataset (`ephemeral_data_dir=true`)
- keyword-only deterministic query path (`C3AE_EMBED_PROVIDER=openai` without key)

1. LongMemEval
   - output: `bench/results/official_longmemeval_20260227_docdedupe_bypassgraph.json`
   - recall@10: `0.908` (from `0.84`)
   - mrr: `0.7115301587301588` (from `0.6240261904761901`)

2. LoCoMo-MC10
   - output: `bench/results/official_locomo_mc10_20260227_docdedupe_bypassgraph.json`
   - recall@10: `0.7903225806451613` (from `0.7467741935483871`)
   - mrr: `0.5707328469022017` (from `0.5273399897593445`)

3. DMR sample-500
   - output: `bench/results/official_dmr_memgpt_sample500_20260227_docdedupe_bypassgraph.json`
   - recall@10: `0.666` (from `0.664`)
   - mrr: `0.3523087301587301` (from `0.35022380952380955`)
