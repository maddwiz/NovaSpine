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
