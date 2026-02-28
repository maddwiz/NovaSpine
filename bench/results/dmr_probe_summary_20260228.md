# DMR Retrieval Probe Summary (2026-02-28)

Quick probe runs to improve DMR retrieval before full QA reruns.

## Results

| Run | Config | doc_hit / recall@10 |
|---|---|---:|
| `dmr_retrieval_probe_base_20260228.json` | keyword-plus + query expansion, sync ingest | 0.706 |
| `dmr_retrieval_probe_variants_20260228.json` | + recall variants (fetch x6, rrf_k=20) | 0.696 |
| `dmr_hybrid_openai_probe_20260228.json` | hybrid embeddings (`text-embedding-3-small`, dims 1536), async batch ingest 128 | **0.808** |
| `dmr_hybrid_openai_qa_probe_extractive_20260228.json` | same hybrid config, QA harness extractive mode | **0.808 doc_hit** |

## Conclusion

- Recall variants hurt DMR in this setup.
- Best improvement came from true hybrid retrieval with batched embedding ingest.
- Next full QA runs should use the hybrid embedding profile to convert retrieval gain into EM/F1 gains.
