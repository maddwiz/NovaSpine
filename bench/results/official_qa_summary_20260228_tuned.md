# NovaSpine QA Summary (Tuned Extractive)

Date: 2026-02-28

Configuration:
- `scripts/run_memory_qa.py`
- `--tune-preset keyword_plus`
- `--query-expansion`
- `--ingest-sync --skip-chunking`
- `--answer-mode extractive`

## Results

| Benchmark | doc_hit | EM | F1 |
|---|---:|---:|---:|
| LongMemEval | 1.000 | 0.052 | 0.122 |
| LoCoMo-MC10 | 0.860 | 0.021 | 0.074 |
| DMR-500 | 0.706 | 0.016 | 0.037 |

Artifacts:
- `bench/results/official_longmemeval_qa_20260228_tuned.json`
- `bench/results/official_locomo_qa_20260228_tuned.json`
- `bench/results/official_dmr_qa_20260228_tuned.json`

Notes:
- Retrieval remains strong (same doc-hit profile as prior tuned retrieval runs).
- Extractive answer mode remains a hard ceiling for EM/F1 on multi-step/semantic QA.
- Use `--answer-mode llm --answer-provider openai` for competitor-style end-to-end QA once `OPENAI_API_KEY` is available in the current shell environment.
