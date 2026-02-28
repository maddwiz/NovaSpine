# NovaSpine QA Summary (OpenAI tuned, 2026-02-28)

Best-per-benchmark runs (same retrieval profile, benchmark-specific answer tuning):

- LongMemEval: `official_longmemeval_qa_openai_20260228_r4.json`
- LoCoMo-MC10: `official_locomo_qa_openai_20260228_r5.json`
- DMR-500: `official_dmr_qa_openai_20260228_r4.json`

## Best Results

| Benchmark | doc_hit | EM | F1 |
|---|---:|---:|---:|
| LongMemEval | 1.000 | 0.218 | 0.343 |
| LoCoMo-MC10 | 0.860 | 0.364 | 0.464 |
| DMR-500 | 0.706 | 0.446 | 0.539 |

## Delta vs 2026-02-27 v2

| Benchmark | EM delta | F1 delta |
|---|---:|---:|
| LongMemEval | +0.046 | +0.057 |
| LoCoMo-MC10 | +0.042 | +0.004 |
| DMR-500 | +0.050 | +0.004 |

## Tuned Profiles Used

### LongMemEval / DMR (`r4`)
- `--answer-context-k 8`
- `--answer-max-context-chars 12000`
- `--answer-chunk-chars 1400`
- `--answer-min-score-ratio 0.0`
- `--answer-max-tokens 256`
- `--answer-reasoning on`

### LoCoMo (`r5`)
- Same as above except: `--answer-reasoning off` and `--answer-max-tokens 220`

Common retrieval flags:
- `--tune-preset keyword_plus`
- `--query-expansion`
- `--ingest-sync --skip-chunking`
- `--answer-mode llm --answer-provider openai --answer-model gpt-4.1-mini`
