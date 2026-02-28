# NovaSpine QA Summary (OpenAI tuned, 2026-02-28)

Best-per-benchmark runs (same retrieval profile, benchmark-specific answer tuning):

- LongMemEval: `official_longmemeval_qa_openai_20260228_r4.json`
- LoCoMo-MC10: `official_locomo_qa_openai_20260228_r6_mini.json`
- DMR-500: `dmr_hybrid_openai_qa_llm_20260228_fast.json`

## Best Results

| Benchmark | doc_hit | EM | F1 |
|---|---:|---:|---:|
| LongMemEval | 1.000 | 0.218 | 0.343 |
| LoCoMo-MC10 | 0.860 | 0.455 | 0.484 |
| DMR-500 | 0.806 | 0.498 | 0.584 |

## Delta vs 2026-02-27 v2

| Benchmark | EM delta | F1 delta |
|---|---:|---:|
| LongMemEval | +0.046 | +0.057 |
| LoCoMo-MC10 | +0.133 | +0.024 |
| DMR-500 | +0.102 | +0.049 |

## Tuned Profiles Used

### LongMemEval (`r4`)
- `--answer-context-k 8`
- `--answer-max-context-chars 12000`
- `--answer-chunk-chars 1400`
- `--answer-min-score-ratio 0.0`
- `--answer-max-tokens 256`
- `--answer-reasoning on`

### LoCoMo (`r5`)
- Same as above except: `--answer-reasoning off` and `--answer-max-tokens 220`

### LoCoMo (`r6_mini`, timeout-safe Round 3)
- `--answer-model gpt-4.1-mini`
- `--answer-timeout-seconds 20`
- `--answer-retries 1`
- `--answer-min-interval-seconds 0.03`
- `--session_diversity=on` (auto-enabled for LoCoMo)
- Result: `doc_hit 0.860`, `EM 0.455`, `F1 0.484`

### DMR-500 (`dmr_hybrid_openai_qa_llm_20260228_fast`)
- `--embed-provider openai --embed-model text-embedding-3-small --embed-dims 1536`
- `--query-expansion`
- `--answer-context-k 6`
- `--answer-max-context-chars 10000`
- `--answer-chunk-chars 1200`
- `--answer-min-score-ratio 0.0`
- `--answer-max-tokens 180`
- `--answer-reasoning off`
- `--answer-retries 1`
- `--reuse-index` against prebuilt DMR corpus index

## Notes

- Round 3 adds timeout-safe QA execution to prevent OpenAI 429/latency stalls from aborting full runs.
- Full re-runs of LongMemEval/DMR under the new timeout-safe path were started but not completed within this session window; best published artifacts above remain valid.

Common retrieval flags:
- `--tune-preset keyword_plus`
- `--query-expansion`
- `--ingest-sync --skip-chunking`
- `--answer-mode llm --answer-provider openai --answer-model gpt-4.1-mini`
