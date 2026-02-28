# NovaSpine QA Summary (OpenAI tuned, overnight 2026-02-28)

Updated from fresh full QA runs using the timeout-safe pipeline and tuned retrieval profiles.

## Best Artifacts

- LongMemEval: `official_longmemeval_qa_openai_20260228_r9.json`
- LoCoMo-MC10 (answer quality profile): `official_locomo_qa_openai_20260228_r6_mini.json`
- LoCoMo-MC10 (high-recall profile): `official_locomo_qa_openai_20260228_r9_k15_lexctx.json`
- DMR-500: `official_dmr_qa_openai_20260228_r9_k15.json`

## Current Results

| Benchmark | Profile | doc_hit | EM | F1 |
|---|---|---:|---:|---:|
| LongMemEval | `r9` | 1.000 | 0.304 | 0.367 |
| LoCoMo-MC10 | `r6_mini` (quality) | 0.860 | 0.455 | 0.484 |
| LoCoMo-MC10 | `r9_k15_lexctx` (recall) | 0.944 | 0.448 | 0.458 |
| DMR-500 | `r9_k15` | 0.884 | 0.590 | 0.579 |

## Delta vs Prior OpenAI Tuned Summary

| Benchmark | doc_hit delta | EM delta | F1 delta |
|---|---:|---:|---:|
| LongMemEval (`r9` vs `r4`) | +0.000 | +0.086 | +0.024 |
| DMR-500 (`r9_k15` vs `fast`) | +0.078 | +0.092 | -0.005 |

Notes:
- DMR improvements are large on retrieval and EM; F1 remains near-flat (slightly lower vs earlier fast profile).
- LoCoMo still has a retrieval-vs-answer tradeoff, but `k15` with lexical context rerank closes much of the EM gap while retaining high doc-hit.
- Additional r10/r11 ablations (legacy normalization variants, wider LoCoMo context, LLM rerank-order-only, stricter typed fallback) were run; none beat the current best-per-benchmark profiles above.

## High-Impact Tuning Confirmed

1. DMR retrieval sensitivity to weighting/profile:
- `keyword_plus` underused semantic retrieval on DMR (doc-hit regressed).
- Default/adaptive weighting with OpenAI embeddings + `top_k=15` produced best DMR doc-hit (`0.882` retrieval-only; `0.884` in QA run).

2. LongMemEval answer routing:
- `gpt-4.1-mini` with reasoning enabled and broader context remained best overall in full-run EM/F1.
- Hard-route to `gpt-4.1` helped some slices but regressed on larger runs.
- Direct `gpt-4.1` probe on a 120-row slice also regressed (`EM 0.375`, `F1 0.392`) vs `gpt-4.1-mini` (`EM 0.525`, `F1 0.547`).

3. GPT-5-mini probe in this pipeline:
- `long_probe_gpt5mini_20260228.json` underperformed (`EM 0.200`, `F1 0.206` on 40-row slice).
- Current chat-completions QA path remains best with `gpt-4.1-mini` for stable benchmark output.

4. Answer-stage normalization + context selection:
- Typed answer normalization (`NUMBER/DATE/YES_NO` focused extraction) improved formatting consistency and reduced verbose answer drift.
- LoCoMo high-recall profile improved with answer-context lexical reranking (`EM 0.420 -> 0.448` at `doc_hit 0.944`).

## Recommended Command Profiles

### LongMemEval (best current)
- `--top-k 10`
- `--tune-preset keyword_plus --query-expansion`
- `--answer-model gpt-4.1-mini --answer-route-hard off`
- `--answer-context-k 8 --answer-max-context-chars 12000 --answer-chunk-chars 1400`
- `--answer-max-tokens 256 --answer-reasoning on`

### LoCoMo-MC10
- Quality profile:
  - `--top-k 10`
  - `--tune-preset keyword_plus --query-expansion`
  - `--answer-model gpt-4.1-mini --answer-reasoning off`
- High-recall profile:
  - same as above but `--top-k 15`
  - add `--answer-context-rerank lexical --answer-context-pool-multiplier 4 --answer-context-overlap-weight 0.45`

### DMR-500 (best current)
- `--top-k 15`
- `--embed-provider openai --embed-model text-embedding-3-small --embed-dims 1536`
- `--query-expansion`
- `--answer-model gpt-4.1-mini --answer-route-hard off --answer-reasoning off`
- `--answer-context-k 6 --answer-max-context-chars 10000 --answer-chunk-chars 1200`

## New/Relevant Artifacts Added This Round

- `bench/results/official_longmemeval_qa_openai_20260228_r9.json`
- `bench/results/official_dmr_qa_openai_20260228_r9_k15.json`
- `bench/results/dmr_retrieval_full_openai_small_r7_default_k15_20260228.json`
- `bench/results/official_locomo_qa_openai_20260228_r9_k15_lexctx.json`
- probe artifacts for ablations in `bench/results/*_20260228.json`
