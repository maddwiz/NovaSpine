# NovaSpine QA Target Bands (2026-02-28)

Practical target bands for OpenAI-powered end-to-end QA (`doc_hit`, `EM`, `F1`).

## Per-Benchmark Targets

| Benchmark | Competitive | Great | Exceptional |
|---|---:|---:|---:|
| LongMemEval | doc_hit >= 0.95, EM >= 0.30, F1 >= 0.45 | doc_hit >= 0.98, EM >= 0.40, F1 >= 0.55 | doc_hit = 1.00, EM >= 0.50, F1 >= 0.65 |
| LoCoMo-MC10 | doc_hit >= 0.82, EM >= 0.32, F1 >= 0.46 | doc_hit >= 0.88, EM >= 0.42, F1 >= 0.56 | doc_hit >= 0.92, EM >= 0.50, F1 >= 0.62 |
| DMR-500 | doc_hit >= 0.74, EM >= 0.46, F1 >= 0.56 | doc_hit >= 0.82, EM >= 0.54, F1 >= 0.64 | doc_hit >= 0.88, EM >= 0.60, F1 >= 0.70 |

## Portfolio-Level Targets

- Competitive overall: average `doc_hit >= 0.84`, average `EM >= 0.36`, average `F1 >= 0.49`.
- Great overall: average `doc_hit >= 0.90`, average `EM >= 0.45`, average `F1 >= 0.58`.
- Exceptional overall: average `doc_hit >= 0.94`, average `EM >= 0.53`, average `F1 >= 0.66`.
