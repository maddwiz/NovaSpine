# Next Full Hybrid QA Commands (OpenAI)

Run these in a shell with `OPENAI_API_KEY` exported.

## 1) Build hybrid index + retrieval check (DMR)

```bash
PYTHONPATH=src .venv/bin/python scripts/run_memory_benchmarks.py \
  --name dmr_hybrid_openai_probe_20260228 \
  --dataset ./bench/official/converted/dmr_memgpt_eval_500.jsonl \
  --corpus ./bench/official/converted/dmr_memgpt_corpus_500.jsonl \
  --top-k 10 \
  --skip-chunking \
  --ingest-batch-size 128 \
  --embed-provider openai \
  --embed-model text-embedding-3-small \
  --embed-dims 1536 \
  --query-expansion \
  --out ./bench/results/dmr_hybrid_openai_probe_20260228.json
```

## 2) Full DMR LLM QA using the built index

```bash
PYTHONPATH=src .venv/bin/python scripts/run_memory_qa.py \
  --name dmr_hybrid_openai_qa_llm_20260228 \
  --dataset ./bench/official/converted/dmr_memgpt_qa_eval.jsonl \
  --top-k 10 \
  --data-dir /var/folders/cr/_248dmcx12b6ykhhf3_8rxlh0000gn/T/novaspine-qa-9x5kvqnt \
  --embed-provider openai \
  --embed-model text-embedding-3-small \
  --embed-dims 1536 \
  --query-expansion \
  --answer-mode llm \
  --answer-provider openai \
  --answer-model gpt-4.1-mini \
  --answer-context-k 8 \
  --answer-max-context-chars 12000 \
  --answer-chunk-chars 1400 \
  --answer-min-score-ratio 0.0 \
  --answer-max-tokens 256 \
  --answer-reasoning on \
  --out ./bench/results/dmr_hybrid_openai_qa_llm_20260228.json
```

Notes:
- Hybrid retrieval already validated at doc_hit/recall@10 `0.808`.
- This full QA run is long (500 LLM answers) and may take extended wall-clock time.
