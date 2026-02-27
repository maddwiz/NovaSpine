#!/usr/bin/env python3
"""Prepare official-source benchmark corpora/eval files for NovaSpine.

Outputs JSONL files under bench/official/converted/:
  - longmemeval_oracle_corpus.jsonl
  - longmemeval_oracle_eval.jsonl
  - longmemeval_oracle_qa_eval.jsonl
  - locomo_mc10_corpus.jsonl
  - locomo_mc10_eval.jsonl
  - locomo_mc10_qa_eval.jsonl
  - dmr_memgpt_corpus.jsonl
  - dmr_memgpt_eval.jsonl
  - dmr_memgpt_qa_eval.jsonl
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


LONGMEMEVAL_URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json"
)
DMR_URL = (
    "https://huggingface.co/datasets/MemGPT/qa_data/resolve/main/"
    "nq-open-30_total_documents_gold_at_14.jsonl.gz"
)
LOCOMO_MC10_DATASET = "Percena/locomo-mc10"
LOCOMO_MC10_URL = (
    "https://huggingface.co/datasets/Percena/locomo-mc10/resolve/main/data/locomo_mc10.json"
)
HF_DATASET_SERVER_ROWS = "https://datasets-server.huggingface.co/rows"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare official-source benchmark conversions.")
    p.add_argument(
        "--out-dir",
        default="bench/official/converted",
        help="Output directory for converted corpus/eval JSONL files",
    )
    p.add_argument(
        "--source-dir",
        default="bench/official",
        help="Directory to store downloaded source benchmark files",
    )
    p.add_argument(
        "--locomo-max-rows",
        type=int,
        default=1986,
        help="Max LoCoMo-MC10 rows to convert",
    )
    p.add_argument(
        "--dmr-max-rows",
        type=int,
        default=2655,
        help="Max MemGPT QA rows to convert",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    source_dir = Path(args.source_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    lme_path = source_dir / "longmemeval_oracle.json"
    dmr_path = source_dir / "dmr_memgpt_gold_at_14.jsonl.gz"
    locomo_path = source_dir / "locomo_mc10.json"

    _download_if_missing(LONGMEMEVAL_URL, lme_path)
    _download_if_missing(DMR_URL, dmr_path)
    _download_if_missing(LOCOMO_MC10_URL, locomo_path)

    summary: dict[str, Any] = {}
    summary["longmemeval"] = _convert_longmemeval(
        src=lme_path,
        corpus_out=out_dir / "longmemeval_oracle_corpus.jsonl",
        eval_out=out_dir / "longmemeval_oracle_eval.jsonl",
        qa_eval_out=out_dir / "longmemeval_oracle_qa_eval.jsonl",
    )
    summary["locomo_mc10"] = _convert_locomo_mc10(
        src=locomo_path,
        corpus_out=out_dir / "locomo_mc10_corpus.jsonl",
        eval_out=out_dir / "locomo_mc10_eval.jsonl",
        qa_eval_out=out_dir / "locomo_mc10_qa_eval.jsonl",
        max_rows=max(1, int(args.locomo_max_rows)),
    )
    summary["dmr_memgpt"] = _convert_dmr_memgpt(
        src=dmr_path,
        corpus_out=out_dir / "dmr_memgpt_corpus.jsonl",
        eval_out=out_dir / "dmr_memgpt_eval.jsonl",
        qa_eval_out=out_dir / "dmr_memgpt_qa_eval.jsonl",
        max_rows=max(1, int(args.dmr_max_rows)),
    )

    summary_path = out_dir / "conversion_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "summary_file": str(summary_path), "summary": summary}, indent=2))


def _download_if_missing(url: str, out_path: Path) -> None:
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    req = Request(url, headers={"User-Agent": "NovaSpine-benchmark-prep/1.0"})
    with urlopen(req, timeout=120) as r, out_path.open("wb") as f:
        f.write(r.read())


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _flatten_session(session: Any) -> str:
    if isinstance(session, str):
        return session
    if not isinstance(session, list):
        return str(session)
    parts: list[str] = []
    for turn in session:
        if isinstance(turn, dict):
            role = str(turn.get("role", "speaker"))
            content = str(turn.get("content", "")).strip()
            if content:
                parts.append(f"{role}: {content}")
        else:
            s = str(turn).strip()
            if s:
                parts.append(s)
    return "\n".join(parts)


def _convert_longmemeval(
    src: Path,
    corpus_out: Path,
    eval_out: Path,
    qa_eval_out: Path,
) -> dict[str, Any]:
    rows = json.loads(src.read_text(encoding="utf-8"))
    corpus_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    qa_eval_rows: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        qid = str(row.get("question_id", f"q_{i:04d}"))
        case_token = f"__LME_CASE_{i:04d}__"
        sid_to_doc_id: dict[str, str] = {}
        haystack_ids = list(row.get("haystack_session_ids", []))
        haystack_sessions = list(row.get("haystack_sessions", []))
        for sid, session in zip(haystack_ids, haystack_sessions):
            sid_s = str(sid)
            doc_id = f"lme_{qid}_{sid_s}"
            sid_to_doc_id[sid_s] = doc_id
            text = _flatten_session(session)
            corpus_rows.append(
                {
                    "doc_id": doc_id,
                    "source_id": f"lme:{qid}:{sid_s}",
                    "text": f"{case_token}\n{text}",
                    "metadata": {
                        "benchmark": "longmemeval_oracle",
                        "question_id": qid,
                        "session_id": sid_s,
                    },
                }
            )

        expected_doc_ids = [
            sid_to_doc_id[sid]
            for sid in [str(x) for x in row.get("answer_session_ids", [])]
            if sid in sid_to_doc_id
        ]
        if not expected_doc_ids:
            continue
        eval_rows.append(
            {
                "query": f"{case_token} {str(row.get('question', '')).strip()}",
                "expected_doc_ids": expected_doc_ids,
            }
        )
        answer = str(row.get("answer", "")).strip()
        if answer:
            qa_eval_rows.append(
                {
                    "query": f"{case_token} {str(row.get('question', '')).strip()}",
                    "expected_doc_ids": expected_doc_ids,
                    "expected_answers": [answer],
                }
            )

    _write_jsonl(corpus_out, corpus_rows)
    _write_jsonl(eval_out, eval_rows)
    _write_jsonl(qa_eval_out, qa_eval_rows)
    return {
        "questions": len(eval_rows),
        "qa_questions": len(qa_eval_rows),
        "documents": len(corpus_rows),
        "source_file": str(src),
        "corpus_file": str(corpus_out),
        "eval_file": str(eval_out),
        "qa_eval_file": str(qa_eval_out),
    }


def _iter_hf_rows(dataset: str, config: str = "default", split: str = "train", page_size: int = 50):
    offset = 0
    while True:
        params = urlencode(
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": offset,
                "length": page_size,
            }
        )
        url = f"{HF_DATASET_SERVER_ROWS}?{params}"
        payload = _get_json_with_retry(url, retries=8, base_sleep=0.5)
        rows = payload.get("rows", [])
        if not rows:
            break
        for item in rows:
            row = item.get("row")
            if isinstance(row, dict):
                yield row
        n = len(rows)
        offset += n
        if n < page_size:
            break
        time.sleep(0.05)


def _get_json_with_retry(url: str, retries: int = 6, base_sleep: float = 0.5) -> dict[str, Any]:
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": "NovaSpine-benchmark-prep/1.0"})
            with urlopen(req, timeout=120) as r:
                return json.loads(r.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as e:
            last_err = e
            sleep_s = base_sleep * (2 ** attempt)
            time.sleep(min(sleep_s, 8.0))
            continue
    if last_err is not None:
        raise last_err
    raise RuntimeError("unexpected retry failure without exception")


def _convert_locomo_mc10(
    src: Path,
    corpus_out: Path,
    eval_out: Path,
    qa_eval_out: Path,
    max_rows: int,
) -> dict[str, Any]:
    dataset_rows: list[dict[str, Any]] = []
    try:
        rows_obj = json.loads(src.read_text(encoding="utf-8"))
        if isinstance(rows_obj, dict):
            maybe_rows = rows_obj.get("data", [])
            if isinstance(maybe_rows, list):
                dataset_rows = [r for r in maybe_rows if isinstance(r, dict)]
        elif isinstance(rows_obj, list):
            dataset_rows = [r for r in rows_obj if isinstance(r, dict)]
    except json.JSONDecodeError:
        # Source file can be newline-delimited JSON.
        with src.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    dataset_rows.append(obj)

    corpus_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    qa_eval_rows: list[dict[str, Any]] = []
    skipped_no_gold = 0
    n_rows = 0
    for i, row in enumerate(dataset_rows):
        if i >= max_rows:
            break
        if not isinstance(row, dict):
            continue
        n_rows += 1
        qid = str(row.get("question_id", f"locomo_{i:04d}"))
        case_token = f"__LOCOMO_CASE_{i:04d}__"
        answer = str(row.get("answer", "")).strip().lower()
        question = str(row.get("question", "")).strip()
        question_type = str(row.get("question_type", "")).strip().lower()

        summaries = list(row.get("haystack_session_summaries", []))
        sessions = list(row.get("haystack_sessions", []))
        session_ids = [str(x) for x in row.get("haystack_session_ids", [])]

        expected_doc_ids: list[str] = []
        for j, sid in enumerate(session_ids):
            summary_text = str(summaries[j]) if j < len(summaries) else ""
            session_text = _flatten_session(sessions[j]) if j < len(sessions) else ""
            doc_text = summary_text.strip() or session_text.strip()
            if not doc_text:
                continue
            doc_id = f"locomo_{qid}_{sid}"
            corpus_rows.append(
                {
                    "doc_id": doc_id,
                    "source_id": f"locomo:{qid}:{sid}",
                    "text": f"{case_token}\n{doc_text}",
                    "metadata": {
                        "benchmark": "locomo_mc10",
                        "question_id": qid,
                        "question_type": question_type,
                        "session_id": sid,
                    },
                }
            )
            hay = f"{summary_text}\n{session_text}".lower()
            if answer and answer != "not answerable" and answer in hay:
                expected_doc_ids.append(doc_id)

        if not expected_doc_ids:
            skipped_no_gold += 1
            continue
        eval_rows.append(
            {
                "query": f"{case_token} {question}",
                "expected_doc_ids": expected_doc_ids,
            }
        )
        if answer and answer != "not answerable":
            qa_eval_rows.append(
                {
                    "query": f"{case_token} {question}",
                    "expected_doc_ids": expected_doc_ids,
                    "expected_answers": [str(row.get("answer", "")).strip()],
                }
            )

    _write_jsonl(corpus_out, corpus_rows)
    _write_jsonl(eval_out, eval_rows)
    _write_jsonl(qa_eval_out, qa_eval_rows)
    return {
        "rows_read": n_rows,
        "questions": len(eval_rows),
        "qa_questions": len(qa_eval_rows),
        "documents": len(corpus_rows),
        "skipped_no_gold": skipped_no_gold,
        "dataset": LOCOMO_MC10_DATASET,
        "source_file": str(src),
        "corpus_file": str(corpus_out),
        "eval_file": str(eval_out),
        "qa_eval_file": str(qa_eval_out),
    }


def _convert_dmr_memgpt(
    src: Path,
    corpus_out: Path,
    eval_out: Path,
    qa_eval_out: Path,
    max_rows: int,
) -> dict[str, Any]:
    corpus_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    qa_eval_rows: list[dict[str, Any]] = []
    skipped_no_gold = 0
    n_rows = 0
    with gzip.open(src, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n_rows += 1
            case_token = f"__DMR_CASE_{i:05d}__"
            question = str(row.get("question", "")).strip()
            ctxs = row.get("ctxs", [])
            if not isinstance(ctxs, list):
                continue

            expected_doc_ids: list[str] = []
            for j, ctx in enumerate(ctxs):
                if not isinstance(ctx, dict):
                    continue
                doc_id = f"dmr_{i:05d}_{j:02d}"
                title = str(ctx.get("title", "")).strip()
                text = str(ctx.get("text", "")).strip()
                corpus_rows.append(
                    {
                        "doc_id": doc_id,
                        "source_id": f"dmr:{i:05d}:{j:02d}",
                        "text": f"{case_token}\n{title}\n{text}",
                        "metadata": {
                            "benchmark": "dmr_memgpt",
                            "row_index": i,
                            "ctx_index": j,
                            "ctx_id": str(ctx.get("id", "")),
                            "hasanswer": bool(ctx.get("hasanswer", False)),
                            "isgold": bool(ctx.get("isgold", False)),
                        },
                    }
                )
                if bool(ctx.get("hasanswer", False)) or bool(ctx.get("isgold", False)):
                    expected_doc_ids.append(doc_id)

            if not expected_doc_ids:
                skipped_no_gold += 1
                continue
            eval_rows.append(
                {
                    "query": f"{case_token} {question}",
                    "expected_doc_ids": expected_doc_ids,
                }
            )
            answers = row.get("answers", [])
            if isinstance(answers, list):
                gold_answers = [str(a).strip() for a in answers if str(a).strip()]
            else:
                gold_answers = []
            if gold_answers:
                qa_eval_rows.append(
                    {
                        "query": f"{case_token} {question}",
                        "expected_doc_ids": expected_doc_ids,
                        "expected_answers": gold_answers,
                    }
                )

    _write_jsonl(corpus_out, corpus_rows)
    _write_jsonl(eval_out, eval_rows)
    _write_jsonl(qa_eval_out, qa_eval_rows)
    return {
        "rows_read": n_rows,
        "questions": len(eval_rows),
        "qa_questions": len(qa_eval_rows),
        "documents": len(corpus_rows),
        "skipped_no_gold": skipped_no_gold,
        "source_file": str(src),
        "corpus_file": str(corpus_out),
        "eval_file": str(eval_out),
        "qa_eval_file": str(qa_eval_out),
    }


if __name__ == "__main__":
    main()
