from __future__ import annotations

import json

from scripts.prepare_official_benchmarks import _convert_longmemeval


def test_prepare_longmemeval_emits_turn_granularity(tmp_path):
    source = tmp_path / "longmemeval_oracle.json"
    source.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "question": "Which shift is Admon assigned?",
                    "answer": "Day Shift",
                    "haystack_session_ids": ["s1"],
                    "answer_session_ids": ["s1"],
                    "haystack_sessions": [
                        [
                            {"role": "user", "content": "Please review the schedule."},
                            {"role": "assistant", "content": "| Day | Person | Shift |\n| Sunday | Admon | Day Shift |"},
                        ]
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    corpus = tmp_path / "corpus.jsonl"
    retrieval = tmp_path / "eval.jsonl"
    qa = tmp_path / "qa.jsonl"

    summary = _convert_longmemeval(source, corpus, retrieval, qa)
    rows = [json.loads(line) for line in corpus.read_text(encoding="utf-8").splitlines()]

    assert summary["documents"] == 2
    assert rows[0]["doc_id"] == rows[1]["doc_id"]
    assert rows[0]["metadata"]["turn_index"] == 0
    assert rows[1]["metadata"]["turn_index"] == 1
    assert "Turn index: 1" in rows[1]["text"]
