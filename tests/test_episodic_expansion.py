from __future__ import annotations

import asyncio

from c3ae.config import Config
from c3ae.memory_spine.spine import MemorySpine
from c3ae.retrieval.planner import plan_memory_query
from c3ae.storage.sqlite_store import SQLiteStore
from c3ae.types import Chunk


def test_build_chunk_metadata_parses_official_turn_headers(tmp_path):
    cfg = Config()
    cfg.data_dir = tmp_path
    cfg.ensure_dirs()
    spine = MemorySpine(cfg)
    try:
        metadata = spine._build_chunk_metadata(
            "\n".join(
                [
                    "Benchmark: longmemeval_m",
                    "Question ID: q-123",
                    "Session ID: session-alpha",
                    "Session date: 2024-02-03",
                    "Turn index: 12",
                    "Turn part: 3",
                    "",
                    "assistant: The table listed Admon on Sunday day shift.",
                ]
            ),
            "memory/official/longmemeval_m/q-123/session-alpha/turn_0012_03.md",
            {},
        )

        assert metadata["benchmark"] == "longmemeval_m"
        assert metadata["question_id"] == "q-123"
        assert metadata["session_id"] == "session-alpha"
        assert metadata["session_date"] == "2024-02-03"
        assert metadata["turn_index"] == 12
        assert metadata["turn_part"] == 3
        assert metadata["role"] == "assistant"
    finally:
        asyncio.run(spine.close())


def test_build_chunk_metadata_parses_locomo_dialogue_headers(tmp_path):
    cfg = Config()
    cfg.data_dir = tmp_path
    cfg.ensure_dirs()
    spine = MemorySpine(cfg)
    try:
        metadata = spine._build_chunk_metadata(
            "\n".join(
                [
                    "Benchmark: LoCoMo",
                    "Conversation sample: conv-26",
                    "Participants: Caroline, Melanie",
                    "Session: 3",
                    "Session date: 7:55 pm on 9 June, 2023",
                    "Dialogue ID: D3:13",
                    "",
                    "Caroline: I'm single, but my friends have been there through everything.",
                ]
            ),
            "memory/official/locomo/conv-26/D3_13.md",
            {"session_id": "conv-26:D3", "benchmark_doc_id": "conv-26:D3:13"},
        )

        assert metadata["benchmark"] == "LoCoMo"
        assert metadata["dialogue_id"] == "D3:13"
        assert metadata["locomo_session_index"] == 3
        assert metadata["turn_index"] == 13
        assert metadata["turn_part"] == 0
        assert metadata["speaker"] == "Caroline"
        assert metadata["session_id"] == "conv-26:D3"
    finally:
        asyncio.run(spine.close())


def test_sqlite_lists_same_episode_neighbor_chunks(tmp_path):
    store = SQLiteStore(tmp_path / "memory.db")
    try:
        before = Chunk(
            content="assistant: Sunday | Admon | Day Shift",
            metadata={"session_id": "s1", "case_id": "case-a", "turn_index": 4, "turn_part": 0},
        )
        current = Chunk(
            content="user: Which shift was Admon assigned?",
            metadata={"session_id": "s1", "case_id": "case-a", "turn_index": 5, "turn_part": 0},
        )
        after = Chunk(
            content="assistant: Monday | Blair | Night Shift",
            metadata={"session_id": "s1", "case_id": "case-a", "turn_index": 6, "turn_part": 0},
        )
        other_case = Chunk(
            content="assistant: Wrong case",
            metadata={"session_id": "s1", "case_id": "case-b", "turn_index": 5, "turn_part": 0},
        )
        for chunk in (before, current, after, other_case):
            store.insert_chunk(chunk)

        neighbors = store.list_neighbor_chunks(
            session_id="s1",
            turn_index=5,
            window=1,
            case_id="case-a",
            exclude_id=current.id,
        )

        assert [chunk.id for chunk in neighbors] == [before.id, after.id]
    finally:
        store.close()


def test_locomo_dialogue_metadata_can_expand_when_query_needs_episode_context(tmp_path):
    cfg = Config()
    cfg.data_dir = tmp_path
    cfg.retrieval.episodic_expansion_enabled = True
    cfg.retrieval.episodic_expansion_window = 1
    cfg.ensure_dirs()
    spine = MemorySpine(cfg)
    try:
        before = Chunk(
            content="Dialogue ID: D3:12\n\nMelanie: Are you dating anyone right now?",
            metadata={
                "benchmark": "locomo",
                "session_id": "conv-26:D3",
                "case_id": "conv-26",
                "turn_index": 12,
                "turn_part": 0,
                "speaker": "Melanie",
                "benchmark_doc_id": "conv-26:D3:12",
            },
        )
        current = Chunk(
            content="Dialogue ID: D3:13\n\nCaroline: I'm single, but my friends support me.",
            metadata={
                "benchmark": "locomo",
                "session_id": "conv-26:D3",
                "case_id": "conv-26",
                "turn_index": 13,
                "turn_part": 0,
                "speaker": "Caroline",
                "benchmark_doc_id": "conv-26:D3:13",
            },
        )
        after = Chunk(
            content="Dialogue ID: D3:14\n\nMelanie: I'm glad your friends are there for you.",
            metadata={
                "benchmark": "locomo",
                "session_id": "conv-26:D3",
                "case_id": "conv-26",
                "turn_index": 14,
                "turn_part": 0,
                "speaker": "Melanie",
                "benchmark_doc_id": "conv-26:D3:14",
            },
        )
        for chunk in (before, current, after):
            spine.sqlite.insert_chunk(chunk)

        row = {
            "id": current.id,
            "content": current.content,
            "score": 1.0,
            "source": "test",
            "metadata": dict(current.metadata),
        }
        expanded = spine._maybe_expand_episodic_row(
            row,
            plan_memory_query("Which things happened before and after Caroline said she was single?"),
        )

        assert expanded["metadata"]["episodic_expanded"] is True
        assert expanded["metadata"]["episodic_neighbor_chunk_ids"] == [before.id, after.id]
        assert expanded["metadata"]["episodic_neighbor_benchmark_doc_ids"] == [
            "conv-26:D3:12",
            "conv-26:D3:14",
        ]
        assert "Melanie: Are you dating anyone right now?" in expanded["content"]
        assert "speaker=Melanie" in expanded["content"]
    finally:
        asyncio.run(spine.close())


def test_episodic_expansion_adds_neighbor_context_for_table_queries(tmp_path):
    cfg = Config()
    cfg.data_dir = tmp_path
    cfg.retrieval.episodic_expansion_enabled = True
    cfg.retrieval.episodic_expansion_window = 1
    cfg.ensure_dirs()
    spine = MemorySpine(cfg)
    try:
        table = Chunk(
            content="assistant: | Day | Person | Shift |\n| Sunday | Admon | Day Shift |",
            metadata={
                "session_id": "schedule-session",
                "case_id": "longmemeval_m::q-table",
                "turn_index": 7,
                "turn_part": 0,
                "role": "assistant",
            },
        )
        followup = Chunk(
            content="user: Let's revisit Admon's Sunday rotation.",
            metadata={
                "session_id": "schedule-session",
                "case_id": "longmemeval_m::q-table",
                "turn_index": 8,
                "turn_part": 0,
                "role": "user",
            },
        )
        spine.sqlite.insert_chunk(table)
        spine.sqlite.insert_chunk(followup)

        row = {
            "id": followup.id,
            "content": followup.content,
            "score": 1.0,
            "source": "test",
            "metadata": dict(followup.metadata),
        }
        expanded = spine._maybe_expand_episodic_row(
            row,
            plan_memory_query("In the schedule table, which shift is Admon assigned on Sunday?"),
        )

        assert expanded["metadata"]["episodic_expanded"] is True
        assert expanded["metadata"]["episodic_neighbor_chunk_ids"] == [table.id]
        assert "Sunday | Admon | Day Shift" in expanded["content"]
        assert "<EpisodicNeighborContext>" in expanded["content"]
    finally:
        asyncio.run(spine.close())


def test_episodic_expansion_max_chars_zero_disables_neighbor_payload(tmp_path):
    cfg = Config()
    cfg.data_dir = tmp_path
    cfg.retrieval.episodic_expansion_enabled = True
    cfg.retrieval.episodic_expansion_max_chars = 0
    cfg.ensure_dirs()
    spine = MemorySpine(cfg)
    try:
        table = Chunk(
            content="assistant: | Day | Person | Shift |\n| Sunday | Admon | Day Shift |",
            metadata={"session_id": "s-max", "turn_index": 1, "turn_part": 0},
        )
        followup = Chunk(
            content="user: Which shift was Admon assigned?",
            metadata={"session_id": "s-max", "turn_index": 2, "turn_part": 0},
        )
        spine.sqlite.insert_chunk(table)
        spine.sqlite.insert_chunk(followup)

        row = {
            "id": followup.id,
            "content": followup.content,
            "score": 1.0,
            "source": "test",
            "metadata": dict(followup.metadata),
        }
        expanded = spine._maybe_expand_episodic_row(
            row,
            plan_memory_query("In the schedule table, which shift is Admon assigned on Sunday?"),
        )

        assert expanded is row
        assert "EpisodicNeighborContext" not in expanded["content"]
    finally:
        asyncio.run(spine.close())
