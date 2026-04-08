from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from c3ae.cli import main
from c3ae.config import Config
from c3ae.memory_spine.spine import MemorySpine


def test_ingest_session_runs_vector_indexing(tmp_path: Path, monkeypatch):
    session_path = tmp_path / "session.jsonl"
    session_path.write_text(
        "\n".join(
            [
                json.dumps({"type": "session", "session_id": "s1"}),
                json.dumps(
                    {
                        "type": "message",
                        "message": {
                            "role": "user",
                            "content": "Desmond is asking for a durable session import test message.",
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "message",
                        "message": {
                            "role": "assistant",
                            "content": "Nemo is replying with enough detail to be preserved for semantic recall.",
                        },
                    }
                ),
            ]
        )
        + "\n"
    )

    cfg = Config()
    cfg.data_dir = tmp_path / "data"
    spine = MemorySpine(cfg)
    calls: list[tuple[list[str], list[str]]] = []

    async def fake_embed_and_index(chunk_ids: list[str], texts: list[str]) -> None:
        calls.append((list(chunk_ids), list(texts)))

    try:
        monkeypatch.setattr(spine, "_embed_and_index", fake_embed_and_index)
        result = spine.ingest_session(session_path)
    finally:
        spine.sqlite.close()

    assert result["chunks_ingested"] == 2
    assert len(calls) == 1
    assert len(calls[0][0]) == 2
    assert calls[0][1] == [
        "Desmond is asking for a durable session import test message.",
        "Nemo is replying with enough detail to be preserved for semantic recall.",
    ]


def test_cli_ingest_passes_source_id(tmp_path: Path, monkeypatch):
    file_path = tmp_path / "note.txt"
    file_path.write_text("hello from file ingest")

    class FakeSQLite:
        def close(self) -> None:
            return None

    class FakeSpine:
        def __init__(self) -> None:
            self.sqlite = FakeSQLite()
            self.calls: list[tuple[Path, str]] = []

        async def ingest_file(self, file_path: Path, metadata=None, source_id: str = ""):
            self.calls.append((file_path, source_id))
            return ["chunk-1"]

    fake_spine = FakeSpine()
    monkeypatch.setattr("c3ae.cli._get_spine", lambda data_dir=None: fake_spine)

    runner = CliRunner()
    result = runner.invoke(main, ["ingest", str(file_path), "--source-id", "unit:test"])

    assert result.exit_code == 0, result.output
    assert fake_spine.calls == [(file_path, "unit:test")]
