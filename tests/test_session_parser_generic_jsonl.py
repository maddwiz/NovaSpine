from __future__ import annotations

from pathlib import Path

from c3ae.ingestion.session_parser import SessionParser


def test_generic_role_content_jsonl(tmp_path: Path) -> None:
    session_path = tmp_path / "generic.jsonl"
    session_path.write_text(
        "\n".join(
            [
                '{"role":"user","content":"Hello from user with enough words to pass threshold."}',
                '{"role":"assistant","content":"Assistant response with enough text to parse cleanly."}',
                '{"role":"tool","content":"Tool call summary with enough content for parser to keep."}',
            ]
        )
    )

    parser = SessionParser()
    chunks = parser.parse_file(session_path)

    assert len(chunks) == 3
    assert chunks[0].role == "user"
    assert chunks[1].role == "assistant"
    assert chunks[2].role == "tool_call"
