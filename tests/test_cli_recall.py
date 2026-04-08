from __future__ import annotations

import json

from click.testing import CliRunner

from c3ae.cli import main


class _FakeSQLite:
    def close(self) -> None:
        return None


class _FakeSpine:
    def __init__(self) -> None:
        self.sqlite = _FakeSQLite()
        self.recall_calls: list[tuple[str, int]] = []

    async def recall(self, query: str, top_k: int = 20):
        self.recall_calls.append((query, top_k))
        return [
            {
                "id": "mem-1",
                "content": "Desmond prefers exact facts over vague summaries.\nHe also likes concise memory previews.",
                "score": 0.8721,
                "source": "hybrid",
                "metadata": {"source_id": "note-1"},
            }
        ]

    def status(self):
        return {
            "chunks": 12,
            "vectors": 12,
            "reasoning_entries": 2,
            "skills": 1,
            "vault_documents": 0,
        }


def test_cli_recall_uses_high_level_recall_and_formats_output(monkeypatch):
    fake_spine = _FakeSpine()
    monkeypatch.setattr("c3ae.cli._get_spine", lambda data_dir=None: fake_spine)

    runner = CliRunner()
    result = runner.invoke(main, ["recall", "what does Desmond prefer?", "--top-k", "1"])

    assert result.exit_code == 0, result.output
    assert fake_spine.recall_calls == [("what does Desmond prefer?", 1)]
    assert "NovaSpine Recall: what does Desmond prefer?" in result.output
    assert "--- Memory 1 (score: 0.8721, source: hybrid) ---" in result.output
    assert "Source ID: note-1" in result.output
    assert "Desmond prefers exact facts over vague summaries. He also likes concise memory previews." in result.output


def test_cli_recall_supports_json_output(monkeypatch):
    fake_spine = _FakeSpine()
    monkeypatch.setattr("c3ae.cli._get_spine", lambda data_dir=None: fake_spine)

    runner = CliRunner()
    result = runner.invoke(main, ["recall", "retrieval", "--json-output"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload[0]["id"] == "mem-1"
    assert payload[0]["source"] == "hybrid"


def test_cli_help_is_novaspine_branded_and_lists_core_commands():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])

    assert result.exit_code == 0, result.output
    assert "NovaSpine CLI" in result.output
    for command_name in ("ingest", "recall", "search", "status", "doctor", "serve"):
        assert command_name in result.output
    assert "C3/Ae" not in result.output


def test_recall_and_search_help_explain_the_distinction():
    runner = CliRunner()

    recall_help = runner.invoke(main, ["recall", "--help"])
    assert recall_help.exit_code == 0, recall_help.output
    assert "preferred high-level retrieval command" in recall_help.output

    search_help = runner.invoke(main, ["search", "--help"])
    assert search_help.exit_code == 0, search_help.output
    assert "lower-level hybrid search" in search_help.output
    assert "debugging and manual review" in search_help.output


def test_status_output_uses_novaspine_branding(monkeypatch):
    fake_spine = _FakeSpine()
    monkeypatch.setattr("c3ae.cli._get_spine", lambda data_dir=None: fake_spine)

    runner = CliRunner()
    result = runner.invoke(main, ["status"])

    assert result.exit_code == 0, result.output
    assert "NovaSpine Status" in result.output
    assert "C3/Ae" not in result.output
