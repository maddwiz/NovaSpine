from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from c3ae.api import routes
from c3ae.types import Chunk


def _seed_chunk(spine, chunk_id: str, content: str, *, source_id: str = "unit:test") -> str:
    chunk = Chunk(
        id=chunk_id,
        source_id=source_id,
        content=content,
        metadata={"source_file": "unit.txt", "role": "user", "session_id": "session-1"},
        created_at=datetime.now(timezone.utc),
    )
    spine.sqlite.insert_chunk(chunk)
    return chunk.id


def test_fact_and_wiki_endpoints_compile_and_resolve_conflicts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("C3AE_AUTH_DISABLED", "1")
    app = routes.create_app(data_dir=str(tmp_path / "data"))
    spine = routes._spine
    assert spine is not None

    chunk_id = _seed_chunk(
        spine,
        "chunk-desmond",
        "Desmond Pottle is also known as Wiz in the operator context.",
    )
    winner_fact_id = spine.sqlite.insert_structured_fact(
        source_chunk_id=chunk_id,
        entity="Desmond Pottle",
        relation="also_known_as",
        value="Wiz",
        confidence=0.96,
        metadata={"fact_status": "current"},
    )
    loser_fact_id = spine.sqlite.insert_structured_fact(
        source_chunk_id=chunk_id,
        entity="Desmond Pottle",
        relation="also_known_as",
        value="Wizard",
        confidence=0.72,
        metadata={"fact_status": "current"},
    )

    with TestClient(app) as client:
        current = client.get("/api/v1/facts/current", params={"entity": "Desmond Pottle"})
        assert current.status_code == 200
        current_payload = current.json()
        assert current_payload["count"] == 2

        truth = client.get("/api/v1/facts/truth", params={"entity": "Desmond Pottle"})
        assert truth.status_code == 200
        truth_payload = truth.json()
        assert truth_payload["count"] == 1
        assert len(truth_payload["fact_groups"][0]["current_facts"]) == 2

        conflicts = client.get("/api/v1/facts/conflicts")
        assert conflicts.status_code == 200
        conflicts_payload = conflicts.json()
        assert conflicts_payload["count"] == 1
        assert conflicts_payload["conflicts"][0]["entity"] == "Desmond Pottle"

        status = client.get("/api/v1/wiki/status")
        assert status.status_code == 200
        status_payload = status.json()
        assert status_payload["service"] == "novaspine"
        assert status_payload["entity_pages"] >= 1
        assert Path(status_payload["vault_root"]).exists()

        search = client.post("/api/v1/wiki/search", json={"query": "Desmond Wiz", "limit": 5})
        assert search.status_code == 200
        search_payload = search.json()
        assert search_payload["count"] >= 1
        assert search_payload["results"][0]["kind"] == "page"
        assert search_payload["results"][0]["title"] == "Desmond Pottle"

        page = client.get("/api/v1/wiki/get", params={"entity": "Desmond Pottle"})
        assert page.status_code == 200
        page_payload = page.json()
        assert page_payload["title"] == "Desmond Pottle"
        assert len(page_payload["current_claims"]) == 2

        applied = client.post(
            "/api/v1/wiki/apply",
            json={
                "entity": "Desmond Pottle",
                "summary": "Desmond Pottle is the operator also known as Wiz.",
                "open_questions": ["Should Wizard remain as a historical alias?"],
                "tags": ["operator", "identity"],
            },
        )
        assert applied.status_code == 200
        applied_payload = applied.json()
        assert applied_payload["summary"] == "Desmond Pottle is the operator also known as Wiz."
        assert applied_payload["manual"]["tags"] == ["operator", "identity"]

        lint = client.get("/api/v1/wiki/lint")
        assert lint.status_code == 200
        lint_payload = lint.json()
        assert lint_payload["counts"]["conflicts"] >= 1

        resolved = client.post(
            "/api/v1/facts/resolve",
            json={
                "winner_fact_id": winner_fact_id,
                "loser_fact_ids": [loser_fact_id],
                "reason": "User confirmed the preferred alias.",
                "user_confirmation": "Keep Wiz as the current alias.",
            },
        )
        assert resolved.status_code == 200
        resolved_payload = resolved.json()
        assert resolved_payload["ok"] is True
        assert resolved_payload["winner_fact"]["id"] == winner_fact_id
        assert resolved_payload["superseded_facts"][0]["id"] == loser_fact_id

        current_after = client.get("/api/v1/facts/current", params={"entity": "Desmond Pottle"})
        assert current_after.status_code == 200
        current_after_payload = current_after.json()
        assert current_after_payload["count"] == 1
        assert current_after_payload["facts"][0]["id"] == winner_fact_id

        conflicts_after = client.get("/api/v1/facts/conflicts")
        assert conflicts_after.status_code == 200
        assert conflicts_after.json()["count"] == 0
