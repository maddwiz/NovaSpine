from __future__ import annotations

from typing import Any
import warnings

from fastapi.testclient import TestClient

from c3ae.api import routes
from c3ae.config import Config


def test_create_app_warns_and_fails_closed_without_auth(tmp_path, monkeypatch):
    monkeypatch.delenv("C3AE_API_TOKEN", raising=False)
    monkeypatch.delenv("C3AE_AUTH_DISABLED", raising=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        app = routes.create_app(data_dir=str(tmp_path / "data"))

    messages = [str(item.message) for item in caught]
    assert any("Non-health and non-docs API routes will return 503" in message for message in messages)

    with TestClient(app) as client:
        assert client.get("/api/v1/health").status_code == 200
        assert client.get("/docs").status_code == 200
        assert client.get("/redoc").status_code == 200
        assert client.get("/openapi.json").status_code == 200

        blocked = client.get("/api/v1/status")
        assert blocked.status_code == 503
        assert blocked.json()["detail"] == "API authentication not configured. Set C3AE_API_TOKEN."


def test_create_app_allows_explicit_auth_disable(tmp_path, monkeypatch):
    monkeypatch.delenv("C3AE_API_TOKEN", raising=False)
    monkeypatch.setenv("C3AE_AUTH_DISABLED", "1")

    app = routes.create_app(data_dir=str(tmp_path / "data"))

    with TestClient(app) as client:
        response = client.get("/api/v1/status")
        assert response.status_code == 200


def test_create_app_enforces_bearer_token(tmp_path, monkeypatch):
    monkeypatch.setenv("C3AE_API_TOKEN", "secret-token")
    monkeypatch.delenv("C3AE_AUTH_DISABLED", raising=False)

    app = routes.create_app(data_dir=str(tmp_path / "data"))

    with TestClient(app) as client:
        unauthorized = client.get("/api/v1/status")
        assert unauthorized.status_code == 401

        authorized = client.get("/api/v1/status", headers={"Authorization": "Bearer secret-token"})
        assert authorized.status_code == 200


def test_embedding_env_aliases_are_supported(monkeypatch):
    monkeypatch.setenv("C3AE_EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("C3AE_EMBEDDING_MODEL", "text-embedding-3-large")
    monkeypatch.setenv("C3AE_EMBEDDING_DIMENSIONS", "3072")
    monkeypatch.setenv("C3AE_EMBEDDING_API_KEY", "embed-key")
    monkeypatch.setenv("C3AE_EMBED_PROVIDER", "venice")
    monkeypatch.setenv("C3AE_EMBED_MODEL", "text-embedding-bge-m3")
    monkeypatch.setenv("C3AE_EMBED_DIMS", "1024")
    monkeypatch.setenv("VENICE_API_KEY", "legacy-key")

    cfg = Config()

    assert cfg.venice.embedding_provider == "openai"
    assert cfg.venice.embedding_model == "text-embedding-3-large"
    assert cfg.venice.embedding_dims == 3072
    assert cfg.venice.api_key == "embed-key"


def test_default_data_dir_uses_user_app_data_location(monkeypatch, tmp_path):
    monkeypatch.delenv("C3AE_DATA_DIR", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg-data"))

    cfg = Config()

    assert cfg.data_dir == (tmp_path / "xdg-data" / "novaspine")


def test_memory_ingest_uses_async_retry_sleep(tmp_path, monkeypatch):
    monkeypatch.setenv("C3AE_AUTH_DISABLED", "1")

    app = routes.create_app(data_dir=str(tmp_path / "data"))

    class FakeSpine:
        def __init__(self) -> None:
            self.calls = 0

        async def ingest_text(self, text: str, source_id: str = "", metadata: dict[str, Any] | None = None):
            self.calls += 1
            if self.calls < 3:
                raise RuntimeError("database is locked")
            return ["chunk-1"]

    fake_spine = FakeSpine()
    routes._spine = fake_spine
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(routes.asyncio, "sleep", fake_sleep)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/memory/ingest",
            json={"text": "hello", "source_id": "unit:test", "metadata": {}},
        )

    assert response.status_code == 200
    assert response.json() == {"chunk_ids": ["chunk-1"], "count": 1}
    assert fake_spine.calls == 3
    assert sleeps == [1.0, 2.0]


def test_api_branding_and_exempt_docs_are_consistent(tmp_path, monkeypatch):
    monkeypatch.setenv("C3AE_AUTH_DISABLED", "1")

    app = routes.create_app(data_dir=str(tmp_path / "data"))

    with TestClient(app) as client:
        health = client.get("/api/v1/health")
        status = client.get("/api/v1/status")
        full_status = client.get("/api/v1/status/full")
        docs = client.get("/docs")
        redoc = client.get("/redoc")
        openapi = client.get("/openapi.json")

    assert health.status_code == 200
    assert health.json()["service"] == "novaspine"

    assert status.status_code == 200
    assert status.json()["service"] == "novaspine"

    assert full_status.status_code == 200
    assert full_status.json()["service"] == "novaspine"

    assert docs.status_code == 200
    assert redoc.status_code == 200
    assert openapi.status_code == 200
    assert openapi.json()["info"]["title"] == "NovaSpine API"
