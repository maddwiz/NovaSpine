from __future__ import annotations

from fastapi.testclient import TestClient

from c3ae.api.routes import create_app


def test_health_is_public(monkeypatch, tmp_path):
    monkeypatch.setenv("C3AE_API_TOKEN", "secret-token")
    monkeypatch.setenv("C3AE_DATA_DIR", str(tmp_path / "data"))
    app = create_app(data_dir=str(tmp_path / "data"))
    client = TestClient(app)

    res = client.get("/api/v1/health")
    assert res.status_code == 200


def test_requires_bearer_token(monkeypatch, tmp_path):
    monkeypatch.setenv("C3AE_API_TOKEN", "secret-token")
    monkeypatch.setenv("C3AE_DATA_DIR", str(tmp_path / "data"))
    app = create_app(data_dir=str(tmp_path / "data"))
    client = TestClient(app)

    unauthorized = client.get("/api/v1/status/full")
    assert unauthorized.status_code == 401

    authorized = client.get(
        "/api/v1/status/full",
        headers={"Authorization": "Bearer secret-token"},
    )
    assert authorized.status_code == 200
