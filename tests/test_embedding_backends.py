from __future__ import annotations

import asyncio

import numpy as np

from c3ae.embeddings.backends import OllamaEmbedder


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict]] = []

    async def post(self, path: str, json: dict) -> _FakeResponse:
        self.calls.append((path, json))
        if not self.responses:
            raise RuntimeError("no more responses")
        return self.responses.pop(0)


def test_ollama_embed_uses_batch_endpoint_when_available() -> None:
    embedder = OllamaEmbedder(model="nomic-embed-text", dims=3)
    client = _FakeClient([
        _FakeResponse({"embeddings": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]}),
    ])

    async def _get_client() -> _FakeClient:
        return client

    embedder._get_client = _get_client  # type: ignore[method-assign]

    arr = asyncio.run(embedder.embed(["hello", "world"]))

    assert arr.shape == (2, 3)
    assert np.allclose(arr[0], np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert client.calls == [
        (
            "/api/embed",
            {"model": "nomic-embed-text", "input": ["hello", "world"], "keep_alive": "30m"},
        ),
    ]


def test_ollama_embed_falls_back_to_legacy_endpoint() -> None:
    embedder = OllamaEmbedder(model="nomic-embed-text", dims=2)

    class _FallbackClient(_FakeClient):
        async def post(self, path: str, json: dict) -> _FakeResponse:
            self.calls.append((path, json))
            if path == "/api/embed":
                raise RuntimeError("batch endpoint unavailable")
            if not self.responses:
                raise RuntimeError("no more responses")
            return self.responses.pop(0)

    client = _FallbackClient([
        _FakeResponse({"embedding": [1.0, 0.0]}),
        _FakeResponse({"embedding": [0.0, 1.0]}),
    ])

    async def _get_client() -> _FallbackClient:
        return client

    embedder._get_client = _get_client  # type: ignore[method-assign]

    arr = asyncio.run(embedder.embed(["hello", "world"]))

    assert arr.shape == (2, 2)
    assert client.calls == [
        (
            "/api/embed",
            {"model": "nomic-embed-text", "input": ["hello", "world"], "keep_alive": "30m"},
        ),
        (
            "/api/embeddings",
            {"model": "nomic-embed-text", "prompt": "hello", "keep_alive": "30m"},
        ),
        (
            "/api/embeddings",
            {"model": "nomic-embed-text", "prompt": "world", "keep_alive": "30m"},
        ),
    ]
