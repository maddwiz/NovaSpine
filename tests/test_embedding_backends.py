from __future__ import annotations

import asyncio

import httpx
import numpy as np

from c3ae.embeddings.backends import OllamaEmbedder, OpenAIEmbedder, _deterministic_embedding


class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("POST", "https://example.test/embeddings")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("bad request", request=request, response=response)
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


def test_ollama_embed_falls_back_to_v1_then_legacy_endpoints() -> None:
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
        _FakeResponse({"data": []}),
        _FakeResponse({"embedding": [1.0, 0.0]}),
        _FakeResponse({"embedding": [0.0, 1.0]}),
    ])

    async def _get_client() -> _FallbackClient:
        return client

    embedder._get_client = _get_client  # type: ignore[method-assign]

    arr = asyncio.run(embedder.embed(["hello", "world"]))

    assert arr.shape == (2, 2)
    paths = [path for path, _ in client.calls]
    assert paths.count("/api/embed") == embedder.max_retries * 3
    assert paths.count("/v1/embeddings") == 1
    assert paths.count("/api/embeddings") == 2
    assert client.calls[3] == (
        "/v1/embeddings",
        {"model": "nomic-embed-text", "input": ["hello", "world"]},
    )
    assert client.calls[7] == (
        "/api/embeddings",
        {"model": "nomic-embed-text", "prompt": "hello", "keep_alive": "30m"},
    )
    assert client.calls[-1] == (
        "/api/embeddings",
        {"model": "nomic-embed-text", "prompt": "world", "keep_alive": "30m"},
    )


def test_openai_embed_retries_items_after_batch_400_and_hashes_only_failed_texts() -> None:
    embedder = OpenAIEmbedder(api_key="test-key", model="text-embedding-3-small", dims=3)
    client = _FakeClient([
        _FakeResponse({"error": {"message": "mixed batch failed"}}, status_code=400),
        _FakeResponse({"data": [{"embedding": [1.0, 0.0, 0.0]}]}),
        _FakeResponse({"error": {"message": "bad text"}}, status_code=400),
        _FakeResponse({"error": {"message": "bad sanitized text"}}, status_code=400),
    ])

    async def _get_client() -> _FakeClient:
        return client

    embedder._get_client = _get_client  # type: ignore[method-assign]

    arr = asyncio.run(embedder.embed(["good", "bad\x00text"]))

    assert arr.shape == (2, 3)
    assert np.allclose(arr[0], np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert np.allclose(arr[1], _deterministic_embedding("bad text", 3))
    assert client.calls == [
        (
            "/embeddings",
            {"model": "text-embedding-3-small", "input": ["good", "bad\x00text"], "dimensions": 3},
        ),
        (
            "/embeddings",
            {"model": "text-embedding-3-small", "input": ["good"], "dimensions": 3},
        ),
        (
            "/embeddings",
            {"model": "text-embedding-3-small", "input": ["bad\x00text"], "dimensions": 3},
        ),
        (
            "/embeddings",
            {"model": "text-embedding-3-small", "input": ["bad text"], "dimensions": 3},
        ),
    ]


def test_ollama_embed_hashes_only_failed_individual_text() -> None:
    embedder = OllamaEmbedder(model="nomic-embed-text", dims=2)
    embedder.max_retries = 1

    class _PartialFailureClient(_FakeClient):
        async def post(self, path: str, json: dict) -> _FakeResponse:
            self.calls.append((path, json))
            if path == "/api/embed" and json["input"] == ["good", "bad\x00text"]:
                return _FakeResponse({"error": {"message": "mixed batch failed"}}, status_code=400)
            if path == "/v1/embeddings" and json["input"] == ["good", "bad\x00text"]:
                return _FakeResponse({"error": {"message": "mixed batch failed"}}, status_code=400)
            if path == "/api/embed" and json["input"] == ["good"]:
                return _FakeResponse({"embeddings": [[0.0, 1.0]]})
            if path == "/api/embed" and json["input"] in (["bad\x00text"], ["bad text"]):
                return _FakeResponse({"error": {"message": "bad text"}}, status_code=400)
            return _FakeResponse({"error": {"message": "unsupported"}}, status_code=400)

    client = _PartialFailureClient([])

    async def _get_client() -> _PartialFailureClient:
        return client

    embedder._get_client = _get_client  # type: ignore[method-assign]

    arr = asyncio.run(embedder.embed(["good", "bad\x00text"]))

    assert arr.shape == (2, 2)
    assert np.allclose(arr[0], np.array([0.0, 1.0], dtype=np.float32))
    assert np.allclose(arr[1], _deterministic_embedding("bad text", 2))
