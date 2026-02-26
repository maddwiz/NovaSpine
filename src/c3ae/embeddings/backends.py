"""Embedding backend abstraction."""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

import httpx
import numpy as np

from c3ae.config import VeniceConfig
from c3ae.embeddings.venice import VeniceEmbedder


@runtime_checkable
class EmbeddingBackend(Protocol):
    async def embed(self, texts: list[str]) -> np.ndarray: ...
    async def embed_single(self, text: str) -> np.ndarray: ...
    async def close(self) -> None: ...


class OpenAIEmbedder:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-large",
        dims: int = 3072,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model
        self.dims = dims
        self.base_url = base_url
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required")
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout,
            )
        return self._client

    async def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dims), dtype=np.float32)
        client = await self._get_client()
        resp = await client.post(
            "/embeddings",
            json={"model": self.model, "input": texts},
        )
        resp.raise_for_status()
        data = resp.json()
        vecs = [x["embedding"] for x in data.get("data", [])]
        arr = np.array(vecs, dtype=np.float32)
        return arr

    async def embed_single(self, text: str) -> np.ndarray:
        return (await self.embed([text]))[0]

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None


class OllamaEmbedder:
    def __init__(
        self,
        model: str = "nomic-embed-text",
        dims: int = 768,
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 30.0,
    ) -> None:
        self.model = model
        self.dims = dims
        self.base_url = base_url
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)
        return self._client

    async def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dims), dtype=np.float32)
        client = await self._get_client()
        out: list[list[float]] = []
        for text in texts:
            resp = await client.post("/api/embeddings", json={"model": self.model, "prompt": text})
            resp.raise_for_status()
            data = resp.json()
            out.append(data.get("embedding", []))
        return np.array(out, dtype=np.float32)

    async def embed_single(self, text: str) -> np.ndarray:
        return (await self.embed([text]))[0]

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None


def create_embedder(config: VeniceConfig | None = None) -> EmbeddingBackend:
    cfg = config or VeniceConfig()
    provider = (cfg.embedding_provider or "venice").strip().lower()
    if provider in {"venice", "default"}:
        return VeniceEmbedder(cfg)
    if provider == "openai":
        return OpenAIEmbedder(dims=cfg.embedding_dims)
    if provider in {"ollama", "local"}:
        return OllamaEmbedder(dims=cfg.embedding_dims)
    raise ValueError(f"Unsupported embedding provider: {cfg.embedding_provider}")
