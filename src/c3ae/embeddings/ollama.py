"""Ollama embedding provider."""

from __future__ import annotations

import httpx
import numpy as np

from c3ae.config import EmbeddingConfig
from c3ae.embeddings.base import EmbeddingProvider
from c3ae.exceptions import EmbeddingError


class OllamaEmbedder(EmbeddingProvider):
    """Local Ollama embeddings via /api/embeddings."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        self._client: httpx.AsyncClient | None = None
        base_url = config.base_url or "http://127.0.0.1:11434"
        self._base_url = base_url.rstrip("/")

    @property
    def dimensions(self) -> int:
        return self.config.dimensions

    @property
    def model_name(self) -> str:
        return self.config.model

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(base_url=self._base_url, timeout=self.config.timeout)
        return self._client

    async def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        client = await self._get_client()
        vectors: list[list[float]] = []
        for text in texts:
            try:
                resp = await client.post(
                    "/api/embeddings",
                    json={"model": self.config.model, "prompt": text},
                )
                resp.raise_for_status()
                payload = resp.json()
                vectors.append(payload["embedding"])
            except httpx.HTTPError as e:
                raise EmbeddingError(f"Ollama embedding request failed: {e}") from e
            except (KeyError, TypeError) as e:
                raise EmbeddingError(f"Unexpected Ollama embedding payload: {e}") from e

        arr = np.array(vectors, dtype=np.float32)
        if arr.ndim != 2:
            raise EmbeddingError("Invalid Ollama embedding response shape")
        if arr.shape[1] != self.dimensions:
            raise EmbeddingError(
                f"Expected embedding dims={self.dimensions}, got dims={arr.shape[1]}"
            )
        return arr

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
