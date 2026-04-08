"""OpenAI-compatible embedding provider."""

from __future__ import annotations

import httpx
import numpy as np

from c3ae.config import EmbeddingConfig
from c3ae.embeddings.base import EmbeddingProvider
from c3ae.exceptions import EmbeddingError


class OpenAIEmbedder(EmbeddingProvider):
    """OpenAI API embeddings via HTTP."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        self._client: httpx.AsyncClient | None = None
        base_url = config.base_url or "https://api.openai.com/v1"
        self._base_url = base_url.rstrip("/")

    @property
    def dimensions(self) -> int:
        return self.config.dimensions

    @property
    def model_name(self) -> str:
        return self.config.model

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            headers = {"Authorization": f"Bearer {self.config.api_key}"}
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers=headers,
                timeout=self.config.timeout,
            )
        return self._client

    async def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        client = await self._get_client()
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.config.max_batch):
            batch = texts[i : i + self.config.max_batch]
            try:
                resp = await client.post(
                    "/embeddings",
                    json={"model": self.config.model, "input": batch},
                )
                resp.raise_for_status()
                payload = resp.json()
                data = payload.get("data", [])
                vectors.extend(item["embedding"] for item in data)
            except httpx.HTTPError as e:
                raise EmbeddingError(f"OpenAI embedding request failed: {e}") from e
            except (KeyError, TypeError) as e:
                raise EmbeddingError(f"Unexpected OpenAI embedding payload: {e}") from e

        arr = np.array(vectors, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] != len(texts):
            raise EmbeddingError("OpenAI embedding response count mismatch")
        if arr.shape[1] != self.dimensions:
            raise EmbeddingError(
                f"Expected embedding dims={self.dimensions}, got dims={arr.shape[1]}"
            )
        return arr

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
