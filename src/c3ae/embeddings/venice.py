"""Venice AI embedding provider."""

from __future__ import annotations

import numpy as np
import httpx

from c3ae.config import VeniceConfig
from c3ae.embeddings.base import EmbeddingProvider
from c3ae.exceptions import EmbeddingError


class VeniceEmbedder(EmbeddingProvider):
    """Async embedding client using Venice API."""

    def __init__(self, config: VeniceConfig | None = None) -> None:
        self.config = config or VeniceConfig()
        self._client: httpx.AsyncClient | None = None

    @property
    def dimensions(self) -> int:
        return self.config.embedding_dims

    @property
    def model_name(self) -> str:
        return self.config.embedding_model

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                timeout=self.config.timeout,
            )
        return self._client

    async def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts. Returns (n, dims) array."""
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        client = await self._get_client()
        all_embeddings = []
        # Process in batches
        for i in range(0, len(texts), self.config.max_batch):
            batch = texts[i : i + self.config.max_batch]
            try:
                resp = await client.post(
                    "/embeddings",
                    json={
                        "model": self.config.embedding_model,
                        "input": batch,
                        "encoding_format": "float",
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                batch_embeddings = [item["embedding"] for item in data["data"]]
                all_embeddings.extend(batch_embeddings)
            except httpx.HTTPError as e:
                raise EmbeddingError(f"Venice API error: {e}") from e
            except (KeyError, IndexError) as e:
                raise EmbeddingError(f"Unexpected Venice response: {e}") from e
        result = np.array(all_embeddings, dtype=np.float32)
        if result.shape[1] != self.dimensions:
            raise EmbeddingError(
                f"Expected {self.dimensions} dims, got {result.shape[1]}"
            )
        return result

    async def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text. Returns (dims,) array."""
        result = await self.embed([text])
        return result[0]

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
