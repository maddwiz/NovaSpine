"""Embedding provider interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class EmbeddingProvider(ABC):
    """Provider-agnostic embedding contract."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Embedding vector dimensionality."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Provider model name for cache scoping."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> np.ndarray:
        """Embed batch of texts, returning (n, dims) float32 array."""

    async def embed_single(self, text: str) -> np.ndarray:
        """Embed one text."""
        vecs = await self.embed([text])
        return vecs[0]

    async def close(self) -> None:
        """Close underlying resources."""
        return None
