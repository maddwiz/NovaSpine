"""Local sentence-transformers embedding provider."""

from __future__ import annotations

import numpy as np

from c3ae.config import EmbeddingConfig
from c3ae.embeddings.base import EmbeddingProvider
from c3ae.exceptions import EmbeddingError


class LocalEmbedder(EmbeddingProvider):
    """Embeddings backed by sentence-transformers."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        self._model = None

    @property
    def dimensions(self) -> int:
        return self.config.dimensions

    @property
    def model_name(self) -> str:
        return self.config.model

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise EmbeddingError(
                    "sentence-transformers is required for provider=local"
                ) from e
            self._model = SentenceTransformer(self.config.model)
        return self._model

    async def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        model = self._get_model()
        vectors = model.encode(
            texts,
            batch_size=self.config.max_batch,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2:
            raise EmbeddingError("Invalid local embedding response shape")
        if arr.shape[1] != self.dimensions:
            raise EmbeddingError(
                f"Expected embedding dims={self.dimensions}, got dims={arr.shape[1]}"
            )
        return arr
