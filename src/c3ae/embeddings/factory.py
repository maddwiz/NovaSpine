"""Embedding provider factory."""

from __future__ import annotations

import os

from c3ae.config import Config
from c3ae.embeddings.base import EmbeddingProvider
from c3ae.config import EmbeddingConfig
from c3ae.embeddings.local import LocalEmbedder
from c3ae.embeddings.ollama import OllamaEmbedder
from c3ae.embeddings.openai import OpenAIEmbedder
from c3ae.embeddings.venice import VeniceEmbedder
from c3ae.exceptions import EmbeddingError


def _resolved_embedding_config(config: Config) -> EmbeddingConfig:
    """Resolve provider-specific defaults for model, dims, auth, and base URL."""
    emb = config.embedding.model_copy(deep=True)
    provider = (emb.provider or "venice").strip().lower()

    if provider == "venice":
        if not emb.model:
            emb.model = config.venice.embedding_model
        if emb.dimensions <= 0:
            emb.dimensions = config.venice.embedding_dims
        if not emb.api_key:
            emb.api_key = config.venice.api_key or os.environ.get("VENICE_API_KEY", "")
        if not emb.base_url:
            emb.base_url = config.venice.base_url
        if emb.max_batch <= 0:
            emb.max_batch = config.venice.max_batch
        if emb.timeout <= 0:
            emb.timeout = config.venice.timeout
        return emb

    if provider == "openai":
        if not emb.model:
            emb.model = "text-embedding-3-small"
        if emb.dimensions <= 0:
            emb.dimensions = 1536
        if not emb.api_key:
            emb.api_key = os.environ.get("OPENAI_API_KEY", "")
        if not emb.base_url:
            emb.base_url = "https://api.openai.com/v1"
        return emb

    if provider == "ollama":
        if not emb.model:
            emb.model = "nomic-embed-text"
        if emb.dimensions <= 0:
            emb.dimensions = 768
        if not emb.base_url:
            emb.base_url = "http://127.0.0.1:11434"
        return emb

    if provider == "local":
        if not emb.model:
            emb.model = "sentence-transformers/all-MiniLM-L6-v2"
        if emb.dimensions <= 0:
            emb.dimensions = 384
        return emb

    raise EmbeddingError(f"Unsupported embedding provider: {provider}")


def build_embedder(config: Config) -> EmbeddingProvider:
    emb = _resolved_embedding_config(config)
    config.embedding = emb
    provider = emb.provider.strip().lower()
    if provider == "venice":
        config.venice.embedding_model = emb.model
        config.venice.embedding_dims = emb.dimensions
        if emb.api_key:
            config.venice.api_key = emb.api_key
        if emb.base_url:
            config.venice.base_url = emb.base_url
        if emb.timeout > 0:
            config.venice.timeout = emb.timeout
        if emb.max_batch > 0:
            config.venice.max_batch = emb.max_batch
        return VeniceEmbedder(config.venice)
    if provider == "openai":
        return OpenAIEmbedder(emb)
    if provider == "ollama":
        return OllamaEmbedder(emb)
    if provider == "local":
        return LocalEmbedder(emb)
    raise EmbeddingError(f"Unsupported embedding provider: {provider}")
