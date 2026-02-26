"""Embedding providers and abstractions."""

from c3ae.embeddings.backends import (
    EmbeddingBackend,
    OllamaEmbedder,
    OpenAIEmbedder,
    create_embedder,
)
from c3ae.embeddings.venice import VeniceEmbedder

__all__ = [
    "EmbeddingBackend",
    "VeniceEmbedder",
    "OpenAIEmbedder",
    "OllamaEmbedder",
    "create_embedder",
]
