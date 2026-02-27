"""Embedding providers and abstractions."""

from c3ae.embeddings.backends import (
    EmbeddingBackend,
    HashEmbedder,
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
    "HashEmbedder",
    "create_embedder",
]
