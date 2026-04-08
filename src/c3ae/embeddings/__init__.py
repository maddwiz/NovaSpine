"""Embedding providers."""

from c3ae.embeddings.base import EmbeddingProvider
from c3ae.embeddings.factory import build_embedder
from c3ae.embeddings.local import LocalEmbedder
from c3ae.embeddings.ollama import OllamaEmbedder
from c3ae.embeddings.openai import OpenAIEmbedder
from c3ae.embeddings.venice import VeniceEmbedder

__all__ = [
    "EmbeddingProvider",
    "build_embedder",
    "LocalEmbedder",
    "OllamaEmbedder",
    "OpenAIEmbedder",
    "VeniceEmbedder",
]
