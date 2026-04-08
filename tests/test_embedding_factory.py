from __future__ import annotations

from c3ae.config import Config
from c3ae.embeddings.factory import build_embedder
from c3ae.embeddings.local import LocalEmbedder
from c3ae.embeddings.ollama import OllamaEmbedder
from c3ae.embeddings.openai import OpenAIEmbedder
from c3ae.embeddings.venice import VeniceEmbedder


def test_factory_defaults_to_venice() -> None:
    cfg = Config()
    cfg.embedding.provider = "venice"
    cfg.embedding.model = ""
    cfg.embedding.dimensions = 0
    embedder = build_embedder(cfg)
    assert isinstance(embedder, VeniceEmbedder)
    assert cfg.embedding.dimensions == cfg.venice.embedding_dims


def test_factory_openai_defaults() -> None:
    cfg = Config()
    cfg.embedding.provider = "openai"
    cfg.embedding.model = ""
    cfg.embedding.dimensions = 0
    embedder = build_embedder(cfg)
    assert isinstance(embedder, OpenAIEmbedder)
    assert cfg.embedding.model == "text-embedding-3-small"
    assert cfg.embedding.dimensions == 1536


def test_factory_ollama_defaults() -> None:
    cfg = Config()
    cfg.embedding.provider = "ollama"
    cfg.embedding.model = ""
    cfg.embedding.dimensions = 0
    embedder = build_embedder(cfg)
    assert isinstance(embedder, OllamaEmbedder)
    assert cfg.embedding.dimensions == 768


def test_factory_local_defaults() -> None:
    cfg = Config()
    cfg.embedding.provider = "local"
    cfg.embedding.model = ""
    cfg.embedding.dimensions = 0
    embedder = build_embedder(cfg)
    assert isinstance(embedder, LocalEmbedder)
    assert cfg.embedding.dimensions == 384
