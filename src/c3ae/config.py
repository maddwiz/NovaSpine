"""C3/Ae configuration."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field


def _default_data_dir() -> Path:
    return Path(os.environ.get("C3AE_DATA_DIR", Path(__file__).resolve().parents[2] / "data"))


class VeniceConfig(BaseModel):
    api_key: str = Field(default_factory=lambda: os.environ.get("VENICE_API_KEY", ""))
    base_url: str = "https://api.venice.ai/api/v1"
    embedding_model: str = "text-embedding-bge-m3"
    embedding_dims: int = 1024
    chat_model: str = "qwen3-235b-a22b-instruct-2507"
    timeout: float = 30.0
    chat_timeout: float = 120.0
    max_batch: int = 64
    temperature: float = 0.3
    max_tokens: int = 4096


class EmbeddingConfig(BaseModel):
    provider: str = Field(default_factory=lambda: os.environ.get("C3AE_EMBEDDING_PROVIDER", "venice"))
    model: str = Field(default_factory=lambda: os.environ.get("C3AE_EMBEDDING_MODEL", ""))
    dimensions: int = Field(default_factory=lambda: int(os.environ.get("C3AE_EMBEDDING_DIMENSIONS", "0")))
    api_key: str = Field(default_factory=lambda: os.environ.get("C3AE_EMBEDDING_API_KEY", ""))
    base_url: str = Field(default_factory=lambda: os.environ.get("C3AE_EMBEDDING_BASE_URL", ""))
    timeout: float = Field(default_factory=lambda: float(os.environ.get("C3AE_EMBEDDING_TIMEOUT", "30")))
    max_batch: int = Field(default_factory=lambda: int(os.environ.get("C3AE_EMBEDDING_MAX_BATCH", "64")))


class RetrievalConfig(BaseModel):
    vector_weight: float = 0.7
    keyword_weight: float = 0.3
    default_top_k: int = 20
    faiss_ivf_threshold: int = 50_000
    faiss_nprobe: int = 16


class GovernanceConfig(BaseModel):
    require_evidence: bool = True
    max_entry_bytes: int = 100_000
    contradiction_check: bool = True


class APIConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8420
    bearer_token: str = Field(default_factory=lambda: os.environ.get("C3AE_API_TOKEN", ""))


class Config(BaseModel):
    data_dir: Path = Field(default_factory=_default_data_dir)
    venice: VeniceConfig = Field(default_factory=VeniceConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    governance: GovernanceConfig = Field(default_factory=GovernanceConfig)
    api: APIConfig = Field(default_factory=APIConfig)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "db" / "c3ae.db"

    @property
    def faiss_dir(self) -> Path:
        return self.data_dir / "faiss"

    @property
    def vault_dir(self) -> Path:
        return self.data_dir / "vault"

    def ensure_dirs(self) -> None:
        for d in [
            self.data_dir,
            self.db_path.parent,
            self.faiss_dir,
            self.vault_dir / "documents",
            self.vault_dir / "evidence",
            self.vault_dir / "raw_logs",
            self.vault_dir / "code_snapshots",
        ]:
            d.mkdir(parents=True, exist_ok=True)
