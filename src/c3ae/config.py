"""C3/Ae configuration."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field


def _default_data_dir() -> Path:
    return Path(os.environ.get("C3AE_DATA_DIR", Path(__file__).resolve().parents[2] / "data"))


class VeniceConfig(BaseModel):
    embedding_provider: str = Field(default_factory=lambda: os.environ.get("C3AE_EMBED_PROVIDER", "venice"))
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


class RetrievalConfig(BaseModel):
    vector_weight: float = 0.7
    keyword_weight: float = 0.3
    default_top_k: int = 20
    faiss_ivf_threshold: int = 50_000
    faiss_nprobe: int = 16
    adaptive_weights: bool = True
    enable_decay: bool = True
    decay_half_life_hours: float = 24.0 * 14.0
    decay_min_factor: float = 0.2
    access_boost_per_hit: float = 0.05
    access_boost_cap: float = 2.0
    evidence_importance_boost: float = 1.25
    graph_weight: float = 0.2


class COSConfig(BaseModel):
    max_key_facts: int = 40
    max_open_questions: int = 20


class GraphConfig(BaseModel):
    enabled: bool = True
    extraction_mode: str = "heuristic"  # heuristic | llm
    max_entities_per_chunk: int = 16
    max_relations_per_chunk: int = 8


class ConsolidationConfig(BaseModel):
    enabled: bool = True
    min_cluster_size: int = 2
    max_clusters_per_run: int = 100
    lookback_hours: int = 24 * 30


class MemoryManagerConfig(BaseModel):
    enabled: bool = True
    use_llm_policy: bool = False
    similarity_noop_threshold: float = 0.92
    similarity_update_threshold: float = 0.80
    llm_provider: str = Field(
        default_factory=lambda: os.environ.get("C3AE_MEMORY_MANAGER_PROVIDER", "venice")
    )
    llm_model: str = Field(
        default_factory=lambda: os.environ.get("C3AE_MEMORY_MANAGER_MODEL", "")
    )
    llm_temperature: float = 0.0
    llm_max_tokens: int = 256


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
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    cos: COSConfig = Field(default_factory=COSConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    consolidation: ConsolidationConfig = Field(default_factory=ConsolidationConfig)
    memory_manager: MemoryManagerConfig = Field(default_factory=MemoryManagerConfig)
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
