"""Embedding backend abstraction."""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
from typing import Protocol, runtime_checkable

import httpx
import numpy as np

from c3ae.config import VeniceConfig
from c3ae.embeddings.venice import VeniceEmbedder
from c3ae.utils import chunk_text_for_embedding, estimate_text_tokens


@runtime_checkable
class EmbeddingBackend(Protocol):
    async def embed(self, texts: list[str]) -> np.ndarray: ...
    async def embed_single(self, text: str) -> np.ndarray: ...
    async def close(self) -> None: ...


class OpenAIEmbedder:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-large",
        dims: int = 3072,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model
        self.dims = dims
        self.base_url = base_url
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required")
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout,
            )
        return self._client

    async def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dims), dtype=np.float32)
        client = await self._get_client()
        payload: dict[str, object] = {"model": self.model, "input": texts}
        if self.model.startswith("text-embedding-3") and self.dims > 0:
            payload["dimensions"] = int(self.dims)
        resp = await client.post(
            "/embeddings",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        vecs = [x["embedding"] for x in data.get("data", [])]
        arr = np.array(vecs, dtype=np.float32)
        return arr

    async def embed_single(self, text: str) -> np.ndarray:
        return (await self.embed([text]))[0]

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None


class OllamaEmbedder:
    def __init__(
        self,
        model: str = "nomic-embed-text",
        dims: int = 768,
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 30.0,
        keep_alive: str = "30m",
    ) -> None:
        self.model = model
        self.dims = dims
        self.base_url = base_url
        self.timeout = timeout
        self.keep_alive = keep_alive
        self.max_retries = 3
        self.safe_embed_tokens = 256
        self.safe_embed_overlap = 32
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)
        return self._client

    async def _reset_client(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    async def _post_json(self, path: str, payload: dict[str, object]) -> dict:
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            client = await self._get_client()
            try:
                resp = await client.post(path, json=payload)
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                last_exc = exc
                await self._reset_client()
                if attempt < self.max_retries:
                    await asyncio.sleep(min(1.5, 0.25 * attempt))
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"Ollama request failed for {path}")

    def _fit_dims(self, arr: np.ndarray) -> np.ndarray:
        if arr.size == 0:
            return np.zeros((0, self.dims), dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] == self.dims:
            out = arr
        elif arr.shape[1] > self.dims:
            out = arr[:, : self.dims]
        else:
            pad = np.zeros((arr.shape[0], self.dims - arr.shape[1]), dtype=np.float32)
            out = np.concatenate([arr, pad], axis=1)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return (out / norms).astype(np.float32, copy=False)

    async def _embed_v1_batch(self, texts: list[str]) -> np.ndarray:
        data = await self._post_json(
            "/v1/embeddings",
            {"model": self.model, "input": texts},
        )
        vecs = [row.get("embedding", []) for row in data.get("data", [])]
        if len(vecs) != len(texts):
            raise RuntimeError(f"ollama /v1/embeddings returned {len(vecs)} embeddings for {len(texts)} texts")
        return self._fit_dims(np.array(vecs, dtype=np.float32))

    async def _embed_single_remote(self, text: str) -> np.ndarray:
        try:
            data = await self._post_json(
                "/api/embed",
                {"model": self.model, "input": [text], "keep_alive": self.keep_alive},
            )
            embeddings = data.get("embeddings", [])
            if len(embeddings) == 1:
                return self._fit_dims(np.array(embeddings, dtype=np.float32))[0]
        except Exception:
            pass

        try:
            data = await self._post_json(
                "/api/embeddings",
                {"model": self.model, "prompt": text, "keep_alive": self.keep_alive},
            )
            if data.get("embedding"):
                return self._fit_dims(np.array([data.get("embedding", [])], dtype=np.float32))[0]
        except Exception:
            pass

        return (await self._embed_v1_batch([text]))[0]

    async def _embed_single_safe(self, text: str) -> np.ndarray:
        try:
            return await self._embed_single_remote(text)
        except Exception:
            parts = chunk_text_for_embedding(
                text,
                max_tokens=self.safe_embed_tokens,
                overlap_tokens=self.safe_embed_overlap,
            )
            if len(parts) <= 1 or (len(parts) == 1 and parts[0].strip() == (text or "").strip()):
                raise
            sub_vecs = await self.embed(parts)
            if len(sub_vecs) == 0:
                raise RuntimeError("No sub-embeddings produced for safe split fallback")
            pooled = np.mean(sub_vecs, axis=0, dtype=np.float32)
            return self._fit_dims(np.asarray([pooled], dtype=np.float32))[0]

    async def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dims), dtype=np.float32)
        try:
            data = await self._post_json(
                "/api/embed",
                {"model": self.model, "input": texts, "keep_alive": self.keep_alive},
            )
            embeddings = data.get("embeddings", [])
            if len(embeddings) == len(texts):
                return self._fit_dims(np.array(embeddings, dtype=np.float32))
        except Exception:
            pass

        try:
            return await self._embed_v1_batch(texts)
        except Exception:
            pass

        out = [await self._embed_single_safe(text) for text in texts]
        return self._fit_dims(np.asarray(out, dtype=np.float32))

    async def embed_single(self, text: str) -> np.ndarray:
        return (await self.embed([text]))[0]

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None


class HashEmbedder:
    """Deterministic local embedder using token hashing (no network/API keys)."""

    _TOKEN_RE = re.compile(r"[a-z0-9_]+")

    def __init__(self, dims: int = 384) -> None:
        self.dims = max(32, int(dims))

    def _encode(self, text: str) -> np.ndarray:
        tokens = self._TOKEN_RE.findall((text or "").lower())
        vec = np.zeros((self.dims,), dtype=np.float32)
        if not tokens:
            return vec

        features = list(tokens)
        features.extend(f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1))
        for feat in features:
            digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "little", signed=False) % self.dims
            sign = 1.0 if (digest[4] & 1) == 0 else -1.0
            vec[idx] += sign
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm
        return vec

    async def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dims), dtype=np.float32)
        return np.stack([self._encode(t) for t in texts]).astype(np.float32, copy=False)

    async def embed_single(self, text: str) -> np.ndarray:
        return self._encode(text)

    async def close(self) -> None:
        return None


class SentenceTransformerEmbedder:
    """Local semantic embedder using sentence-transformers."""

    def __init__(self, model: str = "all-MiniLM-L6-v2", dims: int = 384) -> None:
        self.model_name = model or "all-MiniLM-L6-v2"
        self.dims = max(32, int(dims))
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise RuntimeError(
                "sentence-transformers is required for embedding_provider='sbert'. "
                "Install with: pip install 'novaspine[semantic]'"
            ) from exc
        self._model = SentenceTransformer(self.model_name)
        return self._model

    def _fit_dims(self, arr: np.ndarray) -> np.ndarray:
        if arr.shape[1] == self.dims:
            out = arr
        elif arr.shape[1] > self.dims:
            out = arr[:, : self.dims]
        else:
            pad = np.zeros((arr.shape[0], self.dims - arr.shape[1]), dtype=np.float32)
            out = np.concatenate([arr, pad], axis=1)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return (out / norms).astype(np.float32, copy=False)

    def _encode_sync(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dims), dtype=np.float32)
        model = self._ensure_model()
        vectors = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return self._fit_dims(arr)

    async def embed(self, texts: list[str]) -> np.ndarray:
        return await asyncio.to_thread(self._encode_sync, texts)

    async def embed_single(self, text: str) -> np.ndarray:
        return (await self.embed([text]))[0]

    async def close(self) -> None:
        return None


def create_embedder(config: VeniceConfig | None = None) -> EmbeddingBackend:
    cfg = config or VeniceConfig()
    provider = (cfg.embedding_provider or "venice").strip().lower()
    if provider in {"venice", "default"}:
        return VeniceEmbedder(cfg)
    if provider == "openai":
        return OpenAIEmbedder(
            api_key=cfg.api_key or os.environ.get("OPENAI_API_KEY", ""),
            model=cfg.embedding_model,
            dims=cfg.embedding_dims,
        )
    if provider in {"ollama", "local"}:
        return OllamaEmbedder(
            model=cfg.embedding_model,
            dims=cfg.embedding_dims,
        )
    if provider in {"hash", "localhash"}:
        return HashEmbedder(dims=cfg.embedding_dims)
    if provider in {"sbert", "sentence-transformers", "sentence_transformers"}:
        model = cfg.embedding_model.strip() if cfg.embedding_model else "all-MiniLM-L6-v2"
        if model == "text-embedding-bge-m3":
            model = "all-MiniLM-L6-v2"
        return SentenceTransformerEmbedder(model=model, dims=cfg.embedding_dims)
    raise ValueError(f"Unsupported embedding provider: {cfg.embedding_provider}")
