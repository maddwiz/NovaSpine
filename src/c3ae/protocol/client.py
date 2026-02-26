"""Versioned local and HTTP protocol clients for NovaSpine."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin

try:
    import httpx
except ModuleNotFoundError:  # pragma: no cover - optional at import time
    httpx = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from c3ae.memory_spine.spine import MemorySpine


class SpineClientV1:
    """Stable local adapter over MemorySpine internals."""

    protocol_version = "v1"

    def __init__(self, spine: "MemorySpine") -> None:
        self._spine = spine

    async def ingest(
        self,
        text: str,
        source_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        return await self._spine.ingest(text, source_id=source_id, metadata=metadata)

    async def recall(
        self,
        query: str,
        top_k: int = 20,
        session_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        return await self._spine.recall(query, top_k=top_k, session_filter=session_filter)

    async def augment(
        self,
        query: str,
        top_k: int = 5,
        format: str = "xml",
        min_score: float = 0.005,
        roles: list[str] | None = None,
    ) -> str:
        return await self._spine.augment(
            query=query,
            top_k=top_k,
            format=format,
            min_score=min_score,
            roles=roles,
        )

    def status(self) -> dict[str, Any]:
        payload = dict(self._spine.status())
        payload["protocol_version"] = self.protocol_version
        return payload

    # Backward-compat aliases for callers still using older names.
    async def ingest_text(
        self,
        text: str,
        source_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        warnings.warn(
            "ingest_text() is deprecated on SpineClient; use ingest()",
            DeprecationWarning,
            stacklevel=2,
        )
        return await self.ingest(text, source_id=source_id, metadata=metadata)

    async def search(self, query: str, top_k: int = 20) -> list[dict[str, Any]]:
        warnings.warn(
            "search() is deprecated on SpineClient; use recall()",
            DeprecationWarning,
            stacklevel=2,
        )
        return await self.recall(query, top_k=top_k)


class SpineClientV2(SpineClientV1):
    """V2 protocol client with additive capability methods."""

    protocol_version = "v2"

    def set_decay_config(self, half_life_hours: float) -> None:
        self._spine.set_decay_config(half_life_hours)

    async def graph_query(self, entity: str, depth: int = 2) -> dict[str, Any]:
        graph = await self._spine.graph_query(entity, depth=depth)
        graph["mode"] = "graph"
        return graph


class SpineHttpClientV1:
    """Stable remote adapter bound to /api/v1 endpoints."""

    protocol_version = "v1"

    def __init__(self, base_url: str, token: str = "", timeout: float = 30.0) -> None:
        if httpx is None:
            raise ModuleNotFoundError(
                "httpx is required for SpineHttpClientV1. Install novaspine with HTTP deps."
            )
        base = base_url.strip().rstrip("/") + "/"
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._base_url = base
        self._headers = headers
        self._timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout, headers=headers)

    async def ingest(
        self,
        text: str,
        source_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        resp = await self._client.post(
            urljoin(self._base_url, "api/v1/memory/ingest"),
            json={"text": text, "source_id": source_id, "metadata": metadata or {}},
        )
        resp.raise_for_status()
        data = resp.json()
        return list(data.get("chunk_ids", []))

    async def recall(
        self,
        query: str,
        top_k: int = 20,
        session_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {"query": query, "top_k": top_k}
        if session_filter:
            payload["session_filter"] = session_filter
        resp = await self._client.post(urljoin(self._base_url, "api/v1/memory/recall"), json=payload)
        resp.raise_for_status()
        data = resp.json()
        memories = data.get("memories", [])
        out: list[dict[str, Any]] = []
        for m in memories:
            if not isinstance(m, dict):
                continue
            out.append(
                {
                    "id": str(m.get("id", "")),
                    "content": str(m.get("content", "")),
                    "score": float(m.get("score", 0.0)),
                    "source": "api:v1:recall",
                    "metadata": dict(m.get("metadata", {})),
                }
            )
        return out

    async def augment(
        self,
        query: str,
        top_k: int = 5,
        format: str = "xml",
        min_score: float = 0.005,
        roles: list[str] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "query": query,
            "top_k": top_k,
            "format": format,
            "min_score": min_score,
        }
        if roles is not None:
            payload["roles"] = roles
        resp = await self._client.post(urljoin(self._base_url, "api/v1/memory/augment"), json=payload)
        resp.raise_for_status()
        data = resp.json()
        return str(data.get("context", ""))

    def status(self) -> dict[str, Any]:
        with httpx.Client(timeout=self._timeout, headers=self._headers) as client:
            resp = client.get(urljoin(self._base_url, "api/v1/status"))
            resp.raise_for_status()
            data = dict(resp.json())
            data["protocol_version"] = self.protocol_version
            return data

    async def status_async(self) -> dict[str, Any]:
        resp = await self._client.get(urljoin(self._base_url, "api/v1/status"))
        resp.raise_for_status()
        data = dict(resp.json())
        data["protocol_version"] = self.protocol_version
        return data

    async def close(self) -> None:
        await self._client.aclose()


class SpineHttpClientV2(SpineHttpClientV1):
    protocol_version = "v2"

    async def graph_query(self, entity: str, depth: int = 2) -> dict[str, Any]:
        resp = await self._client.post(
            urljoin(self._base_url, "api/v2/graph/query"),
            json={"entity": entity, "depth": depth},
        )
        resp.raise_for_status()
        return dict(resp.json())

    async def set_decay_config(self, half_life_hours: float) -> dict[str, Any]:
        resp = await self._client.post(
            urljoin(self._base_url, "api/v2/decay/config"),
            json={"half_life_hours": half_life_hours},
        )
        resp.raise_for_status()
        return dict(resp.json())
