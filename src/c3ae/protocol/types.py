"""Versioned Spine protocol contracts for stable integrations."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SpineProtocolV1(Protocol):
    """Stable interface for direct or remote Spine integrations."""

    protocol_version: str

    async def ingest(
        self,
        text: str,
        source_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> list[str]: ...

    async def recall(
        self,
        query: str,
        top_k: int = 20,
        session_filter: str | None = None,
    ) -> list[dict[str, Any]]: ...

    async def augment(
        self,
        query: str,
        top_k: int = 5,
        format: str = "xml",
        min_score: float = 0.005,
        roles: list[str] | None = None,
    ) -> str: ...

    def status(self) -> dict[str, Any]: ...


@runtime_checkable
class SpineProtocolV2(SpineProtocolV1, Protocol):
    """V2 extends V1 without breaking its method contracts."""

    async def graph_query(self, entity: str, depth: int = 2) -> dict[str, Any]: ...

    def set_decay_config(self, half_life_hours: float) -> None: ...
