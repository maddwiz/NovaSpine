"""LLM backend abstraction layer."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from c3ae.llm.venice_chat import ChatResponse, Message, VeniceChat


@runtime_checkable
class ChatBackend(Protocol):
    async def chat(
        self,
        messages: list[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> ChatResponse: ...

    @property
    def stats(self) -> dict[str, Any]: ...


class VeniceBackend(VeniceChat):
    """Default backend implementation (backward compatible)."""

