"""Additional chat backend providers (OpenAI, Anthropic, Ollama)."""

from __future__ import annotations

import os
from typing import Any

import httpx

from c3ae.llm.venice_chat import ChatResponse, Message


class OpenAIBackend:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4.1-mini",
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 120.0,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client: httpx.AsyncClient | None = None
        self._stats = {"calls": 0, "input_tokens": 0, "output_tokens": 0}

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

    async def chat(
        self,
        messages: list[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> ChatResponse:
        client = await self._get_client()
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        resp = await client.post("/chat/completions", json=body)
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        usage = data.get("usage", {})
        self._stats["calls"] += 1
        self._stats["input_tokens"] += int(usage.get("prompt_tokens", 0))
        self._stats["output_tokens"] += int(usage.get("completion_tokens", 0))
        return ChatResponse(
            content=str(choice["message"]["content"]),
            model=str(data.get("model", self.model)),
            usage=usage,
            finish_reason=str(choice.get("finish_reason", "")),
            raw=data,
        )

    @property
    def stats(self) -> dict[str, Any]:
        return {
            **self._stats,
            "total_tokens": self._stats["input_tokens"] + self._stats["output_tokens"],
        }

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None


class AnthropicBackend:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-3-5-sonnet-latest",
        base_url: str = "https://api.anthropic.com/v1",
        timeout: float = 120.0,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client: httpx.AsyncClient | None = None
        self._stats = {"calls": 0, "input_tokens": 0, "output_tokens": 0}

    async def _get_client(self) -> httpx.AsyncClient:
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required")
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                },
                timeout=self.timeout,
            )
        return self._client

    async def chat(
        self,
        messages: list[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> ChatResponse:
        client = await self._get_client()
        system = ""
        chat_msgs: list[dict[str, str]] = []
        for m in messages:
            if m.role == "system":
                system += (m.content + "\n")
            else:
                role = "assistant" if m.role == "assistant" else "user"
                chat_msgs.append({"role": role, "content": m.content})
        body: dict[str, Any] = {
            "model": self.model,
            "system": system.strip() if system else None,
            "messages": chat_msgs,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
        }
        if json_mode:
            body["system"] = ((body.get("system") or "") + "\nRespond with strict JSON.")
        resp = await client.post("/messages", json={k: v for k, v in body.items() if v is not None})
        resp.raise_for_status()
        data = resp.json()
        content_blocks = data.get("content", [])
        text = ""
        for blk in content_blocks:
            if isinstance(blk, dict) and blk.get("type") == "text":
                text += str(blk.get("text", ""))
        usage = data.get("usage", {})
        self._stats["calls"] += 1
        self._stats["input_tokens"] += int(usage.get("input_tokens", 0))
        self._stats["output_tokens"] += int(usage.get("output_tokens", 0))
        return ChatResponse(
            content=text,
            model=self.model,
            usage=usage,
            finish_reason=str(data.get("stop_reason", "")),
            raw=data,
        )

    @property
    def stats(self) -> dict[str, Any]:
        return {
            **self._stats,
            "total_tokens": self._stats["input_tokens"] + self._stats["output_tokens"],
        }

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None


class OllamaBackend:
    def __init__(
        self,
        model: str = "llama3.1:8b-instruct",
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 120.0,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client: httpx.AsyncClient | None = None
        self._stats = {"calls": 0, "input_tokens": 0, "output_tokens": 0}

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)
        return self._client

    async def chat(
        self,
        messages: list[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> ChatResponse:
        client = await self._get_client()
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else self.temperature,
                "num_predict": max_tokens if max_tokens is not None else self.max_tokens,
            },
        }
        if json_mode:
            body["format"] = "json"
        resp = await client.post("/api/chat", json=body)
        resp.raise_for_status()
        data = resp.json()
        msg = data.get("message", {})
        self._stats["calls"] += 1
        return ChatResponse(
            content=str(msg.get("content", "")),
            model=self.model,
            usage={},
            finish_reason=str(data.get("done_reason", "")),
            raw=data,
        )

    @property
    def stats(self) -> dict[str, Any]:
        return {
            **self._stats,
            "total_tokens": self._stats["input_tokens"] + self._stats["output_tokens"],
        }

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
