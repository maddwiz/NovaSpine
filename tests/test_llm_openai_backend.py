from __future__ import annotations

import httpx
import pytest
import respx

from c3ae.llm.providers import OpenAIBackend
from c3ae.llm.venice_chat import Message


@pytest.mark.asyncio
@respx.mock
async def test_openai_backend_uses_chat_completions_for_gpt41() -> None:
    route_chat = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "gpt-4.1-mini-2025-04-14",
                "choices": [
                    {
                        "message": {"content": "{\"answer\":\"ok\"}"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3},
            },
        )
    )
    route_responses = respx.post("https://api.openai.com/v1/responses").mock(
        return_value=httpx.Response(500, json={"error": "should not be used"})
    )
    backend = OpenAIBackend(api_key="test-key", model="gpt-4.1-mini")
    try:
        out = await backend.chat(
            [Message(role="user", content="answer now")],
            json_mode=True,
            temperature=0.0,
            max_tokens=16,
        )
    finally:
        await backend.close()

    assert route_chat.called
    assert not route_responses.called
    assert out.content == "{\"answer\":\"ok\"}"
    assert out.model.startswith("gpt-4.1-mini")


@pytest.mark.asyncio
@respx.mock
async def test_openai_backend_uses_responses_for_gpt5() -> None:
    route_chat = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(500, json={"error": "should not be used"})
    )
    route_responses = respx.post("https://api.openai.com/v1/responses").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "gpt-5-mini-2025-08-07",
                "status": "completed",
                "output": [
                    {"type": "reasoning", "summary": []},
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "{\"answer\":\"ok\"}"}],
                    },
                ],
                "usage": {"input_tokens": 9, "output_tokens": 4},
            },
        )
    )
    backend = OpenAIBackend(api_key="test-key", model="gpt-5-mini")
    try:
        out = await backend.chat(
            [Message(role="system", content="json only"), Message(role="user", content="answer now")],
            json_mode=True,
            temperature=0.0,
            max_tokens=20,
        )
    finally:
        await backend.close()

    assert route_responses.called
    assert not route_chat.called
    body = route_responses.calls[0].request.read().decode("utf-8")
    assert '"text":{"format":{"type":"json_object"}}' in body
    assert out.content == "{\"answer\":\"ok\"}"
    assert out.model.startswith("gpt-5-mini")
