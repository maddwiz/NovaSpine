from __future__ import annotations

from dataclasses import dataclass

import pytest

from c3ae.retrieval.verify import (
    build_verification_messages,
    parse_verification_payload,
    verify_answer_with_llm,
)


@dataclass
class _FakeResp:
    content: str


class _FakeBackend:
    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls = 0

    async def chat(self, *args, **kwargs):
        self.calls += 1
        return _FakeResp(content=self.payload)


def test_parse_verification_payload_uses_proposed_when_verified_answer_missing():
    result = parse_verification_payload('{"verified": true, "answer": ""}', proposed_answer="sunset")
    assert result.verified is True
    assert result.answer == "sunset"


def test_build_verification_messages_contains_question_and_proposed_answer():
    msgs = build_verification_messages(
        query="What did Melanie paint recently?",
        proposed_answer="horse",
        context="Melanie painted a horse in March and a sunset in July.",
    )
    assert len(msgs) == 2
    assert "What did Melanie paint recently?" in msgs[1].content
    assert "horse" in msgs[1].content


@pytest.mark.asyncio
async def test_verify_answer_with_llm_parses_structured_verdict():
    backend = _FakeBackend('{"verified": false, "answer": "sunset", "reason": "latest event"}')
    result = await verify_answer_with_llm(
        llm_backend=backend,
        query="What did Melanie paint recently?",
        proposed_answer="horse",
        context="Melanie painted a horse in March and a sunset in July.",
    )
    assert backend.calls == 1
    assert result.verified is False
    assert result.answer == "sunset"

