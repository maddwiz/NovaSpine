from __future__ import annotations

import pytest

from c3ae.ingestion.fact_extractor import extract_facts, extract_facts_async


def test_extract_facts_pulls_entity_relation_value_and_date():
    text = (
        "Melanie painted a sunset on July 12, 2023. "
        "Caroline moved from Sweden to Denver."
    )
    facts = extract_facts(text, max_facts=8)
    assert facts
    rendered = {(f.entity, f.relation, f.value) for f in facts}
    assert ("Melanie", "painted", "sunset on July 12, 2023") in rendered or ("Melanie", "painted", "sunset on July 12") in rendered
    assert any(f.relation == "moved_from" for f in facts)
    assert ("Caroline", "location", "Denver") in rendered


def test_extract_facts_handles_first_person_current_location_updates():
    text = (
        "User moved from Denver to Santa Fe last month. "
        "Santa Fe is now their current city of residence. "
        "Since March, my home base has been Santa Fe."
    )
    facts = extract_facts(text, max_facts=8)
    rendered = {(f.entity, f.relation, f.value) for f in facts}
    assert ("User", "moved_from", "Denver") in rendered
    assert ("User", "location", "Santa Fe") in rendered


class _FakeResp:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeBackend:
    async def chat(self, *args, **kwargs):
        return _FakeResp(
            '{"facts":[{"entity":"Caroline","relation":"moved_from","value":"Sweden","date":"2019","confidence":0.9}]}'
        )


@pytest.mark.asyncio
async def test_extract_facts_async_llm_mode_parses_json_payload():
    facts = await extract_facts_async(
        "Caroline moved from Sweden in 2019.",
        mode="llm",
        chat_backend=_FakeBackend(),
        max_facts=4,
    )
    assert len(facts) == 1
    assert facts[0].entity == "Caroline"
    assert facts[0].relation == "moved_from"
    assert facts[0].value == "Sweden"
