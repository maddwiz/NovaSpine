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


def test_extract_facts_handles_first_person_historical_location_statements():
    facts = extract_facts(
        "Earlier in the year I was still based in Denver. Since March, my home base has been Santa Fe.",
        max_facts=8,
    )
    rendered = {(f.entity, f.relation, f.value) for f in facts}
    assert ("User", "location", "Denver") in rendered
    assert ("User", "location", "Santa Fe") in rendered


def test_extract_facts_handles_first_person_preferences_and_changes():
    text = (
        "My espresso order now is a flat white made with oat milk. "
        "Before spring, I was carrying an olive Evergoods CPL24 most workdays. "
        "My notebook at the start of the year was a pocket Moleskine. "
        "By April I had switched from the pocket Moleskine to a Field Notes three-pack. "
        "Movie nights still mean kettle corn, and I usually pick up sparkling water too. "
        "The charger that keeps ending up in my airport pouch is the Anker 737. "
        "On planes I nearly always claim the aisle."
    )
    facts = extract_facts(text, max_facts=20)
    rendered = {(f.entity, f.relation, f.value) for f in facts}
    assert ("User", "coffee_order", "flat white made with oat milk") in rendered
    assert ("User", "bag", "olive Evergoods CPL24") in rendered
    assert ("User", "notebook", "pocket Moleskine") in rendered
    assert ("User", "notebook", "Field Notes three-pack") in rendered
    assert ("User", "movie_snack", "kettle corn") in rendered
    assert ("User", "movie_drink", "sparkling water") in rendered
    assert ("User", "charger", "Anker 737") in rendered
    assert ("User", "flight_seat", "aisle") in rendered


def test_extract_facts_handles_semicolon_change_log_sequences():
    facts = extract_facts(
        "Historical changes: cappuccino with whole milk became flat white with oat milk; "
        "olive Evergoods CPL24 became Peak Design 20L; "
        "pocket Moleskine became Field Notes; "
        "Denver became Santa Fe.",
        max_facts=20,
    )
    rendered = {(f.entity, f.relation, f.value) for f in facts}
    assert ("User", "coffee_order", "cappuccino with whole milk") in rendered
    assert ("User", "coffee_order", "flat white with oat milk") in rendered
    assert ("User", "bag", "olive Evergoods CPL24") in rendered
    assert ("User", "bag", "Peak Design 20L") in rendered
    assert ("User", "notebook", "pocket Moleskine") in rendered
    assert ("User", "notebook", "Field Notes") in rendered


def test_extract_facts_handles_profile_and_airport_summary_lines():
    facts = extract_facts(
        "# Travel Profile\n"
        "Current travel profile: flat white with oat milk, aisle seat, Peak Design 20L, Anker 737, Field Notes, Santa Fe base.\n"
        "My boarding routine is simple: aisle seat, Peak Design 20L, Anker 737, and the Field Notes notebook.\n"
        "Field Notes and the Anker 737 keep ending up in the airport kit.\n"
        "Airport kit stays consistent: Peak Design 20L, Anker 737, Field Notes, aisle seat.\n",
        max_facts=24,
    )
    rendered = {(f.entity, f.relation, f.value) for f in facts}
    assert ("User", "coffee_order", "flat white with oat milk") in rendered
    assert ("User", "flight_seat", "aisle seat") in rendered or ("User", "flight_seat", "aisle") in rendered
    assert ("User", "bag", "Peak Design 20L") in rendered
    assert ("User", "charger", "Anker 737") in rendered
    assert ("User", "notebook", "Field Notes") in rendered
    assert ("User", "location", "Santa Fe") in rendered


def test_extract_facts_does_not_turn_team_notebooks_into_user_facts():
    facts = extract_facts(
        "# Team Session: Dev\n"
        "Dev moved from Phoenix to Madison, switched from the olive Evergoods CPL24 to the navy Aer City Pack, "
        "orders black coffee, prefers the rear aisle seat, keeps a Midori notebook, carries a Anker Nano, "
        "and usually buys gummy worms at the theater.\n",
        max_facts=16,
    )
    rendered = {(f.entity, f.relation, f.value) for f in facts}
    assert ("User", "notebook", "Midori") not in rendered
    assert ("Dev", "moved_from", "Phoenix") in rendered
    assert ("Dev", "location", "Madison") in rendered


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
