from __future__ import annotations

from c3ae.eval.temporal import (
    compute_temporal_facts,
    enrich_context_with_temporal_facts,
    extract_temporal_mentions,
)


def test_extract_temporal_mentions_infers_missing_year_from_explicit_context():
    text = "We met on March 5, 2024 and followed up on 3/12."
    mentions = extract_temporal_mentions(text)
    assert len(mentions) == 2
    assert mentions[0].parsed.year == 2024
    assert mentions[1].parsed.year == 2024


def test_compute_temporal_facts_builds_day_delta_lines():
    mentions = extract_temporal_mentions("Workshop on March 5, 2024. Meeting on March 12, 2024.")
    facts = compute_temporal_facts(mentions, max_pairs=4)
    assert facts
    assert any("7 days" in line for line in facts)


def test_enrich_context_with_temporal_facts_appends_block():
    context = "The workshop was on March 5, 2024. The team meeting happened on March 12, 2024."
    out = enrich_context_with_temporal_facts(
        query="How many days between the workshop and the meeting?",
        context=context,
        max_pairs=8,
    )
    assert "[TEMPORAL FACTS computed from context]" in out
    assert "7 days" in out

