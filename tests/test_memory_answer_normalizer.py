from c3ae.qa.normalizer import infer_answer_type, normalize_answer, normalize_for_match


def test_normalizes_relative_year_with_reference_date():
    out = normalize_answer(
        "last year",
        "year",
        {"reference_date": "2023-06-09T12:00:00Z"},
    )

    assert out.answer == "2022"
    assert "resolved_last_year" in out.steps


def test_normalizes_relative_day_with_session_date():
    out = normalize_answer(
        "pottery class yesterday",
        "date",
        {"session_date": "2023-07-03T12:00:00Z"},
    )

    assert out.answer == "2 July 2023"
    assert "resolved_yesterday" in out.steps


def test_normalizes_counts_and_match_text():
    assert normalize_answer("two projects", "count").answer == "2"
    assert normalize_for_match("The Two Projects!") == "2 projects"


def test_infers_common_memory_answer_types():
    assert infer_answer_type("How many projects have I led?") == "count"
    assert infer_answer_type("What is Caroline's relationship status?") == "relationship_status"
    assert infer_answer_type("Where do I live now?") == "location"
