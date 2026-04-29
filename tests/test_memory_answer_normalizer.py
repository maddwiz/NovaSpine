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


def test_normalizes_relative_day_with_locomo_session_date_string():
    out = normalize_answer(
        "yesterday",
        "date",
        {"session_date": "1:36 pm on 3 July, 2023"},
    )

    assert out.answer == "2 July 2023"


def test_session_date_takes_priority_over_storage_created_at():
    out = normalize_answer(
        "yesterday",
        "date",
        {
            "session_date": "1:36 pm on 3 July, 2023",
            "_created_at": "2026-04-28T20:00:00Z",
        },
    )

    assert out.answer == "2 July 2023"


def test_normalizes_relative_day_with_longmem_session_date_string():
    out = normalize_answer(
        "yesterday",
        "date",
        {"session_date": "2023/05/20 (Sat) 09:05"},
    )

    assert out.answer == "19 May 2023"


def test_normalizes_month_day_without_year_from_session_date():
    out = normalize_answer(
        "January 2nd",
        "date",
        {"session_date": "2023/02/20 (Mon) 09:05"},
    )

    assert out.answer == "2 January 2023"


def test_normalizes_weekday_and_month_offsets_from_session_date():
    assert normalize_answer("last Saturday", "date", {"session_date": "2023/05/28 (Sun) 09:05"}).answer == "27 May 2023"
    assert normalize_answer("two months ago", "date", {"session_date": "2023/05/28 (Sun) 09:05"}).answer == "28 March 2023"
    assert normalize_answer("3 weeks ago", "date", {"session_date": "2023/05/20 (Sat) 09:05"}).answer == "29 April 2023"


def test_normalizes_month_phase_from_session_date():
    out = normalize_answer(
        "mid-February",
        "date",
        {"session_date": "2023/03/10 (Fri) 22:50"},
    )

    assert out.answer == "15 February 2023"


def test_normalizes_counts_and_match_text():
    assert normalize_answer("two projects", "count").answer == "2"
    assert normalize_for_match("The Two Projects!") == "2 projects"


def test_infers_common_memory_answer_types():
    assert infer_answer_type("How many projects have I led?") == "count"
    assert infer_answer_type("What is Caroline's relationship status?") == "relationship_status"
    assert infer_answer_type("Where do I live now?") == "location"
