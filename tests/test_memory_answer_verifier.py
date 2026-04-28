from c3ae.qa.verifier import verify_answer_support


def test_verifier_supports_relative_year_with_metadata():
    result = verify_answer_support(
        "2022",
        ["Melanie: Yeah, I painted that lake sunrise last year!"],
        "year",
        metadata={"reference_date": "2023-04-01T00:00:00Z"},
    )

    assert result.status == "supported"


def test_verifier_rejects_unsupported_answer():
    result = verify_answer_support(
        "whole milk",
        ["The note only says cappuccino."],
        "preference",
    )

    assert result.status == "unsupported"
