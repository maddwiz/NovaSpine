from c3ae.qa.reader import answer_from_memory


def _row(content: str, row_id: str = "m1", **metadata):
    return {"id": row_id, "content": content, "metadata": metadata}


def test_reader_extracts_relative_year_with_citation():
    answer = answer_from_memory(
        "When did Melanie paint a sunrise?",
        [
            _row(
                "Melanie: Yeah, I painted that lake sunrise last year!",
                "melanie-1",
                reference_date="2023-08-01T00:00:00Z",
            )
        ],
    )

    assert answer.answer == "2022"
    assert answer.verifier_status == "supported"
    assert answer.citations == ["melanie-1"]


def test_reader_extracts_researched_topic():
    answer = answer_from_memory(
        "What did Caroline research?",
        [_row("Caroline researched adoption agencies before deciding.", "caroline-1")],
    )

    assert answer.answer == "adoption agencies"
    assert answer.verifier_status == "supported"


def test_reader_extracts_relationship_status():
    answer = answer_from_memory(
        "What is Caroline's relationship status?",
        [_row("Caroline said she is currently single.", "caroline-2")],
    )

    assert answer.answer == "Single"
    assert answer.verifier_status == "supported"


def test_reader_abstains_when_requested_relation_missing():
    answer = answer_from_memory(
        "What did my dad give me as a birthday gift?",
        [_row("My sister gave me a Moleskine notebook as a birthday gift.", "gift-1")],
    )

    assert answer.abstain
    assert answer.answer == "not enough information"


def test_reader_counts_project_mentions():
    answer = answer_from_memory(
        "How many projects have I led or am currently leading?",
        [
            _row("I led Project Atlas last winter.", "project-1"),
            _row("I am currently leading Project Orion.", "project-2"),
        ],
    )

    assert answer.answer == "2"
    assert answer.verifier_status == "supported"


def test_reader_prefers_current_over_old_for_current_state():
    answer = answer_from_memory(
        "What does the user currently prefer?",
        [
            _row("The user used to prefer coffee.", "old", entry_status="superseded"),
            _row("The user now prefers matcha.", "new", entry_status="active"),
        ],
    )

    assert answer.answer == "matcha"
    assert answer.citations == ["new"]


def test_reader_can_answer_historical_question_from_old_memory():
    answer = answer_from_memory(
        "Where did the user used to live?",
        [
            _row("The user moved to Chicago.", "new", entry_status="active"),
            _row("The user used to live in Denver.", "old", entry_status="superseded"),
        ],
    )

    assert answer.answer == "Denver"


def test_reader_extracts_current_preference_from_markdown_table():
    answer = answer_from_memory(
        "What drink does the user currently prefer?",
        [
            _row(
                "\n".join(
                    [
                        "| Person | Previous drink | Current drink |",
                        "| --- | --- | --- |",
                        "| user | drip coffee | iced matcha |",
                    ]
                ),
                "prefs-table",
                entry_status="active",
            )
        ],
    )

    assert answer.answer == "iced matcha"
    assert answer.citations == ["prefs-table"]
    assert answer.verifier_status == "supported"


def test_reader_extracts_labeled_bullet_list_answer():
    answer = answer_from_memory(
        "Which allergies does the user have?",
        [
            _row(
                "\n".join(
                    [
                        "- Current city: Boulder",
                        "- Allergies: peanuts, shellfish, and latex",
                    ]
                ),
                "profile-list",
            )
        ],
    )

    assert answer.answer == "peanuts, shellfish, latex"
    assert answer.answer_type == "list"
    assert answer.citations == ["profile-list"]


def test_reader_prefers_current_table_column_over_historical_column():
    answer = answer_from_memory(
        "What notebook does the user currently use?",
        [
            _row(
                "\n".join(
                    [
                        "| Item | Old notebook | Current notebook |",
                        "| --- | --- | --- |",
                        "| user | Moleskine | Leuchtturm1917 |",
                    ]
                ),
                "notebook-table",
            )
        ],
    )

    assert answer.answer == "Leuchtturm1917"
    assert answer.citations == ["notebook-table"]


def test_reader_can_select_historical_table_column():
    answer = answer_from_memory(
        "What drink did the user previously prefer?",
        [
            _row(
                "\n".join(
                    [
                        "| Person | Previous drink | Current drink |",
                        "| --- | --- | --- |",
                        "| user | drip coffee | iced matcha |",
                    ]
                ),
                "prefs-table",
            )
        ],
    )

    assert answer.answer == "drip coffee"
    assert answer.citations == ["prefs-table"]


def test_reader_extracts_shift_cell_from_schedule_table():
    answer = answer_from_memory(
        "In the schedule table, which shift is Admon assigned on Sundays?",
        [
            _row(
                "\n".join(
                    [
                        "| Day | Person | Shift |",
                        "| --- | --- | --- |",
                        "| Sunday | Admon | 8 am - 4 pm (Day Shift) |",
                        "| Monday | Blair | 4 pm - 12 am (Swing Shift) |",
                    ]
                ),
                "shift-table",
            )
        ],
    )

    assert answer.answer == "8 am - 4 pm (Day Shift)"
    assert answer.citations == ["shift-table"]
    assert answer.verifier_status == "supported"


def test_reader_resolves_yesterday_with_session_date():
    answer = answer_from_memory(
        "When was the pottery class?",
        [_row("I had pottery class yesterday.", "pottery", session_date="2023-07-03T12:00:00Z")],
    )

    assert answer.answer == "2 July 2023"
    assert answer.verifier_status == "supported"


def test_reader_aggregates_list_items_across_rows():
    answer = answer_from_memory(
        "Which allergies does the user have?",
        [
            _row("- Allergies: peanuts and shellfish", "allergy-1"),
            _row("- Allergies: latex", "allergy-2"),
        ],
    )

    assert answer.answer == "peanuts, shellfish, latex"
    assert answer.citations == ["allergy-1", "allergy-2"]


def test_reader_computes_simple_day_delta_from_question_and_memory_date():
    answer = answer_from_memory(
        "How many days before 10 July 2023 was the pottery class?",
        [_row("I had pottery class yesterday.", "pottery", session_date="2023-07-03T12:00:00Z")],
    )

    assert answer.answer == "8"
    assert answer.verifier_status == "supported"
