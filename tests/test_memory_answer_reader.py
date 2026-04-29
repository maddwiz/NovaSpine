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


def test_reader_extracts_shift_header_from_crosstab_schedule():
    answer = answer_from_memory(
        "What was the rotation for Admon on a Sunday?",
        [
            _row(
                "\n".join(
                    [
                        "|  | 8 am - 4 pm (Day Shift) | 12 pm - 8 pm (Afternoon Shift) |",
                        "| --- | --- | --- |",
                        "| Sunday | Admon | Magdy |",
                    ]
                ),
                "rotation-table",
            )
        ],
    )

    assert answer.answer == "8 am - 4 pm (Day Shift)"
    assert answer.citations == ["rotation-table"]


def test_reader_resolves_yesterday_with_session_date():
    answer = answer_from_memory(
        "When was the pottery class?",
        [_row("I had pottery class yesterday.", "pottery", session_date="2023-07-03T12:00:00Z")],
    )

    assert answer.answer == "2 July 2023"
    assert answer.verifier_status == "supported"


def test_reader_does_not_treat_speaker_prefix_as_date_answer():
    answer = answer_from_memory(
        "When did Melanie sign up for a pottery class?",
        [
            _row(
                (
                    "Melanie: Wow, Caroline! That's great! "
                    "I just signed up for a pottery class yesterday."
                ),
                "pottery",
                session_date="1:36 pm on 3 July, 2023",
            )
        ],
    )

    assert answer.answer == "2 July 2023"


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


def test_reader_chooses_first_quoted_event_by_relative_dates():
    answer = answer_from_memory(
        "Which event did I attend first, the 'Effective Time Management' workshop or the 'Data Analysis using Python' webinar?",
        [
            _row(
                'I attended a workshop on "Effective Time Management" at the local community center last Saturday.',
                "time-workshop",
                session_date="2023/05/28 (Sun) 21:04",
            ),
            _row(
                'I participated in a webinar on "Data Analysis using Python" two months ago.',
                "python-webinar",
                session_date="2023/05/28 (Sun) 07:17",
            ),
        ],
    )

    assert answer.answer == "Data Analysis using Python webinar"
    assert answer.citations == ["python-webinar"]


def test_reader_chooses_first_vehicle_nearest_date():
    answer = answer_from_memory(
        "Which vehicle did I take care of first in February, the bike or the car?",
        [
            _row(
                "I was looking at a new bike rack for my car. My bike had issues in mid-February, so I took it in for repairs.",
                "bike-repair",
                session_date="2023/03/10 (Fri) 22:50",
            ),
            _row(
                "My car is still in good condition and I use it for errands.",
                "car-note",
                session_date="2023/03/10 (Fri) 08:11",
            ),
        ],
    )

    assert answer.answer == "bike"
    assert answer.citations == ["bike-repair"]


def test_reader_prefers_arrival_over_preorder_for_got_first_question():
    answer = answer_from_memory(
        "Which device did I got first, the Samsung Galaxy S22 or the Dell XPS 13?",
        [
            _row(
                "I pre-ordered the laptop, Dell XPS 13, on January 28th, and it finally arrived on February 25th.",
                "dell",
                session_date="2023/03/15 (Wed) 10:31",
            ),
            _row(
                "I recently got a new Samsung Galaxy S22 from Best Buy on February 20th.",
                "samsung",
                session_date="2023/03/15 (Wed) 00:56",
            ),
        ],
    )

    assert answer.answer == "Samsung Galaxy S22"
    assert answer.citations == ["samsung"]


def test_reader_resolves_option_comparison_with_weeks_and_months_ago():
    answer = answer_from_memory(
        "Which item did I purchase first, dog bed for Max or training pads for Luna?",
        [
            _row("I bought a dog bed for Max about 3 weeks ago.", "dog-bed", session_date="2023/05/28 (Sun) 12:00"),
            _row("I purchased training pads for Luna about a month ago.", "training-pads", session_date="2023/05/28 (Sun) 12:00"),
        ],
    )

    assert answer.answer == "training pads for Luna"
    assert answer.citations == ["training-pads"]


def test_reader_extracts_issue_target():
    answer = answer_from_memory(
        "What was the first issue I had with my new car after its first service?",
        [
            _row(
                "I had an issue with my car's GPS system on 3/22, and I had to take it back to the dealership.",
                "gps",
                session_date="2023/04/10 (Mon) 14:47",
            )
        ],
    )

    assert answer.answer == "GPS system"
    assert answer.citations == ["gps"]


def test_reader_computes_day_delta_from_month_day_dates_in_question():
    answer = answer_from_memory(
        "How many days were there between the January 2nd event and the February 1st event?",
        [_row("The relevant year for these events is 2023.", "calendar", session_date="2023/02/20 (Mon) 09:05")],
    )

    assert answer.answer == "30"


def test_reader_uses_session_date_header_when_metadata_missing():
    answer = answer_from_memory(
        "Which item did I purchase first, the dog bed for Max or the training pads for Luna?",
        [
            _row(
                "Session date: 2023/05/20 (Sat) 23:31\n\nuser: I got a new dog bed for Max 3 weeks ago.",
                "dog-bed",
            ),
            _row(
                "Session date: 2023/05/20 (Sat) 06:19\n\nuser: I got training pads for Luna about a month ago.",
                "training-pads",
            ),
        ],
    )

    assert answer.answer == "training pads for Luna"
    assert answer.citations == ["training-pads"]


def test_reader_extracts_cleaned_shoe_pair():
    answer = answer_from_memory(
        "Which pair of shoes did I clean last month?",
        [
            _row(
                "I finally got around to cleaning my white Adidas sneakers last month, which I'd been meaning to do for weeks.",
                "shoes",
            )
        ],
    )

    assert answer.answer == "white Adidas sneakers"


def test_reader_computes_duration_before_current_job():
    answer = answer_from_memory(
        "How long have I been working before I started my current job at NovaTech?",
        [
            _row("I've been working professionally for 9 years and still keep a notebook for tasks.", "career-total"),
            _row("I've been working at NovaTech for about 4 years and 3 months now.", "current-job"),
        ],
    )

    assert answer.answer == "4 years and 9 months"
    assert answer.verifier_status == "supported"
