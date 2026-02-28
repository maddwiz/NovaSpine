from __future__ import annotations

from c3ae.eval.qa_metrics import (
    best_exact_match,
    best_f1,
    extractive_answer,
    normalize_text,
)


def test_normalize_text_basic():
    assert normalize_text("The Quick, Brown Fox!") == "quick brown fox"


def test_exact_match_and_f1():
    answers = ["Wilhelm Conrad Röntgen", "Rontgen"]
    pred = "Wilhelm Conrad Rontgen"
    assert best_exact_match(pred, answers) == 1.0
    assert best_f1(pred, answers) > 0.7


def test_exact_match_accepts_multi_answer_subclauses():
    answers = ["7 days. 8 days (including the last day) is also acceptable."]
    assert best_exact_match("7 days", answers) == 1.0
    assert best_f1("7 days", answers) >= 0.99


def test_normalize_numeric_words_in_exact_match():
    answers = ["twice"]
    assert best_exact_match("2", answers) == 1.0


def test_extractive_answer_prefers_query_overlap():
    recalled = [
        {"content": "This paragraph is unrelated to the question."},
        {"content": "The first Nobel Prize in Physics was awarded to Wilhelm Conrad Röntgen."},
    ]
    pred = extractive_answer("who got the first nobel prize in physics", recalled)
    assert "wilhelm conrad röntgen" in pred.lower()


def test_extractive_answer_handles_case_tokens_and_options():
    recalled = [
        {
            "content": "I attended the 'Data Analysis using Python' webinar before the "
            "'Effective Time Management' workshop."
        }
    ]
    q = "__LME_CASE_0001__ Which event did I attend first, the 'Effective Time Management' workshop or the 'Data Analysis using Python' webinar?"
    pred = extractive_answer(q, recalled)
    assert pred.lower() == "data analysis using python"


def test_extractive_answer_finds_numeric_answer():
    recalled = [{"content": "I had been preparing for 7 days before the meeting."}]
    pred = extractive_answer("How many days before the meeting was I preparing?", recalled)
    assert pred.lower() == "7 days"
