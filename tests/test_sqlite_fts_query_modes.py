from __future__ import annotations

from c3ae.storage.sqlite_store import _sanitize_fts_query


def test_sanitize_fts_short_query_keeps_precision():
    q = _sanitize_fts_query('user_id abc-123?')
    # Short queries should remain AND-style for precision.
    assert " OR " not in q
    assert '"user_id"' in q
    assert '"abc"' in q
    assert '"123"' in q


def test_sanitize_fts_long_query_uses_or_recall_mode():
    q = _sanitize_fts_query(
        "when did user prefer novaspine over other memory systems in prior sessions"
    )
    assert " OR " in q
    assert '"novaspine"' in q
    assert '"sessions"' in q


def test_sanitize_fts_dedupes_and_caps_tokens():
    query = " ".join(["alpha"] * 30 + ["beta", "gamma", "2026"])
    q = _sanitize_fts_query(query)
    # Dedupe should keep single alpha and preserve added anchors.
    assert q.count('"alpha"') == 1
    assert '"beta"' in q
    assert '"gamma"' in q
    assert '"2026"' in q
