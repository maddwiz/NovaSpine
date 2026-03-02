"""Temporal extraction and context enrichment helpers for QA."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date


_MONTH_NAME_TO_NUM = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

_MONTH_PATTERN = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
_MONTH_DATE_RE = re.compile(
    rf"\b(?P<month>{_MONTH_PATTERN})\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?(?:,\s*(?P<year>\d{{4}}))?\b",
    re.IGNORECASE,
)
_NUMERIC_DATE_RE = re.compile(
    r"\b(?P<month>\d{1,2})/(?P<day>\d{1,2})(?:/(?P<year>\d{2,4}))?\b",
    re.IGNORECASE,
)
_ISO_DATE_RE = re.compile(r"\b(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})\b")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


@dataclass(frozen=True)
class TemporalMention:
    """A date-like value extracted from context."""

    parsed: date
    raw_text: str
    start: int


def _coerce_year(raw_year: str | None) -> int:
    if not raw_year:
        return -1
    year = int(raw_year)
    if year < 100:
        year += 2000 if year < 70 else 1900
    return year


def _make_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def extract_temporal_mentions(text: str, *, default_year: int | None = None) -> list[TemporalMention]:
    """Extract date mentions from text with best-effort year inference."""
    if not text:
        return []
    candidates: list[tuple[int, int, int, str, int]] = []
    explicit_years: list[int] = []

    for m in _MONTH_DATE_RE.finditer(text):
        raw_month = m.group("month").lower()
        month = _MONTH_NAME_TO_NUM.get(raw_month, _MONTH_NAME_TO_NUM.get(raw_month[:3], 0))
        if month <= 0:
            continue
        day = int(m.group("day"))
        year = _coerce_year(m.group("year"))
        if year > 0:
            explicit_years.append(year)
        candidates.append((year, month, day, m.group(0), m.start()))

    for m in _NUMERIC_DATE_RE.finditer(text):
        month = int(m.group("month"))
        day = int(m.group("day"))
        year = _coerce_year(m.group("year"))
        if year > 0:
            explicit_years.append(year)
        candidates.append((year, month, day, m.group(0), m.start()))

    for m in _ISO_DATE_RE.finditer(text):
        year = int(m.group("year"))
        month = int(m.group("month"))
        day = int(m.group("day"))
        explicit_years.append(year)
        candidates.append((year, month, day, m.group(0), m.start()))

    inferred_year = default_year
    if inferred_year is None and explicit_years:
        inferred_year = max(set(explicit_years), key=explicit_years.count)

    mentions: list[TemporalMention] = []
    seen: set[tuple[date, int]] = set()
    for year, month, day, raw, start in candidates:
        yyyy = year if year > 0 else (inferred_year if inferred_year is not None else 2000)
        parsed = _make_date(yyyy, month, day)
        if parsed is None:
            continue
        key = (parsed, start)
        if key in seen:
            continue
        seen.add(key)
        mentions.append(TemporalMention(parsed=parsed, raw_text=raw, start=start))
    mentions.sort(key=lambda m: (m.parsed, m.start))
    return mentions


def _query_default_year(query: str) -> int | None:
    if not query:
        return None
    m = _YEAR_RE.search(query)
    if not m:
        return None
    return int(m.group(0))


def compute_temporal_facts(
    mentions: list[TemporalMention],
    *,
    max_pairs: int = 24,
) -> list[str]:
    """Compute pairwise day deltas from extracted temporal mentions."""
    if len(mentions) < 2:
        return []
    facts: list[str] = []
    seen_pairs: set[tuple[date, date]] = set()
    for i in range(len(mentions)):
        for j in range(i + 1, min(len(mentions), i + 4)):
            left = mentions[i]
            right = mentions[j]
            pair_key = (left.parsed, right.parsed)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            delta_days = (right.parsed - left.parsed).days
            if delta_days <= 0:
                continue
            week_count = delta_days // 7
            week_label = "week" if week_count == 1 else "weeks"
            facts.append(
                f"- {left.raw_text} to {right.raw_text}: {delta_days} days ({week_count} {week_label})"
            )
            if len(facts) >= max(1, int(max_pairs)):
                return facts
    return facts


def enrich_context_with_temporal_facts(
    *,
    query: str,
    context: str,
    max_pairs: int = 24,
) -> str:
    """Append computed temporal facts to context when at least two dates are present."""
    if not context.strip():
        return context
    mentions = extract_temporal_mentions(context, default_year=_query_default_year(query))
    facts = compute_temporal_facts(mentions, max_pairs=max_pairs)
    if not facts:
        return context
    block = "[TEMPORAL FACTS computed from context]\n" + "\n".join(facts)
    return context.rstrip() + "\n\n" + block

