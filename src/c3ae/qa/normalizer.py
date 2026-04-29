"""Deterministic answer-type inference and normalization."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import re
import unicodedata
from typing import Any


ANSWER_TYPES = {
    "date",
    "year",
    "person",
    "location",
    "relationship_status",
    "preference",
    "count",
    "number",
    "yes_no",
    "list",
    "free_text",
    "not_enough_information",
}

_MONTH_RE = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
_NULL_PHRASES = {
    "unknown",
    "not mentioned",
    "not specified",
    "not enough information",
    "insufficient information",
    "no answer",
    "none",
}
_NUM_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}
_MONTHS = {
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
_WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


@dataclass
class NormalizedAnswer:
    answer: str
    answer_type: str
    steps: list[str] = field(default_factory=list)


def clean_answer(text: str) -> str:
    value = re.sub(r"\s+", " ", (text or "").strip())
    value = value.strip(" \t\r\n\"'`.,;:")
    return value


def clean_cell(text: str) -> str:
    value = re.sub(r"\s+", " ", (text or "").strip())
    return value.strip(" \t\r\n\"'`.,;:")


def normalize_for_match(text: str) -> str:
    value = unicodedata.normalize("NFKD", text or "")
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.lower()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    words = [_NUM_WORDS.get(tok, tok) for tok in value.split()]
    return " ".join(str(tok) for tok in words).strip()


def is_abstention(text: str) -> bool:
    norm = normalize_for_match(text)
    return not norm or any(phrase in norm for phrase in _NULL_PHRASES)


def infer_answer_type(question: str) -> str:
    q = normalize_for_match(question)
    if not q:
        return "free_text"
    if q.startswith("how many") or " number of " in f" {q} ":
        return "count"
    if q.startswith("what year") or "which year" in q:
        return "year"
    if q.startswith("when") or "what date" in q:
        return "date"
    if q.startswith("who"):
        return "person"
    if q.startswith("where"):
        return "location"
    if any(term in q for term in ("relationship status", "single", "married", "dating")):
        return "relationship_status"
    if any(term in q for term in ("prefer", "favorite", "favourite", "like", "order", "drink", "bag", "notebook")):
        return "preference"
    if q.split(" ", 1)[0] in {"is", "are", "was", "were", "do", "does", "did", "can", "could", "should", "has", "have"}:
        return "yes_no"
    if any(term in q for term in ("which", "what did", "what was", "what is")):
        return "free_text"
    return "free_text"


def _parse_reference_datetime(metadata: dict[str, Any] | None) -> datetime | None:
    if not metadata:
        return None
    for key in ("reference_date", "session_date", "question_date", "date", "timestamp", "created_at", "_created_at"):
        raw = metadata.get(key)
        if not isinstance(raw, str) or not raw.strip():
            continue
        value = raw.strip()
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            parsed = _parse_loose_reference_datetime(value)
            if parsed is None:
                continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    return None


def _parse_loose_reference_datetime(value: str) -> datetime | None:
    candidates = [value.strip()]
    on_match = re.search(r"\bon\s+(\d{1,2}\s+[A-Za-z]+,?\s+\d{4})\b", value, re.IGNORECASE)
    if on_match:
        candidates.append(on_match.group(1))
    slash_match = re.search(r"\b((?:19|20)\d{2}/\d{1,2}/\d{1,2})\b", value)
    if slash_match:
        candidates.append(slash_match.group(1))
    iso_match = re.search(r"\b((?:19|20)\d{2}-\d{1,2}-\d{1,2})\b", value)
    if iso_match:
        candidates.append(iso_match.group(1))
    month_match = re.search(r"\b(\d{1,2}\s+[A-Za-z]+,?\s+\d{4})\b", value)
    if month_match:
        candidates.append(month_match.group(1))
    for candidate in candidates:
        cleaned = candidate.replace(",", "").strip()
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d %B %Y", "%d %b %Y"):
            try:
                return datetime.strptime(cleaned, fmt)
            except ValueError:
                continue
    return None


def _normalize_year(text: str, metadata: dict[str, Any] | None, steps: list[str]) -> str:
    value = text.lower()
    m = re.search(r"\b(19|20)\d{2}\b", value)
    if m:
        steps.append("extracted_explicit_year")
        return m.group(0)
    ref = _parse_reference_datetime(metadata)
    if ref is not None:
        if re.search(r"\blast year\b", value):
            steps.append("resolved_last_year")
            return str(ref.year - 1)
        if re.search(r"\bthis year\b", value):
            steps.append("resolved_this_year")
            return str(ref.year)
        if re.search(r"\bnext year\b", value):
            steps.append("resolved_next_year")
            return str(ref.year + 1)
    return clean_answer(text)


def _format_date(value: datetime) -> str:
    return f"{value.day} {value.strftime('%B')} {value.year}"


def _number_from_word_or_digits(value: str) -> int | None:
    raw = (value or "").strip().lower()
    if raw in {"a", "an"}:
        return 1
    normalized = normalize_for_match(value)
    if normalized in {"couple"}:
        return 2
    if normalized in {"few"}:
        return 3
    if normalized.isdigit():
        return int(normalized)
    number = _NUM_WORDS.get(normalized)
    return int(number) if isinstance(number, int) else None


def _subtract_months(value: datetime, months: int) -> datetime:
    month_index = (value.year * 12 + value.month - 1) - months
    year = month_index // 12
    month = month_index % 12 + 1
    days_in_month = [31, 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return value.replace(year=year, month=month, day=min(value.day, days_in_month[month - 1]))


def _month_number(value: str) -> int | None:
    return _MONTHS.get(value.lower().rstrip("."))


def _date_from_parts(day: int, month: int, year: int) -> datetime | None:
    try:
        return datetime(year=year, month=month, day=day, tzinfo=timezone.utc)
    except ValueError:
        return None


def _canonicalize_date_span(raw: str, metadata: dict[str, Any] | None, steps: list[str]) -> str:
    value = clean_answer(raw)
    ref = _parse_reference_datetime(metadata)

    iso = re.fullmatch(r"(?P<year>(?:19|20)\d{2})-(?P<month>\d{1,2})-(?P<day>\d{1,2})", value)
    if iso:
        parsed = _date_from_parts(int(iso.group("day")), int(iso.group("month")), int(iso.group("year")))
        if parsed is not None:
            steps.append("canonicalized_explicit_date")
            return _format_date(parsed)

    slash = re.fullmatch(r"(?P<month>\d{1,2})/(?P<day>\d{1,2})(?:/(?P<year>\d{2,4}))?", value)
    if slash:
        year_raw = slash.group("year")
        if year_raw:
            year = int(year_raw)
            if year < 100:
                year += 2000 if year < 70 else 1900
        elif ref is not None:
            year = ref.year
        else:
            return value
        parsed = _date_from_parts(int(slash.group("day")), int(slash.group("month")), year)
        if parsed is not None:
            steps.append("canonicalized_slash_date")
            return _format_date(parsed)

    day_month = re.fullmatch(
        rf"(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s+(?P<month>{_MONTH_RE})(?:,?\s*(?P<year>(?:19|20)\d{{2}}))?",
        value,
        re.IGNORECASE,
    )
    if day_month:
        month = _month_number(day_month.group("month"))
        year = int(day_month.group("year")) if day_month.group("year") else (ref.year if ref is not None else None)
        if month is not None and year is not None:
            parsed = _date_from_parts(int(day_month.group("day")), month, year)
            if parsed is not None:
                steps.append("canonicalized_day_month_date")
                return _format_date(parsed)
        return value

    month_day = re.fullmatch(
        rf"(?P<month>{_MONTH_RE})\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s*(?P<year>(?:19|20)\d{{2}}))?",
        value,
        re.IGNORECASE,
    )
    if month_day:
        month = _month_number(month_day.group("month"))
        year = int(month_day.group("year")) if month_day.group("year") else (ref.year if ref is not None else None)
        if month is not None and year is not None:
            parsed = _date_from_parts(int(month_day.group("day")), month, year)
            if parsed is not None:
                steps.append("canonicalized_month_day_date")
                return _format_date(parsed)
        return value

    phase_month = re.fullmatch(
        rf"(?P<phase>early|mid|late)[-\s]+(?P<month>{_MONTH_RE})(?:,?\s*(?P<year>(?:19|20)\d{{2}}))?",
        value,
        re.IGNORECASE,
    )
    if phase_month:
        month = _month_number(phase_month.group("month"))
        year = int(phase_month.group("year")) if phase_month.group("year") else (ref.year if ref is not None else None)
        day_by_phase = {"early": 5, "mid": 15, "late": 25}
        if month is not None and year is not None:
            parsed = _date_from_parts(day_by_phase[phase_month.group("phase").lower()], month, year)
            if parsed is not None:
                steps.append("canonicalized_month_phase_date")
                return _format_date(parsed)
        return value

    month_year = re.fullmatch(rf"(?P<month>{_MONTH_RE})\s+(?P<year>(?:19|20)\d{{2}})", value, re.IGNORECASE)
    if month_year:
        month = _month_number(month_year.group("month"))
        if month is not None:
            steps.append("canonicalized_month_year")
            return f"{datetime(2000, month, 1).strftime('%B')} {month_year.group('year')}"
    return value


def _normalize_relative_date(text: str, metadata: dict[str, Any] | None, steps: list[str]) -> str:
    ref = _parse_reference_datetime(metadata)
    if ref is None:
        return ""
    value = text.lower()
    if re.search(r"\byesterday\b", value):
        steps.append("resolved_yesterday")
        return _format_date(ref - timedelta(days=1))
    if re.search(r"\btoday\b", value):
        steps.append("resolved_today")
        return _format_date(ref)
    if re.search(r"\btomorrow\b", value):
        steps.append("resolved_tomorrow")
        return _format_date(ref + timedelta(days=1))
    if re.search(r"\blast week\b", value):
        steps.append("resolved_last_week")
        return _format_date(ref - timedelta(days=7))
    if re.search(r"\bnext week\b", value):
        steps.append("resolved_next_week")
        return _format_date(ref + timedelta(days=7))
    m = re.search(
        r"\b(?:about|around|roughly|approximately)?\s*(?P<num>\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|couple|few)\s+"
        r"(?P<unit>day|days|week|weeks|month|months|year|years)\s+ago\b",
        value,
    )
    if m:
        amount = _number_from_word_or_digits(m.group("num"))
        if amount is not None:
            unit = m.group("unit")
            if unit.startswith("day"):
                resolved = ref - timedelta(days=amount)
            elif unit.startswith("week"):
                resolved = ref - timedelta(days=amount * 7)
            elif unit.startswith("month"):
                resolved = _subtract_months(ref, amount)
            else:
                resolved = ref.replace(year=ref.year - amount)
            steps.append(f"resolved_{unit.rstrip('s')}_ago")
            return _format_date(resolved)
    m = re.search(r"\b(?:last|this past)\s+(?P<weekday>monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", value)
    if m:
        target = _WEEKDAYS[m.group("weekday")]
        days_back = (ref.weekday() - target) % 7
        if days_back == 0:
            days_back = 7
        steps.append("resolved_last_weekday")
        return _format_date(ref - timedelta(days=days_back))
    if re.search(r"\brecently\b", value):
        steps.append("resolved_recently_approximate")
        return _format_date(ref - timedelta(days=3))
    return ""


def _normalize_date(text: str, metadata: dict[str, Any] | None, steps: list[str]) -> str:
    relative = _normalize_relative_date(text, metadata, steps)
    if relative:
        return relative
    for pattern in (
        rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH_RE}(?:\s*(?:19|20)\d{{2}})?\b",
        rf"\b{_MONTH_RE}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,\s*(?:19|20)\d{{2}})?\b",
        rf"\b(?:early|mid|late)[-\s]+{_MONTH_RE}(?:,?\s*(?:19|20)\d{{2}})?\b",
        rf"\b{_MONTH_RE}\s+(?:19|20)\d{{2}}\b",
        r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b",
        r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b",
    ):
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            steps.append("extracted_date_span")
            return _canonicalize_date_span(m.group(0), metadata, steps)
    year = _normalize_year(text, metadata, steps=[])
    if re.fullmatch(r"(19|20)\d{2}", year):
        steps.append("normalized_date_to_year")
        return year
    return clean_answer(text)


def _normalize_number(text: str, steps: list[str]) -> str:
    m = re.search(r"\b\d+(?:\.\d+)?\b", text)
    if m:
        steps.append("extracted_number")
        return m.group(0)
    for word, value in _NUM_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE):
            steps.append("normalized_number_word")
            return str(value)
    return clean_answer(text)


def normalize_answer(
    answer: str,
    answer_type: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> NormalizedAnswer:
    inferred = answer_type if answer_type in ANSWER_TYPES else "free_text"
    steps: list[str] = []
    value = clean_answer(answer)
    if is_abstention(value):
        return NormalizedAnswer("not enough information", "not_enough_information", ["normalized_abstention"])
    if inferred == "year":
        value = _normalize_year(value, metadata, steps)
    elif inferred == "date":
        value = _normalize_date(value, metadata, steps)
    elif inferred in {"count", "number"}:
        value = _normalize_number(value, steps)
    elif inferred == "yes_no":
        norm = normalize_for_match(value)
        if any(tok in norm.split() for tok in ("yes", "true", "confirmed")):
            value = "yes"
            steps.append("normalized_yes_no")
        elif any(tok in norm.split() for tok in ("no", "false")):
            value = "no"
            steps.append("normalized_yes_no")
    elif inferred == "relationship_status":
        m = re.search(r"\b(single|married|divorced|engaged|dating|separated|widowed)\b", value, re.IGNORECASE)
        if m:
            value = m.group(1).capitalize()
            steps.append("normalized_relationship_status")
    elif inferred == "list":
        parts = [clean_answer(p) for p in re.split(r",|\band\b", value) if clean_answer(p)]
        if parts:
            value = ", ".join(parts)
            steps.append("normalized_list")
    else:
        value = clean_answer(value)
    return NormalizedAnswer(value, inferred, steps)
