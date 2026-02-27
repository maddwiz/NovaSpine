"""Lightweight QA metrics and extractive answer heuristic."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "to", "of", "in", "on", "for",
    "and", "or", "with", "what", "when", "who", "where", "which", "did", "do",
    "does", "at", "by", "from", "as", "it", "be", "this", "that",
}
_CASE_TOKEN_RE = re.compile(r"__\w+_CASE_\d+__", re.IGNORECASE)
_MONTH_RE = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)


def normalize_text(s: str) -> str:
    t = unicodedata.normalize("NFKD", s)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = t.lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\b(a|an|the)\b", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def token_f1(pred: str, gold: str) -> float:
    p_toks = normalize_text(pred).split()
    g_toks = normalize_text(gold).split()
    if not p_toks and not g_toks:
        return 1.0
    if not p_toks or not g_toks:
        return 0.0
    p_counts: dict[str, int] = {}
    g_counts: dict[str, int] = {}
    for t in p_toks:
        p_counts[t] = p_counts.get(t, 0) + 1
    for t in g_toks:
        g_counts[t] = g_counts.get(t, 0) + 1
    common = 0
    for t, c in p_counts.items():
        common += min(c, g_counts.get(t, 0))
    if common == 0:
        return 0.0
    precision = common / len(p_toks)
    recall = common / len(g_toks)
    return 2 * precision * recall / (precision + recall)


def best_exact_match(pred: str, answers: list[str]) -> float:
    n_pred = normalize_text(pred)
    return 1.0 if any(n_pred == normalize_text(a) for a in answers if a.strip()) else 0.0


def best_f1(pred: str, answers: list[str]) -> float:
    if not answers:
        return 0.0
    return max(token_f1(pred, a) for a in answers if a.strip())


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[\.\?\!])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p and p.strip()]


def _clean_query(query: str) -> str:
    return re.sub(r"\s+", " ", _CASE_TOKEN_RE.sub(" ", query)).strip()


def _query_tokens(query: str) -> set[str]:
    toks = set(re.findall(r"[a-z0-9_]+", _clean_query(query).lower()))
    return {t for t in toks if len(t) > 1 and t not in _STOPWORDS}


def _sentence_score(query: str, sentence: str) -> float:
    q_toks = _query_tokens(query)
    if not q_toks:
        return 0.0
    s_toks = set(re.findall(r"[a-z0-9_]+", sentence.lower()))
    overlap = len(q_toks & s_toks)
    if overlap == 0:
        return 0.0
    return overlap / max(1, len(q_toks)) + min(len(sentence), 240) / 4000.0


def _extract_options(query: str) -> list[str]:
    clean = _clean_query(query)
    quoted = [q for q in re.findall(r"'([^']+)'|\"([^\"]+)\"", clean) for q in q if q.strip()]
    if len(quoted) >= 2:
        return [q.strip() for q in quoted[:4]]

    words = re.findall(r"[A-Za-z0-9_]+", clean)
    if "or" not in [w.lower() for w in words]:
        return []
    lowers = [w.lower() for w in words]
    idx = lowers.index("or")
    left = [w for w in words[max(0, idx - 4):idx] if w.lower() not in _STOPWORDS]
    right = [w for w in words[idx + 1:min(len(words), idx + 5)] if w.lower() not in _STOPWORDS]
    if not left or not right:
        return []
    return [left[-1], right[0]]


def _find_best_options_hit(query: str, sentence: str) -> str:
    hits: list[tuple[int, str]] = []
    for opt in _extract_options(query):
        m = re.search(rf"\b{re.escape(opt)}\b", sentence, re.IGNORECASE)
        if m:
            hits.append((m.start(), opt))
    if hits:
        hits.sort(key=lambda x: x[0])
        return hits[0][1]
    return ""


def _extract_date_or_number(q: str, sentence: str) -> str:
    if "how many" in q:
        m = re.search(r"\b\d+(?:\.\d+)?\s+(?:day|days|week|weeks|month|months|year|years)\b", sentence, re.IGNORECASE)
        if m:
            return m.group(0)
        m = re.search(r"\b\d+(?:\.\d+)?\b", sentence)
        if m:
            return m.group(0)
    if "when" in q or "what date" in q:
        for pat in (
            rf"\b{_MONTH_RE}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,\s*\d{{4}})?\b",
            r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b",
            r"\b\d{4}-\d{2}-\d{2}\b",
        ):
            m = re.search(pat, sentence, re.IGNORECASE)
            if m:
                return m.group(0)
    return ""


def _extract_name_or_phrase(q: str, sentence: str) -> str:
    quoted = [x for x in re.findall(r"'([^']+)'|\"([^\"]+)\"", sentence) for x in x if x.strip()]
    if quoted:
        return quoted[0].strip()

    if q.startswith("who") or "which person" in q:
        candidates = re.findall(r"\b([A-Z][\w\.\-]*(?:\s+[A-Z][\w\.\-]*){0,4})\b", sentence)
        ignore = {"I", "We", "The", "A", "An", "By", "On", "At", "In", "Since"}
        cleaned = [c.strip() for c in candidates if c.strip() and c.strip() not in ignore]
        if cleaned:
            cleaned.sort(key=lambda s: len(s), reverse=True)
            return cleaned[0]
    return ""


def _compress_answer(query: str, sentence: str) -> str:
    clean_q = _clean_query(query)
    q = clean_q.lower()
    if not sentence:
        return ""

    opt = _find_best_options_hit(clean_q, sentence)
    if opt:
        return opt

    date_or_num = _extract_date_or_number(q, sentence)
    if date_or_num:
        return date_or_num

    phrase = _extract_name_or_phrase(q, sentence)
    if phrase:
        return phrase

    if q.startswith("what "):
        m = re.search(r"\b(?:is|are|was|were|did|does|do|has|have|had)\b\s+([^.,;]{2,100})", sentence, re.IGNORECASE)
        if m:
            return m.group(1).strip()

    return sentence.strip()


def extractive_answer(query: str, recalled: list[dict[str, Any]]) -> str:
    clean_q = _clean_query(query)
    best = ""
    best_score = -1.0
    for row in recalled:
        content = str(row.get("content", "")).strip()
        if not content:
            continue
        for sent in _split_sentences(content):
            sc = _sentence_score(clean_q, sent)
            if sc > best_score:
                best_score = sc
                best = sent
    if best:
        return _compress_answer(clean_q, best)
    if recalled:
        return str(recalled[0].get("content", "")).strip()[:220]
    return ""
