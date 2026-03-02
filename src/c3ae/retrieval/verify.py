"""Answer-verification helpers for multi-pass QA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from c3ae.llm import Message
from c3ae.utils import parse_json_object


@dataclass(frozen=True)
class VerificationResult:
    verified: bool
    answer: str
    reason: str


def _coerce_bool(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return False
    text = str(raw).strip().lower()
    return text in {"1", "true", "yes", "y", "verified"}


def build_verification_messages(
    *,
    query: str,
    proposed_answer: str,
    context: str,
) -> list[Message]:
    system = (
        "You verify whether a proposed QA answer is directly supported by context.\n"
        "Return strict JSON only with keys: verified (boolean), answer (string), reason (string).\n"
        "Rules:\n"
        "- verified=true only when the exact answer is directly supported.\n"
        "- if verified=false and a better short answer exists in context, provide it in answer.\n"
        "- if no answer is supported, return answer as empty string.\n"
        "- keep answer concise; no extra keys; no markdown."
    )
    user = (
        f"Question:\n{query}\n\n"
        f"Proposed answer:\n{proposed_answer}\n\n"
        f"Context:\n{context}\n\n"
        "Return JSON now."
    )
    return [Message(role="system", content=system), Message(role="user", content=user)]


def parse_verification_payload(raw_content: str, *, proposed_answer: str = "") -> VerificationResult:
    obj = parse_json_object(raw_content)
    if not obj:
        return VerificationResult(verified=False, answer="", reason="invalid_json")
    verified = _coerce_bool(obj.get("verified"))
    answer = str(obj.get("answer", "") or "").strip()
    reason = str(obj.get("reason", "") or "").strip()
    if verified and not answer:
        answer = proposed_answer.strip()
    return VerificationResult(verified=verified, answer=answer, reason=reason)


async def verify_answer_with_llm(
    *,
    llm_backend: Any,
    query: str,
    proposed_answer: str,
    context: str,
    temperature: float = 0.0,
    max_tokens: int = 160,
) -> VerificationResult:
    messages = build_verification_messages(
        query=query,
        proposed_answer=proposed_answer,
        context=context,
    )
    resp = await llm_backend.chat(
        messages,
        temperature=float(temperature),
        max_tokens=max(96, int(max_tokens)),
        json_mode=True,
    )
    return parse_verification_payload(resp.content, proposed_answer=proposed_answer)

