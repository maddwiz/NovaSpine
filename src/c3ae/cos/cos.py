"""Carry-Over Summary (COS) — session state that persists across reasoning steps."""

from __future__ import annotations

from c3ae.config import COSConfig
from c3ae.storage.sqlite_store import SQLiteStore
from c3ae.types import CarryOverSummary
from c3ae.utils import utcnow


class COSManager:
    """Manages Carry-Over Summaries for reasoning sessions."""

    def __init__(self, store: SQLiteStore, config: COSConfig | None = None) -> None:
        self.store = store
        self.config = config or COSConfig()

    def create(self, session_id: str, summary: str,
               key_facts: list[str] | None = None,
               open_questions: list[str] | None = None) -> CarryOverSummary:
        """Create the initial COS for a session."""
        cos = CarryOverSummary(
            session_id=session_id,
            sequence=0,
            summary=summary,
            key_facts=key_facts or [],
            open_questions=open_questions or [],
        )
        self.store.insert_cos(cos)
        return cos

    def update(self, session_id: str, new_summary: str,
               new_facts: list[str] | None = None,
               resolved_questions: list[str] | None = None,
               new_questions: list[str] | None = None) -> CarryOverSummary:
        """Create a new COS entry by merging with the latest state."""
        latest = self.store.get_latest_cos(session_id)
        if latest is None:
            return self.create(session_id, new_summary, new_facts, new_questions)

        # Merge key facts (deduplicated)
        merged_facts = list(latest.key_facts)
        for f in (new_facts or []):
            if f not in merged_facts:
                merged_facts.append(f)
        if self.config.max_key_facts > 0 and len(merged_facts) > self.config.max_key_facts:
            merged_facts = merged_facts[-self.config.max_key_facts:]

        # Resolve questions
        resolved = set(resolved_questions or [])
        remaining_questions = [q for q in latest.open_questions if q not in resolved]
        for q in (new_questions or []):
            if q not in remaining_questions:
                remaining_questions.append(q)
        if self.config.max_open_questions > 0 and len(remaining_questions) > self.config.max_open_questions:
            remaining_questions = remaining_questions[-self.config.max_open_questions:]

        cos = CarryOverSummary(
            session_id=session_id,
            sequence=latest.sequence + 1,
            summary=new_summary,
            key_facts=merged_facts,
            open_questions=remaining_questions,
        )
        self.store.insert_cos(cos)
        return cos

    def get_latest(self, session_id: str) -> CarryOverSummary | None:
        return self.store.get_latest_cos(session_id)

    def get_history(self, session_id: str) -> list[CarryOverSummary]:
        return self.store.list_cos(session_id)

    def render_prompt(self, session_id: str) -> str:
        """Render the current COS as a prompt section for LLM context."""
        cos = self.store.get_latest_cos(session_id)
        if cos is None:
            return ""
        parts = [f"## Carry-Over Summary (step {cos.sequence})\n"]
        parts.append(cos.summary)
        if cos.key_facts:
            parts.append("\n### Key Facts")
            for f in cos.key_facts:
                parts.append(f"- {f}")
        if cos.open_questions:
            parts.append("\n### Open Questions")
            for q in cos.open_questions:
                parts.append(f"- {q}")
        return "\n".join(parts)
