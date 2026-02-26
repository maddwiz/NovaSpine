"""Memory write manager for ADD/UPDATE/DELETE/NOOP policy decisions."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import json
import os
import re
from typing import Any

from c3ae.config import MemoryManagerConfig
from c3ae.llm import create_chat_backend
from c3ae.llm.venice_chat import Message
from c3ae.reasoning_bank.bank import ReasoningBank
from c3ae.types import ReasoningEntry, SearchResult


@dataclass
class WriteDecision:
    action: str  # ADD | UPDATE | DELETE | NOOP
    target_id: str = ""
    reason: str = ""


class MemoryWriteManager:
    """Heuristic policy manager for memory CRUD decisions.

    This is the practical pre-RL path:
    - ADD new entries
    - UPDATE by superseding near-matching entries
    - NOOP duplicates
    """

    def __init__(self, bank: ReasoningBank, config: MemoryManagerConfig | None = None) -> None:
        self.bank = bank
        self.config = config or MemoryManagerConfig()
        self._chat = None

    def decide(self, entry: ReasoningEntry) -> WriteDecision:
        """Heuristic policy decision."""
        if not self.config.enabled:
            return WriteDecision(action="ADD", reason="manager_disabled")

        query = f"{entry.title} {entry.content[:240]}"
        hits = self.bank.search(query, limit=5)
        if not hits:
            return WriteDecision(action="ADD", reason="no_similar_entries")

        best = hits[0]
        similarity = max(
            self._token_similarity(f"{entry.title} {entry.content}", best.content),
            self._token_similarity(entry.content, best.content),
        )
        if similarity >= float(self.config.similarity_noop_threshold):
            return WriteDecision(
                action="NOOP",
                target_id=best.id,
                reason=f"duplicate_similarity={similarity:.3f}",
            )
        if similarity >= float(self.config.similarity_update_threshold):
            return WriteDecision(
                action="UPDATE",
                target_id=best.id,
                reason=f"update_similarity={similarity:.3f}",
            )
        return WriteDecision(action="ADD", reason=f"novel_similarity={similarity:.3f}")

    async def decide_async(self, entry: ReasoningEntry) -> WriteDecision:
        """Policy decision with optional LLM routing + deterministic fallback."""
        heuristic = self.decide(entry)
        if not self.config.enabled or not self.config.use_llm_policy:
            return heuristic

        hits = self._candidate_hits(entry)
        if not hits:
            return heuristic

        try:
            llm_decision = await self._decide_with_llm(entry, hits)
        except Exception:
            heuristic.reason = f"{heuristic.reason}; llm_fallback"
            return heuristic

        allowed = {"ADD", "UPDATE", "DELETE", "NOOP"}
        action = llm_decision.action.strip().upper()
        if action not in allowed:
            heuristic.reason = f"{heuristic.reason}; llm_invalid_action"
            return heuristic
        llm_decision.action = action

        if action in {"UPDATE", "DELETE", "NOOP"}:
            if not llm_decision.target_id:
                heuristic.reason = f"{heuristic.reason}; llm_missing_target"
                return heuristic

            candidate_ids = [h.id for h in hits]
            if llm_decision.target_id not in candidate_ids:
                prefix_match = next(
                    (cid for cid in candidate_ids if cid.startswith(llm_decision.target_id)),
                    "",
                )
                if not prefix_match:
                    heuristic.reason = f"{heuristic.reason}; llm_unknown_target"
                    return heuristic
                llm_decision.target_id = prefix_match
        else:
            llm_decision.target_id = ""

        if not llm_decision.reason:
            llm_decision.reason = "llm_policy"
        return llm_decision

    def _candidate_hits(self, entry: ReasoningEntry, limit: int = 5) -> list[SearchResult]:
        query = f"{entry.title} {entry.content[:240]}"
        hits = self.bank.search(query, limit=limit)
        if hits:
            return hits

        # Fallback for low-overlap phrasing: still allow policy over recent active entries.
        rows = self.bank.list_active(limit=limit)
        out: list[SearchResult] = []
        for row in rows:
            out.append(
                SearchResult(
                    id=row.id,
                    content=f"{row.title}. {row.content}",
                    score=0.0,
                    source="reasoning_entry",
                    metadata={"status": row.status.value},
                )
            )
        return out

    async def _decide_with_llm(self, entry: ReasoningEntry, hits: list[Any]) -> WriteDecision:
        chat = self._get_chat_backend()
        candidates = []
        for h in hits:
            candidates.append(
                {
                    "id": h.id,
                    "preview": self._trim(h.content, max_chars=220),
                    "score": round(float(h.score), 4),
                }
            )

        payload = {
            "new_entry": {
                "title": entry.title,
                "content": self._trim(entry.content, max_chars=900),
                "tags": entry.tags,
                "evidence_ids_count": len(entry.evidence_ids),
            },
            "candidates": candidates,
            "policy": {
                "noop_if_duplicate": True,
                "update_if_same_fact_rephrased": True,
                "delete_if_existing_is_invalidated": True,
            },
        }

        response = await chat.chat(
            [
                Message(
                    role="system",
                    content=(
                        "You are a memory write policy router. "
                        "Choose exactly one action: ADD, UPDATE, DELETE, or NOOP."
                    ),
                ),
                Message(
                    role="user",
                    content=(
                        "Return strict JSON only with keys: action, target_id, reason.\n"
                        "Rules:\n"
                        "- ADD: new knowledge, target_id must be empty\n"
                        "- UPDATE: revise one candidate, set target_id\n"
                        "- DELETE: retract one stale/incorrect candidate, set target_id\n"
                        "- NOOP: duplicate of one candidate, set target_id\n\n"
                        f"Input:\n{json.dumps(payload, ensure_ascii=True)}"
                    ),
                ),
            ],
            temperature=float(self.config.llm_temperature),
            max_tokens=int(self.config.llm_max_tokens),
            json_mode=True,
        )
        data = self._parse_json_object(response.content)
        return WriteDecision(
            action=str(data.get("action", "ADD")).strip().upper(),
            target_id=str(data.get("target_id", "")).strip(),
            reason=str(data.get("reason", "")).strip(),
        )

    def _get_chat_backend(self):
        if self._chat is not None:
            return self._chat

        provider = str(self.config.llm_provider or "venice").strip().lower()
        kwargs: dict[str, Any] = {
            "temperature": float(self.config.llm_temperature),
            "max_tokens": int(self.config.llm_max_tokens),
        }
        if self.config.llm_model:
            kwargs["model"] = self.config.llm_model

        if provider in {"venice", "default"}:
            kwargs.setdefault("api_key", os.environ.get("VENICE_API_KEY", ""))
        elif provider == "openai":
            kwargs.setdefault("api_key", os.environ.get("OPENAI_API_KEY", ""))
        elif provider == "anthropic":
            kwargs.setdefault("api_key", os.environ.get("ANTHROPIC_API_KEY", ""))

        self._chat = create_chat_backend(provider=provider, **kwargs)
        return self._chat

    async def close(self) -> None:
        if self._chat is None:
            return
        close_fn = getattr(self._chat, "close", None)
        if close_fn is not None:
            maybe = close_fn()
            if inspect.isawaitable(maybe):
                await maybe
        self._chat = None

    @staticmethod
    def _parse_json_object(raw: str) -> dict[str, Any]:
        text = (raw or "").strip()
        if not text:
            return {}
        if "```json" in text:
            m = re.search(r"```json\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
            if m:
                text = m.group(1).strip()
        elif text.startswith("```"):
            m = re.search(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)
            if m:
                text = m.group(1).strip()
        try:
            data = json.loads(text)
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _trim(value: str, max_chars: int = 200) -> str:
        text = (value or "").strip()
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3] + "..."

    @staticmethod
    def _token_similarity(a: str, b: str) -> float:
        ta = {t for t in re.findall(r"[a-z0-9_\-]+", a.lower()) if t}
        tb = {t for t in re.findall(r"[a-z0-9_\-]+", b.lower()) if t}
        if not ta or not tb:
            return 0.0
        inter = len(ta & tb)
        union = len(ta | tb)
        if union == 0:
            return 0.0
        return inter / union
