"""Memory write manager for ADD/UPDATE/NOOP policy decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re

from c3ae.config import MemoryManagerConfig
from c3ae.reasoning_bank.bank import ReasoningBank
from c3ae.types import ReasoningEntry


@dataclass
class WriteDecision:
    action: str  # ADD | UPDATE | NOOP
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

    def decide(self, entry: ReasoningEntry) -> WriteDecision:
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
