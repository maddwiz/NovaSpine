"""Deterministic write-admission policy for memory facts and graph edges."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from c3ae.config import MemoryManagerConfig


SINGLE_VALUED_RELATIONS = {
    "bag",
    "charger",
    "coffee_order",
    "flight_seat",
    "is",
    "location",
    "movie_drink",
    "movie_snack",
    "notebook",
    "owns",
    "prefers",
}


@dataclass(frozen=True)
class AdmissionDecision:
    action: str  # ALLOW | DENY | NOOP | SUPERSEDE
    target_id: str = ""
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class MemoryWriteAdmissionManager:
    """Small deterministic admission layer for extracted memory writes.

    This sits below the reasoning-entry manager. It keeps raw extraction
    deterministic while recording enough provenance/policy metadata for later
    training and self-repair.
    """

    def __init__(self, config: MemoryManagerConfig | None = None) -> None:
        self.config = config or MemoryManagerConfig()

    def decide_structured_fact(
        self,
        store: Any,
        *,
        source_chunk_id: str,
        entity: str,
        relation: str,
        value: str,
        fact_date: str = "",
        confidence: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> AdmissionDecision:
        if not self.config.enabled:
            return AdmissionDecision("ALLOW", reason="manager_disabled")
        entity_clean = " ".join((entity or "").split())
        relation_clean = self._normalize_relation(relation)
        value_clean = " ".join((value or "").split())
        if not source_chunk_id or not entity_clean or not relation_clean or len(value_clean) < 2:
            return AdmissionDecision("DENY", reason="missing_required_fields")
        min_conf = float(getattr(self.config, "admission_min_confidence", 0.0))
        if float(confidence) < min_conf:
            return AdmissionDecision("DENY", reason=f"low_confidence={float(confidence):.3f}")

        existing = store.list_current_structured_facts(
            entity=entity_clean,
            relation=relation_clean,
            limit=25,
        )
        value_norm = self._normalize_value(value_clean)
        for fact in existing:
            if self._normalize_value(str(fact.get("value") or "")) == value_norm:
                return AdmissionDecision(
                    "NOOP",
                    target_id=str(fact.get("id") or ""),
                    reason="duplicate_current_fact",
                    metadata={"duplicate_fact_id": str(fact.get("id") or "")},
                )

        if self.is_single_valued_relation(relation_clean) and existing:
            supersedes = [str(fact.get("id") or "") for fact in existing if str(fact.get("id") or "")]
            return AdmissionDecision(
                "SUPERSEDE",
                target_id=supersedes[0] if supersedes else "",
                reason="single_valued_relation_replacement",
                metadata={"supersedes_fact_ids": supersedes},
            )
        return AdmissionDecision("ALLOW", reason="accepted")

    def decide_graph_edge(
        self,
        store: Any,
        *,
        src_entity_id: str,
        relation: str,
        dst_entity_id: str,
        confidence: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> AdmissionDecision:
        _ = store, metadata
        if not self.config.enabled:
            return AdmissionDecision("ALLOW", reason="manager_disabled")
        if not src_entity_id or not dst_entity_id or not self._normalize_relation(relation):
            return AdmissionDecision("DENY", reason="missing_required_fields")
        min_conf = float(getattr(self.config, "admission_min_confidence", 0.0))
        if float(confidence) < min_conf:
            return AdmissionDecision("DENY", reason=f"low_confidence={float(confidence):.3f}")
        if self.is_single_valued_relation(relation):
            return AdmissionDecision("SUPERSEDE", reason="single_valued_relation")
        return AdmissionDecision("ALLOW", reason="accepted")

    @staticmethod
    def is_single_valued_relation(relation: str) -> bool:
        return MemoryWriteAdmissionManager._normalize_relation(relation) in SINGLE_VALUED_RELATIONS

    @staticmethod
    def _normalize_relation(relation: str) -> str:
        return "_".join((relation or "").strip().lower().split())

    @staticmethod
    def _normalize_value(value: str) -> str:
        return " ".join((value or "").strip().lower().split())

