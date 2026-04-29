"""Lightweight retrieval tracing contracts."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RetrievalTrace:
    """Optional per-query retrieval diagnostics.

    The object is intentionally small and side-effect free so normal retrieval
    output remains unchanged unless a caller explicitly reads the trace.
    """

    timings_ms: dict[str, float] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def add_timing(self, key: str, elapsed_ms: float) -> None:
        self.timings_ms[key] = round(float(elapsed_ms), 3)

    def add_counter(self, key: str, value: int) -> None:
        self.counters[key] = int(value)

    def merge(self, other: "RetrievalTrace | None") -> None:
        if other is None:
            return
        self.timings_ms.update(other.timings_ms)
        self.counters.update(other.counters)
        self.notes.extend(other.notes)

    def to_dict(self) -> dict[str, object]:
        return {
            "timings_ms": dict(self.timings_ms),
            "counters": dict(self.counters),
            "notes": list(self.notes),
        }
