"""Structured answer contracts for memory QA."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class MemoryAnswer(BaseModel):
    answer: str
    answer_type: str
    confidence: float = 0.0
    citations: list[str] = Field(default_factory=list)
    supporting_spans: list[str] = Field(default_factory=list)
    abstain: bool = False
    normalization_steps: list[str] = Field(default_factory=list)
    verifier_status: str = "unchecked"
    metadata: dict[str, Any] = Field(default_factory=dict)


class VerificationResult(BaseModel):
    status: str
    confidence: float = 0.0
    reason: str = ""
