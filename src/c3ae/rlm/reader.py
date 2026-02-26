"""RLM Reader — Recursive out-of-core reader producing Evidence Packs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from c3ae.llm import Message, create_chat_backend
from c3ae.memory_spine.spine import MemorySpine
from c3ae.types import EvidencePack
from c3ae.utils import chunk_text


@dataclass
class ReadBudget:
    """Controls how much the reader processes."""
    max_chunks: int = 100
    max_depth: int = 3
    chunks_processed: int = 0
    depth: int = 0

    @property
    def exhausted(self) -> bool:
        return self.chunks_processed >= self.max_chunks or self.depth >= self.max_depth

    def consume(self, n: int = 1) -> None:
        self.chunks_processed += n


@dataclass
class ReadResult:
    """Output of the RLM reader."""
    chunks_processed: int
    evidence_packs: list[EvidencePack] = field(default_factory=list)
    summaries: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class RLMReader:
    """Recursive paging reader for large documents.

    Processes documents in chunks, extracting evidence packs at each level.
    Budget enforcement prevents unbounded processing.
    """

    def __init__(
        self,
        spine: MemorySpine,
        use_llm_extraction: bool = True,
        max_claims_per_chunk: int = 5,
    ) -> None:
        self.spine = spine
        self.use_llm_extraction = use_llm_extraction
        self.max_claims_per_chunk = max(1, max_claims_per_chunk)
        self._claim_chat = None

    async def read_text(self, text: str, topic: str = "",
                        budget: ReadBudget | None = None) -> ReadResult:
        """Read a text document, producing evidence packs."""
        budget = budget or ReadBudget()
        chunks = chunk_text(text)
        return await self._process_chunks(chunks, topic, budget)

    async def read_file(self, file_path: Path, topic: str = "",
                        budget: ReadBudget | None = None) -> ReadResult:
        """Read a file, producing evidence packs."""
        text = file_path.read_text(errors="replace")
        return await self.read_text(text, topic, budget)

    async def _process_chunks(self, chunks: list[str], topic: str,
                               budget: ReadBudget) -> ReadResult:
        """Process chunks with budget enforcement."""
        result = ReadResult(chunks_processed=0)

        for chunk in chunks:
            if budget.exhausted:
                break

            # Ingest chunk into memory
            chunk_ids = await self.spine.ingest_text(chunk, source_id=f"rlm:{topic}")
            budget.consume(1)
            result.chunks_processed += 1

            # Extract claims from chunk (simplified — in production, LLM would do this)
            claims = await self._extract_claims(chunk, topic)
            for claim, reasoning in claims:
                pack = self.spine.add_evidence(
                    claim=claim,
                    sources=[f"chunk:{chunk_ids[0]}" if chunk_ids else "inline"],
                    confidence=0.5,
                    reasoning=reasoning,
                )
                result.evidence_packs.append(pack)

        result.metadata = {
            "topic": topic,
            "total_chunks": len(chunks),
            "budget_remaining": budget.max_chunks - budget.chunks_processed,
        }
        return result

    async def _extract_claims(self, chunk: str, topic: str) -> list[tuple[str, str]]:
        """Extract claims from text via LLM-first pipeline with deterministic fallback."""
        if self.use_llm_extraction:
            claims = await self._extract_claims_llm(chunk, topic)
            if claims:
                return claims
        return self._extract_claims_heuristic(chunk, topic)

    async def _extract_claims_llm(self, chunk: str, topic: str) -> list[tuple[str, str]]:
        api_key = self.spine.config.venice.api_key
        if not api_key:
            return []

        if self._claim_chat is None:
            self._claim_chat = create_chat_backend(
                provider="venice",
                api_key=api_key,
                model=self.spine.config.venice.chat_model,
                base_url=self.spine.config.venice.base_url,
                timeout=self.spine.config.venice.chat_timeout,
                temperature=0.0,
                max_tokens=900,
            )

        system = (
            "Extract only verifiable factual claims from text. "
            "Return strict JSON with an array under key 'claims'."
        )
        user = (
            f"Topic: {topic or 'general'}\n"
            "Return JSON as {\"claims\":[{\"claim\":\"...\",\"reasoning\":\"...\"}]}. "
            f"Include at most {self.max_claims_per_chunk} claims.\n\n"
            f"Text:\n{chunk}"
        )

        try:
            resp = await self._claim_chat.chat(
                [Message(role="system", content=system), Message(role="user", content=user)],
                json_mode=True,
            )
            payload = json.loads(resp.content)
        except Exception:
            return []

        rows: list[dict[str, Any]]
        if isinstance(payload, list):
            rows = [x for x in payload if isinstance(x, dict)]
        elif isinstance(payload, dict):
            claims_obj = payload.get("claims", [])
            rows = [x for x in claims_obj if isinstance(x, dict)]
        else:
            rows = []

        out: list[tuple[str, str]] = []
        for row in rows[: self.max_claims_per_chunk]:
            claim = str(row.get("claim", "")).strip().rstrip(".")
            reasoning = str(row.get("reasoning", "")).strip()
            if len(claim) < 20:
                continue
            if not reasoning:
                reasoning = (
                    f"LLM-extracted claim from topic '{topic}'"
                    if topic
                    else "LLM-extracted claim from text chunk"
                )
            out.append((claim, reasoning))
        return out

    def _extract_claims_heuristic(self, chunk: str, topic: str) -> list[tuple[str, str]]:
        """Fallback extractor when LLM extraction is unavailable."""
        sentences = [
            s.strip()
            for s in re.split(r"(?<=[\.\!\?])\s+|\n+", chunk)
            if len(s.strip()) >= 20
        ]
        ranked: list[tuple[float, str]] = []
        for sentence in sentences:
            s = sentence.strip().rstrip(".")
            score = 0.0
            if re.search(r"\d", s):
                score += 1.0
            if re.search(r"\b(is|are|was|were|has|have|will|must|shows|indicates)\b", s.lower()):
                score += 0.8
            if re.search(r"\b(increase|decrease|improve|decline|risk|probability|evidence)\b", s.lower()):
                score += 0.7
            score += min(len(s) / 200.0, 1.0)
            ranked.append((score, s))

        ranked.sort(key=lambda x: x[0], reverse=True)
        claims: list[tuple[str, str]] = []
        for _, claim in ranked[: self.max_claims_per_chunk]:
            reasoning = (
                f"Heuristic extraction from topic '{topic}': '{claim[:100]}...'"
                if topic
                else f"Heuristic extraction: '{claim[:100]}...'"
            )
            claims.append((claim, reasoning))
        return claims

    async def close(self) -> None:
        if self._claim_chat is not None:
            await self._claim_chat.close()
            self._claim_chat = None
